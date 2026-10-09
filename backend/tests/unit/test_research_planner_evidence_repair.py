"""Planner evidence sufficiency and repair tests."""

from __future__ import annotations

from o2_evidence_test_utils import read_result, material
from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials

import asyncio
from dataclasses import dataclass
from typing import Any
import pytest
from o2_evidence_test_utils import scope

from app.models.research_planner_contracts import (
    RepairQuery,
    ResearchClaimDraft,
    ResearchPlan,
    ResearchPlanDraft,
    ResearchUnit,
    ResearchUnitResult,
)
from app.models.knowledge_access_contracts import EvidenceReadResult
from app.services.internal_research.research_planner_evidence_repair import (
    build_evidence_sufficiency_report,
    build_repair_query_plans,
    execute_with_repair,
)
from app.services.internal_research.research_planner_executor import (
    PlannerExecutionContext,
    ResearchPlannerExecutor,
)
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from app.services.internal_research.research_planner_pipeline import (
    execute_planner_state,
)
from app.services.internal_research.research_planner_synthesis import (
    synthesize_planner_answer,
)
from research_planner_strategy_test_utils import validate_strategy_fixture
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraft, ResearchClaimStrategyDraftBatch
from research_planner_test_utils import make_draft, make_facet, make_unit, supporting_spans, explicit_claim_provider


def test_missing_comparison_target_triggers_scoped_repair() -> None:
    calls: list[dict[str, Any]] = []
    plan = _comparison_plan()
    executor = ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={"local_direct_text": _repairing_adapter(calls)})

    outcome = _run(_execute_stages(plan=plan, executor=executor, context=_ctx()))

    assert outcome.repair_status == "resolved"
    # 首轮按拓扑序取证 u1(paper-a)、u2(paper-b 空)，再对缺证据的 u2 发起 paper-b 定向修复
    assert calls[0]["target_paper_ids"] == ["paper-a"]
    assert calls[1]["target_paper_ids"] == ["paper-b"]
    assert calls[2]["target_paper_ids"] == ["paper-b"]
    assert outcome.repair_query_plans[0].queries[0].target_paper_id == "paper-b"
    # u1 首轮拿到 paper-a 证据，u2 经修复拿到 paper-b 证据
    assert outcome.results[0].status == "evidence_found"
    assert outcome.results[0].evidence_ids == [_evidence("e-a", "paper-a")["evidence_id"]]
    assert outcome.results[1].status == "evidence_found"
    assert outcome.results[1].evidence_ids == [_evidence("e-b", "paper-b")["evidence_id"]]
    assert outcome.missing_evidence == []


def test_out_of_scope_repair_record_is_rejected() -> None:
    plan = _comparison_plan()
    executor = ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={"local_direct_text": _unscoped_repair_adapter()})

    outcome = _run(_execute_stages(plan=plan, executor=executor, context=_ctx()))

    assert outcome.repair_status == "failed"
    # u1(paper-a) 首轮成功，u2(paper-b) 的修复证据落在范围外被拒，u2 仍取证不足
    assert outcome.results[0].status == "evidence_found"
    assert outcome.results[1].status == "insufficient_evidence"
    assert outcome.repair_merge_report.invalid_evidence_ids == [_evidence("e-unscoped", "paper-outside")["evidence_id"]]
    assert outcome.missing_evidence[0]["paper_id"] == "paper-b"
    assert outcome.missing_evidence[0]["attempted_queries"]


def test_rerun_failure_preserves_repair_trace_and_drops_claims() -> None:
    calls: list[dict[str, Any]] = []
    plan = _comparison_plan()
    executor = ResearchPlannerExecutor(
        adapters={"local_direct_text": _repairing_adapter(calls)},
        reasoner=ResearchUnitReasoner(draft_provider=_rerun_failing_provider),
    )

    with pytest.raises(RuntimeError, match="rerun claim draft failed") as caught:
        _run(_execute_stages(plan=plan, executor=executor, context=_ctx()))
    report = caught.value.planner_execution_report
    assert report.results[1].status == "blocked"
    assert report.results[1].claims == []
    assert _evidence("e-b", "paper-b")["evidence_id"] in {record["evidence_id"] for record in report.evidence_records}
    assert report.read_receipts


def test_sufficiency_report_and_query_plan_are_deterministic() -> None:
    plan = _comparison_plan()
    first_result = _run(
        ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={"local_direct_text": _first_pass_adapter()}).execute_with_evidence(plan, _ctx())
    )

    report = build_evidence_sufficiency_report(
        plan=plan,
        results=first_result.results,
        evidence_records=first_result.evidence_records,
        pass_index=1,
    )
    query_plan = build_repair_query_plans(plan=plan, report=report, direct_adapter_available=True)[0]

    # u1 锁 paper-a 首轮拿到证据，无缺口；缺证据的是锁 paper-b 的 u2
    assert report.unit_reports[0].hard_gaps == []
    gap = next(gap for gap in report.unit_reports[1].hard_gaps if gap.gap_type == "target_coverage_missing")
    assert gap.target_paper_id == "paper-b"
    assert gap.required_count == 1 and gap.found_count == 0
    assert {gap.gap_type for gap in report.unit_reports[1].hard_gaps} >= {"claim_output_missing", "required_claim_type_missing"}
    assert query_plan.queries[0] == RepairQuery(
        query="attention mechanism mechanism", target_paper_id="paper-b", evidence_dimension="mechanism",
        required_sections=["Method"], source_gap_id=gap.gap_id,
    )
    assert report == build_evidence_sufficiency_report(plan=plan, results=first_result.results,
                                                       evidence_records=first_result.evidence_records, pass_index=1)


def test_malformed_final_carrier_remains_failed_with_repair_gaps() -> None:
    plan = _comparison_plan()
    payload = synthesize_planner_answer(
        plan,
        [_blocked_result()],
        project_paper_count=2,
        evidence_records=[_evidence("e-a", "paper-a")],
        final_answer_draft={"blocks": [{"role": "recommendation", "text": "do not use", "claim_ids": [], "evidence_ids": []}]},
        repair_context={
            "planner_repair_status": "failed",
            "planner_repair_reason": "repair failed",
            "missing_evidence": [{"paper_id": "paper-b", "dimension": "mechanism", "reason": "missing"}],
            "final_answer_eligibility_report": {
                "can_call_provider": False,
                "final_status_if_skipped": "skipped",
                "blocking_unit_ids": ["u1"],
                "blocking_gap_ids": ["g1"],
                "partial_allowed": False,
                "reason": "target_paper_direct_text_missing",
            },
        },
    )

    assert payload["source_status"]["planner_repair"] == "failed"
    assert payload["source_status"]["planner_final"] == "failed"
    assert "do not use" not in payload["answer"]
    assert payload["missing_evidence"][0]["paper_id"] == "paper-b"


def test_structure_hint_repair_reruns_direct_verification() -> None:
    calls: list[dict[str, Any]] = []
    plan = _structure_plan()
    executor = ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={
        "local_structure_hint": _structure_adapter(),
        "local_direct_text": _structure_direct_adapter(calls),
    })

    outcome = _run(_execute_stages(plan=plan, executor=executor, context=_ctx()))

    assert outcome.repair_status == "resolved"
    assert outcome.repair_query_plans[0].queries[0].target_paper_id == "paper-a"
    assert calls[0]["unit_id"] == "u1"
    assert calls[1]["target_paper_ids"] == ["paper-a"]
    assert outcome.results[0].status == "evidence_found"
    assert outcome.results[0].claims[0].supporting_evidence_ids == [_evidence("e-a", "paper-a")["evidence_id"]]


def test_staged_execute_state_returns_repair_payload() -> None:
    calls: list[dict[str, Any]] = []
    plan = _comparison_plan()
    project_scope = scope()
    project_scope["paper_ids"] = ["paper-a", "paper-b"]
    project_scope["publication"]["paper_batches"]["paper-b"] = project_scope["publication"]["paper_batches"]["paper-a"]
    state = {"user_query": plan.question, "research_plan": plan.model_dump(mode="json"),
             "run_id": "test-run", "project_scope": project_scope}

    payload = _run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters={"local_direct_text": _repairing_adapter(calls)},
    ))

    assert payload["source_status"]["planner_repair"] == "skipped"
    assert payload["evidence_sufficiency_report"]["pass_index"] == 1
    assert payload["repair_query_plans"] == []
    assert len(calls) == 2
    assert payload["missing_evidence"]


@dataclass
class _Settings:
    rag_research_planner_max_units: int = 8


@dataclass
class _FakeLLM:
    draft: dict[str, Any] | None = None

    def with_structured_output(self, model: Any) -> "_StructuredOutput":
        return _StructuredOutput(model=model, draft=self.draft)


@dataclass
class _StructuredOutput:
    model: Any
    draft: dict[str, Any] | None = None

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self.model is ResearchPlanDraft:
            return ResearchPlanDraft.model_validate(
                self.draft or make_draft([_paper_a_unit(), _paper_b_unit()])
            )
        if self.model is ResearchClaimStrategyDraftBatch:
            # 每个直取单元只能引用喂给它自己的那篇证据，避免跨单元引用不存在的 id
            evidence_ids = _evidence_ids_in_prompt(messages)
            if not evidence_ids:
                return ResearchClaimStrategyDraftBatch(claims=[])
            return ResearchClaimStrategyDraftBatch(claims=[
                ResearchClaimStrategyDraft(
                    text="Direct evidence supports this unit.",
                    claim_type="paper_supported",
                    supporting_evidence_ids=evidence_ids,
                )
            ])
        raise AssertionError(f"unexpected structured output model: {self.model}")


def _evidence_ids_in_prompt(messages: list[Any]) -> list[str]:
    # 从 claim drafting 提示词里解析出本单元证据块的 id 行，模拟只引用在场证据的 LLM 行为
    text = "\n".join(str(getattr(message, "content", "")) for message in messages)
    ids: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("id: "):
            evidence_id = stripped[len("id: "):].strip()
            if evidence_id and evidence_id not in ids:
                ids.append(evidence_id)
    return ids


def _comparison_plan() -> ResearchPlan:
    # 此用例以每篇一个单元表达比较，O3 也支持多个对象同单元或多单元合作
    # 首轮 paper-a 出证据、paper-b 不出，缺证据的 u2 触发对 paper-b 的定向修复
    draft = make_draft(
        [_paper_a_unit(), _paper_b_unit()],
        explicit_facets=[make_facet("f1"), make_facet("f2")],
    )
    result = validate_strategy_fixture(draft)
    assert result.plan is not None
    return result.plan


def _structure_plan() -> ResearchPlan:
    result = validate_strategy_fixture(make_draft([_structure_unit()], explicit_facets=[]))
    assert result.plan is not None
    return result.plan


def _paper_a_unit() -> dict[str, Any]:
    return make_unit(
        "paper-a",
        goal="Locate paper-a attention mechanism evidence",
        action="locate attention mechanism evidence in paper-a",
        paper_ids=["paper-a"],
        allowed_claims=["paper_supported"],
        facets=["f1"],
    )


def _paper_b_unit() -> dict[str, Any]:
    return make_unit(
        "paper-b",
        goal="Locate paper-b attention mechanism evidence",
        action="locate attention mechanism evidence in paper-b",
        paper_ids=["paper-b"],
        allowed_claims=["paper_supported"],
        facets=["f2"],
    )


def _structure_unit() -> dict[str, Any]:
    unit = make_unit(
        "hint",
        goal="Verify graph relation",
        action="verify graph hint with direct text",
        source_type="local_structure_hint",
        allowed_claims=["paper_supported"],
        paper_ids=["paper-a"],
    )
    unit["claim_policy"]["citation_required"] = False
    # 这个单元的正文核实是计划索取的，所以首轮缺正文时补证会去补一次正文检索；
    # 不索取核实的结构线索单元不缺这份正文，补证也不该替它去补
    unit["evidence_policy"]["verify_with_direct_text"] = True
    return unit


def _repairing_adapter(calls: list[dict[str, Any]]):
    async def run(unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        calls.append({
            "unit_id": unit.unit_id,
            "target_paper_ids": list(unit.evidence_policy.target_paper_ids),
            "queries": list(unit.evidence_policy.locator_queries),
        })
        # u1 锁 paper-a，首轮就出证据；u2 锁 paper-b，首轮不出，逼出针对 paper-b 的修复
        if unit.unit_id.endswith("::evidence_repair::p2"):
            return read_result([_evidence("e-b", "paper-b")])
        if "paper-b" in unit.evidence_policy.target_paper_ids:
            return read_result([])
        return read_result([_evidence("e-a", "paper-a")])

    return run


def _structure_adapter():
    async def run(_unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        return read_result([{
            "evidence_id": "g-a",
            "source": "neo4j",
            "layer": "structure_hint",
            "paper_id": "paper-a",
            "summary": "graph relation",
            "usable_as_citation": False,
            "verification_status": "unverified",
        }])

    return run


def _structure_direct_adapter(calls: list[dict[str, Any]]):
    async def run(unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        calls.append({"unit_id": unit.unit_id, "target_paper_ids": list(unit.evidence_policy.target_paper_ids)})
        if unit.unit_id.endswith("::evidence_repair::p2"):
            return read_result([_evidence("e-a", "paper-a")])
        return read_result([])

    return run


def _first_pass_adapter():
    async def run(_unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        return read_result([_evidence("e-a", "paper-a")])

    return run


def _unscoped_repair_adapter():
    async def run(unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        # u2(paper-b) 的修复检索返回一条拥有完整来源但超出授权目标的证据，应在合并阶段被拒
        if unit.unit_id.endswith("::evidence_repair::p2"):
            return read_result([_evidence("e-unscoped", "paper-outside")])
        if "paper-b" in unit.evidence_policy.target_paper_ids:
            return read_result([])
        return read_result([_evidence("e-a", "paper-a")])

    return run


async def _rerun_failing_provider(
    unit: ResearchUnit,
    evidence: list[dict[str, Any]],
    _dependency_results: list[ResearchUnitResult],
    *, read_reports=None, prepared_input=None) -> list[ResearchClaimDraft]:
    # 只在缺证据的 u2 修复后重跑时失败，验证重跑失败会丢弃 claim 并保留修复轨迹
    if unit.unit_id == "u2":
        raise RuntimeError("rerun claim draft failed")
    return [
        ResearchClaimDraft(
            text="Scoped direct evidence is available.",
            claim_type="paper_supported",
            supporting_evidence_ids=[str(evidence[0]["evidence_id"])], supporting_spans=supporting_spans(evidence[:1]),
        )
    ]


def _blocked_result():
    return ResearchUnitResult(
        unit_id="u1",
        status="insufficient_evidence",
        evidence_ids=[_evidence("e-a", "paper-a")["evidence_id"]],
        unresolved_reason="missing required citations for target papers",
    )


def _evidence(evidence_id: str, paper_id: str) -> dict[str, Any]:
    """参数为夹具定位与论文；通过真实转换生成新版范围及哈希身份。"""
    return convert_knowledge_materials((material("attention mechanism", paper_id=paper_id, locator=evidence_id),))[0]


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="Compare attention mechanisms")


def _run(awaitable):
    return asyncio.run(awaitable)


async def _execute_stages(*, plan, executor, context):
    """参数为计划和离线执行器；显式执行首轮与获准补证两个阶段，不恢复自动补证旁路。"""
    initial = await execute_with_repair(plan=plan, executor=executor, context=context)
    return await execute_with_repair(plan=plan, executor=executor, context=context, mode="repair",
                                     previous_results=initial.results, prior_sufficiency=initial.final_sufficiency_report)
