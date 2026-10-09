"""取证能力四态结果合同：success / empty / skipped / failed 各自可辨，最终并进回合终态 capability_results。作者：xxx。"""

from __future__ import annotations

from o2_evidence_test_utils import direct_evidence, read_result, scope as publication_scope

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from app.models.research_planner_contracts import (
    ResearchClaimDraft,
    ResearchPlan,
    ResearchUnit,
)
from app.models.knowledge_access_contracts import EvidenceReadResult
from app.services.internal_research.research_memory_writeback import build_turn_outcome
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError,
    PlannerExecutionContext,
    ResearchPlannerExecutor,
)
from app.services.internal_research.research_planner_pipeline import execute_planner_state
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from research_planner_strategy_test_utils import current_plan_fixture, validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit


def test_success_and_empty_are_distinct_states() -> None:
    """适配器返回记录记 success、返回空列表记 empty，两者不合并且空列表不冒充成功。"""
    plan = _plan([make_unit("a", facets=["f1"]), _metadata_unit("b", facets=["f2"])])
    executor = ResearchPlannerExecutor(
        adapters={
            "local_direct_text": _adapter([_direct_evidence()]),
            "local_metadata": _adapter([]),
        },
        reasoner=_reasoner(),
    )

    report = _run(executor.execute_with_evidence(plan, _ctx()))

    assert report.capability_results == {"local_direct_text": "success", "local_metadata": "empty"}


def test_uncalled_capability_is_skipped() -> None:
    """计划里有该本地取证类型却压根没调用适配器时记 skipped，不冒充 empty，也不记 failed。"""
    # 1. direct_text 缺 locator_queries：适配器在册也不调用
    unit = make_unit("a", facets=["f1"])
    unit["evidence_policy"]["locator_queries"] = []
    calls: list[str] = []
    report = _run(
        ResearchPlannerExecutor(
            adapters={"local_direct_text": _adapter([_direct_evidence()], calls)},
        ).execute_with_evidence(_raw_plan([unit]), _ctx())
    )
    assert report.capability_results == {"local_direct_text": "skipped"}
    assert calls == []
    # 2. 依赖阻塞：被阻塞单元压根没调用适配器，先出的能力照常记 empty
    units = [
        make_unit("a", facets=["f1"]),
        _metadata_unit("b", facets=["f2"], deps=[_dep("a", required_claims=["paper_supported"])]),
    ]
    metadata_calls: list[str] = []
    blocked = _run(
        ResearchPlannerExecutor(
            adapters={
                "local_direct_text": _adapter([]),
                "local_metadata": _adapter([_metadata_evidence()], metadata_calls),
            },
        ).execute_with_evidence(_plan(units, facets=["f1", "f2"]), _ctx())
    )
    assert blocked.capability_results == {"local_direct_text": "empty", "local_metadata": "skipped"}
    assert metadata_calls == []


def test_missing_adapter_leaves_no_state_and_blocks_the_unit() -> None:
    """枚举值没有对应适配器时该能力压根没跑，不进四态表；缺陷本身在单元结果里显式可见。"""
    plan = _plan([make_unit("a", facets=["f1"])])
    report = _run(ResearchPlannerExecutor(adapters={}).execute_with_evidence(plan, _ctx()))

    # 1. 装配缺陷不记 skipped——skipped 是「本轮不需要」，记成它会让你永远看不出枚举有值、能力没接上
    assert report.capability_results == {}
    # 2. 缺陷不被吞：单元如实 blocked，原因写明没注册哪个 source_type
    assert report.results[0].status == "blocked"
    assert "no adapter registered for local_direct_text" in report.results[0].unresolved_reason


def test_adapter_failure_is_failed_and_turn_still_hard_fails() -> None:
    """适配器抛错记 failed 且带可读原因，异常照旧冒到执行层变成硬失败，没被吞。"""
    boom = RuntimeError("milvus query timeout")
    plan = _plan([make_unit("a", facets=["f1"])])
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": _raising_adapter(boom)})

    # 1. 首轮执行不吞错：显式异常带取证面与原因上抛，原始原因链保留
    with pytest.raises(EvidenceCapabilityError) as caught:
        _run(executor.execute_with_evidence(plan, _ctx()))
    assert caught.value.source_type == "local_direct_text"
    assert caught.value.reason == "RuntimeError: milvus query timeout"
    assert caught.value.__cause__ is boom
    # 2. 到 execute_planner_state 仍是既有的硬失败路径：错误码、终止决策与失败载荷不变
    state = {"run_id": "test-run", "project_scope": publication_scope(),
             "user_query": plan.question, "research_plan": plan.model_dump(mode="json")}
    payload = _run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters={"local_direct_text": _raising_adapter(boom)},
    ))
    assert payload["error_code"] == "planner_execution_failed"
    assert payload["planner_loop_decision"] == "terminate_fail"
    assert payload["planner_evidence_capabilities"] == {"local_direct_text": "failed"}
    assert payload["source_status_reasons"]["local_direct_text"] == "RuntimeError: milvus query timeout"
    assert "milvus query timeout" in payload["error_summary"]
    # 3. 四态最终进回合终态：failed 如实出现在 capability_results 里，回合状态仍是 failed
    outcome = build_turn_outcome(payload, memory_writeback={})
    assert outcome["status"] == "failed"
    assert outcome["capability_results"]["local_direct_text"] == "failed"


def test_execute_state_delta_carries_capability_states() -> None:
    """execute 阶段把四态写进 planner_evidence_capabilities；空结果走完补证流程仍如实是 empty。"""
    plan = _plan([make_unit("a", facets=["f1"])])
    state = {"run_id": "test-run", "project_scope": publication_scope(),
             "user_query": plan.question, "research_plan": plan.model_dump(mode="json")}

    success = _run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters={"local_direct_text": _adapter([_direct_evidence()])},
    ))
    assert success["planner_evidence_capabilities"] == {"local_direct_text": "success"}

    empty = _run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters={"local_direct_text": _adapter([])},
    ))
    assert empty["planner_evidence_capabilities"] == {"local_direct_text": "empty"}


def test_capability_states_reach_turn_outcome() -> None:
    """四态并进回合终态 capability_results；取值不在 CapabilityStatus 词表内的整条不出现。"""
    plan = _plan([make_unit("a", facets=["f1"])])
    state = {"run_id": "test-run", "project_scope": publication_scope(),
             "user_query": plan.question, "research_plan": plan.model_dump(mode="json")}
    delta = _run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters={"local_direct_text": _adapter([_direct_evidence()])},
    ))

    # 1. 真实执行增量走完整链路：execute 阶段的四态出现在回合终态里
    outcome = build_turn_outcome({**state, **delta}, memory_writeback={"status": "skipped"})
    assert outcome["capability_results"]["local_direct_text"] == "success"

    # 2. 四种取值各自独立进表，取值不在词表的整条不出现
    four_states = {
        "local_direct_text": "success",
        "local_metadata": "empty",
        "local_structure_hint": "skipped",
        "local_section_structure": "failed",
        "external_context_needed": "unknown",
    }
    capped = build_turn_outcome(
        {**state, **delta, "planner_evidence_capabilities": four_states},
        memory_writeback={"status": "skipped"},
    )
    assert {
        name: capped["capability_results"][name]
        for name in four_states
        if name != "external_context_needed"
    } == {
        name: status
        for name, status in four_states.items()
        if name != "external_context_needed"
    }
    assert "external_context_needed" not in capped["capability_results"]


@dataclass
class _Settings:
    rag_research_planner_max_units: int = 8


@dataclass
class _FakeLLM:
    def with_structured_output(self, model: Any) -> "_StructuredOutput":
        return _StructuredOutput(model=model)


@dataclass
class _StructuredOutput:
    model: Any

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self.model is ResearchClaimStrategyDraftBatch:
            evidence_ids = _evidence_ids_in_prompt(messages)
            claims = [_claim(evidence_ids[0])] if evidence_ids else []
            return ResearchClaimStrategyDraftBatch.model_validate({"claims": [claim.model_dump() for claim in claims]})
        raise AssertionError(f"unexpected structured output model: {self.model}")


def _evidence_ids_in_prompt(messages: list[Any]) -> list[str]:
    """从 claim 提示词里解析在场证据 id，模拟只引用喂进来的证据的模型行为。"""
    ids: list[str] = []
    for message in messages:
        for line in str(getattr(message, "content", "")).splitlines():
            stripped = line.strip()
            if stripped.startswith("id: ") and stripped[len("id: "):].strip() not in ids:
                ids.append(stripped[len("id: "):].strip())
    return ids


def _claim(evidence_id: str) -> ResearchClaimDraft:
    return ResearchClaimDraft(
        text="Scoped evidence supports this finding.",
        claim_type="paper_supported",
        supporting_evidence_ids=[evidence_id],
    )


async def _draft_from_evidence(
    unit: ResearchUnit,
    evidence: list[dict[str, Any]],
    _dependencies: list[Any],
    *, read_reports=None) -> list[ResearchClaimDraft]:
    """有证据才出主张；无证据时留空让单元如实判取证不足。"""
    if not evidence:
        return []
    return [_claim(str(evidence[0]["evidence_id"]))]


def _reasoner() -> ResearchUnitReasoner:
    return ResearchUnitReasoner(draft_provider=_draft_from_evidence)


def _adapter(records: list[dict[str, Any]], calls: list[str] | None = None):
    async def run(unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        if calls is not None:
            calls.append(unit.unit_id)
        return read_result(records)

    return run


def _raising_adapter(exc: Exception):
    async def run(_unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        raise exc

    return run


def _direct_evidence() -> dict[str, Any]:
    """无参数；返回带真实来源范围和稳定身份的正文测试材料。"""
    return direct_evidence("e1")


def _metadata_evidence() -> dict[str, Any]:
    return {"evidence_id": "m1", "source": "inventory", "layer": "metadata", "paper_id": "paper-a", "summary": "Paper A"}


def _metadata_unit(
    unit_id: str,
    *,
    facets: list[str],
    deps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    unit = make_unit(
        unit_id,
        source_type="local_metadata",
        allowed_claims=["evidence_grounded_inference"],
        deps=deps,
        facets=facets,
    )
    unit["claim_policy"]["citation_required"] = False
    return unit


def _dep(unit_id: str, *, required_claims: list[str]) -> dict[str, Any]:
    return {"unit_id": unit_id, "required": True, "required_claim_types": required_claims, "reason": "needed"}


def _plan(units: list[dict[str, Any]], *, facets: list[str] | None = None) -> ResearchPlan:
    facet_ids = facets or [f"f{index}" for index in range(1, len(units) + 1)]
    result = validate_strategy_fixture(make_draft(units, explicit_facets=[make_facet(f) for f in facet_ids]))
    assert result.plan is not None
    return result.plan


def _raw_plan(units: list[dict[str, Any]]) -> ResearchPlan:
    return current_plan_fixture(make_draft(units))


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="How do the papers use attention?")


def _run(awaitable):
    return asyncio.run(awaitable)
