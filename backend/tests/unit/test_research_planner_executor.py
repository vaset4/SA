"""Planner unit executor tests."""

from __future__ import annotations

from o2_evidence_test_utils import read_result

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.models.research_planner_contracts import (
    EvidencePolicy,
    ResearchClaimDraft,
    ResearchPlan,
    ResearchUnit,
    ResearchUnitResult,
)
from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials
from o2_evidence_test_utils import material
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError,
    EvidenceCapabilitySoftError,
    PlannerExecutionContext,
    PlannerUnitExecutor,
    _topological_unit_layers,
)
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit, supporting_spans, explicit_claim_provider


def test_external_unit_without_adapter_is_blocked_visibly() -> None:
    """external 走正常取证路径：适配器不在册时 blocked 原因要点名缺哪个适配器。"""
    plan = _plan([_external_unit()])
    reasoner = StaticReasoner({"u1": [_hypothesis()]})
    results = _run(PlannerUnitExecutor(reasoner=_reasoner(reasoner)).execute_plan(plan))
    assert results[0].status == "blocked"
    assert "no adapter registered for external_context_needed" in results[0].unresolved_reason
    assert results[0].claims == []
    assert reasoner.calls == []


def test_no_external_retrieval_when_adapter_returns_empty() -> None:
    """external 适配器在册就真调用：空命中如实落 empty 四态，不冒充成功。"""
    plan = _plan([_external_unit()])
    calls: list[str] = []
    executor = PlannerUnitExecutor(adapters={"external_context_needed": _adapter([], calls, "external")})
    report = _run(executor.execute_with_evidence(plan))
    assert report.capability_results["external_context_needed"] == "empty"
    assert report.results[0].status == "insufficient_evidence"
    assert calls == ["external"]


def test_external_needed_visible_in_result() -> None:
    """external 适配器故障软失败：单元 blocked、四态 failed，原因可读不吞错。"""
    plan = _plan([_external_unit()])
    reasoner = StaticReasoner({"u1": [_hypothesis()]})

    async def failing_adapter(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        raise EvidenceCapabilitySoftError("external_context_needed", "arXiv API unavailable")

    executor = PlannerUnitExecutor(
        adapters={"external_context_needed": failing_adapter},
        reasoner=_reasoner(reasoner),
    )
    report = _run(executor.execute_with_evidence(plan))
    assert report.capability_results["external_context_needed"] == "failed"
    assert report.results[0].status == "blocked"
    assert "arXiv API unavailable" in report.results[0].unresolved_reason
    assert reasoner.calls == []


def test_dispatch_table_extensible() -> None:
    calls: list[str] = []
    executor = PlannerUnitExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider)).with_adapter("local_direct_text", _adapter([_direct_evidence()], calls, "direct"))
    result = _run(executor.execute_plan(_plan(), context=_ctx(),))
    assert result[0].status == "evidence_found"
    assert calls == ["direct"]


def test_schema_has_no_external_execution_field() -> None:
    fields = EvidencePolicy.model_fields
    assert "external_execution_enabled" not in fields


def test_evidence_record_identity_direct_text() -> None:
    record = convert_knowledge_materials((material(),))[0]
    reasoner = StaticReasoner({"u1": [_supported(record["evidence_id"])]})
    executor = PlannerUnitExecutor(
        adapters={"local_direct_text": _adapter([record], [], "local_direct_text")},
        reasoner=_reasoner(reasoner),
    )
    first = _run(executor.execute_with_evidence(_plan(), context=_ctx()))
    second = _run(executor.execute_with_evidence(_plan(), context=_ctx()))
    assert first.evidence_records
    assert first.evidence_records[0]["evidence_id"] == second.evidence_records[0]["evidence_id"]
    assert first.results[0].claims[0].supporting_evidence_ids == [record["evidence_id"]]


def test_scoped_direct_text_rejects_missing_paper_id() -> None:
    """无参数；实时正文缺论文身份是合同错误，不能经范围过滤悄悄变成空资料。"""
    record = {**_direct_evidence(), "paper_id": ""}
    with pytest.raises(EvidenceCapabilityError, match="论文"):
        _run(_executor(direct_evidence=[record]).execute_plan(_plan(), context=_ctx()))


def test_policy_metadata() -> None:
    result = _run(_executor().execute_plan(_plan([_metadata_unit()]), context=_ctx()))[0]
    assert result.status == "evidence_found"
    assert result.claims[0].claim_type == "evidence_grounded_inference"
    assert result.claims[0].supporting_evidence_ids == ["m1"]


def test_neo4j_verification_gate() -> None:
    verified_unit = _structure_unit(verify_with_direct_text=True)
    report = _run(_executor().execute_with_evidence(_plan([verified_unit]), context=_ctx()))
    result = report.results[0]
    records = {record["evidence_id"]: record for record in report.evidence_records}
    assert result.status == "evidence_found"
    assert result.claims[0].claim_type == "paper_supported"
    assert result.claims[0].supporting_evidence_ids == [_direct_evidence()["evidence_id"]]
    assert records["g1"]["verification_status"] == "unverified"
    assert result.metadata["structure_verification"] == {
        "status": "verified", "hint_evidence_ids": ["g1"], "direct_evidence_ids": [_direct_evidence()["evidence_id"]],
    }
    assert records["g1"]["usable_as_citation"] is False
    failed = _run(_executor(direct_evidence=[]).execute_with_evidence(_plan([verified_unit]), context=_ctx()))
    assert failed.results[0].status == "insufficient_evidence"
    assert failed.results[0].claims == []
    assert failed.evidence_records[0]["verification_status"] == "unverified"


def test_structure_verification_preserves_required_inference_contract() -> None:
    """无参数；明确要求有依据推断的结构核实单元不能被内部强改成论文事实。"""
    unit = _structure_unit(verify_with_direct_text=True)
    unit["claim_policy"]["allowed_claim_types"] = ["evidence_grounded_inference"]
    unit["expected_output"]["required_claim_types"] = ["evidence_grounded_inference"]
    drafts = StaticReasoner({"u1": [_metadata_claim(_direct_evidence()["evidence_id"])]})
    executor = _executor()
    executor = PlannerUnitExecutor(adapters=executor.adapters, reasoner=_reasoner(drafts))
    result = _run(executor.execute_with_evidence(_plan([unit]), context=_ctx())).results[0]
    assert result.status == "evidence_found"
    assert result.claims[0].claim_type == "evidence_grounded_inference"


def test_policy_inference_from_units() -> None:
    units = [
        make_unit("a", facets=["f1"]),
        _inference_unit("b", deps=[_dep("a", required_claims=["paper_supported"])]),
    ]
    results = _run(_executor().execute_plan(_plan(units, facets=["f1", "f2"]), context=_ctx()))
    assert [result.status for result in results] == ["evidence_found", "inferred"]
    assert results[1].claims[0].source_unit_ids == ["u1"]


def test_dependency_chain_reaches_final_recommendation_claim() -> None:
    records = [
        _direct_evidence("e1", "paper-a"),
        _direct_evidence("e2", "paper-b"),
        _direct_evidence("e3", "paper-c"),
    ]
    synthesis = _inference_unit(
        "d",
        deps=[
            _dep("a", required_claims=["paper_supported"]),
            _dep("b", required_claims=["paper_supported"]),
            _dep("c", required_claims=["paper_supported"]),
        ],
        facets=["f4"],
    )
    synthesis["goal"] = "Synthesize cross-paper evidence"
    synthesis["research_action"] = "compare validated claims from source papers"
    recommendation = _inference_unit(
        "e",
        deps=[_dep("d", required_claims=["evidence_grounded_inference"])],
        facets=["f5"],
        output_role="final_recommendation",
    )
    recommendation["goal"] = "Recommend a research direction"
    recommendation["research_action"] = "turn synthesis claims into a concrete research direction"
    units = [
        make_unit(
            "a",
            goal="Verify paper A evidence",
            action="locate paper A evidence",
            paper_ids=["paper-a"],
            facets=["f1"],
        ),
        make_unit(
            "b",
            goal="Verify paper B evidence",
            action="locate paper B evidence",
            paper_ids=["paper-b"],
            facets=["f2"],
        ),
        make_unit(
            "c",
            goal="Verify paper C evidence",
            action="locate paper C evidence",
            paper_ids=["paper-c"],
            facets=["f3"],
        ),
        synthesis,
        recommendation,
    ]
    reasoner = StaticReasoner({
        "u1": [_supported(_direct_evidence("e1", "paper-a")["evidence_id"])],
        "u2": [_supported(_direct_evidence("e2", "paper-b")["evidence_id"])],
        "u3": [_supported(_direct_evidence("e3", "paper-c")["evidence_id"])],
        "u4": [_inference_claim(["u1", "u2", "u3"])],
        "u5": [_inference_claim(["u4"])],
    })

    plan = _plan(units, facets=["f1", "f2", "f3", "f4", "f5"])
    results = _run(_executor(reasoner=reasoner, direct_evidence=records).execute_plan(plan, context=_ctx()))

    assert [result.status for result in results] == [
        "evidence_found",
        "evidence_found",
        "evidence_found",
        "inferred",
        "inferred",
    ]
    assert results[3].claims[0].source_unit_ids == ["u1", "u2", "u3"]
    assert results[4].claims[0].source_unit_ids == ["u4"]
    assert plan.units[4].output_role == "final_recommendation"


def test_dispatch_by_source_type_not_family() -> None:
    calls: list[str] = []
    unit = _metadata_unit(operation_family="comparison")
    result = _run(_executor(calls=calls).execute_plan(_plan([unit]), context=_ctx()))[0]
    assert result.status == "evidence_found"
    assert calls == ["local_metadata"]


def test_no_fallback_when_evidence_is_missing() -> None:
    executor = _executor(direct_evidence=[])
    result = _run(executor.execute_plan(_plan(), context=_ctx()))[0]
    assert result.status == "insufficient_evidence"
    assert "no evidence found" in result.unresolved_reason


def test_single_target_answer_passes_within_multi_paper_scope() -> None:
    # 单一对象问题（如"STD-PLM 有什么创新点"）锁定了项目内多篇论文
    # 只有相关的那一篇出证据，绑定范围内其余论文没提到，仍应取证成功而非判不足
    unit = make_unit(paper_ids=["paper-a", "paper-b", "paper-c"], allowed_claims=["paper_supported"])
    executor = _executor(direct_evidence=[_direct_evidence("e1", "paper-a")])

    result = _run(executor.execute_plan(_plan([unit]), context=_ctx()))[0]

    assert result.status == "evidence_found"
    assert result.claims[0].claim_type == "paper_supported"


def test_multi_target_still_insufficient_when_scope_has_no_evidence() -> None:
    # 绑定范围内没有任何一篇出证据时，仍应显式判取证不足，不因软化而放行
    unit = make_unit(paper_ids=["paper-a", "paper-b"], allowed_claims=["paper_supported"])
    executor = _executor(direct_evidence=[])

    result = _run(executor.execute_plan(_plan([unit]), context=_ctx()))[0]

    assert result.status == "insufficient_evidence"


def test_creative_unit_evidence_first() -> None:
    unit = make_unit(
        action="propose a stability hypothesis from local evidence",
        allowed_claims=["model_prior_hypothesis"],
        operation_family="creative_analysis",
    )
    reasoner = StaticReasoner({"u1": [_hypothesis()]})
    executor = _executor(reasoner=reasoner)
    result = _run(executor.execute_plan(_plan([unit]), context=_ctx()))[0]
    assert result.status == "evidence_found"
    assert reasoner.calls[0]["evidence_count"] == 1
    assert result.claims[0].claim_type == "model_prior_hypothesis"


def test_unit_result_from_execution_not_llm() -> None:
    result = _run(_executor().execute_plan(_plan(), context=_ctx()))[0]
    assert isinstance(result, ResearchUnitResult)
    assert result.metadata["source_type"] == "local_direct_text"


def test_adapter_gating_policy_to_path_mapping() -> None:
    calls: list[str] = []
    units = [make_unit("a", facets=["f1"]), _metadata_unit("b", facets=["f2"]), _structure_unit("c", facets=["f3"])]
    _run(_executor(calls=calls).execute_plan(_plan(units, facets=["f1", "f2", "f3"]), context=_ctx()))
    # 三个单元互无依赖，同层并发执行，三条路的先后不再固定；
    # 本用例要证明的是「各自走对的那一条且只走一次」，不是走的顺序
    # 结构线索单元没索取正文验证，就只走图库这一条路，不额外发正文检索
    assert sorted(calls) == ["local_direct_text", "local_metadata", "local_structure_hint"]


def test_structure_hint_asking_for_verification_adds_a_direct_text_hop() -> None:
    calls: list[str] = []
    units = [_structure_unit("c", facets=["f3"], verify_with_direct_text=True)]
    _run(_executor(calls=calls).execute_plan(_plan(units, facets=["f3"]), context=_ctx()))
    # 计划索取正文验证时才多出这一跳正文检索，与上面那条用例构成对照
    assert calls == ["local_structure_hint", "local_direct_text"]


def test_inference_no_retrieval() -> None:
    calls: list[str] = []
    units = [make_unit("a", facets=["f1"]), _inference_unit("b", deps=[_dep("a")])]
    results = _run(_executor(calls=calls).execute_plan(_plan(units, facets=["f1", "f2"]), context=_ctx()))
    assert results[1].status == "inferred"
    assert results[1].evidence_ids == []
    assert calls == ["local_direct_text"]


def test_inference_blocked_on_insufficient_dep() -> None:
    units = [make_unit("a", facets=["f1"]), _inference_unit("b", deps=[_dep("a", required_claims=["paper_supported"])])]
    results = _run(_executor(direct_evidence=[]).execute_plan(_plan(units, facets=["f1", "f2"]), context=_ctx()))
    assert results[0].status == "insufficient_evidence"
    assert results[1].status == "blocked"


def test_external_context_with_arxiv_records_stays_unresolved_not_crash() -> None:
    """external 拿到记录但记录不可引用：不产 paper_supported 主张，也不硬失败。"""
    records = [{
        "evidence_id": "ev-arxiv-1",
        "source": "arxiv",
        "layer": "metadata",
        "paper_id": "2401.12345",
        "summary": "A paper about attention.",
        "excerpt": "A paper about attention.",
        "usable_as_citation": False,
        "provenance": "arxiv_search",
    }]
    calls: list[str] = []
    executor = PlannerUnitExecutor(
        adapters={"external_context_needed": _adapter(records, calls, "external")},
        reasoner=_reasoner(StaticReasoner({"u1": [_supported("ev-arxiv-1")]})),
    )
    report = _run(executor.execute_with_evidence(_plan([_external_unit()]), context=_ctx()))
    assert report.capability_results["external_context_needed"] == "success"
    # arXiv 记录不可引用：引用级主张被压掉，单元按证据不足如实收尾
    assert report.results[0].status == "insufficient_evidence"
    assert calls == ["external"]


def test_no_metadata_promoted_to_citation() -> None:
    result = _run(_executor().execute_plan(_plan([_metadata_unit()]), context=_ctx()))[0]
    assert result.claims[0].claim_type == "evidence_grounded_inference"
    assert all(claim.claim_type != "paper_supported" for claim in result.claims)


def test_topological_layers_splits_ready_layer_from_dependent_layer() -> None:
    """分层只把「互不依赖」的单元放进同一层，有依赖的单元必须落到后一层。"""
    units = _plan([
        make_unit("a", facets=["f1"]),
        _metadata_unit("b", facets=["f2"]),
        _inference_unit("c", deps=[_dep("a", required_claims=["paper_supported"])], facets=["f3"]),
    ], facets=["f1", "f2", "f3"]).units
    layers = _topological_unit_layers(units)
    assert [[unit.unit_id for unit in layer] for layer in layers] == [["u1", "u2"], ["u3"]]


def test_same_layer_units_run_concurrently() -> None:
    """同层两个单元互相等待：串行实现会死等超时，只有真并发才走得通。"""
    calls: list[str] = []
    gate = asyncio.Event()
    base = _adapter([_direct_evidence()], calls, "a")

    async def adapter_a(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        return await base(unit, context)

    async def adapter_b(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        # b 拿到执行名额说明 a 已经在跑；a 不先返回，b 就永远等不到这把闸门
        calls.append("b")
        await asyncio.wait_for(gate.wait(), timeout=1.0)
        return read_result([_metadata_evidence()])

    async def adapter_c(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        calls.append("c")
        gate.set()
        return read_result([_structure_evidence()])

    units = [make_unit("a", facets=["f1"]), _metadata_unit("b", facets=["f2"]), _structure_unit("c", facets=["f3"])]
    executor = PlannerUnitExecutor(
        adapters={
            "local_direct_text": adapter_a,
            "local_metadata": adapter_b,
            "local_structure_hint": adapter_c,
        },
        reasoner=_reasoner(StaticReasoner({})),
    )
    results = _run(executor.execute_plan(_plan(units, facets=["f1", "f2", "f3"]), context=_ctx()))
    assert [result.status for result in results] == ["evidence_found", "evidence_found", "evidence_found"]
    assert sorted(calls) == ["a", "b", "c"]


@pytest.mark.parametrize("parallel_limit", [0, 2])
def test_dependent_layer_waits_for_previous_layer(parallel_limit) -> None:
    """参数为并发上限；有无上限都须等待前一依赖层完成才能开始后一层，无返回值。"""
    finished: set[str] = set()
    observed: dict[str, bool] = {}

    async def adapter(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        observed[unit.unit_id] = "u1" in finished
        await asyncio.sleep(0)
        finished.add(unit.unit_id)
        return read_result([_direct_evidence()])

    units = [
        make_unit("a", facets=["f1"]),
        make_unit(
            "b",
            goal="Verify the dependent paper evidence",
            action="locate dependent paper evidence",
            deps=[_dep("a", required_claims=["paper_supported"])],
            facets=["f2"],
        ),
    ]
    plan = _plan(units, facets=["f1", "f2"])
    with patch.object(get_settings(), "rag_planner_max_parallel_units", parallel_limit):
        report = _run(asyncio.wait_for(PlannerUnitExecutor(
            adapters={"local_direct_text": adapter},
            reasoner=_reasoner(StaticReasoner({"u1": [_supported(_direct_evidence("e1", "paper-a")["evidence_id"])]})),
        ).execute_with_evidence(plan, context=_ctx()), timeout=1))
    # 跨层串行：u2 起步时 u1 必然已经完成；u1 无前序可等
    assert observed == {"u1": False, "u2": True}
    assert [result.unit_id for result in report.results] == ["u1", "u2"]


@pytest.mark.parametrize("parallel_limit,expected_peak", [(0, 2), (1, 1), (2, 2)])
def test_parallel_unit_count_is_capped_by_setting(parallel_limit, expected_peak) -> None:
    """参数为并发配置和预期峰值；0 放开同层并发，正数继续限制，返回值无。"""
    units = [make_unit("a", facets=["f1"]), _metadata_unit("b", facets=["f2"])]
    plan = _plan(units, facets=["f1", "f2"])
    in_flight = 0
    peak = 0

    async def adapter(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return read_result([_direct_evidence()])

    executor = PlannerUnitExecutor(
        adapters={"local_direct_text": adapter, "local_metadata": adapter},
        reasoner=_reasoner(StaticReasoner({})),
    )
    with patch.object(get_settings(), "rag_planner_max_parallel_units", parallel_limit):
        _run(asyncio.wait_for(executor.execute_plan(plan, context=_ctx()), timeout=1))
        assert peak == expected_peak


def test_same_capability_registration_follows_plan_order() -> None:
    """同名能力取末次执行结果，末次按层内计划序而非完成先后：晚完成的先登记会把结果写反。"""
    units = [
        make_unit("a", facets=["f1"]),
        make_unit("b", goal="Verify the second paper evidence", action="locate second paper evidence", facets=["f2"]),
    ]
    plan = _plan(units, facets=["f1", "f2"])

    async def adapter(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        # 计划序里靠后的单元先返回、靠前的单元后返回，把「按完成先后登记」的实现照出来
        if unit.unit_id == "u1":
            await asyncio.sleep(0.05)
            return read_result([_direct_evidence()])
        return read_result([])

    executor = PlannerUnitExecutor(
        adapters={"local_direct_text": adapter},
        reasoner=_reasoner(StaticReasoner({})),
    )
    report = _run(executor.execute_with_evidence(plan, context=_ctx()))
    assert [result.unit_id for result in report.results] == ["u1", "u2"]
    assert report.capability_results["local_direct_text"] == "empty"


def test_ids_stay_stable_across_repeated_runs() -> None:
    """同一计划跑两次，证据 ID、主张 ID 与结果顺序逐条一致。"""
    plan = _plan([make_unit("a", facets=["f1"]), _metadata_unit("b", facets=["f2"])], facets=["f1", "f2"])
    executor = _executor()
    first = _run(executor.execute_with_evidence(plan, context=_ctx()))
    second = _run(executor.execute_with_evidence(plan, context=_ctx()))
    assert [result.unit_id for result in first.results] == ["u1", "u2"]
    assert [result.unit_id for result in first.results] == [result.unit_id for result in second.results]
    assert [claim.claim_id for result in first.results for claim in result.claims] == [
        claim.claim_id for result in second.results for claim in result.claims
    ]
    assert [record["evidence_id"] for record in first.evidence_records] == [
        record["evidence_id"] for record in second.evidence_records
    ]


def test_first_failure_in_layer_order_is_reraised() -> None:
    """同层两个单元都失败时，抛出的必须是层内计划序里更靠前的那个失败面。"""
    units = [make_unit("a", facets=["f1"]), _metadata_unit("b", facets=["f2"])]
    plan = _plan(units, facets=["f1", "f2"])

    async def failing(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        # 计划序里靠后的单元先抛，若按完成先后重抛，冒出来的会是它的失败面
        if unit.unit_id == "u2":
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.05)
        raise RuntimeError(f"{unit.unit_id} failed")

    executor = PlannerUnitExecutor(
        adapters={"local_direct_text": failing, "local_metadata": failing},
        reasoner=_reasoner(StaticReasoner({})),
    )
    with pytest.raises(EvidenceCapabilityError) as excinfo:
        _run(executor.execute_plan(plan, context=_ctx()))
    assert excinfo.value.source_type == "local_direct_text"
    assert "u1 failed" in excinfo.value.reason


@dataclass
class StaticReasoner:
    drafts_by_unit: dict[str, list[ResearchClaimDraft]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(
        self,
        unit: ResearchUnit,
        evidence: list[dict[str, Any]],
        dependency_results: list[ResearchUnitResult],
        *, read_reports=None, prepared_input=None) -> list[ResearchClaimDraft]:
        self.calls.append({"unit_id": unit.unit_id, "evidence_count": len(evidence)})
        drafts = self.drafts_by_unit.get(unit.unit_id, _default_drafts_for(unit))
        return [draft.model_copy(update={
            "supporting_spans": supporting_spans(evidence, draft.supporting_evidence_ids),
        }) for draft in drafts]


def _executor(
    *,
    reasoner: StaticReasoner | None = None,
    calls: list[str] | None = None,
    direct_evidence: list[dict[str, Any]] | None = None,
) -> PlannerUnitExecutor:
    call_log = calls if calls is not None else []
    return PlannerUnitExecutor(
        adapters={
            "local_direct_text": _adapter(_direct_or_default(direct_evidence), call_log, "local_direct_text"),
            "local_metadata": _adapter([_metadata_evidence()], call_log, "local_metadata"),
            "local_structure_hint": _adapter([_structure_evidence()], call_log, "local_structure_hint"),
        },
        reasoner=_reasoner(reasoner or StaticReasoner({})),
    )


def _reasoner(reasoner: StaticReasoner) -> ResearchUnitReasoner:
    return ResearchUnitReasoner(draft_provider=reasoner)


def _default_drafts_for(unit: ResearchUnit) -> list[ResearchClaimDraft]:
    source_type = unit.evidence_policy.source_type
    if source_type == "local_direct_text":
        return [_supported()]
    if source_type == "local_metadata":
        return [_metadata_claim("m1")]
    if source_type == "local_structure_hint":
        return [_metadata_claim("g1")]
    if source_type == "inference_from_units":
        return [_inference_claim("u1")]
    return []


def _run(awaitable):
    return asyncio.run(awaitable)


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="How do the papers use attention?")


def _plan(units: list[dict[str, Any]] | None = None, *, facets: list[str] | None = None) -> ResearchPlan:
    unit_list = units or [make_unit(facets=["f1"])]
    facet_ids = facets or ["f1"]
    result = validate_strategy_fixture(make_draft(unit_list, explicit_facets=[make_facet(f) for f in facet_ids]))
    assert result.plan is not None
    return result.plan


def _adapter(records: list[dict[str, Any]], calls: list[str], name: str):
    async def run(unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        calls.append(name)
        return read_result(records)

    return run


def _direct_or_default(records: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [_direct_evidence()] if records is None else records


def _direct_evidence(evidence_id: str = "e1", paper_id: str = "paper-a") -> dict[str, Any]:
    """参数为夹具定位与论文；从真实知识材料转换出完整范围合同和内容哈希身份。"""
    return convert_knowledge_materials((material(paper_id=paper_id, locator=evidence_id),))[0]


def _metadata_evidence() -> dict[str, Any]:
    return {"evidence_id": "m1", "source": "inventory", "layer": "metadata", "summary": "Paper A"}


def _structure_evidence() -> dict[str, Any]:
    return {
        "evidence_id": "g1",
        "source": "neo4j",
        "layer": "structure_hint",
        "summary": "related",
        "verification_status": "unverified",
        "usable_as_citation": False,
    }


def _supported(evidence_id: str | None = None) -> ResearchClaimDraft:
    return ResearchClaimDraft(
        text="The paper uses attention.",
        claim_type="paper_supported",
        supporting_evidence_ids=[evidence_id or _direct_evidence()["evidence_id"]],
    )


def _metadata_claim(evidence_id: str) -> ResearchClaimDraft:
    return ResearchClaimDraft(
        text="The local evidence supports a bounded inference.",
        claim_type="evidence_grounded_inference",
        supporting_evidence_ids=[evidence_id],
        reasoning_summary="The evidence summary supports this inference.",
    )


def _inference_claim(unit_ids: str | list[str]) -> ResearchClaimDraft:
    source_unit_ids = [unit_ids] if isinstance(unit_ids, str) else unit_ids
    return ResearchClaimDraft(
        text="The mechanisms can be compared from prior unit claims.",
        claim_type="evidence_grounded_inference",
        source_claim_ids=[f"{unit_id}.c1" for unit_id in source_unit_ids],
        reasoning_summary="Source units produced validated claims.",
    )


def _hypothesis() -> ResearchClaimDraft:
    return ResearchClaimDraft(
        text="This may be unstable under distribution shift.",
        claim_type="model_prior_hypothesis",
        related_evidence_ids=[_direct_evidence()["evidence_id"]],
        validation_tasks=["run a robustness experiment"],
    )


def _external_unit() -> dict[str, Any]:
    unit = make_unit(source_type="external_context_needed", allowed_claims=["external_needed"])
    unit["claim_policy"].update({"citation_required": False, "bounded_inference_allowed": False})
    return unit


def _metadata_unit(
    unit_id: str = "a",
    *,
    facets: list[str] | None = None,
    operation_family: str | None = "evidence_lookup",
) -> dict[str, Any]:
    unit = make_unit(
        unit_id,
        source_type="local_metadata",
        allowed_claims=["evidence_grounded_inference"],
        operation_family=operation_family,
    )
    unit["claim_policy"]["citation_required"] = False
    unit["covered_facet_ids"] = facets or ["f1"]
    return unit


def _structure_unit(
    unit_id: str = "a",
    *,
    facets: list[str] | None = None,
    verify_with_direct_text: bool = False,
) -> dict[str, Any]:
    allowed = ["paper_supported"] if verify_with_direct_text else ["evidence_grounded_inference"]
    unit = make_unit(unit_id, source_type="local_structure_hint", allowed_claims=allowed)
    unit["claim_policy"]["citation_required"] = False
    unit["covered_facet_ids"] = facets or ["f1"]
    # 正文核实由计划按单元索取：不写这个开关的结构线索单元只作规划已知事实，不再多发一次正文检索
    unit["evidence_policy"]["verify_with_direct_text"] = verify_with_direct_text
    return unit


def _inference_unit(
    unit_id: str = "a",
    *,
    deps: list[dict[str, Any]] | None = None,
    facets: list[str] | None = None,
    output_role: str = "synthesis",
) -> dict[str, Any]:
    unit = make_unit(
        unit_id,
        source_type="inference_from_units",
        allowed_claims=["evidence_grounded_inference"],
        deps=deps,
        facets=facets or ["f2"],
        output_role=output_role,
    )
    unit["claim_policy"]["citation_required"] = False
    return unit


def _dep(unit_id: str, *, required: bool = True, required_claims: list[str] | None = None) -> dict[str, Any]:
    return {"unit_id": unit_id, "required": required, "required_claim_types": required_claims or [], "reason": "needed"}
