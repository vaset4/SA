"""【研究规划】【任务满足】O3 实际产物与声明一致性回归。作者：xxx。"""

import asyncio
from dataclasses import replace

import pytest

from app.models.research_planner_contracts import ResearchClaim, ResearchPlan, ResearchUnit, ResearchUnitResult
from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport
from app.services.internal_research.research_planner_evidence_repair import build_evidence_sufficiency_report, execute_with_repair
from app.services.internal_research.research_planner_executor import PlannerExecutionContext, ResearchPlannerExecutor
from app.services.internal_research.research_planner_execution_state import PlannerExecutionState
from app.services.internal_research.research_planner_fulfillment import count_task_materials, evaluate_dependencies
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner, validate_claim_drafts
from app.services.internal_research.research_planner_claim_provider import build_claim_prompt
from app.services.internal_research.research_planner_validator import PlannerValidationOptions, validate_or_repair_plan
from research_planner_strategy_test_utils import validate_strategy_fixture, current_plan_fixture
from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials
from o2_evidence_test_utils import direct_evidence, material, read_result
from research_planner_test_utils import make_draft, make_facet, make_unit, result_with_exposure, supporting_spans, explicit_claim_provider


def test_empty_section_claim_task_keeps_product_gap() -> None:
    """参数无；验证零数量下限不能把空章节主张任务报告为已完成。"""
    raw = make_unit(source_type="local_section_text")
    raw["evidence_policy"]["min_citations_per_target"] = 0
    unit = ResearchUnit.model_validate(raw)
    draft = make_draft([raw])
    plan = current_plan_fixture(draft)
    report = build_evidence_sufficiency_report(
        plan=plan, results=[ResearchUnitResult(unit_id=unit.unit_id, status="evidence_found")],
        evidence_records=[], pass_index=1,
    )
    assert report.unit_reports[0].hard_gaps


def _comparison(*, split: bool = False, kind: str = "claims") -> ResearchPlan:
    """参数为拆分方式与交付种类；返回明确要求 A/B 的新版比较计划。"""
    facet = {**make_facet("f1"), "required_paper_ids": ["paper-a", "paper-b"], "deliverable_kind": kind}
    units = [make_unit("a", paper_ids=["paper-a", "paper-b"])]
    if split:
        units = [make_unit("a", goal="read A", paper_ids=["paper-a"]),
                 make_unit("b", goal="read B", paper_ids=["paper-b"])]
    for unit in units:
        unit["expected_output"]["deliverable_kind"] = kind
        if kind == "materials":
            unit["expected_output"]["required_claim_types"] = []
    return current_plan_fixture(make_draft(units, explicit_facets=[facet]))


def _result(unit: ResearchUnit, records: list[dict], *, associated: bool = True) -> ResearchUnitResult:
    """参数为单元和真实材料；返回显式关联要求的已校验测试主张。"""
    ids = [record["evidence_id"] for record in records]
    claims = [ResearchClaim(claim_id=f"{unit.unit_id}.c1", text="A supported observation", claim_type="paper_supported",
                            validation_status="validated", supporting_evidence_ids=ids, supporting_spans=supporting_spans(records),
                            addressed_facet_ids=["f1"] if associated else [])] if ids else []
    return result_with_exposure(unit, records, claims)


def test_a_quantity_does_not_hide_required_b() -> None:
    """参数无；验证 A 两份材料仍留下比较对象 B 的明确缺口。"""
    plan = _comparison()
    records = [direct_evidence("a1"), direct_evidence("a2")]
    report = build_evidence_sufficiency_report(plan=plan, results=[_result(plan.units[0], records)],
                                               evidence_records=records, pass_index=1)
    assert report.facet_reports[0].missing_paper_ids == ["paper-b"]
    assert report.missing_required_facet_ids == ["f1"]
    assert report.facet_reports[0].claim_ids


def test_split_units_collectively_fulfill_comparison() -> None:
    """参数无；验证 A/B 各一份材料可由两个局部有效单元共同满足要求。"""
    plan = _comparison(split=True)
    a, b = direct_evidence("a"), direct_evidence("b", paper_id="paper-b")
    report = build_evidence_sufficiency_report(plan=plan, results=[_result(plan.units[0], [a]), _result(plan.units[1], [b])],
                                               evidence_records=[a, b], pass_index=1)
    assert all(item.fulfilled for item in report.unit_reports)
    assert report.fulfilled_required_facet_ids == ["f1"]
    assert report.facet_reports[0].target_coverage == {"paper-a": 1, "paper-b": 1}


def test_scope_is_not_per_paper_delivery() -> None:
    """参数无；验证授权 A/B 范围不会为仅要求 A 的任务制造 B 缺口。"""
    plan = _comparison()
    facet = plan.coverage_review.explicit_facets[0].model_copy(update={"required_paper_ids": ["paper-a"]})
    plan = plan.model_copy(update={"coverage_review": plan.coverage_review.model_copy(update={"explicit_facets": [facet]})})
    record = direct_evidence()
    report = build_evidence_sufficiency_report(plan=plan, results=[_result(plan.units[0], [record])],
                                               evidence_records=[record], pass_index=1)
    assert not report.missing_required_facet_ids
    assert not report.blocking_unit_ids


def test_model_missing_facet_link_does_not_get_completed_by_ownership() -> None:
    """参数无；验证主张未声明实际响应要求时，不用单元认领自动补造已满足。"""
    plan = _comparison()
    record = direct_evidence()
    report = build_evidence_sufficiency_report(plan=plan, results=[_result(plan.units[0], [record], associated=False)],
                                               evidence_records=[record], pass_index=1)
    assert report.facet_reports[0].claim_ids == []
    assert report.missing_required_facet_ids == ["f1"]


def test_material_dependency_needs_real_records_but_no_fake_claims() -> None:
    """参数无；验证材料前置消费真实记录，空主张合法，空材料仍阻断。"""
    record = direct_evidence()
    unit = ResearchUnit.model_validate(make_unit("b", deps=[{"unit_id": "a", "required_product": "materials"}]))
    upstream = ResearchUnitResult(unit_id="a", status="evidence_found", evidence_ids=[record["evidence_id"]])
    assert evaluate_dependencies(unit, {"a": upstream}, evidence_records=[record]).satisfied
    assert not evaluate_dependencies(unit, {"a": upstream}, evidence_records=[]).satisfied


def test_claim_dependency_does_not_accept_missing_evidence_in_empty_pool() -> None:
    """参数无；验证标记 validated 的上游事实在材料池为空或缺失引用 ID 时不能作为有效前置。"""
    record = direct_evidence("dependency")
    upstream = _result(ResearchUnit.model_validate(make_unit("a")), [record])
    unit = ResearchUnit.model_validate(make_unit("b", deps=[{"unit_id": "a"}]))
    assert not evaluate_dependencies(unit, {"a": upstream}, evidence_records=[]).satisfied
    assert not evaluate_dependencies(unit, {"a": upstream}, evidence_records=[direct_evidence("other")]).satisfied
    assert evaluate_dependencies(unit, {"a": upstream}, evidence_records=[record]).satisfied


@pytest.mark.parametrize("product,satisfied", [("claims", False), ("hypotheses", True)])
def test_hypothesis_dependency_preserves_unvalidated_identity(product: str, satisfied: bool) -> None:
    """参数为消费用途与期望；验证待验证假设只能履行明确的假设前置。"""
    unit = ResearchUnit.model_validate(make_unit("b", deps=[{"unit_id": "a", "required_product": product,
                                                             "required_claim_types": ["model_prior_hypothesis"]}]))
    claim = ResearchClaim(claim_id="a.c1", text="hypothesis", claim_type="model_prior_hypothesis")
    upstream = result_with_exposure(ResearchUnit.model_validate(make_unit("a", allowed_claims=["model_prior_hypothesis"])), [], [claim])
    outcome = evaluate_dependencies(unit, {"a": upstream})
    assert outcome.satisfied is satisfied
    if satisfied:
        assert not outcome.validated_results
        assert outcome.hypothesis_results[0].claims[0].validation_status == "needs_validation"


def test_required_claim_types_use_all_and_reject_unsupported() -> None:
    """参数无；验证多个必需类型全部取得，unsupported 不作为无类型前置的有效主张。"""
    record = direct_evidence()
    upstream_unit = ResearchUnit.model_validate(make_unit("a"))
    upstream = _result(upstream_unit, [record])
    unit = ResearchUnit.model_validate(make_unit("b", deps=[{"unit_id": "a", "required_claim_types": [
        "paper_supported", "evidence_grounded_inference"]}]))
    assert not evaluate_dependencies(unit, {"a": upstream}, evidence_records=[record]).satisfied
    inference = ResearchClaim(claim_id="a.c2", text="supported inference", claim_type="evidence_grounded_inference",
                              validation_status="validated", supporting_evidence_ids=[record["evidence_id"]],
                              reasoning_summary="derived from the same material", supporting_spans=supporting_spans([record]))
    complete = upstream.model_copy(update={"claims": [*upstream.claims, inference]})
    assert evaluate_dependencies(unit, {"a": complete}, evidence_records=[record]).satisfied
    invalid = upstream.model_copy(update={"claims": [upstream.claims[0].model_copy(update={"validation_status": "unsupported"})]})
    unit = ResearchUnit.model_validate(make_unit("b", deps=[{"unit_id": "a"}]))
    assert not evaluate_dependencies(unit, {"a": invalid}, evidence_records=[record]).satisfied


def test_reasoner_blocks_before_provider_when_dependency_unsatisfied() -> None:
    """参数无；验证前置未成立时不先产生主张再由另一份报告阻断。"""
    calls = []

    def provider(*args, **kwargs):
        """参数为草拟输入；记录不应发生的调用，返回空列表。"""
        calls.append(args)
        return []

    unit = ResearchUnit.model_validate(make_unit("b", deps=[{"unit_id": "a"}]))
    result = asyncio.run(ResearchUnitReasoner(draft_provider=provider).reason(unit, [], []))
    assert result.rejected_reasons
    assert calls == []


def test_source_policy_cannot_silently_remove_required_claim_type() -> None:
    """参数无；验证元数据任务声明必须产论文事实时，在计划阶段显式失败。"""
    raw = make_unit(source_type="local_metadata")
    result = validate_strategy_fixture(make_draft([raw]))
    assert result.failure and result.failure.code == "claim_source_not_citable"


def test_declared_comparison_sources_must_cover_required_objects() -> None:
    """参数无；验证仅声明 A 读取的单元不能凭认领字符串覆盖 A/B 比较。"""
    plan = _comparison(split=True)
    draft = plan.model_dump(mode="json")
    draft["units"] = draft["units"][:1]
    draft["plan_type"] = "single_unit"
    result = validate_strategy_fixture(draft)
    assert result.failure and result.failure.code == "facet_objects_uncovered"


def test_each_comparison_object_needs_declared_claim_path() -> None:
    """参数无；验证 A 承诺论文事实、B 仅交材料且无下游主张时，不能拼成可满足的事实比较。"""
    plan = _comparison(split=True)
    raw = plan.model_dump(mode="json")
    raw["coverage_review"]["explicit_facets"][0]["required_claim_types"] = ["paper_supported"]
    raw["units"][1]["expected_output"].update({"deliverable_kind": "materials", "required_claim_types": []})
    result = validate_strategy_fixture(raw)
    assert result.failure and result.failure.code == "facet_object_output_uncovered"


def test_material_and_inference_units_collectively_fulfill_one_claim_facet() -> None:
    """参数无；同一要求的材料前置只验收所承诺的材料，下游推断才交付该要求的主张。"""
    material_unit = make_unit("materials", goal="read A for the requested finding", paper_ids=["paper-a"])
    material_unit["expected_output"].update({"deliverable_kind": "materials", "required_claim_types": []})
    inference_unit = make_unit("inference", source_type="inference_from_units", goal="infer the requested finding",
                               allowed_claims=["evidence_grounded_inference"], output_role="synthesis",
                               deps=[{"unit_id": "materials", "required_product": "materials"}])
    facet = {**make_facet("f1"), "required_paper_ids": ["paper-a"],
             "required_claim_types": ["evidence_grounded_inference"]}
    validation = validate_strategy_fixture(make_draft([material_unit, inference_unit], explicit_facets=[facet]))
    assert validation.plan is not None
    record = direct_evidence("shared-facet")

    async def adapter(_unit, _context):
        """参数为材料任务与执行上下文；返回固定且有来源范围的正文材料。"""
        return read_result([record])

    execution = asyncio.run(ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={"local_direct_text": adapter}).execute_with_evidence(
        validation.plan,
    ))
    assert [result.status for result in execution.results] == ["evidence_found", "inferred"]
    assert execution.results[0].claims == []
    material_only = build_evidence_sufficiency_report(plan=validation.plan, results=execution.results[:1],
                                                      evidence_records=execution.evidence_records, pass_index=1)
    assert material_only.unit_reports[0].fulfilled
    assert material_only.missing_required_facet_ids == ["f1"]
    report = build_evidence_sufficiency_report(plan=validation.plan, results=execution.results,
                                               evidence_records=execution.evidence_records, pass_index=1)
    assert all(item.fulfilled for item in report.unit_reports)
    assert report.fulfilled_required_facet_ids == ["f1"]


def test_required_inferred_baseline_cannot_be_lowered() -> None:
    """参数无；验证已确认推断要求纳入基准，重规划不能将其变为可选。"""
    inferred = {**make_facet("f2", "inferred"), "required": True}
    draft = make_draft([make_unit(facets=["f1", "f2"])], inferred_facets=[inferred])
    first = validate_strategy_fixture(draft)
    assert first.plan is not None
    draft["coverage_review"]["inferred_facets"][0]["required"] = False
    result = validate_strategy_fixture(draft, options=PlannerValidationOptions(requirements_baseline=first.plan.coverage_review))
    assert result.failure and result.failure.code == "invalid_plan_format"


def test_legacy_plan_not_upgraded_to_current_execution() -> None:
    """参数无；验证缺版本历史计划可读但不能直接执行，也不补造要求事实。"""
    draft = make_draft()
    del draft["contract_version"]
    historical = ResearchPlan.model_validate(draft)
    assert historical.contract_version == "research-planner-v1"
    result = validate_or_repair_plan(draft)
    assert result.failure and result.failure.code == "invalid_plan_format"


def test_current_plan_missing_dependency_product_is_not_guessed() -> None:
    """参数无；验证新版声明缺少依赖用途时显式失败，不沿用旧默认主张用途。"""
    a = make_unit("a")
    b = make_unit("b", goal="dependent work", deps=[{"unit_id": "a"}])
    del b["dependencies"][0]["required_product"]
    result = validate_strategy_fixture(make_draft([a, b]))
    assert result.failure and result.failure.code == "invalid_plan_format"


def test_optional_exploration_gaps_do_not_block_required_answer() -> None:
    """参数无；验证纯可选探索不足保留局部报告，却不增加整体必需缺口。"""
    units = [make_unit("a"), make_unit("b", goal="optional exploration", facets=["f2"])]
    plan = current_plan_fixture(make_draft(units, inferred_facets=[make_facet("f2", "inferred")]))
    record = direct_evidence()
    report = build_evidence_sufficiency_report(plan=plan, results=[_result(plan.units[0], [record])],
                                               evidence_records=[record], pass_index=1)
    assert report.unit_reports[1].hard_gaps
    assert report.blocking_unit_ids == []
    assert report.missing_required_facet_ids == []
    assert report.has_usable_results


def test_cross_entry_same_source_range_does_not_double_count() -> None:
    """参数无；验证父块和章节同源范围虽有不同材料组，只获得一份任务完成信用。"""
    parent = material("same body")
    span = {**parent.source_spans[0], "source_fragments": [{"source_item": "source-1", "source_start": 0,
            "source_end": 9, "text_start": 0, "text_end": 9}]}
    parent = replace(parent, source_spans=(span,))
    section = replace(parent, group_kind="section", group_locator="Method", material_group_id="section-group")
    records = [dict(item) for item in convert_knowledge_materials((parent, section))]
    assert count_task_materials(records) == 1
    other = {**records[1], "material_group_id": "other", "source_spans": [{**span, "paper_id": "paper-b"}]}
    assert count_task_materials([records[0], other]) == 2


def test_claims_cannot_address_unassigned_facet() -> None:
    """参数无；验证模型不能通过越界关联替其他单元宣布要求完成。"""
    unit = ResearchUnit.model_validate(make_unit())
    record = direct_evidence()
    result = validate_claim_drafts(unit, [record], [], [{"text": "fact", "claim_type": "paper_supported",
        "supporting_evidence_ids": [record["evidence_id"]], "addressed_facet_ids": ["other"]}])
    assert not result.claims
    assert "unassigned" in result.rejected_reasons[0]


@pytest.mark.parametrize("status,has_evidence", [("blocked", True), ("insufficient_evidence", True), ("evidence_found", False)])
@pytest.mark.parametrize("claim_type", ["paper_supported", "evidence_grounded_inference",
                                       "model_prior_hypothesis", "memory_continuation"])
def test_optional_invalid_dependency_cannot_be_cited_as_validated_source(
    status: str, has_evidence: bool, claim_type: str,
) -> None:
    """参数为上游状态、材料可用性与主张类型；可选前置不阻断执行，但失效旧主张不得被实际引用。"""
    record = direct_evidence("upstream")
    local_record = direct_evidence("local")
    pool = [local_record, record] if has_evidence else [local_record]
    upstream = _result(ResearchUnit.model_validate(make_unit("a")), [record]).model_copy(update={"status": status})
    unit = ResearchUnit.model_validate(make_unit("b", allowed_claims=[claim_type],
        deps=[{"unit_id": "a", "required": False, "required_product": "claims"}]))
    hypothesis = claim_type in {"model_prior_hypothesis", "memory_continuation"}
    draft = {"text": "a draft", "claim_type": claim_type, "source_claim_ids": ["a.c1"],
             "supporting_evidence_ids": [] if hypothesis else [local_record["evidence_id"]],
             "supporting_spans": [] if hypothesis else supporting_spans([local_record]),
             "reasoning_summary": "using the upstream claim", "addressed_facet_ids": ["f1"]}
    result = validate_claim_drafts(unit, [local_record], [upstream], [draft], memory_grounded_allowed=True,
                                   dependency_evidence_records=pool)
    assert not result.claims
    assert any("source_claim" in reason or "support_evidence" in reason for reason in result.rejected_reasons)
    stale = ResearchUnitResult(unit_id=unit.unit_id, status="inferred", claims=[ResearchClaim(
        **draft, claim_id="b.c1", validation_status="needs_validation" if hypothesis else "validated",
    )])
    plan = current_plan_fixture(make_draft([unit.model_dump(mode="json")]))
    report = build_evidence_sufficiency_report(plan=plan, results=[upstream, stale],
                                               evidence_records=pool, pass_index=1)
    assert report.unit_reports[0].usable_claim_ids == []


@pytest.mark.parametrize("claim_type", ["paper_supported", "model_prior_hypothesis", "memory_continuation"])
def test_claim_source_unit_cannot_point_to_its_own_result(claim_type: str) -> None:
    """参数为主张类型；主张自身归属不构成上游依据，草稿阶段就拒绝 source_unit_ids 自引用。"""
    unit = ResearchUnit.model_validate(make_unit("self", allowed_claims=[claim_type]))
    record = direct_evidence("self-source")
    draft = {"text": "a claim", "claim_type": claim_type, "source_unit_ids": [unit.unit_id],
             "supporting_evidence_ids": [record["evidence_id"]] if claim_type == "paper_supported" else [],
             "addressed_facet_ids": ["f1"]}
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="source_unit_ids"):
        validate_claim_drafts(unit, [record], [], [draft], memory_grounded_allowed=True)


@pytest.mark.parametrize("claim_type,accepted", [("model_prior_hypothesis", True), ("evidence_grounded_inference", False)])
def test_hypothesis_source_preserves_draft_validation_identity(claim_type: str, accepted: bool) -> None:
    """参数为下游主张类型和预期；合格假设前置允许继续提出假设，但不能成为已验证推断依据。"""
    unit = ResearchUnit.model_validate(make_unit("b", source_type="inference_from_units",
        allowed_claims=[claim_type], output_role="synthesis", deps=[{"unit_id": "a", "required_product": "hypotheses",
                                                                   "required_claim_types": ["model_prior_hypothesis"]}]))
    upstream = result_with_exposure(ResearchUnit.model_validate(make_unit("a", allowed_claims=["model_prior_hypothesis"])), [], [ResearchClaim(
        claim_id="a.c1", text="a hypothesis", claim_type="model_prior_hypothesis",
    )])
    draft = {"text": "another hypothesis", "claim_type": claim_type, "source_claim_ids": ["a.c1"],
             "reasoning_summary": "uses unvalidated context", "addressed_facet_ids": ["f1"]}
    result = validate_claim_drafts(unit, [], [upstream], [draft])
    assert bool(result.claims) is accepted
    if accepted:
        assert result.claims[0].validation_status == "needs_validation"
    else:
        assert "hypothesis_not_factual_support" in result.rejected_reasons[0]


def test_claim_prompt_carries_actual_facet_definitions() -> None:
    """参数无；验证真实草拟提示包含对象和要求含义，模型漏关联时仍无程序补造。"""
    plan = _comparison()
    prompt = build_claim_prompt(plan.units[0], [], [], coverage_review=plan.coverage_review)
    assert "Unit and assigned requirements" in prompt
    assert '"required_paper_ids": ["paper-a", "paper-b"]' in prompt
    assert "addressed_facet_ids" in prompt


def test_explicit_material_dependency_reaches_reasoner_without_fake_upstream_claim() -> None:
    """参数无；验证明确材料前置把真实元数据交给推理，并保持论文事实引用资格边界。"""
    record = {"evidence_id": "metadata-a", "layer": "metadata", "paper_id": "paper-a", "summary": "actual metadata"}
    unit = ResearchUnit.model_validate(make_unit("b", source_type="inference_from_units",
        allowed_claims=["evidence_grounded_inference"], output_role="synthesis",
        deps=[{"unit_id": "a", "required_product": "materials"}]))
    upstream = ResearchUnitResult(unit_id="a", status="evidence_found", evidence_ids=["metadata-a"])
    result = asyncio.run(ResearchUnitReasoner(draft_provider=explicit_claim_provider).reason(unit, [], [upstream], dependency_evidence_records=[record]))
    assert result.claims[0].claim_type == "evidence_grounded_inference"
    assert result.claims[0].supporting_evidence_ids == ["metadata-a"]
    assert result.claims[0].source_unit_ids == []


def test_real_directory_material_supports_section_read_dependency_without_citation_upgrade() -> None:
    """参数无；生产目录转换器的 structure_hint 记录可交付材料并支持章节前置，保持不可引用。"""
    from app.nodes.rag_node import _section_structure_record

    directory = _section_structure_record("paper-a", ["Method"])
    outline = make_unit("outline", source_type="local_section_structure", facets=["f1"],
                        allowed_claims=["evidence_grounded_inference"])
    outline["expected_output"].update({"deliverable_kind": "materials", "required_claim_types": []})
    body = make_unit("body", source_type="local_section_text", goal="read outlined section", facets=["f2"],
                     deps=[{"unit_id": "outline", "required_product": "materials"}])
    facets = [{**make_facet("f1"), "deliverable_kind": "materials", "required_paper_ids": ["paper-a"]}, make_facet("f2")]
    validation = validate_strategy_fixture(make_draft([outline, body], explicit_facets=facets))
    assert validation.plan is not None

    async def directory_adapter(_unit, _context):
        """参数为目录任务；返回生产转换器生成的真实记录形状。"""
        return read_result([directory])

    async def body_adapter(_unit, _context):
        """参数为章节任务；返回具备来源范围的固定正文。"""
        body_material = material(locator="section")
        records = convert_knowledge_materials((body_material,), provenance="section_path_read")
        return EvidenceReadResult(tuple(records), KnowledgeReadReport(operation="sections", status="complete",
            requested=("Method",), resolved=("Paper / Method",), read_spans=body_material.source_spans))

    executor = ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={"local_section_structure": directory_adapter,
                                                "local_section_text": body_adapter})
    report = asyncio.run(executor.execute_with_evidence(validation.plan))
    assert [result.status for result in report.results] == ["evidence_found", "evidence_found"]
    assert report.results[0].claims == []
    assert directory["usable_as_citation"] is False
    assert directory["evidence_id"] not in report.results[1].claims[0].supporting_evidence_ids


def test_blocked_material_task_cannot_be_marked_fulfilled_by_saved_records() -> None:
    """参数无；验证失败单元遗留材料仍可保留，却不使当前材料任务自动有效。"""
    plan = _comparison(kind="materials")
    records = [direct_evidence(), direct_evidence("b", paper_id="paper-b")]
    result = _result(plan.units[0], records).model_copy(update={"status": "blocked"})
    report = build_evidence_sufficiency_report(plan=plan, results=[result], evidence_records=records, pass_index=1)
    assert not report.unit_reports[0].fulfilled
    assert report.missing_required_facet_ids == ["f1"]


@pytest.mark.parametrize("error", [RuntimeError("reasoner failed after repair"), asyncio.CancelledError("cancelled after repair")])
def test_repair_failure_preserves_identity_materials_and_revokes_ordinary_dependents(error: BaseException) -> None:
    """参数为硬失败或取消；验证补读材料保留、原异常传播、普通取证下游旧主张同步失效。"""
    async def run_case():
        """参数无；运行同一离线研究的初次与补证，返回失败前原始下游成果供核对。"""
        a1, a2, b = direct_evidence("a1"), direct_evidence("a2"), direct_evidence("b", paper_id="paper-b")
        a_unit = make_unit("a", paper_ids=["paper-a"])
        a_unit["evidence_policy"]["min_citations_per_target"] = 2
        b_unit = make_unit("b", goal="consume A then read B", paper_ids=["paper-b"],
                           deps=[{"unit_id": "a", "required_product": "claims"}])
        plan = current_plan_fixture(make_draft([a_unit, b_unit]))
        context = PlannerExecutionContext(execution_state=PlannerExecutionState(evidence_records=[b]))

        async def adapter(unit, _context):
            """参数为单元与上下文；返回不同阶段实际取得的固定材料。"""
            return read_result([a2 if "::evidence_repair::" in unit.unit_id else a1])

        def provider(unit, records, _dependencies, **kwargs):
            """参数为实际推理材料；补读完成后抛出原错误，其余返回显式关联的主张草稿。"""
            if unit.unit_id == "a" and len(records) > 1:
                raise error
            return [{"text": "actual fact", "claim_type": "paper_supported",
                     "supporting_evidence_ids": [records[0]["evidence_id"]], "supporting_spans": supporting_spans(records[:1]),
                     "addressed_facet_ids": ["f1"]}]

        executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter},
                                            reasoner=ResearchUnitReasoner(draft_provider=provider))
        first = await execute_with_repair(plan=plan, executor=executor, context=context)
        stale_b = _result(plan.units[1], [b])
        previous = [first.results[0], stale_b]
        with pytest.raises(type(error)) as caught:
            await execute_with_repair(plan=plan, executor=executor, context=context, mode="repair",
                                      previous_results=previous, prior_sufficiency=first.final_sufficiency_report)
        assert caught.value is error
        report = caught.value.planner_execution_report
        assert report.results[1].claims == [] and report.results[1].status == "blocked"
        assert stale_b.claims
        ids = {record["evidence_id"] for record in report.evidence_records}
        assert a2["evidence_id"] in ids
        assert all(set(receipt["evidence_ids"]) <= ids for receipt in report.read_receipts)

    asyncio.run(run_case())


def test_successful_repair_runs_ordinary_dependent_then_inference_with_full_material_pool() -> None:
    """参数无；验证 A 补读成功后，首次被阻塞的普通 B 真取证，C 使用完整材料链产生新推断。"""
    async def run_case():
        """参数无；通过正式两阶段入口执行完整依赖链，并检查新增材料进入返回快照。"""
        calls = []
        a1, a2, b = direct_evidence("a1"), direct_evidence("a2"), direct_evidence("b", paper_id="paper-b")
        a_unit = make_unit("a", paper_ids=["paper-a"], facets=["f1"])
        a_unit["evidence_policy"]["min_citations_per_target"] = 2
        b_unit = make_unit("b", goal="read B after A", paper_ids=["paper-b"], facets=["f2"],
                           deps=[{"unit_id": "a", "required_product": "claims"}])
        c_unit = make_unit("c", goal="infer from B", source_type="inference_from_units", facets=["f3"],
                           allowed_claims=["evidence_grounded_inference"], output_role="synthesis",
                           deps=[{"unit_id": "b", "required_product": "claims"}])
        plan = current_plan_fixture(make_draft([c_unit, b_unit, a_unit],
            explicit_facets=[make_facet("f1"), make_facet("f2"), make_facet("f3")]))

        async def adapter(unit, _context):
            """参数为取证单元；按真实目标和补读阶段返回固定材料并记录调用。"""
            calls.append(unit.unit_id)
            record = b if unit.unit_id == "b" else a2 if "::evidence_repair::" in unit.unit_id else a1
            return read_result([record])

        executor = ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={"local_direct_text": adapter})
        context = PlannerExecutionContext()
        first = await execute_with_repair(plan=plan, executor=executor, context=context)
        assert calls == ["a"]
        outcome = await execute_with_repair(plan=plan, executor=executor, context=context, mode="repair",
                                            previous_results=first.results, prior_sufficiency=first.final_sufficiency_report)
        assert calls == ["a", "a::evidence_repair::p2", "b"]
        results = {result.unit_id: result for result in outcome.results}
        assert results["a"].status == results["b"].status == "evidence_found"
        assert results["c"].status == "inferred" and results["c"].claims[0].source_unit_ids == ["b"]
        assert outcome.repair_status == "resolved"
        assert b["evidence_id"] in {record["evidence_id"] for record in outcome.evidence_records}

    asyncio.run(run_case())


def test_structure_repair_reasons_with_upstream_material_registry() -> None:
    """参数无；结构补证后的再次推理保留前置主张材料，不因只传本单元正文而误阻断。"""
    async def run_case():
        """参数无；正式执行先得到 A 事实与 B 线索，再定向补读 B 正文并核实。"""
        a, b = direct_evidence("structure-upstream"), direct_evidence("structure-body", paper_id="paper-b")
        hint = {"evidence_id": "hint-b", "source": "neo4j", "layer": "structure_hint", "paper_id": "paper-b",
                "summary": "a relation to verify", "usable_as_citation": False, "verification_status": "unverified"}
        upstream = make_unit("a", paper_ids=["paper-a"], facets=["f1"])
        structure = make_unit("b", goal="verify B after A", source_type="local_structure_hint",
                              paper_ids=["paper-b"], facets=["f2"], allowed_claims=["paper_supported"],
                              deps=[{"unit_id": "a", "required_product": "claims"}])
        structure["evidence_policy"]["verify_with_direct_text"] = True
        validation = validate_strategy_fixture(make_draft([upstream, structure],
                                            explicit_facets=[make_facet("f1"), make_facet("f2")]))
        assert validation.plan is not None

        async def direct_adapter(unit, _context):
            """参数为真实读取任务；A 正常返回，B 首读为空、定向补读才得到正文。"""
            if unit.evidence_policy.target_paper_ids == ["paper-a"]:
                return read_result([a])
            return read_result([b] if "::evidence_repair::" in unit.unit_id else [])

        async def hint_adapter(_unit, _context):
            """参数为结构任务；返回保留不可引用身份的图线索。"""
            return read_result([hint])

        executor = ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider), adapters={"local_direct_text": direct_adapter,
                                                     "local_structure_hint": hint_adapter})
        context = PlannerExecutionContext()
        first = await execute_with_repair(plan=validation.plan, executor=executor, context=context)
        assert [result.status for result in first.results] == ["evidence_found", "insufficient_evidence"]
        outcome = await execute_with_repair(plan=validation.plan, executor=executor, context=context, mode="repair",
                                            previous_results=first.results, prior_sufficiency=first.final_sufficiency_report)
        assert outcome.repair_status == "resolved"
        assert outcome.results[1].status == "evidence_found"
        assert outcome.results[1].claims[0].supporting_evidence_ids == [b["evidence_id"]]
        assert outcome.results[1].metadata["structure_verification"]["status"] == "verified"
        assert {a["evidence_id"], b["evidence_id"], hint["evidence_id"]} <= {
            record["evidence_id"] for record in outcome.evidence_records
        }

    asyncio.run(run_case())
