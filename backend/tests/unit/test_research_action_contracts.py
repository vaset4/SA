"""【研究动作】【依赖边界】模型适用参数与内部执行投影的离线回归。作者：xxx。"""

from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from app.models.research_action_contracts import AnalyzeMaterials, InitialResearchDecision, ResearchDecision
from app.models.research_planner_contracts import PlannerCoverageReview, ResearchUnitResult
from app.services.internal_research.research_action_validation import _project_action, action_work_keys
from app.services.internal_research.research_planner_fulfillment import evaluate_dependencies
from app.services.internal_research.research_planner_plan_prompt import build_action_prompt


def analysis_payload(dependency):
    """参数为人工构造的依赖；返回完整后续分析决策，不冒充真实模型响应。"""
    return {"decision": {"kind": "execute", "actions": [{"goal": "分析已有成果", "input": {
        "source_type": "inference_from_units", "dependencies": [dependency]}}]}}


@pytest.mark.parametrize("types", [[], ["paper_supported"]])
def test_material_dependency_rejects_even_empty_claim_requirement(types):
    """参数为结论类型列表；材料依赖连空类型选项也不应暴露或接受。"""
    raw = analysis_payload(dict(unit_id="u1", required=True, required_product="materials", required_claim_types=types))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ResearchDecision.model_validate(raw, strict=True)


@pytest.mark.parametrize("product, claim_type", [
    ("claims", "model_prior_hypothesis"), ("claims", "memory_continuation"),
    ("claims", "external_needed"), ("hypotheses", "paper_supported"),
    ("hypotheses", "evidence_grounded_inference"), ("hypotheses", "external_needed"),
])
def test_dependency_rejects_inapplicable_claim_types(product, claim_type):
    """参数为产物和不适用类型；不能把假设、事实与外部缺口互相替代。"""
    raw = analysis_payload(dict(unit_id="u1", required=True, required_product=product, required_claim_types=[claim_type]))
    with pytest.raises(ValidationError, match="literal_error"):
        ResearchDecision.model_validate(raw, strict=True)


@pytest.mark.parametrize("product, types", [
    ("materials", None), ("claims", []), ("hypotheses", []),
    ("claims", ["paper_supported", "evidence_grounded_inference"]),
    ("hypotheses", ["model_prior_hypothesis", "memory_continuation"]),
])
@pytest.mark.parametrize("required", [False, True])
def test_valid_dependency_preserves_internal_semantics(product, types, required):
    """参数为合法输入和可选性；投影保留用途与类型要求，材料内部采用既有空列表。"""
    dependency = dict(unit_id="u1", required=required, required_product=product, reason="用于当前分析")
    if types is not None:
        dependency["required_claim_types"] = types
    draft = ResearchDecision.model_validate(analysis_payload(dependency), strict=True)
    action = draft.decision.actions[0]
    projected = _project_action(action, "u2", PlannerCoverageReview())
    assert projected.dependencies[0].model_dump() == {
        **dependency, "required_claim_types": [] if types is None else types, "selections": None}
    if product == "materials":
        assert "required_claim_types" not in action.input.dependencies[0].model_dump()


def test_dependency_schema_exposes_product_specific_fields():
    """无参数；给模型的模式按产物判别，材料分支不暴露结论类型。"""
    schema = AnalyzeMaterials.model_json_schema()
    items = schema["properties"]["dependencies"]["items"]
    assert items["discriminator"]["propertyName"] == "required_product"
    material_ref = items["discriminator"]["mapping"]["materials"].rsplit("/", 1)[-1]
    assert "required_claim_types" not in schema["$defs"][material_ref]["properties"]


@pytest.mark.parametrize("kind", ["execute", "finish", "clarify"])
def test_initial_and_followup_keep_field_ownership(kind):
    """参数为三种决策；研究意图留在根部，已接受要求不能在后续覆盖。"""
    decisions = {
        "execute": analysis_payload(dict(unit_id="u1", required=True, required_product="materials"))["decision"],
        "finish": dict(kind="finish", reason="研究完成"),
        "clarify": dict(kind="clarify", clarification=dict(needed=True, question="研究哪篇论文？", reason_code="paper_unclear")),
    }
    raw = dict(decision=decisions[kind], research_intent="后续研究思路")
    ResearchDecision.model_validate(raw, strict=True)
    facet = dict(facet_id="f", label="问题", source="explicit", user_text_span="问题", required=True,
                 deliverable_kind="materials")
    InitialResearchDecision.model_validate({**raw, "requested_facets": [facet]}, strict=True)
    for key, value in (("requested_facets", [facet]), ("answer_sections", [])):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ResearchDecision.model_validate({**raw, key: value}, strict=True)
    misplaced = deepcopy(raw)
    misplaced["decision"]["research_intent"] = raw["research_intent"]
    for schema in (ResearchDecision, InitialResearchDecision):
        candidate = misplaced if schema is ResearchDecision else {**misplaced, "requested_facets": [facet]}
        with pytest.raises(ValidationError, match="extra_forbidden"):
            schema.model_validate(candidate, strict=True)


def test_research_intent_does_not_change_actual_work_identity():
    """无参数；只改研究思路不能伪造选中依赖的新执行进展。"""
    raw = analysis_payload(dict(unit_id="u1", required=True, required_product="materials"))
    before = action_work_keys(raw, {})
    assert action_work_keys({**raw, "research_intent": "改写解释"}, {}) == before


def test_initial_and_followup_prompts_use_accepted_requirements():
    """无参数；首次建立要求，后续只消费已接受要求，新提示不再请求独立研究意图。"""
    first = build_action_prompt({})
    later = build_action_prompt({"coverage_review": {"explicit_facets": []}})
    assert "First identify real requested_facets" in first
    assert "First identify real requested_facets" not in later
    assert "Do not resubmit requested_facets or answer_sections" in later
    assert "research_intent" not in first + later


def test_followup_prompt_retains_rejected_support_with_accepted_claims():
    """无参数；部分结论通过时，后续规划仍收到同一单元的具体引用拒绝原因，无返回值。"""
    # 1. 【研究循环】【拒绝反馈】人工构造已读材料、已接受结论与被通用履约缺口覆盖的引用错误
    evidence_id = "ev-r1-f3323f05fa9ecbd1206fabfe255d1887"
    reason = f"support_quote_mismatch:{evidence_id}"
    accepted_claim = {
        "claim_id": "u3.c1", "text": "X 与 Y 的通道使用同一随机置换",
        "claim_type": "paper_supported", "validation_status": "validated",
        "supporting_evidence_ids": [evidence_id], "addressed_facet_ids": ["facet_xy_handling"],
    }
    partial_result = ResearchUnitResult.model_validate({
        "unit_id": "u3", "status": "insufficient_evidence", "claims": [accepted_claim],
        "unresolved_reason": "required facet object lacks eligible output",
        "metadata": {"rejected_claims": [reason], "result_input_signature": "internal-signature"},
    }).model_dump()
    read_result = ResearchUnitResult(unit_id="u2", status="evidence_found", evidence_ids=[evidence_id]).model_dump()
    state = {
        "coverage_review": {"explicit_facets": []},
        "research_unit_results": [read_result, partial_result],
    }
    original_state = deepcopy(state)

    # 2. 【研究循环】【拒绝反馈】真实提示保留错误归属及合法成果，不泄露整份内部元数据
    prompt = build_action_prompt(state)
    result_text = prompt.split("Actual results:\n", 1)[1].split("\nMaterial directory", 1)[0]
    observations = {item["unit_id"]: item for item in json.loads(result_text)}
    assert observations["u3"].get("rejected_claims") == [reason]
    projected = observations["u3"]["claims"][0]
    assert all(projected[key] == value for key, value in accepted_claim.items())
    assert "supporting_spans" not in projected
    assert observations["u3"]["unresolved_reason"] == partial_result["unresolved_reason"]
    assert observations["u3"]["status"] == "insufficient_evidence"
    assert observations["u2"].get("rejected_claims", []) == []
    assert observations["u2"]["evidence_ids"] == [evidence_id]
    assert "internal-signature" not in prompt
    assert "First identify real requested_facets" not in prompt
    assert "rejected citation, span or quote" in prompt
    assert "does not itself mean material is missing" in prompt
    assert "already-read material" in prompt
    assert state == original_state


@pytest.mark.parametrize("metadata", [{}, {"rejected_claims": []}])
def test_clean_action_results_do_not_invent_rejections(metadata):
    """参数为没有拒绝或显式空拒绝的元数据；干净结果不产生虚构错误，无返回值。"""
    result = ResearchUnitResult(unit_id="u1", status="evidence_found", metadata=metadata).model_dump()
    prompt = build_action_prompt({"research_unit_results": [result]})
    result_text = prompt.split("Actual results:\n", 1)[1].split("\nMaterial directory", 1)[0]
    observation = json.loads(result_text)[0]
    assert observation["unit_id"] == "u1"
    assert observation.get("rejected_claims", []) == []
    assert "support_quote_mismatch" not in prompt


@pytest.mark.parametrize("product", ["materials", "claims", "hypotheses"])
@pytest.mark.parametrize("required", [False, True])
def test_projected_dependency_optionality_controls_real_consumption(product, required):
    """参数为三类产物和必需性；真实失败结果只在必需依赖时阻断，且不会提供虚构成果。"""
    dependency = dict(unit_id="u1", required=required, required_product=product)
    draft = ResearchDecision.model_validate(analysis_payload(dependency), strict=True)
    projected = _project_action(draft.decision.actions[0], "u2", PlannerCoverageReview())
    result = ResearchUnitResult(unit_id="u1", status="blocked", unresolved_reason="上游读取已明确失败")
    outcome = evaluate_dependencies(projected, {"u1": result})
    assert outcome.satisfied is (not required)
    assert bool(outcome.blocking_reasons) is required
    assert outcome.validated_results == ()
    assert outcome.hypothesis_results == ()
    assert outcome.material_records == ()
