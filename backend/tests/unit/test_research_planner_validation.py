"""【研究规划】【实时验证】旧自动改写断言改为严格拒绝或无损派生。作者：xxx。"""

from copy import deepcopy

import pytest

from app.services.internal_research import research_planner_validator as validator
from app.services.internal_research.research_planner_plan_prompt import build_planner_prompt
from research_planner_strategy_test_utils import strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit


def _draft():
    """无参数；返回含明确要求的当前策略，不携带服务端派生字段。"""
    return strategy_fixture(make_draft())


def _validate(draft, *, repair=None, papers=None):
    """参数为策略、明确回修和授权范围；返回真实验证结果。"""
    return validator.validate_or_repair_plan(draft, repair_draft=repair,
        options=validator.PlannerValidationOptions(question="text f1 text f2", allowed_paper_ids=papers))


def test_draft_validated_before_execution():
    """无参数；验证身份、计划类型、固定来源角色均由程序派生且未伪造回修。"""
    result = _validate(_draft())
    assert result.is_valid
    assert result.plan.units[0].unit_id == "u1"
    assert result.plan.units[0].output_role == "evidence"
    assert result.plan.plan_type == "single_unit"
    assert not result.plan.metadata.repairs
    assert not result.plan.metadata.repair_attempted


def test_explicit_repair_preserves_both_observations(monkeypatch):
    """参数为观察替身；缺要求覆盖后的明确回修保留首次失败和成功两个事实。"""
    calls = []
    monkeypatch.setattr(validator, "record_plan_validation", lambda raw, result, **kw: calls.append((result, kw)))
    draft = _draft()
    draft["requested_facets"].append(make_facet("f2"))
    repair = deepcopy(draft)
    repair["units"][0]["covered_facet_ids"].append("f2")
    result = _validate(draft, repair=repair)
    assert result.is_valid and result.plan.metadata.repair_attempted
    assert not calls[0][0].is_valid and calls[1][0].is_valid
    assert calls[1][1]["phase"] == "explicit_repair"


def test_invalid_json_keeps_standard_error_path():
    """无参数；JSON错误保留Pydantic标准错误类型，不使用围栏抽取。"""
    result = _validate("{not-json")
    assert result.failure.code == "invalid_plan_format"
    errors = result.failure.trace["validation_errors"][0]["details"]["field_errors"]
    assert errors[0]["type"] == "json_invalid"


@pytest.mark.parametrize("change,code", [
    ("plan_type", "invalid_plan_format"), ("unknown_source", "invalid_plan_format"),
    ("missing_policy", "invalid_plan_format"), ("duplicate_unit", "duplicate_unit_id"),
    ("unknown_facet", "invalid_facet_id"), ("scope", "out_of_scope_paper_id"),
    ("noncitable_claim", "claim_source_not_citable"), ("external_intent", "external_intent_source_conflict"),
])
def test_old_destructive_repairs_are_explicit_failures(change, code):
    """参数为过去会自动改写的错误；新版保留输入并明确失败，不补facet或删除单元。"""
    draft = _draft()
    unit = draft["units"][0]
    if change == "plan_type":
        draft["plan_type"] = "multi_unit"
    elif change == "unknown_source":
        unit["evidence_policy"]["source_type"] = "invented"
    elif change == "missing_policy":
        del unit["evidence_policy"]
    elif change == "duplicate_unit":
        draft["units"].append(deepcopy(unit))
    elif change == "unknown_facet":
        unit["covered_facet_ids"].append("undeclared")
    elif change == "scope":
        unit["evidence_policy"]["target_paper_ids"] = ["private"]
    elif change == "noncitable_claim":
        unit["evidence_policy"]["source_type"] = "local_metadata"
    else:
        unit["claim_intents"] = [{"claim_type": "external_needed", "required": True}]
    original = deepcopy(draft)
    result = _validate(draft, papers=frozenset({"paper-a"}))
    assert result.failure.code == code
    assert draft == original
    assert not result.failure.trace["repairs"]


def test_external_words_do_not_rewrite_source():
    """无参数；论文正文中的背景研究词语不能触发来源改写。"""
    draft = _draft()
    draft["units"][0].update(goal="Need external background", research_action="read the paper background section")
    result = _validate(draft)
    assert result.is_valid
    assert result.plan.units[0].evidence_policy.source_type == "local_direct_text"


@pytest.mark.parametrize("source,key,value,valid", [
    ("local_direct_text", "evidence_granularity", "parent_chunk", True),
    ("local_section_text", "evidence_granularity", "parent_chunk", False),
    ("local_structure_hint", "verify_with_direct_text", True, True),
    ("local_metadata", "verify_with_direct_text", True, False),
])
def test_source_options_are_kept_or_rejected_without_normalization(source, key, value, valid):
    """参数为来源专属选项；生效选项原样保留，无效选项拒绝而非归零。"""
    draft = _draft()
    unit = draft["units"][0]
    unit["evidence_policy"].update(source_type=source, **{key: value})
    if source == "local_metadata":
        unit["claim_intents"] = [{"claim_type": "evidence_grounded_inference", "required": True}]
    result = _validate(draft)
    assert result.is_valid is valid
    if valid:
        assert getattr(result.plan.units[0].evidence_policy, key) == value
    else:
        assert result.failure.code.endswith("not_supported_by_source")


def test_declared_facets_are_not_invented_or_dropped():
    """无参数；显式缺口与未知要求同时保留，可选推断要求仍可未覆盖。"""
    draft = _draft()
    draft["units"][0]["covered_facet_ids"].append("unknown")
    draft["requested_facets"].append(make_facet("f2"))
    result = _validate(draft)
    codes = {item["code"] for item in result.failure.trace["validation_errors"]}
    assert {"invalid_facet_id", "explicit_facet_uncovered"} <= codes
    optional = _draft()
    optional["requested_facets"].append(make_facet("i1", "inferred"))
    result = _validate(optional)
    assert result.is_valid
    assert result.plan.coverage_review.missing_inferred_facet_ids == ["i1"]
    assert result.plan.coverage_review.unit_facet_map == {"u1": ["f1"]}


@pytest.mark.parametrize("role,valid", [("synthesis", False), ("final_recommendation", True)])
def test_recommendation_requires_the_declared_role(role, valid):
    """参数为合成角色；推荐要求必须由真正推荐单元承接。"""
    upstream = make_unit("a", facets=[])
    consumer = make_unit("b", source_type="inference_from_units", allowed_claims=["evidence_grounded_inference"],
                         output_role=role, deps=[{"unit_id": "a", "required_product": "claims"}])
    draft = strategy_fixture(make_draft([upstream, consumer]))
    draft["requested_facets"][0]["required_output_role"] = "final_recommendation"
    result = _validate(draft)
    assert result.is_valid is valid
    if not valid:
        assert result.failure.code == "facet_output_uncovered"


def test_prompt_keeps_followup_and_source_context_without_field_whitelists():
    """无参数；后续问题保留上下文和来源含义，不再重复执行字段表。"""
    prompt = build_planner_prompt({"user_query": "详细展开说明", "chat_history": [
        {"role": "user", "content": "融合三篇论文"}, {"role": "assistant", "content": "上一轮的结构"}],
        "previous_rag_context": {"query": "融合三篇论文", "answer": "结合各篇思想", "evidence_records": [
            {"evidence_id": "ev-1", "expanded_text": "Original method context."}]}})
    assert "上一轮的结构" in prompt
    assert "previous_evidence ev-1" in prompt
    assert "Project-library design or synthesis is not an external-background request" in prompt
    assert "Allowed evidence_policy keys only" not in prompt
    assert "Required JSON shape example" not in prompt
    assert "claim_policy" not in prompt
