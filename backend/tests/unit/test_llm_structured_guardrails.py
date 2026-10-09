"""【模型合同】【严格校验】旧漂移样本保留为拒绝回归，不清洗依赖或补写覆盖。作者：xxx。"""

import json

import pytest
from pydantic import ValidationError

from app.nodes.sufficiency_grader import SufficiencyDecision
from app.models.research_planner_contracts import ResearchPlanDraft
from app.services.llm_service import parse_structured_response
from research_planner_test_utils import make_draft, make_facet, make_unit


@pytest.mark.parametrize("content", ["sufficient=true", '{"sufficient":"false"}', '{"sufficient":true}'])
def test_sufficiency_requires_json_boolean_and_reason(content):
    """旧自然文本、字符串布尔及缺理由均保留原错误，不能生成默认理由。"""
    with pytest.raises(ValidationError):
        parse_structured_response(content, SufficiencyDecision)


@pytest.mark.parametrize("dependency", [
    "u1",
    {"depends_on_unit_id": "u1", "dependency_type": "required", "rationale": "need text"},
    {"source_unit_id": "u1", "dependency_type": "evidence_required"},
    {"from_unit_id": "u1", "dependency_type": "optional"},
    {"dependency_type": "evidence_required"},
    {"dependency_unit_id": "u1", "required_product": "claims"},
])
def test_dependency_drift_is_rejected_without_dropping_edges(dependency):
    """旧依赖字符串、别名和缺上游 ID 不再变成另一条依赖或被删空。"""
    synthesis = make_unit("u2", source_type="inference_from_units", facets=["f2"])
    synthesis["dependencies"] = [dependency]
    draft = make_draft([make_unit("u1"), synthesis])
    original = json.dumps(draft)
    with pytest.raises(ValidationError) as caught:
        parse_structured_response(original, ResearchPlanDraft)
    assert any("dependencies" in item["loc"] for item in caught.value.errors())
    assert json.dumps(draft) == original


@pytest.mark.parametrize(("container", "field", "value"), [
    (None, "operation_family", "analysis"),
    (None, "operation_family", "not_a_real_family"),
    ("claim_policy", "allowed_claim_types", ["inference"]),
    ("evidence_policy", "source_type", "external_context"),
    ("evidence_policy", "input_unit_ids", ["u1"]),
    ("evidence_policy", "synthesis_approach", "contrastive"),
    ("evidence_policy", "requested_topics", ["current benchmarks"]),
])
def test_enum_and_policy_drift_remains_a_field_error(container, field, value):
    """原别名和额外取证字段由合同拒绝，不能偷偷映射或删掉。"""
    unit = make_unit()
    target = unit[container] if container else unit
    target[field] = value
    with pytest.raises(ValidationError):
        parse_structured_response(json.dumps(make_draft([unit])), ResearchPlanDraft)


@pytest.mark.parametrize("field", ["user_text_span", "rationale"])
def test_null_facet_text_is_not_replaced_with_empty_string(field):
    """模型将文本写成 null 时直接报字段错误，不补空串掩盖原输入。"""
    draft = make_draft()
    draft["coverage_review"]["explicit_facets"][0][field] = None
    with pytest.raises(ValidationError):
        parse_structured_response(json.dumps(draft), ResearchPlanDraft)


def test_required_product_is_preserved_without_schema_name_whitelist():
    """曾被旧白名单删除的 required_product 必须与响应逐值一致。"""
    dependency = {"unit_id": "u1", "required_product": "claims", "required": True,
                  "required_claim_types": ["paper_supported"], "reason": "need claims"}
    draft = make_draft([make_unit("u1"), make_unit("u2", deps=[dependency])])
    result = parse_structured_response(json.dumps(draft), ResearchPlanDraft)
    assert result.units[1].dependencies[0].model_dump() == dependency


def test_coverage_map_is_not_promoted_into_model_units():
    """覆盖表与单元声明不一致留给领域校验，公共解析层不替模型补认领。"""
    draft = make_draft([make_unit("u1"), make_unit("u2", facets=[])])
    draft["coverage_review"]["explicit_facets"].append(make_facet("f2"))
    draft["coverage_review"]["unit_facet_map"] = {"u1": ["f1"], "u2": ["f2"]}
    result = parse_structured_response(json.dumps(draft), ResearchPlanDraft)
    assert result.units[1].covered_facet_ids == []
