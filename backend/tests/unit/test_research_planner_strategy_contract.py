"""【研究规划】【策略合同】最小构造回归，不冒称完整历史响应重放。作者：xxx。"""

from __future__ import annotations

import copy
import json
import asyncio

import pytest
from pydantic import ValidationError

from app.models.research_planner_contracts import ResearchUnit
from app.services.internal_research.research_planner_validation_rules import dependency_issues
from research_planner_test_utils import make_unit


def test_inference_without_dependencies_is_rejected() -> None:
    """无参数；验证推断单元缺上游不会被作为可执行计划接受。"""
    unit = ResearchUnit.model_validate(make_unit(source_type="inference_from_units"))
    assert any(item.code == "missing_inference_dependencies" for item in dependency_issues([unit]))


def test_historical_sections_retain_unknown_presentation() -> None:
    """无参数；验证旧章节未保存展示语义时保留未知，而非自动升级。"""
    from app.models.research_planner_contracts import AnswerSectionRequirement

    section = AnswerSectionRequirement(section_id="s1", title="Evidence", required=True, source="inferred")
    assert section.presentation_kind is None


def _strategy() -> dict:
    """无参数；返回独立构造的双单元策略，保留非规范化ID用于映射验证。"""
    return {
        "requested_facets": [{"facet_id": "method", "label": "方法", "source": "explicit", "required": True,
            "user_text_span": "解释方法", "deliverable_kind": "claims", "required_claim_types": ["paper_supported"],
            "required_paper_ids": ["paper-a"]}],
        "answer_sections": [{"section_id": "s1", "title": "Evidence", "facet_ids": ["method"], "required": True,
            "source": "inferred", "rationale": "组织方法说明", "presentation_kind": "internal", "user_text_span": ""}],
        "units": [
            {"unit_id": "read-a", "goal": "方法证据", "research_action": "查找方法正文",
             "evidence_policy": {"source_type": "local_direct_text", "target_paper_ids": ["paper-a"]},
             "output_goal": "方法事实", "deliverable_kind": "claims",
             "claim_intents": [{"claim_type": "paper_supported", "required": True}], "covered_facet_ids": ["method"]},
            {"unit_id": "explain", "goal": "解释方法", "research_action": "综合证据进行解释",
             "evidence_policy": {"source_type": "inference_from_units"}, "output_goal": "有证据的解释",
             "deliverable_kind": "claims", "claim_intents": [{"claim_type": "evidence_grounded_inference", "required": True}],
             "output_role": "synthesis", "dependencies": [{"unit_id": "read-a", "required_product": "claims", "required": True,
                 "required_claim_types": ["paper_supported"]}], "covered_facet_ids": ["method"]},
        ],
    }


def _validate(payload: dict, **options):
    """参数为策略及可信选项；返回正式验证结果，不调用模型或外部数据。"""
    from app.services.internal_research.research_planner_validator import PlannerValidationOptions, validate_or_repair_plan

    return validate_or_repair_plan(payload, options=PlannerValidationOptions(
        question="请解释方法", allowed_paper_ids=frozenset({"paper-a"}), **options))


def test_strategy_preserves_intent_and_maps_every_edge() -> None:
    """无参数；验证推断保持自身类型，不连带允许先验/记忆，也不改成外部请求。"""
    payload = _strategy()
    original = copy.deepcopy(payload)
    result = _validate(payload)
    assert result.is_valid, result.failure
    plan = result.plan
    assert plan.contract_version == "research-planner-v4"
    assert plan.question == "请解释方法"
    assert plan.units[1].dependencies[0].unit_id == "u1"
    assert plan.units[1].dependencies[0].required_product == "claims"
    assert plan.units[1].claim_policy.allowed_claim_types == ["evidence_grounded_inference"]
    assert plan.units[1].claim_policy.bounded_inference_allowed
    assert not plan.units[1].claim_policy.citation_required
    assert plan.units[0].expected_output.answer_section_ids == ["s1"]
    assert payload == original


@pytest.mark.parametrize("limit,accepted", [(0, True), (2, True), (1, False), (-1, False)])
def test_strategy_zero_unit_limit_matches_incremental_batch_semantics(limit, accepted) -> None:
    """参数为研究单元上限及预期结果；整份策略同样仅0不限，保留原单元及依赖，无返回值。"""
    payload = _strategy()
    original = copy.deepcopy(payload)
    result = _validate(payload, max_units=limit)
    assert payload == original
    assert result.is_valid is accepted
    if accepted:
        assert len(result.plan.units) == len(payload["units"])
        assert result.plan.units[1].dependencies[0].unit_id == "u1"
    else:
        assert result.failure.code == "too_many_units"


@pytest.mark.parametrize("field,value", [
    ("question", "fake"), ("plan_type", "multi_unit"), ("metadata", {}),
    ("contract_version", "research-planner-v4"), ("coverage_review", {}),
])
def test_strategy_rejects_server_owned_fields(field: str, value: object) -> None:
    """参数为服务端派生字段；验证模型不能输出执行状态或问题副本。"""
    payload = _strategy()
    payload[field] = value
    assert _validate(payload).failure.code == "invalid_plan_format"


@pytest.mark.parametrize("mutation", ["missing_product", "null_product", "missing_required", "wrong_upstream", "string_edge"])
def test_dependency_errors_are_not_deleted(mutation: str) -> None:
    """参数为依赖违约种类；验证错边原样报错，不能删掉后成为合法推断。"""
    payload = _strategy()
    edge = payload["units"][1]["dependencies"][0]
    if mutation == "missing_product":
        edge.pop("required_product")
    elif mutation == "null_product":
        edge["required_product"] = None
    elif mutation == "missing_required":
        edge.pop("required")
    elif mutation == "wrong_upstream":
        edge["unit_id"] = "u1"
    else:
        payload["units"][1]["dependencies"] = ["read-a"]
    original = copy.deepcopy(payload)
    result = _validate(payload)
    assert not result.is_valid
    assert payload == original


@pytest.mark.parametrize("mutation,code", [
    ("duplicate_id", "duplicate_unit_id"), ("unknown_facet", "invalid_facet_id"),
    ("scope", "out_of_scope_paper_id"), ("wrong_source", "claim_source_not_citable"),
    ("memory", "memory_intent_without_memory"), ("cycle", "cyclic_dependency"),
])
def test_semantic_errors_fail_without_repair(mutation: str, code: str) -> None:
    """参数为领域错误；验证唯一入口暴露失败，不补facet、不删单元、不换来源。"""
    payload = _strategy()
    first, second = payload["units"]
    if mutation == "duplicate_id":
        second["unit_id"] = first["unit_id"]
    elif mutation == "unknown_facet":
        first["covered_facet_ids"].append("invented")
    elif mutation == "scope":
        first["evidence_policy"]["target_paper_ids"] = ["private"]
    elif mutation == "wrong_source":
        first["evidence_policy"]["source_type"] = "local_metadata"
    elif mutation == "memory":
        second["claim_intents"].append({"claim_type": "memory_continuation", "required": False})
    else:
        first["dependencies"] = [{"unit_id": "explain", "required_product": "claims", "required": True}]
    original = copy.deepcopy(payload)
    result = _validate(payload)
    assert not result.is_valid
    assert code in {issue["code"] for issue in result.failure.trace["validation_errors"]}
    assert payload == original


def test_explicit_memory_intent_requires_real_memory_input() -> None:
    """无参数；显式选择记忆且可信输入确认可用时保留待验证记忆类型。"""
    payload = _strategy()
    payload["units"][1]["claim_intents"].append({"claim_type": "memory_continuation", "required": False})
    result = _validate(payload, memory_grounded_allowed=True)
    assert result.is_valid, result.failure
    assert result.plan.units[1].claim_policy.allowed_claim_types == ["evidence_grounded_inference", "memory_continuation"]


@pytest.mark.parametrize("product", ["materials", "claims", "hypotheses"])
def test_explicit_dependency_products_and_optional_edges_survive(product: str) -> None:
    """参数为三种上游产物；验证必需/可选声明及产物能力均由同一合同承接。"""
    payload = _strategy()
    first, second = payload["units"]
    edge = second["dependencies"][0]
    edge.update(required_product=product, required=False, required_claim_types=[])
    if product == "materials":
        first.update(deliverable_kind="materials", claim_intents=[])
        payload["requested_facets"][0].update(deliverable_kind="materials", required_claim_types=[])
    elif product == "hypotheses":
        first["claim_intents"] = [{"claim_type": "model_prior_hypothesis", "required": True}]
        payload["requested_facets"][0]["required_claim_types"] = ["model_prior_hypothesis"]
    result = _validate(payload)
    assert result.is_valid, result.failure
    assert result.plan.units[1].dependencies[0].required_product == product
    assert result.plan.units[1].dependencies[0].required is False
    if product == "materials":
        assert result.plan.units[0].claim_policy.allowed_claim_types == []


def test_replan_preserves_requirements_and_rejects_requirement_replacement() -> None:
    """无参数；只改行动可以重规划，降级既有要求无法进入只行动schema。"""
    first = _validate(_strategy()).plan
    payload = {"units": _strategy()["units"]}
    payload["units"][0]["unit_id"] = "renamed"
    payload["units"][1]["dependencies"][0]["unit_id"] = "renamed"
    result = _validate(payload, requirements_baseline=first.coverage_review)
    assert result.is_valid, result.failure
    assert result.plan.coverage_review.explicit_facets == first.coverage_review.explicit_facets
    assert result.plan.coverage_review.answer_sections == first.coverage_review.answer_sections
    payload["requested_facets"] = []
    assert not _validate(payload, requirements_baseline=first.coverage_review).is_valid


@pytest.mark.parametrize("kind,span,title,valid", [
    ("internal", "", "Evidence", True), ("user_section", "解释方法", "方法说明", True),
    ("exact_heading", "解释方法", "解释方法", True), ("user_section", "不存在", "方法", False),
    ("exact_heading", "解释方法", "Evidence", False), (None, "", "Evidence", False),
])
def test_presentation_basis_is_explicit_and_traceable(kind, span: str, title: str, valid: bool) -> None:
    """参数为展示语义和用户原话；验证展示依据可回查，未保存事实不能默认通过。"""
    payload = _strategy()
    payload["answer_sections"][0].update(presentation_kind=kind, user_text_span=span, title=title)
    assert _validate(payload).is_valid is valid


def test_claim_quote_and_user_text_are_not_stripped() -> None:
    """无参数；严格JSON处理保留换行、代码围栏与首尾原字符。"""
    from app.models.research_planner_contracts import ResearchClaimDraftBatch
    from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft

    quote = " \n```python\nx = 1\n```\n "
    raw = {"claims": [{"text": "  文本  ", "claim_type": "paper_supported", "supporting_evidence_ids": ["e1"],
        "supporting_spans": [{"evidence_id": "e1", "evidence_version": "v1", "field": "excerpt", "start": 0,
                              "end": len(quote), "quote": quote}]}]}
    batch = ResearchClaimDraftBatch.model_validate_json(json.dumps(raw), strict=True)
    assert batch.claims[0].supporting_spans[0].quote == quote
    assert batch.claims[0].text == "  文本  "
    payload = _strategy()
    payload["requested_facets"][0]["user_text_span"] = "  原文\n"
    assert ResearchPlanStrategyDraft.model_validate(payload).requested_facets[0].user_text_span == "  原文\n"


@pytest.mark.parametrize("mutation", ["duplicate", "no_required", "materials_with_claim", "policy", "bool_string"])
def test_single_intent_contract_rejects_conflicting_declarations(mutation: str) -> None:
    """参数为重复或冲突声明；验证类型、必需性和材料模式不能被自动清洗。"""
    payload = _strategy()
    unit = payload["units"][0]
    if mutation == "duplicate":
        unit["claim_intents"] *= 2
    elif mutation == "no_required":
        unit["claim_intents"][0]["required"] = False
    elif mutation == "materials_with_claim":
        unit["deliverable_kind"] = "materials"
    elif mutation == "policy":
        unit["claim_policy"] = {"bounded_inference_allowed": True}
    else:
        unit["claim_intents"][0]["required"] = "true"
    assert _validate(payload).failure.code == "invalid_plan_format"


def test_revalidate_model_copy_projection_and_preserve_program_failure(monkeypatch) -> None:
    """参数为补丁工具；验证未校验model_copy不能穿过边界，程序异常不会冒充模型违约。"""
    from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft
    from app.services.internal_research import research_planner_validator as validator

    draft = ResearchPlanStrategyDraft.model_validate(_strategy())
    corrupted = draft.model_copy(update={"units": [{"unknown": "x"}]})
    assert not _validate(corrupted).is_valid

    def fail(*args, **kwargs):
        """参数为任意投影输入；模拟真实程序错误，不调用外部服务。"""
        raise RuntimeError("projection bug")

    monkeypatch.setattr(validator, "project_strategy_units", fail)
    observations = []
    monkeypatch.setattr(validator, "record_plan_validation", lambda raw, result, **kw: observations.append(result))
    with pytest.raises(RuntimeError, match="projection bug"):
        _validate(_strategy())
    assert observations[-1].failure.trace["failure_stage"] == "program_conversion"


@pytest.mark.parametrize("version", ["research-planner-v1", "research-planner-v2", "research-planner-v3"])
def test_history_reads_without_upgrade_and_checkpoint_requires_replan(version: str) -> None:
    """参数为历史合同版本；验证读取不补事实，执行恢复保留快照并明确重规划。"""
    from app.models.research_planner_contracts import ResearchPlan
    from app.services.internal_research.research_planner_conversion import migrate_execution_checkpoint

    current = _validate(_strategy()).plan.model_dump()
    current["contract_version"] = version
    current["coverage_review"]["answer_sections"][0].pop("presentation_kind")
    history = ResearchPlan.model_validate(current)
    assert history.contract_version == version
    assert history.coverage_review.answer_sections[0].presentation_kind is None
    delta = migrate_execution_checkpoint({"research_plan": current})
    assert delta["research_plan"] is None
    assert delta["planner_execution_history"][0]["snapshot"]["research_plan"] == current
    current["contract_version"] = "research-planner-v4"
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(current)


def test_current_strategy_checkpoint_keeps_failed_attempt_context() -> None:
    """无参数；新草案没有执行版本字段，回规划不能因此被误迁移成历史并丢失取证上下文。"""
    from app.services.internal_research.research_planner_conversion import migrate_execution_checkpoint

    assert migrate_execution_checkpoint({"planner_draft": _strategy(), "planner_budget": {"steps": 2}}) == {}


@pytest.mark.parametrize("field", ["goal", "research_action", "output_goal", "unit_id"])
def test_business_text_cannot_be_only_whitespace(field: str) -> None:
    """参数为必需业务文本；保持原字符的同时，用字段约束拒绝空白冒充非空内容。"""
    payload = _strategy()
    payload["units"][0][field] = "  \n\t"
    assert _validate(payload).failure.code == "invalid_plan_format"


def test_whitespace_claim_and_facet_are_rejected_without_stripping_originals() -> None:
    """无参数；空白主张和要求标签拒绝，真实原文前后空格继续保留。"""
    from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraft, RequestedFacetDraft

    with pytest.raises(ValidationError):
        ResearchClaimStrategyDraft(text=" \n ", claim_type="model_prior_hypothesis")
    facet = _strategy()["requested_facets"][0]
    with pytest.raises(ValidationError):
        RequestedFacetDraft.model_validate({**facet, "label": "  "})
    claim = ResearchClaimStrategyDraft(text="  有内容  ", claim_type="model_prior_hypothesis")
    assert claim.text == "  有内容  "


def test_unknown_dependency_cannot_collide_with_projected_id() -> None:
    """无参数；未知u2不能在alpha/beta重编号后冒充合法上游beta。"""
    payload = _strategy()
    payload["units"][0]["unit_id"] = "alpha"
    payload["units"][1]["unit_id"] = "beta"
    payload["units"][1]["dependencies"][0]["unit_id"] = "u2"
    result = _validate(payload)
    assert result.failure.code == "bad_dependency_id"
    assert result.plan is None


def test_claim_model_selects_quote_and_server_binds_actual_version() -> None:
    """无参数；模型只选择原文，版本绑定到同次可见材料，外部版本字段严格拒绝。"""
    from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
    from app.services.internal_research.research_planner_claim_provider import bind_claim_versions
    from app.services.internal_research.research_planner_consumer_budget import prepare_consumer_input

    text = " \n```python\nx=1\n```\n "
    prepared = prepare_consumer_input([{"evidence_id": "e1", "excerpt": text, "paper_id": "paper-a"}], [],
                                       evidence_max_chars=1000, prompt_version="research-claim-v4")
    payload = {"claims": [{"text": "代码", "claim_type": "paper_supported", "supporting_evidence_ids": ["e1"],
        "supporting_spans": [{"evidence_id": "e1", "field": "excerpt", "start": 0, "end": len(text), "quote": text}]}]}
    batch = ResearchClaimStrategyDraftBatch.model_validate_json(json.dumps(payload), strict=True)
    claim = bind_claim_versions(batch, prepared)[0]
    assert claim.supporting_spans[0].quote == text
    assert claim.supporting_spans[0].evidence_version == prepared.exposure.evidence[0].evidence_version
    payload["claims"][0]["supporting_spans"][0]["evidence_version"] = "model-copy"
    with pytest.raises(ValidationError, match="evidence_version"):
        ResearchClaimStrategyDraftBatch.model_validate(payload)


def test_claim_version_binding_does_not_invent_an_unknown_material() -> None:
    """无参数；未送入本次模型的材料不能由服务端补版本变成合法支持。"""
    from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
    from app.services.internal_research.research_planner_claim_provider import bind_claim_versions
    from app.services.internal_research.research_planner_consumer_budget import prepare_consumer_input

    prepared = prepare_consumer_input([], [], evidence_max_chars=1000, prompt_version="test")
    batch = ResearchClaimStrategyDraftBatch.model_validate({"claims": [{"text": "x", "claim_type": "paper_supported",
        "supporting_spans": [{"evidence_id": "absent", "field": "excerpt", "start": 0, "end": 1, "quote": "x"}]}]})
    with pytest.raises(ValueError, match="outside this model input"):
        bind_claim_versions(batch, prepared)


def test_pipeline_requests_initial_and_replan_schema_without_requirement_echo() -> None:
    """无参数；实际模型接线首次识别要求，重规划schema不再让模型复写基准。"""
    from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft, ResearchReplanStrategyDraft
    from app.services.internal_research.research_planner_pipeline import _analyze_plan, validate_planner_state
    from types import SimpleNamespace

    class Model:
        """【研究规划】【离线模型】只校验测试输入，不实现网络调用。"""

        def __init__(self):
            """无参数；创建schema观察列表。"""
            self.schemas = []

        def with_structured_output(self, schema):
            """参数为运行请求的schema；记录真实选择并返回当前离线端口。"""
            self.schemas.append(schema)
            return self

        async def ainvoke(self, messages):
            """参数为业务提示；按运行选择的真实schema返回固定测试策略。"""
            schema = self.schemas[-1]
            payload = _strategy() if schema is ResearchPlanStrategyDraft else {"units": _strategy()["units"]}
            assert "claim_policy" not in messages[0].content
            assert "research-plan-strategy-v5" in messages[0].content
            assert "nonempty user_text_span" in messages[0].content
            assert "user (verbatim excerpt):\n  历史原话\n保留空白 " in messages[0].content
            if schema is ResearchPlanStrategyDraft:
                assert "description" in schema.model_json_schema()["properties"]["requested_facets"]
            return schema.model_validate(payload)

    model = Model()
    state = {"user_query": "请解释方法", "project_scope": {"paper_ids": ["paper-a"]},
             "chat_history": [{"role": "user", "content": "  历史原话\n保留空白 "}]}
    first = asyncio.run(_analyze_plan(state, model))
    validated = validate_planner_state({**state, "planner_draft": first.model_dump()}, settings=SimpleNamespace(
        rag_research_planner_max_units=8))
    assert validated["research_plan"]["contract_version"] == "research-planner-v4"
    second = asyncio.run(_analyze_plan({**state, "research_plan": validated["research_plan"]}, model))
    assert model.schemas == [ResearchPlanStrategyDraft, ResearchReplanStrategyDraft]
    assert "requested_facets" not in second.model_dump()
