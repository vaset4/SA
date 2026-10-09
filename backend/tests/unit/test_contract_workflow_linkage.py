"""【合同观察】【流程关联】从真实模型边界验证新策略、转换与离线覆盖；作者：xxx。"""

from __future__ import annotations

import json

import pytest

from app.middleware.trace_context import export_evidence_observations
from app.models.research_planner_draft_contracts import (
    ResearchClaimStrategyDraftBatch, ResearchPlanStrategyDraft, ResearchReplanStrategyDraft,
)
from app.services import llm_observation
from app.services.internal_research.research_planner_validator import PlannerValidationOptions, validate_or_repair_plan
from evals.agent_outcome.workflow_delivery import assess_delivery
from evals.agent_outcome.workflow_facts import read_turn
from evals.agent_outcome.workflow_structure import grade_structure
from evals.contracts import snapshot
from test_llm_contract_standardization import response

from test_research_planner_strategy_contract import _strategy

pytest_plugins = ["test_llm_contract_standardization"]


def turn(trace, *, values=None):
    """参数为真实 trace 和可选终态；返回经正式冻结读取的工作流轮次。"""
    state = values or {}
    return read_turn({"turn": 1, "input_snapshot": snapshot({"question": "请解释方法"}),
                      "values_snapshot": snapshot(state), "assistant_output": state.get("answer", ""),
                      "terminal_state": "completed", "evidence_observation_snapshot": snapshot(export_evidence_observations(trace))})


def event_payloads(trace, stage):
    """参数为 trace 和阶段；返回本次冻结事件载荷。"""
    return [json.loads(item)["payload"] for item in trace.evidence_observations if json.loads(item)["stage"] == stage]


@pytest.mark.parametrize("replan", [False, True])
def test_new_plan_actual_validation_is_in_business_coverage(runtime, replan):
    """首次及重规划经实际结构调用和领域校验后，工作流业务覆盖须关联同一操作。"""
    client, post, _cache, trace = runtime
    raw = _strategy()
    options = PlannerValidationOptions(question="请解释方法", allowed_paper_ids=frozenset({"paper-a"}))
    schema = ResearchPlanStrategyDraft
    if replan:
        accepted = validate_or_repair_plan(raw, options=options)
        assert accepted.is_valid
        options = PlannerValidationOptions(question="请解释方法", allowed_paper_ids=frozenset({"paper-a"}),
                                           requirements_baseline=accepted.plan.coverage_review)
        schema, raw = ResearchReplanStrategyDraft, {"units": raw["units"]}
    post.side_effect = None
    post.return_value = response(json.dumps(raw, ensure_ascii=False))
    draft = client.with_structured_output(schema).invoke("请解释方法")
    result = validate_or_repair_plan(draft, options=options)
    assert result.is_valid
    validations = event_payloads(trace, "structured_response_validation")
    business = event_payloads(trace, "structured_business_validation")[-1]
    assert business["schema"] == schema.__name__
    assert business["operation_id"] == validations[0]["operation_id"]
    metrics, structure = grade_structure((turn(trace),))
    coverage = next(metric for metric in metrics if metric.name == "structured_business_observation_coverage")
    assert coverage.denominator == 1 and coverage.value == 1
    assert structure["business_validation_missing_operation_ids"] == []


@pytest.mark.parametrize("schema", [ResearchPlanStrategyDraft, ResearchReplanStrategyDraft, ResearchClaimStrategyDraftBatch])
def test_new_schema_without_business_observation_reports_missing_coverage(runtime, schema):
    """新计划与主张若没有实际业务核验，不能因名字未登记而从应观察分母消失。"""
    client, post, _cache, trace = runtime
    raw = _strategy() if schema is ResearchPlanStrategyDraft else {"units": _strategy()["units"]}
    if schema is ResearchClaimStrategyDraftBatch:
        raw = {"claims": []}
    post.side_effect = None
    post.return_value = response(json.dumps(raw))
    client.with_structured_output(schema).invoke("question")
    metrics, structure = grade_structure((turn(trace),))
    coverage = next(metric for metric in metrics if metric.name == "structured_business_observation_coverage")
    assert coverage.denominator == 1 and coverage.value == 0
    assert len(structure["business_validation_missing_operation_ids"]) == 1


@pytest.mark.parametrize("duplicate_source", [False, True])
def test_explicit_conversion_links_only_a_unique_source_operation(runtime, duplicate_source):
    """服务端添加字段后仅凭显式转换关联；相同源输出来自多次操作时保留歧义。"""
    client, post, _cache, trace = runtime
    post.side_effect = None
    post.return_value = response('{"claims":[]}')
    batch = client.with_structured_output(ResearchClaimStrategyDraftBatch).invoke("question")
    if duplicate_source:
        client.with_structured_output(ResearchClaimStrategyDraftBatch).invoke("another question")
    converted = {"claims": [], "server_only_fact": "bound-evidence-version"}
    llm_observation.record_structured_conversion(batch, converted, schema=ResearchClaimStrategyDraftBatch.__name__)
    llm_observation.record_business_validation(converted, {"schema": ResearchClaimStrategyDraftBatch.__name__, "valid": True})
    conversion = event_payloads(trace, "structured_conversion")[0]
    business = event_payloads(trace, "structured_business_validation")[0]
    assert conversion["input_fingerprint"] != conversion["output_fingerprint"]
    assert business["conversion_ids"] == [conversion["conversion_id"]]
    metrics, _structure = grade_structure((turn(trace),))
    coverage = next(metric for metric in metrics if metric.name == "structured_business_observation_coverage")
    if duplicate_source:
        assert conversion["operation_id"] is None and business["operation_id"] is None
        assert len(conversion["matching_operation_ids"]) == 2
        assert coverage.value == 0
    else:
        source = event_payloads(trace, "structured_response_validation")[0]
        assert conversion["operation_id"] == business["operation_id"] == source["operation_id"]
        assert coverage.value == 1


def test_conversion_without_observed_source_never_invents_operation(runtime):
    """没有模型边界证据的本地转换可以记录指纹，但不能造出模型操作关联。"""
    _client, _post, _cache, trace = runtime
    raw, converted = {"claims": []}, {"claims": [], "source": "unobserved"}
    llm_observation.record_structured_conversion(raw, converted, schema=ResearchClaimStrategyDraftBatch.__name__)
    llm_observation.record_business_validation(converted, {"schema": ResearchClaimStrategyDraftBatch.__name__, "valid": True})
    assert event_payloads(trace, "structured_business_validation")[0]["operation_id"] is None
    assert event_payloads(trace, "structured_conversion")[0]["matching_operation_ids"] == []


def test_without_conversion_mapping_no_server_field_is_guessed_away(runtime):
    """原输出已经被观察也不能靠删除服务端字段猜来源，缺少显式映射时保持关联缺失。"""
    client, post, _cache, trace = runtime
    post.side_effect = None
    post.return_value = response('{"claims":[]}')
    client.with_structured_output(ResearchClaimStrategyDraftBatch).invoke("question")
    llm_observation.record_business_validation({"claims": [], "server_fact": "v4"}, {
        "schema": ResearchClaimStrategyDraftBatch.__name__, "valid": True,
    })
    business = event_payloads(trace, "structured_business_validation")[0]
    assert business["operation_id"] is None and business["conversion_ids"] == []


def test_dictionary_plan_without_explicit_schema_keeps_identity_unknown(runtime):
    """直接记录未携带可信schema的字典时保留未知，不能按字段猜成历史或新计划。"""
    from types import SimpleNamespace

    _client, _post, _cache, trace = runtime
    result = SimpleNamespace(plan=None, failure=SimpleNamespace(trace={}), is_valid=False)
    llm_observation.record_plan_validation({"units": []}, result, phase="initial")
    assert event_payloads(trace, "structured_business_validation")[0]["schema"] is None


@pytest.mark.parametrize("version,expected", [("research-planner-v4", "unknown"), ("research-planner-v3", "unknown"), (None, "minimal")])
def test_missing_report_on_current_plan_is_not_minimal_success(runtime, version, expected):
    """冻结回合带现行计划却缺报告时保持未知；真正无计划的旧minimal路径保持可判。"""
    _client, _post, _cache, trace = runtime
    state = {"turn_depth": "minimal", "answer": "answer", "turn_outcome": {"status": "completed"}}
    if version:
        state["research_plan"] = {"contract_version": version}
    assessment = assess_delivery(turn(trace, values=state))
    assert assessment.contract == expected
    assert assessment.false_success is (False if expected == "minimal" else None)
