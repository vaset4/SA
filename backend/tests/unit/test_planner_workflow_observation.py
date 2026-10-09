"""【评测观察】【执行校验】验证真实工具四态与引用验证观测。作者：xxx。"""

from __future__ import annotations

from o2_evidence_test_utils import direct_evidence, read_result

import asyncio
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import Mock

import pytest

from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx
from app.models.research_planner_contracts import ResearchClaim, ResearchPlan, ResearchUnit
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError, PlannerExecutionContext, ResearchPlannerExecutor, _call_adapter,
)
from app.services.internal_research.research_planner_reasoner import validate_claim_drafts
from app.services.internal_research.research_planner_execution_state import PlannerExecutionState
from evals.agent_outcome.workflow_execution import unit_checks
from evals.agent_outcome.workflow_facts import read_turn
from evals.contracts import snapshot
from research_planner_test_utils import make_draft, make_unit, result_with_exposure, supporting_spans


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """参数为补丁工具；封锁未替换的网络和数据库边界，并断言无意外外联。"""
    blocked = Mock(side_effect=AssertionError("流程观察回归禁止真实外联"))
    for target in ("httpx.Client.send", "httpx.AsyncClient.send", "requests.sessions.Session.request",
                   "app.services.database.get_pool", "socket.create_connection"):
        monkeypatch.setattr(target, blocked)
    yield
    assert blocked.call_count == 0


@pytest.fixture
def traced_execution(monkeypatch):
    """参数为替身；返回trace，限定工具重试但不发真实请求。"""
    config = SimpleNamespace(rag_planner_evidence_max_retries=1,
                             rag_planner_evidence_retry_backoff_seconds=0,
                             rag_planner_evidence_timeout_seconds=2)
    monkeypatch.setattr("app.services.internal_research.research_planner_executor.get_settings", lambda: config)
    trace = TraceContext(request_id="planner-observation")
    token = set_trace_ctx(trace)
    try:
        yield trace
    finally:
        reset_trace_ctx(token)


def events(trace: TraceContext, stage: str) -> list[dict]:
    """参数为trace及阶段；返回同一次执行的冻结事实。"""
    return [item["payload"] for item in export_evidence_observations(trace)["evidence_observations"]
            if item["stage"] == stage]


def test_tool_retry_preserves_attempts_without_duplicate_logical_calls(traced_execution):
    """工具连接错误后原重试成功，两次attempt保留相同逻辑工具ID。"""
    adapter = Mock(side_effect=[OSError("connection interrupted"), read_result([])])
    unit = ResearchUnit.model_validate(make_unit())
    assert asyncio.run(_call_adapter(adapter, unit, PlannerExecutionContext())).records == ()
    outcomes = events(traced_execution, "tool_execution_result")
    assert [item["status"] for item in outcomes] == ["failed", "empty"]
    assert [item["tool_attempt"] for item in outcomes] == [1, 2]
    assert len({item["tool_call_id"] for item in outcomes}) == 1
    assert outcomes[0]["error_type"] == "OSError"


def test_tool_failure_keeps_original_cause(traced_execution):
    """非重试工具失败仍按原规则抛出取证错误，观察不吞掉原始原因。"""
    error = RuntimeError("original adapter failure")
    adapter = Mock(side_effect=error)
    unit = ResearchUnit.model_validate(make_unit())
    with pytest.raises(EvidenceCapabilityError) as caught:
        asyncio.run(_call_adapter(adapter, unit, PlannerExecutionContext()))
    assert caught.value.__cause__ is error and adapter.call_count == 1
    assert events(traced_execution, "tool_execution_result")[0]["status"] == "failed"


def test_unit_empty_and_skipped_are_not_merged(traced_execution):
    """相同来源的实际空返回与缺定位词跳过单独保存，跳过不产生工具尝试。"""
    adapter = Mock(return_value=read_result([]))
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter})
    first = ResearchUnit.model_validate(make_unit("first"))
    skipped = make_unit("second")
    skipped["evidence_policy"]["locator_queries"] = []
    second = ResearchUnit.model_validate(skipped)
    context = PlannerExecutionContext()
    asyncio.run(executor._execute_unit(first, [], context))
    asyncio.run(executor._execute_unit(second, [], context))
    outcomes = events(traced_execution, "unit_execution_result")
    assert [item["capability_status"] for item in outcomes] == ["empty", "skipped"]
    assert adapter.call_count == 1 and len(events(traced_execution, "tool_execution_result")) == 1
    assert all(item["status"] == "completed" for item in outcomes)


def test_unit_dependency_skip_retains_declared_dependency(traced_execution):
    """缺少已声明前序结果时仍按原规则阻塞，记录依赖来源与未调用状态。"""
    adapter = Mock(return_value=read_result([]))
    unit = ResearchUnit.model_validate(make_unit(deps=[{"unit_id": "missing", "required": True}]))
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter})
    asyncio.run(executor._execute_unit(unit, [], PlannerExecutionContext()))
    outcome = events(traced_execution, "unit_execution_result")[0]
    assert outcome["capability_status"] == "skipped" and outcome["unit_status"] == "blocked"
    assert outcome["dependencies"][0]["unit_id"] == "missing" and not adapter.called


def test_claim_reference_rejection_is_business_validation(traced_execution):
    """主张schema合法但引用不存在的Evidence时记录真实业务拒绝。"""
    unit = ResearchUnit.model_validate(make_unit())
    outcome = validate_claim_drafts(unit, [], [], [{"text": "claim", "claim_type": "paper_supported",
                                                   "supporting_evidence_ids": ["nonexistent"]}])
    assert not outcome.claims and outcome.rejected_reasons
    event = events(traced_execution, "structured_business_validation")[0]
    assert event["validation_stage"] == "claim_evidence_references" and event["valid"] is False
    assert event["errors"] == outcome.rejected_reasons
    assert event["operation_id"] is None


@pytest.mark.parametrize(("status", "validation", "required_types", "called"), [
    ("planned", "needs_validation", [], False),
    ("evidence_found", "validated", ["paper_supported"], True),
    ("evidence_found", "needs_validation", ["paper_supported"], False),
])
def test_real_dependency_observation_matches_workflow_rules(traced_execution, status, validation, required_types, called):
    """参数为合法前序状态和依赖；真实执行器允许/阻塞的结果必须被评测正确识别，不另造完成条件。"""
    record = direct_evidence("prior")
    prior_unit = ResearchUnit.model_validate(make_unit("prior"))
    claim = ResearchClaim.model_validate({
        "claim_id": "claim-prior", "text": "observed claim", "claim_type": "paper_supported",
        "validation_status": validation, "supporting_evidence_ids": [record["evidence_id"]],
        "supporting_spans": supporting_spans([record]),
    })
    prior = result_with_exposure(prior_unit, [record], [claim]).model_copy(update={"status": status})
    unit = ResearchUnit.model_validate(make_unit(deps=[{
        "unit_id": "prior", "required": True, "required_claim_types": required_types,
    }]))
    adapter = Mock(return_value=read_result([]))
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter})
    context = PlannerExecutionContext(execution_state=PlannerExecutionState(evidence_records=[record]))
    result, _records, _capability = asyncio.run(executor._execute_unit(unit, [prior], context))
    assert adapter.called is called
    assert (result.status == "blocked") is not called
    turn = read_turn({"turn": 1, "input_snapshot": snapshot({}), "values_snapshot": snapshot({}),
                      "evidence_observation_snapshot": snapshot(export_evidence_observations(traced_execution))})
    assert unit_checks(turn) == ([], [])


@pytest.mark.parametrize("provider_failed", [False, True])
def test_final_rejection_metrics_count_positions_not_error_messages(provider_failed: bool) -> None:
    """参数为模型失败分支；真实最终校验的局部多错误只计一个坏块，报告级错误不计正文块。"""
    from app.services.internal_research.research_planner_final_contracts import FINAL_CONTRACT_VERSION
    from app.services.internal_research.research_planner_final_retry import _updated_trace_summary
    from app.services.internal_research.research_planner_observability import build_planner_trace_payload
    from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer

    plan = ResearchPlan.model_validate(make_draft())
    record = direct_evidence("metrics")
    claim = ResearchClaim(claim_id="u1.c1", text="有来源的事实", claim_type="paper_supported",
                          validation_status="validated", supporting_evidence_ids=[record["evidence_id"]],
                          supporting_spans=supporting_spans([record]), addressed_facet_ids=["f1"])
    result = result_with_exposure(plan.units[0], [record], [claim])
    valid = {"block_id": "good", "text": "有来源的正文", "content_kind": "fact", "facet_ids": ["f1"],
             "section_ids": [], "claim_ids": [claim.claim_id], "evidence_ids": [record["evidence_id"]]}
    invalid = {**valid, "block_id": "bad", "text": "可读但关系错误的正文",
               "claim_ids": ["missing"], "evidence_ids": ["missing"]}
    output = synthesize_planner_answer(plan, [result], project_paper_count=1, evidence_records=[record],
             final_answer_draft={"contract_version": FINAL_CONTRACT_VERSION,
                                 "blocks": [valid, invalid, None], "extra": True},
             final_provider_error="provider unavailable" if provider_failed else "")
    expected = 0 if provider_failed else 2
    assert output["final_delivery_report"]["status"] == ("failed" if provider_failed else "partial")
    assert len(output["planner_final_validation_errors"]) > expected
    observed = build_planner_trace_payload(output)["planner_metrics"]
    retried = _updated_trace_summary({}, output)
    assert observed["planner_final_rejected_block_count"] == expected
    assert retried["planner_final_rejected_block_count"] == expected
