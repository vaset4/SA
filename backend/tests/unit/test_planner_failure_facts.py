"""【研究失败】【读取事实】累计读取与失败出口的离线回归。作者：xxx。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.models.research_planner_contracts import PlannerFailure
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research.research_planner_executor import read_evidence
from app.services.internal_research.research_planner_execution_state import ReadReceipt
from app.services.llm_service import StructuredOutputError
from o2_evidence_test_utils import direct_evidence, scope


def _state() -> dict[str, Any]:
    """无参数；返回具备本轮身份且尚未正式执行的离线状态。"""
    return {"run_id": "failure-facts-run", "project_scope": scope(), "turn_depth": "lookup",
            "user_query": "Explain A Method", "planner_read_receipts": [], "evidence_records": []}


def _reads(state: dict[str, Any], outcomes: list[str], *, calls: int = 1, prefix: str = "read") -> dict[str, Any]:
    """参数为运行、结果与调用次数；返回合法材料/回执快照，不调用任何外部服务。"""
    identity = pipeline._execution_context({**state, "planner_read_receipts": [], "evidence_records": []}).execution_state.identity
    record = direct_evidence()
    receipts = [ReadReceipt(
        receipt_id=f"{prefix}-{index}", identity=identity, source_execution_id="probe-execution",
        source_unit_id="probe-context", source_type="local_section_text", request=None, request_key="",
        outcome=outcome, actual_call_count=calls,
        evidence_ids=(record["evidence_id"],) if outcome == "success" else (),
        read_report={"operation": "sections", "status": "complete"}
        if outcome in {"success", "empty", "unavailable"} else None,
    ).model_dump(mode="json") for index, outcome in enumerate(outcomes)]
    return {**state, "planner_read_receipts": receipts,
            "evidence_records": [record] if "success" in outcomes else []}


def _structured_failure() -> StructuredOutputError:
    """无参数；返回真实冻结失败形状的结构化异常，不伪造成功响应。"""
    return StructuredOutputError("structured_output_failed", "initial action plan requires requested_facets",
                                 user_summary="研究计划字段不符合契约", details={"repair_count": 1})


def _analyze(monkeypatch: pytest.MonkeyPatch, state: dict[str, Any], exc: Exception) -> dict[str, Any]:
    """参数为注入器、状态与失败；经过正式分析入口返回失败增量。"""
    monkeypatch.setattr(pipeline, "_first_analysis_delta", AsyncMock(return_value={}))
    monkeypatch.setattr(pipeline, "_analyze_plan", AsyncMock(side_effect=exc))
    return asyncio.run(pipeline.analyze_planner_state(state, object(), SimpleNamespace()))


def _assert_read_fact(delta: dict[str, Any], *, calls: int, success: bool, formal: bool = False) -> None:
    """参数为正式失败增量与可证事实；核对正文、状态、trace和交付不相互矛盾。"""
    expected = "attempted" if calls else "not_run"
    assert delta["source_status"]["local_evidence_execution"] == expected
    assert delta["planner_trace"]["local_evidence_execution"] == expected
    facts = delta["planner_trace"]["read_facts"]
    assert facts["actual_call_count"] == calls
    assert facts["has_successful_read"] is success
    assert facts["execution_started"] is formal
    assert ("正式研究执行已开始" if formal else "研究计划未能完成") in delta["answer"]
    if calls:
        assert ("已读取部分本地资料" if success else "已尝试查询本地资料") in delta["answer"]
    else:
        assert "本轮尚未调用本地资料读取工具" in delta["answer"]


def test_saved_probe_success_survives_structured_plan_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身；三次正常回边读取后结构错误仍显示已读，且不假报研究已执行。"""
    state = _reads(_state(), ["success"] * 3)
    original = deepcopy(state)
    delta = _analyze(monkeypatch, state, _structured_failure())
    _assert_read_fact(delta, calls=3, success=True)
    assert delta["final_delivery_applicability"] == "pre_execution_failure"
    assert delta["research_unit_results"] == []
    assert state == original


@pytest.mark.parametrize("outcome", ["empty", "failed", "cancelled", "unavailable"])
def test_attempted_probe_without_success_never_claims_read_content(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    """参数为实际发出但无成功材料的读取结果；失败摘要只承认尝试。"""
    delta = _analyze(monkeypatch, _reads(_state(), [outcome], calls=2), _structured_failure())
    _assert_read_fact(delta, calls=2, success=False)


@pytest.mark.parametrize("outcomes", [[], ["success"], ["failed"]])
def test_zero_call_records_and_cached_material_do_not_create_attempts(monkeypatch: pytest.MonkeyPatch, outcomes: list[str]) -> None:
    """参数为空回执、零调用成功或失败；材料/记录存在不能冒充工具调用。"""
    delta = _analyze(monkeypatch, _reads(_state(), outcomes, calls=0), _structured_failure())
    _assert_read_fact(delta, calls=0, success=False)


def test_exception_snapshot_merges_cumulative_receipts_without_double_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身；异常累计快照包含旧回执和新增失败时，旧成功保留且不重复统计。"""
    state = _reads(_state(), ["success"] * 3)
    snapshot = _reads(_state(), ["failed"], prefix="new")
    snapshot["planner_read_receipts"] = [*state["planner_read_receipts"], *snapshot["planner_read_receipts"]]
    snapshot["evidence_records"] = state["evidence_records"]
    exc = _structured_failure()
    exc.planner_execution_snapshot = snapshot
    delta = _analyze(monkeypatch, state, exc)
    _assert_read_fact(delta, calls=4, success=True)
    assert len(delta["planner_read_receipts"]) == 4
    assert delta["evidence_records"] == state["evidence_records"]


def test_initial_analysis_failure_uses_saved_current_run_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身；首次分析自身报错也通过共同事实出口，不丢当前运行已保存的读取。"""
    monkeypatch.setattr(pipeline, "_first_analysis_delta", AsyncMock(side_effect=RuntimeError("context failed")))
    delta = asyncio.run(pipeline.analyze_planner_state(_reads(_state(), ["success"]), object(), SimpleNamespace()))
    _assert_read_fact(delta, calls=1, success=True)


def test_domain_rejection_only_projects_on_terminal_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为校验替身；重复领域拒绝前的重规划不落终态，终止时仍携带已读事实。"""
    from app.models.research_planner_contracts import PlannerValidationResult

    failure = PlannerFailure(code="unverified_facet_basis", message="用户原文依据不成立")
    monkeypatch.setattr(pipeline, "validate_or_repair_plan", lambda *_args, **_kwargs: PlannerValidationResult(failure=failure))
    state = _reads(_state(), ["success"] * 4)
    first = pipeline.validate_planner_state(state, SimpleNamespace())
    assert first["planner_loop_decision"] == "replan"
    assert not first["error_code"]
    second = pipeline.validate_planner_state({**state, **first}, SimpleNamespace())
    assert second["planner_loop_decision"] == "terminate_fail"
    _assert_read_fact(second, calls=4, success=True)


@pytest.mark.parametrize("snapshot_kind", ["incremental", "repeated"])
def test_exception_snapshot_preserves_earlier_success(monkeypatch: pytest.MonkeyPatch, snapshot_kind: str) -> None:
    """参数为仅新增或重复累计快照；旧回执不得被遗漏，重复快照不能成为新读取。"""
    state = _reads(_state(), ["success"])
    exc = _structured_failure()
    exc.planner_execution_snapshot = state if snapshot_kind == "repeated" else _reads(_state(), ["failed"], prefix="new")
    delta = _analyze(monkeypatch, state, exc)
    _assert_read_fact(delta, calls=1 if snapshot_kind == "repeated" else 2, success=True)
    assert delta["evidence_records"] == state["evidence_records"]


@pytest.mark.parametrize("missing", ["run_id", "planner_read_receipts"])
def test_missing_historical_facts_remain_unknown_in_actual_trace(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """参数为缺失的历史事实；正式保存trace不把未知变成未读，输入历史正文不被修改。"""
    from app.services.internal_research.research_planner_observability import build_planner_stage, build_planner_trace_payload

    state = {**_state(), "answer": "历史正文原样保留"}
    state.pop(missing)
    original = deepcopy(state)
    delta = _analyze(monkeypatch, state, _structured_failure())
    assert "local_evidence_execution" not in delta["source_status"]
    assert "local_evidence_execution" not in delta["planner_trace"]
    assert "尚未调用" not in delta["answer"]
    assert "已读取" not in delta["answer"]
    assert delta["planner_trace"]["read_facts"]["known"] is False
    stage = build_planner_stage({**delta, **build_planner_trace_payload({**state, **delta})})
    assert stage["read_facts"] == delta["planner_trace"]["read_facts"]
    assert "local_evidence_execution" not in stage["source_status"]
    assert state == original


def test_current_identity_cannot_consume_other_runs_exception_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身；本轮失败快照仍遵守既有同运行约束，不把其他回合的读取记入本轮。"""
    state = _reads(_state(), ["success"])
    exc = _structured_failure()
    exc.planner_execution_snapshot = _reads({**_state(), "run_id": "other-run"}, ["success"], prefix="foreign")
    with pytest.raises(ValueError, match="研究运行、用户、范围或发布快照不一致"):
        _analyze(monkeypatch, state, exc)


def test_new_question_and_nonlocal_receipts_do_not_claim_local_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身；新回合空状态或仅外部工具回执均不能被当作本地资料读取。"""
    state = _reads(_state(), ["empty"])
    state["planner_read_receipts"][0]["source_type"] = "external_context_needed"
    delta = _analyze(monkeypatch, state, _structured_failure())
    _assert_read_fact(delta, calls=0, success=False)
    next_state = {**_state(), "run_id": "new-question-run"}
    next_delta = _analyze(monkeypatch, next_state, _structured_failure())
    _assert_read_fact(next_delta, calls=0, success=False)


@pytest.mark.parametrize("prior_reads", [False, True])
def test_pre_execution_access_failure_preserves_only_prior_read_fact(monkeypatch: pytest.MonkeyPatch, prior_reads: bool) -> None:
    """参数为是否有先前探测；权限失败不新增读取或正式执行，但不得否认先前实际读取。"""
    from test_planner_o3_pipeline import _plan

    state = _reads(_state(), ["success"] if prior_reads else [])
    state["research_plan"] = _plan().model_dump(mode="json")
    resolver = AsyncMock(side_effect=PermissionError("access revoked"))
    adapter = AsyncMock()
    delta = asyncio.run(pipeline.execute_planner_state(state, llm=object(), settings=SimpleNamespace(),
        adapters={"local_direct_text": adapter}, request_resolver=resolver))
    assert delta["source_status"]["local_evidence_execution"] == ("attempted" if prior_reads else "not_run")
    assert delta["planner_trace"]["read_facts"]["actual_call_count"] == int(prior_reads)
    assert delta["final_delivery_applicability"] == "pre_execution_failure"
    assert delta["error_code"] == "planner_execution_before_read_failed"
    assert delta["planner_trace"]["read_facts"]["execution_started"] is False
    adapter.assert_not_awaited()


@pytest.mark.parametrize("reuse", [False, True])
def test_formal_execution_failure_keeps_attempt_fact_and_applicability(monkeypatch: pytest.MonkeyPatch, reuse: bool) -> None:
    """参数为正文是否复用；真实读取或复用后的推理失败均保留正式执行状态，调用数不虚增。"""
    from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport
    from test_planner_o3_pipeline import _plan, _request_key

    state = _state()
    plan = _plan()
    adapter = AsyncMock(return_value=EvidenceReadResult((direct_evidence(),),
                              KnowledgeReadReport(operation="sections", status="complete")))
    provider = AsyncMock(side_effect=RuntimeError("reasoning failed"))
    monkeypatch.setattr(pipeline, "LLMClaimDraftProvider", lambda **_kwargs: provider)

    async def scenario() -> dict[str, Any]:
        """无参数；可先读取同请求，再经正式执行复用它并触发推理错误。"""
        current = state
        if reuse:
            context = pipeline._execution_context(current, plan, request_resolver=_request_key)
            await read_evidence(adapter, plan.units[0], context)
            current = {**current, **pipeline._execution_snapshot(context)}
        return await pipeline.execute_planner_state({**current, "research_plan": plan.model_dump(mode="json")},
            llm=object(), settings=SimpleNamespace(), adapters={"local_direct_text": adapter}, request_resolver=_request_key)

    delta = asyncio.run(scenario())
    _assert_read_fact(delta, calls=1, success=True, formal=True)
    assert adapter.await_count == 1
    assert delta["final_delivery_applicability"] == "research"
    assert delta["final_delivery_report"]["status"] == "failed"
    assert delta["error_code"] == ("planner_execution_before_read_failed" if reuse else "planner_execution_failed")


def test_synthesis_input_failure_uses_receipts_not_old_source_status() -> None:
    """无参数；综合输入失败时旧状态不能覆盖当前失败投影，正式研究仍未完成。"""
    state = {**_reads(_state(), ["success"]), "source_status": {"planner": "executed", "local_evidence_execution": "not_run"}}
    delta = asyncio.run(pipeline.synthesize_planner_state(state, object()))
    _assert_read_fact(delta, calls=1, success=True, formal=True)
    assert delta["planner_trace"]["read_facts"]["stage"] == "synthesize"


def test_provider_http_failure_shares_safe_summary_with_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身；模型供应商短摘要进入用户回答，技术消息仍留诊断而不泄漏到正文。"""
    import requests
    from app.services.llm_service import OpenAICompatibleLLM, classify_llm_upstream_error

    response = requests.Response()
    response.status_code = 404
    response._content = b'{"error":{"message":"No endpoints found for model"}}'
    model = OpenAICompatibleLLM(base_url="https://offline.invalid/v1", api_key="offline-test", model="offline",
                                temperature=0.0, timeout=1)
    error = model._decorate_http_error(
        requests.exceptions.HTTPError("404", response=response), "https://offline.invalid/v1/chat/completions")
    delta = _analyze(monkeypatch, _reads(_state(), ["success"]), error)
    assert classify_llm_upstream_error(error)[1] == delta["error_summary"]
    assert "https://offline.invalid" not in delta["answer"]
    assert delta["planner_trace"]["trace"]["technical_message"] == str(error)


def test_final_provider_failure_keeps_body_report_and_safe_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为最终模型替身；通过真实综合入口验证安全原因、累计读取与原同次交付字段不被重写。"""
    from app.services.llm_service import ProviderHTTPError
    from test_planner_o3_pipeline import _plan

    state = _reads(_state(), ["success"] * 3)
    plan = _plan(materials=True)
    state.update(research_plan=plan.model_dump(mode="json"), research_unit_results=[{
        "unit_id": "u1", "status": "evidence_found", "evidence_ids": [direct_evidence()["evidence_id"]], "claims": [],
    }])
    error = ProviderHTTPError("technical requested_url=https://offline.invalid", response=None,
                              user_summary="模型服务返回404。服务商说明：No endpoints found for model")
    provider = SimpleNamespace(generate=AsyncMock(side_effect=error))
    monkeypatch.setattr(pipeline, "LLMPlannerFinalAnswerProvider", lambda **_kwargs: provider)
    original_synthesis = pipeline.synthesize_planner_answer_async
    generated = []

    async def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """参数沿用正式综合合同；保存同次原始交付载荷，供核对后续只追加诊断说明。"""
        payload = await original_synthesis(*args, **kwargs)
        generated.append(deepcopy(payload))
        return payload

    monkeypatch.setattr(pipeline, "synthesize_planner_answer_async", capture)
    delta = asyncio.run(pipeline.synthesize_planner_state(state, object()))
    _assert_read_fact(delta, calls=3, success=True, formal=True)
    assert provider.generate.await_count == 1
    assert delta["final_delivery_report"]["status"] == "failed"
    assert delta["answer"].startswith(generated[0]["answer"])
    assert "requested_url" not in str(delta["final_delivery_report"])
    assert delta["source_status_reasons"]["planner_final"] == error.user_summary
    for key in ("final_delivery_report", "final_citation_bindings", "final_generation_attempt", "planner_final_blocks"):
        assert delta[key] == generated[0][key]


@pytest.mark.parametrize("error", [RuntimeError("pure reasoning failed"), _structured_failure()])
def test_formal_reasoning_without_local_calls_keeps_execution_error_identity(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """参数为执行期模型错误；零读取的实际工作仍属正式执行，原执行错误码不被结构错误重分类。"""
    from test_planner_o3_pipeline import _plan

    async def fail_reasoning(**kwargs: Any) -> Any:
        """参数为正式执行上下文；替代执行依赖，在真实工作标志写入后抛出模型错误。"""
        kwargs["context"].execution_state.mark_execution_started()
        raise error

    monkeypatch.setattr(pipeline, "execute_with_repair", fail_reasoning)
    state = {**_state(), "research_plan": _plan().model_dump(mode="json")}
    adapter = AsyncMock()
    delta = asyncio.run(pipeline.execute_planner_state(state, llm=object(), settings=SimpleNamespace(),
                         adapters={"local_direct_text": adapter}))
    _assert_read_fact(delta, calls=0, success=False, formal=True)
    assert delta["error_code"] == "planner_execution_before_read_failed"
    assert delta["final_delivery_report"]["status"] == "failed"
    adapter.assert_not_awaited()


def test_protocol_failure_uses_safe_summary_at_actual_analysis_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为规划端口替身；正式分析入口保留调用失败和已读资料，协议错误不成为结构回修。"""
    from app.services.llm_service import ProviderProtocolError, classify_llm_upstream_error

    error = ProviderProtocolError("invalid_sse_json", line_number=2, data_text="脱敏坏片段")
    state = _reads(_state(), ["success"])
    original = deepcopy(state)
    delta = _analyze(monkeypatch, state, error)
    _assert_read_fact(delta, calls=1, success=True)
    classified = classify_llm_upstream_error(error)
    assert classified is not None
    assert delta["error_summary"] == classified[1] == error.user_summary
    assert delta["error_code"] != "structured_output_failed"
    assert "ProviderProtocolError" not in delta["answer"] and "脱敏坏片段" not in delta["answer"]
    assert delta["final_delivery_applicability"] == "pre_execution_failure" and state == original


def test_protocol_failure_at_actual_synthesis_entry_keeps_saved_research(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为最终模型替身；正式综合入口保留研究成果，交付明确失败且安全原因不带技术类名。"""
    from app.services.llm_service import ProviderProtocolError
    from app.models.research_planner_contracts import ResearchUnitResult
    from test_planner_o3_pipeline import _plan

    state = _reads(_state(), ["success"])
    plan = _plan(materials=True)
    state = {**state, "research_plan": plan.model_dump(mode="json"), "research_unit_results": [{
        "unit_id": "u1", "status": "evidence_found", "evidence_ids": [direct_evidence()["evidence_id"]], "claims": [],
    }]}
    original = deepcopy(state)
    error = ProviderProtocolError("non_object_sse_data", line_number=2, data_text="[]")
    provider = SimpleNamespace(generate=AsyncMock(side_effect=error))
    monkeypatch.setattr(pipeline, "LLMPlannerFinalAnswerProvider", lambda **_kwargs: provider)
    delta = asyncio.run(pipeline.synthesize_planner_state(state, object()))
    assert provider.generate.await_count == 1 and delta["final_delivery_report"]["status"] == "failed"
    assert delta["source_status_reasons"]["planner_final"] == error.user_summary
    assert "ProviderProtocolError" not in delta["answer"]
    assert delta["research_plan"] == state["research_plan"]
    assert delta["research_unit_results"] == [ResearchUnitResult.model_validate(item).model_dump(mode="json")
                                              for item in state["research_unit_results"]]
    assert delta["evidence_records"] == state["evidence_records"] and state == original
