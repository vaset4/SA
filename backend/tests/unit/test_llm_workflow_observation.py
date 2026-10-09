"""【评测观察】【真实边界】用本地HTTP替身验证归因、结构回修和业务校验。作者：xxx。"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import Mock

import pytest
import requests
from pydantic import BaseModel, ConfigDict

from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx
from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft
from app.services.internal_research.research_planner_validator import PlannerValidationOptions, validate_or_repair_plan
from app.services.llm_observation import redact_error, safe_endpoint
from app.services.llm_service import OpenAICompatibleLLM, StructuredOutputError
from research_planner_test_utils import make_draft, make_unit
from research_planner_strategy_test_utils import strategy_fixture

SECRET = "fixture-private-key"


class Draft(BaseModel):
    """结构字段通过不代表工作流成功；此模型仅验证真实解析边界。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str


def response(content: str, *, status: int = 200, envelope: bool = True) -> requests.Response:
    """参数为助手内容、状态及信封开关；返回本地响应，不进行网络调用。"""
    result = requests.Response()
    result.status_code = status
    result.headers["Content-Type"] = "application/json"
    result._content = (json.dumps({"choices": [{"message": {"content": content}}]}) if envelope else content).encode()
    return result


@pytest.fixture
def observed_runtime(monkeypatch):
    """参数为替身管理器；返回真实客户端、HTTP替身、缓存及独立trace，所有网络入口已替换。"""
    cache = Mock()
    cache.get_llm_response.return_value = None
    post = Mock(side_effect=AssertionError("unexpected network"))
    monkeypatch.setattr("app.services.llm_service.get_cache", lambda: cache)
    monkeypatch.setattr("app.services.llm_service.requests.post", post)
    monkeypatch.setattr("app.services.llm_service.time.sleep", lambda seconds: None)
    monkeypatch.setattr("app.services.llm_service._structured_output_native_enabled", lambda: False)
    client = OpenAICompatibleLLM("fixture", SECRET, "https://example.test/v1", 0, 5,
                                 request_exception_max_retries=1, empty_response_max_retries=1)
    monkeypatch.setattr(client, "_streaming_enabled", lambda: False)
    trace = TraceContext(request_id="workflow-observation")
    token = set_trace_ctx(trace)
    try:
        yield client, post, cache, trace
    finally:
        reset_trace_ctx(token)


def events(trace: TraceContext, stage: str) -> list[dict]:
    """参数为trace及事件名；返回冻结后的指定事件载荷。"""
    return [event["payload"] for event in export_evidence_observations(trace)["evidence_observations"]
            if event["stage"] == stage]


def test_http_retry_is_not_schema_repair(observed_runtime):
    """503恢复保留两次HTTP尝试，但首次完整结构响应仅一次且直接通过。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response('{"error":{"code":"unavailable"}}', status=503, envelope=False),
                        response('{"answer":"ok"}')]
    assert client.with_structured_output(Draft).invoke("question").answer == "ok"
    attempts = events(trace, "model_response_usage")
    assert [item["http_status"] for item in attempts] == [503, 200]
    assert [item["http_attempt"] for item in attempts] == [1, 2]
    assert attempts[0]["provider_error_code"] == "unavailable"
    validations = events(trace, "structured_response_validation")
    assert len(validations) == 1 and validations[0]["schema_valid"] is True
    assert validations[0]["repair_count"] == 0
    assert {item["operation_id"] for item in attempts} == {validations[0]["operation_id"]}
    assert events(trace, "structured_operation_end")[0]["status"] == "success"


def test_invalid_assistant_json_and_repair_api_failure_coexist(observed_runtime):
    """首次助手JSON失败后回修403，不得用API失败抹掉首轮格式错误。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response("bad-json"), response("forbidden", status=403, envelope=False)]
    with pytest.raises(requests.HTTPError):
        client.with_structured_output(Draft).invoke("question")
    validation = events(trace, "structured_response_validation")
    assert len(validation) == 1 and validation[0]["raw_json_valid"] is False
    assert validation[0]["schema_valid"] is False
    attempts = events(trace, "model_response_usage")
    assert [item["status"] for item in attempts] == ["completed", "failed"]
    assert attempts[1]["error_boundary"] == "http_status"
    end = events(trace, "structured_operation_end")[0]
    assert end["phase"] == "repair" and end["repair_count"] == 1 and end["status"] == "failed"


def test_api_envelope_error_has_no_assistant_validation(observed_runtime):
    """HTTP200信封损坏发生在助手内容之前，不给模型结构内容编造零分。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response('{"broken":', envelope=False)]
    with pytest.raises(ValueError, match="LLM JSON"):
        client.with_structured_output(Draft).invoke("question")
    assert not events(trace, "structured_response_validation")
    attempt = events(trace, "model_response_usage")[0]
    assert attempt["error_boundary"] == "api_envelope" and attempt["http_status"] == 200


def test_schema_repair_preserves_original_and_final_compliance(observed_runtime):
    """缺字段后的严格回修成功不能覆盖首次错误，完整两次响应均可查。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response('{"wrong":1}'), response('{"answer":"fixed"}')]
    assert client.with_structured_output(Draft).invoke("question").answer == "fixed"
    validations = events(trace, "structured_response_validation")
    assert [(item["phase"], item["schema_valid"]) for item in validations] == [("initial", False), ("repair", True)]
    assert validations[1]["raw_json_valid"] is True
    assert validations[1]["normalization_applied"] is False
    assert [item["response_text"] for item in validations] == ['{"wrong":1}', '{"answer":"fixed"}']
    end = events(trace, "structured_operation_end")[0]
    assert end["original_compliance"] is False and end["final_compliance"] is True
    assert events(trace, "structured_operation_end")[0]["repair_succeeded"] is True


def test_bad_cache_uses_existing_repair_budget(observed_runtime):
    """坏缓存不算新生成，但重新请求只使用一次已存在的格式回修额度。"""
    client, post, cache, trace = observed_runtime
    cache.get_llm_response.return_value = "bad cached JSON"
    post.side_effect = [response('{"answer":"fresh"}')]
    assert client.with_structured_output(Draft).invoke("question").answer == "fresh"
    validations = events(trace, "structured_response_validation")
    assert [item["cache_hit"] for item in validations] == [True, False]
    assert [item["phase"] for item in validations] == ["initial", "repair"]
    assert len(events(trace, "model_response_usage")) == 1
    assert events(trace, "structured_operation_end")[0]["repair_count"] == 1


def test_cache_success_records_no_new_http(observed_runtime):
    """有效缓存保留schema检查，但不新增模型生成或HTTP尝试。"""
    client, post, cache, trace = observed_runtime
    cache.get_llm_response.return_value = '{"answer":"cached"}'
    assert client.with_structured_output(Draft).invoke("question").answer == "cached"
    assert not post.called and not events(trace, "model_response_usage")
    assert events(trace, "structured_response_validation")[0]["cache_hit"] is True


def test_native_rejection_is_separate_from_model_repair(observed_runtime, monkeypatch):
    """原生格式 400 拒绝是供应商错误，不去掉配置额外请求或记为模型回修。"""
    client, post, _cache, trace = observed_runtime
    monkeypatch.setattr("app.services.llm_service._structured_output_native_enabled", lambda: True)
    post.side_effect = [response("response_format unsupported", status=400, envelope=False),
                        response('{"answer":"ok"}')]
    with pytest.raises(requests.HTTPError):
        client.with_structured_output(Draft).invoke("question")
    assert post.call_count == 1
    assert [item["path"] for item in events(trace, "structured_path")] == ["native_rejected"]
    assert events(trace, "structured_operation_end")[0]["repair_count"] == 0
    assert not events(trace, "structured_response_validation")
    assert events(trace, "structured_operation_end")[0]["native_mode"] == "rejected"


def test_errors_and_endpoints_exclude_credentials(observed_runtime, caplog):
    """上游回显密钥仍需脱敏，接口身份移除用户、查询和片段。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response(json.dumps({"error": {"message": f"api_key={SECRET}"}}), status=403, envelope=False)]
    with pytest.raises(requests.HTTPError):
        client.with_structured_output(Draft).invoke("question")
    captured = json.dumps(export_evidence_observations(trace))
    assert SECRET not in captured and SECRET not in caplog.text
    assert safe_endpoint("https://user:pass@example.test/v1?api_key=private#secret") == "https://example.test/v1"
    assert "private" not in redact_error("https://user:pass@example.test/v1?api_key=private#secret")
    assert "opaque-token" not in redact_error("Authorization: Bearer opaque-token")


def test_business_scope_failure_links_actual_schema_operation(observed_runtime):
    """schema合法但论文越权由真实validator拒绝，保留操作关联及具体错误。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response(json.dumps(strategy_fixture(make_draft([make_unit(paper_ids=["other-paper"])]))))]
    draft = client.with_structured_output(ResearchPlanStrategyDraft).invoke("question")
    result = validate_or_repair_plan(draft.model_dump(mode="json"),
                                    options=PlannerValidationOptions(question="question", allowed_paper_ids=frozenset({"paper-a"})))
    assert not result.is_valid
    business = events(trace, "structured_business_validation")[0]
    validation = events(trace, "structured_response_validation")[0]
    assert business["operation_id"] == validation["operation_id"]
    assert business["valid"] is False and business["accepted"] is False and business["errors"]


def test_concurrent_operations_keep_call_identity(observed_runtime):
    """同一trace下并发结构化操作的HTTP、解析和终态不能串到兄弟调用。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = None
    post.return_value = response('{"answer":"ok"}')

    async def run():
        """无参数；并发执行两个本地替身调用并返回响应。"""
        return await asyncio.gather(client.with_structured_output(Draft).ainvoke("one"),
                                    client.with_structured_output(Draft).ainvoke("two"))

    assert len(asyncio.run(run())) == 2
    attempts = events(trace, "model_response_usage")
    assert len({item["operation_id"] for item in attempts}) == 2
    validations = {item["operation_id"]: item for item in events(trace, "structured_response_validation")}
    assert all(item["call_id"] == validations[item["operation_id"]]["call_id"] for item in attempts)


def test_exhausted_schema_repair_preserves_both_invalid_responses(observed_runtime):
    """两次完整响应均非法时保留两份事实及原StructuredOutputError，不回填最终成功。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response("bad one"), response("bad two")]
    with pytest.raises(StructuredOutputError):
        client.with_structured_output(Draft).invoke("question")
    assert [item["schema_valid"] for item in events(trace, "structured_response_validation")] == [False, False]
    assert events(trace, "structured_operation_end")[0]["status"] == "failed"


def test_empty_assistant_is_protocol_missing_not_schema_failure(observed_runtime):
    """空助手字段保留原回修行为，但未取得内容的尝试属于协议缺失，不能记JSON零分。"""
    client, post, _cache, trace = observed_runtime
    post.side_effect = [response(""), response('{"answer":"recovered"}')]
    assert client.with_structured_output(Draft).invoke("question").answer == "recovered"
    attempts = events(trace, "model_response_usage")
    assert attempts[0]["protocol_status"] == "failed" and attempts[0]["error_boundary"] == "api_envelope"
    validations = events(trace, "structured_response_validation")
    assert len(validations) == 1 and validations[0]["schema_valid"] is True
    assert events(trace, "structured_response_unavailable")[0]["reason"] == "empty_assistant_content"


def test_duplicate_output_fingerprints_do_not_guess_business_operation(observed_runtime):
    """相同合法输出发生两次时保留两候选ID，不能按出现顺序猜业务校验归属。"""
    client, post, _cache, trace = observed_runtime
    raw = json.dumps(strategy_fixture(make_draft()))
    post.side_effect = [response(raw), response(raw)]
    client.with_structured_output(ResearchPlanStrategyDraft).invoke("question one")
    second = client.with_structured_output(ResearchPlanStrategyDraft).invoke("question two")
    validate_or_repair_plan(second, options=PlannerValidationOptions(question="question"))
    business = events(trace, "structured_business_validation")[0]
    assert business["operation_id"] is None and len(business["matching_operation_ids"]) == 2


def test_malformed_error_url_preserves_original_transport_exception(observed_runtime):
    """错误正文的畸形URL只影响地址采集，不能取代原连接异常或改变既有重试次数。"""
    client, post, _cache, trace = observed_runtime
    original = requests.ConnectionError("upstream supplied https://[broken/path?token=private")
    post.side_effect = original
    with pytest.raises(requests.ConnectionError) as caught:
        client.with_structured_output(Draft).invoke("question")
    assert caught.value is original and post.call_count == 2
    assert [item["error_type"] for item in events(trace, "model_response_usage")] == ["ConnectionError"] * 2
    capture = export_evidence_observations(trace)
    assert any(item["stage"] == "endpoint_redaction" for item in capture["evidence_capture_errors"])
    assert "private" not in json.dumps(capture)


@pytest.mark.parametrize("structured", [False, True])
def test_transport_error_bare_credential_is_redacted_everywhere(observed_runtime, caplog, structured):
    """接口异常回显裸密钥时，HTTP、调用终态与重试日志均脱敏，原异常对象仍原样抛出。"""
    client, post, _cache, trace = observed_runtime
    caplog.set_level(logging.INFO, logger="app.services.llm_service")
    original = requests.ConnectionError(f"rejected credential {SECRET}")
    post.side_effect = original
    generator = client.with_structured_output(Draft) if structured else client
    with pytest.raises(requests.ConnectionError) as caught:
        generator.invoke("question")
    assert caught.value is original and SECRET in str(original)
    assert SECRET not in json.dumps(export_evidence_observations(trace))
    assert SECRET not in caplog.text
    assert "连接重试" in caplog.text
