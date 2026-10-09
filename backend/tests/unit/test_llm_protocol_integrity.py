"""【模型服务】【传输完整性】真实调用入口、结束事实及隔离持久化回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from pydantic import BaseModel, ConfigDict

from app.middleware.trace_context import (
    TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx,
)
from app.services.llm_service import OpenAICompatibleLLM, ProviderProtocolError, classify_llm_upstream_error
from test_llm_streaming import _FakeSSEResponse, _sse_chunk

SECRET = 'fixture-key-"quoted\\private'
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "completion_tokens_details": {"reasoning_tokens": 3}}
OUTPUT_LIMIT = 321
LONG_DIAGNOSTIC_CHARS = 600


class ProtocolDraft(BaseModel):
    """只约束完整助手正文，用于区分传输损坏与结构回修。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str


class InterruptedResponse(_FakeSSEResponse):
    """已发送用量后抛出原连接异常，证明异常身份及已见事实同时保留。"""

    def __init__(self, lines, error):
        """参数为已收到行和原异常；初始化本地响应，无返回值。"""
        super().__init__(lines)
        self.error = error

    def iter_content(self, chunk_size=None):
        """参数为分片大小；先产出原字节，再抛出同一个异常对象。"""
        yield from super().iter_content(chunk_size)
        raise self.error


class ConcurrentResponse(_FakeSSEResponse):
    """两个响应在读取阶段会合，验证同一客户端的实际重叠调用。"""

    def __init__(self, lines, gate):
        """参数为响应行与同步点；初始化独立本地字节流，无返回值。"""
        super().__init__(lines)
        self.gate = gate

    def iter_content(self, chunk_size=None):
        """参数为分片大小；两次调用都进入后产出各自字节。"""
        self.gate.wait(timeout=5)
        yield from super().iter_content(chunk_size)


@pytest.fixture
def runtime(monkeypatch):
    """参数为测试注入器；返回客户端、HTTP/缓存/等待替身与trace，禁止真实队列和网络调用。"""
    queue = Mock()
    queue.policy = {"enabled": False}
    cache = Mock()
    cache.get_llm_response.return_value = None
    post = Mock(side_effect=AssertionError("未声明的HTTP请求"))
    wait = Mock()
    monkeypatch.setattr("app.services.llm_service.get_provider_queue", lambda: queue)
    monkeypatch.setattr("app.services.llm_service.get_cache", lambda: cache)
    monkeypatch.setattr("app.services.llm_service.requests.post", post)
    monkeypatch.setattr("app.services.llm_service.wait_for_provider_retry", wait)
    monkeypatch.setattr("app.services.llm_service._structured_output_native_enabled", lambda: False)
    monkeypatch.setattr("app.services.llm_service._structured_output_repair_max", lambda: 1)
    client = OpenAICompatibleLLM("fixture", SECRET, "https://example.test/v1", 0, 5,
        request_exception_max_retries=2, empty_response_max_retries=2)
    monkeypatch.setattr(client, "_streaming_enabled", lambda: True)
    trace = TraceContext(request_id="protocol-integrity", user_id="fixture")
    token = set_trace_ctx(trace)
    try:
        yield SimpleNamespace(client=client, post=post, cache=cache, wait=wait, trace=trace)
    finally:
        reset_trace_ctx(token)


def events(trace, stage):
    """参数为trace和事件阶段；返回冻结后同阶段的原始载荷。"""
    return [item["payload"] for item in export_evidence_observations(trace)["evidence_observations"]
            if item["stage"] == stage]


def data_line(payload):
    """参数为合成供应商对象；返回单条JSON data行。"""
    return "data: " + json.dumps(payload, ensure_ascii=False)


def response_lines(content, *, reason=None, usage=None, done=True):
    """参数为正文、真实结束原因、用量和DONE选择；返回合成响应行，不补造默认stop。"""
    lines = [": comment", "", _sse_chunk(content)]
    if reason is not None:
        lines.append(data_line({"choices": [{"delta": {}, "finish_reason": reason}]}))
    if usage is not None:
        lines.append(data_line({"choices": [], "usage": usage}))
    return [*lines, "data: [DONE]"] if done else lines


def json_response(content, *, reason=None, usage=None):
    """参数为正文及供应商原元数据；返回本地requests响应，不执行网络。"""
    body = {"choices": [{"message": {"content": content}, "finish_reason": reason}]}
    if usage is not None:
        body["usage"] = usage
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(body).encode("utf-8")
    return response


@pytest.mark.parametrize("bad_data,code", [
    ('{"broken":', "invalid_sse_json"), ("[]", "non_object_sse_data"),
    ("null", "non_object_sse_data"), ('"text"', "non_object_sse_data"),
    ("1", "non_object_sse_data"), ("true", "non_object_sse_data"),
])
def test_bad_data_fails_once_even_when_remaining_content_would_be_valid(runtime, bad_data, code):
    """参数为坏data及对应机器码；真实结构入口立即失败，不回修、不缓存或继续读取DONE。"""
    runtime.post.side_effect = [_FakeSSEResponse([
        _sse_chunk('{"answer":"A'), "data: " + bad_data, _sse_chunk('B"}'), "data: [DONE]",
    ])]
    with pytest.raises(ProviderProtocolError) as caught:
        runtime.client.with_structured_output(ProtocolDraft).invoke("固定问题")
    assert runtime.post.call_count == 1
    runtime.wait.assert_not_called()
    runtime.cache.set_llm_response.assert_not_called()
    assert classify_llm_upstream_error(caught.value) == (502, caught.value.user_summary)
    assert str(caught.value) == caught.value.user_summary
    content = events(runtime.trace, "model_response_content")
    assert len(content) == 1 and content[0]["response_text"] == '{"answer":"A'
    assert content[0]["http_status"] == 200 and content[0]["status"] == "failed"
    assert content[0]["stream_done_received"] is False and content[0]["content_complete"] is False
    assert content[0]["protocol_error"]["code"] == code
    assert content[0]["protocol_error"]["line_number"] == 2
    assert content[0]["error_boundary"] == "stream_protocol"
    assert not events(runtime.trace, "structured_response_validation")
    assert not events(runtime.trace, "structured_path")
    end = events(runtime.trace, "structured_operation_end")[0]
    assert end["status"] == "failed" and end["repair_count"] == 0 and not end["repair_succeeded"]
    calls = export_evidence_observations(runtime.trace)["model_call_observations"]
    assert calls[0]["retry_count"] == 0 and calls[0]["http_attempt_count"] == 1


def test_protocol_diagnostic_redacts_complete_secret_before_preview_limit(runtime):
    """参数为真实调用夹具；凭证先完整脱敏后截断，错误摘要与日志不含原始坏片段。"""
    bad = json.dumps({"authorization": "Bearer other-secret", "message": SECRET,
                      "url": "https://alice:password@example.test/p?token=query-secret"})[:-1]
    runtime.post.side_effect = [_FakeSSEResponse(["data: " + bad, "data: [DONE]"])]
    with pytest.raises(ProviderProtocolError) as caught:
        runtime.client.invoke("固定问题")
    diagnostic = events(runtime.trace, "model_response_content")[0]["protocol_error"]
    for sensitive in (SECRET, "other-secret", "alice:password", "query-secret", "quoted"):
        assert sensitive not in json.dumps(diagnostic, ensure_ascii=False)
        assert sensitive not in str(caught.value)
    assert diagnostic["json_error_offset"] is not None
    assert "[REDACTED]" in diagnostic["data_preview"]


def test_long_bad_data_keeps_redacted_tail_beyond_preview(runtime):
    """参数为真实调用夹具；240字符后的错误与完整脱敏尾部仍可复查，用户摘要保持固定。"""
    body = json.dumps({"padding": "长" * LONG_DIAGNOSTIC_CHARS, "secret": SECRET,
                       "tail": "尾部定位", "url": "https://alice:password@example.test/p?token=query-secret"},
                      ensure_ascii=False)
    bad = body[:-1] + ', "broken":'
    runtime.post.side_effect = [_FakeSSEResponse(["data: " + bad, "data: [DONE]"])]
    with pytest.raises(ProviderProtocolError) as caught:
        runtime.client.invoke("固定问题")
    diagnostic = events(runtime.trace, "model_response_content")[0]["protocol_error"]
    assert diagnostic["json_error_offset"] > LONG_DIAGNOSTIC_CHARS
    assert len(diagnostic["data_text"]) > len(diagnostic["data_preview"])
    assert "尾部定位" in diagnostic["data_text"] and diagnostic["data_text"].endswith('"broken":')
    assert "尾部定位" not in diagnostic["data_preview"]
    for sensitive in (SECRET, "quoted", "alice:password", "query-secret"):
        assert sensitive not in json.dumps(diagnostic, ensure_ascii=False)
        assert sensitive not in str(caught.value)
    assert str(caught.value) == caught.value.user_summary
    assert runtime.post.call_count == 1
    runtime.cache.set_llm_response.assert_not_called()


@pytest.mark.parametrize("reason", ["stop", "length", "content_filter", "vendor_reason", None])
def test_sse_retains_original_finish_reason_and_usage_without_automatic_retry(runtime, reason):
    """参数为供应商真实或缺失原因；原值进入raw及两份观察，原因本身不触发重试。"""
    runtime.post.side_effect = [_FakeSSEResponse(response_lines("中文😀", reason=reason, usage=USAGE), chunk_split=3)]
    result = runtime.client.invoke("固定问题")
    assert result.content == "中文😀" and result.raw["choices"][0]["finish_reason"] == reason
    assert result.raw["usage"] == USAGE and runtime.post.call_count == 1
    content = events(runtime.trace, "model_response_content")[0]
    usage = events(runtime.trace, "model_response_usage")[0]
    assert content["finish_reason"] == usage["finish_reason"] == reason
    assert content["usage"] == usage["usage"] == USAGE
    assert content["usage_status"] == usage["usage_status"] == "measured"
    assert content["stream_done_received"] is True and content["actual_response_mode"] == "sse"
    assert content["requested_output_limits"] == {key: {"present": False, "value": None}
                                                 for key in ("max_tokens", "max_completion_tokens")}


@pytest.mark.parametrize("requested,actual", [(True, "json"), (False, "sse"), (False, "json"), (True, "sse")])
def test_requested_mode_is_separate_from_actual_response(runtime, monkeypatch, requested, actual):
    """参数为请求与实际模式；模式适配只发一次HTTP，非流式原因同样如实保留。"""
    monkeypatch.setattr(runtime.client, "_streaming_enabled", lambda: requested)
    response = (_FakeSSEResponse(response_lines("正文", reason="length", usage=USAGE)) if actual == "sse"
                else json_response("正文", reason="length", usage=USAGE))
    runtime.post.side_effect = [response]
    result = runtime.client.invoke("固定问题")
    observed = events(runtime.trace, "model_response_content")[0]
    assert runtime.post.call_count == 1 and result.content == "正文"
    assert observed["requested_stream"] is requested and observed["actual_response_mode"] == actual
    assert observed["finish_reason"] == "length" and observed["usage"] == USAGE
    assert observed["stream_done_received"] == (True if actual == "sse" else None)


@pytest.mark.parametrize("kind", ["bad_data", "missing_done", "chunked_error", "provider_error"])
def test_failure_keeps_usage_and_reason_seen_before_failure(runtime, kind):
    """参数为实际失败边界；用量和原因保留，未读DONE不补造，原连接异常身份不变。"""
    lines = response_lines("半截正文", reason="length", usage=USAGE, done=False)
    original = requests.exceptions.ChunkedEncodingError("fixture broken chunk")
    if kind == "bad_data":
        response = _FakeSSEResponse([*lines, 'data: {"broken":', "data: [DONE]"])
    elif kind == "chunked_error":
        response = InterruptedResponse(lines, original)
    elif kind == "provider_error":
        response = _FakeSSEResponse([*lines, data_line({"error": {"message": "upstream failure"}})])
    else:
        response = _FakeSSEResponse(lines)
    runtime.client.request_exception_max_retries = 0
    runtime.post.side_effect = [response]
    expected = ProviderProtocolError if kind == "bad_data" else (
        requests.exceptions.ChunkedEncodingError if kind == "chunked_error" else requests.ConnectionError)
    with pytest.raises(expected) as caught:
        runtime.client.invoke("固定问题")
    if kind == "chunked_error":
        assert caught.value is original
    content = events(runtime.trace, "model_response_content")[0]
    usage = events(runtime.trace, "model_response_usage")[0]
    assert content["usage"] == usage["usage"] == USAGE
    assert content["finish_reason"] == usage["finish_reason"] == "length"
    assert content["usage_status"] == usage["usage_status"] == "measured"
    assert content["response_text"] == "半截正文" and content["stream_done_received"] is False
    assert runtime.post.call_count == 1
    runtime.cache.set_llm_response.assert_not_called()


def test_retry_snapshots_do_not_reuse_previous_attempt_facts(runtime):
    """参数为调用夹具；断流后的成功尝试不继承此前用量、原因、正文和失败诊断。"""
    runtime.post.side_effect = [
        _FakeSSEResponse(response_lines("旧片段", reason="length", usage=USAGE, done=False)),
        _FakeSSEResponse(response_lines("新正文")),
    ]
    result = runtime.client.invoke("固定问题")
    records = events(runtime.trace, "model_response_content")
    assert result.content == "新正文" and runtime.post.call_count == 2
    assert [item["http_attempt"] for item in records] == [1, 2]
    assert records[0]["call_id"] == records[1]["call_id"]
    assert records[0]["usage"] == USAGE and records[1]["usage"] is None
    assert records[1]["finish_reason"] is None and records[1]["protocol_error"] is None
    assert [item["content_complete"] for item in records] == [False, True]
    runtime.cache.set_llm_response.assert_called_once()


def test_protocol_failure_during_existing_repair_does_not_add_another_repair(runtime):
    """参数为结构入口夹具；既有一次回修遇坏协议立即失败，不伪报回修成功或再发请求。"""
    runtime.post.side_effect = [json_response("bad assistant JSON"), _FakeSSEResponse(['data: {"broken":'])]
    with pytest.raises(ProviderProtocolError):
        runtime.client.with_structured_output(ProtocolDraft).invoke("固定问题")
    assert runtime.post.call_count == 2
    validations = events(runtime.trace, "structured_response_validation")
    assert len(validations) == 1 and validations[0]["raw_json_valid"] is False
    end = events(runtime.trace, "structured_operation_end")[0]
    assert end["repair_count"] == 1 and not end["repair_succeeded"] and end["status"] == "failed"
    runtime.wait.assert_not_called()
    runtime.cache.set_llm_response.assert_called_once()
    assert runtime.cache.set_llm_response.call_args.args[1] == "bad assistant JSON"


def test_complete_assistant_json_error_still_uses_existing_repair(runtime):
    """参数为结构入口夹具；完整传输的助手JSON错误仍回修一次，不误判为流协议错误。"""
    runtime.post.side_effect = [_FakeSSEResponse(response_lines("bad assistant JSON")),
                                _FakeSSEResponse(response_lines('{"answer":"修复正文"}'))]
    assert runtime.client.with_structured_output(ProtocolDraft).invoke("固定问题").answer == "修复正文"
    assert runtime.post.call_count == 2
    assert all(item["content_complete"] and item["protocol_error"] is None
               for item in events(runtime.trace, "model_response_content"))
    assert events(runtime.trace, "structured_operation_end")[0]["repair_succeeded"] is True


def test_cache_hit_has_no_new_http_or_response_facts(runtime):
    """参数为缓存夹具；命中正文保持公开合同，不补造历史停止原因或本次HTTP记录。"""
    runtime.cache.get_llm_response.return_value = "历史缓存正文"
    result = runtime.client.invoke("固定问题")
    assert result.raw == {"cache_hit": True}
    runtime.post.assert_not_called()
    assert not events(runtime.trace, "model_response_content")
    assert not events(runtime.trace, "model_response_usage")


def test_no_response_headers_leaves_actual_mode_unknown(runtime):
    """参数为调用夹具；连接失败前没有响应头，不把请求模式写成实际模式。"""
    runtime.post.side_effect = requests.ConnectionError("fixture offline")
    with pytest.raises(requests.ConnectionError):
        runtime.client.invoke("固定问题")
    records = events(runtime.trace, "model_response_content")
    assert len(records) == runtime.post.call_count == 3
    assert all(item["actual_response_mode"] is None and item["response_content_type"] is None
               and item["http_status"] is None and item["stream_done_received"] is None for item in records)


def test_output_limit_observation_uses_actual_admitted_payload(runtime):
    """参数为调用夹具；仅观察实际已发送字段，不修改常规请求配置或补造未设置限制。"""
    runtime.post.side_effect = [json_response("正文")]
    payload = {"model": "fixture", "messages": [{"role": "user", "content": "固定问题"}],
               "max_completion_tokens": OUTPUT_LIMIT}
    result = runtime.client._invoke_http_attempt({}, payload, use_stream=False, call_record=None,
                                                request_identity={"call_id": "explicit-payload", "attempt": 1})
    assert result.content == "正文" and runtime.post.call_args.kwargs["json"] == payload
    observed = events(runtime.trace, "model_response_content")[0]
    assert observed["call_id"] == "explicit-payload" and observed["http_attempt"] == 1
    assert observed["requested_output_limits"]["max_completion_tokens"] == {"present": True, "value": OUTPUT_LIMIT}
    assert observed["requested_output_limits"]["max_tokens"] == {"present": False, "value": None}


@pytest.mark.asyncio
async def test_same_client_concurrent_calls_keep_distinct_success_and_failure(runtime):
    """参数为共享客户端；两次重叠请求不共享可变快照，用量和协议失败只属于各自call_id。"""
    gate = threading.Barrier(2)
    runtime.post.side_effect = [
        ConcurrentResponse(response_lines("成功正文"), gate),
        ConcurrentResponse([*response_lines("失败片段", reason="length", usage=USAGE, done=False), "data: []"], gate),
    ]
    results = await asyncio.gather(runtime.client.ainvoke("第一题"), runtime.client.ainvoke("第二题"),
                                   return_exceptions=True)
    assert sum(isinstance(item, ProviderProtocolError) for item in results) == 1
    records = events(runtime.trace, "model_response_content")
    assert len({item["call_id"] for item in records}) == runtime.post.call_count == 2
    by_text = {item["response_text"]: item for item in records}
    assert by_text["成功正文"]["usage"] is None and by_text["成功正文"]["finish_reason"] is None
    assert by_text["成功正文"]["protocol_error"] is None and by_text["成功正文"]["content_complete"] is True
    assert by_text["失败片段"]["usage"] == USAGE and by_text["失败片段"]["finish_reason"] == "length"
    assert by_text["失败片段"]["content_complete"] is False


@pytest.mark.asyncio
async def test_success_and_failure_facts_survive_full_trace_size_cap_and_repository(runtime, monkeypatch):
    """参数为真实调用与隔离仓储；完整trace经强制裁剪和实际序列化写入后，两次事实不丢失。"""
    from app.middleware import trace_context
    from app.middleware.trace_builders import build_full_trace_payload
    from app.repositories import observability_repository

    runtime.post.side_effect = [
        _FakeSSEResponse(response_lines("完整正文", reason="stop", usage=USAGE)),
        _FakeSSEResponse([*response_lines("保留片段", reason="length", usage=USAGE, done=False), "data: []"]),
    ]
    runtime.client.invoke("成功问题")
    with pytest.raises(ProviderProtocolError):
        runtime.client.invoke("失败问题")
    expected = events(runtime.trace, "model_response_content")
    full = build_full_trace_payload(runtime.trace, {"query": "固定问题"},
        call_summary_fn=trace_context.model_call_to_summary, call_debug_fn=trace_context.model_call_to_debug)
    execute = AsyncMock()
    repo = observability_repository.ObservabilityRepository()
    monkeypatch.setattr(observability_repository, "execute", execute)
    monkeypatch.setattr(observability_repository, "get_observability_repository", lambda: repo)
    monkeypatch.setattr(trace_context, "OBS_TRACE_MAX_BYTES", 1)
    await trace_context.safe_write_trace(full, "protocol-integrity", "fixture-session", "固定问题", "fixture")
    execute.assert_awaited_once()
    recorded_call = execute.await_args
    assert recorded_call is not None
    saved = json.loads(recorded_call.args[4])
    assert saved["debug"] == {"truncated": True, "reason": "size_cap"}
    assert [item["payload"] for item in saved["model_response_observations"]] == expected
    assert [item["status"] for item in expected] == ["completed", "failed"]
    assert [item["http_status"] for item in expected] == [200, 200]
    assert full["debug"].get("reason") != "size_cap"
