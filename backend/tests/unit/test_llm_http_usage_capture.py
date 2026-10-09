"""实际HTTP尝试、完整SSE响应与provider用量采集回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import pytest
import requests

from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx
from app.services.llm_service import OpenAICompatibleLLM
from evals.suites.runtime import ModelRequestObserver, _capture_agent_resources

USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
         "completion_tokens_details": {"reasoning_tokens": 3}, "prompt_tokens_details": {"cached_tokens": 2}, "cost": 0.004}


class Response:
    """保持requests接口的本地响应，测试不会发出网络请求。"""

    status_code = 200

    def __init__(self, *, streaming: bool, body: dict | None = None, finished: bool = True, on_body=None):
        """参数为响应模式、正文和流终止状态；初始化可控本地响应。"""
        self.headers = {"Content-Type": "text/event-stream" if streaming else "application/json"}
        self.body = body or {"choices": [{"message": {"content": "实际回答"}}], "usage": USAGE}
        self.text = json.dumps(self.body)
        self.finished = finished
        self.on_body = on_body
        self.closed = False

    def close(self):
        """无参数；标记本地响应已释放，与真实requests响应保持一致。"""
        self.closed = True

    def raise_for_status(self):
        """无参数；当前响应状态为200，无返回值。"""

    def json(self):
        """无参数；返回本地JSON响应体。"""
        return self.body

    def iter_content(self, *, chunk_size):
        """参数为分片大小；产出正文、用量及可选DONE，允许检查读取正文前的采集状态。"""
        if self.on_body is not None:
            self.on_body()
        yield ('data: ' + json.dumps({"choices": [{"delta": {"content": "实际回答"}}]}) + '\n').encode()
        if "usage" in self.body:
            yield ('data: ' + json.dumps({"choices": [], "usage": self.body["usage"]}) + '\n').encode()
        if self.finished:
            yield b'data: [DONE]\n'


@pytest.fixture
def setup_runtime(monkeypatch):
    """参数为替身；隔离缓存和HTTP端口，返回客户端工厂与实际请求Mock。"""
    cache = Mock()
    cache.get_llm_response.return_value = None
    monkeypatch.setattr("app.services.llm_service.get_cache", lambda: cache)
    monkeypatch.setattr("app.services.llm_service.time.sleep", lambda seconds: None)
    post = Mock(side_effect=AssertionError("unexpected network attempt"))
    monkeypatch.setattr("app.services.llm_service.requests.post", post)
    return cache, post


def client(streaming: bool) -> OpenAICompatibleLLM:
    """参数为流模式；返回无真实凭证的客户端，重试次数沿用测试固定值。"""
    value = OpenAICompatibleLLM(model="test-model", api_key="fixture-key", base_url="https://example.test/v1", temperature=0.0,
                                timeout=10, request_exception_max_retries=1, empty_response_max_retries=1)
    value._streaming_enabled = lambda: streaming
    return value


def capture(value: OpenAICompatibleLLM, *, use_cache: bool = False):
    """参数为客户端与缓存选择；调用真实invoke并返回原始响应和冻结观察。"""
    trace = TraceContext(request_id="usage-capture", user_id="fixture")
    token = set_trace_ctx(trace)
    try:
        result = value.invoke([{"role": "user", "content": "固定测试问题"}], use_cache=use_cache)
        return result, export_evidence_observations(trace)
    finally:
        reset_trace_ctx(token)


@pytest.mark.parametrize("streaming", [False, True])
def test_success_retains_actual_attempt_and_complete_provider_usage(setup_runtime, streaming):
    """参数为隔离端口和流模式；一次真实传输计一次，保留provider所有原始计费字段。"""
    _cache, post = setup_runtime
    post.side_effect = None
    post.return_value = Response(streaming=streaming)
    result, observed = capture(client(streaming))
    assert observed["model_call_observations"][0]["http_attempt_count"] == post.call_count == 1
    events = [event["payload"] for event in observed["evidence_observations"] if event["stage"] == "model_response_usage"]
    assert len(events) == 1 and events[0]["status"] == "completed"
    assert events[0]["usage"] == USAGE == result.raw["usage"]
    assert events[0]["call_id"] == observed["model_call_observations"][0]["call_id"]
    assert events[0]["usage_status"] == "measured" and events[0]["duration_seconds"] >= 0
    sent = post.call_args.kwargs["json"]
    assert "stream_options" not in sent
    assert sent.get("stream", False) is streaming
    assert post.return_value.closed


def test_stream_headers_are_not_reported_as_completed(setup_runtime, monkeypatch):
    """参数为本地HTTP端口；收到响应头但尚未消费SSE时不能产生完成或用量记录。"""
    _cache, post = setup_runtime
    trace = TraceContext(request_id="headers", user_id="fixture")

    def check_headers():
        """无参数；在首次消费正文时确认此前只有请求参数记录。"""
        stages = [event["stage"] for event in export_evidence_observations(trace)["evidence_observations"]
                  if event["stage"].startswith("model_")]
        assert stages == ["model_request_parameters"]

    post.side_effect = None
    post.return_value = Response(streaming=True, on_body=check_headers)
    observer = ModelRequestObserver(send=OpenAICompatibleLLM._post_chat_completions_stream)
    monkeypatch.setattr(OpenAICompatibleLLM, "_post_chat_completions_stream", observer)
    token = set_trace_ctx(trace)
    try:
        client(True).invoke([{"role": "user", "content": "固定问题"}], use_cache=False)
    finally:
        reset_trace_ctx(token)


def test_interrupted_stream_retry_retains_failure_and_success(setup_runtime):
    """参数为HTTP替身；流未DONE的尝试记录失败，重试成功独立保存，不能头部就报成功。"""
    _cache, post = setup_runtime
    post.side_effect = [Response(streaming=True, body={"choices": []}, finished=False), Response(streaming=True)]
    _result, observed = capture(client(True))
    assert observed["model_call_observations"][0]["http_attempt_count"] == post.call_count == 2
    events = [event["payload"] for event in observed["evidence_observations"] if event["stage"] == "model_response_usage"]
    assert [event["status"] for event in events] == ["failed", "completed"]
    assert events[0]["usage_status"] == "missing" and events[0]["error_type"] == "ConnectionError"
    assert events[1]["usage"] == USAGE


def test_cached_response_has_zero_attempts_and_no_provider_usage(setup_runtime):
    """参数为缓存替身；缓存命中不调用HTTP，也不把旧token当成本次用量。"""
    cache, post = setup_runtime
    cache.get_llm_response.return_value = "缓存答案"
    _result, observed = capture(client(True), use_cache=True)
    assert observed["model_call_observations"][0]["http_attempt_count"] == post.call_count == 0
    assert observed["model_call_observations"][0]["cache_hit"] is True
    assert not observed["evidence_observations"]


def test_absent_usage_remains_missing_in_suite_resources(setup_runtime):
    """参数为HTTP替身；真实完整响应不含usage时仍为missing，套件不估算token。"""
    _cache, post = setup_runtime
    post.side_effect = None
    post.return_value = Response(streaming=True, body={"choices": []})
    _result, observed = capture(client(True))
    from evals.contracts import read_snapshot, snapshot

    raw = {"observation_snapshot": snapshot({"turns": [{"evidence_observation_snapshot": snapshot(observed)}]})}
    record = _capture_agent_resources({"run_record": raw, "agent_run": raw}, duration=0.01)
    resources = read_snapshot(record["run_record"]["observation_snapshot"])["usage"]
    assert resources["observed_call_count"] == 1 and resources["status"] == "missing"
    assert resources["model_http_calls"][0]["usage"] is None


def test_http_403_is_one_failed_attempt_without_usage(setup_runtime):
    """参数为HTTP替身；不重试403，保留一次实际请求失败，不能伪造用量或完成。"""
    _cache, post = setup_runtime
    response = requests.Response()
    response.status_code = 403
    response._content = b'{"error":"forbidden"}'
    post.side_effect = None
    post.return_value = response
    trace = TraceContext(request_id="forbidden", user_id="fixture")
    token = set_trace_ctx(trace)
    try:
        with pytest.raises(requests.exceptions.HTTPError):
            client(True).invoke([{"role": "user", "content": "固定问题"}], use_cache=False)
        observed = export_evidence_observations(trace)
    finally:
        reset_trace_ctx(token)
    call = observed["model_call_observations"][0]
    assert call["http_attempt_count"] == post.call_count == 1
    assert "HTTPError" in call["error"]
    outcome = next(event["payload"] for event in observed["evidence_observations"] if event["stage"] == "model_response_usage")
    assert outcome["status"] == "failed" and outcome["usage_status"] == "missing"
    assert outcome["usage"] is None


def test_exhausted_connections_preserve_each_real_attempt(setup_runtime):
    """参数为HTTP替身；耗尽重试时每次连接尝试均单列失败，错误继续向调用方抛出。"""
    _cache, post = setup_runtime
    post.side_effect = requests.exceptions.ConnectionError("fixture connection failure")
    trace = TraceContext(request_id="connections", user_id="fixture")
    token = set_trace_ctx(trace)
    try:
        with pytest.raises(requests.exceptions.ConnectionError, match="fixture connection failure"):
            client(False).invoke([{"role": "user", "content": "固定问题"}], use_cache=False)
        observed = export_evidence_observations(trace)
    finally:
        reset_trace_ctx(token)
    assert observed["model_call_observations"][0]["http_attempt_count"] == post.call_count == 2
    events = [event["payload"] for event in observed["evidence_observations"] if event["stage"] == "model_response_usage"]
    assert [event["http_attempt"] for event in events] == [1, 2]
    assert all(event["status"] == "failed" and event["usage"] is None for event in events)


def test_async_calls_keep_independent_attempt_usage_identity(setup_runtime):
    """参数为HTTP替身；同一trace内并发ainvoke保持独立call_id和尝试编号，不串用其他调用用量。"""
    _cache, post = setup_runtime
    post.side_effect = None
    post.return_value = Response(streaming=True)
    trace = TraceContext(request_id="async-usage", user_id="fixture")

    async def run_calls():
        """无参数；并发执行两个真实ainvoke适配器，无网络请求。"""
        token = set_trace_ctx(trace)
        try:
            await asyncio.gather(client(True).ainvoke("第一题", use_cache=False),
                                 client(True).ainvoke("第二题", use_cache=False))
        finally:
            reset_trace_ctx(token)

    asyncio.run(run_calls())
    observed = export_evidence_observations(trace)
    calls = {call["call_id"]: call for call in observed["model_call_observations"]}
    events = [event["payload"] for event in observed["evidence_observations"] if event["stage"] == "model_response_usage"]
    assert len(calls) == post.call_count == len(events) == 2
    assert {event["call_id"] for event in events} == calls.keys()
    assert all(event["http_attempt"] == calls[event["call_id"]]["http_attempt_count"] == 1 for event in events)


def test_observation_does_not_parse_response_twice(setup_runtime, monkeypatch):
    """参数为HTTP替身；套件observer不额外读取正文或重复JSON解析，原provider响应只消费一次。"""
    _cache, post = setup_runtime
    response = Response(streaming=False)
    response.json = Mock(return_value=response.body)
    post.side_effect = None
    post.return_value = response
    observer = ModelRequestObserver(send=OpenAICompatibleLLM._post_chat_completions)
    monkeypatch.setattr(OpenAICompatibleLLM, "_post_chat_completions", observer)
    _result, observed = capture(client(False))
    response.json.assert_called_once()
    assert [event["stage"] for event in observed["evidence_observations"] if event["stage"].startswith("model_")] == ["model_request_parameters", "model_response_content", "model_response_usage"]


def test_usage_survives_suite_freezing_without_fake_total(setup_runtime):
    """参数为HTTP替身；套件完整保留provider原始usage对象，不生成自行估算的token总数。"""
    from evals.contracts import read_snapshot, snapshot

    _cache, post = setup_runtime
    post.side_effect = None
    post.return_value = Response(streaming=True)
    _result, observed = capture(client(True))
    raw = {"observation_snapshot": snapshot({"turns": [{"evidence_observation_snapshot": snapshot(observed)}]})}
    record = _capture_agent_resources({"run_record": raw, "agent_run": raw}, duration=0.01)
    resources = read_snapshot(record["run_record"]["observation_snapshot"])["usage"]
    assert resources["status"] == "measured" and resources["observed_call_count"] == 1
    assert resources["model_http_calls"][0]["usage"] == USAGE
    assert "total_tokens" not in resources


@pytest.mark.parametrize("streaming", [False, True])
def test_usage_serialization_failure_preserves_answer_and_missing_resources(setup_runtime, streaming):
    """参数为HTTP替身和流模式；一次用量无法冻结不能改写答案，也不能由另一次有效用量掩盖缺失。"""
    from evals.contracts import read_snapshot, snapshot

    _cache, post = setup_runtime
    invalid_body = {"choices": [{"message": {"content": "实际回答"}}], "usage": {"cost": float("nan")}}
    post.side_effect = [Response(streaming=streaming, body=invalid_body), Response(streaming=streaming)]
    trace = TraceContext(request_id="capture-error", user_id="fixture")
    token = set_trace_ctx(trace)
    try:
        answers = [client(streaming).invoke("固定问题", use_cache=False).content for _ in range(2)]
        observed = export_evidence_observations(trace)
    finally:
        reset_trace_ctx(token)
    assert answers == ["实际回答", "实际回答"]
    assert post.call_count == 2
    assert all(call["http_attempt_count"] == 1 and call["error"] is None for call in observed["model_call_observations"])
    assert [error["stage"] for error in observed["evidence_capture_errors"]] == ["model_response_content", "model_response_usage"]
    raw = {"observation_snapshot": snapshot({"turns": [{"evidence_observation_snapshot": snapshot(observed)}]})}
    record = _capture_agent_resources({"run_record": raw, "agent_run": raw}, duration=0.01)
    resources = read_snapshot(record["run_record"]["observation_snapshot"])["usage"]
    assert resources["observed_call_count"] == 1
    assert resources["status"] == "missing"


def test_empty_completed_stream_retries_without_premature_success(setup_runtime):
    """参数为HTTP替身；只有DONE没有正文的流保留失败，后续重试成功才记录完成。"""
    _cache, post = setup_runtime
    empty = Response(streaming=True)
    empty.iter_content = Mock(return_value=iter([b"data: [DONE]\n"]))
    post.side_effect = [empty, Response(streaming=True)]
    result, observed = capture(client(True))
    assert result.content == "实际回答"
    assert observed["model_call_observations"][0]["http_attempt_count"] == post.call_count == 2
    events = [event["payload"] for event in observed["evidence_observations"] if event["stage"] == "model_response_usage"]
    assert [event["status"] for event in events] == ["failed", "completed"]
    assert events[0]["error_type"] == "ValueError" and events[0]["usage_status"] == "missing"


def test_json_parse_failure_preserves_original_error_and_one_attempt(setup_runtime):
    """参数为HTTP替身；JSON解析错误按既有规则抛出且不重试，记录一次失败传输。"""
    _cache, post = setup_runtime
    response = Response(streaming=False)
    response.json = Mock(side_effect=ValueError("fixture invalid JSON"))
    post.side_effect = None
    post.return_value = response
    trace = TraceContext(request_id="parse-error", user_id="fixture")
    token = set_trace_ctx(trace)
    try:
        with pytest.raises(ValueError, match="LLM JSON 解析失败"):
            client(False).invoke("固定问题", use_cache=False)
        observed = export_evidence_observations(trace)
    finally:
        reset_trace_ctx(token)
    assert observed["model_call_observations"][0]["http_attempt_count"] == post.call_count == 1
    outcome = next(event["payload"] for event in observed["evidence_observations"] if event["stage"] == "model_response_usage")
    assert outcome["status"] == "failed" and outcome["error_type"] == "ValueError"
    assert outcome["usage_status"] == "missing" and outcome["usage"] is None
