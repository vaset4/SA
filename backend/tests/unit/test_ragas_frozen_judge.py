"""项目 judge 传输审计、敏感信息与缓存隔离回归。作者：xxx。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from langchain_core.messages import HumanMessage

from app.middleware.trace_context import TraceContext, get_trace_ctx, reset_trace_ctx, set_trace_ctx
from app.services.cache import get_cache
from app.services.llm_service import OpenAICompatibleLLM
from evals.ragas.frozen_judge import _AuditedProjectLLM, _sanitize


def _settings() -> SimpleNamespace:
    """无参数；返回不访问网络的专用 judge 测试配置。"""
    return SimpleNamespace(llm_model_name="fixture-judge", llm_api_key="fixture-secret-key",
                           llm_base_url="https://fixture.invalid/v1", llm_timeout=5,
                           ragas_llm_min_request_interval_seconds=0.0)


def _response(headers: dict[str, str], payload: dict[str, Any]) -> requests.Response:
    """参数为实际请求头与正文；返回带真实结构 usage 的 HTTP 测试响应。"""
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "application/json"
    body = {"choices": [{"message": {"content": '{"verdict":1}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}}
    response._content = json.dumps(body).encode()
    return response


@pytest.mark.asyncio
async def test_judge_captures_request_response_usage_and_cleans_only_own_cache(monkeypatch: Any) -> None:
    """审计保存真实请求/响应；judge 唯一缓存键不会覆盖产品同提示词缓存。"""
    monkeypatch.setattr(OpenAICompatibleLLM, "_streaming_enabled", lambda self: False)
    monkeypatch.setattr(OpenAICompatibleLLM, "_post_chat_completions", lambda self, headers, payload: _response(headers, payload))
    settings = _settings()
    llm = _AuditedProjectLLM(settings)
    client = llm._build_client(temperature=0.0)
    monkeypatch.setattr(llm, "_build_client", lambda **options: client)
    product = OpenAICompatibleLLM(model=settings.llm_model_name, api_key=settings.llm_api_key,
                                 base_url=settings.llm_base_url, temperature=0.0, timeout=5)
    messages = [{"role": "user", "content": "same prompt"}]
    product_key = product._cache_key(messages)
    get_cache().set_llm_response(product_key, "existing product response")
    output = await llm._ainvoke_once(payload=[HumanMessage(content="same prompt")], temperature=0.0)
    assert output == '{"verdict":1}'
    audit = llm.calls[0]
    assert audit["requests"] == [{"model": settings.llm_model_name, "messages": messages, "temperature": 0.0}]
    assert audit["response"]["usage"]["total_tokens"] == 16
    assert audit["usage_status"] == "measured"
    assert get_cache().get_llm_response(client.judge_cache_key) is None
    assert get_cache().get_llm_response(product_key) == "existing product response"
    get_cache().delete_llm_response(product_key)
    assert settings.llm_api_key not in json.dumps(audit)


def test_sanitizer_preserves_content_but_removes_nested_credentials() -> None:
    """审计边界保留正文和用量，去掉敏感键及异常字符串里的配置秘密。"""
    raw = {"headers": {"Authorization": "Bearer fixture-secret"}, "output": "真实判定",
           "response": {"api_key": "fixture-secret", "usage": {"total_tokens": 3}},
           "error": "fixture-secret at https://fixture.invalid/v1", "base_url": "hidden"}
    cleaned = _sanitize(raw, secrets=("fixture-secret", "https://fixture.invalid/v1"))
    assert cleaned == {"output": "真实判定", "response": {"usage": {"total_tokens": 3}},
                       "error": "[redacted] at [redacted]"}
    assert raw["response"]["api_key"] == "fixture-secret"


@pytest.mark.asyncio
async def test_product_trace_can_be_restored_after_judge_scope(monkeypatch: Any) -> None:
    """不同 trace 的实际传输只写 judge，退出后恢复产品 trace。"""
    monkeypatch.setattr(OpenAICompatibleLLM, "_streaming_enabled", lambda self: False)
    monkeypatch.setattr(OpenAICompatibleLLM, "_post_chat_completions", lambda self, headers, payload: _response(headers, payload))
    product_trace = TraceContext(request_id="product")
    outer = set_trace_ctx(product_trace)
    judge_trace = TraceContext(request_id="judge")
    inner = set_trace_ctx(judge_trace)
    try:
        await _AuditedProjectLLM(_settings())._ainvoke_once(payload="prompt", temperature=0.0)
    finally:
        reset_trace_ctx(inner)
    assert get_trace_ctx() is product_trace
    assert product_trace.model_calls == []
    assert len(judge_trace.model_calls) == 1
    reset_trace_ctx(outer)
