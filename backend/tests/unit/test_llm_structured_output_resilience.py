"""【模型合同】【调用预算】验证原生格式失败与一次严格回修。作者：xxx。"""

from __future__ import annotations

from typing import Any

import pytest
import requests
from pydantic import BaseModel, ConfigDict, Field

from app.services.llm_service import (
    LLMResponse,
    StructuredOutputError,
    StructuredOutputLLM,
    _is_native_format_rejected,
)


class TinyDraft(BaseModel):
    """最小 schema，仅用于韧性路径单测。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1)
    score: int = 1


class _StubClient:
    """可控的 LLM 客户端替身：按调用序号返回预设响应或抛错。"""

    def __init__(self, behaviors: list[Any]) -> None:
        self.behaviors = list(behaviors)
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        messages: Any,
        *,
        use_cache: bool = True,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "use_cache": use_cache,
                "response_format": response_format,
            }
        )
        if not self.behaviors:
            raise AssertionError("stub client exhausted")
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        if isinstance(behavior, LLMResponse):
            return behavior
        return LLMResponse(content=str(behavior), raw={})

    def invalidate_cached_response(self, messages: Any, **kwargs: Any) -> None:
        return None


def _http_error(status: int, body: str) -> requests.exceptions.HTTPError:
    response = requests.models.Response()
    response.status_code = status
    response._content = body.encode("utf-8")
    return requests.exceptions.HTTPError(body, response=response)


def test_native_format_rejected_marker_detection() -> None:
    assert _is_native_format_rejected(_http_error(400, "response_format json_schema unsupported"))
    assert not _is_native_format_rejected(_http_error(429, "rate limit"))
    assert not _is_native_format_rejected(_http_error(500, "internal"))


def test_structured_output_native_reject_raises_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_native_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_repair_max",
        lambda: 1,
    )
    client = _StubClient(
        [
            _http_error(400, "response_format is not supported"),
            '{"answer":"ok","score":2}',
        ]
    )
    with pytest.raises(requests.HTTPError):
        StructuredOutputLLM(client, TinyDraft).invoke("give json")
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"] is not None


def test_structured_output_repair_succeeds_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_native_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_repair_max",
        lambda: 1,
    )
    client = _StubClient(
        [
            "not-json-at-all",
            '{"answer":"repaired","score":3}',
        ]
    )
    result = StructuredOutputLLM(client, TinyDraft).invoke("give json")
    assert result.answer == "repaired"
    assert len(client.calls) == 2
    assert client.calls[0]["use_cache"] is True
    assert client.calls[1]["use_cache"] is False


def test_structured_output_repair_exhausted_raises_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_native_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_repair_max",
        lambda: 1,
    )
    client = _StubClient(
        [
            "first-bad",
            "second-still-bad",
            "should-not-be-called",
        ]
    )
    with pytest.raises(StructuredOutputError) as exc_info:
        StructuredOutputLLM(client, TinyDraft).invoke("give json")
    error = exc_info.value
    assert error.code == "structured_output_failed"
    assert error.details.get("repair_attempted") is True
    assert error.details.get("repair_count") == 1
    assert error.details.get("repair_succeeded") is False
    assert len(client.calls) == 2


def test_structured_output_repair_max_zero_no_extra_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_native_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "app.services.llm_service._structured_output_repair_max",
        lambda: 0,
    )
    client = _StubClient(["still-not-json", "should-not-be-called"])
    with pytest.raises(StructuredOutputError):
        StructuredOutputLLM(client, TinyDraft).invoke("give json")
    assert len(client.calls) == 1
