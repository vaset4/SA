from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests

import app.services.llm_service as llm_service
from app.services.llm_service import OpenAICompatibleLLM


def _response(status_code: int, body: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    response.url = "https://example.com/v1/chat/completions"
    return response


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = _response(status_code, {"detail": f"status-{status_code}"})
    return requests.exceptions.HTTPError(f"{status_code} error", response=response)


class _DummyCache:
    def get_llm_response(self, _key: str):
        return None

    def set_llm_response(self, _key: str, _value: str) -> None:
        return None

    def invalidate_llm_response(self, _key: str) -> None:
        return None


def _client() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        model="test-model",
        api_key="test-key",
        base_url="https://example.com/v1",
        temperature=0.0,
        timeout=30,
    )


def test_llm_invoke_retries_http_500(monkeypatch) -> None:
    client = _client()
    calls: list[int] = []

    def fake_post(_headers, _payload):  # noqa: ANN001
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(500)
        return _response(
            200,
            {"choices": [{"message": {"content": "retry ok"}}]},
        )

    monkeypatch.setattr(client, "_post_chat_completions", fake_post)
    monkeypatch.setattr(llm_service, "get_cache", lambda: _DummyCache())
    monkeypatch.setattr(llm_service.time, "sleep", lambda _seconds: None)

    result = client.invoke([{"role": "user", "content": "hello"}], use_cache=False)

    assert result.content == "retry ok"
    assert len(calls) == 2


def test_llm_invoke_does_not_retry_http_403(monkeypatch) -> None:
    client = _client()
    calls: list[int] = []

    def fake_post(_headers, _payload):  # noqa: ANN001
        calls.append(1)
        raise _http_error(403)

    monkeypatch.setattr(client, "_post_chat_completions", fake_post)
    monkeypatch.setattr(llm_service, "get_cache", lambda: _DummyCache())
    monkeypatch.setattr(llm_service.time, "sleep", lambda _seconds: None)

    with pytest.raises(requests.exceptions.HTTPError):
        client.invoke([{"role": "user", "content": "hello"}], use_cache=False)

    assert len(calls) == 1
