from __future__ import annotations

from typing import Any

import pytest
import requests

from pydantic import BaseModel

import app.services.cache as cache_module
from app.models.structured_outputs import AnswerEvaluation, Citation, CitedAnswer
from app.services.llm_service import (
    OpenAICompatibleLLM,
    build_structured_output_prompt,
    parse_structured_response,
)


class FakeResponse:
    def __init__(
        self,
        *,
        text: str = "",
        content: bytes = b"",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        json_data: Any = None,
        json_error: Exception | None = None,
    ) -> None:
        self.text = text
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self._json_data = json_data
        self._json_error = json_error

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        """无参数；记录本地响应已释放，对齐requests响应关闭合同。"""
        self.closed = True

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


@pytest.fixture(autouse=True)
def isolate_llm_response_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替换器；每项根入口测试独享真实内存缓存，避免前项成功正文污染后项重试断言。"""
    monkeypatch.setattr(cache_module, "_cache", None)


def _build_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        model="test-model",
        api_key="test-key",
        base_url="http://test-host/v1",
        temperature=0.0,
        timeout=30,
    )


def _build_connection_error() -> requests.exceptions.ConnectionError:
    return requests.exceptions.ConnectionError(
        ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接。")
    )


class ExampleSchema(BaseModel):
    value: int


def test_invoke_retries_once_when_body_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.llm_service.time.sleep", lambda *_args, **_kwargs: None)
    responses = [
        FakeResponse(headers={"Content-Length": "0"}),
        FakeResponse(
            text='{"choices":[{"message":{"content":"ok"}}]}',
            content=b'{"choices":[{"message":{"content":"ok"}}]}',
            headers={"Content-Type": "application/json"},
            json_data={"choices": [{"message": {"content": "ok"}}]},
        ),
    ]
    call_count = {"value": 0}

    def _fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        call_count["value"] += 1
        return responses.pop(0)

    monkeypatch.setattr("app.services.llm_service.requests.post", _fake_post)

    llm = _build_llm()
    result = llm.invoke("hello")

    assert result.content == "ok"
    assert call_count["value"] == 2


def test_invoke_raises_after_retry_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.llm_service.time.sleep", lambda *_args, **_kwargs: None)
    responses = [
        FakeResponse(headers={"Content-Length": "0"}),
        FakeResponse(headers={"Content-Length": "0"}),
    ]
    call_count = {"value": 0}

    def _fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        call_count["value"] += 1
        return responses.pop(0)

    monkeypatch.setattr("app.services.llm_service.requests.post", _fake_post)

    llm = _build_llm()
    with pytest.raises(ValueError, match="LLM 返回空响应体"):
        llm.invoke("hello")
    assert call_count["value"] == 2


def test_build_response_reads_utf8_bytes_when_text_empty() -> None:
    """无参数；验证私有解析返回快照后仍从UTF-8字节恢复原正文。"""
    llm = _build_llm()
    response = FakeResponse(
        text="",
        content=b'{"content":"from-bytes"}',
        headers={"Content-Type": "application/json"},
        json_data={"content": "from-bytes"},
    )

    parsed, _facts = llm._build_response(response)
    assert parsed.content == "from-bytes"


def test_invoke_retries_once_when_connection_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.llm_service.time.sleep", lambda *_args, **_kwargs: None)
    responses = [
        _build_connection_error(),
        FakeResponse(
            text='{"choices":[{"message":{"content":"ok"}}]}',
            content=b'{"choices":[{"message":{"content":"ok"}}]}',
            headers={"Content-Type": "application/json"},
            json_data={"choices": [{"message": {"content": "ok"}}]},
        ),
    ]
    call_count = {"value": 0}

    def _fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        call_count["value"] += 1
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("app.services.llm_service.requests.post", _fake_post)

    llm = _build_llm()
    result = llm.invoke("hello")

    assert result.content == "ok"
    assert call_count["value"] == 2


def test_invoke_throttles_requests_per_host(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    call_count = {"value": 0}
    now = {"value": 100.0}

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["value"] += seconds

    def _fake_monotonic() -> float:
        return now["value"]

    def _fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        call_count["value"] += 1
        return FakeResponse(
            text='{"choices":[{"message":{"content":"ok"}}]}',
            content=b'{"choices":[{"message":{"content":"ok"}}]}',
            headers={"Content-Type": "application/json"},
            json_data={"choices": [{"message": {"content": "ok"}}]},
        )

    monkeypatch.setattr("app.services.llm_service.time.sleep", _fake_sleep)
    monkeypatch.setattr("app.services.llm_service.time.monotonic", _fake_monotonic)
    monkeypatch.setattr("app.services.llm_service.requests.post", _fake_post)
    OpenAICompatibleLLM._host_next_request_at.clear()

    llm = OpenAICompatibleLLM(
        model="test-model",
        api_key="test-key",
        base_url="http://test-host/v1",
        temperature=0.0,
        timeout=30,
        min_request_interval_seconds=2.2,
    )

    first = llm.invoke("hello-1", use_cache=False)
    second = llm.invoke("hello-2", use_cache=False)

    assert first.content == "ok"
    assert second.content == "ok"
    assert call_count["value"] == 2
    assert sleeps == [pytest.approx(2.2)]


def test_invoke_raises_after_connection_retry_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.llm_service.time.sleep", lambda *_args, **_kwargs: None)
    call_count = {"value": 0}

    def _fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        call_count["value"] += 1
        raise _build_connection_error()

    monkeypatch.setattr("app.services.llm_service.requests.post", _fake_post)

    llm = _build_llm()
    with pytest.raises(requests.exceptions.ConnectionError):
        llm.invoke("hello")
    assert call_count["value"] == 6


def test_structured_output_error_contains_schema_and_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse(
        text="milvus\ngraph_mode: local",
        content=b"milvus\ngraph_mode: local",
        headers={"Content-Type": "text/plain"},
        json_error=ValueError("not json"),
    )

    monkeypatch.setattr("app.services.llm_service.requests.post", lambda *args, **kwargs: response)

    llm = _build_llm()
    with pytest.raises(ValueError, match="schema=ExampleSchema") as exc_info:
        llm.with_structured_output(ExampleSchema).invoke("hello")

    assert "response_preview=milvus graph_mode: local" in str(exc_info.value)


def test_build_structured_output_prompt_enforces_json_only() -> None:
    prompt = build_structured_output_prompt("Pick a mode", ExampleSchema)

    assert "Return ONLY one valid JSON object." in prompt
    assert "Do not wrap JSON in markdown fences." in prompt
    assert "Schema: ExampleSchema" in prompt


def test_parse_structured_response_normalizes_cited_answer_payload() -> None:
    parsed = parse_structured_response(
        """
        ```json
        {
          "answer": "Chain-of-Thought 是一种逐步推理提示方法。[1]",
          "citations": ["[1]"],
          "confidence": 0.9,
          "confidence_reasoning": "evidence is direct"
        }
        ```
        """,
        CitedAnswer,
    )

    assert parsed.answer.startswith("Chain-of-Thought")
    assert parsed.confidence == "high"
    assert parsed.citations == []


def test_parse_structured_response_fills_missing_citation_fields() -> None:
    parsed = parse_structured_response(
        """
        {
          "answer": "STGCN 解决了交通预测中的时空依赖建模问题。",
          "citations": [
            {
              "paper_id": "upload_stgcn",
              "section": "Abstract",
              "page_num": "1"
            }
          ],
          "confidence": "medium",
          "confidence_reasoning": "supported by abstract"
        }
        """,
        CitedAnswer,
    )

    assert parsed.citations == [
        Citation(
            paper_id="upload_stgcn",
            chunk_id="",
            section="Abstract",
            page_num=1,
            excerpt="",
        )
    ]


def test_parse_structured_response_normalizes_answer_evaluation_payload() -> None:
    parsed = parse_structured_response(
        """
        {
          "relevance": "high",
          "faithfulness": "medium",
          "completeness": "medium",
          "quality_score": "0.8",
          "grounded": "true",
          "missing_info": "缺少方法细节",
          "follow_up_queries": "这篇论文的方法是什么？",
          "has_hallucination": "false"
        }
        """,
        AnswerEvaluation,
    )

    assert parsed.completeness == "partial"
    assert parsed.quality_score == 0.8
    assert parsed.grounded is True
    assert parsed.missing_info == ["缺少方法细节"]
    assert parsed.follow_up_queries == ["这篇论文的方法是什么？"]
    assert parsed.has_hallucination is False


def test_structured_output_invalidates_bad_cache_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    cache_module._cache = None
    try:
        llm = _build_llm()
        prompt_key = llm._cache_key(llm._normalize_messages("hello"))
        cache_module.get_cache().set_llm_response(prompt_key, "not-json")

        response = FakeResponse(
            text='{"choices":[{"message":{"content":"{\\"value\\": 7}"}}]}',
            content=b'{"choices":[{"message":{"content":"{\\"value\\": 7}"}}]}',
            headers={"Content-Type": "application/json"},
            json_data={"choices": [{"message": {"content": '{"value": 7}'}}]},
        )
        call_count = {"value": 0}

        def _fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
            call_count["value"] += 1
            return response

        monkeypatch.setattr("app.services.llm_service.requests.post", _fake_post)

        parsed = llm.with_structured_output(ExampleSchema).invoke("hello")

        assert parsed.value == 7
        assert call_count["value"] == 1
        assert cache_module.get_cache().get_llm_response(prompt_key) == '{"value": 7}'
    finally:
        cache_module._cache = None
