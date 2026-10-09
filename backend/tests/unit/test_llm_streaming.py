"""
测试 OpenAICompatibleLLM 的 SSE 流式传输与聚合行为

作者: xxx
创建时间: 2026-09-12 12:30:00
"""
import json
from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest
import requests
from urllib3.exceptions import ProtocolError

from app.services.llm_service import EMPTY_RESPONSE_ERROR_PREFIX, OpenAICompatibleLLM
from app.config import Settings


class _FakeSSEResponse:
    """模拟 requests 的 SSE 响应：按字节流产出预置行，可指定内容类型与分片大小。"""

    def __init__(self, lines, content_type="text/event-stream", chunk_split=None):
        self.status_code = 200
        self.headers = {"Content-Type": content_type}
        # 1. 预置行统一按 UTF-8 编码并补行尾换行，模拟真实 SSE 字节流
        blob = "".join(line + "\n" for line in lines).encode("utf-8")
        if chunk_split:
            self._chunks = [blob[i:i + chunk_split] for i in range(0, len(blob), chunk_split)]
        else:
            self._chunks = [blob]

    def raise_for_status(self):
        return None

    def close(self):
        """无参数；记录流式测试替身允许传输所有者关闭响应。"""
        return None

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)


def _sse_chunk(content_piece, usage=None, error=None):
    """构造一行 OpenAI 风格的 SSE data chunk。"""
    payload = {
        "id": "chunk-x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content_piece}, "finish_reason": None}],
    }
    if usage is not None:
        payload["usage"] = usage
    if error is not None:
        payload["error"] = error
    return "data: " + json.dumps(payload, ensure_ascii=False)


def _client():
    """创建测试用 LLM 客户端（禁用缓存与限流）。"""
    return OpenAICompatibleLLM(
        model="test-model",
        api_key="test-key",
        base_url="https://api.example.com",
        temperature=0.0,
        timeout=30,
        min_request_interval_seconds=0.0,
    )


def _streaming_settings(enabled=True):
    """参数为流式开关；返回完整配置合同并明确关闭测试的共享数据库队列。"""
    return Settings(_env_file=None, llm_streaming_enabled=enabled, llm_queue_enabled=False)


def _interrupted_sse_chunks(error: ProtocolError) -> Iterator[bytes]:
    """参数为底层协议异常；产出部分正文后抛出该异常，交给 Requests 转换。"""
    yield (_sse_chunk("partial") + "\n\n").encode("utf-8")
    raise error


def _interrupted_sse_response(error: ProtocolError) -> requests.Response:
    """参数为底层协议异常；返回使用真实 iter_content 的离线断流响应。"""
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/event-stream"
    response.raw = Mock()
    response.raw.stream.return_value = _interrupted_sse_chunks(error)
    return response


class TestLLMStreamingAggregation:
    """流式请求体、聚合结果与非流式的等价性"""

    def test_stream_payload_and_aggregated_content(self):
        """
        测试场景：开关开启时发送 stream:true，多个 delta.content 聚合为完整文本
        预期：1. 请求体带 "stream": true
              2. content 与手工拼接各片段一致
        """
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None, stream=False):
            captured["json"] = json
            captured["stream"] = stream
            return _FakeSSEResponse([
                _sse_chunk("Hel"),
                _sse_chunk("lo "),
                _sse_chunk("world"),
                "data: [DONE]",
            ])

        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            with patch("app.services.llm_service.requests.post", side_effect=fake_post):
                result = _client().invoke([{"role": "user", "content": "hi"}], use_cache=False)

        assert captured["json"]["stream"] is True
        assert captured["stream"] is True
        assert result.content == "Hello world"

    def test_usage_chunk_absorbed_into_raw(self):
        """
        测试场景：流末尾带 usage chunk（OpenAI stream_options 常见形态）
        预期：聚合结果的 usage 进入 raw，content 不受影响
        """
        usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        response = _FakeSSEResponse([
            _sse_chunk("OK"),
            _sse_chunk("", usage=usage),
            "data: [DONE]",
        ])
        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            result, _facts = _client()._build_response(response)

        assert result.content == "OK"
        assert result.raw["usage"] == usage

    def test_reasoning_delta_ignored_content_only(self):
        """
        测试场景：推理型模型 chunk 携带 delta.reasoning，content 为空
        预期：只聚合 delta.content，reasoning 不混入答案文本
        """
        lines = [
            "data: " + json.dumps({
                "choices": [{"index": 0, "delta": {"content": "", "reasoning": "thinking"}}],
            }, ensure_ascii=False),
            _sse_chunk("answer"),
            "data: [DONE]",
        ]
        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            result, _facts = _client()._build_response(_FakeSSEResponse(lines))

        assert result.content == "answer"

    def test_response_format_survives_in_stream_payload(self):
        """
        测试场景：流式 + response_format(json_object) 组合
        预期：response_format 原样保留在流式请求体中
        """
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None, stream=False):
            captured["json"] = json
            return _FakeSSEResponse([_sse_chunk('{"ok": true}'), "data: [DONE]"])

        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            with patch("app.services.llm_service.requests.post", side_effect=fake_post):
                _client().invoke(
                    [{"role": "user", "content": "hi"}],
                    use_cache=False,
                    response_format={"type": "json_object"},
                )

        assert captured["json"]["response_format"] == {"type": "json_object"}
        assert captured["json"]["stream"] is True


class TestLLMStreamingFailureBranches:
    """流中断、SSE 错误事件、空流的显式暴露"""

    def test_chunked_encoding_error_retries_and_caches_only_complete_response(self):
        """无参数；协议断流重试后只返回并缓存完整正文，同时关闭失败响应。"""
        bad = _interrupted_sse_response(ProtocolError("Response ended prematurely"))
        good = _FakeSSEResponse([_sse_chunk("recovered"), "data: [DONE]"])
        cache = Mock()
        cache.get_llm_response.return_value = None
        with (
            patch("app.config.get_settings", return_value=_streaming_settings()),
            patch("app.services.llm_service.requests.post", side_effect=[bad, good]) as post,
            patch("app.services.llm_service.get_cache", return_value=cache),
            patch("app.services.llm_service.wait_for_provider_retry") as wait,
        ):
            result = _client().invoke("hi")

        assert result.content == "recovered"
        assert post.call_count == 2
        wait.assert_called_once()
        cache.set_llm_response.assert_called_once()
        assert cache.set_llm_response.call_args.args[1] == "recovered"
        bad.raw.close.assert_called_once()
        bad.raw.release_conn.assert_called_once()

    @pytest.mark.parametrize("retry_budget", [0, 2])
    def test_chunked_encoding_error_exhausts_request_budget(self, retry_budget):
        """参数为原HTTP重试额度；耗尽后保留最终协议异常，不缓存半截正文。"""
        errors = [ProtocolError("Response ended prematurely") for _ in range(retry_budget + 1)]
        responses = [_interrupted_sse_response(error) for error in errors]
        cache = Mock()
        client = _client()
        client.request_exception_max_retries = retry_budget
        with (
            patch("app.config.get_settings", return_value=_streaming_settings()),
            patch("app.services.llm_service.requests.post", side_effect=responses) as post,
            patch("app.services.llm_service.get_cache", return_value=cache),
            patch("app.services.llm_service.wait_for_provider_retry") as wait,
            pytest.raises(requests.exceptions.ChunkedEncodingError) as caught,
        ):
            client.invoke("hi", use_cache=False)

        assert post.call_count == retry_budget + 1
        assert wait.call_count == retry_budget
        assert caught.value.args[0] is errors[-1]
        assert caught.value.__context__ is errors[-1]
        cache.set_llm_response.assert_not_called()
        for response in responses:
            response.raw.close.assert_called_once()
            response.raw.release_conn.assert_called_once()

    def test_stream_without_done_raises_connection_error(self):
        """
        测试场景：流在 [DONE] 之前断开（即使已聚合到部分内容）
        预期：抛 ConnectionError（瞬时连接故障），走既有重试，不静默返回半截内容
        """
        response = _FakeSSEResponse([_sse_chunk("partial")])
        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
                _client()._build_response(response)

        assert "[DONE]" in str(exc_info.value)

    def test_sse_error_event_raises_connection_error(self):
        """
        测试场景：SSE 事件中携带 error 字段（上游流被中断）
        预期：抛 ConnectionError（瞬时连接故障），走既有重试，错误信息可读
        """
        response = _FakeSSEResponse([
            _sse_chunk("", error={"message": "upstream overloaded"}),
        ])
        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
                _client()._build_response(response)

        assert "upstream overloaded" in str(exc_info.value)

    def test_sse_error_event_retried_until_success(self):
        """
        测试场景：上游流被中断后重试（线上实测中转站常见瞬时错误）
        预期：invoke 循环捕获 ConnectionError 重试，第二次成功返回完整内容
        """
        good = _FakeSSEResponse([_sse_chunk("recovered"), "data: [DONE]"])
        bad = _FakeSSEResponse([
            _sse_chunk("", error={"code": "upstream_stream_read_error", "message": "interrupted"}),
        ])
        call_count = 0

        def fake_post(url, headers=None, json=None, timeout=None, stream=False):
            nonlocal call_count
            call_count += 1
            return bad if call_count == 1 else good

        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            with patch("app.services.llm_service.requests.post", side_effect=fake_post):
                with patch("app.services.llm_service.time.sleep"):
                    result = _client().invoke([{"role": "user", "content": "hi"}], use_cache=False)

        assert result.content == "recovered"
        assert call_count == 2

    def test_multibyte_content_survives_chunk_split(self):
        """
        测试场景：多字节中文字符被网络分片劈开（线上实测乱码根因）
        预期：按字节缓冲切行后整行 UTF-8 解码，content 完整无乱码
        """
        chinese = "对比实验与消融分析"
        response = _FakeSSEResponse([
            _sse_chunk(chinese),
            "data: [DONE]",
        ], chunk_split=3)
        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            result, _facts = _client()._build_response(response)

        assert result.content == chinese

    def test_empty_stream_reuses_empty_response_prefix(self):
        """
        测试场景：流正常结束但没有任何 content
        预期：抛既有空响应前缀错误，沿用空响应重试语义
        """
        response = _FakeSSEResponse(["data: [DONE]"])
        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            with pytest.raises(ValueError) as exc_info:
                _client()._build_response(response)

        assert str(exc_info.value).startswith(EMPTY_RESPONSE_ERROR_PREFIX)


class TestLLMStreamingSwitch:
    """开关关闭与非 SSE 响应的回退路径"""

    def test_switch_off_keeps_non_stream_payload(self):
        """
        测试场景：llm_streaming_enabled=false
        预期：请求体不带 stream 字段，走非流式解析
        """
        captured = {}
        mock_resp = Mock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({"choices": [{"message": {"content": "plain"}}]})
        mock_resp.json.return_value = {"choices": [{"message": {"content": "plain"}}]}

        def fake_post(url, headers=None, json=None, timeout=None, stream=False):
            captured["json"] = json
            return mock_resp

        with patch("app.config.get_settings", return_value=_streaming_settings(False)):
            with patch("app.services.llm_service.requests.post", side_effect=fake_post):
                result = _client().invoke([{"role": "user", "content": "hi"}], use_cache=False)

        assert "stream" not in captured["json"]
        assert result.content == "plain"

    def test_non_sse_content_type_parsed_as_json(self):
        """
        测试场景：请求带了 stream:true 但上游返回普通 JSON（不支持流式）
        预期：按非流式响应体解析，同一份响应体两种解法
        """
        body = {"choices": [{"message": {"content": "fallback"}}]}
        response = Mock()
        response.headers = {"Content-Type": "application/json"}
        response.status_code = 200
        response.text = json.dumps(body)
        response.json.return_value = body

        with patch("app.config.get_settings", return_value=_streaming_settings(True)):
            result, _facts = _client()._build_response(response)

        assert result.content == "fallback"
