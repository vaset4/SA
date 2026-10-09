"""【模型服务】【供应商失败】验证安全说明、真实状态及原始异常链。作者：xxx。"""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from app.services.llm_service import OpenAICompatibleLLM, classify_llm_upstream_error


PROVIDER_REASON = "No endpoints found for deepseek-v4.1-flash"
UNKNOWN_404 = "模型服务返回404，具体原因未明确"
API_KEY = "fixture-provider-secret-12345"
LONG_MESSAGE_SIZE = 2000
MAX_SUMMARY_SIZE = 300
TRUNCATION_BOUNDARY_PREFIX_CHARS = 205


@pytest.fixture
def client():
    """无参数；返回禁用自动HTTP重试的测试客户端，外部请求由测试替换。"""
    return OpenAICompatibleLLM(
        model="fixture-model", api_key=API_KEY, base_url="https://provider.example/v1",
        temperature=0, timeout=30, request_exception_max_retries=0,
    )


def error_response(body, *, status=404):
    """参数为供应商原始正文和状态；返回已缓冲的真实 requests 响应对象。"""
    response = requests.Response()
    response.status_code = status
    response.url = "https://provider.example/v1/chat/completions"
    response.encoding = "utf-8"
    response._content = body.encode("utf-8")
    response._content_consumed = True
    response.request = requests.Request("POST", response.url).prepare()
    return response


def decorated_error(client, response):
    """参数为客户端和HTTP错误响应；返回客户端装饰结果，不发生网络调用。"""
    original = requests.HTTPError("upstream HTTP failure", response=response)
    return client._decorate_http_error(original, response.url)


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_404_keeps_safe_reason_and_original_error(client, streaming):
    """参数为客户端与传输模式；验证两个正式出口只请求一次并保留供应商事实。"""
    response = error_response(json.dumps({"error": {
        "message": PROVIDER_REASON, "type": "upstream_error", "code": "route_unavailable",
        "routing": "Filter by Fallback: 0; rate candidates: 0",
    }}))
    with patch.object(client, "_streaming_enabled", return_value=streaming), \
            patch("app.services.llm_service.requests.post", return_value=response) as post:
        with pytest.raises(requests.HTTPError) as caught:
            client.invoke("test", use_cache=False)
    failure = caught.value
    expected = f"模型服务返回404。服务商说明：{PROVIDER_REASON}"
    assert classify_llm_upstream_error(failure) == (502, expected)
    assert failure.user_summary == expected
    assert failure.provider_error_type == "upstream_error"
    assert failure.provider_error_code == "route_unavailable"
    assert failure.response is response
    assert failure.request is response.request
    assert isinstance(failure.__cause__, requests.HTTPError)
    assert failure.__cause__.response is response
    assert "route_unavailable" in failure.technical_message
    assert "requested_url=" not in failure.user_summary
    assert "routing" not in failure.user_summary
    assert "endpoint not found" not in str(failure)
    assert post.call_count == 1
    assert post.call_args.kwargs.get("stream", False) is streaming


@pytest.mark.parametrize("body", [
    "", "<html><body>rate page missing</body></html>", "{broken json",
    "null", "[]", '{"message":"untrusted top-level reason"}',
    '{"error":"string error"}', '{"error":{"message":42}}',
    '{"error":{"message":"   "}}', '{"error":{"message":{"secret":"x"}}}',
])
def test_missing_structured_reason_is_explicitly_unknown(client, body):
    """参数为客户端和非标准错误正文；验证不猜原因也不把HTML或整个JSON显示给用户。"""
    failure = decorated_error(client, error_response(body))
    assert classify_llm_upstream_error(failure) == (502, UNKNOWN_404)
    assert failure.user_summary == UNKNOWN_404
    assert failure.response.text == body


@pytest.mark.parametrize("message", [
    "rate", "ResourceExhausted", "timeout occurred", "https://rate.example/timeout occurred",
])
def test_plain_404_ignores_long_error_text(message):
    """参数为普通HTTPError的误导文本；验证无客户端安全摘要时仅消费真实404状态。"""
    response = error_response('{"error":{"message":"not trusted by classifier"}}')
    failure = requests.HTTPError(message, response=response)
    assert classify_llm_upstream_error(failure) == (502, UNKNOWN_404)


def test_provider_summary_redacts_before_truncation_and_renders_literal_text(client, caplog):
    """参数为客户端和日志采集器；验证秘密、URL认证和富文本不会进入用户说明或日志。"""
    prefix = "x" * TRUNCATION_BOUNDARY_PREFIX_CHARS
    body = json.dumps({"error": {"message": (
        f"{prefix}{API_KEY} Authorization: Bearer stolen-token "
        "api_key=other-key https://alice:password@provider.example/path?token=query-secret "
        '<script>alert("x")</script> [open](https://evil.example/steal)'
    ), "type": "upstream_error", "code": "unavailable"}})
    failure = decorated_error(client, error_response(body))
    for value in (failure.user_summary, str(failure), caplog.text):
        assert API_KEY not in value
        assert API_KEY[:10] not in value
        assert "stolen-token" not in value
        assert "other-key" not in value
        assert "alice:password" not in value
        assert "query-secret" not in value
    assert len(failure.user_summary) <= MAX_SUMMARY_SIZE
    assert failure.response.text == body


def test_provider_special_characters_cannot_create_html_or_markdown(client):
    """参数为客户端；验证标准message中的特殊字符只能作为文本呈现，返回空。"""
    from markdown_it import MarkdownIt

    message = '<img src=x onerror="alert(1)"> [click](https://evil.example/path) `code` & 中文'
    failure = decorated_error(client, error_response(json.dumps({"error": {"message": message}})))
    rendered = MarkdownIt("commonmark").render(failure.user_summary)
    assert "<img" not in rendered
    assert "<a " not in rendered
    assert "<code>" not in rendered
    assert "中文" in failure.user_summary
    assert "&lt;img" in rendered


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_json_escaped_configured_key_is_redacted_everywhere(client, caplog, ensure_ascii):
    """参数为客户端、日志与JSON编码模式；验证含引号、反斜杠及中文的配置凭证也能完整脱敏。"""
    client.api_key = 'private-credential-"quoted\\secret\n凭证'
    body = json.dumps({"error": {"message": f"rejected {client.api_key} end"}}, ensure_ascii=ensure_ascii)
    failure = decorated_error(client, error_response(body))
    for value in (failure.user_summary, str(failure), caplog.text):
        assert "private-credential-" not in value
        assert "quoted" not in value
        assert "secret" not in value
        assert "凭证" not in value
    assert failure.response.text == body


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_http_observation_redacts_json_escaped_key(ensure_ascii):
    """参数为JSON编码模式；验证保留原response后，观察事件再次读取正文也不会暴露转义凭证。"""
    from app.services.llm_observation import http_error_facts

    secret = 'fixture-"quoted\\secret\n凭证'
    body = json.dumps({"error": {"message": f"rejected {secret} end"}}, ensure_ascii=ensure_ascii)
    response = error_response(body)
    failure = requests.HTTPError("upstream HTTP failure", response=response)
    facts = http_error_facts(failure, boundary="http_status", secret=secret)
    assert "fixture-" not in facts["response_summary"]
    assert "quoted" not in facts["response_summary"]
    assert "secret" not in facts["response_summary"]
    assert response.text == body


def test_control_characters_are_not_preserved_in_display(client):
    """参数为客户端；验证控制字符无法伪造日志行或用户说明，保留可读原因。"""
    message = "provider\x00\x1b[31m rejected\r\nfor\tunknown\u202ereason"
    failure = decorated_error(client, error_response(json.dumps({"error": {"message": message}})))
    for character in ("\x00", "\x1b", "\r", "\n", "\t", "\u202e"):
        assert character not in failure.user_summary
    assert "provider" in failure.user_summary


def test_long_standard_reason_is_bounded_without_affecting_classification(client):
    """参数为客户端；验证长供应商说明只截显示摘要，原response与404分类不变。"""
    message = "rate " + "长" * LONG_MESSAGE_SIZE
    failure = decorated_error(client, error_response(json.dumps({"error": {"message": message}})))
    assert len(failure.user_summary) <= MAX_SUMMARY_SIZE
    assert classify_llm_upstream_error(failure) == (502, failure.user_summary)
    assert failure.response.json()["error"]["message"] == message


@pytest.mark.parametrize("status, expected", [
    (401, (502, "上游模型服务鉴权失败，请检查 API Key 配置")),
    (403, None),
    (429, (503, "上游模型服务当前繁忙（额度或并发已满），请稍后重试")),
    (503, (503, "上游模型服务暂时不可用，请稍后重试")),
    (524, (504, "上游模型服务处理超时，请简化问题或稍后重试")),
])
@pytest.mark.parametrize("streaming", [False, True])
def test_other_status_policies_keep_mapping_and_attempt_count(client, status, expected, streaming):
    """参数为客户端、状态、旧映射和传输模式；验证既有HTTP政策及单次调用额度不变。"""
    response = error_response('{"error":{"message":"provider rejected request"}}', status=status)
    queue = Mock()
    queue.policy = {"enabled": False}
    with patch.object(client, "_streaming_enabled", return_value=streaming), \
            patch("app.services.llm_service.requests.post", return_value=response) as post, \
            patch("app.services.llm_service.get_provider_queue", return_value=queue):
        with pytest.raises(requests.HTTPError) as caught:
            client.invoke("test", use_cache=False)
    assert classify_llm_upstream_error(caught.value) == expected
    assert caught.value.response is response
    assert post.call_count == 1
    assert queue.cooldown.call_count == (1 if status == 429 else 0)


@pytest.mark.parametrize("failure, expected_status", [
    (requests.Timeout("slow"), 504), (requests.ConnectionError("offline"), 502),
])
def test_transport_classification_is_unchanged(failure, expected_status):
    """参数为传输异常和旧HTTP状态；验证HTTP说明修复不扩展传输错误政策，返回空。"""
    assert classify_llm_upstream_error(failure)[0] == expected_status
