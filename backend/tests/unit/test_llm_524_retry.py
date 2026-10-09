"""
测试 LLM 524 及同类 Cloudflare 网关错误的重试行为

作者: xxx
创建时间: 2026-06-21 15:30:00
"""
from unittest.mock import Mock, patch

import pytest
import requests

from app.services.llm_service import OpenAICompatibleLLM, classify_llm_upstream_error


class TestLLM524Retry:
    """测试 524 Gateway Timeout 及同类状态码的重试逻辑"""

    @pytest.fixture
    def llm_client(self):
        """创建 LLM 客户端实例（禁用缓存和限流，加速测试）"""
        return OpenAICompatibleLLM(
            model="test-model",
            api_key="test-key",
            base_url="https://api.example.com",
            temperature=0.7,
            timeout=30,
            min_request_interval_seconds=0.0,
        )

    @pytest.fixture
    def mock_success_response(self):
        """模拟成功的 LLM 响应"""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        return mock_resp

    def test_524_triggers_retry_and_succeeds(self, llm_client, mock_success_response):
        """
        测试场景：首次请求返回 524，重试后成功
        预期：1. 发生至少 1 次重试
              2. 最终返回成功响应
        """
        mock_524_response = Mock()
        mock_524_response.status_code = 524
        mock_524_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_524_response
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 1. 首次请求返回 524
                raise requests.exceptions.HTTPError(response=mock_524_response)
            # 2. 重试后成功
            return mock_success_response

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep"):
                messages = [{"role": "user", "content": "test"}]
                result = llm_client.invoke(messages, use_cache=False)

                # 验证：最终成功返回
                assert result.content == "test response"
                # 验证：发生了重试（调用了 2 次 post）
                assert call_count == 2

    def test_524_retry_exhausted_raises_error(self, llm_client):
        """
        测试场景：持续返回 524，重试耗尽
        预期：抛出 HTTPError（不静默）
        """
        mock_524_response = Mock()
        mock_524_response.status_code = 524

        def side_effect(*args, **kwargs):
            raise requests.exceptions.HTTPError(response=mock_524_response)

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep"):
                messages = [{"role": "user", "content": "test"}]

                # 验证：重试耗尽后显式抛错
                with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                    llm_client.invoke(messages, use_cache=False)

                # 验证：错误响应的状态码是 524
                assert exc_info.value.response.status_code == 524

    @pytest.mark.parametrize("status_code", [520, 522, 523, 524])
    def test_cloudflare_gateway_errors_are_retryable(
        self, llm_client, mock_success_response, status_code
    ):
        """
        测试场景：验证 Cloudflare 网关错误（520/522/523/524）均可重试
        预期：首次返回指定状态码，重试后成功
        """
        mock_error_response = Mock()
        mock_error_response.status_code = status_code

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.HTTPError(response=mock_error_response)
            return mock_success_response

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep"):
                messages = [{"role": "user", "content": "test"}]
                result = llm_client.invoke(messages, use_cache=False)

                # 验证：最终成功
                assert result.content == "test response"
                # 验证：发生了重试
                assert call_count == 2

    def test_oversized_retry_after_is_capped(self, llm_client, mock_success_response):
        """
        测试场景：上游 524 响应带超大 Retry-After（120s），单次 sleep 应被配置上限收敛
        预期：1. 实际传给 time.sleep 的延迟不超过硬上限（默认 15s），而非照睡 120s
              2. 收敛后仍能正常重试并成功
        """
        mock_524_response = Mock()
        mock_524_response.status_code = 524
        mock_524_response.headers = {"Retry-After": "120"}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 1. 首次返回带 Retry-After: 120 的 524
                raise requests.exceptions.HTTPError(response=mock_524_response)
            # 2. 重试后成功
            return mock_success_response

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep") as mock_sleep:
                messages = [{"role": "user", "content": "test"}]
                result = llm_client.invoke(messages, use_cache=False)

                # 验证：最终成功
                assert result.content == "test response"
                # 验证：单次 sleep 被封到硬上限（默认 15s），未照搬 120s
                cap = llm_client._retry_delay_cap_seconds()
                assert mock_sleep.call_count >= 1
                for sleep_call in mock_sleep.call_args_list:
                    slept_seconds = sleep_call.args[0]
                    assert slept_seconds <= cap
                    assert slept_seconds < 120.0

    @pytest.mark.asyncio
    async def test_524_retry_works_with_ainvoke(
        self, llm_client, mock_success_response
    ):
        """
        测试场景：验证异步调用路径（ainvoke）也能正确处理 524 重试
        预期：ainvoke 复用 invoke 的重试逻辑，同样能重试成功
        """
        mock_524_response = Mock()
        mock_524_response.status_code = 524

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.HTTPError(response=mock_524_response)
            return mock_success_response

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep"):
                messages = [{"role": "user", "content": "test"}]
                result = await llm_client.ainvoke(messages, use_cache=False)

                # 验证：最终成功
                assert result.content == "test response"
                # 验证：发生了重试
                assert call_count == 2


class TestLLMRetryTuning:
    """测试适配 NVIDIA 免费上游的重试/限流调优行为"""

    @pytest.fixture
    def mock_success_response(self):
        """模拟成功的 LLM 响应"""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        return mock_resp

    def test_rpm_derives_request_interval(self):
        """
        测试场景：仅给 requests_per_minute_limit=30，未给显式间隔
        预期：有效间隔 = 60/30 = 2.0s（RPM 反推），不再是 0
        """
        client = OpenAICompatibleLLM(
            model="m",
            api_key="k",
            base_url="https://api.example.com",
            temperature=0.0,
            timeout=30,
            requests_per_minute_limit=30,
        )
        assert client.min_request_interval_seconds == 2.0

    def test_explicit_interval_wins_when_larger(self):
        """
        测试场景：显式间隔 5s 与 RPM 反推 2s 同时给出
        预期：取较大值 5s，避免突破更严格的显式约束
        """
        client = OpenAICompatibleLLM(
            model="m",
            api_key="k",
            base_url="https://api.example.com",
            temperature=0.0,
            timeout=30,
            min_request_interval_seconds=5.0,
            requests_per_minute_limit=30,
        )
        assert client.min_request_interval_seconds == 5.0

    def test_configurable_retry_count_is_honored(self, mock_success_response):
        """
        测试场景：把 request_exception_max_retries 设为 3，持续 503 到第 3 次才成功
        预期：能重试满 3 次（共 4 次 post 调用）而非写死的 5 次
        """
        client = OpenAICompatibleLLM(
            model="m",
            api_key="k",
            base_url="https://api.example.com",
            temperature=0.0,
            timeout=30,
            request_exception_max_retries=3,
        )
        mock_503 = Mock()
        mock_503.status_code = 503
        mock_503.headers = {}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 前 3 次返回 503，第 4 次成功
            if call_count <= 3:
                raise requests.exceptions.HTTPError(response=mock_503)
            return mock_success_response

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep"):
                result = client.invoke([{"role": "user", "content": "t"}], use_cache=False)
                assert result.content == "test response"
                assert call_count == 4

    def test_503_uses_short_flat_backoff_not_exponential(self, mock_success_response):
        """
        测试场景：503 且无 Retry-After 头，连续失败多次
        预期：每次 sleep 都是固定短退避（不随 attempt 指数增长），
              这是让高发 503 快速重试而非越等越久的关键
        """
        client = OpenAICompatibleLLM(
            model="m",
            api_key="k",
            base_url="https://api.example.com",
            temperature=0.0,
            timeout=30,
            request_exception_max_retries=5,
        )
        mock_503 = Mock()
        mock_503.status_code = 503
        mock_503.headers = {}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise requests.exceptions.HTTPError(response=mock_503)
            return mock_success_response

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep") as mock_sleep:
                client.invoke([{"role": "user", "content": "t"}], use_cache=False)
                slept = [c.args[0] for c in mock_sleep.call_args_list]
                # 所有 sleep 相等且都很短（固定退避），而非 1→2→4→8 指数
                assert len(slept) == 4
                assert len(set(slept)) == 1
                assert slept[0] <= 2.0

    def test_retry_backoff_seconds_constructor_is_honored(self, mock_success_response):
        """
        测试场景：构造时传入 retry_backoff_seconds=3.0，连续 503 无 Retry-After
        预期：每次 sleep 使用 3.0（受 cap 约束后仍为 3.0），证明预设退避能进入运行时
        """
        client = OpenAICompatibleLLM(
            model="m",
            api_key="k",
            base_url="https://api.example.com",
            temperature=0.0,
            timeout=30,
            request_exception_max_retries=3,
            retry_backoff_seconds=3.0,
        )
        mock_503 = Mock()
        mock_503.status_code = 503
        mock_503.headers = {}

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise requests.exceptions.HTTPError(response=mock_503)
            return mock_success_response

        with patch("app.services.llm_service.requests.post", side_effect=side_effect):
            with patch("app.services.llm_service.time.sleep") as mock_sleep:
                client.invoke([{"role": "user", "content": "t"}], use_cache=False)
                slept = [c.args[0] for c in mock_sleep.call_args_list]
                assert slept
                assert all(value == 3.0 for value in slept)


class TestClassifyLlmUpstreamError:
    """上游异常翻译成可读 HTTP 语义，供 chat/research 路由透传"""

    def test_timeout_maps_to_504(self):
        status, reason = classify_llm_upstream_error(requests.exceptions.Timeout())
        assert status == 504
        assert "超时" in reason

    def test_503_http_error_maps_to_busy(self):
        response = Mock()
        response.status_code = 503
        error = requests.exceptions.HTTPError("503 Server Error", response=response)
        status, reason = classify_llm_upstream_error(error)
        assert status == 503
        assert "不可用" in reason or "繁忙" in reason

    def test_unrelated_exception_returns_none(self):
        assert classify_llm_upstream_error(ValueError("not llm")) is None


class TestGetLlmBackoffWiring:
    """get_llm 必须把 settings.llm_retry_backoff_seconds 传入运行时实例"""

    def test_get_llm_passes_retry_backoff_from_settings(self):
        """
        测试场景：settings.llm_retry_backoff_seconds=4.25 时重建单例
        预期：get_llm() 返回实例的 retry_backoff_seconds == 4.25
        """
        import app.services.llm_service as llm_mod
        from app.services.llm_service import get_llm, reset_llm

        reset_llm()
        fake_settings = Mock()
        fake_settings.llm_api_key = "k"
        fake_settings.llm_base_url = "https://api.example.com/v1"
        fake_settings.llm_model_name = "m"
        fake_settings.llm_temperature = 0.0
        fake_settings.llm_timeout = 30
        fake_settings.llm_min_request_interval_seconds = 0.0
        fake_settings.llm_http_max_retries = 2
        fake_settings.llm_requests_per_minute_limit = 0
        fake_settings.llm_rate_limit_max_retries = 0
        fake_settings.llm_retry_backoff_seconds = 4.25

        # get_llm 内部是 `from app.config import get_settings`，需 patch 源模块
        with patch("app.config.get_settings", return_value=fake_settings):
            client = get_llm()
            assert client.retry_backoff_seconds == 4.25
        reset_llm()
        assert llm_mod._llm_instance is None
