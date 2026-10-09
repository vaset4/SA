"""【模型服务】【共享排队】确定性传输与取消回归。作者：xxx。"""

import asyncio
import threading
import time
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
import requests

from app.config import Settings
from app.services import provider_request_queue as queue_module
from app.services.llm_service import OpenAICompatibleLLM
from app.services.provider_request_queue import (
    ProviderRequestCancelled, ProviderRequestQueue, provider_request_context,
    queue_policy_identity, retry_after_seconds, run_provider_thread,
)


def policy():
    """无参数；返回本次批准的五次每分钟策略。"""
    return queue_policy_identity(Settings(_env_file=None, llm_queue_enabled=True, llm_queue_group="unit"))


@pytest.fixture(autouse=True)
def silence_observation(monkeypatch):
    """参数为测试替换器；隔离持久观察，保留回调断言。"""
    monkeypatch.setattr(queue_module, "_emit", Mock())


def test_retry_after_supports_seconds_dates_and_invalid_values():
    """无参数；真实等待头支持秒和HTTP日期，不截断为十五秒。"""
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    assert retry_after_seconds("120") == 120
    assert retry_after_seconds("Sun, 04 Oct 2026 00:02:00 GMT", now=now) == 120
    for value in (None, "bad", "nan", "inf", "-1"):
        assert retry_after_seconds(value) is None


def test_cancelled_wait_never_consumes_slot():
    """无参数；已取消运行不会加入协调队列或消费额度。"""
    repository = Mock()
    event = threading.Event()
    event.set()
    with provider_request_context(cancel_event=event):
        with pytest.raises(ProviderRequestCancelled):
            ProviderRequestQueue(policy(), repository).acquire({})
    repository.enqueue.assert_not_called()
    repository.poll.assert_not_called()


def test_waiting_cancel_removes_unsent_entry(monkeypatch):
    """参数为替换器；排队中取消清理等待条目，不假造发送事实。"""
    repository = Mock()
    repository.poll.return_value = {"admitted": False, "estimated_wait_seconds": 12}
    event = threading.Event()
    monkeypatch.setattr(queue_module, "wait_for_provider_retry", lambda seconds: event.set())
    with provider_request_context(cancel_event=event):
        with pytest.raises(ProviderRequestCancelled):
            ProviderRequestQueue(policy(), repository).acquire({"call_id": "c"})
    repository.cancel.assert_called_once()
    assert not any(call.args[0] == "provider_admitted" for call in queue_module._emit.call_args_list)


def test_cooldown_exceeding_deadline_fails_without_waiting():
    """无参数；共享冷却超出剩余运行期限时立即终止而不延长期限。"""
    repository = Mock()
    repository.poll.return_value = {"admitted": False, "estimated_wait_seconds": 60}
    with provider_request_context(cancel_event=threading.Event(), deadline=time.monotonic() + 10):
        with pytest.raises(TimeoutError):
            ProviderRequestQueue(policy(), repository).acquire({})
    repository.cancel.assert_called_once()


def test_shared_cooldown_preserves_header_and_configured_source():
    """无参数；两种冷却来源独立可查并写入共享仓储。"""
    repository = Mock()
    queue = ProviderRequestQueue(policy(), repository)
    assert queue.cooldown("120", {}) == 120
    repository.cooldown.assert_called_with("unit", 120)
    assert queue.cooldown(None, {}) == 60
    assert queue_module._emit.call_args.args[1]["source"] == "configured"


def test_queue_database_unavailable_is_not_bypassed():
    """无参数；协调库故障明确失败，不返回发送许可。"""
    repository = Mock()
    repository.enqueue.side_effect = ConnectionError("coordination unavailable")
    with pytest.raises(ConnectionError):
        ProviderRequestQueue(policy(), repository).acquire({})
    repository.poll.assert_not_called()


def test_enabled_queue_requires_explicit_database():
    """无参数；启用组不能使用隐式业务库作为协调介质。"""
    with pytest.raises(ValueError, match="协调数据库"):
        queue_module._configured_queue("", tuple(policy().items()))


@pytest.mark.asyncio
async def test_async_cancel_waits_until_real_thread_stops():
    """无参数；异步取消先通知线程，线程退出前调用方不能宣告排空。"""
    entered = threading.Event()
    release = threading.Event()
    cancel = threading.Event()

    def work():
        """无参数；模拟收到取消后仍在收尾的真实同步传输。"""
        entered.set()
        cancel.wait(2)
        release.wait(2)

    with provider_request_context(cancel_event=cancel):
        task = asyncio.create_task(run_provider_thread(work))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert cancel.is_set()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_http_retry_acquires_each_attempt_and_keeps_budget(monkeypatch):
    """参数为替换器；每次真实HTTP重试重新准入，429不增加既有尝试上限。"""
    from app.services import llm_service

    queue = Mock(policy={"enabled": True})
    monkeypatch.setattr(llm_service, "get_provider_queue", lambda: queue)
    monkeypatch.setattr(llm_service, "wait_for_provider_retry", lambda seconds: None)
    client = OpenAICompatibleLLM("model", "key", "https://unit.invalid", 0, 30,
                                 request_exception_max_retries=1, empty_response_max_retries=1)
    monkeypatch.setattr(client, "_streaming_enabled", lambda: False)
    response = requests.Response()
    response.status_code = 429
    response._content = b"rate limited"
    response._content_consumed = True
    response.headers["Retry-After"] = "120"
    post = Mock(return_value=response)
    monkeypatch.setattr(llm_service.requests, "post", post)
    with pytest.raises(requests.HTTPError):
        client.invoke("question", use_cache=False)
    assert post.call_count == queue.acquire.call_count == 2
    assert queue.cooldown.call_count == 2


def test_cancel_before_admission_has_no_http_observation(monkeypatch):
    """参数为替换器；未发取消不得计作失败HTTP或增加尝试计数。"""
    from app.services import llm_service

    queue = Mock(policy={"enabled": True})
    queue.acquire.side_effect = ProviderRequestCancelled("cancelled")
    monkeypatch.setattr(llm_service, "get_provider_queue", lambda: queue)
    client = OpenAICompatibleLLM("model", "key", "https://unit.invalid", 0, 30)
    record = Mock(http_attempt_count=0, call_id="call")
    post = Mock()
    monkeypatch.setattr(client, "_post_chat_completions", post)
    with pytest.raises(ProviderRequestCancelled):
        client._invoke_http_attempt({}, {}, use_stream=False, call_record=record)
    assert record.http_attempt_count == 0
    post.assert_not_called()


def test_cancel_after_admission_releases_without_http_count(monkeypatch):
    """参数为替换器；获得资格后用户取消，最后检查拒绝发送并释放未计费资格。"""
    from app.services import llm_service

    event = threading.Event()
    repository = Mock()
    repository.poll.return_value = {"admitted": True, "_connection": object()}
    queue = ProviderRequestQueue(policy(), repository)
    monkeypatch.setattr(queue_module, "_emit", lambda name, data: event.set())
    monkeypatch.setattr(llm_service, "get_provider_queue", lambda: queue)
    client = OpenAICompatibleLLM("model", "key", "https://unit.invalid", 0, 30)
    record = Mock(http_attempt_count=0, call_id="call")
    post = Mock()
    monkeypatch.setattr(client, "_post_chat_completions", post)
    with provider_request_context(cancel_event=event):
        with pytest.raises(ProviderRequestCancelled):
            client._invoke_http_attempt({}, {}, use_stream=False, call_record=record)
    assert record.http_attempt_count == 0
    repository.mark_started.assert_not_called()
    repository.release.assert_called_once()
    post.assert_not_called()


@pytest.mark.parametrize("cancelled", [True, False])
def test_continuous_sse_obeys_cancel_and_total_deadline(monkeypatch, cancelled):
    """参数为替换器和终止原因；持续返回数据也必须结束已取消或到期的流并关闭响应。"""
    from app.services import llm_service

    event = threading.Event()
    queue = Mock(policy={"enabled": True})
    monkeypatch.setattr(llm_service, "get_provider_queue", lambda: queue)
    response = Mock(status_code=200, headers={"Content-Type": "text/event-stream"})

    def chunks(**kwargs):
        """参数为读取选项；在持续心跳前触发真实取消或总期限。"""
        if cancelled:
            event.set()
        else:
            time.sleep(0.02)
        yield b":heartbeat\n\n"
        raise AssertionError("cancelled stream consumed again")

    response.iter_content.side_effect = chunks
    monkeypatch.setattr(llm_service.requests, "post", lambda *args, **kwargs: response)
    client = OpenAICompatibleLLM("model", "key", "https://unit.invalid", 0, 30)
    with provider_request_context(cancel_event=event, deadline=time.monotonic() + (5 if cancelled else 0.01)):
        with pytest.raises(ProviderRequestCancelled if cancelled else TimeoutError):
            client.invoke("question", use_cache=False)
    response.close.assert_called_once()


def test_cooldown_store_failure_keeps_original_429_and_stops_retry(monkeypatch):
    """参数为替换器；协调写失败不能覆盖已发生的429，也不能继续发送重试。"""
    from app.services import llm_service

    queue = Mock(policy={"enabled": True})
    queue.cooldown.side_effect = ConnectionError("coordinator unavailable")
    monkeypatch.setattr(llm_service, "get_provider_queue", lambda: queue)
    response = requests.Response()
    response.status_code = 429
    response._content = b"rate limited"
    response._content_consumed = True
    response.close = Mock()
    post = Mock(return_value=response)
    monkeypatch.setattr(llm_service.requests, "post", post)
    client = OpenAICompatibleLLM("model", "key", "https://unit.invalid", 0, 30)
    with pytest.raises(requests.HTTPError) as error:
        client.invoke("question", use_cache=False)
    assert error.value.response.status_code == 429
    assert isinstance(error.value, llm_service.ProviderCooldownError)
    post.assert_called_once()
    response.close.assert_called_once()
