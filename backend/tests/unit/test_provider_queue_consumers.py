"""【模型服务】【消费接线】评分和配置探活共用真实准入。作者：xxx。"""

from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest
import requests

from app.config import get_settings
from app.services import llm_profile_service, provider_request_queue
from evals.ragas.frozen_judge import _AuditedProjectLLM
from evals.suites.identity import describe_model_client


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """参数为补丁工具；任何遗漏的真实HTTP都明确失败。"""
    monkeypatch.setattr(requests.Session, "request", Mock(side_effect=AssertionError("unexpected network")))


@pytest.mark.parametrize("same_key", [True, False])
def test_profile_probe_only_current_credential_uses_queue(monkeypatch, same_key):
    """参数为凭证关系；当前凭证探活领取、发送、释放，其他配置不占本组。"""
    settings = get_settings().model_copy(update={"llm_api_key": "fixture-key", "llm_base_url": "https://fixture.invalid/v1"})
    monkeypatch.setattr(llm_profile_service, "get_settings", lambda: settings)
    actions = []
    queue = SimpleNamespace(acquire=lambda identity: actions.append("acquire") or {},
        mark_started=lambda ticket: actions.append("started"),
        release=lambda ticket, **options: actions.append("released"))
    monkeypatch.setattr(provider_request_queue, "get_provider_queue", lambda: queue)

    def send(*args, **kwargs):
        """参数为探活HTTP；返回真实结构的隔离响应并记录发送顺序。"""
        actions.append("http")
        response = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setattr(requests, "post", send)
    result = llm_profile_service.test_connection(base_url=settings.llm_base_url, model_name="fixture",
        api_key=settings.llm_api_key if same_key else "other-key")
    assert result.ok
    assert actions == (["acquire", "started", "http", "released"] if same_key else ["http"])


def test_profile_probe_cancelled_before_send_releases_ticket(monkeypatch):
    """参数为补丁工具；准入后的取消不发送HTTP，仍归还未使用资格。"""
    settings = get_settings()
    released = Mock()
    queue = SimpleNamespace(acquire=lambda identity: {}, release=released,
        mark_started=Mock(side_effect=provider_request_queue.ProviderRequestCancelled("cancelled")))
    monkeypatch.setattr(provider_request_queue, "get_provider_queue", lambda: queue)
    send = Mock(side_effect=AssertionError("cancelled request sent"))
    monkeypatch.setattr(requests, "post", send)
    with pytest.raises(provider_request_queue.ProviderRequestCancelled):
        llm_profile_service.test_connection(base_url=settings.llm_base_url, model_name="fixture", api_key=settings.llm_api_key)
    send.assert_not_called()
    released.assert_called_once()


def test_judge_transport_and_identity_share_current_policy():
    """无参数；评分真实重试上限与新身份一致，公开身份不携带协调连接或密钥。"""
    settings = get_settings()
    client = _AuditedProjectLLM(settings)._build_client(temperature=0)
    identity = describe_model_client(client)
    assert client.request_exception_max_retries == settings.llm_http_max_retries
    assert identity["provider_queue"]["http_max_retries"] == client.request_exception_max_retries
    assert "database_url" not in str(identity)
    assert "api_key" not in identity


def test_probe_publishes_cooldown_before_releasing_send_ticket(monkeypatch):
    """参数为补丁工具；429冷却先提交，避免下一进程在旧额度事实下继续发送。"""
    settings = get_settings()
    actions = []
    queue = SimpleNamespace(acquire=lambda identity: {}, mark_started=lambda ticket: None,
        cooldown=lambda header, identity: actions.append(("cooldown", header)),
        release=lambda ticket, **options: actions.append(("release", None)))
    monkeypatch.setattr(provider_request_queue, "get_provider_queue", lambda: queue)
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "60"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    result = llm_profile_service.test_connection(base_url=settings.llm_base_url,
        model_name="fixture", api_key=settings.llm_api_key)
    assert result.status_code == 429
    assert actions == [("cooldown", "60"), ("release", None)]


def test_probe_cancellation_after_send_commit_still_sends(monkeypatch):
    """参数为补丁工具；发送提交后的取消不撤回计费请求，真实HTTP结束后释放。"""
    settings = get_settings()
    cancel = threading.Event()
    release = Mock()
    queue = SimpleNamespace(acquire=lambda identity: {}, mark_started=lambda ticket: cancel.set(), release=release)
    monkeypatch.setattr(provider_request_queue, "get_provider_queue", lambda: queue)
    response = requests.Response()
    response.status_code = 200
    send = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", send)
    with provider_request_queue.provider_request_context(cancel_event=cancel):
        result = llm_profile_service.test_connection(base_url=settings.llm_base_url,
            model_name="fixture", api_key=settings.llm_api_key)
    assert result.ok
    send.assert_called_once()
    release.assert_called_once()
