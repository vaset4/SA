from __future__ import annotations

import os

import pytest
from fastapi import Request

os.environ["SSLKEYLOGFILE"] = ""

import app.api.facades.system_facade as system_facade_module
import app.api.routes.system_api as system_api_module
import app.middleware.auth as auth_module
from app.services import auth_service, jwt_service, rate_limit_service


def _patch_settings(monkeypatch, **overrides):
    settings = type(
        "S",
        (),
        {
            "app_env": "local",
            "auth_enabled": True,
            "api_key": "test-api-key",
            "jwt_secret_key": "phase4-secret",
            "jwt_expire_hours": 24,
            "rate_limit": "60/minute",
            **overrides,
        },
    )()
    monkeypatch.setattr(auth_service, "_settings", lambda: settings)
    monkeypatch.setattr(jwt_service, "_settings", auth_service._settings)
    monkeypatch.setattr(rate_limit_service, "_settings", auth_service._settings)
    return settings


def _request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "scheme": "http",
        "server": ("testserver", 80),
        "query_string": b"",
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_rate_limit_rejects_burst_requests_on_protected_endpoint(monkeypatch) -> None:
    fake_time = {"value": 100.0}
    _patch_settings(monkeypatch, rate_limit="2/minute")
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    auth_module.reset_rate_limiter()

    first = await auth_module.get_current_user_with_rate_limit(_request("/research"), None, "test-api-key")
    second = await auth_module.get_current_user_with_rate_limit(_request("/research"), None, "test-api-key")
    with pytest.raises(Exception) as exc_info:
        await auth_module.get_current_user_with_rate_limit(_request("/research"), None, "test-api-key")

    assert first["sub"] == "api_key_user"
    assert second["sub"] == "api_key_user"
    assert exc_info.value.status_code == 429
    assert "请求过于频繁" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rate_limit_window_resets_after_expiration(monkeypatch) -> None:
    fake_time = {"value": 200.0}
    _patch_settings(monkeypatch, rate_limit="1/second")
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    auth_module.reset_rate_limiter()

    first = await auth_module.get_current_user_with_rate_limit(_request("/chat"), None, "test-api-key")
    with pytest.raises(Exception) as exc_info:
        await auth_module.get_current_user_with_rate_limit(_request("/chat"), None, "test-api-key")
    fake_time["value"] = 202.0
    third = await auth_module.get_current_user_with_rate_limit(_request("/chat"), None, "test-api-key")

    assert first["sub"] == "api_key_user"
    assert exc_info.value.status_code == 429
    assert third["sub"] == "api_key_user"


@pytest.mark.asyncio
async def test_health_remains_public_and_not_rate_limited(monkeypatch) -> None:
    async def fake_health_payload():
        return {"status": "healthy"}

    monkeypatch.setattr(system_facade_module, "build_health_payload", fake_health_payload)

    first = await system_api_module.health()
    second = await system_api_module.health()

    assert first == {"status": "healthy"}
    assert second == {"status": "healthy"}


@pytest.mark.asyncio
async def test_rate_limit_still_applies_when_auth_disabled(monkeypatch) -> None:
    fake_time = {"value": 400.0}
    _patch_settings(monkeypatch, auth_enabled=False, rate_limit="1/minute")
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    auth_module.reset_rate_limiter()

    first = await auth_module.get_current_user_with_rate_limit(_request("/compare"), None, None)
    with pytest.raises(Exception) as exc_info:
        await auth_module.get_current_user_with_rate_limit(_request("/compare"), None, None)

    assert first["role"] == "developer"
    assert exc_info.value.status_code == 429
