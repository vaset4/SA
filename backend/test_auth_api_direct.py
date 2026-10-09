from __future__ import annotations

import asyncio
import os

import pytest
from fastapi import HTTPException, Request

os.environ["SSLKEYLOGFILE"] = ""

import app.api.routes.auth_api as auth_api
from app.api.schemas.auth import LoginRequest, RefreshTokenRequest, RegisterRequest
from app.services import auth_service, jwt_service, rate_limit_service


def _request(path: str = "/api/v1/auth/login") -> Request:
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


def _patch_settings(monkeypatch, **overrides):
    settings = type(
        "S",
        (),
        {
            "app_env": "local",
            "auth_enabled": True,
            "api_key": "test-api-key",
            "jwt_secret_key": "test-secret",
            "jwt_expire_hours": 24,
            "jwt_refresh_expire_days": 7,
            "rate_limit": "1/minute",
            **overrides,
        },
    )()
    monkeypatch.setattr(auth_service, "_settings", lambda: settings)
    monkeypatch.setattr(jwt_service, "_settings", auth_service._settings)
    monkeypatch.setattr(rate_limit_service, "_settings", auth_service._settings)
    return settings


def test_login_route_rate_limits_anonymous_attempts(monkeypatch) -> None:
    fake_time = {"value": 100.0}
    calls: list[str] = []
    _patch_settings(monkeypatch)
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    rate_limit_service.reset_rate_limiter()

    async def fake_authenticate(username, password):  # noqa: ANN001
        calls.append(username)
        return {"user_id": "user-1", "username": username}

    monkeypatch.setattr(auth_api.auth_service, "authenticate_user", fake_authenticate)
    monkeypatch.setattr(auth_api.auth_service, "build_token_payload", lambda user: {"user": user})

    body = LoginRequest(username="alice", password="secret1")
    first = asyncio.run(auth_api.login(body, _request()))
    with pytest.raises(Exception) as exc_info:
        asyncio.run(auth_api.login(body, _request()))

    assert first["user"]["username"] == "alice"
    assert exc_info.value.status_code == 429
    assert calls == ["alice"]


def test_auth_route_limits_are_separated_by_identifier(monkeypatch) -> None:
    fake_time = {"value": 200.0}
    calls: list[str] = []
    _patch_settings(monkeypatch)
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    rate_limit_service.reset_rate_limiter()

    async def fake_register(username, password):  # noqa: ANN001
        calls.append(username)
        return {"user_id": f"user-{username}", "username": username}

    monkeypatch.setattr(auth_api.auth_service, "register_user", fake_register)
    monkeypatch.setattr(auth_api.auth_service, "build_token_payload", lambda user: {"user": user})

    alice = asyncio.run(
        auth_api.register(RegisterRequest(username="alice", password="secret1"), _request("/api/v1/auth/register"))
    )
    bob = asyncio.run(
        auth_api.register(RegisterRequest(username="bob", password="secret1"), _request("/api/v1/auth/register"))
    )

    assert alice["user"]["username"] == "alice"
    assert bob["user"]["username"] == "bob"
    assert calls == ["alice", "bob"]


def test_refresh_route_rate_limits_token_fingerprint(monkeypatch) -> None:
    fake_time = {"value": 300.0}
    calls: list[str] = []
    _patch_settings(monkeypatch)
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    rate_limit_service.reset_rate_limiter()

    async def fake_refresh(token):  # noqa: ANN001
        calls.append(token)
        return {"ok": True}

    monkeypatch.setattr(auth_api.auth_service, "refresh_access_payload", fake_refresh)

    body = RefreshTokenRequest(refresh_token="refresh-token-value")
    first = asyncio.run(auth_api.refresh_token(body, _request("/api/v1/auth/refresh")))
    with pytest.raises(Exception) as exc_info:
        asyncio.run(auth_api.refresh_token(body, _request("/api/v1/auth/refresh")))

    assert first == {"ok": True}
    assert exc_info.value.status_code == 429
    assert calls == ["refresh-token-value"]


def test_logout_route_revokes_authenticated_user(monkeypatch) -> None:
    """logout route 必须把服务端身份交给吊销服务。"""
    calls: list[str] = []

    async def fake_revoke(user_id: str) -> None:
        """记录路由提交的用户标识，无返回值。"""
        calls.append(user_id)

    monkeypatch.setattr(auth_api.auth_service, "revoke_user_sessions", fake_revoke)

    response = asyncio.run(auth_api.logout({"user_id": "user-1"}))

    assert response == {"revoked": True}
    assert calls == ["user-1"]


def test_logout_route_rejects_api_key_identity(monkeypatch) -> None:
    """API key 合成身份不得进入用户 refresh 会话吊销流程。"""
    calls: list[str] = []

    async def fake_revoke(user_id: str) -> None:
        """记录意外进入吊销服务的用户标识。"""
        calls.append(user_id)

    monkeypatch.setattr(auth_api.auth_service, "revoke_user_sessions", fake_revoke)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth_api.logout({"user_id": "api_key_user"}))

    assert exc_info.value.status_code == 422
    assert calls == []
