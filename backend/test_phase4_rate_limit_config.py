from __future__ import annotations

import os
import pytest
from fastapi import Request

os.environ["SSLKEYLOGFILE"] = ""

import app.middleware.auth as auth_module
from app.services import auth_service, rate_limit_service


def _request(path: str = "/research") -> Request:
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


def test_parse_rate_limit_supports_minute_second_and_hour() -> None:
    assert auth_module.parse_rate_limit("2/minute") == (2, 60.0)
    assert auth_module.parse_rate_limit("1/second") == (1, 1.0)
    assert auth_module.parse_rate_limit("120/hour") == (120, 3600.0)


def test_parse_rate_limit_rejects_invalid_strings() -> None:
    with pytest.raises(ValueError):
        auth_module.parse_rate_limit("abc")
    with pytest.raises(ValueError):
        auth_module.parse_rate_limit("0/minute")
    with pytest.raises(ValueError):
        auth_module.parse_rate_limit("3/day")


def test_enforce_rate_limit_uses_settings_rate_limit(monkeypatch) -> None:
    auth_module.reset_rate_limiter()
    fake_time = {"value": 10.0}
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    monkeypatch.setattr(
        rate_limit_service,
        "_settings",
        lambda: type("S", (), {"rate_limit": "2/minute"})(),
    )

    request = _request()
    user = {"sub": "user-1"}

    rate_limit_service.enforce_rate_limit(request, user)
    rate_limit_service.enforce_rate_limit(request, user)
    with pytest.raises(Exception) as exc_info:
        rate_limit_service.enforce_rate_limit(request, user)

    assert exc_info.value.status_code == 429
    auth_module.reset_rate_limiter()


def test_anonymous_auth_limit_uses_identifier_dimension(monkeypatch) -> None:
    auth_module.reset_rate_limiter()
    fake_time = {"value": 30.0}
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    monkeypatch.setattr(
        rate_limit_service,
        "_settings",
        lambda: type("S", (), {"rate_limit": "1/minute"})(),
    )

    request = _request("/api/v1/auth/login")
    rate_limit_service.enforce_anonymous_auth_rate_limit(request, "login", "alice")
    rate_limit_service.enforce_anonymous_auth_rate_limit(request, "login", "bob")
    with pytest.raises(Exception) as exc_info:
        rate_limit_service.enforce_anonymous_auth_rate_limit(request, "login", "alice")

    assert exc_info.value.status_code == 429
    auth_module.reset_rate_limiter()


def test_anonymous_auth_token_fingerprint_is_case_sensitive(monkeypatch) -> None:
    auth_module.reset_rate_limiter()
    fake_time = {"value": 40.0}
    monkeypatch.setattr(rate_limit_service.time, "monotonic", lambda: fake_time["value"])
    monkeypatch.setattr(
        rate_limit_service,
        "_settings",
        lambda: type("S", (), {"rate_limit": "1/minute"})(),
    )

    request = _request("/api/v1/auth/refresh")
    rate_limit_service.enforce_anonymous_auth_rate_limit(request, "refresh", "TokenValue")
    rate_limit_service.enforce_anonymous_auth_rate_limit(request, "refresh", "tokenvalue")
    with pytest.raises(Exception) as exc_info:
        rate_limit_service.enforce_anonymous_auth_rate_limit(request, "refresh", "TokenValue")

    assert exc_info.value.status_code == 429
    auth_module.reset_rate_limiter()


def test_non_local_env_rejects_auth_disabled() -> None:
    with pytest.raises(Exception) as exc_info:
        auth_service.validate_auth_configuration(
            type("S", (), {
                "app_env": "prod",
                "auth_enabled": False,
                "jwt_secret_key": "phase4-secret",
            })()
        )

    assert exc_info.value.status_code == 500
    assert "不允许关闭认证" in exc_info.value.detail


def test_local_loopback_allows_default_secret_for_development() -> None:
    auth_service.validate_auth_configuration(
        type("S", (), {
            "app_env": "local",
            "auth_enabled": True,
            "jwt_secret_key": "change-me-in-production",
            "host": "127.0.0.1",
        })()
    )


def test_local_public_bind_rejects_default_secret() -> None:
    with pytest.raises(Exception) as exc_info:
        auth_service.validate_auth_configuration(
            type("S", (), {
                "app_env": "local",
                "auth_enabled": True,
                "jwt_secret_key": "change-me-in-production",
                "host": "0.0.0.0",
            })()
        )

    assert exc_info.value.status_code == 500
    assert "公开地址" in exc_info.value.detail


def test_local_public_bind_rejects_auth_disabled() -> None:
    with pytest.raises(Exception) as exc_info:
        auth_service.validate_auth_configuration(
            type("S", (), {
                "app_env": "local",
                "auth_enabled": False,
                "jwt_secret_key": "dev-secret",
                "host": "0.0.0.0",
            })()
        )

    assert exc_info.value.status_code == 500
    assert "不允许关闭认证" in exc_info.value.detail
