from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os

from fastapi import HTTPException, Request
import pytest

os.environ["SSLKEYLOGFILE"] = ""

import app.middleware.auth as auth_module
import app.middleware.observability as observability
import app.repositories.user_repository as user_repository_module
from app.api.routes.observability_api import _require_developer
from app.services import auth_service, jwt_service


class FakeRefreshRepository:
    """认证刷新测试仓储，只记录账号查询和吊销调用。"""

    def __init__(self, user: dict | None, next_version: int | None = 4) -> None:
        """初始化测试仓储。

        Args:
            user: 查询账号时返回的预设记录。
            next_version: 吊销全部令牌后返回的新版本号。

        Returns:
            None.
        """
        self.user = user
        self.next_version = next_version
        self.requested_user_id: str | None = None
        self.revoked_user_id: str | None = None

    async def ensure_schema(self) -> None:
        """模拟完成 users schema 初始化，无传参且无返回值。"""

    async def get_by_user_id(self, user_id: str) -> dict | None:
        """记录查询身份并返回预设账号。

        Args:
            user_id: refresh token 中解析出的用户标识。

        Returns:
            预设账号记录，账号删除场景返回 None。
        """
        self.requested_user_id = user_id
        return self.user

    async def revoke_all_tokens(self, user_id: str) -> int | None:
        """记录吊销身份并返回预设版本。

        Args:
            user_id: 需要吊销全部 refresh token 的用户标识。

        Returns:
            吊销后的令牌版本，账号不存在时返回 None。
        """
        self.revoked_user_id = user_id
        if self.user is not None and self.next_version is not None:
            self.user = {**self.user, "token_version": self.next_version}
        return self.next_version


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


def _patch_settings(monkeypatch, **overrides):
    settings = type(
        "S",
        (),
        {
            "app_env": "local",
            "auth_enabled": True,
            "api_key": "test-api-key",
            "jwt_secret_key": "phase11-secret",
            "jwt_expire_hours": 24,
            "jwt_refresh_expire_days": 7,
            "rate_limit": "60/minute",
            "host": "127.0.0.1",
            "developer_usernames": "",
            "admin_usernames": "",
            **overrides,
        },
    )()
    monkeypatch.setattr(auth_service, "_settings", lambda: settings)
    monkeypatch.setattr(jwt_service, "_settings", auth_service._settings)
    return settings


def test_auth_accepts_api_key(monkeypatch) -> None:
    request = _request()
    _patch_settings(monkeypatch)

    user = asyncio.run(auth_module.get_current_user(request, bearer=None, api_key="test-api-key"))

    assert user == {
        "sub": "api_key_user",
        "user_id": "api_key_user",
        "username": "api_key_user",
        "role": "user",
    }
    assert request.state.current_user == user


def test_token_payload_uses_configured_developer_role(monkeypatch) -> None:
    settings = _patch_settings(monkeypatch, developer_usernames="alice")

    payload = auth_service.build_token_payload(
        {
            "user_id": "user-1",
            "username": "Alice",
            "role": "user",
            "token_version": 0,
        },
        settings=settings,
    )
    access_payload = jwt_service.verify_jwt(payload["access_token"], settings=settings)

    assert payload["user"]["role"] == "developer"
    assert access_payload["role"] == "developer"


def test_admin_role_takes_precedence_over_developer(monkeypatch) -> None:
    settings = _patch_settings(
        monkeypatch,
        developer_usernames="alice",
        admin_usernames="alice",
    )

    payload = auth_service.build_token_payload(
        {
            "user_id": "user-1",
            "username": "alice",
            "role": "user",
            "token_version": 0,
        },
        settings=settings,
    )

    assert payload["user"]["role"] == "admin"


def test_register_rejects_preconfigured_privileged_username(monkeypatch) -> None:
    class FakeRepository:
        created = False

        async def ensure_schema(self):
            return None

        async def get_by_username(self, username):
            return None

        async def create_user(self, username, password_hash, role="user"):
            self.created = True
            return {"user_id": "user-1", "username": username, "role": role}

    repository = FakeRepository()
    _patch_settings(monkeypatch, developer_usernames="alice")
    monkeypatch.setattr(auth_service, "get_user_repository", lambda: repository)

    with pytest.raises(ValueError, match="高权限用户名"):
        asyncio.run(auth_service.register_user("alice", "secret1"))

    assert repository.created is False


def test_observability_accepts_admin_and_developer_roles() -> None:
    assert _require_developer({"user_id": "u1", "role": "developer"}) == "u1"
    assert _require_developer({"user_id": "u2", "role": "admin"}) == "u2"

    with pytest.raises(Exception) as exc_info:
        _require_developer({"user_id": "u3", "role": "user"})

    assert getattr(exc_info.value, "status_code", None) == 403


def test_refresh_uses_database_user_and_preserves_absolute_expiry(monkeypatch) -> None:
    """刷新必须读取当前账号，并且新 refresh 的 exp 不得晚于旧 refresh。"""
    settings = _patch_settings(monkeypatch)
    repository = FakeRefreshRepository(
        {
            "user_id": "user-1",
            "username": "alice",
            "role": "user",
            "is_active": True,
            "token_version": 3,
        }
    )
    monkeypatch.setattr(auth_service, "get_user_repository", lambda: repository)
    token = jwt_service.create_refresh_token(
        "user-1",
        username="stale-name",
        role="user",
        token_version=3,
        expires_delta=timedelta(minutes=10),
        settings=settings,
    )
    old_payload = jwt_service.verify_refresh_token(token, settings=settings)

    result = asyncio.run(auth_service.refresh_access_payload(token, settings=settings))

    access_payload = jwt_service.verify_jwt(result["access_token"], settings=settings)
    new_payload = jwt_service.verify_refresh_token(result["refresh_token"], settings=settings)
    rotated_result = asyncio.run(
        auth_service.refresh_access_payload(result["refresh_token"], settings=settings)
    )
    rotated_payload = jwt_service.verify_refresh_token(
        rotated_result["refresh_token"],
        settings=settings,
    )
    assert repository.requested_user_id == "user-1"
    assert access_payload["username"] == "alice"
    assert new_payload["exp"] <= old_payload["exp"]
    assert rotated_payload["exp"] <= old_payload["exp"]
    assert new_payload["token_version"] == 3


@pytest.mark.parametrize(
    "user",
    [
        None,
        {
            "user_id": "user-1",
            "username": "alice",
            "role": "user",
            "is_active": False,
            "token_version": 3,
        },
        {
            "user_id": "user-1",
            "username": "alice",
            "role": "user",
            "is_active": True,
            "token_version": 4,
        },
        {
            "user_id": "user-1",
            "username": "alice",
            "role": "developer",
            "is_active": True,
            "token_version": 3,
        },
    ],
    ids=["deleted", "inactive", "version_changed", "role_changed"],
)
def test_refresh_rejects_revoked_account_states(monkeypatch, user) -> None:
    """删除、禁用、版本变化或角色变化都必须让旧 refresh 返回统一 401。"""
    settings = _patch_settings(monkeypatch)
    repository = FakeRefreshRepository(user)
    monkeypatch.setattr(auth_service, "get_user_repository", lambda: repository)
    token = jwt_service.create_refresh_token(
        "user-1",
        username="alice",
        role="user",
        token_version=3,
        settings=settings,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth_service.refresh_access_payload(token, settings=settings))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "登录状态已失效，请重新登录"


def test_legacy_refresh_without_version_is_rejected(monkeypatch) -> None:
    """迁移前没有 token_version 的 refresh 必须明确要求重新登录。"""
    settings = _patch_settings(monkeypatch)
    now = datetime.now(timezone.utc)
    token = jwt_service._encode_token(
        {
            "sub": "user-1",
            "type": "refresh",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=10)).timestamp()),
        },
        settings=settings,
    )

    with pytest.raises(HTTPException) as exc_info:
        jwt_service.verify_refresh_token(token, settings=settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == auth_service.AUTH_SESSION_EXPIRED_DETAIL


def test_refresh_token_failures_use_one_public_error(monkeypatch) -> None:
    """签名、过期、类型和版本错误不能向客户端暴露不同账号状态。"""
    settings = _patch_settings(monkeypatch)
    invalid_tokens = [
        "not-a-jwt",
        jwt_service.create_access_token("user-1", token_version=3, settings=settings),
        jwt_service.create_refresh_token(
            "user-1",
            token_version=3,
            expires_delta=timedelta(seconds=-1),
            settings=settings,
        ),
        jwt_service._encode_token(
            {
                "sub": "user-1",
                "type": "refresh",
                "token_version": "invalid",
                "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()),
            },
            settings=settings,
        ),
    ]

    for token in invalid_tokens:
        with pytest.raises(HTTPException) as exc_info:
            jwt_service.verify_refresh_token(token, settings=settings)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == auth_service.AUTH_SESSION_EXPIRED_DETAIL


def test_logout_increments_account_token_version(monkeypatch) -> None:
    """服务端 logout 必须通过仓储原子递增账号 token_version。"""
    repository = FakeRefreshRepository(user=None, next_version=5)
    monkeypatch.setattr(auth_service, "get_user_repository", lambda: repository)

    asyncio.run(auth_service.revoke_user_sessions("user-1"))

    assert repository.revoked_user_id == "user-1"


def test_logout_makes_existing_refresh_token_invalid(monkeypatch) -> None:
    """账号级 logout 后，吊销前签发的 refresh token 必须统一返回 401。"""
    settings = _patch_settings(monkeypatch)
    repository = FakeRefreshRepository(
        {
            "user_id": "user-1",
            "username": "alice",
            "role": "user",
            "is_active": True,
            "token_version": 3,
        },
        next_version=4,
    )
    monkeypatch.setattr(auth_service, "get_user_repository", lambda: repository)
    token = jwt_service.create_refresh_token(
        "user-1",
        username="alice",
        role="user",
        token_version=3,
        settings=settings,
    )

    asyncio.run(auth_service.revoke_user_sessions("user-1"))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth_service.refresh_access_payload(token, settings=settings))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == auth_service.AUTH_SESSION_EXPIRED_DETAIL


def test_revoke_all_tokens_uses_one_atomic_update(monkeypatch) -> None:
    """仓储吊销必须用单条参数化 UPDATE 原子递增 token_version。"""
    calls: list[tuple[str, tuple[object, ...]]] = []

    async def fake_fetchrow(query: str, *params: object) -> dict:
        """记录 SQL 与参数，并返回数据库递增后的版本。"""
        calls.append((query, params))
        return {"token_version": 6}

    monkeypatch.setattr(user_repository_module, "fetchrow", fake_fetchrow)

    result = asyncio.run(user_repository_module.UserRepository().revoke_all_tokens("user-1"))

    assert result == 6
    assert len(calls) == 1
    assert "SET token_version = token_version + 1" in calls[0][0]
    assert calls[0][1] == ("user-1",)


def test_record_timed_event_writes_structured_log(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(observability.time, "monotonic", lambda: 12.5)

    def fake_write_log(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(observability.paper_db, "write_log", fake_write_log)

    observability.record_timed_event(
        "research_completed",
        "graph_facade",
        12.0,
        query="rag",
        result_count=1,
        session_id="session-1",
    )

    assert captured == {
        "event_type": "research_completed",
        "node_name": "graph_facade",
        "query": "rag",
        "result_count": 1,
        "latency_ms": 500,
        "session_id": "session-1",
    }
