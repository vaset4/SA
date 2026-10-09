from __future__ import annotations

import os

import pytest
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials

os.environ["SSLKEYLOGFILE"] = ""

import app.api.facades.system_facade as system_facade_module
import app.api.routes.research_api as research_api_module
import app.api.routes.system_api as system_api_module
import app.middleware.auth as auth_module
from app.api.schemas.research import ResearchRequest
from app.services import auth_service, jwt_service


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
    return settings


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


class _FakeResearchSessionLifecycle:
    """为旧 API 形状测试提供 durable session 边界。"""

    def __init__(self, rows=None):
        """保存可选 session 列表。"""
        self.rows = list(rows or [])

    async def prepare_for_run(self, **payload):
        """返回与请求 session 一致的 running 记录。"""
        session_id = payload.get("session_id") or "session-test"
        return {
            "session_id": session_id,
            "conversation_id": payload.get("conversation_id"),
            "thread_id": session_id,
            "status": "running",
            "attempt_count": 1,
        }

    async def record_result(self, **payload):
        """把测试 graph 结果映射为 completed。"""
        session_id = payload["session_id"]
        return {
            "session_id": session_id,
            "conversation_id": None,
            "thread_id": session_id,
            "status": "completed",
            "attempt_count": 1,
        }

    async def list_sessions(self, **kwargs):
        """返回预置 durable session 列表。"""
        return self.rows


@pytest.mark.asyncio
async def test_protected_endpoint_requires_authentication(monkeypatch):
    _patch_settings(monkeypatch)

    with pytest.raises(Exception) as exc_info:
        await auth_module.get_current_user(_request(), bearer=None, api_key=None)

    assert exc_info.value.status_code == 401
    assert "JWT token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_protected_endpoint_accepts_api_key(monkeypatch):
    request = _request()
    _patch_settings(monkeypatch)

    user = await auth_module.get_current_user(request, bearer=None, api_key="test-api-key")

    assert user == {
        "sub": "api_key_user",
        "user_id": "api_key_user",
        "username": "api_key_user",
        "role": "user",
    }
    assert request.state.current_user == user


@pytest.mark.asyncio
async def test_protected_endpoint_accepts_jwt(monkeypatch):
    request = _request()
    _patch_settings(monkeypatch)
    token = auth_module.create_access_token("user-1", role="admin")
    bearer = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = await auth_module.get_current_user(request, bearer=bearer, api_key=None)

    assert user["sub"] == "user-1"
    assert user["role"] == "admin"
    assert request.state.current_user == user


@pytest.mark.asyncio
async def test_research_route_keeps_payload_shape(monkeypatch):
    async def fake_invoke(user_query: str, extra: dict | None = None):
        return {"report": f"stub:{user_query}"}, extra.get("session_id") if extra else "session-test"

    monkeypatch.setattr(research_api_module, "invoke_graph", fake_invoke)
    monkeypatch.setattr(research_api_module, "_resolve_identity_or_404", lambda *args: _async_none())
    monkeypatch.setattr(
        research_api_module,
        "get_research_session_lifecycle",
        lambda: _FakeResearchSessionLifecycle(),
    )
    async def fake_build_project_context(project_id, requested_paper_ids=None, user_id="anonymous"):
        return {
            "user_id": user_id,
            "project_id": project_id or "",
            "library_ids": [],
            "resolved_scope": {"paper_ids": requested_paper_ids or []} if requested_paper_ids else None,
            "filter_expr": "",
            "paper_ids": requested_paper_ids or [],
            "local_retrieval_allowed": bool(requested_paper_ids),
        }
    monkeypatch.setattr(research_api_module, "build_project_context", fake_build_project_context)

    response = await research_api_module.research(
        ResearchRequest(query="analyze rag", session_id="session-1", paper_ids=["paper-1"]),
        user={"sub": "user-1", "user_id": "user-1", "role": "user"},
    )

    assert response["report"] == "stub:analyze rag"
    assert response["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_list_research_sessions_filters_non_research_entries(monkeypatch):
    lifecycle = _FakeResearchSessionLifecycle([
        {
            "session_id": "chat-1",
            "topic": "默认对话",
            "status": "completed",
            "state_json": "{\"intent\":\"free_chat\",\"report\":\"hello\"}",
            "step_name": "chat",
            "paper_ids": [],
            "created_at": "2026-03-30T10:00:00Z",
            "updated_at": "2026-03-30T10:10:00Z",
        },
        {
            "session_id": "research-1",
            "conversation_id": "conv-1",
            "project_id": "project-1",
            "thread_id": "research-1",
            "research_mode": "internal",
            "topic": "深度研究 · Graph RAG",
            "status": "completed",
            "last_error": None,
            "attempt_count": 1,
            "state_json": "{\"intent\":\"deep_research\",\"report\":\"# Report\"}",
            "step_name": "done",
            "paper_ids": ["paper-1"],
            "created_at": "2026-03-30T11:00:00Z",
            "updated_at": "2026-03-30T11:30:00Z",
        },
    ])
    monkeypatch.setattr(
        research_api_module,
        "get_research_session_lifecycle",
        lambda: lifecycle,
    )

    response = await research_api_module.list_research_sessions(
        user={"sub": "user-1", "user_id": "user-1", "role": "user"},
    )

    assert response == {
        "sessions": [
            {
                "session_id": "research-1",
                "conversation_id": "conv-1",
                "project_id": "project-1",
                "thread_id": "research-1",
                "topic": "深度研究 · Graph RAG",
                "status": "completed",
                "last_error": None,
                "attempt_count": 1,
                "updated_at": "2026-03-30T11:30:00Z",
                "created_at": "2026-03-30T11:00:00Z",
                "step_name": "done",
                "paper_ids": ["paper-1"],
                "preview": "# Report",
                "intent": "deep_research",
                "research_mode": "internal",
                "evidence_count": 0,
                "idea_count": 0,
                "candidate_paper_count": 0,
                "external_gap_count": 0,
                "follow_up_count": 0,
                "key_findings": [],
            }
        ]
    }


async def _async_none():
    """返回 None 的异步测试辅助。"""
    return None


def test_staging_rejects_default_secret(monkeypatch):
    _patch_settings(
        monkeypatch,
        app_env="staging",
        jwt_secret_key=jwt_service.DEFAULT_DEV_SECRET,
    )

    with pytest.raises(Exception) as exc_info:
        auth_service.validate_auth_configuration(auth_service._settings())

    assert exc_info.value.status_code == 500
    assert "默认 JWT secret" in exc_info.value.detail


@pytest.mark.asyncio
async def test_health_remains_public(monkeypatch):
    async def fake_health_payload():
        return {"status": "healthy"}

    monkeypatch.setattr(system_facade_module, "build_health_payload", fake_health_payload)

    response = await system_api_module.health()

    assert response == {"status": "healthy"}
