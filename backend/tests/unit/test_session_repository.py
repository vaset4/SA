"""
SessionRepository owner 与状态 CAS 回归测试。

作者: xxx
时间: 2026-08-18 16:00:00
"""

from __future__ import annotations

import pytest

from app.repositories import session_repository
from app.repositories.session_repository import SessionRepository


@pytest.mark.asyncio
async def test_get_session_query_is_owner_scoped(monkeypatch) -> None:
    """用户侧 session 读取必须在 SQL 中同时限制 user_id 与 session_id。"""
    captured: dict = {}

    async def fake_fetchrow(query: str, *args):
        """记录 SQL 与参数并返回一行 session。"""
        captured["query"] = query
        captured["args"] = args
        return {
            "session_id": "session-1",
            "user_id": "user-1",
            "paper_ids": [],
            "chat_history": [],
            "attempt_count": 0,
        }

    monkeypatch.setattr(session_repository, "fetchrow", fake_fetchrow)

    row = await SessionRepository().get_session("session-1", user_id="user-1")

    assert "WHERE user_id=$1 AND session_id=$2" in captured["query"]
    assert captured["args"] == ("user-1", "session-1")
    assert row["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_transition_status_query_uses_owner_and_expected_status(monkeypatch) -> None:
    """状态迁移 SQL 必须以 owner 和 current status 作为 CAS 条件。"""
    captured: dict = {}

    async def fake_fetchrow(query: str, *args):
        """记录状态迁移 SQL，并返回迁移后的 session。"""
        captured["query"] = query
        captured["args"] = args
        return {
            "session_id": "session-1",
            "user_id": "user-1",
            "status": "running",
            "paper_ids": [],
            "chat_history": [],
            "attempt_count": 1,
        }

    monkeypatch.setattr(session_repository, "fetchrow", fake_fetchrow)

    row = await SessionRepository().transition_status(
        "session-1",
        user_id="user-1",
        expected_status="created",
        target_status="running",
        last_error=None,
        increment_attempt=True,
    )

    assert "WHERE user_id=$5 AND session_id=$6 AND status=$7" in captured["query"]
    assert captured["args"][4:] == ("user-1", "session-1", "created", None)
    assert "current_run_id IS NOT DISTINCT FROM $8::text" in captured["query"]
    assert row["status"] == "running"


@pytest.mark.asyncio
async def test_scope_binding_cannot_write_checkpoint_thread(monkeypatch) -> None:
    """scope 补齐不得绕过 legacy thread 的独立一次性绑定边界。"""
    captured: dict = {}

    async def fake_fetchrow(query: str, *args):
        """记录 scope SQL 并返回已补齐的 session。"""
        captured["query"] = query
        captured["args"] = args
        return {
            "session_id": "session-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "paper_ids": ["paper-1"],
            "chat_history": [],
            "attempt_count": 0,
        }

    monkeypatch.setattr(session_repository, "fetchrow", fake_fetchrow)

    await SessionRepository().bind_session_scope(
        "session-1",
        user_id="user-1",
        conversation_id="conv-1",
        project_id="project-1",
        research_mode="internal",
        paper_ids=["paper-1"],
    )

    assert "thread_id" not in captured["query"]
    assert "WHERE user_id=$6 AND session_id=$7" in captured["query"]
    assert captured["args"][-2:] == ("user-1", "session-1")
