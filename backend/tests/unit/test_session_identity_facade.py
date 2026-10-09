from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.api.facades import session_identity_facade


class _FakeSessionRepository:
    """为 identity facade 提供 owner-aware 异步 session 查询。"""

    def __init__(self, session: dict | None) -> None:
        self.session = session

    async def get_session(self, session_id: str, *, user_id: str | None = None) -> dict | None:
        if self.session is None:
            return None
        return {**self.session, "session_id": session_id, "user_id": user_id}


def _patch_session_repository(monkeypatch, session: dict | None) -> None:
    repository = _FakeSessionRepository(session)
    monkeypatch.setattr(session_identity_facade, "get_session_repository", lambda: repository)


@pytest.mark.asyncio
async def test_resolve_identity_uses_explicit_owned_conversation(monkeypatch) -> None:
    async def fake_get_conversation_or_none(user_id: str, conversation_id: str):
        assert user_id == "user-1"
        assert conversation_id == "conv-1"
        return {"id": "conv-1", "session_id": "session-1"}

    monkeypatch.setattr(
        session_identity_facade.conversation_facade,
        "get_conversation_or_none",
        fake_get_conversation_or_none,
    )
    _patch_session_repository(monkeypatch, None)

    identity = await session_identity_facade.resolve_conversation_identity(
        user_id="user-1",
        conversation_id=" conv-1 ",
        session_id=None,
    )

    assert identity.conversation_id == "conv-1"
    assert identity.session_id is None


@pytest.mark.asyncio
async def test_resolve_identity_uses_session_mapping(monkeypatch) -> None:
    async def fake_get_conversation_by_session_id_or_none(user_id: str, session_id: str):
        assert user_id == "user-1"
        assert session_id == "session-1"
        return {"id": "conv-1", "session_id": "session-1"}

    monkeypatch.setattr(
        session_identity_facade.conversation_facade,
        "get_conversation_by_session_id_or_none",
        fake_get_conversation_by_session_id_or_none,
    )
    _patch_session_repository(monkeypatch, {})

    identity = await session_identity_facade.resolve_conversation_identity(
        user_id="user-1",
        conversation_id=None,
        session_id=" session-1 ",
    )

    assert identity.conversation_id == "conv-1"
    assert identity.session_id == "session-1"


@pytest.mark.asyncio
async def test_resolve_identity_rejects_missing_session(monkeypatch) -> None:
    _patch_session_repository(monkeypatch, None)

    with pytest.raises(LookupError, match="会话不存在"):
        await session_identity_facade.resolve_conversation_identity(
            user_id="user-1",
            conversation_id=None,
            session_id="missing-session",
        )


@pytest.mark.asyncio
async def test_resolve_identity_rejects_unowned_conversation(monkeypatch) -> None:
    async def fake_get_conversation_or_none(user_id: str, conversation_id: str):
        assert user_id == "user-1"
        assert conversation_id == "conv-other"
        return None

    monkeypatch.setattr(
        session_identity_facade.conversation_facade,
        "get_conversation_or_none",
        fake_get_conversation_or_none,
    )

    with pytest.raises(LookupError, match="对话不存在"):
        await session_identity_facade.resolve_conversation_identity(
            user_id="user-1",
            conversation_id="conv-other",
            session_id=None,
        )


@pytest.mark.asyncio
async def test_resolve_identity_rejects_session_without_conversation_mapping(monkeypatch) -> None:
    async def fake_get_conversation_by_session_id_or_none(user_id: str, session_id: str):
        assert user_id == "user-1"
        assert session_id == "session-1"
        return None

    monkeypatch.setattr(
        session_identity_facade.conversation_facade,
        "get_conversation_by_session_id_or_none",
        fake_get_conversation_by_session_id_or_none,
    )
    _patch_session_repository(monkeypatch, {})

    with pytest.raises(LookupError, match="会话未关联对话"):
        await session_identity_facade.resolve_conversation_identity(
            user_id="user-1",
            conversation_id=None,
            session_id="session-1",
        )


@pytest.mark.asyncio
async def test_resolve_identity_can_explicitly_allow_session_without_conversation(monkeypatch) -> None:
    async def fake_get_conversation_by_session_id_or_none(user_id: str, session_id: str):
        assert user_id == "user-1"
        assert session_id == "session-1"
        return None

    monkeypatch.setattr(
        session_identity_facade.conversation_facade,
        "get_conversation_by_session_id_or_none",
        fake_get_conversation_by_session_id_or_none,
    )
    _patch_session_repository(monkeypatch, {})

    identity = await session_identity_facade.resolve_conversation_identity(
        user_id="user-1",
        conversation_id=None,
        session_id="session-1",
        allow_missing_conversation=True,
    )

    assert identity.conversation_id is None
    assert identity.session_id == "session-1"


def test_route_local_conversation_resolvers_are_retired() -> None:
    routes_dir = Path(__file__).parents[2] / "app" / "api" / "routes"
    pattern = re.compile(r"async\s+def\s+_resolve_conversation_id|def\s+_resolve_conversation_id")
    offenders = []
    for route_path in (routes_dir / name for name in ("chat_api.py", "research_api.py", "stream_api.py")):
        if pattern.search(route_path.read_text(encoding="utf-8")):
            offenders.append(route_path.name)

    assert offenders == []
