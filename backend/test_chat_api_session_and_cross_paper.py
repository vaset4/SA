from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import app.api.routes.chat_api as chat_api
from app.api.schemas.research import CrossPaperQARequest, SessionCreateRequest
from app.services.research_session_lifecycle import ACTION_CANCEL, ResearchSessionNotFound


def _stub_cross_paper_dependencies(monkeypatch) -> None:
    """隔离同步跨论文 route 的身份、范围、历史与持久化依赖。

    参数:
        monkeypatch: pytest 提供的依赖替换工具
    返回:
        None
    """
    project_context = {
        "project_id": None,
        "resolved_scope": {"paper_ids": ["paper-1"]},
        "filter_expr": 'paper_id == "paper-1"',
        "paper_ids": ["paper-1"],
    }
    monkeypatch.setattr(chat_api, "_resolve_identity_or_404", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_api, "build_project_context", AsyncMock(return_value=project_context))
    monkeypatch.setattr(chat_api, "_previous_rag_context", AsyncMock(return_value={}))
    monkeypatch.setattr(chat_api, "_load_research_memory", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat_api, "_create_cross_paper_run", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_api, "_persist_completed_messages", AsyncMock(return_value=None))


def test_create_session_endpoint_returns_real_session_id(monkeypatch) -> None:
    lifecycle = AsyncMock()
    lifecycle.ensure_session.return_value = {
        "session_id": "session-for-新对话",
        "conversation_id": None,
        "thread_id": "session-for-新对话",
        "status": "created",
        "allowed_actions": [ACTION_CANCEL],
    }
    monkeypatch.setattr(chat_api, "get_research_session_lifecycle", lambda: lifecycle)

    payload = asyncio.run(chat_api.create_session(SessionCreateRequest(topic="新对话"), user={}))

    assert payload == {
        "session_id": "session-for-新对话",
        "conversation_id": None,
        "thread_id": "session-for-新对话",
        "status": "created",
        "allowed_actions": [ACTION_CANCEL],
    }


def test_cross_paper_qa_returns_fallback_answer_when_graph_finishes_empty(monkeypatch) -> None:
    _stub_cross_paper_dependencies(monkeypatch)

    async def fake_invoke_graph(query, extra):  # noqa: ANN001
        return (
            {
                "answer": "   ",
                "citations": [],
                "answer_confidence": None,
            },
            "session-1",
        )

    monkeypatch.setattr(chat_api, "invoke_graph", fake_invoke_graph)
    payload = asyncio.run(
        chat_api.cross_paper_qa(
            CrossPaperQARequest(query="当前论文的贡献是什么", paper_ids=["paper-1"], session_id="session-1"),
            user={},
        )
    )

    assert payload["session_id"] == "session-1"
    assert payload["answer"] == "当前论文中未找到可用于回答该问题的证据，请换个问法，或先确认论文已完成索引。"
    assert payload["filter_expr"] == 'paper_id == "paper-1"'


def test_cross_paper_qa_returns_internal_research_contract_fields(monkeypatch) -> None:
    _stub_cross_paper_dependencies(monkeypatch)

    async def fake_invoke_graph(query, extra):  # noqa: ANN001
        return (
            {
                "answer": "方法以图卷积为主。",
                "citations": [{"paper_id": "paper-1"}],
                "research_scope": {"mode": "single_paper", "summary": "聚焦单篇论文。"},
                "research_objects": [{"key": "methods", "label": "方法", "items": [{"label": "STGCN"}]}],
                "note_actions": [{"key": "copy", "type": "copy_note", "label": "复制结论", "note": "note"}],
            },
            "session-2",
        )

    monkeypatch.setattr(chat_api, "invoke_graph", fake_invoke_graph)
    payload = asyncio.run(
        chat_api.cross_paper_qa(
            CrossPaperQARequest(query="方法是什么", paper_ids=["paper-1"], session_id="session-2"),
            user={},
        )
    )

    assert payload["session_id"] == "session-2"
    assert payload["research_scope"]["mode"] == "single_paper"
    assert payload["research_objects"][0]["key"] == "methods"
    assert payload["note_actions"][0]["type"] == "copy_note"


def test_cross_paper_qa_maps_milvus_timeout_to_gateway_timeout(monkeypatch) -> None:
    _stub_cross_paper_dependencies(monkeypatch)

    async def fake_invoke_graph(query, extra):  # noqa: ANN001
        raise RuntimeError("milvus failed: Milvus query timed out after 12.0s")

    monkeypatch.setattr(chat_api, "invoke_graph", fake_invoke_graph)
    try:
        asyncio.run(
            chat_api.cross_paper_qa(
                CrossPaperQARequest(query="论文的贡献是什么", paper_ids=["paper-1"], session_id="session-1"),
                user={},
            )
        )
    except Exception as exc:  # noqa: BLE001
        assert getattr(exc, "status_code", None) == 504
        assert "cross-paper retrieval timed out" in getattr(exc, "detail", "")
    else:
        raise AssertionError("expected cross_paper_qa to raise HTTPException for milvus timeout")


def test_get_session_returns_empty_payload_for_created_session_without_checkpoint(monkeypatch) -> None:
    async def fake_get_shared_agent_state(  # noqa: ANN001, ARG001
        session_id,
        conversation_id=None,
        *,
        thread_id=None,
    ):
        return None

    session_row = {
        "session_id": "session-empty",
        "topic": "空会话",
        "thread_id": "session-empty",
        "status": "created",
        "paper_ids": [],
        "attempt_count": 0,
    }
    lifecycle = AsyncMock()
    lifecycle.get_session.return_value = session_row
    lifecycle.bind_legacy_thread.return_value = session_row
    monkeypatch.setattr(chat_api, "get_research_session_lifecycle", lambda: lifecycle)
    monkeypatch.setattr(chat_api, "get_shared_agent_state", fake_get_shared_agent_state)
    monkeypatch.setattr(chat_api, "_resolve_identity_or_404", AsyncMock(return_value=None))

    payload = asyncio.run(chat_api.get_session("session-empty"))

    assert payload["session_id"] == "session-empty"
    assert payload["topic"] == "空会话"
    assert payload["allowed_actions"] == [ACTION_CANCEL]
    assert payload["chat_history"] == []
    assert payload["values"]["chat_history"] == []


def test_get_session_still_raises_404_for_unknown_session(monkeypatch) -> None:
    lifecycle = AsyncMock()
    lifecycle.get_session.side_effect = ResearchSessionNotFound("会话不存在")
    monkeypatch.setattr(chat_api, "get_research_session_lifecycle", lambda: lifecycle)

    try:
        asyncio.run(chat_api.get_session("missing-session"))
    except Exception as exc:  # noqa: BLE001
        assert getattr(exc, "status_code", None) == 404
        assert getattr(exc, "detail", "") == "会话不存在"
    else:
        raise AssertionError("expected get_session to raise HTTPException for missing session")
