from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi import HTTPException

os.environ["SSLKEYLOGFILE"] = ""

import app.api.routes.stream_api as stream_api_module
from app.api.schemas.research import ResearchStreamRequest


async def _values_stream(events):
    for event in events:
        yield event


async def _error_stream(message: str):
    raise RuntimeError(message)
    yield  # pragma: no cover


async def _collect_sse(response) -> list[dict]:
    events: list[dict] = []
    async for chunk in response.body_iterator:
        text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        for frame in text.split("\n\n"):
            if not frame.strip():
                continue
            for line in frame.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    return events


def _patch_context(monkeypatch, *, session_exists: bool = True):
    monkeypatch.setattr(
        stream_api_module.paper_db,
        "get_session",
        lambda session_id, user_id=None: {"session_id": session_id} if session_exists else None,
    )
    async def fake_get_conversation_by_session_id_or_none(user_id, session_id):  # noqa: ANN001
        return None

    monkeypatch.setattr(
        stream_api_module.conversation_facade,
        "get_conversation_by_session_id_or_none",
        fake_get_conversation_by_session_id_or_none,
    )

    async def fake_build_project_context(project_id, requested_paper_ids=None, user_id="anonymous"):
        return {
            "project_id": project_id or "",
            "resolved_scope": {"paper_ids": requested_paper_ids or ["paper-1"]} if (project_id or requested_paper_ids) else None,
            "filter_expr": 'paper_id == "paper-1"' if project_id else "",
            "paper_ids": requested_paper_ids or (["paper-1"] if project_id else []),
        }

    monkeypatch.setattr(stream_api_module, "build_project_context", fake_build_project_context)


@pytest.mark.asyncio
async def test_stream_sse_emits_report_deltas(monkeypatch):
    _patch_context(monkeypatch)

    async def fake_open_graph_stream(query: str, extra: dict | None = None):
        assert query == "研究 streaming"
        return (
            "values",
            _values_stream(
                [
                    {"user_query": query, "intent": None, "report": None, "session_id": "sess-1"},
                    {"user_query": query, "intent": "deep_research", "report": "Hello", "session_id": "sess-1"},
                    {"user_query": query, "intent": "deep_research", "report": "Hello world", "session_id": "sess-1"},
                ]
            ),
            "sess-1",
        )

    monkeypatch.setattr(stream_api_module, "open_graph_stream", fake_open_graph_stream)

    response = await stream_api_module.stream(
        ResearchStreamRequest(query="研究 streaming", session_id="sess-1"),
        user={"sub": "user-1", "user_id": "user-1", "role": "user"},
    )
    events = await _collect_sse(response)

    assert events == [
        {"type": "event", "stage": "session", "session_id": "sess-1"},
        {"type": "token", "content": "Hello"},
        {"type": "token", "content": " world"},
        {
            "type": "done",
            "intent": "deep_research",
            "session_id": "sess-1",
            "research_mode": None,
            "key_findings": [],
            "uncertainties": [],
            "idea_candidates": [],
            "external_gaps": [],
            "follow_up_questions": [],
            "candidate_papers_to_import": [],
            "evidence_count": 0,
        },
    ]


@pytest.mark.asyncio
async def test_stream_sse_emits_external_research_structures_in_done(monkeypatch):
    _patch_context(monkeypatch)
    candidate_paper = {
        "title": "LangGraph interrupt docs",
        "url": "https://example.com/langgraph-interrupt",
        "reason_to_import": "适合作为后续内部研究的补库入口。",
    }

    async def fake_open_graph_stream(_query: str, _extra: dict | None = None):
        return (
            "values",
            _values_stream(
                [
                    {
                        "intent": "deep_research",
                        "report": "# Final report",
                        "session_id": "sess-structured",
                        "research_mode": "external_only",
                        "key_findings": ["外部资料已覆盖 interrupt / resume 主路径。"],
                        "uncertainties": ["仍需补充生产环境恢复案例。"],
                        "idea_candidates": ["把 checkpoint 恢复和前端 session 语义统一验证。"],
                        "external_gaps": ["缺少生产环境 session 恢复的外部案例对照。"],
                        "follow_up_questions": ["哪些资料值得补库后回到内部研究继续比对？"],
                        "candidate_papers_to_import": [candidate_paper],
                        "evidence_cards": [{"card_id": "ev-1"}],
                    },
                ]
            ),
            "sess-structured",
        )

    monkeypatch.setattr(stream_api_module, "open_graph_stream", fake_open_graph_stream)

    response = await stream_api_module.stream(
        ResearchStreamRequest(query="研究 structured done", session_id="sess-structured"),
        user={"sub": "user-1", "user_id": "user-1", "role": "user"},
    )
    events = await _collect_sse(response)

    assert events == [
        {"type": "event", "stage": "session", "session_id": "sess-structured"},
        {"type": "token", "content": "# Final report"},
        {
            "type": "done",
            "intent": "deep_research",
            "session_id": "sess-structured",
            "research_mode": "external_only",
            "key_findings": ["外部资料已覆盖 interrupt / resume 主路径。"],
            "uncertainties": ["仍需补充生产环境恢复案例。"],
            "idea_candidates": ["把 checkpoint 恢复和前端 session 语义统一验证。"],
            "external_gaps": ["缺少生产环境 session 恢复的外部案例对照。"],
            "follow_up_questions": ["哪些资料值得补库后回到内部研究继续比对？"],
            "candidate_papers_to_import": [candidate_paper],
            "evidence_count": 1,
        },
    ]


@pytest.mark.asyncio
async def test_stream_sse_emits_clarification_status(monkeypatch):
    _patch_context(monkeypatch)
    question = "请先说明你更关心 API 用法还是工程接入。"

    async def fake_open_graph_stream(_query: str, _extra: dict | None = None):
        return (
            "values",
            _values_stream(
                [
                    {
                        "intent": "deep_research",
                        "__interrupt__": [
                            {
                                "value": {
                                    "kind": "deep_research_clarification",
                                    "question": question,
                                    "report": question,
                                }
                            }
                        ],
                        "session_id": "sess-clarify",
                    },
                ]
            ),
            "sess-clarify",
        )

    monkeypatch.setattr(stream_api_module, "open_graph_stream", fake_open_graph_stream)

    response = await stream_api_module.stream(
        ResearchStreamRequest(query="研究 clarify", session_id="sess-clarify"),
        user={"sub": "user-1", "user_id": "user-1", "role": "user"},
    )
    events = await _collect_sse(response)

    assert events == [
        {"type": "event", "stage": "session", "session_id": "sess-clarify"},
        {"type": "token", "content": question},
        {
            "type": "done",
            "intent": "deep_research",
            "session_id": "sess-clarify",
            "research_mode": None,
            "key_findings": [],
            "uncertainties": [],
            "idea_candidates": [],
            "external_gaps": [],
            "follow_up_questions": [],
            "candidate_papers_to_import": [],
            "evidence_count": 0,
            "awaiting_clarification": True,
            "clarification_question": question,
        },
    ]


@pytest.mark.asyncio
async def test_stream_sse_surfaces_stream_error(monkeypatch):
    _patch_context(monkeypatch)

    async def fake_open_graph_stream(_query: str, _extra: dict | None = None):
        return ("values", _error_stream("stream exploded"), "sess-error")

    monkeypatch.setattr(stream_api_module, "open_graph_stream", fake_open_graph_stream)

    response = await stream_api_module.stream(
        ResearchStreamRequest(query="研究 error"),
        user={"sub": "user-1", "user_id": "user-1", "role": "user"},
    )
    events = await _collect_sse(response)

    assert events == [
        {"type": "event", "stage": "session", "session_id": "sess-error"},
        {"type": "error", "detail": "stream exploded"},
    ]


@pytest.mark.asyncio
async def test_stream_sse_rejects_missing_session(monkeypatch):
    _patch_context(monkeypatch, session_exists=False)

    with pytest.raises(HTTPException) as exc_info:
        await stream_api_module.stream(
            ResearchStreamRequest(query="研究 missing", session_id="missing-session"),
            user={"sub": "user-1", "user_id": "user-1", "role": "user"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "会话不存在"


@pytest.mark.asyncio
async def test_stream_sse_passes_project_scope_and_intent(monkeypatch):
    _patch_context(monkeypatch)
    captured: dict = {}

    async def fake_open_graph_stream(query: str, extra: dict | None = None):
        captured["query"] = query
        captured["extra"] = extra
        return ("values", _values_stream([]), "sess-intent")

    monkeypatch.setattr(stream_api_module, "open_graph_stream", fake_open_graph_stream)

    response = await stream_api_module.stream(
        ResearchStreamRequest(
            query="研究 preset intent",
            session_id="sess-intent",
            project_id="proj-1",
            research_mode="paper_first",
            paper_ids=["paper-a", "paper-b"],
            intent="deep_research",
        ),
        user={"sub": "user-1", "user_id": "user-1", "role": "user"},
    )
    await _collect_sse(response)

    assert captured["query"] == "研究 preset intent"
    assert captured["extra"] == {
        "intent": "deep_research",
        "session_id": "sess-intent",
        "conversation_id": None,
        "filter_expr": 'paper_id == "paper-1"',
        "structured_query": {
            "project_id": "proj-1",
            "resolved_scope": {"paper_ids": ["paper-a", "paper-b"]},
            "research_mode": "external_only",
            "paper_ids": ["paper-a", "paper-b"],
        },
    }
