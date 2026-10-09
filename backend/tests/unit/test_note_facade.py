from __future__ import annotations

import pytest

from app.api.facades import paper_facade


def test_create_note_normalizes_extended_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_add_note(paper_id, content, user_id="anonymous", **kwargs):
        captured.update(
            {
                "paper_id": paper_id,
                "content": content,
                "user_id": user_id,
                **kwargs,
            }
        )
        return "note-1"

    monkeypatch.setattr(paper_facade.paper_db, "add_note", fake_add_note)

    payload = paper_facade.create_note(
        {
            "paper_id": "paper-1",
            "project_id": "proj-1",
            "conversation_id": "conv-1",
            "session_id": "sess-1",
            "content": "  记录结论  ",
            "note_type": "Comparison Note",
            "source_type": "local rag",
            "citation": "RAG, 2024",
            "excerpt": "model beats baseline",
            "evidence_reference": {"card_id": "card-1"},
        },
        user_id="user-1",
    )

    assert payload == {"note_id": "note-1"}
    assert captured == {
        "paper_id": "paper-1",
        "content": "记录结论",
        "user_id": "user-1",
        "project_id": "proj-1",
        "conversation_id": "conv-1",
        "session_id": "sess-1",
        "note_type": "comparison_note",
        "source_type": "local_rag",
        "citation": "RAG, 2024",
        "excerpt": "model beats baseline",
        "evidence_reference": {"card_id": "card-1"},
    }


def test_create_note_requires_anchor_object() -> None:
    with pytest.raises(ValueError, match="至少提供一个关联对象"):
        paper_facade.create_note({"content": "只有正文"})


def test_list_notes_normalizes_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_list_notes(**kwargs):
        captured.update(kwargs)
        return [{"note_id": "note-1"}]

    monkeypatch.setattr(paper_facade.paper_db, "list_notes", fake_list_notes)

    payload = paper_facade.list_notes(
        user_id="user-1",
        project_id="proj-1",
        note_type="Idea Note",
        source_type="Deep Research",
        include_project_sessions=True,
        limit=20,
    )

    assert payload == {"notes": [{"note_id": "note-1"}]}
    assert captured["user_id"] == "user-1"
    assert captured["project_id"] == "proj-1"
    assert captured["note_type"] == "idea_note"
    assert captured["source_type"] == "deep_research"
    assert captured["include_project_sessions"] is True
    assert captured["limit"] == 20


def test_create_research_note_requires_project_session_and_cited_papers() -> None:
    with pytest.raises(ValueError, match="正式研究笔记需要绑定"):
        paper_facade.create_note(
            {
                "project_id": "proj-1",
                "session_id": "sess-1",
                "content": "记录研究结论",
                "note_type": "research_note",
                "source_type": "internal_research",
            }
        )


def test_create_research_note_accepts_project_session_and_cited_papers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_add_note(paper_id, content, user_id="anonymous", **kwargs):
        captured.update({"paper_id": paper_id, "content": content, "user_id": user_id, **kwargs})
        return "note-research-1"

    monkeypatch.setattr(paper_facade.paper_db, "add_note", fake_add_note)

    payload = paper_facade.create_note(
        {
            "project_id": "proj-1",
            "session_id": "sess-1",
            "content": "记录研究结论",
            "note_type": "research_note",
            "source_type": "internal_research",
            "evidence_reference": {"paper_ids": ["paper-1", "paper-2"]},
        },
        user_id="user-1",
    )

    assert payload == {"note_id": "note-research-1"}
    assert captured["project_id"] == "proj-1"
    assert captured["session_id"] == "sess-1"
    assert captured["note_type"] == "research_note"
    assert captured["source_type"] == "internal_research"
