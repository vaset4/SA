from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

os.environ["SSLKEYLOGFILE"] = ""

from app.api.server import app
import app.runtime.lifespan as runtime_lifespan
import app.services.paper_db as paper_db


def _make_paper(
    paper_id: str,
    *,
    title: str,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    return {
        "paper_id": paper_id,
        "arxiv_id": paper_id,
        "doi": None,
        "title": title,
        "authors": ["Author A"],
        "abstract": f"Abstract for {title}",
        "pdf_url": "",
        "pdf_path": None,
        "categories": categories or [],
        "published_date": "2026-03-08",
        "source": "test",
        "analysis": None,
        "tags": tags or [],
    }


def _set_added_at(db_file: str, paper_id: str, added_at: str) -> None:
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            "UPDATE papers SET added_at=? WHERE paper_id=?",
            (added_at, paper_id),
        )
        connection.commit()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    class NoopRuntime:
        async def initialize(self):
            return None

        async def close(self) -> None:
            return None

    noop_runtime = NoopRuntime()
    monkeypatch.setattr(runtime_lifespan, "get_agent_runtime", lambda: noop_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_db_runtime", lambda: noop_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_conversation_runtime", lambda: noop_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_runtime_container", lambda: object())
    monkeypatch.setattr(paper_db.settings, "data_dir", str(tmp_path))
    paper_db.init_db()
    with TestClient(app) as test_client:
        yield test_client


def test_list_papers_endpoint_returns_database_total_not_page_size(client):
    paper_db.add_paper(_make_paper("paper_001", title="Paper 1", categories=["cs.AI"]))
    paper_db.add_paper(_make_paper("paper_002", title="Paper 2", categories=["cs.AI"]))
    paper_db.add_paper(_make_paper("paper_003", title="Paper 3", categories=["cs.LG"]))

    response = client.get("/papers", params={"limit": 1, "offset": 0})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["papers"]) == 1
    assert payload["total"] == 3


def test_list_papers_filters_before_applying_pagination(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_db.settings, "data_dir", str(tmp_path))
    paper_db.init_db()

    paper_db.add_paper(_make_paper("paper_old_ai", title="AI Paper", categories=["cs.AI"]))
    paper_db.add_paper(_make_paper("paper_new_cv", title="CV Paper", categories=["cs.CV"]))

    db_file = str(tmp_path / "papers.db")
    _set_added_at(db_file, "paper_old_ai", "2026-03-08 00:00:00")
    _set_added_at(db_file, "paper_new_cv", "2026-03-08 00:00:01")

    papers = paper_db.list_papers(categories=["cs.AI"], limit=1, offset=0)

    assert [paper["paper_id"] for paper in papers] == ["paper_old_ai"]
