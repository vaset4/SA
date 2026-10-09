from __future__ import annotations

import os

os.environ["SSLKEYLOGFILE"] = ""

import app.api.facades.paper_facade as paper_facade
import app.api.facades.system_facade as system_facade
import app.services.library_service as library_service
import pytest


def test_list_papers_page_defaults_to_default_library(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_list_papers(*args, **kwargs):
        captured["list_library_id"] = kwargs.get("library_id")
        return []

    def fake_count_papers(*args, **kwargs):
        captured["count_library_id"] = kwargs.get("library_id")
        return 0

    monkeypatch.setattr(paper_facade.paper_db, "list_papers", fake_list_papers)
    monkeypatch.setattr(paper_facade.paper_db, "count_papers", fake_count_papers)

    payload = paper_facade.list_papers_page(limit=10, offset=0)

    assert payload["library_id"] == "default"
    assert captured["list_library_id"] == "default"
    assert captured["count_library_id"] == "default"


def test_search_papers_scopes_to_library(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMilvus:
        def hybrid_search(self, *, query: str, top_k: int, filter_expr: str):
            captured["query"] = query
            captured["top_k"] = top_k
            captured["filter_expr"] = filter_expr
            return [{"paper_id": "paper-1", "preview": "x", "score": 0.9, "chunk_id": "c1"}]

    monkeypatch.setattr(paper_facade.paper_db, "list_paper_ids", lambda library_id: ["paper-1", "paper-2"])
    monkeypatch.setattr(paper_facade, "get_milvus_store", lambda: FakeMilvus())

    payload = paper_facade.search_papers(
        query="rag",
        top_k=5,
        library_id="lib_scope",
        year_min=2024,
        year_max=None,
        category=None,
        authors=None,
    )

    assert payload["library_id"] == "lib_scope"
    assert "paper_id == \"paper-1\"" in str(captured["filter_expr"])
    assert "paper_id == \"paper-2\"" in str(captured["filter_expr"])
    assert "year >= 2024" in str(captured["filter_expr"])


@pytest.mark.asyncio
async def test_ingest_uploaded_paper_requires_library_and_forwards_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_require(library_id: str):
        captured["required"] = library_id
        return {"library_id": library_id}

    async def fake_add(filename: str, raw: bytes, library_id: str):
        captured["filename"] = filename
        captured["library_id"] = library_id
        return {"paper_id": "paper-1", "library_id": library_id}

    monkeypatch.setattr(library_service, "require_library", fake_require)
    monkeypatch.setattr(paper_facade, "add_uploaded_paper", fake_add)

    payload = await paper_facade.ingest_uploaded_paper("draft.pdf", b"body", "lib_ingest")

    assert captured["required"] == "lib_ingest"
    assert captured["filename"] == "draft.pdf"
    assert captured["library_id"] == "lib_ingest"
    assert payload["library_id"] == "lib_ingest"


def test_search_papers_returns_empty_when_library_has_no_papers(monkeypatch) -> None:
    monkeypatch.setattr(paper_facade.paper_db, "list_paper_ids", lambda library_id: [])

    payload = paper_facade.search_papers(
        query="rag",
        top_k=5,
        library_id="lib_empty",
        year_min=None,
        year_max=None,
        category=None,
        authors=None,
    )

    assert payload == {"results": [], "filter_expr": "", "library_id": "lib_empty"}


@pytest.mark.asyncio
async def test_graph_query_payload_forwards_library_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLightRAG:
        async def query(self, *, query: str, mode: str, top_k: int, library_id: str | None = None):
            captured["query"] = query
            captured["mode"] = mode
            captured["top_k"] = top_k
            captured["library_id"] = library_id
            return "graph-result"

    monkeypatch.setattr(system_facade, "get_lightrag_service", lambda: FakeLightRAG())

    payload = await system_facade.graph_query_payload("agents", "hybrid", 3, "lib_graph")

    assert payload["result"] == "graph-result"
    assert captured["library_id"] == "lib_graph"


def test_library_service_normalizes_default_library() -> None:
    assert library_service.normalize_library_id(None) == "default"
    assert library_service.normalize_library_id("") == "default"
    assert library_service.normalize_library_id("lib_x") == "lib_x"
