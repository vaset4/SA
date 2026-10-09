from __future__ import annotations

import asyncio
import os

os.environ.pop("SSLKEYLOGFILE", None)

from app.services.ingest import (
    document_analysis_service,
    document_graph_service,
    document_index_service,
    document_parse_service,
)
import app.services.lightrag_service as lightrag_service


def test_merge_parent_texts_dedupes_parent_blocks() -> None:
    chunks = [
        {"chunk_id": "c1", "parent_id": "p1", "text": "a", "full_text": "parent-1", "section": "", "page_num": 1},
        {"chunk_id": "c2", "parent_id": "p1", "text": "b", "full_text": "parent-1", "section": "", "page_num": 1},
        {"chunk_id": "c3", "parent_id": "p2", "text": "c", "full_text": "parent-2", "section": "", "page_num": 2},
    ]

    merged = document_parse_service.merge_parent_texts(chunks)

    assert merged == "parent-1\n\nparent-2"


def test_build_upload_paper_uses_upload_defaults() -> None:
    paper = document_analysis_service.build_upload_paper(
        "draft.pdf",
        {"file_path": "x.pdf", "chunks": [], "full_text": "body text", "file_type": "pdf"},
        "upload_1234",
    )

    assert paper["paper_id"] == "upload_1234"
    assert paper["title"] == "draft.pdf"
    assert paper["source"] == "upload"
    assert paper["abstract"] == "body text"


def test_build_upload_paper_prefers_extracted_metadata() -> None:
    paper = document_analysis_service.build_upload_paper(
        "draft.pdf",
        {
            "file_path": "x.pdf",
            "chunks": [],
            "full_text": "body text",
            "file_type": "pdf",
            "metadata": {
                "title": "Metadata Title",
                "authors": ["Alice", "Bob"],
                "published_date": "2024-03-30",
                "abstract": "Metadata abstract",
            },
        },
        "upload_meta",
    )

    assert paper["title"] == "Metadata Title"
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["published_date"] == "2024-03-30"
    assert paper["abstract"] == "Metadata abstract"


def test_build_upload_metadata_extracts_year_and_lists() -> None:
    metadata = document_index_service.build_upload_metadata(
        {
            "paper_id": "p1",
            "arxiv_id": None,
            "doi": None,
            "title": "Title",
            "authors": ["Alice", "Bob"],
            "abstract": "Abstract",
            "pdf_url": "",
            "pdf_path": "x.pdf",
            "categories": ["cs.AI", "cs.LG"],
            "published_date": "2024-01-02",
            "source": "upload",
            "analysis": None,
            "tags": [],
        },
        "pdf",
    )

    assert metadata == {
        "year": 2024,
        "authors": "Alice, Bob",
        "venue": "",
        "category": "cs.AI, cs.LG",
        "file_type": "pdf",
    }


def test_insert_to_lightrag_trims_and_uses_paper_id(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeService:
        async def insert(self, text: str, paper_id: str = "", *, library_id: str = "default") -> None:
            calls.append((text, paper_id, library_id))

    monkeypatch.setattr(lightrag_service, "get_lightrag_service", lambda: FakeService())

    oversized_text = " body text " + ("x" * (document_graph_service.LIGHTRAG_INSERT_CHAR_LIMIT + 32))
    asyncio.run(document_graph_service.insert_to_lightrag("paper_x", oversized_text))

    assert calls[0][1:] == ("paper_x", "default")
    assert calls[0][0].startswith("body text")
    assert len(calls[0][0]) <= document_graph_service.LIGHTRAG_INSERT_CHAR_LIMIT


def test_build_graph_source_text_prefers_high_value_sections_and_excludes_references() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "parent_id": "p1",
            "text": "intro",
            "full_text": "Introduction context",
            "section": "Introduction",
            "page_num": 1,
        },
        {
            "chunk_id": "c2",
            "parent_id": "p2",
            "text": "method",
            "full_text": "Method details and experiment setup",
            "section": "Method",
            "page_num": 2,
        },
        {
            "chunk_id": "c3",
            "parent_id": "p3",
            "text": "refs",
            "full_text": "[1] reference entry",
            "section": "References",
            "page_num": 9,
        },
        {
            "chunk_id": "c4",
            "parent_id": "p4",
            "text": "conclusion",
            "full_text": "Conclusion and limitations",
            "section": "Conclusion",
            "page_num": 10,
        },
    ]

    text, audit = document_graph_service.build_graph_source_text(
        "front matter only",
        chunks=chunks,
        metadata={"title": "MinerU Paper", "abstract": "Abstract summary"},
    )

    assert text.startswith("# MinerU Paper")
    assert "## Method" in text
    assert "## Conclusion" in text
    assert "reference entry" not in text.lower()
    assert "References" in audit["excluded_sections"]
