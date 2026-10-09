from __future__ import annotations

import pytest

import app.api.facades.paper_facade as paper_facade
import app.api.routes.papers_api as papers_api
from app.api.schemas.papers import PaperMetadataUpdateRequest
from app.services import paper_metadata_service


def _paper(**overrides):
    return {
        "paper_id": "paper-1",
        "library_id": "default",
        "arxiv_id": None,
        "doi": None,
        "title": "Original title",
        "authors": ["Alice"],
        "abstract": "",
        "pdf_url": "",
        "pdf_path": "",
        "categories": [],
        "published_date": "",
        "source": "upload",
        "analysis": {"keywords": ["rag"]},
        "tags": [],
        **overrides,
    }


def test_get_paper_or_none_adds_view_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_facade.paper_db,
        "get_paper",
        lambda paper_id, user_id=None: _paper(
            authors=["Alice", "Bob"],
            categories=["LLM", "RAG"],
            published_date="2025-01-01",
        ),
    )

    result = paper_facade.get_paper_or_none("paper-1", user_id="tester")

    assert result["year"] == "2025"
    assert result["category"] == "LLM, RAG"
    assert result["keywords"] == ["rag"]
    assert result["authors"] == ["Alice", "Bob"]


def test_update_paper_metadata_or_raise_normalizes_fields(monkeypatch) -> None:
    state = {"paper": _paper()}
    captured: dict[str, object] = {}

    def fake_get_paper(paper_id, user_id=None):
        return state["paper"]

    def fake_update_paper_metadata(paper_id, patch, user_id=None):
        captured.update(patch)
        state["paper"] = {**state["paper"], **patch}
        return state["paper"]

    monkeypatch.setattr(paper_facade.paper_db, "get_paper", fake_get_paper)
    monkeypatch.setattr(paper_facade.paper_db, "update_paper_metadata", fake_update_paper_metadata)

    result = paper_facade.update_paper_metadata_or_raise(
        "paper-1",
        {
            "title": "New title",
            "authors": "Alice; Bob",
            "year": "2026",
            "category": "LLM, RAG",
            "abstract": "Updated abstract",
            "keywords": "graph neural network, traffic forecasting",
            "problem": "预测交通流量。",
            "method": "Graph WaveNet",
            "contribution": "贡献一；贡献二",
            "limitation": "暂无明确说明",
            "tldr": "一段简短总结",
            "methodology_type": "experimental",
            "methodology_detail": "使用公开交通数据集和多指标评估。",
        },
        user_id="tester",
    )

    assert captured["title"] == "New title"
    assert captured["authors"] == ["Alice", "Bob"]
    assert captured["published_date"] == "2026-01-01"
    assert captured["categories"] == ["LLM", "RAG"]
    assert captured["analysis"]["keywords"] == ["graph neural network", "traffic forecasting"]
    assert captured["analysis"]["method"] == "Graph WaveNet"
    assert captured["analysis"]["contribution"] == ["贡献一", "贡献二"]
    assert captured["analysis"]["methodology_type"] == "experimental"
    assert result["year"] == "2026"
    assert result["category"] == "LLM, RAG"
    assert result["analysis"]["problem"] == "预测交通流量。"
    assert result["keywords"] == ["graph neural network", "traffic forecasting"]


@pytest.mark.asyncio
async def test_autofill_route_returns_facade_payload(monkeypatch) -> None:
    async def fake_autofill(paper_id: str, *, user_id: str = "anonymous"):
        return {"task_id": "metadata-task", "status": "queued", "payload": {"paper_id": paper_id, "operation": "metadata"}}

    monkeypatch.setattr(papers_api.paper_facade, "autofill_paper_metadata_or_raise", fake_autofill)

    payload = await papers_api.autofill_paper_metadata("paper-1", user={"user_id": "tester"})

    assert payload["task_id"] == "metadata-task"
    assert payload["status"] == "queued"
    assert payload["payload"]["operation"] == "metadata"


@pytest.mark.asyncio
async def test_update_route_uses_patch_schema(monkeypatch) -> None:
    def fake_update(paper_id: str, payload: dict, *, user_id: str = "anonymous"):
        return {"paper_id": paper_id, **payload}

    monkeypatch.setattr(papers_api.paper_facade, "update_paper_metadata_or_raise", fake_update)

    payload = await papers_api.update_paper_metadata(
        "paper-1",
        PaperMetadataUpdateRequest(
            title="Edited title",
            year="2024",
            category="Machine Learning",
            keywords="rag, graph",
            problem="比较两篇论文的差异",
        ),
        user={"user_id": "tester"},
    )

    assert payload["title"] == "Edited title"
    assert payload["year"] == "2024"
    assert payload["category"] == "Machine Learning"
    assert payload["keywords"] == "rag, graph"
    assert payload["problem"] == "比较两篇论文的差异"


def test_build_autofill_metadata_patch_replaces_affiliation_like_abstract(monkeypatch) -> None:
    paper = _paper(
        abstract="1Centre for Artificial Intelligence, FEIT, University of Technology Sydney, Australia",
        pdf_path="demo.pdf",
    )

    monkeypatch.setattr(paper_metadata_service, "_resolve_autofill_text", lambda _paper: "full text")
    monkeypatch.setattr(
        paper_metadata_service.document_parse_service,
        "extract_document_metadata",
        lambda *args, **kwargs: {
            "title": "",
            "authors": [],
            "published_date": "",
            "abstract": "这是一段正确的论文摘要，说明研究问题、方法和结果。",
        },
    )
    monkeypatch.setattr(paper_metadata_service, "_infer_category", lambda *args, **kwargs: "")

    patch = paper_metadata_service.build_autofill_metadata_patch(paper)

    assert patch["abstract"] == "这是一段正确的论文摘要，说明研究问题、方法和结果。"


def test_normalize_analysis_payload_keeps_source_and_reference_fields() -> None:
    normalized = paper_metadata_service.normalize_analysis_payload({
        "problem": "问题",
        "method": "方法",
        "references": ["Ref A", "Ref B"],
        "source_markdown_path": "data/paper/source.md",
        "section_summaries": {"method": {"heading": "Method", "formula_blocks": ["$x+y$"]}},
    })

    assert normalized["references"] == ["Ref A", "Ref B"]
    assert normalized["source_markdown_path"] == "data/paper/source.md"
    assert "section_summaries" in normalized
