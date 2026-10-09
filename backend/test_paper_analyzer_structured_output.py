from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app.models.structured_outputs import PaperAnalysisOutput


def _load_paper_analyzer_module():
    module_path = Path(__file__).resolve().parent / "app" / "nodes" / "paper_analyzer.py"
    spec = importlib.util.spec_from_file_location("test_paper_analyzer_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper_analyzer_module = _load_paper_analyzer_module()


class FakeLLM:
    def __init__(self, payload: PaperAnalysisOutput | Exception) -> None:
        self.payload = payload
        self.calls = 0

    def with_structured_output(self, schema):  # noqa: ANN001
        assert schema is PaperAnalysisOutput
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _build_paper():
    return {
        "paper_id": "paper-1",
        "arxiv_id": "1234.5678",
        "doi": None,
        "title": "Agentic Retrieval for Research",
        "authors": ["Alice", "Bob"],
        "abstract": "This paper studies agentic retrieval and structured academic analysis.",
        "pdf_url": "",
        "pdf_path": None,
        "categories": ["cs.AI"],
        "published_date": "2026-01-01",
        "source": "arxiv",
        "analysis": None,
        "tags": [],
    }


def test_analyze_node_reads_markdown_json_block(monkeypatch) -> None:
    fake_llm = FakeLLM(
        PaperAnalysisOutput(
            problem="Improve retrieval planning.",
            method="Agentic retrieval pipeline.",
            contribution=["Better planning", "Structured analysis"],
            limitation="Needs more benchmarks.",
            keywords=["agent", "retrieval", "planning"],
            related_work=["GraphRAG", "LinearRAG"],
            tldr="Agentic retrieval improves planning quality for academic QA.",
            methodology_type="experimental",
            methodology_detail="Evaluates planning quality and answer quality on retrieval benchmarks.",
        )
    )
    monkeypatch.setattr(paper_analyzer_module, "llm", fake_llm)

    result = paper_analyzer_module.analyze_node(
        {"paper": _build_paper(), "full_text": None, "analysis": None},
    )

    assert result["analysis"]["problem"] == "Improve retrieval planning."
    assert result["analysis"]["method"] == "Agentic retrieval pipeline."
    assert result["analysis"]["keywords"] == ["agent", "retrieval", "planning"]
    assert result["analysis"]["tldr"] == "Agentic retrieval improves planning quality for academic QA."
    assert result["analysis"]["methodology_type"] == "experimental"
    assert result["paper"]["analysis"] == result["analysis"]


def test_analyze_node_raises_when_schema_validation_fails(monkeypatch) -> None:
    fake_llm = FakeLLM(ValueError("schema validation failed"))
    monkeypatch.setattr(paper_analyzer_module, "llm", fake_llm)

    with pytest.raises(ValueError, match="schema validation failed"):
        paper_analyzer_module.analyze_node(
            {"paper": _build_paper(), "full_text": None, "analysis": None},
        )


def test_analyze_node_skips_llm_when_analysis_exists(monkeypatch) -> None:
    fake_llm = FakeLLM(
        PaperAnalysisOutput(
            problem="unused",
            method="unused",
            contribution=[],
            limitation="unused",
            keywords=[],
            related_work=[],
            tldr="unused",
            methodology_type="survey",
            methodology_detail="unused",
        )
    )
    monkeypatch.setattr(paper_analyzer_module, "llm", fake_llm)

    existing_analysis = {
        "problem": "Known problem",
        "method": "Known method",
        "contribution": ["Known contribution"],
        "limitation": "Known limitation",
        "keywords": ["known"],
        "related_work": ["KnownWork"],
        "tldr": "Known summary",
        "methodology_type": "hybrid",
        "methodology_detail": "Known setup",
    }
    paper = {**_build_paper(), "analysis": existing_analysis}

    result = paper_analyzer_module.analyze_node(
        {"paper": paper, "full_text": None, "analysis": existing_analysis},
    )

    assert result["analysis"] == existing_analysis
    assert fake_llm.calls == 0


def test_analyze_node_force_analysis_ignores_existing_analysis(monkeypatch) -> None:
    fake_llm = FakeLLM(
        PaperAnalysisOutput(
            problem="Fresh problem",
            method="Fresh method",
            contribution=["Fresh contribution"],
            limitation="Fresh limitation",
            keywords=["fresh"],
            related_work=["FreshWork"],
            tldr="Fresh summary",
            methodology_type="experimental",
            methodology_detail="Fresh setup",
        )
    )
    monkeypatch.setattr(paper_analyzer_module, "llm", fake_llm)

    paper = {**_build_paper(), "analysis": {"problem": "Known problem"}}
    result = paper_analyzer_module.analyze_node(
        {"paper": paper, "full_text": None, "analysis": paper["analysis"], "force_analysis": True},
    )

    assert result["analysis"]["problem"] == "Fresh problem"
    assert fake_llm.calls == 1
