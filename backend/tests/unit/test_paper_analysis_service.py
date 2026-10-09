from __future__ import annotations

import pytest

import app.services.paper_analysis_service as paper_analysis_service
from app.services.paper_analysis_material_service import build_analysis_bundle


def test_build_analysis_bundle_keeps_method_figures_formula_and_references() -> None:
    bundle = build_analysis_bundle(
        paper_id="paper_demo",
        source_markdown=(
            "# Abstract\n"
            "We study retrieval.\n\n"
            "# Method\n"
            "We use $L = x+y$.\n\n"
            "# Experiments\n"
            "Results improve.\n\n"
            "# References\n"
            "[1] Demo Reference"
        ),
        content_list=[
            {"type": "image", "caption": "Figure 1", "img_path": "fig-1.png"},
            {"type": "formula", "latex": "$L = x+y$"},
        ],
        source_markdown_path="data/paper_demo/source.md",
    )

    assert "method" in bundle.sections
    assert bundle.sections["method"].formula_blocks == ["$L = x+y$"]
    assert bundle.figures[0]["caption"] == "Figure 1"
    assert bundle.references == ["[1] Demo Reference"]
    assert bundle.source_markdown_path == "data/paper_demo/source.md"


@pytest.mark.asyncio
async def test_analyze_paper_by_id_marks_failed_without_placeholder(monkeypatch) -> None:
    status_updates: list[dict[str, object]] = []

    monkeypatch.setattr(
        paper_analysis_service.paper_db,
        "get_paper",
        lambda paper_id, user_id="anonymous": {
            "paper_id": paper_id,
            "title": "Slow Paper",
            "abstract": "Existing abstract should not become fake analysis.",
            "source": "upload",
            "analysis": {"problem": "old fake summary"},
        },
    )
    monkeypatch.setattr(
        paper_analysis_service,
        "_read_mineru_artifacts",
        lambda paper_id: ("markdown body", [], "source.md"),
    )
    monkeypatch.setattr(
        paper_analysis_service,
        "_build_paper_analysis_llm",
        lambda: object(),
    )

    async def fail_analysis(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise TimeoutError("LLM request timed out after 180s")

    def record_status(paper_id, status, **kwargs):  # noqa: ANN001
        status_updates.append({"paper_id": paper_id, "status": status, **kwargs})

    monkeypatch.setattr(paper_analysis_service, "analyze_single_paper", fail_analysis)
    monkeypatch.setattr(paper_analysis_service.paper_db, "update_analysis_status", record_status)
    monkeypatch.setattr(paper_analysis_service.paper_db, "update_analysis", lambda *args, **kwargs: None)

    with pytest.raises(TimeoutError, match="180s"):
        await paper_analysis_service.analyze_paper_by_id("paper-1", user_id="user-1")

    assert status_updates == [
        {"paper_id": "paper-1", "status": "running", "clear_analysis": True, "user_id": "user-1"},
        {
            "paper_id": "paper-1",
            "status": "failed",
            "error": "LLM request timed out after 180s",
            "clear_analysis": True,
            "user_id": "user-1",
        },
    ]
