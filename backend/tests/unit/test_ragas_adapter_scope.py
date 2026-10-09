from __future__ import annotations

from evals.ragas import rag_adapter


def test_live_rag_adapter_initial_state_includes_source_status_reasons() -> None:
    adapter = rag_adapter.LiveRAGAdapter(mode="auto")

    payload = adapter._build_live_input(
        {
            "question": "Which prediction task does PDFormer target?",
            "paper_ids": ["paper-pdformer"],
        }
    )

    assert payload["source_status_reasons"] == {}


def test_live_rag_adapter_initial_state_includes_paper_titles_for_expansion() -> None:
    adapter = rag_adapter.LiveRAGAdapter(mode="auto")

    payload = adapter._build_live_input(
        {
            "question": "Summarize the paper in one sentence.",
            "paper_ids": ["paper-pdformer"],
            "paper_titles": ["PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer"],
        }
    )

    assert payload["structured_query"]["paper_ids"] == ["paper-pdformer"]
    assert payload["structured_query"]["paper_titles"] == [
        "PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer"
    ]


def test_live_rag_adapter_recovers_resolved_mode_from_trace() -> None:
    adapter = rag_adapter.LiveRAGAdapter(mode="auto")

    record = adapter._build_live_record(
        {"question": "Q", "reference_answer": "A"},
        {
            "answer": "A",
            "search_results": [],
            "source_status": {"milvus": "ok", "graph": "skipped"},
            "trace": {"mode": "milvus", "graph_mode": "hybrid"},
        },
    )

    assert record["metadata"]["resolved_mode"] == "milvus"
    assert record["metadata"]["retrieval_mode"] == "milvus"


def test_live_rag_adapter_records_scope_resolution_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        rag_adapter,
        "_resolve_live_paper_ids_from_titles",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("title did not resolve")),
    )

    adapter = rag_adapter.LiveRAGAdapter(mode="auto")
    resolved = adapter._resolve_sample_scope(
        {
            "question": "Q",
            "reference_answer": "A",
            "paper_titles": ["Missing Paper"],
        }
    )

    assert resolved["metadata"]["scope_resolution_warnings"] == ["title did not resolve"]


def test_live_rag_adapter_records_project_scope_mismatch_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        rag_adapter,
        "_resolve_live_project_context",
        lambda **kwargs: {
            "paper_ids": ["paper-b"],
            "resolved_scope": {"project_id": kwargs["project_id"], "paper_ids": ["paper-b"]},
        },
    )

    adapter = rag_adapter.LiveRAGAdapter(mode="auto")
    resolved = adapter._resolve_sample_scope(
        {
            "question": "Q",
            "reference_answer": "A",
            "project_id": "project-1",
            "paper_ids": ["paper-a"],
        }
    )

    warning_text = "\n".join(resolved["metadata"]["scope_resolution_warnings"])
    assert "requested_paper_ids" in warning_text
    assert "paper-a" in warning_text
    assert "paper-b" in warning_text


def test_resolve_live_paper_ids_from_titles_accepts_dict_papers(monkeypatch) -> None:
    from app.services import paper_db

    calls: list[tuple[str, str | None, str | None]] = []

    def fake_search_by_exact_title(
        title: str,
        library_id: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, str]]:
        calls.append((title, library_id, user_id))
        return [{"paper_id": "paper-by-title-1"}]

    monkeypatch.setattr(paper_db, "search_by_exact_title", fake_search_by_exact_title)

    result = rag_adapter._resolve_live_paper_ids_from_titles(
        ["MinerU Markdown Pipeline Paper"],
        library_id="default",
        user_id="anonymous",
    )

    assert result == ["paper-by-title-1"]
    assert calls == [("MinerU Markdown Pipeline Paper", "default", "anonymous")]


def test_resolve_live_paper_ids_from_titles_reports_dict_match_ids(monkeypatch) -> None:
    from app.services import paper_db

    def fake_search_by_exact_title(
        title: str,
        library_id: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, str]]:
        del title, library_id, user_id
        return [{"paper_id": "paper-a"}, {"paper_id": "paper-b"}]

    monkeypatch.setattr(paper_db, "search_by_exact_title", fake_search_by_exact_title)

    try:
        rag_adapter._resolve_live_paper_ids_from_titles(["Duplicated Title"])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ValueError for duplicated paper title matches")

    assert "paper-a" in message
    assert "paper-b" in message
