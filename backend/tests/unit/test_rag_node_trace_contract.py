from app.nodes import rag_node


def test_update_trace_generation_accepts_write_back_trace_fields(monkeypatch) -> None:
    calls: list[tuple[object, dict]] = []

    def fake_update_trace_generation(trace: object, **kwargs: object) -> None:
        calls.append((trace, kwargs))

    monkeypatch.setattr(
        rag_node.rag_trace_service,
        "update_trace_generation",
        fake_update_trace_generation,
    )
    trace = object()

    rag_node._update_trace_generation(
        trace,
        citations=[{"paper_id": "p1"}],
        selected_count=2,
        evidence_selection_stats={"source": "focused_text"},
        generation_status="failed",
        error_kind="generation_evidence_mismatch",
        reason="selected=2 citations=1",
        quality_score=0.7,
        iteration=1,
    )

    assert calls == [
        (
            trace,
            {
                "citations": [{"paper_id": "p1"}],
                "selected_count": 2,
                "evidence_selection_stats": {"source": "focused_text"},
                "generation_status": "failed",
                "error_kind": "generation_evidence_mismatch",
                "reason": "selected=2 citations=1",
                "quality_score": 0.7,
                "iteration": 1,
            },
        )
    ]
