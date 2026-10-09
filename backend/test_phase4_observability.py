from __future__ import annotations

import asyncio

import app.middleware.observability as observability
import app.services.paper_db as paper_db


def test_rag_trace_to_dict_collects_metrics(monkeypatch) -> None:
    points = iter([1.0, 1.1, 1.3, 1.5, 1.9, 2.4])
    monkeypatch.setattr(observability.time, "monotonic", lambda: next(points))

    trace = observability.RAGTrace(session_id="session-1", query="what is self-rag")
    trace.rewritten_queries = ["self rag", "self-rag paper"]
    trace.retrieval["hybrid_results"] = 5
    trace.retrieval["reranked"] = 3
    trace.self_rag_iterations = 1
    trace.final_quality_score = 0.87
    trace.citations_count = 2

    trace.checkpoint("retrieval_start")
    trace.checkpoint("retrieval_end")
    trace.checkpoint("generation_start")
    trace.checkpoint("generation_end")

    payload = trace.to_dict()

    assert payload["request_id"] == trace.request_id
    assert payload["session_id"] == "session-1"
    assert payload["query"] == "what is self-rag"
    assert payload["rewritten_queries"] == ["self rag", "self-rag paper"]
    assert payload["retrieval"] == {
        "hybrid_results": 5,
        "reranked": 3,
        "graded_relevant": 0,
        "graded_irrelevant": 0,
    }
    assert payload["generation"] == {
        "self_rag_iterations": 1,
        "final_quality_score": 0.87,
        "citations_count": 2,
    }
    assert payload["latency_ms"] == {
        "total": 1400,
        "retrieval": 199,
        "rerank": 0,
        "grading": 0,
        "generation": 399,
        "evaluation": 0,
    }


def test_save_rag_trace_delegates_to_paper_db(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_write_rag_trace(request_id: str, session_id: str, query: str, trace_data: dict) -> None:
        captured["request_id"] = request_id
        captured["session_id"] = session_id
        captured["query"] = query
        captured["trace_data"] = trace_data

    monkeypatch.setattr(observability.paper_db, "write_rag_trace", fake_write_rag_trace)

    trace = observability.RAGTrace(session_id="session-2", query="agent memory")

    asyncio.run(observability.save_rag_trace(trace))

    assert captured["request_id"] == trace.request_id
    assert captured["session_id"] == "session-2"
    assert captured["query"] == "agent memory"
    assert captured["trace_data"]["request_id"] == trace.request_id


def test_save_rag_trace_swallows_storage_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        observability.paper_db,
        "write_rag_trace",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db down")),
    )

    trace = observability.RAGTrace(session_id="session-3", query="fallback")

    asyncio.run(observability.save_rag_trace(trace))


def test_init_db_async_creates_rag_trace_schema(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def fake_execute(query: str, *args) -> None:  # noqa: ARG001
        captured["query"] = query

    monkeypatch.setattr(paper_db, "_execute", fake_execute)

    asyncio.run(paper_db._init_db_async())

    query = captured["query"]
    assert "CREATE TABLE IF NOT EXISTS rag_traces" in query
    assert "trace_data JSONB" in query
    assert "idx_traces_session" in query
    assert "idx_traces_created" in query


def test_write_rag_trace_swallows_database_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        paper_db,
        "init_db",
        lambda: (_ for _ in ()).throw(RuntimeError("init failed")),
    )

    paper_db.write_rag_trace(
        request_id="request-1",
        session_id="session-4",
        query="grounded generation",
        trace_data={"status": "ok"},
    )
