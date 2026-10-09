from __future__ import annotations

import asyncio
import os

os.environ.pop("SSLKEYLOGFILE", None)

import app.middleware.observability as observability
import app.nodes.rag_node as rag_node
import app.services.recommendation as recommendation_module
from app.models.structured_outputs import AnswerEvaluation
from types import SimpleNamespace


def _mode(mode: str, graph_mode: str = "hybrid") -> SimpleNamespace:
    return SimpleNamespace(mode=mode, graph_mode=graph_mode, reasoning=f"{mode}-trace")


def test_rag_search_node_creates_trace_and_records_retrieval(monkeypatch) -> None:
    results = [
        {"chunk_id": "c1", "paper_id": "p1", "preview": "alpha", "score": 0.91},
        {"chunk_id": "c2", "paper_id": "p2", "preview": "beta", "score": 0.61},
    ]

    monkeypatch.setattr(rag_node, "_run_rewritten_search", lambda vs, queries, filter_expr: results)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node.settings, "tracing_enabled", True, raising=False)
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("milvus")))
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(
        rag_node.rag_search_node({"user_query": "trace rag", "session_id": "session-1", "context_summary": ""})
    )

    trace = result["trace"]
    assert isinstance(trace, observability.RAGTrace)
    assert trace.session_id == "session-1"
    assert trace.query == "trace rag"
    assert trace.rewritten_queries == ["trace rag"]
    assert trace.retrieval["hybrid_results"] == 2
    assert trace.to_dict()["latency_ms"]["retrieval"] >= 0


def test_document_grading_and_evaluate_update_trace_metrics(monkeypatch) -> None:
    trace = observability.RAGTrace(session_id="session-2", query="quality")
    state = {
        "trace": trace,
        "user_query": "quality",
        "search_results": [{"chunk_id": "c1", "paper_id": "p1", "preview": "doc"}],
        "answer": "answer",
        "self_rag_iteration": 0,
    }

    async def fake_grade_documents(question: str, documents: list[dict]) -> tuple[list[dict], list[dict]]:
        return (documents[:1], [{"chunk_id": "c2", "paper_id": "p2", "preview": "other"}])

    async def fake_evaluate_answer(question: str, answer: str, evidence: str) -> AnswerEvaluation:
        return AnswerEvaluation(
            relevance="high",
            faithfulness="high",
            completeness="complete",
            quality_score=0.88,
            missing_info=[],
            has_hallucination=False,
        )

    monkeypatch.setattr("app.nodes.document_grading.grade_documents", fake_grade_documents)
    monkeypatch.setattr("app.nodes.self_rag.evaluate_answer", fake_evaluate_answer)
    monkeypatch.setattr("app.nodes.self_rag.should_refine", lambda evaluation, iteration: False)

    graded = asyncio.run(rag_node.document_grading_node(state))
    evaluated = asyncio.run(rag_node.evaluate_node({**state, **graded}))

    updated_trace = evaluated["trace"]
    assert updated_trace.retrieval["graded_relevant"] == 1
    assert updated_trace.retrieval["graded_irrelevant"] == 1
    assert updated_trace.final_quality_score == 0.88
    assert updated_trace.self_rag_iterations == 0


def test_write_back_persists_trace(monkeypatch) -> None:
    saved: dict[str, observability.RAGTrace] = {}
    trace = observability.RAGTrace(session_id="session-3", query="persist")

    monkeypatch.setattr(observability, "persist_rag_trace", lambda current: saved.setdefault("trace", current))
    monkeypatch.setattr(recommendation_module, "build_profile", lambda: {})
    monkeypatch.setattr(recommendation_module, "score_paper", lambda paper, profile: (0.1, "skip"))
    monkeypatch.setattr(recommendation_module, "push_notification", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_node.paper_db, "get_paper", lambda paper_id: None)

    result = rag_node.write_back_node(
        {
            "trace": trace,
            "rag_result": {"sources": [], "chunks": [], "scores": [], "is_sufficient": False},
            "evaluation": {"quality_score": 0.91},
            "citations": [{"paper_id": "p1", "chunk_id": "c1", "excerpt": "evidence"}],
            "self_rag_iteration": 1,
        }
    )

    assert result["trace"] is trace
    assert saved["trace"] is trace
    assert trace.citations_count == 1
    assert trace.final_quality_score == 0.91
    assert trace.self_rag_iterations == 1


def test_rag_search_node_skips_trace_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(rag_node, "_run_rewritten_search", lambda vs, queries, filter_expr: [])
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node.settings, "tracing_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("milvus")))
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(rag_node.rag_search_node({"user_query": "no trace", "context_summary": ""}))

    assert "trace" not in result
