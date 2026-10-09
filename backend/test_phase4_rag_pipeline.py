from __future__ import annotations

import asyncio
import os

os.environ.pop("SSLKEYLOGFILE", None)

import app.nodes.rag_node as rag_node
import app.services.cache as cache_module
from types import SimpleNamespace


def _reset_cache_singleton() -> None:
    cache_module._cache = None


class FakeReranker:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls: list[list[dict]] = []

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        self.calls.append(documents)
        return self.results[:top_k]


def _mode(mode: str, graph_mode: str = "hybrid") -> SimpleNamespace:
    return SimpleNamespace(mode=mode, graph_mode=graph_mode, reasoning=f"{mode}-test")


def test_rag_search_node_filters_low_similarity_before_rerank(monkeypatch) -> None:
    _reset_cache_singleton()
    candidates = [
        {"chunk_id": "c1", "paper_id": "p1", "preview": "high", "score": 0.91},
        {"chunk_id": "c2", "paper_id": "p2", "preview": "low", "score": 0.11},
        {"chunk_id": "c3", "paper_id": "p3", "preview": "mid", "score": 0.45},
    ]
    fake_reranker = FakeReranker(
        [
            {**candidates[0], "rerank_score": 0.82},
            {**candidates[2], "rerank_score": 0.73},
        ]
    )

    monkeypatch.setattr(rag_node, "_run_rewritten_search", lambda vs, queries, filter_expr: candidates)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "get_reranker", lambda: fake_reranker, raising=False)
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("milvus")))
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", True, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.3)
    monkeypatch.setattr(rag_node, "_rerank_threshold", lambda: 0.5)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(rag_node.rag_search_node({"user_query": "rag", "context_summary": ""}))

    assert [item["chunk_id"] for item in fake_reranker.calls[0]] == ["c1", "c3"]
    assert [item["chunk_id"] for item in result["search_results"]] == ["c1", "c3"]


def test_rag_search_node_filters_low_rerank_scores(monkeypatch) -> None:
    _reset_cache_singleton()
    candidates = [
        {"chunk_id": "c1", "paper_id": "p1", "preview": "high", "score": 0.91},
        {"chunk_id": "c2", "paper_id": "p2", "preview": "mid", "score": 0.61},
    ]
    fake_reranker = FakeReranker(
        [
            {**candidates[0], "rerank_score": 0.84},
            {**candidates[1], "rerank_score": 0.32},
        ]
    )

    monkeypatch.setattr(rag_node, "_run_rewritten_search", lambda vs, queries, filter_expr: candidates)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "get_reranker", lambda: fake_reranker, raising=False)
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("milvus")))
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", True, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.3)
    monkeypatch.setattr(rag_node, "_rerank_threshold", lambda: 0.5)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(rag_node.rag_search_node({"user_query": "rag", "context_summary": ""}))

    assert [item["chunk_id"] for item in result["search_results"]] == ["c1"]
    assert result["rag_result"]["sources"] == ["p1"]
