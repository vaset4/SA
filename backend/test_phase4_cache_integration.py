from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

os.environ.pop("SSLKEYLOGFILE", None)

import app.nodes.rag_node as rag_node
import app.services.cache as cache_module


def _reset_cache_singleton() -> None:
    cache_module._cache = None


def _mode(mode: str, graph_mode: str = "hybrid") -> SimpleNamespace:
    return SimpleNamespace(mode=mode, graph_mode=graph_mode, reasoning=f"{mode}-test")


def test_rag_search_node_reuses_cached_search_results(monkeypatch) -> None:
    _reset_cache_singleton()
    search_calls = {"count": 0}
    results = [{"chunk_id": "c1", "paper_id": "p1", "preview": "cached", "score": 0.91}]

    def fake_search(vs, queries, filter_expr):  # noqa: ANN001
        search_calls["count"] += 1
        return results

    monkeypatch.setattr(rag_node, "_run_rewritten_search", fake_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("milvus")))
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    state = {"user_query": "rag cache", "context_summary": ""}

    first = asyncio.run(rag_node.rag_search_node(state))
    second = asyncio.run(rag_node.rag_search_node(state))

    assert search_calls["count"] == 1
    assert first["search_results"] == second["search_results"] == results
    stats = cache_module.get_cache().get_stats()
    assert stats["search_hits"] == 1
    assert stats["search_misses"] == 1


def test_rag_search_node_distinguishes_filter_expr_cache_keys(monkeypatch) -> None:
    _reset_cache_singleton()
    search_calls = {"count": 0}

    def fake_search(vs, queries, filter_expr):  # noqa: ANN001
        search_calls["count"] += 1
        return [{"chunk_id": filter_expr or "all", "paper_id": "p1", "preview": filter_expr or "all", "score": 0.9}]

    monkeypatch.setattr(rag_node, "_run_rewritten_search", fake_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("milvus")))
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    base_state = {"user_query": "rag cache", "context_summary": ""}

    first = asyncio.run(rag_node.rag_search_node({**base_state, "filter_expr": ""}))
    second = asyncio.run(rag_node.rag_search_node({**base_state, "filter_expr": "year > 2024"}))
    third = asyncio.run(rag_node.rag_search_node({**base_state, "filter_expr": "year > 2024"}))

    assert search_calls["count"] == 2
    assert first["search_results"][0]["chunk_id"] == "all"
    assert second["search_results"][0]["chunk_id"] == "year > 2024"
    assert third["search_results"] == second["search_results"]


def test_rag_search_node_skips_cache_when_disabled(monkeypatch) -> None:
    _reset_cache_singleton()
    search_calls = {"count": 0}

    def fake_search(vs, queries, filter_expr):  # noqa: ANN001
        search_calls["count"] += 1
        return [{"chunk_id": "c1", "paper_id": "p1", "preview": "no-cache", "score": 0.9}]

    monkeypatch.setattr(rag_node, "_run_rewritten_search", fake_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("milvus")))
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    state = {"user_query": "rag cache", "context_summary": ""}

    asyncio.run(rag_node.rag_search_node(state))
    asyncio.run(rag_node.rag_search_node(state))

    assert search_calls["count"] == 2


def test_rag_search_node_reuses_cached_graph_results(monkeypatch) -> None:
    _reset_cache_singleton()
    graph_calls = {"count": 0}

    class FakeGraphService:
        async def query(self, query: str, mode: str, top_k: int) -> str:
            graph_calls["count"] += 1
            return f"graph:{query}:{mode}:{top_k}"

    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("graph", "global")))
    monkeypatch.setattr(rag_node, "get_lightrag_service", lambda: FakeGraphService())
    monkeypatch.setattr(rag_node.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 3, raising=False)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    state = {"user_query": "knowledge graph", "context_summary": ""}

    first = asyncio.run(rag_node.rag_search_node(state))
    second = asyncio.run(rag_node.rag_search_node(state))

    assert graph_calls["count"] == 1
    assert first["search_results"] == second["search_results"] == []
    assert first["graph_context"] == second["graph_context"] == "graph:knowledge graph:global:3"
    assert first["source_status"] == {"milvus": "skipped", "graph": "ok"}


def test_rag_search_node_fusion_keeps_graph_context_outside_search_results(monkeypatch) -> None:
    _reset_cache_singleton()
    search_calls = {"count": 0}
    graph_calls = {"count": 0}
    vector_results = [{"chunk_id": "c1", "paper_id": "p1", "preview": "vector", "score": 0.88}]

    def fake_search(vs, queries, filter_expr):  # noqa: ANN001
        search_calls["count"] += 1
        return vector_results

    class FakeGraphService:
        async def query(self, query: str, mode: str, top_k: int) -> str:
            graph_calls["count"] += 1
            return "graph summary"

    monkeypatch.setattr(rag_node, "_run_rewritten_search", fake_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "get_lightrag_service", lambda: FakeGraphService())
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("fusion", "hybrid")))
    monkeypatch.setattr(rag_node.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    state = {"user_query": "fusion query", "context_summary": ""}

    first = asyncio.run(rag_node.rag_search_node(state))
    second = asyncio.run(rag_node.rag_search_node(state))

    assert search_calls["count"] == 1
    assert graph_calls["count"] == 1
    assert first["search_results"] == second["search_results"] == vector_results
    assert all(item["paper_id"] != "lightrag" for item in first["search_results"])
    assert first["graph_context"] == second["graph_context"] == "graph summary"
    assert first["source_status"] == {"milvus": "ok", "graph": "ok"}


def test_rag_search_node_exposes_partial_failure_status(monkeypatch) -> None:
    _reset_cache_singleton()

    def fake_search(vs, queries, filter_expr):  # noqa: ANN001
        raise RuntimeError("milvus down")

    class FakeGraphService:
        async def query(self, query: str, mode: str, top_k: int) -> str:
            return "graph survives"

    monkeypatch.setattr(rag_node, "_run_rewritten_search", fake_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "get_lightrag_service", lambda: FakeGraphService())
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("fusion", "hybrid")))
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(rag_node.rag_search_node({"user_query": "partial failure", "context_summary": ""}))

    assert result["search_results"] == []
    assert result["graph_context"] == "graph survives"
    assert result["source_status"] == {"milvus": "failed", "graph": "ok"}
    assert "milvus down" in result["failure_reason"]


def test_rag_search_node_graph_timeout_does_not_block_fusion(monkeypatch) -> None:
    _reset_cache_singleton()
    vector_results = [{"chunk_id": "c1", "paper_id": "p1", "preview": "vector survives", "score": 0.88}]

    def fake_search(vs, queries, filter_expr):  # noqa: ANN001
        return vector_results

    class SlowGraphService:
        async def query(self, query: str, mode: str, top_k: int) -> str:
            await asyncio.sleep(0.05)
            return "late graph result"

    monkeypatch.setattr(rag_node, "_run_rewritten_search", fake_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "get_lightrag_service", lambda: SlowGraphService())
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("fusion", "hybrid")))
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_graph_query_timeout_seconds", 0.01, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(rag_node.rag_search_node({"user_query": "当前论文的贡献是什么", "context_summary": ""}))

    assert result["search_results"] == vector_results
    assert result["graph_context"] == ""
    assert result["source_status"] == {"milvus": "ok", "graph": "failed"}
    assert "timed out" in result["failure_reason"]


def test_rag_search_node_vector_timeout_does_not_block_fusion(monkeypatch) -> None:
    _reset_cache_singleton()

    def slow_search(vs, queries, filter_expr):  # noqa: ANN001
        time.sleep(0.05)
        return [{"chunk_id": "late", "paper_id": "p1", "preview": "late vector", "score": 0.88}]

    class FastGraphService:
        async def query(self, query: str, mode: str, top_k: int) -> str:
            return "graph survives"

    monkeypatch.setattr(rag_node, "_run_rewritten_search", slow_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "get_lightrag_service", lambda: FastGraphService())
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("fusion", "hybrid")))
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_vector_query_timeout_seconds", 0.01, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(rag_node.rag_search_node({"user_query": "当前论文的贡献是什么", "context_summary": ""}))

    assert result["search_results"] == []
    assert result["graph_context"] == "graph survives"
    assert result["source_status"] == {"milvus": "failed", "graph": "ok"}
    assert "Milvus query timed out" in result["failure_reason"]


def test_rag_search_node_forces_milvus_when_paper_scope_is_constrained(monkeypatch) -> None:
    _reset_cache_singleton()
    vector_calls = {"count": 0}
    graph_calls = {"count": 0}
    vector_results = [{"chunk_id": "c1", "paper_id": "paper-1", "preview": "scoped hit", "score": 0.92}]

    def fake_search(vs, queries, filter_expr):  # noqa: ANN001
        vector_calls["count"] += 1
        assert filter_expr == 'paper_id == "paper-1"'
        return vector_results

    class FakeGraphService:
        async def query(self, query: str, mode: str, top_k: int) -> str:
            graph_calls["count"] += 1
            return "graph result should not be used"

    monkeypatch.setattr(rag_node, "_run_rewritten_search", fake_search)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "get_lightrag_service", lambda: FakeGraphService())
    monkeypatch.setattr(rag_node, "route_retrieval", lambda query: asyncio.sleep(0, result=_mode("graph", "local")))
    monkeypatch.setattr(rag_node.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_filter_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "query_rewrite_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rerank_enabled", False, raising=False)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 5, raising=False)
    monkeypatch.setattr(rag_node, "_similarity_threshold", lambda: 0.0)
    monkeypatch.setattr(rag_node.paper_db, "write_log", lambda **kwargs: None)

    result = asyncio.run(
        rag_node.rag_search_node(
            {
                "user_query": "Chain-of-Thought是什么",
                "context_summary": "",
                "filter_expr": 'paper_id == "paper-1"',
                "structured_query": {"paper_ids": ["paper-1"]},
            }
        )
    )

    assert vector_calls["count"] == 1
    assert graph_calls["count"] == 0
    assert result["search_results"] == vector_results
    assert result["graph_context"] == ""
    assert result["source_status"] == {"milvus": "ok", "graph": "skipped"}
