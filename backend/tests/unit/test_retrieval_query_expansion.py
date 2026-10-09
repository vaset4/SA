from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.services import retrieval_service
from app.services.retrieval_query_expansion import (
    expand_lightrag_locator_queries,
    expand_single_paper_summary_queries,
)


def test_expand_single_paper_summary_queries_adds_section_terms() -> None:
    queries = expand_single_paper_summary_queries(
        query="Within the bound paper scope, summarize PDFormer in one sentence.",
        queries=["Within the bound paper scope, summarize PDFormer in one sentence."],
        structured_query={
            "paper_ids": ["paper-pdformer"],
            "resolved_scope": {"paper_ids": ["paper-pdformer"]},
            "paper_titles": ["PDFormer"],
        },
    )

    assert queries[0] == "Within the bound paper scope, summarize PDFormer in one sentence."
    assert "Within the bound paper scope, summarize PDFormer in one sentence. abstract" in queries
    assert "Within the bound paper scope, summarize PDFormer in one sentence. conclusion" in queries
    assert "PDFormer abstract" in queries


def test_expand_single_paper_summary_queries_ignores_unscoped_summary() -> None:
    queries = expand_single_paper_summary_queries(
        query="summarize the field",
        queries=["summarize the field"],
        structured_query={},
    )

    assert queries == ["summarize the field"]


def test_lightrag_expansion_adds_locator_queries_without_evidence_records() -> None:
    provider = _LightRAGProvider("adaptive adjacency relation\n- temporal attention")
    result = asyncio.run(
        expand_lightrag_locator_queries(
            query="compare traffic forecasting methods",
            queries=["attention mechanism"],
            provider=provider,
            library_id="library-a",
            top_k=3,
        )
    )

    assert result.locator_queries == ["attention mechanism", "adaptive adjacency relation", "temporal attention"]
    assert result.evidence_records == []
    assert result.metadata["source"] == "lightrag"
    assert provider.calls == [{"query": "compare traffic forecasting methods", "library_id": "library-a", "top_k": 3}]


def test_execute_rag_search_expands_single_paper_summary_without_changing_filter() -> None:
    captured: dict[str, object] = {}

    async def fake_route_retrieval(query: str, *, structured_query=None):  # noqa: ANN001, ARG001
        return SimpleNamespace(mode="milvus", graph_mode="hybrid", reasoning="test")

    def fake_search(store, queries, filter_expr):  # noqa: ANN001
        del store
        captured["queries"] = list(queries)
        captured["filter_expr"] = filter_expr
        return [{"chunk_id": "c1", "paper_id": "paper-pdformer", "preview": "PDFormer abstract", "score": 0.9}]

    settings = SimpleNamespace(
        query_filter_enabled=False,
        query_rewrite_enabled=False,
        vector_rerank_enabled=False,
        rag_top_k=5,
        rag_vector_query_timeout_seconds=0,
        rag_graph_query_timeout_seconds=0,
    )

    result = asyncio.run(
        retrieval_service.execute_rag_search(
            {
                "user_query": "Within the bound paper scope, summarize PDFormer in one sentence.",
                "structured_query": {
                    "paper_ids": ["paper-pdformer"],
                    "resolved_scope": {"paper_ids": ["paper-pdformer"]},
                },
                "filter_expr": 'paper_id == "paper-pdformer"',
            },
            settings=settings,
            route_retrieval=fake_route_retrieval,
            get_cache=lambda: SimpleNamespace(get_search=lambda *args: None, set_search=lambda *args: None),
            get_vector_store=lambda: object(),
            get_lightrag_service=lambda: None,
            get_reranker=lambda: None,
            paper_db=SimpleNamespace(write_log=lambda *args, **kwargs: None, get_paper=lambda paper_id: {"paper_id": paper_id}),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            ensure_trace=lambda state: None,
            with_trace=lambda payload, trace: payload,
            run_rewritten_search_fn=fake_search,
            similarity_threshold_fn=lambda: 0.0,
            rerank_threshold_fn=lambda: 0.0,
        )
    )

    assert captured["filter_expr"] == 'paper_id == "paper-pdformer"'
    assert "abstract" in " ".join(captured["queries"])
    assert result["search_results"][0]["chunk_id"] == "c1"


class _LightRAGProvider:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def collect_library_graph_hint(self, query: str, mode: str, top_k: int, *, library_id: str | None = None) -> str:
        del mode
        self.calls.append({"query": query, "library_id": library_id, "top_k": top_k})
        return self.response
