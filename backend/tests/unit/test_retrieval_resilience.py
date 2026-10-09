from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest

from app.services import retrieval_service


class _FakeCache:
    def __init__(self) -> None:
        self.saved = None

    def get_search(self, cache_key: str, filter_expr: str, mode: str):  # noqa: ANN001
        return None

    def set_search(self, cache_key: str, filter_expr: str, payload: dict, mode: str) -> None:  # noqa: ANN001
        self.saved = (cache_key, filter_expr, payload, mode)


class _FakePaperDb:
    def __init__(self) -> None:
        self.logs: list[tuple] = []

    def write_log(self, *args, **kwargs):  # noqa: ANN002
        self.logs.append((args, kwargs))

    def get_paper(self, paper_id: str):  # noqa: ANN001
        return {"paper_id": paper_id}


def _trusted_scope(paper_ids: tuple[str, ...] = ("paper-1",)) -> dict:
    """构造测试用的已认证、已归属校验的本地检索范围。"""
    ids = list(paper_ids)
    filter_expr = " or ".join(f'paper_id == "{paper_id}"' for paper_id in ids)
    resolved_scope = {
        "project_id": "project-1",
        "library_ids": ["lib-1"],
        "paper_ids": ids,
        "excluded_paper_ids": [],
        "year_from": None,
        "year_to": None,
        "task_filters": [],
        "method_filters": [],
    }
    return {
        "user_id": "user-1",
        "project_id": "project-1",
        "library_ids": ["lib-1"],
        "paper_ids": ids,
        "filter_expr": filter_expr,
        "local_retrieval_allowed": True,
        "resolved_scope": resolved_scope,
    }


class _FailingGraphProvider:
    async def query(self, query: str, mode: str, top_k: int) -> str:
        raise RuntimeError("graph embedding unavailable")


def test_execute_rag_search_returns_failure_payload_when_all_sources_fail() -> None:
    async def fake_route_retrieval(query: str):  # noqa: ARG001
        return SimpleNamespace(mode="fusion", graph_mode="hybrid", reasoning="test")

    settings = SimpleNamespace(
        query_filter_enabled=False,
        query_rewrite_enabled=False,
        rerank_enabled=False,
        rag_top_k=5,
        rerank_top_k=5,
        rag_vector_query_timeout_seconds=0,
        rag_graph_query_timeout_seconds=0,
    )
    cache = _FakeCache()
    paper_db = _FakePaperDb()

    result = asyncio.run(
        retrieval_service.execute_rag_search(
            {
                "user_query": "stgcn解决了什么问题",
                "structured_query": {
                    "project_scope": _trusted_scope(),
                    "dialogue_mode": "research",
                },
            },
            settings=settings,
            route_retrieval=fake_route_retrieval,
            get_cache=lambda: cache,
            get_vector_store=lambda: object(),
            get_lightrag_service=lambda: _FailingGraphProvider(),
            get_reranker=lambda: None,
            paper_db=paper_db,
            logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            ensure_trace=lambda state: None,
            with_trace=lambda payload, trace: payload,
            run_rewritten_search_fn=lambda store, queries, filter_expr: (_ for _ in ()).throw(
                RuntimeError("embedding 模型不可用")
            ),
            similarity_threshold_fn=lambda: 0.0,
            rerank_threshold_fn=lambda: 0.0,
        )
    )

    assert result["search_results"] == []
    assert result["graph_context"] == ""
    assert result["source_status"] == {"milvus": "failed", "graph": "failed"}
    assert "milvus failed: embedding 模型不可用" in result["failure_reason"]
    assert "graph failed: graph embedding unavailable" in result["failure_reason"]


def test_execute_document_grading_preserves_existing_failure_reason() -> None:
    async def fake_grade_documents(query: str, results: list[dict]):  # noqa: ARG001
        return [], []

    result = asyncio.run(
        retrieval_service.execute_document_grading(
            {
                "user_query": "stgcn解决了什么问题",
                "search_results": [],
                "graph_context": "",
                "failure_reason": "milvus failed: embedding 模型不可用",
            },
            grade_documents=fake_grade_documents,
            with_trace=lambda payload, trace: payload,
            format_search_results_fn=lambda results: str(results),
        )
    )

    assert result["grading_action"] == "retry_search"
    assert result["failure_reason"] == "milvus failed: embedding 模型不可用"


def test_execute_document_grading_keeps_graph_context_in_messages() -> None:
    async def fake_grade_documents(query: str, results: list[dict]):  # noqa: ARG001
        return [], []

    result = asyncio.run(
        retrieval_service.execute_document_grading(
            {
                "user_query": "比较这两个论文的方法",
                "search_results": [],
                "graph_context": "Graph says both methods use spatiotemporal modeling.",
                "failure_reason": "",
            },
            grade_documents=fake_grade_documents,
            with_trace=lambda payload, trace: payload,
            format_search_results_fn=lambda results: str(results),
        )
    )

    assert result["grading_action"] == "proceed"
    assert "=== Graph Context ===" in result["messages"][0].content
    assert "Graph says both methods use spatiotemporal modeling." in result["messages"][0].content


def test_execute_rag_search_supports_vector_rerank_alias_fields() -> None:
    async def fake_route_retrieval(query: str):  # noqa: ARG001
        return SimpleNamespace(mode="milvus", graph_mode="hybrid", reasoning="test")

    class FakeReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[dict], int]] = []

        def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
            self.calls.append((query, list(documents), top_k))
            return list(reversed(documents))

    candidates = [
        {"chunk_id": "c1", "paper_id": "p1", "preview": "A", "score": 0.8},
        {"chunk_id": "c2", "paper_id": "p2", "preview": "B", "score": 0.7},
    ]
    settings = SimpleNamespace(
        query_filter_enabled=False,
        query_rewrite_enabled=False,
        vector_rerank_enabled=True,
        vector_rerank_top_k=7,
        rag_top_k=5,
        rag_vector_query_timeout_seconds=0,
        rag_graph_query_timeout_seconds=0,
    )
    cache = _FakeCache()
    paper_db = _FakePaperDb()
    fake_reranker = FakeReranker()

    result = asyncio.run(
        retrieval_service.execute_rag_search(
            {
                "user_query": "vector rerank alias",
                "structured_query": {"project_scope": _trusted_scope(("p1", "p2"))},
            },
            settings=settings,
            route_retrieval=fake_route_retrieval,
            get_cache=lambda: cache,
            get_vector_store=lambda: object(),
            get_lightrag_service=lambda: None,
            get_reranker=lambda: fake_reranker,
            paper_db=paper_db,
            logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            ensure_trace=lambda state: None,
            with_trace=lambda payload, trace: payload,
            run_rewritten_search_fn=lambda store, queries, filter_expr: candidates,
            similarity_threshold_fn=lambda: 0.0,
            rerank_threshold_fn=lambda: 0.0,
        )
    )

    assert fake_reranker.calls == [("vector rerank alias", candidates, 7)]
    assert [item["chunk_id"] for item in result["search_results"]] == ["c2", "c1"]
    assert result["source_status"] == {"milvus": "ok", "graph": "skipped"}


def test_execute_rag_search_marks_rerank_failure_explicitly() -> None:
    async def fake_route_retrieval(query: str):  # noqa: ARG001
        return SimpleNamespace(mode="milvus", graph_mode="hybrid", reasoning="test")

    class FailingReranker:
        def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:  # noqa: ARG002
            raise RuntimeError("local reranker cache missing")

    candidates = [
        {"chunk_id": "c1", "paper_id": "p1", "preview": "A", "score": 0.8},
        {"chunk_id": "c2", "paper_id": "p2", "preview": "B", "score": 0.7},
    ]
    settings = SimpleNamespace(
        query_filter_enabled=False,
        query_rewrite_enabled=False,
        vector_rerank_enabled=True,
        vector_rerank_top_k=7,
        rag_top_k=5,
        rag_vector_query_timeout_seconds=0,
        rag_graph_query_timeout_seconds=0,
    )
    cache = _FakeCache()

    result = asyncio.run(
        retrieval_service.execute_rag_search(
            {
                "user_query": "rerank failure",
                "structured_query": {"project_scope": _trusted_scope(("p1", "p2"))},
            },
            settings=settings,
            route_retrieval=fake_route_retrieval,
            get_cache=lambda: cache,
            get_vector_store=lambda: object(),
            get_lightrag_service=lambda: None,
            get_reranker=lambda: FailingReranker(),
            paper_db=_FakePaperDb(),
            logger=logging.getLogger(__name__),
            ensure_trace=lambda state: None,
            with_trace=lambda payload, trace: payload,
            run_rewritten_search_fn=lambda store, queries, filter_expr: candidates,
            similarity_threshold_fn=lambda: 0.0,
            rerank_threshold_fn=lambda: 0.0,
        )
    )

    assert result["search_results"] == []
    assert result["source_status"] == {"milvus": "ok", "graph": "skipped", "rerank": "failed"}
    assert result["error_kind"] == "rerank_unavailable"
    assert "local reranker cache missing" in result["source_status_reasons"]["rerank"]
    assert "rerank failed" in result["failure_reason"]
    assert cache.saved is None


def test_filter_hits_by_business_state_drops_stale_and_out_of_scope_hits() -> None:
    class FakePaperDb:
        def get_paper(self, paper_id: str):
            return {"paper_id": paper_id} if paper_id == "paper-1" else None

    results = [
        {"chunk_id": "c1", "paper_id": "paper-1", "preview": "valid", "score": 0.9},
        {"chunk_id": "c2", "paper_id": "paper-2", "preview": "stale", "score": 0.8},
        {"chunk_id": "c3", "paper_id": "paper-3", "preview": "out-of-scope", "score": 0.7},
    ]

    filtered = retrieval_service.filter_hits_by_business_state(
        results,
        paper_db=FakePaperDb(),
        structured_query={"paper_ids": ["paper-1", "paper-3"]},
    )

    assert filtered == [{"chunk_id": "c1", "paper_id": "paper-1", "preview": "valid", "score": 0.9}]


def test_execute_rag_search_filters_cached_results_by_business_state() -> None:
    async def fake_route_retrieval(query: str):  # noqa: ARG001
        return SimpleNamespace(mode="milvus", graph_mode="hybrid", reasoning="test")

    settings = SimpleNamespace(
        query_filter_enabled=False,
        query_rewrite_enabled=False,
        rerank_enabled=False,
        rag_top_k=5,
        rag_vector_query_timeout_seconds=0,
        rag_graph_query_timeout_seconds=0,
    )
    cache = _FakeCache()
    cache.saved = (
        "cached scope query",
        'paper_id == "paper-1"',
        {
            "search_results": [
                {"chunk_id": "c1", "paper_id": "paper-1", "preview": "valid", "score": 0.9},
                {"chunk_id": "c2", "paper_id": "paper-2", "preview": "stale", "score": 0.8},
            ],
            "graph_context": "",
            "source_status": {"milvus": "ok", "graph": "skipped"},
        },
        "milvus",
    )
    cache.get_search = lambda cache_key, filter_expr, mode: cache.saved[2]  # noqa: E731

    class ScopedPaperDb(_FakePaperDb):
        def get_paper(self, paper_id: str):  # noqa: ANN001
            return {"paper_id": paper_id} if paper_id == "paper-1" else None

    result = asyncio.run(
        retrieval_service.execute_rag_search(
            {
                "user_query": "cached scope query",
                "structured_query": {"project_scope": _trusted_scope()},
            },
            settings=settings,
            route_retrieval=fake_route_retrieval,
            get_cache=lambda: cache,
            get_vector_store=lambda: object(),
            get_lightrag_service=lambda: None,
            get_reranker=lambda: None,
            paper_db=ScopedPaperDb(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            ensure_trace=lambda state: None,
            with_trace=lambda payload, trace: payload,
            run_rewritten_search_fn=lambda store, queries, filter_expr: [],
            similarity_threshold_fn=lambda: 0.0,
            rerank_threshold_fn=lambda: 0.0,
        )
    )

    assert result["search_results"] == [{"chunk_id": "c1", "paper_id": "paper-1", "preview": "valid", "score": 0.9}]


def test_build_rag_init_payload_preserves_request_scope_fields() -> None:
    payload = retrieval_service.build_rag_init_payload(
        {
            "filter_expr": 'paper_id == "paper-1"',
            "structured_query": {"paper_ids": ["paper-1"]},
            "session_id": "session-1",
            "conversation_id": "conv-1",
            "chat_history": [{"role": "user", "content": "hi"}],
        },
        ensure_trace=lambda state: None,
        with_trace=lambda body, trace: body,
    )

    assert payload["filter_expr"] == 'paper_id == "paper-1"'
    assert payload["structured_query"] == {"paper_ids": ["paper-1"]}
    assert payload["session_id"] == "session-1"
    assert payload["conversation_id"] == "conv-1"
    assert payload["chat_history"] == [{"role": "user", "content": "hi"}]


def test_execute_rag_search_keeps_fusion_for_scoped_research_dialogue() -> None:
    async def fake_route_retrieval(query: str, structured_query=None):  # noqa: ARG001
        return SimpleNamespace(mode="fusion", graph_mode="global", reasoning="test")

    captured: dict[str, object] = {}

    class FakeGraphProvider:
        async def collect_library_graph_hint(self, query: str, mode: str, top_k: int, library_id=None) -> str:  # noqa: ANN001
            captured["library_id"] = library_id
            return "graph hint"

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
                "user_query": "比较当前项目里的论文方法差异",
                "structured_query": {
                    "project_id": "proj-1",
                    "dialogue_mode": "research",
                    "paper_ids": ["paper-1", "paper-2"],
                    "project_scope": _trusted_scope(("paper-1", "paper-2")),
                    "paper_focus": {"mode": "multi_paper"},
                    "resolved_scope": {"paper_ids": ["paper-1", "paper-2"], "library_ids": ["lib-1"]},
                },
                "filter_expr": 'paper_id == "paper-1" or paper_id == "paper-2"',
            },
            settings=settings,
            route_retrieval=fake_route_retrieval,
            get_cache=lambda: _FakeCache(),
            get_vector_store=lambda: object(),
            get_lightrag_service=lambda: FakeGraphProvider(),
            get_reranker=lambda: None,
            paper_db=_FakePaperDb(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
            ensure_trace=lambda state: None,
            with_trace=lambda payload, trace: payload,
            run_rewritten_search_fn=lambda store, queries, filter_expr: [
                {"chunk_id": "c1", "paper_id": "paper-1", "preview": "valid", "score": 0.9}
            ],
            similarity_threshold_fn=lambda: 0.0,
            rerank_threshold_fn=lambda: 0.0,
        )
    )

    assert result["retrieval_mode"] == "fusion"
    assert result["graph_context"] == "graph hint"
    assert result["source_status"] == {"milvus": "ok", "graph": "ok"}
    assert captured["library_id"] == "lib-1"


async def _run_threaded_rerank(reranker):
    """接收可控同步重排端口，运行真实检索编排并返回结果；所有存储均为测试替身。"""
    async def route(query):
        """接收查询并返回确定的向量检索路由。"""
        return SimpleNamespace(mode="milvus", graph_mode="hybrid", reasoning="test")

    candidates = [{"chunk_id": "c1", "paper_id": "paper-1", "preview": "A", "score": 0.8}]
    settings = SimpleNamespace(query_filter_enabled=False, query_rewrite_enabled=False,
        vector_rerank_enabled=True, vector_rerank_top_k=5, rag_top_k=5,
        rag_vector_query_timeout_seconds=0, rag_graph_query_timeout_seconds=0)
    return await retrieval_service.execute_rag_search(
        {"user_query": "threaded rerank", "structured_query": {"project_scope": _trusted_scope()}},
        settings=settings, route_retrieval=route, get_cache=_FakeCache,
        get_vector_store=object, get_lightrag_service=lambda: None,
        get_reranker=lambda: reranker, paper_db=_FakePaperDb(), logger=logging.getLogger(__name__),
        ensure_trace=lambda state: None, with_trace=lambda payload, trace: payload,
        run_rewritten_search_fn=lambda store, queries, scope: candidates,
        similarity_threshold_fn=lambda: 0.0, rerank_threshold_fn=lambda: 0.0,
    )


@pytest.mark.asyncio
async def test_rerank_sync_work_does_not_block_event_loop():
    """同步重排等待期间事件循环仍能触发释放信号，返回材料及分数保持原样。"""
    released = threading.Event()
    owner_thread = threading.get_ident()

    def rerank(query, documents, top_k):
        """接收真实重排参数；等待循环信号后返回原有材料。"""
        assert threading.get_ident() != owner_thread
        assert released.wait(timeout=2)
        return documents

    handle = asyncio.get_running_loop().call_later(0.05, released.set)
    try:
        result = await _run_threaded_rerank(SimpleNamespace(rerank=rerank))
    finally:
        released.set()
        handle.cancel()
    assert result["search_results"][0]["chunk_id"] == "c1"
    assert result["source_status"] == {"milvus": "ok", "graph": "skipped"}


@pytest.mark.asyncio
async def test_rerank_repeated_cancellation_drains_actual_sync_work():
    """连续取消不能提前释放调用所有权；实际同步请求结束后才返回取消。"""
    started = threading.Event()
    released = threading.Event()
    finished = threading.Event()

    def rerank(query, documents, top_k):
        """接收重排参数并模拟可控同步请求，退出前记录实际结束。"""
        started.set()
        assert released.wait(timeout=2)
        finished.set()
        return documents

    async def wait_started():
        """等待实际重排线程开始；不阻塞事件循环。"""
        while not started.is_set():
            await asyncio.sleep(0.001)

    task = asyncio.create_task(_run_threaded_rerank(SimpleNamespace(rerank=rerank)))
    try:
        await asyncio.wait_for(wait_started(), timeout=1)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        assert not finished.is_set()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        released.set()
        await asyncio.gather(task, return_exceptions=True)
