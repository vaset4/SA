"""
RAG 冷启动检索节点辅助。

这个文件是一个很薄的转发层：
- `rag_node.py` 负责子图装配
- `retrieval_service.py` 负责真正的检索逻辑
- 这里负责把 LangGraph node 和 service 函数粘起来

这样做的好处是，子图装配代码不会被检索细节淹没。
"""

from __future__ import annotations

from typing import Any

import app.services.retrieval_service as retrieval_service


def run_rag_init(
    state: dict[str, Any],
    *,
    ensure_trace: Any,
    with_trace: Any,
) -> dict[str, Any]:
    """初始化 RAG 子图所需的基础状态字段。"""
    return retrieval_service.build_rag_init_payload(
        state,
        ensure_trace=ensure_trace,
        with_trace=with_trace,
    )


async def run_rag_search(
    state: dict[str, Any],
    *,
    settings: Any,
    route_retrieval: Any,
    get_cache: Any,
    get_vector_store: Any,
    get_lightrag_service: Any,
    get_reranker: Any,
    paper_db: Any,
    logger: Any,
    ensure_trace: Any,
    with_trace: Any,
    run_rewritten_search_fn: Any,
    similarity_threshold_fn: Any,
    rerank_threshold_fn: Any,
) -> dict[str, Any]:
    """执行 RAG 首轮检索，并返回子图可直接消费的状态更新。"""
    return await retrieval_service.execute_rag_search(
        state,
        settings=settings,
        route_retrieval=route_retrieval,
        get_cache=get_cache,
        get_vector_store=get_vector_store,
        get_lightrag_service=get_lightrag_service,
        get_reranker=get_reranker,
        paper_db=paper_db,
        logger=logger,
        ensure_trace=ensure_trace,
        with_trace=with_trace,
        run_rewritten_search_fn=run_rewritten_search_fn,
        similarity_threshold_fn=similarity_threshold_fn,
        rerank_threshold_fn=rerank_threshold_fn,
    )
