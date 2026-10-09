"""
RAG 检索编排与结果格式化服务。

这是 RAG 子图里最核心的服务层之一，负责把“一个用户问题”转成可用检索结果。
它本身不关心 HTTP，也不直接生成最终答案，而是集中处理：

1. query 改写和过滤表达式准备
2. 向量检索与图检索的路由、并发和回退
3. 结果去重、阈值过滤、rerank、缓存
4. 把检索结果整理成后续节点统一可消费的状态结构

如果你想理解“RAG 到底是怎么跑起来的”，这个文件是最值得先读的入口之一。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from app.config import get_vector_rerank_top_k, is_vector_rerank_enabled
from app.middleware.observability import log_structured_event
from app.models.models import RAGResult, RAGState
from app.models.research_contracts import RetrievalScopeError
from app.services.provider_request_queue import run_provider_thread
from app.services.retrieval_query_expansion import expand_single_paper_summary_queries
from app.services.retrieval_provider_registry import (
    get_default_vector_provider,
    get_graph_provider,
)

_RERANK_SOURCE = "rerank"
_RERANK_ERROR_KIND = "rerank_unavailable"


def _log_event(current_logger: Any, event: str, *, level: int = 20, **fields: Any) -> None:
    """
    兼容真实 logger 和测试里的轻量替身。

    生产环境继续走统一结构化日志；测试若只提供 `info/warning`，这里做一层降级。
    """
    if hasattr(current_logger, "log"):
        log_structured_event(current_logger, event, level=level, **fields)
        return
    message = f"event={event} fields={fields}"
    if level >= 40 and callable(getattr(current_logger, "warning", None)):
        current_logger.warning(message)
        return
    if callable(getattr(current_logger, "info", None)):
        current_logger.info(message)


def format_search_results(results: List[dict]) -> str:
    """把检索结果转换成适合塞进消息流的纯文本。"""
    if not results:
        return "(no results)"
    lines = []
    for item in results:
        lines.append(
            f'  chunk_id={item["chunk_id"]}  paper={item["paper_id"]}  '
            f'score={item.get("score", 0):.3f}\n'
            f'  preview: {item["preview"]}'
        )
    return "\n".join(lines)


def format_read_result(result: dict) -> str:
    """把 read_chunk 结果转成展示文本。"""
    if result.get("skipped"):
        return result["full_text"]
    return (
        f'[{result["section"]} p.{result["page_num"]}]\n'
        f'{result["full_text"]}'
    )


def search_chunks(
    store: Any,
    query: str,
    top_k: int,
    *,
    scope: dict[str, Any],
) -> List[dict]:
    """
    统一封装迁移期检索入口。

    项目当前兼容不同向量存储接口实现，这里做一层适配，
    让上层不用关心底层究竟暴露的是 `hybrid_search` 还是 `local_rag_search`。
    """
    trusted_scope = require_local_retrieval_scope(scope)
    published_search = getattr(store, "search_scope", None)
    if callable(published_search):
        return published_search(query, top_k=top_k, scope=trusted_scope)
    filter_expr = str(trusted_scope["filter_expr"]).strip()
    paper_ids = [str(item).strip() for item in trusted_scope["paper_ids"]]
    probe_health = getattr(store, "probe_health", None)
    if callable(probe_health) and not probe_health():
        last_error = getattr(store, "last_health_error", None)
        detail = last_error() if callable(last_error) else ""
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Milvus health probe failed{suffix}")
    hybrid_search = getattr(store, "hybrid_search", None)
    if callable(hybrid_search):
        return hybrid_search(query, top_k=top_k, filter_expr=filter_expr)
    local_rag_search = getattr(store, "local_rag_search", None)
    if callable(local_rag_search):
        return local_rag_search(query, top_k=top_k, filter_paper_ids=paper_ids)
    raise AttributeError("Vector store must implement hybrid_search or local_rag_search")


def dedupe_hits(groups: List[List[dict]]) -> List[dict]:
    """按 chunk_id 合并多组检索结果。"""
    merged: List[dict] = []
    seen_chunk_ids: set[str] = set()
    for group in groups:
        for item in group:
            chunk_id = item.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            merged.append(item)
    return merged


def _project_scope_contract(structured_query: dict | None) -> dict[str, Any]:
    structured = structured_query if isinstance(structured_query, dict) else {}
    project_scope = structured.get("project_scope")
    if isinstance(project_scope, dict):
        return project_scope
    return {}


def require_local_retrieval_scope(scope: dict[str, Any] | None) -> dict[str, Any]:
    """
    校验本地向量检索所需的可信范围合同。

    Args:
        scope: route/facade 构造并注入 structured_query 的项目范围合同

    Returns:
        dict[str, Any]: 可直接交给向量适配器的非空 scope

    Raises:
        RetrievalScopeError: scope 缺失、未授权或没有论文过滤条件
    """
    if not isinstance(scope, dict):
        raise RetrievalScopeError()
    user_id = str(scope.get("user_id") or "").strip()
    paper_ids = [
        str(item or "").strip()
        for item in scope.get("paper_ids", [])
        if str(item or "").strip()
    ]
    filter_expr = str(scope.get("filter_expr") or "").strip()
    if not user_id or not scope.get("local_retrieval_allowed") or not paper_ids or not filter_expr:
        raise RetrievalScopeError()
    return scope


def require_local_scope_from_query(structured_query: dict | None) -> dict[str, Any]:
    """从结构化请求中提取并校验本地检索范围。"""
    return require_local_retrieval_scope(_project_scope_contract(structured_query))


def _paper_focus_contract(structured_query: dict | None) -> dict[str, Any]:
    structured = structured_query if isinstance(structured_query, dict) else {}
    paper_focus = structured.get("paper_focus")
    if isinstance(paper_focus, dict):
        return paper_focus
    return {}


def _project_scope_id(structured_query: dict | None) -> str:
    project_scope = _project_scope_contract(structured_query)
    resolved_scope = structured_query.get("resolved_scope") if isinstance(structured_query, dict) and isinstance(structured_query.get("resolved_scope"), dict) else {}
    return str(
        project_scope.get("project_id")
        or resolved_scope.get("project_id")
        or (structured_query or {}).get("project_id")
        or ""
    ).strip()


def resolve_graph_library_id(structured_query: dict | None) -> str | None:
    """尽量把图检索对齐到当前项目的 library 作用域。"""
    structured = structured_query if isinstance(structured_query, dict) else {}
    explicit_library_id = str(structured.get("library_id") or "").strip()
    if explicit_library_id:
        return explicit_library_id
    for scope in (_project_scope_contract(structured_query), structured.get("resolved_scope")):
        if not isinstance(scope, dict):
            continue
        library_ids = scope.get("library_ids")
        if isinstance(library_ids, list):
            normalized = [str(item).strip() for item in library_ids if str(item).strip()]
            if normalized:
                return normalized[0]
    return None


def allow_graph_hint_under_scope(structured_query: dict | None) -> bool:
    """研究型对话在有项目范围时允许保留图谱全局线索。"""
    structured = structured_query if isinstance(structured_query, dict) else {}
    dialogue_mode = str(structured.get("dialogue_mode") or "").strip().lower()
    if dialogue_mode == "research":
        return True
    paper_focus = _paper_focus_contract(structured_query)
    return str(paper_focus.get("mode") or "").strip().lower() == "multi_paper"


def resolve_allowed_paper_ids(structured_query: dict | None) -> set[str]:
    """从请求级结构化范围里提取允许命中的 paper_id 集合。"""
    structured = structured_query if isinstance(structured_query, dict) else {}
    allowed: set[str] = set()
    for raw in (structured.get("paper_ids"), (structured.get("resolved_scope") or {}).get("paper_ids")):
        if not isinstance(raw, list):
            continue
        for item in raw:
            paper_id = str(item or "").strip()
            if paper_id:
                allowed.add(paper_id)
    return allowed


def filter_hits_by_business_state(
    results: List[dict],
    *,
    paper_db: Any,
    structured_query: dict | None = None,
) -> List[dict]:
    """
    过滤掉业务库中已不存在，或已超出当前允许范围的命中。

    这样即使 Milvus 里还残留旧 chunk，也不会继续进入 RAG 主链。

    Args:
        results: 原始向量召回结果
        paper_db: 业务论文库，用于确认论文是否仍然存在
        structured_query: 当前请求的结构化范围合同

    Returns:
        List[dict]: 通过业务存在性与作用域校验后的命中列表
    """
    allowed_paper_ids = resolve_allowed_paper_ids(structured_query)
    get_paper = getattr(paper_db, "get_paper", None)
    filtered: list[dict] = []
    for item in results:
        paper_id = str(item.get("paper_id") or "").strip()
        if not paper_id:
            continue
        if allowed_paper_ids and paper_id not in allowed_paper_ids:
            continue
        if callable(get_paper) and not get_paper(paper_id):
            continue
        filtered.append(item)
    return filtered


def run_rewritten_search(
    store: Any,
    queries: List[str],
    scope: dict[str, Any],
    *,
    candidate_limit: int = 15,
) -> List[dict]:
    """执行多查询改写后的批量检索。"""
    trusted_scope = require_local_retrieval_scope(scope)
    groups = [
        search_chunks(store, query=item, top_k=candidate_limit, scope=trusted_scope)
        for item in queries
    ]
    return dedupe_hits(groups)


def similarity_threshold(settings: Any) -> float:
    """读取向量检索阈值。"""
    return float(getattr(settings, "rag_similarity_threshold", getattr(settings, "rag_score_threshold", 0.0)))


def rerank_threshold(settings: Any) -> float:
    """读取 rerank 阈值。"""
    return float(getattr(settings, "rag_rerank_threshold", 0.0))


def filter_by_score(results: List[dict], field: str, threshold: float) -> List[dict]:
    """按指定分数字段过滤结果。"""
    if threshold <= 0:
        return results
    return [item for item in results if float(item.get(field, 0.0)) >= threshold]


def search_cache_key(queries: List[str]) -> str:
    """为查询改写结果生成稳定缓存 key。"""
    return "\n".join(queries)


def format_graph_context(content: str) -> str:
    """把图谱聚合文本裁成适合消息流展示的上下文。"""
    return content.strip()


def cached_search_payload(payload: Any) -> tuple[List[dict], str, Dict[str, str]]:
    """
    兼容旧 list 缓存和新的分源 payload。

    这说明缓存结构经历过演进：早期只有 search_results，
    后来又补进了 graph_context 和 source_status。
    """
    if isinstance(payload, list):
        return payload, "", {"milvus": "ok", "graph": "skipped"}
    if not isinstance(payload, dict):
        return [], "", {"milvus": "skipped", "graph": "skipped"}
    return (
        payload.get("search_results", []),
        payload.get("graph_context", ""),
        payload.get("source_status", {"milvus": "skipped", "graph": "skipped"}),
    )


def build_search_cache_payload(search_results: List[dict], graph_context: str, source_status: Dict[str, str]) -> dict:
    """统一缓存 payload。"""
    return {
        "search_results": search_results,
        "graph_context": graph_context,
        "source_status": source_status,
    }


def build_rag_search_response(
    query: str,
    rewritten_query: str,
    queries: List[str],
    merged: List[dict],
    graph_context: str = "",
    source_status: Optional[Dict[str, str]] = None,
    source_status_reasons: Optional[Dict[str, str]] = None,
    failure_reason: str = "",
    error_kind: str = "",
    retrieval_mode: Any = None,
) -> dict:
    """把首轮检索结果收敛成统一状态更新结构。"""
    sections = [
        f"=== Initial Search Results for: {query!r} ===\n" + format_search_results(merged)
    ]
    if graph_context:
        sections.append(f"=== Graph Context ===\n{format_graph_context(graph_context)}")
    result_text = "\n\n".join(sections)
    response = {
        "messages": [HumanMessage(content=result_text)],
        "rewritten_query": rewritten_query,
        "rewritten_queries": queries,
        "search_results": merged,
        "graph_context": graph_context,
        "source_status": source_status or {"milvus": "skipped", "graph": "skipped"},
        "source_status_reasons": source_status_reasons or {},
        "rag_result": RAGResult(
            chunks=[item["preview"] for item in merged],
            sources=[item["paper_id"] for item in merged],
            scores=[item.get("score", 0.0) for item in merged],
            is_sufficient=False,
        ),
    }
    if failure_reason:
        response["failure_reason"] = failure_reason
    if error_kind:
        response["error_kind"] = error_kind
    if retrieval_mode is not None:
        response["retrieval_mode"] = retrieval_mode.mode
        response["retrieval_reasoning"] = retrieval_mode.reasoning
        response["graph_mode"] = retrieval_mode.graph_mode
    return response


async def run_vector_retrieval(
    mode: str,
    queries: List[str],
    scope: dict[str, Any] | None,
    *,
    settings: Any,
    get_vector_store: Any | None = None,
    run_rewritten_search_fn: Any,
) -> List[dict]:
    """
    按路由决定是否执行 Milvus 检索。

    这里是“执行层”，不负责决定 route；route 已经由上游 `route_retrieval`
    计算好，这里只根据 mode 做实际调用和超时控制。

    Args:
        mode: 上游已决策好的检索模式
        queries: query rewrite 后的查询列表
        scope: 当前已完成归属校验的本地检索范围
        settings: 全局配置
        get_vector_store: 可注入的向量存储 getter
        run_rewritten_search_fn: 真正执行多 query 检索的函数

    Returns:
        List[dict]: Milvus 命中结果；若本轮不该走向量检索则返回空列表
    """
    if mode not in ("milvus", "fusion"):
        return []
    trusted_scope = require_local_retrieval_scope(scope)
    vector_store_getter = get_vector_store or get_default_vector_provider
    store = vector_store_getter()
    probe_health = getattr(store, "probe_health", None)
    if callable(probe_health) and not probe_health():
        last_error = getattr(store, "last_health_error", None)
        detail = last_error() if callable(last_error) else ""
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Milvus health probe failed{suffix}")
    timeout_seconds = float(getattr(settings, "rag_vector_query_timeout_seconds", 12.0))
    retrieval_call = asyncio.to_thread(run_rewritten_search_fn, store, queries, trusted_scope)
    if timeout_seconds <= 0:
        return await retrieval_call
    try:
        return await asyncio.wait_for(retrieval_call, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Milvus query timed out after {timeout_seconds:.1f}s") from exc


async def run_graph_retrieval(
    query: str,
    mode: str,
    graph_mode: str,
    *,
    settings: Any,
    library_id: str | None = None,
    get_lightrag_service: Any | None = None,
) -> str:
    """
    按路由决定是否执行 LightRAG / 图提示检索。

    Args:
        query: 当前问题文本
        mode: 上游已决策好的检索模式
        graph_mode: LightRAG 侧的 local/global/hybrid 模式
        settings: 全局配置
        library_id: 可选图谱库 ID，用于 project-scoped 图提示
        get_lightrag_service: 可注入的图检索 provider getter

    Returns:
        str: 图检索返回的聚合上下文文本；本轮不走图检索时返回空串
    """
    if mode not in ("graph", "fusion"):
        return ""
    graph_provider_getter = get_lightrag_service or get_graph_provider
    provider = graph_provider_getter()
    if provider is None:
        return ""
    collect_hint = getattr(provider, "collect_library_graph_hint", None)
    target_callable = collect_hint if callable(collect_hint) else getattr(provider, "query", None)
    if target_callable is None:
        return ""
    try:
        parameters = inspect.signature(target_callable).parameters
    except (TypeError, ValueError):
        parameters = {}
    supports_library_id = "library_id" in parameters
    if callable(collect_hint):
        if supports_library_id:
            graph_call = collect_hint(query=query, mode=graph_mode, top_k=settings.rag_top_k, library_id=library_id)
        else:
            graph_call = collect_hint(query=query, mode=graph_mode, top_k=settings.rag_top_k)
    else:
        if supports_library_id:
            graph_call = provider.query(query=query, mode=graph_mode, top_k=settings.rag_top_k, library_id=library_id)
        else:
            graph_call = provider.query(query=query, mode=graph_mode, top_k=settings.rag_top_k)
    timeout_seconds = float(getattr(settings, "rag_graph_query_timeout_seconds", 12.0))
    if timeout_seconds <= 0:
        return await graph_call
    try:
        return await asyncio.wait_for(graph_call, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"LightRAG query timed out after {timeout_seconds:.1f}s") from exc


def collect_source_error(errors: Dict[str, Exception]) -> str:
    """把分源异常压成可显式暴露的失败说明。"""
    if not errors:
        return ""
    return "; ".join(f"{name} failed: {exc}" for name, exc in errors.items())


def build_rag_init_payload(
    state: RAGState,
    *,
    ensure_trace: Any,
    with_trace: Any,
) -> dict:
    """
    构造 RAG 子图的初始化状态。

    这个初始化 payload 的意义是：把后续节点要依赖的关键字段一次性补齐，
    避免每个节点自己猜默认值。

    Args:
        state: 主图桥接进来的初始 RAGState
        ensure_trace: 负责初始化 trace 的函数
        with_trace: 负责把 trace 绑定回 payload 的函数

    Returns:
        dict: 适合直接进入 RAG 子图第一轮节点执行的初始化状态
    """
    return with_trace(
        {
            "iteration": 0,
            "should_continue": True,
            "already_read": set(),
            "retrieval_keys": set(),
            "rewritten_queries": [],
            "filter_expr": state.get("filter_expr"),
            "structured_query": state.get("structured_query") or {},
            "dialogue_mode": state.get("dialogue_mode") or "",
            "project_scope": state.get("project_scope"),
            "paper_focus": state.get("paper_focus"),
            "session_id": state.get("session_id") or "",
            "conversation_id": state.get("conversation_id") or "",
            "chat_history": state.get("chat_history") or [],
            "failure_reason": "",
            "context_summary": state.get("context_summary") or "",
            "rag_result": RAGResult(chunks=[], sources=[], scores=[], is_sufficient=False),
            "search_results": [],
            "graph_context": state.get("graph_context") or "",
            "graph_evidence": state.get("graph_evidence") or [],
            "paper_sources": state.get("paper_sources") or [],
            "inventory_contexts": state.get("inventory_contexts") or [],
            "inventory_source_ids": state.get("inventory_source_ids") or [],
            "chunk_citations": state.get("chunk_citations") or [],
            "source_status": state.get("source_status") or {"milvus": "unused", "graph": "unused", "neo4j": "unused"},
            "retrieval_mode": state.get("retrieval_mode") or "",
            "retrieval_reasoning": state.get("retrieval_reasoning") or "",
            "graph_mode": state.get("graph_mode") or "",
            "grading_action": "",
            "evaluation": None,
            "self_rag_action": "",
            "self_rag_iteration": 0,
            "answer_confidence": None,
            "citations": [],
            "messages": [],
        },
        ensure_trace(state),
    )


def resolve_retrieval_mode_override(state: RAGState) -> Any | None:
    """
    仅供离线评估进程强制指定检索模式。

    在线主链默认不写这个字段，因此这里不会改变线上路由行为。
    """
    mode = str(state.get("retrieval_mode_override") or "").strip().lower()
    if mode not in ("milvus", "graph", "fusion"):
        return None
    graph_mode = str(state.get("graph_mode_override") or "hybrid").strip().lower() or "hybrid"
    if graph_mode not in ("local", "global", "hybrid"):
        graph_mode = "hybrid"
    return SimpleNamespace(
        mode=mode,
        graph_mode=graph_mode,
        reasoning=f"离线评估强制指定检索模式: {mode}",
    )


async def resolve_retrieval_mode(
    route_retrieval: Any,
    *,
    query: str,
    structured_query: dict | None,
) -> Any:
    """兼容旧测试替身和新签名的检索路由入口。"""
    try:
        parameters = inspect.signature(route_retrieval).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "structured_query" in parameters:
        return await route_retrieval(query, structured_query=structured_query)
    return await route_retrieval(query)


async def execute_rag_search(
    state: RAGState,
    *,
    settings: Any,
    route_retrieval: Any,
    get_cache: Any,
    get_vector_store: Any | None = None,
    get_lightrag_service: Any | None = None,
    get_reranker: Any,
    paper_db: Any,
    logger: Any,
    ensure_trace: Any,
    with_trace: Any,
    run_rewritten_search_fn: Any,
    similarity_threshold_fn: Any,
    rerank_threshold_fn: Any,
) -> dict:
    """
    执行冷启动检索节点。

    这是 RAG 检索主流程的总编排函数，顺序大致是：
    1. 解析作用域 / filter_expr
    2. 可选 query parse 与 query rewrite
    3. 路由到 milvus / graph / fusion
    4. 查缓存，命中则直接返回
    5. 并发检索、聚合结果、阈值过滤、rerank
    6. 写缓存、写 trace、返回统一结构

    Args:
        state: 当前 RAGState
        settings: 全局配置
        route_retrieval: 检索模式路由函数
        get_cache: 检索缓存 getter
        get_vector_store: 向量存储 getter
        get_lightrag_service: 图检索 provider getter
        get_reranker: reranker getter
        paper_db: 业务论文库
        logger: 当前模块 logger 或测试替身
        ensure_trace: 初始化 trace 的函数
        with_trace: 把 trace 绑定回 payload 的函数
        run_rewritten_search_fn: 执行多 query 向量检索的函数
        similarity_threshold_fn: 相似度阈值读取函数
        rerank_threshold_fn: rerank 阈值读取函数

    Returns:
        dict: rag_init 节点后续节点可直接消费的统一检索状态更新
    """
    query = state.get("user_query", "")
    structured_query = state.get("structured_query") or {}
    scope = _project_scope_contract(structured_query)
    filter_expr = str(scope.get("filter_expr") or "").strip()
    dialogue_mode = str((structured_query.get("dialogue_mode") if isinstance(structured_query, dict) else "") or "").strip().lower()

    trace = ensure_trace(state)
    if (
        not scope.get("local_retrieval_allowed")
        and str(structured_query.get("research_mode") or "").strip().lower() == "external"
    ):
        return with_trace(
            build_rag_search_response(
                query,
                "",
                [query],
                [],
                source_status={"milvus": "skipped", "graph": "skipped"},
                failure_reason="外部 web-only 研究不进入本地检索",
            ),
            trace,
        )
    if not scope.get("local_retrieval_allowed"):
        raise RetrievalScopeError()

    queries = [query]
    rewritten_query = state.get("rewritten_query", "")
    project_id = _project_scope_id(structured_query)
    paper_focus_mode = str(_paper_focus_contract(structured_query).get("mode") or "").strip()
    if settings.query_rewrite_enabled:
        # query rewrite 的目标不是覆盖原问题，而是扩成多条可并行检索的候选查询。
        try:
            from app.nodes.query_rewrite import get_all_queries, rewrite_query

            rewritten = rewrite_query(query=query, context=state.get("context_summary", ""))
            queries = get_all_queries(rewritten)
            rewritten_query = rewritten.rewritten
        except Exception as exc:
            _log_event(
                logger,
                "rag_query_rewrite_failed",
                level=40,
                query=query,
                outcome="fallback_original_query",
                error=str(exc),
            )
    elif rewritten_query:
        queries = [rewritten_query]
    queries = expand_single_paper_summary_queries(
        query=query,
        queries=queries,
        structured_query=structured_query,
    )

    if trace is not None:
        trace.rewritten_queries = list(queries)

    retrieval_mode = resolve_retrieval_mode_override(state)
    if retrieval_mode is None:
        retrieval_mode = await resolve_retrieval_mode(
            route_retrieval,
            query=query,
            structured_query=structured_query,
        )
    if trace is not None:
        trace.mode = retrieval_mode.mode
        trace.graph_mode = retrieval_mode.graph_mode
        trace.retrieval_reasoning = str(retrieval_mode.reasoning or "").strip()
    if state.get("retrieval_mode_override"):
        _log_event(
            logger,
            "rag_retrieval_mode_forced",
            request_id=getattr(trace, "request_id", None),
            session_id=getattr(trace, "session_id", None),
            mode=retrieval_mode.mode,
            graph_mode=retrieval_mode.graph_mode,
            dialogue_mode=dialogue_mode,
            project_id=project_id or None,
            paper_focus_mode=paper_focus_mode or None,
            retrieval_reasoning=retrieval_mode.reasoning,
            outcome="override",
        )
    has_scope_constraint = bool(scope.get("local_retrieval_allowed"))
    if has_scope_constraint and retrieval_mode.mode == "graph":
        # 当前 GraphRAG 还不能承担项目内的严格主证据角色，纯 graph 模式直接回退到 Milvus。
        retrieval_mode.mode = "milvus"
        retrieval_mode.graph_mode = "hybrid"
        retrieval_mode.reasoning = (
            "存在 paper 范围约束，纯图检索当前不能承担主证据，已回退到 Milvus"
        )
        if trace is not None:
            trace.mode = retrieval_mode.mode
            trace.graph_mode = retrieval_mode.graph_mode
            trace.retrieval_reasoning = str(retrieval_mode.reasoning or "").strip()
        _log_event(
            logger,
            "rag_retrieval_mode_adjusted",
            request_id=getattr(trace, "request_id", None),
            session_id=getattr(trace, "session_id", None),
            mode=retrieval_mode.mode,
            graph_mode=retrieval_mode.graph_mode,
            dialogue_mode=dialogue_mode,
            project_id=project_id or None,
            paper_focus_mode=paper_focus_mode or None,
            retrieval_reasoning=retrieval_mode.reasoning,
            outcome="scope_forced_milvus",
            source_status={"milvus": "pending", "graph": "skipped"},
            paper_count=len(structured_query.get("paper_ids") or []) if isinstance(structured_query, dict) else 0,
        )
    elif has_scope_constraint and retrieval_mode.mode == "fusion" and allow_graph_hint_under_scope(structured_query):
        retrieval_mode.reasoning = (
            "当前问题在项目范围内使用 Milvus 主证据，并保留 LightRAG 作为全局线索层"
        )
        _log_event(
            logger,
            "rag_retrieval_mode_adjusted",
            request_id=getattr(trace, "request_id", None),
            session_id=getattr(trace, "session_id", None),
            mode=retrieval_mode.mode,
            graph_mode=retrieval_mode.graph_mode,
            outcome="scope_kept_fusion_hint",
            source_status={"milvus": "pending", "graph": "pending"},
            dialogue_mode=dialogue_mode,
            project_id=project_id or None,
            paper_focus_mode=paper_focus_mode or None,
            retrieval_reasoning=retrieval_mode.reasoning,
            paper_count=len(structured_query.get("paper_ids") or []) if isinstance(structured_query, dict) else 0,
        )
    elif has_scope_constraint and retrieval_mode.mode == "fusion":
        retrieval_mode.mode = "milvus"
        retrieval_mode.graph_mode = "hybrid"
        retrieval_mode.reasoning = (
            "当前范围约束下不保留图谱线索，优先使用 Milvus 原文证据"
        )
        if trace is not None:
            trace.mode = retrieval_mode.mode
            trace.graph_mode = retrieval_mode.graph_mode
            trace.retrieval_reasoning = str(retrieval_mode.reasoning or "").strip()
        _log_event(
            logger,
            "rag_retrieval_mode_adjusted",
            request_id=getattr(trace, "request_id", None),
            session_id=getattr(trace, "session_id", None),
            mode=retrieval_mode.mode,
            graph_mode=retrieval_mode.graph_mode,
            dialogue_mode=dialogue_mode,
            project_id=project_id or None,
            paper_focus_mode=paper_focus_mode or None,
            retrieval_reasoning=retrieval_mode.reasoning,
            outcome="scope_forced_milvus",
            source_status={"milvus": "pending", "graph": "skipped"},
            paper_count=len(structured_query.get("paper_ids") or []) if isinstance(structured_query, dict) else 0,
        )
    cache = get_cache()
    cache_key = search_cache_key(queries)
    # 缓存 key 同时绑定 query/filter/mode，避免不同检索范围互相串结果。
    cached_payload = cache.get_search(cache_key, filter_expr, retrieval_mode.mode)
    if cached_payload is not None:
        cached_results, cached_graph_context, cached_status = cached_search_payload(cached_payload)
        cached_results = filter_hits_by_business_state(
            cached_results,
            paper_db=paper_db,
            structured_query=structured_query,
        )
        if trace is not None:
            trace.mode = retrieval_mode.mode
            trace.graph_mode = retrieval_mode.graph_mode
            trace.retrieval_reasoning = str(retrieval_mode.reasoning or "").strip()
            trace.source_status = dict(cached_status)
            trace.outcome = "cache_hit"
            trace.checkpoint("retrieval_start")
            trace.retrieval["hybrid_results"] = len(cached_results)
            trace.retrieval["reranked"] = len(cached_results)
            trace.checkpoint("retrieval_end")
        await asyncio.to_thread(
            paper_db.write_log,
            event_type="rag_hit",
            node_name="rag_search_node_cache",
            query=query,
            result_count=len(cached_results),
            latency_ms=0,
        )
        _log_event(
            logger,
            "rag_retrieval_completed",
            request_id=getattr(trace, "request_id", None),
            session_id=getattr(trace, "session_id", None),
            mode=retrieval_mode.mode,
            graph_mode=retrieval_mode.graph_mode,
            dialogue_mode=dialogue_mode,
            project_id=project_id or None,
            paper_focus_mode=paper_focus_mode or None,
            retrieval_reasoning=retrieval_mode.reasoning,
            source_status=cached_status,
            outcome="cache_hit",
            result_count=len(cached_results) + (1 if cached_graph_context else 0),
            duration_ms=0,
        )
        return with_trace(
            build_rag_search_response(
                query,
                rewritten_query,
                queries,
                cached_results,
                graph_context=cached_graph_context,
                source_status=cached_status,
                retrieval_mode=retrieval_mode,
            ),
            trace,
        )

    t0 = time.monotonic()
    if trace is not None:
        trace.checkpoint("retrieval_start")

    trusted_scope = (
        require_local_retrieval_scope(scope)
        if retrieval_mode.mode in ("milvus", "fusion")
        else None
    )
    vector_task = run_vector_retrieval(
        retrieval_mode.mode,
        queries,
        trusted_scope,
        settings=settings,
        get_vector_store=get_vector_store,
        run_rewritten_search_fn=run_rewritten_search_fn,
    )
    graph_task = run_graph_retrieval(
        query,
        retrieval_mode.mode,
        retrieval_mode.graph_mode,
        settings=settings,
        library_id=resolve_graph_library_id(structured_query),
        get_lightrag_service=get_lightrag_service,
    )
    # 两路检索并发执行；即使其中一路失败，也尽量保留另一路结果。
    vector_result, graph_result = await asyncio.gather(vector_task, graph_task, return_exceptions=True)
    errors: Dict[str, Exception] = {}
    merged: List[dict] = []
    graph_context = ""
    source_status = {"milvus": "skipped", "graph": "skipped"}

    if retrieval_mode.mode in ("milvus", "fusion"):
        if isinstance(vector_result, Exception):
            errors["milvus"] = vector_result
            source_status["milvus"] = "failed"
            logger.warning(
                "milvus retrieval failed: query=%s filter=%s error=%s",
                query[:80],
                filter_expr[:160],
                repr(vector_result),
                exc_info=vector_result,
            )
        else:
            merged = filter_hits_by_business_state(
                filter_by_score(vector_result, "score", similarity_threshold_fn()),
                paper_db=paper_db,
                structured_query=structured_query,
            )
            source_status["milvus"] = "ok"
            if trace is not None:
                trace.retrieval["hybrid_results"] = len(merged)

    if retrieval_mode.mode in ("graph", "fusion"):
        if isinstance(graph_result, Exception):
            errors["graph"] = graph_result
            source_status["graph"] = "failed"
            logger.warning(
                "graph retrieval failed: query=%s mode=%s error=%s",
                query[:80],
                retrieval_mode.graph_mode,
                repr(graph_result),
                exc_info=graph_result,
            )
        else:
            graph_context = format_graph_context(graph_result)
            source_status["graph"] = "ok" if graph_context else "empty"

    # 即使所有实际执行的数据源都失败，也不要直接把异常冒泡成 500；
    # 主链应该保留 source_status 和 failure_reason，交给后续节点做受控降级。
    all_attempted_sources_failed = bool(errors) and all(
        status == "failed" for status in source_status.values() if status != "skipped"
    )
    failure_reason = collect_source_error(errors)

    if trace is not None:
        trace.checkpoint("retrieval_end")

    vector_rerank_enabled = is_vector_rerank_enabled(settings)
    source_status_reasons: Dict[str, str] = {}
    error_kind = ""
    rerank_failed = False
    if trace is not None:
        trace.retrieval["vector_rerank_enabled"] = vector_rerank_enabled
    if vector_rerank_enabled and merged:
        # vector_rerank 只对向量结果生效；graph_context 仍作为补充上下文直接透传。
        rerank_top_k = max(settings.rag_top_k, get_vector_rerank_top_k(settings))
        if trace is not None:
            trace.checkpoint("rerank_start")
        try:
            # 1. 【检索】【候选重排】同步模型请求交由受管理线程执行，取消先等待实际请求结束
            merged = await run_provider_thread(get_reranker().rerank, query, merged, rerank_top_k)
        except Exception as exc:
            rerank_failed = True
            error_kind = _RERANK_ERROR_KIND
            source_status[_RERANK_SOURCE] = "failed"
            source_status_reasons[_RERANK_SOURCE] = f"{_RERANK_ERROR_KIND}: {exc}"
            failure_reason = f"{failure_reason}; " if failure_reason else ""
            failure_reason += f"{_RERANK_SOURCE} failed: {exc}"
            merged = []
            logger.error(
                "【检索】【候选重排】vector rerank failed: query=%s error=%s",
                query[:80],
                repr(exc),
                exc_info=exc,
            )
            if trace is not None:
                trace.retrieval["reranked"] = 0
                trace.retrieval["vector_reranked"] = 0
                trace.retrieval["rerank_error_kind"] = _RERANK_ERROR_KIND
                trace.checkpoint("rerank_end")
        if trace is not None:
            if not rerank_failed:
                trace.retrieval["reranked"] = len(merged)
                trace.retrieval["vector_reranked"] = len(merged)
                trace.checkpoint("rerank_end")
        if not rerank_failed:
            merged = filter_by_score(merged, "rerank_score", rerank_threshold_fn())
            merged = merged[: settings.rag_top_k]
    else:
        merged = merged[: settings.rag_top_k]

    if not all_attempted_sources_failed and not rerank_failed:
        cache.set_search(
            cache_key,
            filter_expr,
            build_search_cache_payload(merged, graph_context, source_status),
            retrieval_mode.mode,
        )

    latency = int((time.monotonic() - t0) * 1000)
    event_type = "rag_hit" if merged or graph_context else "rag_miss"
    outcome = "rerank_failed" if rerank_failed else ("all_sources_failed" if all_attempted_sources_failed else event_type)
    if trace is not None:
        trace.source_status = dict(source_status)
        trace.graph_mode = retrieval_mode.graph_mode
        trace.retrieval_reasoning = str(retrieval_mode.reasoning or "").strip()
        trace.outcome = outcome
    await asyncio.to_thread(
        paper_db.write_log,
        event_type=event_type,
        node_name="rag_search_node",
        query=query,
        result_count=len(merged) + (1 if graph_context else 0),
        latency_ms=latency,
    )
    _log_event(
        logger,
        "rag_retrieval_completed",
        level=40 if all_attempted_sources_failed else 20,
        request_id=getattr(trace, "request_id", None),
        session_id=getattr(trace, "session_id", None),
        mode=retrieval_mode.mode,
        graph_mode=retrieval_mode.graph_mode,
        dialogue_mode=dialogue_mode,
        project_id=project_id or None,
        paper_focus_mode=paper_focus_mode or None,
        retrieval_reasoning=retrieval_mode.reasoning,
        source_status=source_status,
        source_status_reasons=source_status_reasons,
        outcome=outcome,
        result_count=len(merged) + (1 if graph_context else 0),
        duration_ms=latency,
        error=failure_reason if all_attempted_sources_failed or rerank_failed else None,
    )
    return with_trace(
        build_rag_search_response(
            query,
            rewritten_query,
            queries,
            merged,
            graph_context=graph_context,
            source_status=source_status,
            source_status_reasons=source_status_reasons,
            failure_reason=failure_reason,
            error_kind=error_kind,
            retrieval_mode=retrieval_mode,
        ),
        trace,
    )


async def execute_document_grading(
    state: RAGState,
    *,
    grade_documents: Any,
    with_trace: Any,
    format_search_results_fn: Any,
) -> dict:
    """执行首轮检索结果的相关性过滤。"""
    trace = state.get("trace")
    if trace is not None:
        trace.checkpoint("grading_start")
    relevant, irrelevant = await grade_documents(
        state.get("user_query", ""),
        state.get("search_results", []),
    )
    if trace is not None:
        trace.retrieval["graded_relevant"] = len(relevant)
        trace.retrieval["graded_irrelevant"] = len(irrelevant)
        trace.checkpoint("grading_end")
    if relevant:
        return with_trace(
            {
                "search_results": relevant,
                "grading_action": "proceed",
                "messages": [
                    HumanMessage(
                        content="=== Document Grading Results ===\n"
                        + format_search_results_fn(relevant)
                    )
                ],
                "rag_result": RAGResult(
                    chunks=[item["preview"] for item in relevant],
                    sources=[item["paper_id"] for item in relevant],
                    scores=[item.get("score", 0.0) for item in relevant],
                    is_sufficient=False,
                ),
            },
            trace,
        )
    if state.get("graph_context"):
        graph_context = str(state.get("graph_context") or "").strip()
        return with_trace(
            {
                "search_results": [],
                "grading_action": "proceed",
                "failure_reason": state.get("failure_reason", ""),
                "messages": [
                    HumanMessage(
                        content="=== Document Grading Results ===\n"
                        "(no vector hits to grade; using graph context)"
                        + (
                            f"\n\n=== Graph Context ===\n{graph_context}"
                            if graph_context
                            else ""
                        )
                    )
                ],
                "rag_result": RAGResult(chunks=[], sources=[], scores=[], is_sufficient=False),
            },
            trace,
        )
    return with_trace(
        {
            "search_results": [],
            "grading_action": "retry_search",
            "failure_reason": state.get("failure_reason") or "all initial search results were graded as irrelevant",
            "messages": [
                HumanMessage(
                    content="=== Document Grading Results ===\n"
                    "(no relevant documents remained after grading)"
                )
            ],
            "rag_result": RAGResult(chunks=[], sources=[], scores=[], is_sufficient=False),
        },
        trace,
    )
