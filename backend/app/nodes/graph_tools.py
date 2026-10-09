"""
GraphRAG 工具包装。

这个模块不是 LightRAG 主实现，而是把图谱检索能力包装成 LangGraph/ToolNode
 可调用的工具接口。

需要特别区分：
  - 真正执行 GraphRAG 查询的是 `services/lightrag_service.py`
  - 这里负责把查询结果转换成 ToolMessage 和 Command 更新
  - 它服务于 `rag_node.py` 中的 ReAct 工具调用路径
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

# ToolMessage 用于把工具结果写回消息流。
from langchain_core.messages import ToolMessage
# `@tool` 把函数注册成可被 ToolNode 调用的工具。
from langchain_core.tools import tool
# ToolRuntime 提供当前 tool_call 的上下文信息。
from langgraph.prebuilt import InjectedState, ToolRuntime
# Command 用于在工具执行后把更新写回状态。
from langgraph.types import Command

# 导入 LightRAG 服务单例获取函数；真正的图谱检索逻辑在服务层。
from app.middleware.observability import log_structured_event
from app.services.graph.explainable_answer_service import get_explainable_answer_service
from app.services.graph.neo4j_client import Neo4jDegradedError
from app.services.graph.research_object_query_service import get_research_object_query_service
from app.services.lightrag_service import get_lightrag_service
from app.services.retrieval_service import resolve_graph_library_id

from app.config import get_settings


logger = logging.getLogger(__name__)

# 当前允许的图谱查询模式集合。
VALID_GRAPH_MODES = {"local", "global", "hybrid"}

# 这个文件把 GraphRAG 检索包装成 LangGraph 工具，与向量检索工具是平级关系。


def _merge_source_status(
    source_status: dict[str, str] | None,
    *,
    source: str,
    status: str,
) -> dict[str, str]:
    merged = dict(source_status or {})
    merged[source] = status
    return merged


def _merge_source_status_reasons(
    source_status_reasons: dict[str, str] | None,
    *,
    source: str,
    reason: str,
) -> dict[str, str]:
    """合并 source_status_reasons 单字段。

    Phase 1 R2.4：把异常类名写入 source_status_reasons.<source>，
    供 analyze 节点渲染 graph_hint_section（"图谱线索本次不可用：<reason>"）。
    """
    merged = dict(source_status_reasons or {})
    merged[source] = reason
    return merged


def _merge_retrieval_mode(current_mode: str, *, source: str) -> str:
    normalized_current = str(current_mode or "").strip().lower()
    if source == "graph":
        return "fusion" if normalized_current in {"milvus", "fusion"} else "graph"
    if source == "neo4j":
        return "fusion" if normalized_current in {"milvus", "graph", "fusion"} else "graph"
    return normalized_current or source


def _merge_graph_context(existing_context: str, new_context: str) -> str:
    existing = str(existing_context or "").strip()
    new = str(new_context or "").strip()
    if not new:
        return existing
    if not existing:
        return new
    if new in existing:
        return existing
    return f"{existing}\n\n{new}"


def _project_paper_ids(structured_query: dict | None) -> list[str]:
    structured = structured_query if isinstance(structured_query, dict) else {}
    paper_ids: list[str] = []
    for raw in (
        structured.get("paper_ids"),
        ((structured.get("resolved_scope") or {}) if isinstance(structured.get("resolved_scope"), dict) else {}).get("paper_ids"),
    ):
        if not isinstance(raw, list):
            continue
        for item in raw:
            paper_id = str(item or "").strip()
            if paper_id and paper_id not in paper_ids:
                paper_ids.append(paper_id)
    return paper_ids


@tool("graph_search")
async def graph_search_tool(
    query: str,
    mode: str = "hybrid",
    structured_query: Annotated[dict | None, InjectedState("structured_query")] = None,
    graph_context: Annotated[str | None, InjectedState("graph_context")] = None,
    source_status: Annotated[dict[str, str] | None, InjectedState("source_status")] = None,
    retrieval_mode: Annotated[str | None, InjectedState("retrieval_mode")] = None,
    runtime: ToolRuntime = None,
) -> Command:
    """
    知识图谱检索工具。

    用于查询 LightRAG 中的实体、关系和主题概览信息。

    Args:
        query: 用户查询文本。
        mode: 图检索模式，仅支持 local、global、hybrid。
    """
    # 对外部传入的 mode 做兜底校验，非法值统一回退为 hybrid。
    actual_mode = mode if mode in VALID_GRAPH_MODES else "hybrid"
    # 05-23-rag-recall-coverage-guard-fix R3 防护：用 settings.rag_graph_query_timeout_seconds
    # 包一层 asyncio.wait_for；LightRAG 内部 LLM keyword 抽取若挂死（外部 mimo2api 不稳），
    # 这里超时优雅失败成 source_status.graph=failed，让 vector 部分的答案仍能返回，
    # 避免外层 180s 整体超时导致用户拿不到任何答案。
    settings = get_settings()
    graph_timeout = float(getattr(settings, "rag_graph_query_timeout_seconds", 120.0) or 0.0)
    try:
        # 真正的图谱检索由 LightRAG 服务执行；这里不直接碰索引文件或图谱存储。
        coro = get_lightrag_service().query(
            query,
            mode=actual_mode,
            top_k=5,
            library_id=resolve_graph_library_id(structured_query),
        )
        if graph_timeout > 0:
            result = await asyncio.wait_for(coro, timeout=graph_timeout)
        else:
            result = await coro
        # 把 GraphRAG 返回的聚合文本包装成工具消息文本。
        content = (
            f'graph_search("{query}", mode={actual_mode}):\n{result}'
            if result
            else f'graph_search("{query}", mode={actual_mode}): 未找到相关知识图谱信息'
        )
        # 成功时 failure_reason 置空。
        failure_reason = ""
        status = "ok" if result else "empty"
    except asyncio.TimeoutError:
        # 显式可见的超时降级，不静默吞——写入 source_status_reasons 让上层渲染
        # "图谱线索本次不可用：&lt;reason&gt;" 而不是让整条请求 180s 超时。
        timeout_msg = f"LightRAG query timed out after {graph_timeout:.1f}s"
        content = f'graph_search("{query}", mode={actual_mode}): 执行超时（{timeout_msg}）'
        failure_reason = f'graph_search timeout for "{query}": {timeout_msg}'
        result = ""
        status = "failed"
    except Exception as exc:
        # 工具层不抛异常打断整条 ReAct 链，而是把失败信息回写给状态和消息流。
        content = f'graph_search("{query}", mode={actual_mode}): 执行失败: {exc}'
        failure_reason = f'graph_search failed for "{query}": {exc}'
        result = ""
        status = "failed"
    return Command(
        update={
            # 用 ToolMessage 把工具结果追加回图状态消息列表。
            "messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
            "graph_context": _merge_graph_context(str(graph_context or ""), str(result or "")),
            "source_status": _merge_source_status(source_status, source="graph", status=status),
            "retrieval_mode": _merge_retrieval_mode(str(retrieval_mode or ""), source="graph"),
            "graph_mode": actual_mode,
            # 同时把失败摘要放到状态里，供上层节点决定后续动作。
            "failure_reason": failure_reason,
            "graph_call_count": 1,
        }
    )


@tool("neo4j_graph_search")
async def neo4j_graph_search_tool(
    query: str = "",
    structured_query: Annotated[dict | None, InjectedState("structured_query")] = None,
    graph_evidence: Annotated[list[dict[str, Any]] | None, InjectedState("graph_evidence")] = None,
    source_status: Annotated[dict[str, str] | None, InjectedState("source_status")] = None,
    source_status_reasons: Annotated[
        dict[str, str] | None, InjectedState("source_status_reasons")
    ] = None,
    retrieval_mode: Annotated[str | None, InjectedState("retrieval_mode")] = None,
    runtime: ToolRuntime = None,
) -> Command:
    """
    读取项目范围内的 Neo4j 结构关系和最小解释路径。

    这里不做复杂图推理，只把当前项目范围内最关键的对象关系与路径摘要暴露给模型。

    Phase 1 R2 优雅降级：
    - 客户端层（neo4j_client.py）捕获白名单异常 → 抛 Neo4jDegradedError
    - 本工具捕获 Neo4jDegradedError，把 source_status.neo4j 标记为 "failed"，
      把异常类名写入 source_status_reasons.neo4j，发结构化日志，**不**向上抛
    - 白名单外异常仍走原有 except Exception 分支，行为不变
    """
    paper_ids = _project_paper_ids(structured_query)
    if not paper_ids:
        content = 'neo4j_graph_search(): 当前范围内没有可查询的 paper_ids'
        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
                "source_status": _merge_source_status(source_status, source="neo4j", status="empty"),
                "failure_reason": "",
            }
        )
    try:
        subgraph = await get_research_object_query_service().query_project_subgraph(paper_ids)
        evidence_bundle = await get_explainable_answer_service().build_bundle(
            paper_ids=paper_ids,
            query=query or "project graph evidence",
            path_type="support",
        )
        graph_paths = evidence_bundle.get("graph_evidence", [])
        path_summaries = [
            str(item.get("summary") or "").strip()
            for item in graph_paths
            if isinstance(item, dict) and str(item.get("summary") or "").strip()
        ]
        meta = subgraph.get("meta") if isinstance(subgraph, dict) else {}
        content_lines = [
            f'neo4j_graph_search("{query}"):' if query else "neo4j_graph_search():",
            f'- papers: {len(subgraph.get("papers", [])) if isinstance(subgraph, dict) else 0}',
            f'- nodes: {int((meta or {}).get("node_count") or 0)}',
            f'- edges: {int((meta or {}).get("edge_count") or 0)}',
        ]
        if path_summaries:
            content_lines.append("- evidence paths:")
            content_lines.extend(f"  - {summary}" for summary in path_summaries[:3])
        else:
            content_lines.append("- evidence paths: (none)")
        return Command(
            update={
                "messages": [ToolMessage(content="\n".join(content_lines), tool_call_id=runtime.tool_call_id)],
                "graph_evidence": [*list(graph_evidence or []), *graph_paths],
                "source_status": _merge_source_status(source_status, source="neo4j", status="ok"),
                "retrieval_mode": _merge_retrieval_mode(str(retrieval_mode or ""), source="neo4j"),
                "failure_reason": "",
                "graph_call_count": 1,
            }
        )
    except Neo4jDegradedError as exc:
        # 白名单内已知失败：标记 source_status.neo4j=failed，记录原因，不向上抛
        log_structured_event(
            logger,
            "rag_source_failed",
            source="neo4j",
            reason=exc.reason,
        )
        content = (
            f'neo4j_graph_search("{query}"): 图谱查询不可用（{exc.reason}）'
            if query
            else f"neo4j_graph_search(): 图谱查询不可用（{exc.reason}）"
        )
        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
                "source_status": _merge_source_status(source_status, source="neo4j", status="failed"),
                "source_status_reasons": _merge_source_status_reasons(
                    source_status_reasons,
                    source="neo4j",
                    reason=exc.reason,
                ),
                "failure_reason": f"neo4j_graph_search degraded: {exc.reason}",
                "graph_call_count": 1,
            }
        )
    except Exception as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f'neo4j_graph_search("{query}"): 执行失败: {exc}',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "source_status": _merge_source_status(source_status, source="neo4j", status="failed"),
                "failure_reason": f'neo4j_graph_search failed for "{query}": {exc}',
                "graph_call_count": 1,
            }
        )
