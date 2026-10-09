"""
项目作用域解析服务。

如果说 `project_service` 负责“管理 project”，
那这个文件负责把 project 变成“本次请求可直接消费的检索上下文”。

典型职责有两个：
1. 把持久化 scope 解析成标准化内部结构
2. 把这个结构进一步压缩成 filter_expr / paper_ids，供 RAG 层直接使用
"""

from __future__ import annotations

from typing import Any

from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.project_models import ResolvedProjectScope
from app.models.research_contracts import RetrievalScopeError, ScopeOwnershipError
from app.repositories.paper_repository import get_paper_repository
from app.services import library_service
from app.services.project_service import get_project_scope, require_project


def _normalize_paper_ids(paper_ids: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_paper_id in paper_ids or []:
        paper_id = str(raw_paper_id or "").strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        normalized.append(paper_id)
    return normalized


async def _resolve_owned_paper_ids(
    paper_ids: list[str],
    user_id: str,
) -> tuple[list[str], list[str]]:
    """一次批量查询论文归属，并按请求顺序拆分可用与拒绝 ID。"""
    papers = await get_paper_repository().get_papers_by_ids(paper_ids, user_id=user_id)
    owned_ids = {str(paper["paper_id"]) for paper in papers}
    accepted = [paper_id for paper_id in paper_ids if paper_id in owned_ids]
    rejected = [paper_id for paper_id in paper_ids if paper_id not in owned_ids]
    return accepted, rejected


def _raise_rejected_paper_ids(rejected_paper_ids: list[str]) -> None:
    if not rejected_paper_ids:
        return
    raise ScopeOwnershipError("论文范围不存在或无权访问")


def build_scope_filter_expr(scope: ResolvedProjectScope) -> str:
    """
    把解析后的作用域转换成 Milvus filter_expr。

    这里生成的是“查询时可直接使用”的字符串表达式，
    因此需要把 paper/year 约束提前编译好，避免每个 route/service 重复拼装。
    """

    escaped = [
        str(paper_id).replace("\\", "\\\\").replace('"', '\\"')
        for paper_id in scope["paper_ids"]
        if paper_id
    ]
    if not escaped:
        raise RetrievalScopeError("本地检索需要至少一篇已授权论文")
    filters: list[str] = ["(" + " or ".join(f'paper_id == "{paper_id}"' for paper_id in escaped) + ")"]
    if scope["year_from"] is not None:
        filters.append(f"year >= {scope['year_from']}")
    if scope["year_to"] is not None:
        filters.append(f"year <= {scope['year_to']}")
    return " and ".join(filters)


async def resolve_project_scope(project_id: str | None, *, user_id: str = "anonymous") -> ResolvedProjectScope | None:
    """
    把 project_id 解析为统一内部作用域。

    关键点：
    - 如果 scope 没显式给 paper_ids，就根据 library_ids 展开
    - 排除项 `excluded_paper_ids` 会在这里真正生效
    - 返回值保证字段齐全，便于下游不用再补默认值
    """

    normalized = (project_id or "").strip()
    if not normalized:
        return None
    project = await require_project(normalized, user_id=user_id)
    scope = await get_project_scope(normalized, user_id=user_id)
    library_ids = _normalize_paper_ids(scope.get("library_ids")) or [
        project.get("default_library_id", DEFAULT_LIBRARY_ID)
    ]
    for library_id in library_ids:
        try:
            await library_service.require_library(library_id, owner=user_id)
        except LookupError as exc:
            raise ScopeOwnershipError("知识库范围不存在或无权访问") from exc
    paper_ids = _normalize_paper_ids(scope.get("paper_ids"))
    if not paper_ids:
        # scope 允许只配置 library，不必在保存时展开全部 paper_id；
        # 真正展开推迟到这里，避免持久化层长期保存大量冗余数据。
        for library_id in library_ids:
            paper_ids.extend(
                await get_paper_repository().list_paper_ids_by_library(
                    library_id,
                    user_id=user_id,
                )
            )
    owned_paper_ids, rejected_paper_ids = await _resolve_owned_paper_ids(paper_ids, user_id)
    _raise_rejected_paper_ids(rejected_paper_ids)
    excluded = {item for item in scope.get("excluded_paper_ids", []) if item}
    resolved_paper_ids: list[str] = []
    seen: set[str] = set()
    for paper_id in owned_paper_ids:
        if not paper_id or paper_id in excluded or paper_id in seen:
            continue
        seen.add(paper_id)
        resolved_paper_ids.append(paper_id)
    return {
        "project_id": normalized,
        "library_ids": library_ids,
        "paper_ids": resolved_paper_ids,
        "excluded_paper_ids": list(excluded),
        "year_from": scope.get("year_from"),
        "year_to": scope.get("year_to"),
        "task_filters": [item for item in scope.get("task_filters", []) if item],
        "method_filters": [item for item in scope.get("method_filters", []) if item],
    }


async def _build_project_context_scope(
    project_id: str | None, *, requested_paper_ids: list[str] | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """核对显式论文归属并构造请求范围；返回项目、论文、过滤条件和授权结果。

    显式指定的自有论文可以来自项目外；未指定论文时采用项目的默认范围。
    """
    requested_ids = _normalize_paper_ids(requested_paper_ids)
    accepted_ids, rejected_ids = (
        await _resolve_owned_paper_ids(requested_ids, user_id) if requested_ids else ([], [])
    )
    _raise_rejected_paper_ids(rejected_ids)
    scope = await resolve_project_scope(project_id, user_id=user_id)
    effective_ids = accepted_ids or (scope["paper_ids"] if scope is not None else [])
    resolved_scope = None
    if scope is not None:
        resolved_scope = {**scope, "paper_ids": effective_ids}
    elif effective_ids:
        resolved_scope = {
            "project_id": "", "library_ids": [], "paper_ids": effective_ids, "excluded_paper_ids": [],
            "year_from": None, "year_to": None, "task_filters": [], "method_filters": [],
        }
    return {
        "user_id": user_id,
        "project_id": scope["project_id"] if scope is not None else "",
        "library_ids": scope["library_ids"] if scope is not None else [],
        "resolved_scope": resolved_scope,
        "filter_expr": build_scope_filter_expr(resolved_scope) if resolved_scope and effective_ids else "",
        "paper_ids": effective_ids,
        "local_retrieval_allowed": bool(effective_ids),
        "accepted_paper_ids": accepted_ids,
        "rejected_paper_ids": rejected_ids,
    }


async def build_project_context(project_id: str | None, *, requested_paper_ids: list[str] | None = None,
                                user_id: str = "anonymous") -> dict[str, Any]:
    """建立授权项目范围并固定当前发布批次，重建中的新结果不进入本轮检索。"""
    from app.services.upload_read_service import attach_publication_snapshot

    scope = await _build_project_context_scope(project_id, requested_paper_ids=requested_paper_ids, user_id=user_id)
    return await attach_publication_snapshot(scope)
