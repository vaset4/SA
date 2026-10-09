"""
项目研究空间服务。

这一层负责把“项目”真正变成可用的研究空间抽象，核心职责包括：
- 确保底层 schema 已就绪
- 统一用户级 project CRUD
- 管理项目 scope
- 管理项目与 paper/session 的关联

和 route 层相比，这里开始承载真实业务规则；
和 repository 层相比，这里更关注“语义正确性”，而不是 SQL 细节。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.project_models import DEFAULT_PROJECT_PAPER_ROLE, empty_scope
from app.models.research_contracts import ScopeOwnershipError
from app.repositories.project_repository import get_project_repository
from app.services import conversation_service, library_service, paper_db, project_activity_service

_schema_ready = False
_schema_lock = asyncio.Lock()
logger = logging.getLogger(__name__)


async def ensure_project_schema() -> None:
    # project 功能依赖 library schema 和 paper/session 表，因此这里统一做级联初始化。
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return
        await library_service.ensure_library_schema()
        await asyncio.to_thread(paper_db.init_db)
        await get_project_repository().ensure_schema()
        await project_activity_service.ensure_project_activity_schema()
        _schema_ready = True


async def _append_activity_safe(project_id: str, event_type: str, payload: dict[str, Any]) -> None:
    try:
        await project_activity_service.append_activity(project_id, event_type, payload)
    except Exception as exc:
        logger.warning("项目活动流写入失败 project_id=%s event_type=%s: %s", project_id, event_type, exc)


def normalize_project_id(project_id: str | None) -> str:
    """统一处理 project_id 的空值与首尾空白。"""
    return (project_id or "").strip()


async def get_project_or_none(project_id: str | None, *, user_id: str = "anonymous") -> dict[str, Any] | None:
    normalized = normalize_project_id(project_id)
    if not normalized:
        return None
    await ensure_project_schema()
    return await get_project_repository().get_project(normalized, user_id=user_id)


async def require_project(project_id: str | None, *, user_id: str = "anonymous") -> dict[str, Any]:
    normalized = normalize_project_id(project_id)
    if not normalized:
        raise LookupError("project_id 不能为空")
    project = await get_project_or_none(normalized, user_id=user_id)
    if project is None:
        raise LookupError(f"项目不存在: {normalized}")
    return project


async def list_projects(*, status: str | None = None, user_id: str = "anonymous") -> list[dict[str, Any]]:
    await ensure_project_schema()
    return await get_project_repository().list_projects(status=status, user_id=user_id)


async def create_project(payload: dict[str, Any], *, user_id: str = "anonymous") -> dict[str, Any]:
    # 创建 project 的同时，立即补一份 scope，避免后续每个调用方都处理“scope 不存在”。
    await ensure_project_schema()
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("项目名称不能为空")
    default_library_id = (payload.get("default_library_id") or DEFAULT_LIBRARY_ID).strip() or DEFAULT_LIBRARY_ID
    await library_service.require_library(default_library_id, owner=user_id)
    repository = get_project_repository()
    project = await repository.create_project(
        {
            **payload,
            "user_id": user_id,
            "owner": payload.get("owner") or user_id,
            "default_library_id": default_library_id,
        }
    )
    await repository.upsert_scope(project["project_id"], empty_scope(default_library_id))
    return project


async def update_project(project_id: str, payload: dict[str, Any], *, user_id: str = "anonymous") -> dict[str, Any]:
    await ensure_project_schema()
    if "default_library_id" in payload:
        await library_service.require_library(payload.get("default_library_id"), owner=user_id)
    project = await get_project_repository().update_project(project_id, {**payload, "user_id": user_id}, user_id=user_id)
    if project is None:
        raise LookupError(f"项目不存在: {project_id}")
    return project


async def delete_project(project_id: str, *, user_id: str = "anonymous") -> None:
    await ensure_project_schema()
    await require_project(project_id, user_id=user_id)
    deleted = await get_project_repository().delete_project(project_id, user_id=user_id)
    if not deleted:
        raise LookupError(f"项目不存在: {project_id}")


async def get_project_scope(project_id: str, *, user_id: str = "anonymous") -> dict[str, Any]:
    await ensure_project_schema()
    project = await require_project(project_id, user_id=user_id)
    scope = await get_project_repository().get_scope(project_id)
    if scope is not None:
        return scope
    return await get_project_repository().upsert_scope(
        project_id,
        empty_scope(project.get("default_library_id", DEFAULT_LIBRARY_ID)),
    )


async def update_project_scope(project_id: str, payload: dict[str, Any], *, user_id: str = "anonymous") -> dict[str, Any]:
    """
    更新项目 scope。

    这里采用 merge 语义而不是整块替换：未显式给出的字段沿用旧值，
    这样前端可以只更新 scope 的某一部分而不用回传整份结构。
    """
    await ensure_project_schema()
    project = await require_project(project_id, user_id=user_id)
    current = await get_project_scope(project_id, user_id=user_id)
    merged = {
        "library_ids": payload.get("library_ids", current.get("library_ids", [project.get("default_library_id", DEFAULT_LIBRARY_ID)])),
        "paper_ids": payload.get("paper_ids", current.get("paper_ids", [])),
        "excluded_paper_ids": payload.get("excluded_paper_ids", current.get("excluded_paper_ids", [])),
        "year_from": payload.get("year_from", current.get("year_from")),
        "year_to": payload.get("year_to", current.get("year_to")),
        "task_filters": payload.get("task_filters", current.get("task_filters", [])),
        "method_filters": payload.get("method_filters", current.get("method_filters", [])),
    }
    validated_library_ids: list[str] = []
    for raw_library_id in merged["library_ids"]:
        library_id = str(raw_library_id or "").strip()
        if not library_id:
            raise ValueError("知识库范围不能包含空 ID")
        await library_service.require_library(library_id, owner=user_id)
        if library_id not in validated_library_ids:
            validated_library_ids.append(library_id)
    merged["library_ids"] = validated_library_ids
    validated_paper_ids: list[str] = []
    for raw_paper_id in merged["paper_ids"]:
        paper_id = str(raw_paper_id or "").strip()
        if not paper_id:
            raise ValueError("论文范围不能包含空 ID")
        if await asyncio.to_thread(paper_db.get_paper, paper_id, user_id=user_id) is None:
            raise ScopeOwnershipError("论文范围不存在或无权访问")
        if paper_id not in validated_paper_ids:
            validated_paper_ids.append(paper_id)
    merged["paper_ids"] = validated_paper_ids
    return await get_project_repository().upsert_scope(project_id, merged)


async def add_project_paper(
    project_id: str,
    paper_id: str,
    *,
    role: str = DEFAULT_PROJECT_PAPER_ROLE,
    note: str = "",
    user_id: str = "anonymous",
) -> dict[str, Any]:
    await ensure_project_schema()
    await require_project(project_id, user_id=user_id)
    normalized_paper_id = str(paper_id or "").strip()
    if not normalized_paper_id:
        raise ValueError("论文 ID 不能为空")
    paper = await asyncio.to_thread(
        paper_db.get_paper,
        normalized_paper_id,
        user_id=user_id,
    )
    if paper is None:
        raise ScopeOwnershipError("论文不存在或无权访问")
    link = await get_project_repository().link_paper(project_id, normalized_paper_id, role=role, note=note)
    # project_paper_links 和 scope.paper_ids 两套数据都要维护：
    # 前者适合关系型展示，后者适合后续直接生成检索范围。
    scope = await get_project_scope(project_id, user_id=user_id)
    paper_ids = [item for item in scope.get("paper_ids", []) if item]
    if normalized_paper_id not in paper_ids:
        paper_ids.append(normalized_paper_id)
        await update_project_scope(project_id, {"paper_ids": paper_ids}, user_id=user_id)
    await _append_activity_safe(
        project_id,
        "paper_added",
        {
            "paper_id": normalized_paper_id,
            "title": paper.get("title") or normalized_paper_id,
        },
    )
    return link


async def remove_project_paper(project_id: str, paper_id: str, *, user_id: str = "anonymous") -> None:
    await ensure_project_schema()
    await require_project(project_id, user_id=user_id)
    await get_project_repository().unlink_paper(project_id, paper_id)
    # 删除关联后同步缩小 scope，避免检索范围还残留已解绑论文。
    scope = await get_project_scope(project_id, user_id=user_id)
    if paper_id in scope.get("paper_ids", []):
        await update_project_scope(
            project_id,
            {"paper_ids": [item for item in scope.get("paper_ids", []) if item != paper_id]},
            user_id=user_id,
        )


async def list_project_papers(project_id: str, *, user_id: str = "anonymous") -> list[dict[str, Any]]:
    await ensure_project_schema()
    await require_project(project_id, user_id=user_id)
    return await get_project_repository().list_project_papers(project_id)


async def attach_session(
    project_id: str, session_id: str,
    *,
    user_id: str = "anonymous",
    session_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    把 owner-scoped session 幂等挂到项目下。

    Args:
        project_id: 目标项目 ID
        session_id: durable 研究 session ID，也兼容旧 conversation ID
        user_id: 当前登录用户 ID
        session_row: 上游已按 owner 读取的 session，传入后避免跨事件循环重复查询

    Returns:
        dict[str, Any]: 已写入并重新读取的 project-session 关联
    """
    await ensure_project_schema()
    await require_project(project_id, user_id=user_id)
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        raise LookupError("session_id 不能为空")
    resolved_session_id = normalized_session_id
    if session_row is not None:
        row_session_id = str(session_row.get("session_id") or "").strip()
        row_user_id = str(session_row.get("user_id") or "").strip()
        if row_session_id != resolved_session_id or row_user_id != user_id:
            raise LookupError("会话不存在或无权访问")
    else:
        session_row = await asyncio.to_thread(
            paper_db.get_session,
            resolved_session_id,
            user_id=user_id,
        )
    if session_row is None:
        conversation = await conversation_service.get_conversation(resolved_session_id, user_id=user_id)
        conversation_session_id = str((conversation or {}).get("session_id") or "").strip()
        if conversation_session_id:
            resolved_session_id = conversation_session_id
            session_row = await asyncio.to_thread(
                paper_db.get_session,
                resolved_session_id,
                user_id=user_id,
            )
    if session_row is None:
        raise LookupError(f"会话不存在，无法绑定到项目: {normalized_session_id}")
    link = await get_project_repository().link_session(project_id, resolved_session_id)
    if not link:
        raise RuntimeError("项目会话关联写入后无法读取")
    await _append_activity_safe(
        project_id,
        "internal_research_started",
        {
            "session_id": resolved_session_id,
            "topic": session_row.get("topic") or "未命名研究",
            "paper_ids": list(session_row.get("paper_ids") or []),
        },
    )
    return link


async def list_project_sessions(project_id: str, *, user_id: str = "anonymous") -> list[dict[str, Any]]:
    await ensure_project_schema()
    await require_project(project_id, user_id=user_id)
    return await get_project_repository().list_project_sessions(project_id)


async def list_project_activities(project_id: str, *, limit: int = 50, user_id: str = "anonymous") -> list[dict[str, Any]]:
    await ensure_project_schema()
    await require_project(project_id, user_id=user_id)
    return await project_activity_service.list_activities(project_id, limit=limit)


async def record_paper_analysis_completed(paper: dict[str, Any], *, user_id: str = "anonymous") -> None:
    await ensure_project_schema()
    paper_id = str(paper.get("paper_id") or "").strip()
    if not paper_id:
        return
    project_ids = await get_project_repository().list_project_ids_by_paper(paper_id)
    summary = str(((paper.get("analysis") or {}).get("tldr") or (paper.get("analysis") or {}).get("method") or "")).strip()
    for project_id in project_ids:
        await _append_activity_safe(
            project_id,
            "analysis_completed",
            {
                "paper_id": paper_id,
                "title": paper.get("title") or paper_id,
                "summary": summary or "论文已具备进入正式研究范围的基础材料。",
            },
        )


async def remove_paper_references(paper_id: str, *, user_id: str = "anonymous") -> dict[str, Any]:
    """清理项目 scope 中残留的 `paper_ids/excluded_paper_ids`。"""
    await ensure_project_schema()
    project_ids: list[str] = []
    for project in await list_projects(user_id=user_id):
        project_id = str(project.get("project_id") or "").strip()
        if not project_id:
            continue
        scope = await get_project_scope(project_id, user_id=user_id)
        paper_ids = [item for item in scope.get("paper_ids", []) if item]
        excluded_paper_ids = [item for item in scope.get("excluded_paper_ids", []) if item]
        next_paper_ids = [item for item in paper_ids if item != paper_id]
        next_excluded_paper_ids = [item for item in excluded_paper_ids if item != paper_id]
        if next_paper_ids == paper_ids and next_excluded_paper_ids == excluded_paper_ids:
            continue
        await update_project_scope(
            project_id,
            {
                "paper_ids": next_paper_ids,
                "excluded_paper_ids": next_excluded_paper_ids,
            },
            user_id=user_id,
        )
        project_ids.append(project_id)
    return {
        "project_count": len(project_ids),
        "project_ids": project_ids,
    }
