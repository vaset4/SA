"""
后台任务创建、查询与进度事件服务。

这是后台任务体系的控制面：创建 task、记录事件、更新进度、完成/失败/取消状态迁移。
真正执行任务的是 worker 和具体 task handler。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from app.config import get_settings
from app.models.task_models import TaskEvent, TaskRun
from app.repositories.paper_deletion_repository import get_paper_deletion_repository
import app.services.paper_db as paper_db

DEFAULT_LIST_LIMIT = 50
DEFAULT_EVENT_LIMIT = 200


async def append_event(
    task_id: str,
    event_type: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> TaskEvent:
    """为任务追加一条结构化事件。"""
    return await asyncio.to_thread(
        paper_db.add_task_event,
        task_id,
        event_type,
        message,
        payload or {},
    )


async def create_task(
    task_type: str,
    payload: dict[str, Any] | None = None,
    *,
    message: str = "任务已入队",
    task_id: str | None = None,
    user_id: str,
) -> TaskRun:
    """创建一条后台任务；user_id 为服务端归属，必填。"""
    task = await asyncio.to_thread(
        paper_db.create_task_run,
        task_type,
        payload or {},
        message,
        task_id=task_id,
        user_id=user_id,
    )
    await append_event(
        task["task_id"],
        "queued",
        message,
        {"task_type": task_type},
    )
    return task


def require_upload_owner(user_id: object) -> str:
    """【上传】【用户归属】在文件暂存或任务执行前核对持久 owner。

    Args:
        user_id: 服务端认证或任务记录提供的用户 ID。

    Returns:
        已规范化且允许执行上传的 owner。

    Raises:
        ValueError: owner 缺失，或账号模式下仍使用 anonymous。
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("上传任务缺少有效 owner")
    owner_id = user_id.strip()
    if owner_id == "anonymous" and get_settings().auth_enabled:
        raise ValueError("旧 anonymous 上传需先明确归属，不能在账号模式直接继续")
    return owner_id


def get_upload_task_owner(task_run: dict[str, Any]) -> str:
    """【上传】【用户归属】以任务表 owner 为准，并拒绝旧 payload 中的冲突身份。

    Args:
        task_run: worker 领取的持久任务记录。

    Returns:
        任务表中经过核对的 owner。

    Raises:
        ValueError: 任务 owner 无效、payload 非对象或其中的旧身份发生冲突。
    """
    owner_id = require_upload_owner(task_run.get("user_id"))
    payload = task_run.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("上传任务 payload 必须是对象")
    if "user_id" in payload and require_upload_owner(payload["user_id"]) != owner_id:
        raise ValueError("上传任务 owner 与 payload 用户不一致")
    return owner_id


async def create_upload_task(
    filename: str,
    content: bytes,
    *,
    library_id: str | None = None,
    project_id: str | None = None,
    user_id: str,
    request_id: str | None = None,
    graph_requested: bool = False,
) -> TaskRun:
    """【上传】【创建任务】为认证用户暂存原文件并创建持久任务。

    Args:
        filename: 原文件名。
        content: 已接收的原文件内容。
        library_id: 目标文献库，可空。
        project_id: 目标项目，可空。
        user_id: 认证入口解析的用户 ID。

    Returns:
        保存了唯一用户归属和稳定 paper_id 的任务记录。
    """
    from typing import cast
    from app.services.upload_service import receive_upload

    owner_id = require_upload_owner(user_id)
    task = await receive_upload(filename, content, library_id=library_id, project_id=project_id,
        user_id=owner_id, request_id=request_id or str(uuid.uuid4()), graph_requested=graph_requested)
    return cast(TaskRun, task)


async def create_research_task(query: str, session_id: str | None = None, *, user_id: str) -> TaskRun:
    """创建 deep research 任务；owner 必填。"""
    return await create_task(
        "deep_research",
        {"query": query, "session_id": session_id, "user_id": user_id},
        message="deep research 任务已入队",
        user_id=user_id,
    )

async def create_lightrag_task(
    paper_id: str,
    *,
    user_id: str,
) -> TaskRun:
    """【上传】【可选图谱】只从当前账号已发布的论文排队建图，返回同一恢复体系的任务。"""
    from typing import cast
    from app.services.upload_service import request_upload_operation

    return cast(TaskRun, await request_upload_operation(paper_id, "graph", user_id=require_upload_owner(user_id)))


async def create_paper_analysis_task(paper_id: str, *, user_id: str) -> TaskRun:
    """创建手动论文分析任务；owner 必填。"""
    await asyncio.to_thread(
        paper_db.update_analysis_status,
        paper_id,
        "queued",
        clear_analysis=True,
        user_id=user_id,
    )
    return await create_task(
        "paper_analysis",
        {"paper_id": paper_id, "user_id": user_id},
        message="论文 AI 分析任务已入队",
        user_id=user_id,
    )


async def create_paper_delete_task(paper_id: str, *, user_id: str) -> TaskRun:
    """创建论文删除任务；论文状态和任务记录由同一数据库事务写入。"""
    task = await get_paper_deletion_repository().request_deletion(paper_id, user_id=user_id)
    if task is None:
        raise LookupError("论文不存在")
    return task


async def get_task_or_none(task_id: str, *, user_id: str) -> TaskRun | None:
    """按归属读取单条任务；跨用户或不存在返回 None。"""
    from typing import cast
    from app.repositories.upload_repository import get_upload_repository

    task = await asyncio.to_thread(paper_db.get_task_run, task_id, user_id=user_id)
    return cast(TaskRun, await get_upload_repository().describe_task(task)) if task is not None else None


async def get_task_or_raise(task_id: str, *, user_id: str) -> TaskRun:
    """按归属读取单条任务，不存在或越权时抛错。"""
    task = await get_task_or_none(task_id, user_id=user_id)
    if task is None:
        raise LookupError(f"任务不存在: {task_id}")
    return task


async def list_tasks(
    statuses: list[str] | None = None,
    *,
    user_id: str,
    limit: int = DEFAULT_LIST_LIMIT,
    offset: int = 0,
) -> list[TaskRun]:
    """按当前用户分页列出任务。"""
    tasks = await asyncio.to_thread(
        paper_db.list_task_runs,
        statuses,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    from typing import cast
    from app.repositories.upload_repository import get_upload_repository

    repository = get_upload_repository()
    return cast(list[TaskRun], [await repository.describe_task(task) for task in tasks])


async def list_task_events(task_id: str, *, user_id: str, limit: int = DEFAULT_EVENT_LIMIT) -> list[TaskEvent]:
    """按任务归属列出事件。"""
    return await asyncio.to_thread(paper_db.list_task_events, task_id, user_id=user_id, limit=limit)


async def report_progress(
    task_id: str,
    progress: int,
    message: str,
    payload: dict[str, Any] | None = None,
) -> TaskRun:
    """更新任务进度，并追加进度事件。"""
    normalized_progress = max(0, min(100, int(progress)))
    task = await asyncio.to_thread(
        paper_db.update_task_progress,
        task_id,
        normalized_progress,
        message,
    )
    if task is None:
        raise LookupError(f"任务不存在: {task_id}")
    await append_event(
        task_id,
        "progress",
        message,
        {"progress": normalized_progress, **(payload or {})},
    )
    return task


async def complete_task(
    task_id: str,
    result: Any,
    *,
    worker_id: str,
    message: str = "任务执行完成",
) -> TaskRun | None:
    """由指定 worker 写入完成终态；租约 CAS 失败时返回 None，且不追加完成事件。"""
    if isinstance(result, dict) and ("_upload_publication" in result or "_upload_graph" in result):
        from typing import cast
        from app.services.ingest.upload_pipeline import commit_upload_result

        return cast(TaskRun, await commit_upload_result(result, task_id=task_id, worker_id=worker_id))
    task = await asyncio.to_thread(
        paper_db.complete_task_run,
        task_id,
        result,
        message,
        worker_id=worker_id,
    )
    if task is None:
        return None
    await append_event(task_id, "completed", message, {"result": result})
    return task


async def fail_task(
    task_id: str,
    error: str,
    *,
    worker_id: str,
    message: str = "任务执行失败",
) -> TaskRun | None:
    """由指定 worker 写入失败终态；租约 CAS 失败时返回 None，且不追加失败事件。"""
    task = await asyncio.to_thread(
        paper_db.fail_task_run,
        task_id,
        error,
        message,
        worker_id=worker_id,
    )
    if task is None:
        return None
    await append_event(task_id, "failed", message, {"error": error})
    return task


async def cancel_task(task_id: str, *, user_id: str, message: str = "已请求取消任务") -> TaskRun:
    """
    请求取消任务。

    它表示“希望取消”，不一定代表 handler 已经中断。按归属校验，越权返回 404。
    """
    task = await asyncio.to_thread(paper_db.request_task_cancel, task_id, message, user_id=user_id)
    if task is None:
        raise LookupError(f"任务不存在: {task_id}")
    event_type = "cancelled" if task["status"] == "cancelled" else "cancel_requested"
    if task["task_type"] == "upload_ingest" and task["status"] == "cancelled":
        from app.repositories.upload_repository import get_upload_repository
        await get_upload_repository().record_cancelled(task)
    await append_event(task_id, event_type, message)
    return task


async def mark_cancelled(
    task_id: str,
    *,
    worker_id: str,
    message: str = "任务已取消",
) -> TaskRun | None:
    """由指定 worker 写入取消终态；租约 CAS 失败时返回 None，且不追加取消事件。"""
    task = await asyncio.to_thread(
        paper_db.mark_task_cancelled,
        task_id,
        message,
        worker_id=worker_id,
    )
    if task is None:
        return None
    if task["task_type"] == "upload_ingest":
        from app.repositories.upload_repository import get_upload_repository
        await get_upload_repository().record_cancelled(task)
    await append_event(task_id, "cancelled", message)
    return task
