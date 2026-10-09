"""论文跨存储幂等删除任务 handler。

Author: xxx
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.config import get_settings
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.repositories.paper_deletion_repository import get_paper_deletion_repository
from app.repositories.source_access_repository import get_source_access_repository
from app.services.cache import get_cache
from app.services.graph import get_neo4j_maintenance_service
from app.services.lightrag_service import get_lightrag_service
from app.services.milvus_store import get_milvus_store
from app.services.paper_file_cleanup_service import delete_managed_paper_files
from app.services.source_io_service import run_source_io

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PaperDeletionContext:
    """保存一次删除执行所需的稳定论文身份与文件信息。"""

    task_id: str
    paper_id: str
    user_id: str
    library_id: str
    pdf_path: str
    already_deleted: bool = False


DeleteStep = Callable[[PaperDeletionContext], Awaitable[dict[str, Any]]]
UPLOAD_STOP_POLL_SECONDS = 1.0


async def _load_context(task_run: dict[str, Any]) -> PaperDeletionContext:
    """
    校验任务 owner、payload 和论文 deleting 状态，并生成不可变执行上下文。

    Args:
        task_run: worker 已领取的 owner-aware 任务记录
    Returns:
        经过状态校验的删除执行上下文
    """
    payload = task_run.get("payload") or {}
    task_id = str(task_run.get("task_id") or "").strip()
    paper_id = str(payload.get("paper_id") or "").strip()
    user_id = str(task_run.get("user_id") or "").strip()
    if not task_id or not paper_id or not user_id:
        raise ValueError("paper_delete 任务缺少 task_id、paper_id 或 user_id")
    if str(payload.get("user_id") or "").strip() != user_id:
        raise ValueError("paper_delete 任务 owner 与 payload 不一致")
    state = await get_paper_deletion_repository().get_deletion_state(paper_id, user_id=user_id)
    if state is None:
        return PaperDeletionContext(task_id, paper_id, user_id, DEFAULT_LIBRARY_ID, "", True)
    if state.get("deletion_status") != "deleting":
        raise RuntimeError(f"论文不处于 deleting 状态: {paper_id}")
    if str(state.get("deletion_task_id") or "") != task_id:
        raise RuntimeError(f"论文删除任务已被替换: {paper_id}")
    return PaperDeletionContext(
        task_id,
        paper_id,
        user_id,
        str(state.get("library_id") or DEFAULT_LIBRARY_ID),
        str(state.get("pdf_path") or ""),
    )


async def _delete_milvus(context: PaperDeletionContext) -> dict[str, Any]:
    """
    清空进程缓存并幂等删除论文的全部 Milvus chunk。

    Args:
        context: 已校验的删除执行上下文
    Returns:
        Milvus 删除步骤结果
    """
    repository = get_paper_deletion_repository()
    while not await repository.stop_uploads(context.paper_id, user_id=context.user_id):
        await asyncio.sleep(UPLOAD_STOP_POLL_SECONDS)
    get_cache().clear_all()
    store = get_milvus_store()
    await asyncio.to_thread(store.delete_paper, context.paper_id)
    await asyncio.to_thread(store.flush_collection)
    return {"status": "done", "paper_id": context.paper_id}


async def _delete_lightrag(context: PaperDeletionContext) -> dict[str, Any]:
    """
    按稳定 paper_id/doc_id 幂等删除 LightRAG 文档。

    Args:
        context: 已校验的删除执行上下文
    Returns:
        LightRAG 删除步骤结果
    """
    if not get_settings().graph_rag_enabled:
        return {"status": "skipped", "reason": "disabled"}
    result = await get_lightrag_service().delete_by_doc_id(
        context.paper_id,
        library_id=context.library_id,
    )
    return {**result, "status": "done"}


async def _delete_neo4j(context: PaperDeletionContext) -> dict[str, Any]:
    """
    幂等删除 Neo4j 论文子图及无关联实体。

    Args:
        context: 已校验的删除执行上下文
    Returns:
        Neo4j 删除步骤结果
    """
    result = await get_neo4j_maintenance_service().delete_paper_subgraph(context.paper_id)
    status = "done" if result.get("enabled", True) else "skipped"
    return {**result, "status": status}


async def _delete_database_references(context: PaperDeletionContext) -> dict[str, Any]:
    """
    在一个 PostgreSQL 事务中清理论文的项目、会话、笔记和记忆引用。

    Args:
        context: 已校验的删除执行上下文
    Returns:
        各类数据库引用清理统计
    """
    result = await get_paper_deletion_repository().cleanup_database_references(
        context.paper_id,
        user_id=context.user_id,
    )
    return {"status": "done", **result}


async def _delete_files(context: PaperDeletionContext) -> dict[str, Any]:
    """
    在线程池内删除 DATA_DIR 受控原文件和 MinerU 产物。

    Args:
        context: 已校验的删除执行上下文
    Returns:
        文件删除数量与对象类型
    """
    # 1. 【论文删除】【文件占用】请求已撤销读取资格，排他占用仅等待已经启动的文件工作
    async with get_source_access_repository().file_access((context.paper_id,), exclusive=True):
        result = await run_source_io(delete_managed_paper_files, context.paper_id, pdf_path=context.pdf_path)
    return {"status": "done", **result}


async def _delete_paper_record(context: PaperDeletionContext) -> dict[str, Any]:
    """
    在所有派生数据清理完成后，以 owner/task CAS 最终硬删除论文主记录。

    Args:
        context: 已校验的删除执行上下文
    Returns:
        最终硬删除结果
    """
    repository = get_paper_deletion_repository()
    deleted = await repository.hard_delete_paper(
        context.paper_id,
        user_id=context.user_id,
        task_id=context.task_id,
    )
    if not deleted and await repository.get_deletion_state(context.paper_id, user_id=context.user_id) is not None:
        raise RuntimeError(f"论文最终硬删除 CAS 失败: {context.paper_id}")
    return {"status": "done", "paper_id": context.paper_id, "already_deleted": not deleted}


async def _persist_failure(context: PaperDeletionContext, step_name: str, error: Exception) -> None:
    """
    把失败步骤写回 paper 生命周期；状态写入失败时显式抛错。

    Args:
        context: 已校验的删除执行上下文
        step_name: 当前固定删除步骤名
        error: 当前步骤原始异常
    Returns:
        无返回值
    """
    persisted = await get_paper_deletion_repository().mark_deletion_failed(
        context.paper_id,
        user_id=context.user_id,
        task_id=context.task_id,
        failed_step=step_name,
        error=str(error),
    )
    if not persisted:
        raise RuntimeError(f"论文删除失败状态 CAS 失败: {context.paper_id}") from error


async def run_paper_delete_task(task_run: dict[str, Any]) -> dict[str, Any]:
    """
    按固定顺序执行论文跨存储删除，失败时保留当前步骤供新任务重试。

    Args:
        task_run: worker 已领取的 paper_delete 任务记录
    Returns:
        可序列化的删除结果和已执行步骤
    """
    # 1. 先校验 owner、任务绑定与 deleting 状态
    context = await _load_context(task_run)
    if context.already_deleted:
        return {"paper_id": context.paper_id, "status": "deleted", "already_deleted": True, "steps": []}
    # 2. 外部存储和引用按固定顺序清理，论文主记录始终排在最后
    steps: tuple[tuple[str, DeleteStep], ...] = (
        ("milvus", _delete_milvus),
        ("lightrag", _delete_lightrag),
        ("neo4j", _delete_neo4j),
        ("database_references", _delete_database_references),
        ("files", _delete_files),
        ("paper_record", _delete_paper_record),
    )
    completed: list[dict[str, Any]] = []
    for step_name, step in steps:
        try:
            completed.append({"step": step_name, **await step(context)})
        except Exception as exc:
            logger.error(
                "【论文删除】【步骤失败】paper_id=%s task_id=%s step=%s error=%s",
                context.paper_id,
                context.task_id,
                step_name,
                exc,
                exc_info=True,
            )
            # 3. 任一步失败即停止，并持久化失败步骤供新任务重试
            await _persist_failure(context, step_name, exc)
            raise
    logger.info(
        "【论文删除】【任务完成】paper_id=%s task_id=%s",
        context.paper_id,
        context.task_id,
    )
    return {"paper_id": context.paper_id, "status": "deleted", "steps": completed}
