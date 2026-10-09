"""论文删除生命周期的事务仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.repositories.common import normalize_record
from app.repositories.task_repository import get_task_repository
from app.services.database import get_pool

_repository: "PaperDeletionRepository | None" = None
PAPER_DELETE_TASK_TYPE = "paper_delete"

_DELETE_PROJECT_LINKS_SQL = """
DELETE FROM project_paper_links AS link
USING projects AS project
WHERE link.project_id=project.project_id AND project.user_id=$1 AND link.paper_id=$2
RETURNING link.project_id
"""
_CLEAN_PROJECT_SCOPES_SQL = """
UPDATE project_scopes AS scope
SET paper_ids=COALESCE(scope.paper_ids, '[]'::jsonb) - $2,
    excluded_paper_ids=COALESCE(scope.excluded_paper_ids, '[]'::jsonb) - $2,
    updated_at=NOW()
FROM projects AS project
WHERE scope.project_id=project.project_id AND project.user_id=$1
  AND (COALESCE(scope.paper_ids, '[]'::jsonb) ? $2
       OR COALESCE(scope.excluded_paper_ids, '[]'::jsonb) ? $2)
RETURNING scope.project_id
"""
_CLEAN_SESSIONS_SQL = """
UPDATE sessions
SET paper_ids=COALESCE(paper_ids, '[]'::jsonb) - $2, updated_at=NOW()
WHERE user_id=$1 AND COALESCE(paper_ids, '[]'::jsonb) ? $2
RETURNING session_id
"""
_CLEAN_CONVERSATIONS_SQL = """
UPDATE conversations
SET associated_papers=COALESCE(associated_papers, '[]'::jsonb) - $2, updated_at=NOW()
WHERE user_id=$1 AND COALESCE(associated_papers, '[]'::jsonb) ? $2
RETURNING id
"""
_DELETE_NOTES_SQL = """
DELETE FROM notes WHERE user_id=$1 AND paper_id=$2 RETURNING note_id
"""


class PaperDeletionRepository:
    """负责论文删除任务创建、状态持久化、引用事务和最终硬删除。"""

    async def request_deletion(self, paper_id: str, *, user_id: str) -> dict[str, Any] | None:
        """
        在同一 PostgreSQL 事务中把论文标记为 deleting 并创建 owner-aware 任务。

        Args:
            paper_id: 要删除的论文主键
            user_id: 当前用户主键，用于论文和任务双重归属校验
        Returns:
            已创建或已存在的删除任务；论文不存在时返回 None
        """
        if not paper_id.strip() or not user_id.strip():
            raise ValueError("paper_id 和 user_id 不能为空")
        pool = await get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                paper = await connection.fetchrow(
                    """
                    SELECT paper_id, user_id, library_id, deletion_status, deletion_task_id
                    FROM papers
                    WHERE paper_id=$1 AND user_id=$2
                    FOR UPDATE
                    """,
                    paper_id,
                    user_id,
                )
                if paper is None:
                    return None
                if str(paper["deletion_status"] or "active") == "deleting":
                    return await self._get_existing_task(paper, user_id=user_id, connection=connection)
                return await self._create_task(paper_id, user_id=user_id, connection=connection)

    async def _get_existing_task(self, paper: Any, *, user_id: str, connection: Any) -> dict[str, Any]:
        """读取 deleting 论文已绑定的删除任务，并暴露不一致状态。"""
        task_id = str(paper["deletion_task_id"] or "").strip()
        if not task_id:
            raise RuntimeError(f"论文删除状态缺少 task_id: {paper['paper_id']}")
        task = await get_task_repository().get_task_run(task_id, user_id=user_id, connection=connection)
        if task is None or task.get("task_type") != PAPER_DELETE_TASK_TYPE:
            raise RuntimeError(f"论文删除任务不存在: {task_id}")
        return task

    async def _create_task(self, paper_id: str, *, user_id: str, connection: Any) -> dict[str, Any]:
        """在当前事务连接中创建删除任务、更新论文状态并记录 queued 事件。"""
        task_id = str(uuid.uuid4())
        task_repository = get_task_repository()
        task = await task_repository.create_task_run(
            PAPER_DELETE_TASK_TYPE,
            {"paper_id": paper_id, "user_id": user_id},
            "论文删除任务已入队",
            task_id=task_id,
            user_id=user_id,
            connection=connection,
        )
        await connection.execute(
            """
            UPDATE papers
            SET deletion_status='deleting', deletion_task_id=$3,
                deletion_error='', deletion_failed_step=NULL,
                deletion_requested_at=NOW()
            WHERE paper_id=$1 AND user_id=$2
            """,
            paper_id,
            user_id,
            task_id,
        )
        from app.repositories.memory_source_repository import invalidate_source

        # 1. 【研究记忆】【删除撤权】删除受理与支持失效同事务，不等待后台物理清理
        await invalidate_source(connection, user_id=user_id, paper_id=paper_id, reason=f"证据论文已删除或撤权: {paper_id}")
        await task_repository.add_task_event(
            task_id,
            "queued",
            "论文删除任务已入队",
            {"task_type": PAPER_DELETE_TASK_TYPE, "paper_id": paper_id},
            connection=connection,
        )
        return task

    async def get_deletion_state(self, paper_id: str, *, user_id: str) -> dict[str, Any] | None:
        """读取论文删除状态；仅用于删除执行器和状态诊断。"""
        pool = await get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM papers WHERE paper_id=$1 AND user_id=$2",
                paper_id,
                user_id,
            )
        return normalize_record(row)

    async def mark_deletion_failed(
        self,
        paper_id: str,
        *,
        user_id: str,
        task_id: str,
        failed_step: str,
        error: str,
    ) -> bool:
        """
        将当前删除任务绑定的论文标记为 delete_failed。

        Args:
            paper_id: 失败论文主键
            user_id: 论文 owner
            task_id: 当前删除任务主键
            failed_step: 固定删除步骤名
            error: 原始失败信息
        Returns:
            当前任务仍持有删除状态时返回 True
        """
        pool = await get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE papers
                SET deletion_status='delete_failed', deletion_failed_step=$4,
                    deletion_error=$5
                WHERE paper_id=$1 AND user_id=$2 AND deletion_task_id=$3
                  AND deletion_status='deleting'
                RETURNING paper_id
                """,
                paper_id,
                user_id,
                task_id,
                failed_step,
                error,
            )
        return row is not None

    async def cleanup_database_references(self, paper_id: str, *, user_id: str) -> dict[str, Any]:
        """
        在同一 PostgreSQL 事务中清理论文的全部业务引用。

        Args:
            paper_id: 待清理的论文主键
            user_id: 论文 owner
        Returns:
            各引用类型受影响数量与记忆失效结果
        """
        pool = await get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                project_links = await connection.fetch(_DELETE_PROJECT_LINKS_SQL, user_id, paper_id)
                project_scopes = await connection.fetch(_CLEAN_PROJECT_SCOPES_SQL, user_id, paper_id)
                sessions = await connection.fetch(_CLEAN_SESSIONS_SQL, user_id, paper_id)
                conversations = await connection.fetch(_CLEAN_CONVERSATIONS_SQL, user_id, paper_id)
                notes = await connection.fetch(_DELETE_NOTES_SQL, user_id, paper_id)
                # 1. 【研究记忆】【保留历史】删除受理已撤销复用资格，历史引文和原版本事实继续保留
                memory = {"references_preserved": True}
        return {
            "project_link_count": len(project_links),
            "project_scope_count": len(project_scopes),
            "session_count": len(sessions),
            "conversation_count": len(conversations),
            "note_count": len(notes),
            "memory": memory,
        }

    async def hard_delete_paper(self, paper_id: str, *, user_id: str, task_id: str) -> bool:
        """
        在所有派生清理完成后硬删除当前任务绑定的论文主记录。

        Args:
            paper_id: 待删除论文主键
            user_id: 论文 owner
            task_id: 当前删除任务主键
        Returns:
            CAS 删除成功时返回 True
        """
        pool = await get_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                DELETE FROM papers
                WHERE paper_id=$1 AND user_id=$2 AND deletion_task_id=$3
                  AND deletion_status='deleting'
                RETURNING paper_id
                """,
                paper_id,
                user_id,
                task_id,
            )
            if row is not None:
                await connection.execute("UPDATE upload_documents SET state='deleted',source_ref='{}'::jsonb,updated_at=NOW() WHERE paper_id=$1 AND user_id=$2", paper_id, user_id)
                await connection.execute("DELETE FROM upload_checkpoints WHERE paper_id=$1 AND user_id=$2", paper_id, user_id)
                await connection.execute("UPDATE upload_batches SET state='retired',manifest_ref=NULL,retired_at=NOW() WHERE paper_id=$1 AND user_id=$2", paper_id, user_id)
        return row is not None

    async def stop_uploads(self, paper_id: str, *, user_id: str) -> bool:
        """删除前取消同论文上传及增强任务，等待有效执行自然离开外部步骤。"""
        pool = await get_pool()
        async with pool.acquire() as connection:
            await connection.execute("""UPDATE task_runs SET cancel_requested=TRUE,
                status=CASE WHEN status IN ('queued','paused') THEN 'cancelled' ELSE status END,
                updated_at=NOW() WHERE user_id=$1 AND payload->>'paper_id'=$2 AND task_type='upload_ingest'
                AND status IN ('queued','paused','running')""", user_id, paper_id)
            running = await connection.fetchval("""SELECT EXISTS(SELECT 1 FROM task_runs WHERE user_id=$1
                AND payload->>'paper_id'=$2 AND task_type='upload_ingest' AND status='running' AND lease_until>NOW())""", user_id, paper_id)
        return not running


def get_paper_deletion_repository() -> PaperDeletionRepository:
    """返回论文删除仓储单例。"""
    global _repository
    if _repository is None:
        _repository = PaperDeletionRepository()
    return _repository
