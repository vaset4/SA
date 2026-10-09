"""后台任务仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.repositories.common import dump_json, load_json, normalize_record
from app.services.database import fetch, fetchrow, fetchval, get_pool

_repository: "TaskRepository | None" = None

# 这里实现的是“数据库驱动的任务队列最小内核”：
#   - task_runs 记录任务主状态
#   - task_events 记录时间线事件
#   - worker 通过 claim/heartbeat/cancel 在数据库里协调


def _task_run_data(row: Any) -> dict[str, Any] | None:
    """把 task_runs 行记录转换成上层 service 可直接返回的字典。"""
    data = normalize_record(row)
    if data is None:
        return None
    data["progress"] = int(data.get("progress") or 0)
    data["payload"] = load_json(data.get("payload"), {})
    data["result"] = load_json(data.get("result"), None)
    data["cancel_requested"] = bool(data.get("cancel_requested"))
    return data


def _task_event_data(row: Any) -> dict[str, Any] | None:
    """把 task_events 行记录转换成字典，并还原 payload JSON。"""
    data = normalize_record(row)
    if data is None:
        return None
    data["payload"] = load_json(data.get("payload"), {})
    return data


class TaskRepository:
    """封装 task_runs 与 task_events 访问。"""

    async def create_task_run(
        self,
        task_type: str,
        payload: dict[str, Any],
        message: str,
        *,
        task_id: str | None = None,
        user_id: str,
        connection: Any | None = None,
    ) -> dict[str, Any]:
        """创建排队中的任务主记录；user_id 为服务端归属，必填。"""
        if not user_id or not user_id.strip():
            raise ValueError("任务 owner 必须是非空 user_id")
        resolved_task_id = task_id or str(uuid.uuid4())
        query = """
            INSERT INTO task_runs (task_id, user_id, task_type, status, progress, message, payload, updated_at)
            VALUES ($1, $2, $3, 'queued', 0, $4, $5::jsonb, NOW())
            RETURNING *
            """
        args = (resolved_task_id, user_id, task_type, message, dump_json(payload))
        row = await (connection.fetchrow(query, *args) if connection is not None else fetchrow(query, *args))
        return _task_run_data(row) or {}

    async def add_task_event(
        self,
        task_id: str,
        event_type: str,
        message: str,
        payload: dict[str, Any],
        *,
        connection: Any | None = None,
    ) -> dict[str, Any]:
        """追加任务时间线事件，供前端轮询或调试页展示。"""
        query = """
            INSERT INTO task_events (event_id, task_id, event_type, message, payload)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING *
            """
        args = (str(uuid.uuid4()), task_id, event_type, message, dump_json(payload))
        row = await (connection.fetchrow(query, *args) if connection is not None else fetchrow(query, *args))
        return _task_event_data(row) or {}

    async def get_task_run(self, task_id: str, *, user_id: str, connection: Any | None = None) -> dict[str, Any] | None:
        """按 (task_id, user_id) 读取单条任务；跨用户或不存在返回 None。"""
        query = "SELECT * FROM task_runs WHERE task_id=$1 AND user_id=$2"
        row = await (connection.fetchrow(query, task_id, user_id) if connection is not None else fetchrow(query, task_id, user_id))
        return _task_run_data(row)

    async def list_task_runs(
        self,
        statuses: list[str] | None,
        *,
        user_id: str,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        """按当前用户分页列出任务。"""
        rows = await fetch(
            """
            SELECT * FROM task_runs
            WHERE user_id=$1 AND ($2::text[] IS NULL OR status = ANY($2::text[]))
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
            """,
            user_id,
            statuses or None,
            limit,
            offset,
        )
        return [_task_run_data(row) or {} for row in rows]

    async def list_task_events(
        self,
        task_id: str,
        *,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """按任务归属列出事件；无权限时返回空，避免事件侧泄露他人任务。"""
        rows = await fetch(
            """
            SELECT e.* FROM task_events e
            JOIN task_runs t ON t.task_id = e.task_id
            WHERE e.task_id=$1 AND t.user_id=$2
            ORDER BY e.created_at ASC
            LIMIT $3
            """,
            task_id,
            user_id,
            limit,
        )
        return [_task_event_data(row) or {} for row in rows]

    async def update_task_progress(self, task_id: str, progress: int, message: str) -> dict[str, Any] | None:
        """更新运行中任务的进度与状态文案。"""
        row = await fetchrow(
            """
            UPDATE task_runs
            SET progress=$2, message=$3, updated_at=NOW()
            WHERE task_id=$1
            RETURNING *
            """,
            task_id,
            progress,
            message,
        )
        return _task_run_data(row)

    async def complete_task_run(
        self,
        task_id: str,
        result: Any,
        message: str,
        *,
        worker_id: str,
    ) -> dict[str, Any] | None:
        """由仍持有效租约的 worker 把任务收敛为 completed；失租时返回 None。"""
        row = await fetchrow(
            """
            UPDATE task_runs
            SET status='completed', progress=100, result=$2::jsonb, error=NULL, lease_until=NULL,
                updated_at=NOW(), finished_at=NOW(), message=$3
            WHERE task_id=$1
              AND worker_id=$4
              AND status='running'
              AND lease_until IS NOT NULL
              AND lease_until > NOW()
            RETURNING *
            """,
            task_id,
            dump_json(result),
            message,
            worker_id,
        )
        return _task_run_data(row)

    async def fail_task_run(
        self,
        task_id: str,
        error: str,
        message: str,
        *,
        worker_id: str,
    ) -> dict[str, Any] | None:
        """由仍持有效租约的 worker 把任务收敛为 failed；失租时返回 None。"""
        row = await fetchrow(
            """
            UPDATE task_runs
            SET status='failed', error=$2, lease_until=NULL, updated_at=NOW(), finished_at=NOW(), message=$3
            WHERE task_id=$1
              AND worker_id=$4
              AND status='running'
              AND lease_until IS NOT NULL
              AND lease_until > NOW()
            RETURNING *
            """,
            task_id,
            error,
            message,
            worker_id,
        )
        return _task_run_data(row)

    async def mark_task_cancelled(
        self,
        task_id: str,
        message: str,
        *,
        worker_id: str,
    ) -> dict[str, Any] | None:
        """由仍持有效租约的 worker 把运行中任务收敛为 cancelled；失租时返回 None。"""
        row = await fetchrow(
            """
            UPDATE task_runs
            SET status='cancelled', cancel_requested=TRUE, lease_until=NULL,
                updated_at=NOW(), finished_at=COALESCE(finished_at, NOW()), message=$2
            WHERE task_id=$1
              AND worker_id=$3
              AND status='running'
              AND lease_until IS NOT NULL
              AND lease_until > NOW()
            RETURNING *
            """,
            task_id,
            message,
            worker_id,
        )
        return _task_run_data(row)

    async def request_task_cancel(
        self,
        task_id: str,
        message: str,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        """
        请求取消任务。

        queued 任务可以直接取消；running 任务则只打 `cancel_requested` 标记，等待 worker
        在安全点自行停止。按 (task_id, user_id) 归属校验，跨用户返回 None。
        """
        row = await fetchrow("SELECT * FROM task_runs WHERE task_id=$1 AND user_id=$2", task_id, user_id)
        if row is None:
            return None
        if row["task_type"] == "paper_delete":
            raise ValueError("论文删除任务不可取消")
        if row["status"] in {"completed", "failed", "cancelled"}:
            return _task_run_data(row)
        if row["status"] in {"queued", "paused"}:
            row = await fetchrow(
                """
                UPDATE task_runs
                SET status='cancelled', cancel_requested=TRUE, lease_until=NULL,
                    updated_at=NOW(), finished_at=COALESCE(finished_at, NOW()), message=$3
                WHERE task_id=$1 AND user_id=$2 AND status IN ('queued','paused')
                RETURNING *
                """,
                task_id,
                user_id,
                message,
            )
            if row is not None:
                return _task_run_data(row)
        row = await fetchrow(
            """
            UPDATE task_runs
            SET cancel_requested=TRUE, updated_at=NOW(), message=$3
            WHERE task_id=$1 AND user_id=$2 AND status='running'
            RETURNING *
            """,
            task_id,
            user_id,
            message,
        )
        return _task_run_data(row) if row is not None else await self.get_task_run(task_id, user_id=user_id)

    async def claim_next_task(self, worker_id: str, lease_seconds: int, task_types: list[str] | None) -> dict[str, Any] | None:
        """
        原子地抢占下一条可执行任务。

        这里依赖数据库事务 + `FOR UPDATE SKIP LOCKED`，避免多个 worker 抢到同一条任务。
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    WITH candidate AS (
                        SELECT task_id
                        FROM task_runs
                        WHERE (
                            status='queued'
                            OR (status='running' AND lease_until IS NOT NULL AND lease_until <= NOW())
                        )
                        AND ($2::text[] IS NULL OR task_type = ANY($2::text[]))
                        ORDER BY created_at
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE task_runs
                    SET status='running', worker_id=$1,
                        lease_until=NOW() + make_interval(secs => $3::int),
                        started_at=COALESCE(started_at, NOW()),
                        updated_at=NOW(), message='任务运行中'
                    WHERE task_id = (SELECT task_id FROM candidate)
                    RETURNING *
                    """,
                    worker_id,
                    task_types or None,
                    int(lease_seconds),
                )
        return _task_run_data(row)

    async def heartbeat_task(self, task_id: str, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        """续租运行中任务，防止 worker 被误判为失活。"""
        row = await fetchrow(
            """
            UPDATE task_runs
            SET lease_until=NOW() + make_interval(secs => $3::int), updated_at=NOW()
            WHERE task_id=$1 AND worker_id=$2 AND status='running'
              AND lease_until IS NOT NULL AND lease_until > NOW()
            RETURNING *
            """,
            task_id,
            worker_id,
            int(lease_seconds),
        )
        return _task_run_data(row)

    async def is_task_cancel_requested(self, task_id: str) -> bool:
        """查询任务是否已收到取消请求，供 worker 轮询检查。"""
        return bool(await fetchval("SELECT cancel_requested FROM task_runs WHERE task_id=$1", task_id))


def get_task_repository() -> TaskRepository:
    """返回任务仓储单例。"""
    global _repository
    if _repository is None:
        _repository = TaskRepository()
    return _repository
