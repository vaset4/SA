"""Repository for durable user-visible RAG run lifecycle state."""

from __future__ import annotations

from typing import Any

from app.repositories.common import dump_json, load_json, normalize_record, utc_now
from app.services import database
from app.services.database import execute, fetch, fetchrow


def _run_data(row: Any) -> dict[str, Any] | None:
    data = normalize_record(row)
    if data is None:
        return None
    data["metadata"] = load_json(data.get("metadata"), {})
    data["pending_artifact"] = load_json(data.get("pending_artifact"), None)
    return data


def _event_data(row: Any) -> dict[str, Any] | None:
    """把事件记录归一化为可序列化字典，并还原 JSON payload。"""
    data = normalize_record(row)
    if data is None:
        return None
    data["payload"] = load_json(data.get("payload"), {})
    data["sequence"] = int(data.get("sequence") or 0)
    return data


class RAGRunRepository:
    """Owns the `rag_runs` table and SQL queries."""

    async def ensure_schema(self) -> None:
        """幂等创建 run、事件表以及跨进程幂等/并发约束。"""
        await execute(
            """
            CREATE TABLE IF NOT EXISTS rag_runs (
                run_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                conversation_id TEXT,
                session_id TEXT,
                request_id TEXT,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                answer TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ
            );
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS idempotency_key TEXT;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS thread_id TEXT;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT FALSE;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS deadline_at TIMESTAMPTZ;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS next_event_sequence BIGINT NOT NULL DEFAULT 0;

            CREATE TABLE IF NOT EXISTS rag_run_events (
                run_id TEXT NOT NULL REFERENCES rag_runs(run_id) ON DELETE CASCADE,
                sequence BIGINT NOT NULL,
                event_type TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (run_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS idx_rag_runs_user_conversation_updated
                ON rag_runs(user_id, conversation_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_rag_runs_user_status_updated
                ON rag_runs(user_id, status, updated_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_runs_user_idempotency
                ON rag_runs(user_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_runs_user_active_thread
                ON rag_runs(user_id, thread_id)
                WHERE thread_id IS NOT NULL AND status IN ('queued', 'running');
            CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_run_terminal_event
                ON rag_run_events(run_id)
                WHERE event_type='terminal';
            CREATE INDEX IF NOT EXISTS idx_rag_run_events_run_sequence
                ON rag_run_events(run_id, sequence);
            """
        )

        await self._ensure_operation_schema()

    async def _ensure_operation_schema(self) -> None:
        """无参数；补齐命令、执行归属和结果版本字段，返回前建好排他约束。"""
        await execute(
            """
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS operation_kind TEXT;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS target_run_id TEXT;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS request_fingerprint TEXT;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS owner_instance_id TEXT;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS execution_started_at TIMESTAMPTZ;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS execution_stopped_at TIMESTAMPTZ;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS delivery_revision BIGINT NOT NULL DEFAULT 0;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS expected_delivery_revision BIGINT;
            ALTER TABLE rag_runs ADD COLUMN IF NOT EXISTS pending_artifact JSONB;
            ALTER TABLE sessions ADD COLUMN IF NOT EXISTS current_run_id TEXT;
            ALTER TABLE sessions ADD COLUMN IF NOT EXISTS result_run_id TEXT;
            ALTER TABLE sessions ADD COLUMN IF NOT EXISTS current_result JSONB;
            ALTER TABLE sessions ADD COLUMN IF NOT EXISTS delivery_revision BIGINT NOT NULL DEFAULT 0;
            CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_runs_owned_thread
                ON rag_runs(user_id, thread_id)
                WHERE thread_id IS NOT NULL AND
                    (status IN ('queued', 'running') OR
                     (owner_instance_id IS NOT NULL AND execution_stopped_at IS NULL));
            CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_runs_active_target
                ON rag_runs(user_id, target_run_id)
                WHERE target_run_id IS NOT NULL AND status IN ('queued', 'running');
            """
        )

    async def create_operation(
        self, *, operation: dict[str, Any], session_command: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """参数为操作与可选 session 前提；同库领取并绑定执行权，返回行及是否新领取。"""
        pool = await database.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            existing = await conn.fetchrow(
                "SELECT * FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2",
                operation["user_id"], operation["idempotency_key"],
            )
            if existing is not None:
                return _run_data(existing), False
            if session_command is not None:
                existing = await self._lock_session_command(conn, operation, session_command)
                if existing is not None:
                    return existing, False
            if operation.get("target_run_id"):
                existing = await self._lock_delivery_target(conn, operation)
                if existing is not None:
                    return existing, False
            created = await self._insert_operation(conn, operation)
            if created is None:
                existing = await conn.fetchrow(
                    "SELECT * FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2",
                    operation["user_id"], operation["idempotency_key"],
                )
                return _run_data(existing), False
            if session_command is not None:
                # 1. 【研究执行】【原子领取】只有新建操作可以推进 session 并增长真正研究次数
                await conn.execute(
                    """UPDATE sessions SET current_run_id=$1, status='running', last_error=NULL,
                        attempt_count=attempt_count + $2, updated_at=$3
                        WHERE session_id=$4 AND user_id=$5""",
                    operation["run_id"], int(session_command["increment_attempt"]), utc_now(),
                    operation["session_id"], operation["user_id"],
                )
            return created, True

    async def _lock_session_command(self, conn: Any, operation: dict, command: dict) -> dict | None:
        """参数为同库连接、操作和命令前提；锁定 owner/scope/status，失败不产生运行。"""
        from app.repositories.session_repository import _session_data

        session = _session_data(await conn.fetchrow(
            "SELECT * FROM sessions WHERE session_id=$1 AND user_id=$2 FOR UPDATE",
            operation["session_id"], operation["user_id"],
        ))
        if session is None:
            raise LookupError("研究会话不存在")
        existing = await conn.fetchrow(
            "SELECT * FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2",
            operation["user_id"], operation["idempotency_key"],
        )
        if existing is not None:
            return _run_data(existing)
        if session["status"] != command["expected_status"]:
            raise ValueError("研究会话状态已变化，请读取当前操作")
        if session.get("thread_id") != operation["thread_id"]:
            raise ValueError("研究会话 checkpoint 身份已变化")
        for field in ("project_id", "research_mode", "conversation_id"):
            if session.get(field) != command.get(field):
                raise ValueError(f"研究会话的 {field} 已变化")
        if set(session["paper_ids"]) != set(command["paper_ids"]):
            raise ValueError("研究会话论文范围已变化")
        if session.get("current_run_id"):
            previous = await conn.fetchrow(
                "SELECT * FROM rag_runs WHERE run_id=$1 AND user_id=$2",
                session["current_run_id"], operation["user_id"],
            )
            if previous is None or previous["execution_stopped_at"] is None:
                raise ValueError("无法确认上次研究执行已经退出，当前线程不可重入")
            if previous["status"] in {"queued", "running"}:
                raise ValueError("上次研究尚未完成终态保存，当前线程不可重入")
            if previous["pending_artifact"] is not None:
                raise ValueError("上次生成产物尚未保存，请先重试保存")
        return None

    async def _lock_delivery_target(self, conn: Any, operation: dict) -> dict | None:
        """参数为连接与最终操作；核对目标归属及版本，返回前释放只随事务结束。"""
        target = await conn.fetchrow(
            "SELECT * FROM rag_runs WHERE run_id=$1 AND user_id=$2 FOR UPDATE",
            operation["target_run_id"], operation["user_id"],
        )
        if target is None:
            raise LookupError("目标研究结果不存在")
        existing = await conn.fetchrow(
            "SELECT * FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2",
            operation["user_id"], operation["idempotency_key"],
        )
        if existing is not None:
            return _run_data(existing)
        if target["status"] in {"queued", "running"}:
            raise ValueError("目标研究尚未结束")
        if target["execution_started_at"] is not None and target["execution_stopped_at"] is None:
            raise ValueError("目标研究执行尚未确认退出，不能重试最终结果")
        if target["delivery_revision"] != operation["expected_delivery_revision"]:
            raise ValueError("目标研究结果版本已变化")
        if operation["operation_kind"] == "final_only" and target["pending_artifact"] is not None:
            raise ValueError("当前结果存在待保存产物，请先重试保存")
        return None

    async def _insert_operation(self, conn: Any, operation: dict) -> dict[str, Any] | None:
        """参数为领取事务连接和操作；应用幂等与活动唯一约束，返回新建行或冲突。"""
        return _run_data(await conn.fetchrow(
            """INSERT INTO rag_runs (run_id, user_id, conversation_id, session_id, thread_id,
                    idempotency_key, query, status, metadata, deadline_at, created_at, updated_at,
                    operation_kind, target_run_id, request_fingerprint, expected_delivery_revision)
                VALUES ($1,$2,$3,$4,$5,$6,$7,'queued',$8::jsonb,$9,$10,$10,$11,$12,$13,$14)
                ON CONFLICT DO NOTHING RETURNING *""",
            operation["run_id"], operation["user_id"], operation["conversation_id"],
            operation["session_id"], operation["thread_id"], operation["idempotency_key"],
            operation["query"], dump_json(operation.get("metadata") or {}), operation["deadline_at"],
            utc_now(), operation["operation_kind"], operation["target_run_id"],
            operation["request_fingerprint"], operation["expected_delivery_revision"],
        ))

    async def claim_execution(self, *, user_id: str, run_id: str, owner_instance_id: str) -> dict | None:
        """参数为已领取操作和进程身份；只有未被执行器持有的 queued 行能取得图执行权。"""
        return _run_data(await fetchrow(
            """UPDATE rag_runs SET owner_instance_id=$1, execution_started_at=$2,
                status='running', updated_at=$2
                WHERE user_id=$3 AND run_id=$4 AND status='queued' AND owner_instance_id IS NULL
                RETURNING *""", owner_instance_id, utc_now(), user_id, run_id,
        ))

    async def record_execution_stopped(self, *, user_id: str, run_id: str, owner_instance_id: str) -> dict | None:
        """参数为实际退出工作的操作和进程；记录停止事实，不凭期限解除别人的执行权。"""
        return _run_data(await fetchrow(
            """UPDATE rag_runs SET execution_stopped_at=$1, updated_at=$1
                WHERE user_id=$2 AND run_id=$3 AND owner_instance_id=$4 RETURNING *""",
            utc_now(), user_id, run_id, owner_instance_id,
        ))

    async def settle_session_operation(self, *, user_id: str, run_id: str,
                                       status: str, error: str) -> dict | None:
        """参数为真实退出操作和失败事实；同事务更新当前 session 及唯一 terminal 事件。"""
        pool = await database.get_pool()
        async with pool.acquire() as conn, conn.transaction():
            run = await conn.fetchrow(
                "SELECT * FROM rag_runs WHERE user_id=$1 AND run_id=$2 FOR UPDATE", user_id, run_id,
            )
            if run is None:
                raise LookupError("研究操作不存在")
            if run["status"] not in {"queued", "running"}:
                return _run_data(run)
            if run["execution_started_at"] is not None and run["execution_stopped_at"] is None:
                raise ValueError("研究工作尚未确认退出，不能提交终态")
            if run["execution_started_at"] is None:
                await conn.execute(
                    "UPDATE rag_runs SET execution_stopped_at=$1 WHERE user_id=$2 AND run_id=$3",
                    utc_now(), user_id, run_id,
                )
            updated = await self.finish_run(
                user_id=user_id, run_id=run_id, expected_status=run["status"], status=status,
                error=error, payload={"type": "terminal", "run_id": run_id, "run_status": status,
                                      "detail": error}, connection=conn,
            )
            if updated is not None and run["session_id"] is not None:
                await conn.execute(
                    """UPDATE sessions SET status=$1, last_error=$2, updated_at=$3
                        WHERE user_id=$4 AND session_id=$5 AND current_run_id=$6""",
                    status, error, utc_now(), user_id, run["session_id"], run_id,
                )
            return updated

    async def create_run(
        self,
        *,
        run_id: str,
        user_id: str,
        conversation_id: str | None,
        session_id: str | None,
        query: str,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        thread_id: str | None = None,
        deadline_at: Any = None,
    ) -> dict[str, Any]:
        """创建一条 run；同步兼容入口可不提供幂等和 thread 字段。"""
        now = utc_now()
        row = await fetchrow(
            """
            INSERT INTO rag_runs (
                run_id, user_id, conversation_id, session_id, query,
                status, metadata, idempotency_key, thread_id, deadline_at,
                created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, 'queued', $6::jsonb, $7, $8, $9, $10, $11)
            RETURNING *
            """,
            run_id,
            user_id,
            conversation_id,
            session_id,
            query,
            dump_json(metadata or {}),
            idempotency_key,
            thread_id,
            deadline_at,
            now,
            now,
        )
        data = _run_data(row)
        if data is None:
            raise RuntimeError("rag run create returned no row")
        return data

    async def create_or_get_run(
        self,
        *,
        run_id: str,
        user_id: str,
        conversation_id: str | None,
        session_id: str | None,
        thread_id: str,
        idempotency_key: str,
        query: str,
        deadline_at: Any,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """原子创建流式 run；同 key 重复请求返回原 run，不重复执行。"""
        now = utc_now()
        row = await fetchrow(
            """
            WITH inserted AS (
                INSERT INTO rag_runs (
                    run_id, user_id, conversation_id, session_id, thread_id,
                    idempotency_key, query, status, metadata, deadline_at,
                    created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'queued', $8::jsonb, $9, $10, $11)
                ON CONFLICT DO NOTHING
                RETURNING *, TRUE AS created
            )
            SELECT * FROM inserted
            UNION ALL
            SELECT rag_runs.*, FALSE AS created
            FROM rag_runs
            WHERE user_id=$2 AND idempotency_key=$6
            LIMIT 1
            """,
            run_id,
            user_id,
            conversation_id,
            session_id,
            thread_id,
            idempotency_key,
            query,
            dump_json(metadata or {}),
            deadline_at,
            now,
            now,
        )
        data = _run_data(row)
        if data is None:
            return None, False
        created = bool(data.pop("created", False))
        return data, created

    async def update_run(
        self,
        *,
        run_id: str,
        user_id: str,
        status: str,
        request_id: str | None = None,
        session_id: str | None = None,
        answer: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        completed: bool = False,
        expected_status: str | None = None,
    ) -> dict[str, Any]:
        """按 owner 更新非终态 run；可选 expected_status 提供 CAS 保护。"""
        now = utc_now()
        row = await fetchrow(
            """
            UPDATE rag_runs
            SET status=$1,
                request_id=COALESCE($2, request_id),
                session_id=COALESCE($3, session_id),
                answer=COALESCE($4, answer),
                error=$5,
                metadata=metadata || $6::jsonb,
                updated_at=$7,
                completed_at=CASE WHEN $8 THEN $7 ELSE completed_at END
            WHERE user_id=$9 AND run_id=$10
              AND ($11::text IS NULL OR status=$11)
            RETURNING *
            """,
            status,
            request_id,
            session_id,
            answer,
            error,
            dump_json(metadata or {}),
            now,
            completed,
            user_id,
            run_id,
            expected_status,
        )
        data = _run_data(row)
        if data is None:
            raise LookupError("RAG run not found")
        return data

    async def append_event(
        self,
        *,
        user_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """在单条 SQL 内分配递增序号并写入一个非终态业务事件。"""
        row = await fetchrow(
            """
            WITH owner AS (
                UPDATE rag_runs
                SET next_event_sequence=next_event_sequence + 1,
                    updated_at=$1
                WHERE user_id=$2 AND run_id=$3
                  AND status IN ('queued', 'running')
                RETURNING run_id, next_event_sequence
            )
            INSERT INTO rag_run_events (run_id, sequence, event_type, payload, created_at)
            SELECT run_id, next_event_sequence, $4, $5::jsonb, $1
            FROM owner
            RETURNING *
            """,
            utc_now(),
            user_id,
            run_id,
            event_type,
            dump_json(payload),
        )
        data = _event_data(row)
        if data is None:
            raise LookupError("RAG run 不存在或已结束")
        return data

    async def finish_run(
        self,
        *,
        user_id: str,
        run_id: str,
        expected_status: str,
        status: str,
        payload: dict[str, Any],
        request_id: str | None = None,
        session_id: str | None = None,
        answer: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        connection: Any = None,
    ) -> dict[str, Any] | None:
        """用一次 CAS 同时写终态和唯一 terminal 事件。"""
        now = utc_now()
        row = await (connection.fetchrow if connection is not None else fetchrow)(
            """
            WITH transitioned AS (
                UPDATE rag_runs
                SET status=$1,
                    request_id=COALESCE($2, request_id),
                    session_id=COALESCE($3, session_id),
                    answer=COALESCE($4, answer),
                    error=$5,
                    metadata=metadata || $6::jsonb,
                    cancel_requested=cancel_requested OR $1='cancelled',
                    next_event_sequence=next_event_sequence + 1,
                    updated_at=$7,
                    completed_at=$7
                WHERE user_id=$8 AND run_id=$9 AND status=$10
                  AND ($1<>'completed' OR NOT cancel_requested)
                RETURNING *
            ), terminal_event AS (
                INSERT INTO rag_run_events (run_id, sequence, event_type, payload, created_at)
                SELECT run_id, next_event_sequence, 'terminal', $11::jsonb, $7
                FROM transitioned
                RETURNING run_id
            )
            SELECT transitioned.*
            FROM transitioned
            JOIN terminal_event USING (run_id)
            """,
            status,
            request_id,
            session_id,
            answer,
            error,
            dump_json(metadata or {}),
            now,
            user_id,
            run_id,
            expected_status,
            dump_json(payload),
        )
        return _run_data(row)

    async def request_cancel(self, *, user_id: str, run_id: str) -> dict[str, Any] | None:
        """为 active run 写入显式取消请求，供执行器安全点读取。"""
        row = await fetchrow(
            """
            UPDATE rag_runs
            SET cancel_requested=TRUE, updated_at=$1
            WHERE user_id=$2 AND run_id=$3 AND status IN ('queued', 'running')
            RETURNING *
            """,
            utc_now(),
            user_id,
            run_id,
        )
        return _run_data(row)

    async def get_run(self, *, user_id: str, run_id: str) -> dict[str, Any] | None:
        row = await fetchrow(
            "SELECT * FROM rag_runs WHERE user_id=$1 AND run_id=$2",
            user_id,
            run_id,
        )
        return _run_data(row)

    async def get_by_idempotency_key(
        self,
        *,
        user_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """按 owner 和幂等键读取已有 run。"""
        row = await fetchrow(
            "SELECT * FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2",
            user_id,
            idempotency_key,
        )
        return _run_data(row)

    async def get_active_for_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        """按 owner/thread 读取唯一 queued 或 running run。"""
        row = await fetchrow(
            """
            SELECT * FROM rag_runs
            WHERE user_id=$1 AND thread_id=$2 AND
                (status IN ('queued', 'running') OR
                 (owner_instance_id IS NOT NULL AND execution_stopped_at IS NULL))
            ORDER BY created_at DESC
            LIMIT 1
            """,
            user_id,
            thread_id,
        )
        return _run_data(row)

    async def get_active_for_target(self, *, user_id: str, target_run_id: str) -> dict | None:
        """参数为 owner 和结果目标；返回仍在修改该目标的唯一操作。"""
        return _run_data(await fetchrow(
            """SELECT * FROM rag_runs WHERE user_id=$1 AND target_run_id=$2
                AND status IN ('queued','running') LIMIT 1""", user_id, target_run_id,
        ))

    async def list_events(
        self,
        *,
        user_id: str,
        run_id: str,
        after_sequence: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """按递增序号返回 owner 可见的 run 事件。"""
        rows = await fetch(
            """
            SELECT event.*
            FROM rag_run_events AS event
            JOIN rag_runs AS run ON run.run_id=event.run_id
            WHERE run.user_id=$1 AND event.run_id=$2 AND event.sequence>$3
            ORDER BY event.sequence ASC
            LIMIT $4
            """,
            user_id,
            run_id,
            after_sequence,
            limit,
        )
        return [data for row in rows if (data := _event_data(row)) is not None]

    async def list_expired_active_runs(self) -> list[dict[str, Any]]:
        """列出超过自身 deadline 的 active run，供启动和定时收敛。"""
        rows = await fetch(
            """
            SELECT * FROM rag_runs
            WHERE status IN ('queued', 'running')
              AND deadline_at IS NOT NULL
              AND deadline_at <= NOW()
            ORDER BY deadline_at ASC
            """
        )
        return [data for row in rows if (data := _run_data(row)) is not None]

    async def get_latest_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        row = await fetchrow(
            """
            SELECT *
            FROM rag_runs
            WHERE user_id=$1 AND conversation_id=$2
                AND COALESCE(operation_kind, 'chat') NOT IN ('final_only', 'persist_only')
            ORDER BY CASE WHEN status IN ('queued', 'running') THEN 0 ELSE 1 END,
                created_at DESC, run_id DESC
            LIMIT 1
            """,
            user_id,
            conversation_id,
        )
        return _run_data(row)


_repository: RAGRunRepository | None = None


def get_rag_run_repository() -> RAGRunRepository:
    global _repository
    if _repository is None:
        _repository = RAGRunRepository()
    return _repository
