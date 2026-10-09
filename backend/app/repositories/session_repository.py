"""历史兼容 session 仓储。"""

from __future__ import annotations

from typing import Any

from app.repositories.common import dump_json, load_json, normalize_record, utc_now
from app.services.database import execute, fetch, fetchrow

_repository: "SessionRepository | None" = None

# 这里维护的是“历史 session 表”，它和前端 conversations 表不是一回事：
#   - sessions：偏 LangGraph checkpoint / 旧会话兼容
#   - conversations：偏前端消息展示
# 因此很多 API 会同时接触两套 session/conversation 概念。


def _session_data(row: Any) -> dict[str, Any] | None:
    """
    把 session 数据库记录归一化为领域服务可消费的字典。

    Args:
        row: asyncpg 返回的单行记录

    Returns:
        dict[str, Any] | None: 已还原 JSON 字段的 session，空记录返回 None
    """
    data = normalize_record(row)
    if data is None:
        return None
    data["paper_ids"] = load_json(data.get("paper_ids"), [])
    data["chat_history"] = load_json(data.get("chat_history"), [])
    data["attempt_count"] = int(data.get("attempt_count") or 0)
    data["current_result"] = load_json(data.get("current_result"), None)
    return data


class SessionRepository:
    """封装历史兼容 session 表访问。"""

    async def create_session(
        self,
        user_id: str,
        session_id: str,
        topic: str,
        *,
        conversation_id: str | None = None,
        project_id: str | None = None,
        thread_id: str | None = None,
        research_mode: str | None = None,
        paper_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        在 graph 执行前创建 durable session。

        Args:
            user_id: session 所属用户 ID
            session_id: 对外暴露的研究会话主键
            topic: 会话展示标题
            conversation_id: 可选前端对话 ID
            project_id: 可选项目 ID
            thread_id: LangGraph checkpoint key
            research_mode: internal 或 external
            paper_ids: 已完成 owner 校验的论文范围

        Returns:
            dict[str, Any] | None: 新建行；主键已存在时返回 None
        """
        row = await fetchrow(
            """
            INSERT INTO sessions (
                session_id, user_id, topic, conversation_id, project_id,
                thread_id, research_mode, status, last_error, attempt_count,
                paper_ids, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'created', NULL, 0, $8::jsonb, $9, $10)
            ON CONFLICT (session_id) DO NOTHING
            RETURNING *
            """,
            session_id,
            user_id,
            topic,
            conversation_id,
            project_id,
            thread_id,
            research_mode,
            dump_json(paper_ids or []),
            utc_now(),
            utc_now(),
        )
        return _session_data(row)

    async def bind_session_scope(
        self,
        session_id: str,
        *,
        user_id: str,
        conversation_id: str | None,
        project_id: str | None,
        research_mode: str | None,
        paper_ids: list[str],
    ) -> dict[str, Any] | None:
        """
        只补齐旧 session 缺失的身份与范围，已有范围不得被空请求覆盖。

        Args:
            session_id: 目标研究会话 ID
            user_id: session 所属用户 ID
            conversation_id: 待补齐的对话 ID
            project_id: 待补齐的项目 ID
            research_mode: 待补齐的研究模式
            paper_ids: 待补齐的论文范围

        Returns:
            dict[str, Any] | None: 更新后的 owner-scoped session
        """
        row = await fetchrow(
            """
            UPDATE sessions
            SET conversation_id=COALESCE(conversation_id, $1),
                project_id=COALESCE(project_id, $2),
                research_mode=COALESCE(research_mode, $3),
                paper_ids=CASE
                    WHEN COALESCE(jsonb_array_length(paper_ids), 0)=0
                         AND jsonb_array_length($4::jsonb)>0
                    THEN $4::jsonb
                    ELSE paper_ids
                END,
                updated_at=$5
            WHERE user_id=$6 AND session_id=$7
            RETURNING *
            """,
            conversation_id,
            project_id,
            research_mode,
            dump_json(paper_ids),
            utc_now(),
            user_id,
            session_id,
        )
        return _session_data(row)

    async def bind_thread_id(
        self,
        session_id: str,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        """
        对 legacy session 执行一次性 thread 绑定。

        Args:
            session_id: 目标研究会话 ID
            user_id: session 所属用户 ID
            thread_id: 已由 checkpoint/conversation 证据确认的 thread ID

        Returns:
            dict[str, Any] | None: 成功绑定后的 session；已被并发绑定时返回 None
        """
        row = await fetchrow(
            """
            UPDATE sessions
            SET thread_id=$1, updated_at=$2
            WHERE user_id=$3 AND session_id=$4 AND thread_id IS NULL
            RETURNING *
            """,
            thread_id,
            utc_now(),
            user_id,
            session_id,
        )
        return _session_data(row)

    async def initialize_legacy_status(
        self,
        session_id: str,
        *,
        user_id: str,
        status: str,
    ) -> dict[str, Any] | None:
        """
        首次读取 legacy session 时持久化一次推导状态。

        Args:
            session_id: 目标研究会话 ID
            user_id: session 所属用户 ID
            status: 从旧 state_json/step_name 明确推导出的状态

        Returns:
            dict[str, Any] | None: 初始化后的 session；并发已写入时返回 None
        """
        row = await fetchrow(
            """
            UPDATE sessions
            SET status=$1, updated_at=$2
            WHERE user_id=$3 AND session_id=$4 AND (status IS NULL OR status='')
            RETURNING *
            """,
            status,
            utc_now(),
            user_id,
            session_id,
        )
        return _session_data(row)

    async def transition_status(
        self,
        session_id: str,
        *,
        user_id: str,
        expected_status: str,
        target_status: str,
        last_error: str | None,
        increment_attempt: bool,
        expected_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        使用 owner + current status 做 CAS 状态迁移。

        Args:
            session_id: 目标研究会话 ID
            user_id: session 所属用户 ID
            expected_status: 调用方已验证的当前状态
            target_status: 本次目标状态
            last_error: 失败原因；非失败状态传 None
            increment_attempt: 是否增加执行尝试次数

        Returns:
            dict[str, Any] | None: 迁移后的 session；CAS 未命中时返回 None
        """
        row = await fetchrow(
            """
            UPDATE sessions
            SET status=$1,
                last_error=$2,
                attempt_count=attempt_count + CASE WHEN $3 THEN 1 ELSE 0 END,
                updated_at=$4
            WHERE user_id=$5 AND session_id=$6 AND status=$7
                AND current_run_id IS NOT DISTINCT FROM $8::text
            RETURNING *
            """,
            target_status,
            last_error,
            increment_attempt,
            utc_now(),
            user_id,
            session_id,
            expected_status,
            expected_run_id,
        )
        return _session_data(row)

    async def save_checkpoint(self, session_id: str, step_name: str, state_json: str, *, user_id: str | None = None) -> None:
        """
        把 checkpoint 写回 sessions 表。

        这里采用 upsert，是因为 session 可能先由 API 显式创建，也可能第一次保存状态时才出现。
        """
        await execute(
            """
            INSERT INTO sessions (session_id, user_id, step_name, state_json, updated_at)
            VALUES ($1, COALESCE($2, 'anonymous'), $3, $4, $5)
            ON CONFLICT (session_id) DO UPDATE
            SET step_name=EXCLUDED.step_name, state_json=EXCLUDED.state_json, updated_at=EXCLUDED.updated_at
            """,
            session_id,
            user_id,
            step_name,
            state_json,
            utc_now(),
        )

    async def load_checkpoint(self, session_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        """读取旧 session 表里的 checkpoint 快照。"""
        if user_id:
            row = await fetchrow(
                "SELECT step_name, state_json FROM sessions WHERE user_id=$1 AND session_id=$2",
                user_id,
                session_id,
            )
        else:
            row = await fetchrow("SELECT step_name, state_json FROM sessions WHERE session_id=$1", session_id)
        return normalize_record(row)

    async def update_session_papers(
        self,
        session_id: str,
        paper_ids: list[str],
        *,
        user_id: str | None = None,
    ) -> None:
        """更新 session 关联的 paper_ids，服务于旧链路的会话范围记忆。"""
        if user_id:
            await execute(
                "UPDATE sessions SET paper_ids=$1::jsonb, updated_at=$2 WHERE user_id=$3 AND session_id=$4",
                dump_json(paper_ids),
                utc_now(),
                user_id,
                session_id,
            )
            return
        await execute("UPDATE sessions SET paper_ids=$1::jsonb, updated_at=$2 WHERE session_id=$3", dump_json(paper_ids), utc_now(), session_id)

    async def get_session(self, session_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        """读取单个 session 完整记录。"""
        if user_id:
            row = await fetchrow("SELECT * FROM sessions WHERE user_id=$1 AND session_id=$2", user_id, session_id)
        else:
            row = await fetchrow("SELECT * FROM sessions WHERE session_id=$1", session_id)
        return _session_data(row)

    async def list_sessions(
        self,
        *,
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """按最近更新时间倒序列出 session；常用于会话列表页或恢复入口。"""
        if user_id:
            rows = await fetch(
                """
                SELECT session_id, user_id, topic, conversation_id, project_id, thread_id,
                       research_mode, status, last_error, attempt_count, paper_ids,
                       current_run_id, result_run_id, current_result, delivery_revision,
                       chat_history, state_json, step_name, created_at, updated_at
                FROM sessions
                WHERE user_id=$1
                ORDER BY updated_at DESC, created_at DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        else:
            rows = await fetch(
                """
                SELECT session_id, user_id, topic, conversation_id, project_id, thread_id,
                       research_mode, status, last_error, attempt_count, paper_ids,
                       current_run_id, result_run_id, current_result, delivery_revision,
                       chat_history, state_json, step_name, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC, created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
        return [_session_data(row) or {} for row in rows]

    async def remove_paper_from_all_sessions(self, paper_id: str, *, user_id: str | None = None) -> int:
        """从所有历史 session 的 `paper_ids` 中移除指定论文。"""
        if user_id:
            rows = await fetch(
                "SELECT session_id, paper_ids FROM sessions WHERE user_id=$1",
                user_id,
            )
        else:
            rows = await fetch("SELECT session_id, paper_ids FROM sessions")
        affected = 0
        for row in rows:
            session_id = str(row["session_id"])
            paper_ids = [item for item in load_json(row["paper_ids"], []) if item]
            if paper_id not in paper_ids:
                continue
            next_paper_ids = [item for item in paper_ids if item != paper_id]
            if user_id:
                await execute(
                    "UPDATE sessions SET paper_ids=$1::jsonb, updated_at=$2 WHERE user_id=$3 AND session_id=$4",
                    dump_json(next_paper_ids),
                    utc_now(),
                    user_id,
                    session_id,
                )
            else:
                await execute(
                    "UPDATE sessions SET paper_ids=$1::jsonb, updated_at=$2 WHERE session_id=$3",
                    dump_json(next_paper_ids),
                    utc_now(),
                    session_id,
                )
            affected += 1
        return affected


def get_session_repository() -> SessionRepository:
    """返回 session 仓储单例。"""
    global _repository
    if _repository is None:
        _repository = SessionRepository()
    return _repository
