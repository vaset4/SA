"""业务日志与 RAG trace 仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.repositories.common import dump_json
from app.services.database import execute

_repository: "ObservabilityRepository | None" = None

# 这里的定位是“原始观测数据落库层”，不做聚合分析、不做 trace 对象构建。
# 上层 observability/rag_trace service 会先把数据整理好，再调用这里持久化。


class ObservabilityRepository:
    """封装 logs / rag_traces 写入。"""

    async def write_log(
        self,
        event_type: str,
        node_name: str,
        query: str,
        result_count: int,
        latency_ms: int,
        session_id: str | None,
        user_id: str | None = None,
    ) -> None:
        """
        写一条业务日志事件；适合节点耗时、命中数等轻量观测。

        面向用户的 push_notification 通知写入时必须显式传 user_id；普通观测日志可不传。
        """
        if event_type == "push_notification":
            if not user_id or not user_id.strip():
                raise ValueError("push_notification 必须显式传入 user_id")
        await execute(
            """
            INSERT INTO logs (log_id, user_id, session_id, event_type, node_name, query, result_count, latency_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            str(uuid.uuid4()),
            user_id,
            session_id,
            event_type,
            node_name,
            query,
            result_count,
            latency_ms,
        )

    async def write_rag_trace(
        self,
        request_id: str,
        session_id: str | None,
        query: str,
        trace_data: dict[str, Any],
        user_id: str = "unknown",
    ) -> None:
        """写完整 RAG trace 快照；trace_data 由上层 service 负责序列化组织。"""
        await execute(
            """
            INSERT INTO rag_traces (request_id, session_id, query, trace_data, user_id)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (request_id) DO UPDATE SET
                trace_data = EXCLUDED.trace_data,
                user_id = EXCLUDED.user_id
            """,
            request_id,
            session_id,
            query,
            dump_json(trace_data),
            user_id,
        )

    async def list_rag_traces(
        self,
        user_id: str | None = None,
        limit: int = 50,
        request_id: str | None = None,
        session_id: str | None = None,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出 trace 摘要。user_id 为 None 时不按用户过滤（供 developer 角色使用）。"""
        from app.services.database import fetch
        params: list[Any] = []
        clauses: list[str] = []
        if user_id is not None:
            params.append(user_id)
            clauses.append(f"user_id = ${len(params)}")
        if request_id:
            params.append(request_id)
            clauses.append(f"request_id = ${len(params)}")
        if session_id:
            params.append(session_id)
            clauses.append(f"session_id = ${len(params)}")
        if q:
            params.append(q)
            idx = len(params)
            clauses.append(f"(request_id = ${idx} OR session_id = ${idx})")
        params.append(min(max(int(limit), 1), 200))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT request_id, session_id, query, user_id, created_at,
                   trace_data->'summary' AS summary
            FROM rag_traces
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ${len(params)}
        """
        rows = await fetch(sql, *params)
        return [dict(r) for r in rows]

    async def get_rag_trace(self, request_id: str) -> dict[str, Any] | None:
        """按 request_id 取完整 trace（不做权限过滤，由上层校验）。"""
        from app.services.database import fetchrow
        row = await fetchrow(
            "SELECT request_id, session_id, query, user_id, created_at, trace_data "
            "FROM rag_traces WHERE request_id = $1",
            request_id,
        )
        return dict(row) if row else None


def get_observability_repository() -> ObservabilityRepository:
    """返回可观测性仓储单例。"""
    global _repository
    if _repository is None:
        _repository = ObservabilityRepository()
    return _repository
