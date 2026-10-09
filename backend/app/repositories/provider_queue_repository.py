"""【模型服务】【共享额度】PostgreSQL 原子准入仓储。作者：xxx。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.types.json import Jsonb

LEASE_SECONDS = 30.0
DATABASE_TIMEOUT_SECONDS = 5


class ProviderQueueRepository:
    """只保存调度身份及时间，不保存提示词、正文或供应商凭证。"""

    def __init__(self, database_url: str) -> None:
        """参数为明确指定的协调库地址；不连接或回落到业务数据库。"""
        self._database_url = database_url

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """无参数；返回短事务连接，连接及锁等待失败直接暴露。"""
        with psycopg.connect(
            self._database_url, connect_timeout=DATABASE_TIMEOUT_SECONDS,
            options="-c statement_timeout=5000 -c lock_timeout=5000",
        ) as connection:
            yield connection

    def initialize(self) -> None:
        """无参数；幂等创建协调表，初始化失败阻止模型发送。"""
        with self.connection() as connection:
            # 1. 【模型服务】【协调初始化】并发进程串行建表，不重置已有窗口历史
            connection.execute("SELECT pg_advisory_xact_lock(734910621)")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS provider_quota_groups (
                    group_id TEXT PRIMARY KEY, policy JSONB NOT NULL,
                    next_send_at TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01 UTC',
                    cooldown_until TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01 UTC'
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS provider_quota_requests (
                    sequence BIGSERIAL PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
                    group_id TEXT NOT NULL REFERENCES provider_quota_groups(group_id),
                    run_id TEXT NOT NULL, call_id TEXT NOT NULL, attempt INTEGER NOT NULL,
                    lease_until TIMESTAMPTZ NOT NULL,
                    sent_at TIMESTAMPTZ, charged_at TIMESTAMPTZ, completed_at TIMESTAMPTZ,
                    ownership_lost_at TIMESTAMPTZ
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS provider_quota_group_sequence
                ON provider_quota_requests(group_id, sequence)
            """)

    def enqueue(self, group: str, policy: dict, request: dict) -> None:
        """参数为额度组、冻结策略和请求身份；原子核对策略后加入 FIFO。"""
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO provider_quota_groups(group_id, policy) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING", (group, Jsonb(policy)),
            )
            row = connection.execute(
                "SELECT policy FROM provider_quota_groups WHERE group_id=%s FOR UPDATE",
                (group,),
            ).fetchone()
            if row[0] != policy:
                raise ValueError(f"模型额度组策略冲突: {group}")
            connection.execute("""
                INSERT INTO provider_quota_requests
                    (request_id, group_id, run_id, call_id, attempt, lease_until)
                VALUES (%s, %s, %s, %s, %s, clock_timestamp() + %s * interval '1 second')
            """, (request["request_id"], group, request["run_id"], request["call_id"],
                  request["attempt"], LEASE_SECONDS))

    def poll(self, group: str, policy: dict, request_id: str, check_active) -> dict:
        """参数为组、策略、请求及取消检查；返回准入连接，资格持有至实际 HTTP 返回。"""
        connection = psycopg.connect(self._database_url, connect_timeout=DATABASE_TIMEOUT_SECONDS,
                                     options="-c statement_timeout=5000 -c lock_timeout=5000")
        admitted = False
        try:
            updated = connection.execute("""
                UPDATE provider_quota_requests
                SET lease_until=clock_timestamp() + %s * interval '1 second'
                WHERE request_id=%s AND sent_at IS NULL AND lease_until > clock_timestamp()
                RETURNING sequence
            """, (LEASE_SECONDS, request_id)).fetchone()
            if updated is None:
                raise RuntimeError("模型排队租约已失效，当前请求不得继续发送")
            # 1. 【模型服务】【发送资格】try-lock不阻塞等待者；没有事务跨越轮询睡眠
            locked = connection.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 734910622))", (group,),
            ).fetchone()[0]
            if not locked:
                connection.commit()
                return {"admitted": False, "estimated_wait_seconds": 0.0}
            result = self._admission(connection, group, policy, request_id)
            check_active()
            connection.commit()
            admitted = result["admitted"]
            return {**result, "_connection": connection} if admitted else result
        finally:
            if not admitted:
                connection.close()

    def _admission(self, connection, group: str, policy: dict, request_id: str) -> dict:
        """参数为独占发送资格连接及请求；核对队首和保守计费窗口，返回是否可发送。"""
        with connection.transaction():
            # 1. 【模型服务】【崩溃恢复】已发但未释放的资格只能在旧连接断开后被观察
            recovered = connection.execute("""
                UPDATE provider_quota_requests
                SET ownership_lost_at=clock_timestamp(), charged_at=clock_timestamp()
                WHERE group_id=%s AND sent_at IS NOT NULL
                  AND completed_at IS NULL AND ownership_lost_at IS NULL RETURNING request_id
            """, (group,)).fetchall()
            if recovered:
                connection.execute("""
                    UPDATE provider_quota_groups
                    SET next_send_at=GREATEST(next_send_at, clock_timestamp() + %s * interval '1 second')
                    WHERE group_id=%s
                """, (policy["min_interval_seconds"], group))
            row = connection.execute("""
                SELECT next_send_at, cooldown_until FROM provider_quota_groups
                WHERE group_id=%s FOR UPDATE
            """, (group,)).fetchone()
            now = connection.execute("SELECT clock_timestamp()").fetchone()[0]
            # 1. 【模型服务】【队首回收】过期等待者不得复活；已发送记录只按窗口清理
            connection.execute("""
                DELETE FROM provider_quota_requests WHERE group_id=%s AND
                ((sent_at IS NULL AND lease_until <= %s) OR
                 charged_at <= %s - %s * interval '1 second')
            """, (group, now, now, policy["window_seconds"]))
            head = connection.execute("""
                SELECT request_id FROM provider_quota_requests
                WHERE group_id=%s AND sent_at IS NULL ORDER BY sequence LIMIT 1
            """, (group,)).fetchone()[0]
            history = connection.execute("""
                SELECT charged_at FROM provider_quota_requests
                WHERE group_id=%s AND charged_at IS NOT NULL ORDER BY charged_at
            """, (group,)).fetchall()
            wait = max(0.0, (row[0] - now).total_seconds(), (row[1] - now).total_seconds())
            if len(history) >= policy["requests_per_window"]:
                wait = max(wait, (history[0][0] - now).total_seconds() + policy["window_seconds"])
            if head != request_id or wait > 0:
                return {"admitted": False, "estimated_wait_seconds": wait}
            return {"admitted": True, "admitted_at": now.isoformat(), "estimated_wait_seconds": 0.0}

    def mark_started(self, connection, request_id: str) -> str:
        """参数为资格连接和请求；发送起点持久计费，进程崩溃也不退还未知结果额度。"""
        with connection.transaction():
            row = connection.execute("""
                UPDATE provider_quota_requests SET sent_at=clock_timestamp(), charged_at=clock_timestamp()
                WHERE request_id=%s RETURNING sent_at
            """, (request_id,)).fetchone()
        return row[0].isoformat()

    def release(self, connection, request_id: str, policy: dict) -> None:
        """参数为资格连接、请求及策略；记录响应返回边界，释放资格且不冒充实际发送时间。"""
        try:
            with connection.transaction():
                row = connection.execute("""
                    UPDATE provider_quota_requests
                    SET completed_at=clock_timestamp(), charged_at=clock_timestamp()
                    WHERE request_id=%s AND sent_at IS NOT NULL RETURNING group_id, completed_at
                """, (request_id,)).fetchone()
                if row is None:
                    connection.execute("DELETE FROM provider_quota_requests WHERE request_id=%s", (request_id,))
                else:
                    # 1. 【模型服务】【保守计费】以响应返回上界推进额度，准入后暂停不会造成突发
                    connection.execute("""
                        UPDATE provider_quota_groups SET next_send_at=%s + %s * interval '1 second'
                        WHERE group_id=%s
                    """, (row[1], policy["min_interval_seconds"], row[0]))
        finally:
            connection.close()

    def cancel(self, request_id: str) -> None:
        """参数为等待请求身份；只删除未发条目，不退还已发或结果未知额度。"""
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM provider_quota_requests WHERE request_id=%s AND sent_at IS NULL",
                (request_id,),
            )

    def cooldown(self, group: str, seconds: float) -> None:
        """参数为组和冷却秒数；延长共享冷却，不缩短其他进程观察到的期限。"""
        with self.connection() as connection:
            connection.execute("""
                UPDATE provider_quota_groups SET cooldown_until=GREATEST(
                    cooldown_until, clock_timestamp() + %s * interval '1 second')
                WHERE group_id=%s
            """, (seconds, group))
