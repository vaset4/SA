"""来源文件读取与显式删除的跨进程临界区，作者：xxx。"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
import hashlib
from typing import Any

from app.repositories.upload_repository import upload_row
from app.services.database import get_pool


def source_lock_key(paper_id: str) -> int:
    """参数为论文身份；返回读者与删除 worker 共用的稳定 PostgreSQL bigint 锁键。"""
    digest = hashlib.sha256(("studyassistant:source-files:" + paper_id).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


class SourceAccessRepository:
    """只协调正在进行的文件工作；长研究租约不承担物理删除互斥。"""

    def __init__(self, pool_provider: Callable[[], Awaitable[Any]] = get_pool) -> None:
        """参数为连接池端口；创建不主动访问数据库的来源占用仓储。"""
        self._pool_provider = pool_provider

    @asynccontextmanager
    async def file_access(self, paper_ids: tuple[str, ...], *, exclusive: bool = False) -> AsyncIterator[Any]:
        """参数为论文集合及是否删除；按稳定顺序取得占用，返回同连接供持锁后的授权核对。"""
        if not paper_ids or any(not isinstance(item, str) or not item.strip() for item in paper_ids):
            raise ValueError("来源文件占用需要明确论文身份")
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            statement = "SELECT pg_advisory_xact_lock($1::bigint)" if exclusive else "SELECT pg_advisory_xact_lock_shared($1::bigint)"
            # 1. 【来源】【文件占用】所有入口使用相同顺序，避免批量读取与删除相互等待
            for paper_id in sorted(set(paper_ids)):
                await connection.execute(statement, source_lock_key(paper_id))
            yield connection

    async def source_reference(self, connection: Any, paper_id: str, *, user_id: str) -> dict[str, Any]:
        """参数为持锁连接、论文及账号；返回当前有权下载的不可变原文件记录。"""
        row = await connection.fetchrow("""SELECT d.* FROM upload_documents d LEFT JOIN papers p ON p.paper_id=d.paper_id
            WHERE d.paper_id=$1 AND d.user_id=$2 AND d.state<>'deleted'
            AND (p.paper_id IS NULL OR (p.user_id=$2 AND p.deletion_status='active'))""", paper_id, user_id)
        result = upload_row(row)
        if result is None:
            raise LookupError("原文不存在或无权访问")
        return result


def get_source_access_repository() -> SourceAccessRepository:
    """无参数；返回使用现有数据库的跨进程文件占用端口。"""
    return SourceAccessRepository()
