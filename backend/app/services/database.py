"""
PostgreSQL 业务库共享访问入口。

这个文件是项目里所有 async PostgreSQL 访问的公共底座，负责：
- 读取数据库连接配置
- 为不同事件循环维护各自的 asyncpg 连接池
- 提供 execute/fetch/fetchrow/fetchval 这些统一访问原语

最关键的设计点是“按事件循环隔离连接池”。
因为 FastAPI、测试环境、后台任务可能运行在不同 loop 上，
直接全局共享一个 asyncpg pool 容易产生跨 loop 复用问题。
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from app.config import get_settings


@dataclass(slots=True)
class _PoolEntry:
    """记录每个事件循环对应的 asyncpg 连接池。"""

    loop: asyncio.AbstractEventLoop
    pool: Any


_pools: dict[int, _PoolEntry] = {}
_pools_lock = threading.Lock()


def get_database_url() -> str:
    """返回业务数据库连接串。"""
    return get_settings().database_url


def _load_asyncpg():
    """
    延迟导入 asyncpg，缺失时明确抛错。

    这样模块导入本身不会因为环境缺依赖而立即失败，
    只有真正需要访问 PostgreSQL 时才会暴露清晰错误。
    """
    try:
        import asyncpg
    except ImportError as exc:
        raise RuntimeError("缺少 asyncpg 依赖，无法创建 PostgreSQL 连接池") from exc
    return asyncpg


async def get_pool():
    """
    获取或创建当前事件循环专属的 asyncpg 连接池。

    这里不是简单的进程级单例，而是“每个 loop 一份 pool”。
    如果同一 loop 下发生并发初始化，后到者会复用先创建成功的 pool，
    自己新建的 pool 会立即关闭，避免泄漏。
    """
    loop = asyncio.get_running_loop()
    loop_key = id(loop)
    with _pools_lock:
        existing_entry = _pools.get(loop_key)
    if existing_entry is not None:
        return existing_entry.pool

    asyncpg = _load_asyncpg()
    settings = get_settings()
    pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
        command_timeout=settings.database_command_timeout,
    )
    with _pools_lock:
        existing_entry = _pools.get(loop_key)
        if existing_entry is not None:
            # 同一事件循环并发初始化时，复用先创建成功的连接池。
            entry_pool = existing_entry.pool
        else:
            _pools[loop_key] = _PoolEntry(loop=loop, pool=pool)
            entry_pool = pool

    if entry_pool is not pool:
        await pool.close()
    return entry_pool


async def execute(query: str, *args):
    """执行写操作 SQL。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args):
    """执行多行查询 SQL。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args):
    """执行单行查询 SQL。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args):
    """执行单值查询 SQL。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def close_pool() -> None:
    """
    关闭当前事件循环持有的业务库连接池。

    这通常在应用生命周期结束时调用，而不是每次查询后调用。
    """
    loop_key = id(asyncio.get_running_loop())
    with _pools_lock:
        entry = _pools.pop(loop_key, None)
    if entry is None:
        return
    await entry.pool.close()
