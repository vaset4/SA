"""
研究记忆层真库测试基座。

阶段 2-4/8/10 的仓储与服务测试需要真实 Postgres 才能验证建表、CRUD 往返、
外键级联删除等行为。本模块提供：
- can_connect_db(): 探测本地库是否可连，不可连时测试整体 skip（无库环境不报错）
- temp_project(): 建一个临时 project 作外键归属，测试结束级联删除清理

清理策略：删 project 触发 ON DELETE CASCADE，研究记忆各表数据随之清空，
表结构保留（对齐「测试后自动清理数据」的约定）。
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

from app.repositories.research_memory_repository import get_research_memory_repository
from app.services.database import close_pool, execute, fetchval


def can_connect_db() -> bool:
    """探测业务库是否可连；不可连时返回 False，供 skipif 跳过真库测试。"""
    async def _probe() -> bool:
        try:
            value = await fetchval("SELECT 1")
            return value == 1
        except Exception:
            return False
        finally:
            # 探测用的临时池随本次 loop 一起关闭，避免残留死池
            await close_pool()

    try:
        return asyncio.run(_probe())
    except Exception:
        return False


@asynccontextmanager
async def temp_project():
    """
    建一个临时 project 并保证研究记忆 6 张表存在，退出时级联删除清理。

    产出:
        project_id: 临时项目 ID，作各记忆实体的外键归属
    """
    project_id = f"proj_rmemtest_{uuid.uuid4().hex[:8]}"
    await get_research_memory_repository().ensure_schema()
    await execute(
        "INSERT INTO projects (project_id, user_id, name) VALUES ($1,$2,$3)",
        project_id,
        "rmem_tester",
        "研究记忆测试项目",
    )
    try:
        yield project_id
    finally:
        # 删 project 触发级联，研究记忆各表随之清空
        await execute("DELETE FROM projects WHERE project_id=$1", project_id)
        # 关闭当前 event loop 的连接池：多个 asyncio.run 会各起新 loop，
        # 而连接池按 loop id 缓存；不清理会让下个复用同 id 的 loop 拿到指向已关闭
        # loop 的死池，acquire 永久挂起。
        await close_pool()
