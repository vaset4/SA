"""
共享 graph 单例与官方 checkpointer 生命周期管理。

这层的存在，是为了把两个概念绑定在一起管理：
1. 编译后的 LangGraph 主图
2. 该主图依赖的 PostgreSQL checkpointer

为什么这里要做成模块级单例：
- compile 图本身有成本，没必要每个请求重建
- checkpointer 需要和图保持一致的生命周期
- API 层只应该“拿现成可用的 graph”，不应该知道底层怎么建和怎么关
"""

import asyncio
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agents.graph_factory import create_compiled_graph
from app.services.database import get_database_url
from app.services.checkpoint_service import read_graph_snapshot

_agent = None
_checkpoint_manager = None
_init_lock = asyncio.Lock()


async def _build_agent():
    """
    构建共享 CompiledGraph，并接入官方 PostgreSQL checkpointer。

    `AsyncPostgresSaver` 用上下文管理器创建，因此这里同时返回 manager，
    关闭时再统一执行 `__aexit__`，避免泄漏连接。

    Returns:
        tuple: `(compiled_graph, manager)`，供模块级单例生命周期统一管理
    """
    # 1. 先创建官方 PostgreSQL checkpointer 的上下文管理器
    manager = AsyncPostgresSaver.from_conn_string(get_database_url())
    checkpointer = await manager.__aenter__()
    # 2. 再初始化 checkpointer 所需表结构，并编译共享 graph
    await checkpointer.setup()
    return create_compiled_graph(checkpointer=checkpointer), manager


async def get_shared_agent():
    """
    返回全局单例 graph。

    这里用 `_init_lock` 做双重检查，防止并发首访时重复构建 graph/checkpointer。

    Returns:
        Any: 全局共享的 compiled graph 实例
    """
    global _agent, _checkpoint_manager
    if _agent is not None:
        return _agent

    async with _init_lock:
        if _agent is None:
            _agent, _checkpoint_manager = await _build_agent()
    return _agent


async def get_shared_agent_state(
    session_id: str,
    conversation_id: str | None = None,
    *,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """
    读取某个 session_id 对应的官方 LangGraph 状态快照。

    注意这里读的是 LangGraph checkpoint 状态，不是前端 conversations 表。
    这也是本项目里 `session` 与 `conversation` 容易混淆的地方之一。

    Args:
        session_id: LangGraph 会话 ID
        conversation_id: 可选前端对话 ID，用于补齐 thread 语义
        thread_id: sessions 表已持久化的 checkpoint key

    Returns:
        dict[str, Any] | None: 当前 session 对应的 graph checkpoint 快照
    """
    agent = await get_shared_agent()
    return await read_graph_snapshot(
        agent,
        session_id,
        conversation_id,
        thread_id=thread_id,
    )


async def close_shared_agent():
    """
    关闭共享 PostgreSQL checkpointer，并清空模块级单例引用。

    先清空引用再退出上下文，可以避免关闭过程中被新的调用误读到旧对象。

    Returns:
        None: 只做资源释放，不返回业务值
    """
    global _agent, _checkpoint_manager
    async with _init_lock:
        # 1. 先摘掉模块级单例引用，避免关闭过程中被新请求误读到旧对象
        manager = _checkpoint_manager
        _agent = None
        _checkpoint_manager = None

    if manager is not None:
        # 2. 再安全退出 checkpointer 上下文，释放底层数据库连接
        await manager.__aexit__(None, None, None)
