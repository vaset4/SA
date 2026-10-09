"""
运行时容器，统一暴露核心资源入口。

可以把这个文件理解成“进程级资源注册表”。它不负责业务逻辑，
只负责把几个需要跨请求复用、又必须在应用生命周期里显式初始化/释放的资源
收拢到一个地方，避免这些初始化逻辑散落在 route 或 service 里。

当前主要管理三类资源：
1. AgentRuntime: 共享 LangGraph + checkpointer
2. DatabaseRuntime: 业务数据库与通用连接池
3. ConversationRuntime: 前端会话展示库
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.api.agent_singleton import close_shared_agent, get_shared_agent
from app.repositories.project_repository import get_project_repository
from app.repositories.research_memory_repository import get_research_memory_repository
from app.services.conversation_service import close_conversation_db, ensure_conversation_schema
from app.services.database import close_pool
from app.services.rag_run_service import ensure_rag_run_schema, expire_overdue_runs
import app.services.paper_db as paper_db


@dataclass(slots=True)
class AgentRuntime:
    """共享 graph/checkpointer 运行时。"""

    async def initialize(self) -> Any:
        """
        预热共享 agent 与官方 checkpointer。

        这里不创建“每个请求独立的 agent”，而是拿进程级单例，
        这样所有请求共享同一份 compile 后的图结构和 checkpoint 基础设施。
        """
        return await get_shared_agent()

    async def close(self) -> None:
        """释放共享 agent 与官方 checkpointer。"""
        await close_shared_agent()


@dataclass(slots=True)
class DatabaseRuntime:
    """业务数据库运行时。"""

    async def initialize(self) -> None:
        """无参数；按基础用户论文、项目依赖顺序初始化业务表与后台线程。"""
        await asyncio.to_thread(paper_db.init_db)
        # 1. 【应用启动】【项目依赖】记忆表依赖项目外键，先完成项目仓储再启动会话消费者
        await get_project_repository().ensure_schema()

    async def close(self) -> None:
        """关闭业务数据库连接，并补充关闭通用 asyncpg 连接池。"""
        await asyncio.to_thread(paper_db.close_db)
        await close_pool()


@dataclass(slots=True)
class ConversationRuntime:
    """前端会话展示库运行时。"""

    async def initialize(self) -> None:
        """初始化 conversations/messages 表与单例仓储。"""
        await ensure_conversation_schema()
        await ensure_rag_run_schema()
        # 【研究执行】【启动核对】期限只发停止信号，不把失联的旧执行者视为已经退出
        await expire_overdue_runs()
        # 研究记忆层 6 张表随会话展示库一同建好（同为纯 async 业务表）
        await get_research_memory_repository().ensure_schema()

    async def close(self) -> None:
        """关闭前端会话展示库连接。"""
        close_conversation_db()


@dataclass(slots=True)
class RuntimeContainer:
    """
    集中维护生命周期涉及的核心资源。

    FastAPI `app.state` 里会持有这个容器，便于调试、测试替换和按需访问。
    """

    agent: AgentRuntime = field(default_factory=AgentRuntime)
    db: DatabaseRuntime = field(default_factory=DatabaseRuntime)
    conversation: ConversationRuntime = field(default_factory=ConversationRuntime)


_RUNTIME_CONTAINER = RuntimeContainer()


def get_runtime_container() -> RuntimeContainer:
    """返回进程级运行时容器。"""
    return _RUNTIME_CONTAINER


def get_agent_runtime() -> AgentRuntime:
    """返回共享 agent 运行时。"""
    return get_runtime_container().agent


def get_db_runtime() -> DatabaseRuntime:
    """返回业务数据库运行时。"""
    return get_runtime_container().db


def get_conversation_runtime() -> ConversationRuntime:
    """返回前端会话展示库运行时。"""
    return get_runtime_container().conversation
