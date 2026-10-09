"""
统一管理 FastAPI 生命周期。

这个文件回答两个关键问题：
1. 应用启动时，哪些资源先初始化，顺序是什么
2. 应用关闭时，哪些资源必须显式释放，顺序是什么

对理解项目很重要的一点是：
启动逻辑没有散落在 `main.py`、`server.py`、各个 service 中，
而是统一收敛到这里，便于排查“为什么服务能启动 / 为什么退出时资源没关干净”。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from contextlib import suppress
import time
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI

from app.config import get_settings

logger = logging.getLogger(__name__)
OPTIONAL_WARM_TIMEOUT_SECONDS = 8.0
PROTECTED_ENVS = {"staging", "prod"}
KNOWN_WEAK_SECRETS = {
    "",
    "123456",
    "change-me-in-production",
    "changeme",
    "minioadmin",
    "neo4j_password",
    "password",
    "postgres",
    "redis_password",
    "studyassistant2024!",
}


def get_runtime_container():
    """惰性导入运行时容器，便于测试替换。"""
    from app.runtime.container import get_runtime_container as _get_runtime_container

    return _get_runtime_container()


def get_db_runtime():
    """惰性导入数据库 runtime getter，便于测试替换。"""
    from app.runtime.container import get_db_runtime as _get_db_runtime

    return _get_db_runtime()


def get_conversation_runtime():
    """惰性导入会话 runtime getter，便于测试替换。"""
    from app.runtime.container import (
        get_conversation_runtime as _get_conversation_runtime,
    )

    return _get_conversation_runtime()


def get_agent_runtime():
    """惰性导入 agent runtime getter，便于测试替换。"""
    from app.runtime.container import get_agent_runtime as _get_agent_runtime

    return _get_agent_runtime()


async def close_rag_run_executors() -> None:
    """取消并等待本进程持有的流式 run，在关闭数据库前写入终态。"""
    from app.services.rag_run_executor_service import get_rag_run_executor_registry

    await get_rag_run_executor_registry().close()


async def close_embedded_workers() -> None:
    """通知内嵌 worker 停止领取，并等待当前持租任务结束。"""
    from app.services.embedded_worker_service import close_embedded_workers as close_workers

    await close_workers()


def start_upload_worker() -> None:
    """【上传】【启动接续】消费已排队及失租任务；暂停记录不会被队列领取，无返回值。"""
    from app.services.embedded_worker_service import ensure_embedded_worker_running

    ensure_embedded_worker_running(task_types=["upload_ingest"], stop_when_idle=False)


def _is_known_weak_secret(value: object) -> bool:
    """
    判断凭证是否为空或命中仓库历史公开弱值。

    Args:
        value: 需要检查的凭证值。

    Returns:
        bool: 为空或已知弱值时返回 True。
    """
    normalized = unquote(str(value or "")).strip().casefold()
    return normalized in KNOWN_WEAK_SECRETS


def _database_password(database_url: str) -> str:
    """
    从 PostgreSQL 连接串提取密码并校验基础结构。

    Args:
        database_url: PostgreSQL 业务库连接串。

    Returns:
        str: URL 中经过解码的数据库密码。
    """
    try:
        parsed = urlsplit(database_url)
    except ValueError as exc:
        raise ValueError("DATABASE_URL 格式无效") from exc
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("DATABASE_URL 必须是完整的 PostgreSQL 连接串")
    return unquote(parsed.password or "")


def validate_startup_infrastructure_configuration(settings: object) -> None:
    """
    校验启动所需数据库与已启用图数据库凭证。

    Args:
        settings: 当前应用配置。

    Returns:
        None: 配置合法时无返回；配置不合法时抛出 ValueError。
    """
    app_env = str(getattr(settings, "app_env", "local")).strip().lower()
    database_url = str(getattr(settings, "database_url", "") or "").strip()
    if getattr(settings, "db_backend", "postgresql") == "postgresql":
        password = _database_password(database_url)
        if app_env in PROTECTED_ENVS and _is_known_weak_secret(password):
            raise ValueError("staging/prod 的 DATABASE_URL 必须包含非弱值密码")

    if not bool(getattr(settings, "neo4j_enabled", False)):
        return
    neo4j_uri = str(getattr(settings, "neo4j_uri", "") or "").strip()
    neo4j_username = str(getattr(settings, "neo4j_username", "") or "").strip()
    neo4j_password = getattr(settings, "neo4j_password", "")
    if not neo4j_uri or not neo4j_username or not str(neo4j_password or "").strip():
        raise ValueError("已启用 Neo4j 时必须配置有效 URI、用户名和密码")
    if app_env in PROTECTED_ENVS and _is_known_weak_secret(neo4j_password):
        raise ValueError("staging/prod 的 Neo4j 密码不能使用已知弱值")


def validate_startup_auth_configuration() -> None:
    """
    执行启动期认证、数据库与已启用基础设施配置校验。

    Args:
        无

    Returns:
        None: 配置合法时无返回；配置不合法时抛异常并阻断启动
    """
    from app.services.auth_service import validate_auth_configuration

    settings = get_settings()
    # 1. 先校验认证边界，再校验外部连接凭证，任何失败都发生在 runtime 初始化前
    validate_auth_configuration(settings)
    validate_startup_infrastructure_configuration(settings)


async def _warm_optional_resources() -> None:
    """
    预热缓存、向量与图谱增强资源；失败时记录告警但不阻断启动。

    这些资源属于“可选增强能力”而不是服务硬前置：
    即使预热失败，主 API 也应该先起来，避免启动期被外部依赖全部卡死。

    Returns:
        None: 预热只做副作用操作，不返回业务值
    """
    # 1. 先做最轻量的缓存预热，避免后续请求首次命中时再初始化
    try:
        from app.services.cache import get_cache

        get_cache()
    except Exception as exc:  # pragma: no cover - 防御性日志
        logger.warning("缓存预热失败: %s", exc)

    async def _run_step(name: str, runner) -> None:
        started_at = time.monotonic()
        try:
            await asyncio.wait_for(runner(), timeout=OPTIONAL_WARM_TIMEOUT_SECONDS)
            logger.info("%s 预热完成 (%.2fs)", name, time.monotonic() - started_at)
        except asyncio.TimeoutError:
            logger.warning("%s 预热超时，已跳过阻塞式等待", name)
        except Exception as exc:  # pragma: no cover - 防御性日志
            logger.warning("%s 预热失败: %s", name, exc)

    settings = get_settings()
    # 2. 再按开关条件预热 Milvus / LightRAG / Neo4j 这些增强能力
    if settings.vector_backend == "milvus":
        from app.services.milvus_store import get_milvus_store

        await _run_step(
            "Milvus",
            lambda: asyncio.to_thread(get_milvus_store().warmup),
        )

    if settings.graph_rag_enabled:
        from app.services.lightrag_service import get_lightrag_service

        await _run_step("LightRAG", lambda: get_lightrag_service().initialize())
    if settings.neo4j_enabled:
        from app.services.graph import get_neo4j_schema_service

        await _run_step("Neo4j", lambda: get_neo4j_schema_service().initialize())


def build_app_lifespan():
    """返回 FastAPI lifespan：依次初始化业务库、会话库、agent 和后台资源，关闭时逆序释放。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 安全配置必须先于 runtime/container 解析，避免 getter 引入连接副作用后绕过启动守卫
        validate_startup_auth_configuration()
        db_runtime = get_db_runtime()
        conversation_runtime = get_conversation_runtime()
        agent_runtime = get_agent_runtime()
        db_initialized = False
        conversation_initialized = False
        agent_initialized = False
        # 把核心运行时挂到 `app.state`，便于诊断和测试时访问。
        app.state.runtime_container = get_runtime_container()
        app.state.optional_warmup_task = None

        try:
            # 2. 安全配置通过后再初始化核心运行时，确保服务最小可用
            await db_runtime.initialize()
            db_initialized = True
            await conversation_runtime.initialize()
            conversation_initialized = True
            await agent_runtime.initialize()
            agent_initialized = True
            # 3. 【上传】【启动接续】持续检查持久队列，进程重启后等待旧租约过期也能接续
            start_upload_worker()
            # 3. 再把可选增强资源放到后台预热，避免启动阶段被这些能力拖慢
            app.state.optional_warmup_task = asyncio.create_task(_warm_optional_resources())
            await asyncio.sleep(0)
            yield
        finally:
            warmup_task = getattr(app.state, "optional_warmup_task", None)
            if warmup_task is not None and not warmup_task.done():
                warmup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup_task
            # 4. 先收敛请求外执行体和 worker，再释放它们依赖的 agent / 数据库
            await close_rag_run_executors()
            await close_embedded_workers()
            if agent_initialized:
                await agent_runtime.close()
            if agent_initialized and get_settings().neo4j_enabled:
                from app.services.graph import get_neo4j_client

                await get_neo4j_client().close()
            if conversation_initialized:
                await conversation_runtime.close()
            if db_initialized:
                await db_runtime.close()
            app.state.optional_warmup_task = None
            app.state.runtime_container = None

    return lifespan
