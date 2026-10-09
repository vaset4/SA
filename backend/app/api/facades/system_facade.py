"""系统接口相关 facade。"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
import app.services.paper_db as paper_db
from app.services.library_service import require_library
from app.services.database import get_pool
from app.services.capability_registry import capability_snapshot
from app.services.graph import get_neo4j_maintenance_service, get_research_object_query_service
from app.services.lightrag_service import get_lightrag_service
from app.services.milvus_store import get_milvus_store
from app.services.paper_orphan_reconciliation_service import reconcile_paper_orphans
from app.services.project_scope_service import resolve_project_scope

settings = get_settings()

# 这个 facade 把“系统状态查询”从 route 层抽出来，便于：
#   - 统一聚合 Postgres / Milvus / LightRAG / Neo4j 健康状态
#   - 复用 project -> library 的解析逻辑
#   - 让 route 保持非常薄，只做鉴权与参数传递


async def _postgres_health() -> bool:
    """检查 PostgreSQL 连接池与基础查询。"""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return (await conn.fetchval("SELECT 1")) == 1
    except Exception:
        return False


def _milvus_health() -> bool:
    """检查 Milvus 健康状态。"""
    try:
        return get_milvus_store().is_healthy()
    except Exception:
        return False


def _llm_health() -> bool:
    """检查 LLM 实例是否已成功初始化。"""
    try:
        from app.services.llm_service import get_llm

        return get_llm() is not None
    except Exception:
        return False


def _cache_stats() -> dict[str, Any]:
    """返回缓存统计。"""
    try:
        from app.services.cache import get_cache

        return get_cache().get_stats()
    except Exception:
        return {}


async def _redis_health() -> bool:
    """检查 Redis 二级缓存连通性。"""
    if not getattr(settings, "redis_cache_enabled", False):
        return False
    try:
        from app.services.cache import get_cache

        cache = get_cache()
        l2_cache = getattr(cache, "l2", None)
        if l2_cache is None:
            return False
        redis_client = await l2_cache._get_redis()
        return bool(await redis_client.ping())
    except Exception:
        return False


async def _lightrag_health() -> bool:
    """检查 LightRAG 健康状态。"""
    try:
        return await get_lightrag_service().is_healthy()
    except Exception:
        return False


async def _neo4j_health() -> dict[str, Any]:
    """检查 Neo4j 研究对象图谱健康状态。"""
    try:
        return await get_research_object_query_service().health()
    except Exception:
        return {"enabled": bool(getattr(settings, "neo4j_enabled", False)), "connected": False, "schema_ready": False}


async def build_health_payload() -> dict[str, Any]:
    """
    聚合基础依赖健康状态。

    这里并不是所有依赖都一票否决：
      - Postgres / Milvus / LLM 属于主链核心
      - Redis / LightRAG / Neo4j 是否纳入健康判定，取决于功能开关
    """
    checks: dict[str, Any] = {
        "postgres": await _postgres_health(),
        "milvus": _milvus_health(),
        "llm": _llm_health(),
        "cache": _cache_stats(),
        "redis": await _redis_health(),
    }
    healthy_dependencies = [checks["postgres"], checks["milvus"], checks["llm"]]
    if getattr(settings, "redis_cache_enabled", False):
        healthy_dependencies.append(checks["redis"])
    if settings.graph_rag_enabled:
        checks["lightrag"] = await _lightrag_health()
        healthy_dependencies.append(checks["lightrag"])
    if getattr(settings, "neo4j_enabled", False):
        checks["neo4j"] = await _neo4j_health()
        healthy_dependencies.append(bool(checks["neo4j"].get("connected")))
    return {
        "status": "healthy" if all(healthy_dependencies) else "degraded",
        "postgres": checks["postgres"],
        "milvus": checks["milvus"],
        "checks": checks,
    }


async def _resolve_library_id(
    library_id: str | None = None,
    project_id: str | None = None,
    *,
    user_id: str = "anonymous",
) -> str | None:
    """
    根据用户上下文解析目标 library。

    优先级是显式 `library_id`，否则再尝试通过 `project_id` 反查 project scope。
    这是系统类接口里最常见的“范围归一化”前置步骤。
    """
    if library_id:
        library = await require_library(library_id, owner=user_id)
        return str(library.get("library_id") or library_id)
    if not project_id:
        return None
    scope = await resolve_project_scope(project_id, user_id=user_id)
    library_ids = (scope or {}).get("library_ids", [])
    return library_ids[0] if library_ids else None


async def graph_stats_payload(
    library_id: str | None = None,
    project_id: str | None = None,
    *,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """返回 LightRAG 图谱统计。"""
    resolved_library_id = await _resolve_library_id(library_id, project_id, user_id=user_id)
    return await get_lightrag_service().get_stats(library_id=resolved_library_id)


async def neo4j_graph_stats_payload(
    library_id: str | None = None,
    project_id: str | None = None,
    *,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """返回 Neo4j 图谱统计与对账信息。"""
    resolved_scope = await resolve_project_scope(project_id, user_id=user_id) if project_id else None
    resolved_library_id = await _resolve_library_id(library_id, project_id, user_id=user_id)
    paper_ids = (resolved_scope or {}).get("paper_ids", []) if resolved_scope else []
    # 这里把 project scope 里的 paper_ids 也带给维护服务，便于做“图数据库 vs 主库”的局部对账。
    return await get_neo4j_maintenance_service().stats(resolved_library_id, paper_ids)


async def graph_query_payload(
    query: str,
    mode: str,
    top_k: int,
    library_id: str | None = None,
    project_id: str | None = None,
    *,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """直接调用 LightRAG 查询；这是绕过主图的调试型查询入口。"""
    resolved_library_id = await _resolve_library_id(library_id, project_id, user_id=user_id)
    result = await get_lightrag_service().query(
        query=query,
        mode=mode,
        top_k=top_k,
        library_id=resolved_library_id,
    )
    return {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "library_id": resolved_library_id,
        "project_id": project_id,
        "result": result,
    }


async def reconcile_paper_orphans_payload(*, apply: bool = False) -> dict[str, Any]:
    """
    返回全局论文孤儿对账报告，仅在显式 apply 时执行清理。

    Args:
        apply: False 仅 dry-run，True 执行报告中的精确清理
    Returns:
        五类派生存储的对账结果
    """
    return await reconcile_paper_orphans(apply=apply)


def list_notifications_payload(user_id: str) -> dict[str, Any]:
    """读取当前用户通知列表；数据库异常必须显式暴露给调用方。"""
    notifications = paper_db.list_notifications(user_id)
    return {"notifications": notifications}


def mark_notification_read_payload(log_id: str, user_id: str, read: bool) -> dict[str, Any]:
    """按归属更新通知已读状态；越权或不存在时抛 404。"""
    updated = paper_db.set_notification_read_state(log_id, user_id, read)
    if not updated:
        raise LookupError("通知不存在")
    return {"log_id": log_id, "read": read}


def capability_snapshot_payload() -> dict[str, Any]:
    """返回当前能力注册快照；常用于前端展示当前部署启用了哪些能力。"""
    return capability_snapshot()
