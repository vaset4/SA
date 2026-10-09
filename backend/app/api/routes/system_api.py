"""系统与运维辅助路由。"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.facades import system_facade
from app.api.schemas.research import GraphQueryRequest
from app.api.schemas.system import NotificationPatch
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.services.auth_roles import can_view_observability

router = APIRouter()
logger = logging.getLogger(__name__)

# 这些接口不是主问答链的一部分，但它们对“理解系统当前状态”很关键：
#   - `/health` 看底层依赖是否可用
#   - `/graph/*` 看图谱侧统计和调试查询
#   - `/notifications` 看后台任务或系统通知


def _require_developer(user: dict) -> None:
    """
    限制破坏性系统维护入口只能由 developer 或 admin 调用。

    Args:
        user: 当前认证用户
    Returns:
        无返回值；权限不足时抛出 403
    """
    if not can_view_observability(user):
        raise HTTPException(status_code=403, detail="developer or admin role required")


@router.get("/health")
async def health():
    """聚合基础依赖健康状态；方便前端或运维快速判断当前系统是否降级。"""
    return await system_facade.build_health_payload()


@router.get("/graph/stats")
async def graph_stats(
    library_id: str | None = None,
    project_id: str | None = None,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """返回 LightRAG 图谱统计；支持按显式 library 或按 project 上下文解析。"""
    return await system_facade.graph_stats_payload(library_id, project_id, user_id=get_user_id(user))


@router.get("/graph/neo4j/stats")
async def neo4j_graph_stats(
    library_id: str | None = None,
    project_id: str | None = None,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """返回 Neo4j 图谱统计与数据库对账结果。"""
    return await system_facade.neo4j_graph_stats_payload(library_id, project_id, user_id=get_user_id(user))


@router.post("/graph/query")
async def graph_query(req: GraphQueryRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    """
    直接调用 LightRAG 查询接口。

    这是偏调试/运维用途的入口，绕过主图直接观察 GraphRAG 层返回。
    """
    return await system_facade.graph_query_payload(
        req.query,
        req.mode,
        req.top_k,
        req.library_id,
        req.project_id,
        user_id=get_user_id(user),
    )


@router.post("/system/reconciliation/paper-orphans")
async def reconcile_paper_orphans(
    apply: bool = Query(default=False),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """
    默认 dry-run 对账论文派生存储；只有显式 apply=true 才执行清理。

    Args:
        apply: 是否执行对账报告中的精确清理
        user: 当前认证用户
    Returns:
        五类派生存储的对账报告
    """
    _require_developer(user)
    try:
        return await system_facade.reconcile_paper_orphans_payload(apply=apply)
    except Exception as exc:
        logger.error(
            "【论文对账】【请求失败】apply=%s error=%s",
            apply,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail="论文孤儿数据对账失败") from exc


@router.get("/notifications")
async def get_notifications(user: dict = Depends(get_current_user_with_rate_limit)):
    """读取当前用户的通知列表。"""
    return system_facade.list_notifications_payload(get_user_id(user))


@router.patch("/notifications/{log_id}")
async def mark_notification_read(log_id: str, body: NotificationPatch, user: dict = Depends(get_current_user_with_rate_limit)):
    """更新当前用户通知的已读状态；越权或找不到记录时转换为 404。"""
    try:
        return system_facade.mark_notification_read_payload(log_id, get_user_id(user), body.read)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
