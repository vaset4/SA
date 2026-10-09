"""
Neo4j 研究对象图谱接口。

这组路由提供的是“图谱视角”的能力，不直接走主图问答：
- 图谱健康检查
- 单篇论文最小子图
- 论文到论文聚合网络
- 可解释证据路径
- 演化图与争议图
- Neo4j 重建/回填运维接口

所以它更像“图谱查询与管理 API”，而不是聊天型 API。
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.schemas.graph_admin import Neo4jBackfillLibraryRequest, Neo4jRebuildPaperRequest
from app.api.schemas.controversy import ControversyMapRequest, ControversyMapResponse
from app.api.schemas.evolution import EvolutionEntryResponse, EvolutionGraphResponse
from app.api.schemas.graph import PaperGraphResponse
from app.api.schemas.graph_evidence import EvidencePathRequest
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.services import paper_db
from app.services.graph import (
    get_controversy_map_service,
    get_evolution_graph_service,
    get_explainable_answer_service,
    get_neo4j_maintenance_service,
    get_paper_graph_service,
    get_research_object_query_service,
)
from app.services.project_scope_service import resolve_project_scope

router = APIRouter()


def _ensure_owned_paper(paper_id: str, user_id: str) -> None:
    """确保请求的 paper_id 属于当前用户可访问范围。"""
    normalized = paper_id.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="paper_id 不能为空")
    if paper_db.get_paper(normalized, user_id=user_id) is None:
        raise HTTPException(status_code=404, detail=f"论文不存在: {normalized}")


def _ensure_owned_paper_ids(paper_ids: list[str], user_id: str) -> list[str]:
    """批量校验 paper_ids 的归属，并返回规范化后的列表。"""
    normalized_ids = [paper_id.strip() for paper_id in paper_ids if str(paper_id or "").strip()]
    missing: list[str] = []
    for paper_id in normalized_ids:
        if paper_db.get_paper(paper_id, user_id=user_id) is None:
            missing.append(paper_id)
    if missing:
        raise HTTPException(status_code=404, detail=f"论文不存在: {missing[0]}")
    return normalized_ids


@router.get("/graph/health")
async def graph_health(user: dict = Depends(get_current_user_with_rate_limit)):
    """返回 Neo4j graph 健康状态。"""
    return await get_research_object_query_service().health()


@router.get("/graph/papers/{paper_id}/subgraph")
async def paper_subgraph(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """
    按 `paper_id` 查询最小子图。

    这个接口适合前端做“单篇论文详情页里的图谱侧边栏”之类的小范围展示。
    """
    await asyncio.to_thread(_ensure_owned_paper, paper_id, get_user_id(user))
    return await get_research_object_query_service().get_paper_subgraph(paper_id)


@router.get("/graph/papers/{paper_id}/evidence")
async def paper_evidence(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """按论文返回最小图证据。"""
    await asyncio.to_thread(_ensure_owned_paper, paper_id, get_user_id(user))
    return await get_explainable_answer_service().build_bundle(
        paper_ids=[paper_id],
        query=f"evidence for {paper_id}",
        path_type="support",
    )


@router.get("/graph/papers/network", response_model=PaperGraphResponse)
async def paper_graph(
    project_id: str = Query(default=""),
    library_id: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=500),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """
    返回论文到论文的聚合关系图。

    若提供 `project_id`，优先按项目作用域收窄；
    否则退化为按 `library_id` 或当前用户论文集合生成图。
    """
    current_user_id = get_user_id(user)
    resolved_scope = await resolve_project_scope(project_id, user_id=current_user_id)
    if resolved_scope is None:
        # 没有项目作用域时，按 library 或当前用户可见论文构造一个临时 scope。
        resolved_scope = {
            "project_id": "",
            "library_ids": [library_id] if library_id else [],
            "paper_ids": await asyncio.to_thread(
                paper_db.list_paper_ids,
                library_id or None,
                user_id=current_user_id,
            ),
            "excluded_paper_ids": [],
            "year_from": None,
            "year_to": None,
            "task_filters": [],
            "method_filters": [],
        }
    return await get_paper_graph_service().get_paper_graph(
        project_id=project_id,
        library_id=library_id,
        resolved_scope=resolved_scope,
        limit=limit,
    )


@router.post("/graph/evidence-paths")
async def evidence_paths(
    req: EvidencePathRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """按论文集合查询最小图证据路径。"""
    owned_paper_ids = await asyncio.to_thread(
        _ensure_owned_paper_ids,
        req.paper_ids,
        get_user_id(user),
    )
    return await get_explainable_answer_service().build_bundle(
        paper_ids=owned_paper_ids,
        query=req.query,
        path_type=req.path_type,
    )


def _build_evolution_query(
    *,
    view_mode: str,
    paper_id: str,
    method: str,
    task: str,
    dataset: str,
    from_year: int | None,
    to_year: int | None,
    depth: int,
    project_id: str,
    resolved_scope: dict | None,
) -> dict[str, object]:
    """统一归一化研究演化图查询参数。"""
    return {
        "view_mode": view_mode if view_mode in {"paper", "method", "task"} else "paper",
        "paper_id": paper_id.strip(),
        "method": method.strip(),
        "task": task.strip(),
        "dataset": dataset.strip(),
        "from_year": from_year,
        "to_year": to_year,
        "depth": max(1, min(depth, 3)),
        "project_id": project_id.strip(),
        "resolved_scope": resolved_scope,
    }


@router.get("/graph/evolution", response_model=EvolutionGraphResponse)
async def graph_evolution(
    view_mode: str = Query(default="paper"),
    paper_id: str = Query(default=""),
    method: str = Query(default=""),
    task: str = Query(default=""),
    dataset: str = Query(default=""),
    from_year: int | None = Query(default=None),
    to_year: int | None = Query(default=None),
    depth: int = Query(default=1),
    project_id: str = Query(default=""),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """返回研究演化地图主 payload。"""
    resolved_scope = await resolve_project_scope(project_id, user_id=get_user_id(user))
    query = _build_evolution_query(
        view_mode=view_mode,
        paper_id=paper_id,
        method=method,
        task=task,
        dataset=dataset,
        from_year=from_year,
        to_year=to_year,
        depth=depth,
        project_id=project_id,
        resolved_scope=resolved_scope,
    )
    return await get_evolution_graph_service().get_evolution(query)


@router.get("/graph/evolution/export", response_model=EvolutionGraphResponse)
async def graph_evolution_export(
    view_mode: str = Query(default="paper"),
    paper_id: str = Query(default=""),
    method: str = Query(default=""),
    task: str = Query(default=""),
    dataset: str = Query(default=""),
    from_year: int | None = Query(default=None),
    to_year: int | None = Query(default=None),
    depth: int = Query(default=1),
    project_id: str = Query(default=""),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """返回供前端直接消费的研究演化导出结果。"""
    resolved_scope = await resolve_project_scope(project_id, user_id=get_user_id(user))
    query = _build_evolution_query(
        view_mode=view_mode,
        paper_id=paper_id,
        method=method,
        task=task,
        dataset=dataset,
        from_year=from_year,
        to_year=to_year,
        depth=depth,
        project_id=project_id,
        resolved_scope=resolved_scope,
    )
    return await get_evolution_graph_service().get_evolution_export(query)


@router.get("/graph/papers/{paper_id}/evolution-entry", response_model=EvolutionEntryResponse)
async def paper_evolution_entry(
    paper_id: str,
    project_id: str = Query(default=""),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """返回论文跳入研究演化页面的默认入口信息。"""
    current_user_id = get_user_id(user)
    await asyncio.to_thread(_ensure_owned_paper, paper_id, current_user_id)
    resolved_scope = await resolve_project_scope(project_id, user_id=current_user_id)
    return await get_evolution_graph_service().get_evolution_entry(
        paper_id,
        project_id=project_id,
        resolved_scope=resolved_scope,
    )


@router.post("/graph/controversy-map", response_model=ControversyMapResponse)
async def controversy_map(
    req: ControversyMapRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """
    返回结构化争议与 gap 地图。

    这类接口通常面向“跨论文综合分析”页面，而不是单篇论文详情。
    """
    current_user_id = get_user_id(user)
    resolved_scope = await resolve_project_scope(req.project_id, user_id=current_user_id)
    owned_paper_ids = (
        await asyncio.to_thread(_ensure_owned_paper_ids, req.paper_ids, current_user_id)
        if req.paper_ids
        else []
    )
    return await get_controversy_map_service().build_map(
        {
            "paper_ids": owned_paper_ids,
            "topic": req.topic.strip(),
            "include_gaps": req.include_gaps,
            "include_conflicts": req.include_conflicts,
            "project_id": (req.project_id or "").strip(),
            "resolved_scope": resolved_scope,
        }
    )


@router.post("/graph/neo4j/rebuild-paper")
async def rebuild_paper_graph(
    req: Neo4jRebuildPaperRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """按 paper_id 重建 Neo4j 子图。"""
    await asyncio.to_thread(_ensure_owned_paper, req.paper_id, get_user_id(user))
    return await get_neo4j_maintenance_service().rebuild_paper_subgraph(req.paper_id)


@router.post("/graph/neo4j/backfill-library")
async def backfill_library_graph(
    req: Neo4jBackfillLibraryRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """按 library 批量回填 Neo4j 子图。"""
    return await get_neo4j_maintenance_service().backfill_library(req.library_id)
