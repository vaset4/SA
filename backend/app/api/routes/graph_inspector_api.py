"""图谱数据诊断查看器接口（只读）。

把两套图的真实内容与质量信号摆给诊断页：
- LightRAG 侧：实体/关系明细 + 类型分布 + 可疑/孤儿/重名信号（读 graphml 权威源）
- Neo4j 研究对象图侧：项目子图 / 单篇子图（复用 research_object_query_service）

全部 GET、只读：不写、不删、不重抽，也不进 planner 主链。
不可用（文件缺失 / Neo4j 关闭）时显式返回 available=false + reason，不静默空白、不抛 500。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.middleware.auth import get_current_user_with_rate_limit
from app.services import lightrag_inspector_service as inspector
from app.services.graph import get_research_object_query_service

router = APIRouter()

# 分页上限，防止一次拉爆大库
_PAGE_MAX = 2000


@router.get("/graph-inspector/lightrag/overview")
async def lightrag_overview(
    library_id: str = Query(default=""),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """LightRAG 图概览：实体/关系总数、按类型分布、可疑/孤儿/重名计数。"""
    return inspector.read_overview(library_id)


@router.get("/graph-inspector/lightrag/entities")
async def lightrag_entities(
    library_id: str = Query(default=""),
    suspect: bool | None = Query(default=None),
    entity_type: str = Query(default=""),
    q: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=_PAGE_MAX),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """LightRAG 实体明细：支持 suspect/type/关键词过滤 + 分页。"""
    return inspector.read_entities(
        library_id,
        suspect=suspect,
        entity_type=entity_type or None,
        q=q or None,
        limit=limit,
        offset=offset,
    )


@router.get("/graph-inspector/lightrag/relations")
async def lightrag_relations(
    library_id: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=_PAGE_MAX),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """LightRAG 关系明细：源/目标/描述/关键词/权重 + 分页。"""
    return inspector.read_relations(library_id, limit=limit, offset=offset)


@router.get("/graph-inspector/lightrag/documents")
async def lightrag_documents(
    library_id: str = Query(default=""),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """LightRAG 文档摄入状态：processed/failed 计数 + 每篇标题/状态/chunk/长度/错误。"""
    return inspector.read_documents(library_id)


@router.get("/graph-inspector/lightrag/chunks")
async def lightrag_chunks(
    library_id: str = Query(default=""),
    full_doc_id: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=_PAGE_MAX),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """LightRAG 文本 chunk 明细：原文内容 + 归属论文 + 顺序 + tokens（by_doc 覆盖全库）。"""
    return inspector.read_chunks(library_id, full_doc_id=full_doc_id or None, limit=limit, offset=offset)


@router.get("/graph-inspector/neo4j/overview")
async def neo4j_overview(
    paper_ids: str = Query(default=""),
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """
    Neo4j 研究对象图概览：复用 query_project_subgraph 返回节点/边及计数。

    Neo4j 不可用时显式返回 available=false + reason（满足 AC6，不静默吞错）。
    """
    ids = [pid.strip() for pid in paper_ids.split(",") if pid.strip()]
    service = get_research_object_query_service()
    try:
        result = await service.query_project_subgraph(ids)
    except Exception as exc:  # 诊断用途：把不可用如实暴露给前端，而非 500
        return {"available": False, "reason": f"Neo4j 研究对象图不可用: {exc}"}
    meta = result.get("meta", {}) if isinstance(result, dict) else {}
    return {
        "available": True,
        "node_count": meta.get("node_count", len(result.get("nodes", []))),
        "edge_count": meta.get("edge_count", len(result.get("edges", []))),
        "paper_count": meta.get("paper_count", len(result.get("papers", []))),
        "nodes": result.get("nodes", []),
        "edges": result.get("edges", []),
        "papers": result.get("papers", []),
    }


@router.get("/graph-inspector/neo4j/paper/{paper_id}")
async def neo4j_paper(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
) -> dict[str, Any]:
    """Neo4j 单篇研究对象子图（复用 get_paper_subgraph）；不可用时显式标注。"""
    service = get_research_object_query_service()
    try:
        result = await service.get_paper_subgraph(paper_id)
    except Exception as exc:  # 同上：不可用如实暴露
        return {"available": False, "reason": f"Neo4j 研究对象图不可用: {exc}"}
    if isinstance(result, dict):
        result.setdefault("available", True)
    return result
