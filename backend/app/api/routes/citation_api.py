"""引用相关路由。"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.facades.citation_facade import (
    batch_format_citations_payload,
    format_citation_payload,
    get_citation_tree_payload,
)
from app.api.schemas.citation import CitationBatchRequest, CitationFormatRequest
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id

router = APIRouter()

# 这组接口把“引用树浏览”和“引用文本格式化”拆成两类能力：
#   - GET `/citations/{paper_id}`：偏结构化浏览
#   - POST `/citation/*`：偏文献导出/展示


@router.get("/citations/{paper_id}")
async def citations(
    paper_id: str,
    depth: int = 1,
    include_graph_paths: bool = False,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """读取某篇论文的引用树；可选附带图谱证据路径增强。"""
    try:
        return await get_citation_tree_payload(
            paper_id,
            depth=depth,
            user_id=get_user_id(user),
            include_graph_paths=include_graph_paths,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/citation/format")
async def format_citation(req: CitationFormatRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    """格式化单条引用；至少要能定位到一条论文身份。"""
    try:
        return await format_citation_payload(req.paper_id, req.doi, req.arxiv_id, req.style)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/citation/batch")
async def batch_format_citations(req: CitationBatchRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    """批量格式化引用。"""
    try:
        return await batch_format_citations_payload(req.paper_ids, req.style)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
