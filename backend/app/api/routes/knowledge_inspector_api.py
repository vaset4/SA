"""Chunk 与向量明细的透明化查询路由，登录用户均可查看。"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.models.knowledge_inspector_contracts import (
    ChunkDetailResponse,
    PaperChunkAuditResponse,
    PaperChunkMapResponse,
)
from app.services import knowledge_inspector_service as inspector
from app.services import paper_db
from app.services.milvus_store import get_milvus_store

router = APIRouter(prefix="/knowledge", tags=["knowledge-inspector"])


@router.get("/chunks/{chunk_id}", response_model=ChunkDetailResponse)
async def get_chunk_detail(
    chunk_id: str,
    include_text: Literal["preview", "full"] = Query(default="preview"),
    max_text_chars: int = Query(default=4000, ge=1, le=20000),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """Inspect one Milvus v2 chunk row by chunk_id."""
    try:
        detail = await asyncio.to_thread(
            inspector.get_chunk_detail,
            chunk_id,
            include_text=include_text,
            max_text_chars=max_text_chars,
            store=get_milvus_store().for_user(get_user_id(user)),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="chunk not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"chunk inspector unavailable: {exc}") from exc
    if not detail.vector_row_found:
        raise HTTPException(status_code=404, detail="chunk not found")
    return detail


@router.get("/papers/{paper_id}/chunk-map", response_model=PaperChunkMapResponse)
async def get_paper_chunk_map(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """Return metadata-grouped chunk rows for one paper."""
    if not await asyncio.to_thread(paper_db.get_paper, paper_id, user_id=get_user_id(user)):
        raise HTTPException(status_code=404, detail="paper not found")
    try:
        return await asyncio.to_thread(inspector.get_paper_chunk_map, paper_id, store=get_milvus_store().for_user(get_user_id(user)))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"chunk map unavailable: {exc}") from exc


@router.get("/papers/{paper_id}/chunk-audit", response_model=PaperChunkAuditResponse)
async def get_paper_chunk_audit(
    paper_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """Run structural chunk/vector audit for one paper."""
    if not await asyncio.to_thread(paper_db.get_paper, paper_id, user_id=get_user_id(user)):
        raise HTTPException(status_code=404, detail="paper not found")
    try:
        return await asyncio.to_thread(inspector.get_paper_chunk_audit, paper_id, store=get_milvus_store().for_user(get_user_id(user)))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"chunk audit unavailable: {exc}") from exc
