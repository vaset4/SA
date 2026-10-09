"""
Library 治理路由。

`library` 是论文/文档的归档容器；与 `project` 不同，它更偏“资料库”而非“研究范围”。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas.library import CreateLibraryRequest
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.services import library_service

router = APIRouter()


@router.get("/libraries")
async def list_libraries(user: dict = Depends(get_current_user_with_rate_limit)):
    """列出当前用户可见知识库。"""
    libraries = await library_service.list_libraries(owner=get_user_id(user))
    return {"libraries": libraries}


@router.post("/libraries")
async def create_library(
    body: CreateLibraryRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """创建最小知识库。"""
    try:
        library = await library_service.create_library(
            body.model_dump(),
            owner=get_user_id(user),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return library


@router.get("/libraries/{library_id}")
async def get_library(library_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """读取单个知识库详情。"""
    library = await library_service.get_library_or_none(library_id, owner=get_user_id(user))
    if library is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return library
