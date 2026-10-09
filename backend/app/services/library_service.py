"""
Library 治理服务。

这是 library 领域的轻业务层，负责 schema、默认库、ID 规范化和存在性校验。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.models.library_models import DEFAULT_LIBRARY_ID, Library
from app.repositories.library_repository import get_library_repository

_schema_ready = False
_schema_lock = asyncio.Lock()


async def ensure_library_schema() -> None:
    """确保 library schema 与默认库已就绪。"""
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return
        repository = get_library_repository()
        await repository.ensure_schema()
        await repository.ensure_default_library()
        _schema_ready = True


def normalize_library_id(library_id: str | None) -> str:
    """统一规范 library_id。"""
    candidate = (library_id or "").strip()
    return candidate or DEFAULT_LIBRARY_ID


async def require_library(library_id: str | None, *, owner: str | None = None) -> Library:
    """确保目标 library 存在。"""
    await ensure_library_schema()
    normalized = normalize_library_id(library_id)
    library = await get_library_repository().get_library(normalized, owner=owner)
    if library is None:
        raise LookupError(f"知识库不存在: {normalized}")
    return library


async def get_library_or_none(library_id: str | None, *, owner: str | None = None) -> Library | None:
    """按 ID 读取知识库。"""
    await ensure_library_schema()
    return await get_library_repository().get_library(normalize_library_id(library_id), owner=owner)


async def list_libraries(*, status: str | None = None, owner: str | None = None) -> list[Library]:
    """列出知识库。"""
    await ensure_library_schema()
    return await get_library_repository().list_libraries(status=status, owner=owner)


async def create_library(payload: dict[str, Any], *, owner: str) -> Library:
    """
    创建归属于认证用户的知识库。

    Args:
        payload: 知识库名称、可见性和描述等业务字段
        owner: 已由认证上下文解析出的用户 ID

    Returns:
        Library: 创建后的知识库记录
    """
    await ensure_library_schema()
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("知识库名称不能为空")
    normalized_owner = str(owner or "").strip()
    if not normalized_owner:
        raise ValueError("知识库 owner 不能为空")
    return await get_library_repository().create_library(payload, owner=normalized_owner)
