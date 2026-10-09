"""Library 仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.models.library_models import (
    DEFAULT_LIBRARY_ID,
    DEFAULT_LIBRARY_NAME,
    DEFAULT_LIBRARY_OWNER,
    DEFAULT_LIBRARY_STATUS,
    DEFAULT_LIBRARY_VISIBILITY,
)
from app.repositories.common import normalize_record
from app.services.database import execute, fetch, fetchrow, fetchval

_repository: "LibraryRepository | None" = None

# library 是项目里“知识库/资料库”这一层作用域实体。
# 它和 project 的关系是：
#   - project 更偏用户工作空间
#   - library 更偏论文/知识内容归属容器
# 很多 service 会先解析 project，再落到一个或多个 library。

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS libraries (
    library_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    owner TEXT NOT NULL DEFAULT 'system',
    visibility TEXT NOT NULL DEFAULT 'private',
    status TEXT NOT NULL DEFAULT 'active',
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_libraries_owner ON libraries(owner);
CREATE INDEX IF NOT EXISTS idx_libraries_status ON libraries(status);
"""


def _library_data(row: Any) -> dict[str, Any] | None:
    return normalize_record(row)


class LibraryRepository:
    """封装 knowledge library 元数据读写。"""

    async def ensure_schema(self) -> None:
        """初始化 libraries 表。"""
        await execute(SCHEMA_SQL)

    async def ensure_default_library(self) -> dict[str, Any]:
        """
        确保默认 library 存在。

        这是历史兼容关键点：老数据没有显式分库时，会统一落到 default library。
        """
        await execute(
            """
            INSERT INTO libraries (library_id, name, owner, visibility, status, description)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (library_id) DO UPDATE
            SET updated_at = NOW()
            """,
            DEFAULT_LIBRARY_ID,
            DEFAULT_LIBRARY_NAME,
            DEFAULT_LIBRARY_OWNER,
            DEFAULT_LIBRARY_VISIBILITY,
            DEFAULT_LIBRARY_STATUS,
            "系统默认知识库，用于承接未显式分库的历史数据。",
        )
        row = await fetchrow("SELECT * FROM libraries WHERE library_id=$1", DEFAULT_LIBRARY_ID)
        return _library_data(row) or {}

    async def create_library(self, payload: dict[str, Any], *, owner: str) -> dict[str, Any]:
        """
        创建显式 library，并使用服务端传入的 owner 写入归属。

        Args:
            payload: 知识库持久化字段
            owner: 已完成认证的用户 ID

        Returns:
            dict[str, Any]: 新建知识库记录
        """
        library_id = (payload.get("library_id") or f"lib_{uuid.uuid4().hex[:8]}").strip()
        await execute(
            """
            INSERT INTO libraries (library_id, name, owner, visibility, status, description)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            library_id,
            payload["name"].strip(),
            owner.strip(),
            payload.get("visibility") or DEFAULT_LIBRARY_VISIBILITY,
            payload.get("status") or DEFAULT_LIBRARY_STATUS,
            (payload.get("description") or "").strip(),
        )
        row = await fetchrow("SELECT * FROM libraries WHERE library_id=$1", library_id)
        return _library_data(row) or {}

    async def get_library(self, library_id: str, owner: str | None = None) -> dict[str, Any] | None:
        """读取单个 library；带 owner 时允许访问系统默认库。"""
        if owner:
            row = await fetchrow(
                "SELECT * FROM libraries WHERE library_id=$1 AND (owner=$2 OR owner=$3)",
                library_id,
                owner,
                DEFAULT_LIBRARY_OWNER,
            )
        else:
            row = await fetchrow("SELECT * FROM libraries WHERE library_id=$1", library_id)
        return _library_data(row)

    async def list_libraries(self, *, status: str | None = None, owner: str | None = None) -> list[dict[str, Any]]:
        """列出 library；owner 过滤同样会把系统默认库一并暴露。"""
        if status and owner:
            rows = await fetch(
                """
                SELECT * FROM libraries
                WHERE status=$1 AND (owner=$2 OR owner=$3)
                ORDER BY created_at ASC
                """,
                status,
                owner,
                DEFAULT_LIBRARY_OWNER,
            )
        elif status:
            rows = await fetch(
                "SELECT * FROM libraries WHERE status=$1 ORDER BY created_at ASC",
                status,
            )
        elif owner:
            rows = await fetch(
                """
                SELECT * FROM libraries
                WHERE owner=$1 OR owner=$2
                ORDER BY created_at ASC
                """,
                owner,
                DEFAULT_LIBRARY_OWNER,
            )
        else:
            rows = await fetch("SELECT * FROM libraries ORDER BY created_at ASC")
        return [_library_data(row) or {} for row in rows]

    async def exists(self, library_id: str, owner: str | None = None) -> bool:
        """判断 library 是否存在；带 owner 时同样兼容默认库共享语义。"""
        if owner:
            return bool(
                await fetchval(
                    """
                    SELECT 1 FROM libraries
                    WHERE library_id=$1 AND (owner=$2 OR owner=$3)
                    LIMIT 1
                    """,
                    library_id,
                    owner,
                    DEFAULT_LIBRARY_OWNER,
                )
            )
        return bool(await fetchval("SELECT 1 FROM libraries WHERE library_id=$1 LIMIT 1", library_id))


def get_library_repository() -> LibraryRepository:
    """返回 library 仓储单例。"""
    global _repository
    if _repository is None:
        _repository = LibraryRepository()
    return _repository
