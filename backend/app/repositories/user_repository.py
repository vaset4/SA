"""用户仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.repositories.common import normalize_record
from app.services.database import execute, fetchrow

_repository: "UserRepository | None" = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    token_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user';
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
"""


class UserRepository:
    """封装用户名密码用户的读写。"""

    async def ensure_schema(self) -> None:
        """初始化 users 表；认证链在 register/login 前都会先确保这一层可用。"""
        await execute(SCHEMA_SQL)

    async def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "user",
    ) -> dict[str, Any]:
        """创建用户并返回认证签发需要的账号状态。

        Args:
            username: 已完成业务校验的用户名。
            password_hash: 已生成的密码摘要。
            role: 初始数据库角色。

        Returns:
            包含账号身份、状态和令牌版本的新用户记录。
        """
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        row = await fetchrow(
            """
            INSERT INTO users (user_id, username, password_hash, role)
            VALUES ($1, $2, $3, $4)
            RETURNING user_id, username, role, is_active, token_version, created_at
            """,
            user_id,
            username,
            password_hash,
            role,
        )
        return normalize_record(row) or {}

    async def get_by_username(self, username: str) -> dict[str, Any] | None:
        """按用户名查完整记录；登录流程需要取出 password_hash 做校验。"""
        row = await fetchrow("SELECT * FROM users WHERE username=$1", username)
        return normalize_record(row)

    async def get_by_user_id(self, user_id: str) -> dict[str, Any] | None:
        """按用户标识查询刷新校验所需的公开账号字段。

        Args:
            user_id: 需要查询的用户标识。

        Returns:
            不含密码摘要的账号记录，账号不存在时返回 None。
        """
        row = await fetchrow(
            """
            SELECT user_id, username, role, is_active, token_version, created_at
            FROM users
            WHERE user_id=$1
            """,
            user_id,
        )
        return normalize_record(row)

    async def revoke_all_tokens(self, user_id: str) -> int | None:
        """递增账号令牌版本并返回新版本。

        Args:
            user_id: 需要吊销全部 refresh token 的用户标识。

        Returns:
            吊销后的令牌版本，账号不存在时返回 None。
        """
        row = await fetchrow(
            """
            UPDATE users
            SET token_version = token_version + 1
            WHERE user_id=$1
            RETURNING token_version
            """,
            user_id,
        )
        record = normalize_record(row)
        return int(record["token_version"]) if record is not None else None


def get_user_repository() -> UserRepository:
    """返回用户仓储单例。"""
    global _repository
    if _repository is None:
        _repository = UserRepository()
    return _repository
