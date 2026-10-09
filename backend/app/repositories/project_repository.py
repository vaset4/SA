"""
Project 仓储。

这一层把项目研究空间相关的数据库读写全部收口到一起，包括：
- projects 主表
- project_scopes 作用域表
- project_paper_links / project_session_links 关联表

从代码阅读角度看，它体现了一个重要设计：
“项目元信息、项目范围、项目关联对象”被拆成多张表，
这样每种信息都能独立演化，而不会把所有字段硬塞进一个大表。
"""

from __future__ import annotations

import uuid
from typing import Any

from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.project_models import (
    DEFAULT_PROJECT_OWNER,
    DEFAULT_PROJECT_PAPER_ROLE,
    DEFAULT_PROJECT_STATUS,
    empty_scope,
)
from app.repositories.common import dump_json, load_json, normalize_record
from app.services.database import execute, fetch, fetchrow, get_pool

_repository: "ProjectRepository | None" = None

SCHEMA_SQL = """
-- projects: 项目主表，保存项目基础元信息与归属用户
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    owner TEXT NOT NULL DEFAULT 'system',
    status TEXT NOT NULL DEFAULT 'active',
    default_library_id TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE IF EXISTS projects ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'anonymous';
UPDATE projects SET user_id = COALESCE(NULLIF(owner, ''), 'anonymous') WHERE user_id IS NULL OR user_id = '';
CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);

-- project_scopes: 检索和筛选相关的作用域定义，和主表拆开存储
CREATE TABLE IF NOT EXISTS project_scopes (
    project_id TEXT PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
    library_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    paper_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    excluded_paper_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    year_from INTEGER,
    year_to INTEGER,
    task_filters JSONB NOT NULL DEFAULT '[]'::jsonb,
    method_filters JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS project_paper_links (
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'core',
    note TEXT NOT NULL DEFAULT '',
    pinned_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (project_id, paper_id)
);

CREATE TABLE IF NOT EXISTS project_session_links (
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (project_id, session_id)
);
"""


def _project_data(row: Any) -> dict[str, Any] | None:
    return normalize_record(row)


def _scope_data(row: Any) -> dict[str, Any] | None:
    data = normalize_record(row)
    if data is None:
        return None
    data["library_ids"] = load_json(data.get("library_ids"), [])
    data["paper_ids"] = load_json(data.get("paper_ids"), [])
    data["excluded_paper_ids"] = load_json(data.get("excluded_paper_ids"), [])
    data["task_filters"] = load_json(data.get("task_filters"), [])
    data["method_filters"] = load_json(data.get("method_filters"), [])
    return data


class ProjectRepository:
    async def ensure_schema(self) -> None:
        # 仓储层负责保证 project 功能运行所需的表结构存在。
        await execute(SCHEMA_SQL)

    async def create_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        # project_id 支持外部传入；否则自动生成短 ID，便于调试和前端展示。
        project_id = (payload.get("project_id") or f"proj_{uuid.uuid4().hex[:8]}").strip()
        await execute(
            """
            INSERT INTO projects (project_id, user_id, name, description, owner, status, default_library_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            project_id,
            (payload.get("user_id") or DEFAULT_PROJECT_OWNER).strip(),
            payload["name"].strip(),
            (payload.get("description") or "").strip(),
            (payload.get("owner") or DEFAULT_PROJECT_OWNER).strip(),
            payload.get("status") or DEFAULT_PROJECT_STATUS,
            (payload.get("default_library_id") or DEFAULT_LIBRARY_ID).strip() or DEFAULT_LIBRARY_ID,
        )
        return await self.get_project(project_id, user_id=payload.get("user_id")) or {}

    async def get_project(self, project_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        if user_id:
            row = await fetchrow("SELECT * FROM projects WHERE user_id=$1 AND project_id=$2", user_id, project_id)
        else:
            row = await fetchrow("SELECT * FROM projects WHERE project_id=$1", project_id)
        return _project_data(row)

    async def list_projects(self, *, status: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        if status and user_id:
            rows = await fetch(
                "SELECT * FROM projects WHERE user_id=$1 AND status=$2 ORDER BY created_at DESC",
                user_id,
                status,
            )
        elif status:
            rows = await fetch("SELECT * FROM projects WHERE status=$1 ORDER BY created_at DESC", status)
        elif user_id:
            rows = await fetch("SELECT * FROM projects WHERE user_id=$1 ORDER BY created_at DESC", user_id)
        else:
            rows = await fetch("SELECT * FROM projects ORDER BY created_at DESC")
        return [_project_data(row) or {} for row in rows]

    async def update_project(self, project_id: str, payload: dict[str, Any], *, user_id: str | None = None) -> dict[str, Any] | None:
        current = await self.get_project(project_id, user_id=user_id)
        if current is None:
            return None
        from app.repositories.memory_source_repository import invalidate_source, invalidate_project_scope

        pool = await get_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.fetch("""SELECT p.paper_id FROM papers p WHERE EXISTS(
                SELECT 1 FROM memory_evidence_refs r JOIN research_memory_items i ON i.memory_id=r.memory_id
                WHERE i.project_id=$1 AND r.paper_id=p.paper_id) ORDER BY p.paper_id FOR SHARE""", project_id)
            await connection.fetchrow("SELECT project_id FROM projects WHERE project_id=$1 FOR UPDATE", project_id)
            await connection.execute(
                """
                UPDATE projects
                SET user_id=$2, name=$3, description=$4, owner=$5, status=$6, default_library_id=$7, updated_at=NOW()
                WHERE project_id=$1 AND user_id=$8
                """,
                project_id,
                payload.get("user_id", current.get("user_id", DEFAULT_PROJECT_OWNER)),
                payload.get("name", current["name"]),
                payload.get("description", current.get("description", "")),
                payload.get("owner", current.get("owner", DEFAULT_PROJECT_OWNER)),
                payload.get("status", current.get("status", DEFAULT_PROJECT_STATUS)),
                payload.get("default_library_id", current.get("default_library_id", DEFAULT_LIBRARY_ID)),
                user_id or current.get("user_id", DEFAULT_PROJECT_OWNER),
            )
            if payload.get("user_id", current["user_id"]) != current["user_id"]:
                await invalidate_source(connection, user_id=current["user_id"], project_id=project_id,
                                        paper_id=None, reason="项目归属已变化，需要重新核对来源")
            elif payload.get("default_library_id", current["default_library_id"]) != current["default_library_id"]:
                await invalidate_project_scope(connection, project_id=project_id, user_id=current["user_id"])

        return await self.get_project(project_id, user_id=user_id)

    async def delete_project(self, project_id: str, *, user_id: str | None = None) -> bool:
        if user_id:
            row = await fetchrow(
                "DELETE FROM projects WHERE project_id=$1 AND user_id=$2 RETURNING project_id",
                project_id,
                user_id,
            )
        else:
            row = await fetchrow("DELETE FROM projects WHERE project_id=$1 RETURNING project_id", project_id)
        return bool(row)

    async def get_scope(self, project_id: str) -> dict[str, Any] | None:
        return _scope_data(await fetchrow("SELECT * FROM project_scopes WHERE project_id=$1", project_id))

    async def upsert_scope(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        插入或更新项目 scope。

        `scope` 之所以单独 upsert，是因为它比项目主信息变化更频繁，
        且天然适合 JSONB + 标量字段混合存储。
        """
        from app.repositories.memory_source_repository import invalidate_project_scope

        pool = await get_pool()
        async with pool.acquire() as connection, connection.transaction():
            # 1. 【研究记忆】【范围变化】先锁依赖论文，再锁项目，沿用发布与确认的锁顺序
            await connection.fetch("""SELECT p.paper_id FROM papers p WHERE EXISTS(
                SELECT 1 FROM memory_evidence_refs r JOIN research_memory_items i ON i.memory_id=r.memory_id
                WHERE i.project_id=$1 AND r.paper_id=p.paper_id) ORDER BY p.paper_id FOR SHARE""", project_id)
            project = await connection.fetchrow("SELECT * FROM projects WHERE project_id=$1 FOR UPDATE", project_id)
            if project is None:
                raise LookupError("项目不存在")
            defaults = empty_scope()
            await connection.execute(
                """
                INSERT INTO project_scopes (
                    project_id, library_ids, paper_ids, excluded_paper_ids,
                    year_from, year_to, task_filters, method_filters
                )
                VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb, $5, $6, $7::jsonb, $8::jsonb)
                ON CONFLICT (project_id) DO UPDATE SET
                    library_ids=EXCLUDED.library_ids,
                    paper_ids=EXCLUDED.paper_ids,
                    excluded_paper_ids=EXCLUDED.excluded_paper_ids,
                    year_from=EXCLUDED.year_from,
                    year_to=EXCLUDED.year_to,
                    task_filters=EXCLUDED.task_filters,
                    method_filters=EXCLUDED.method_filters,
                    updated_at=NOW()
                """,
                project_id,
                dump_json(payload.get("library_ids", defaults["library_ids"])),
                dump_json(payload.get("paper_ids", defaults["paper_ids"])),
                dump_json(payload.get("excluded_paper_ids", defaults["excluded_paper_ids"])),
                payload.get("year_from"),
                payload.get("year_to"),
                dump_json(payload.get("task_filters", defaults["task_filters"])),
                dump_json(payload.get("method_filters", defaults["method_filters"])),
            )
            await invalidate_project_scope(connection, project_id=project_id, user_id=project["user_id"])
        return await self.get_scope(project_id) or {}

    async def link_paper(self, project_id: str, paper_id: str, *, role: str = DEFAULT_PROJECT_PAPER_ROLE, note: str = "") -> dict[str, Any]:
        # ON CONFLICT 让“重复添加同一篇论文”退化成更新角色/备注，而不是报错。
        await execute(
            """
            INSERT INTO project_paper_links (project_id, paper_id, role, note)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (project_id, paper_id) DO UPDATE
            SET role=EXCLUDED.role, note=EXCLUDED.note, pinned_at=NOW()
            """,
            project_id,
            paper_id,
            role,
            note,
        )
        return normalize_record(
            await fetchrow(
                "SELECT * FROM project_paper_links WHERE project_id=$1 AND paper_id=$2",
                project_id,
                paper_id,
            )
        ) or {}

    async def unlink_paper(self, project_id: str, paper_id: str) -> None:
        """参数为项目和论文；取消关联与依赖记忆失效在同一事务完成。"""
        from app.repositories.memory_source_repository import invalidate_source

        pool = await get_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.fetchrow("SELECT paper_id FROM papers WHERE paper_id=$1 FOR SHARE", paper_id)
            project = await connection.fetchrow("SELECT user_id FROM projects WHERE project_id=$1 FOR UPDATE", project_id)
            if project is None:
                raise LookupError("项目不存在")
            await connection.execute("DELETE FROM project_paper_links WHERE project_id=$1 AND paper_id=$2", project_id, paper_id)
            await invalidate_source(connection, user_id=project["user_id"], project_id=project_id, paper_id=paper_id,
                                    reason=f"支持资料已移出项目: {paper_id}")

    async def list_project_papers(self, project_id: str) -> list[dict[str, Any]]:
        # 查询结果把论文基础信息和项目内角色/备注一起拼回，便于前端直接展示。
        rows = await fetch(
            """
            SELECT p.*, l.role AS project_role, l.note AS project_note, l.pinned_at
            FROM project_paper_links l
            JOIN papers p ON p.paper_id = l.paper_id
            WHERE l.project_id=$1
            ORDER BY l.pinned_at DESC
            """,
            project_id,
        )
        return [normalize_record(row) or {} for row in rows]

    async def link_session(self, project_id: str, session_id: str) -> dict[str, Any]:
        # session 关联采用幂等写法，重复挂载不会报错。
        await execute(
            """
            INSERT INTO project_session_links (project_id, session_id)
            VALUES ($1, $2)
            ON CONFLICT (project_id, session_id) DO NOTHING
            """,
            project_id,
            session_id,
        )
        return normalize_record(
            await fetchrow(
                "SELECT * FROM project_session_links WHERE project_id=$1 AND session_id=$2",
                project_id,
                session_id,
            )
        ) or {}

    async def list_project_sessions(self, project_id: str) -> list[dict[str, Any]]:
        rows = await fetch(
            """
            SELECT l.project_id, l.session_id, l.created_at, s.topic, s.updated_at, s.paper_ids
            FROM project_session_links l
            JOIN sessions s ON s.session_id = l.session_id
            WHERE l.project_id=$1
            ORDER BY l.created_at DESC
            """,
            project_id,
        )
        return [normalize_record(row) or {} for row in rows]

    async def list_project_ids_by_paper(self, paper_id: str) -> list[str]:
        rows = await fetch(
            """
            SELECT project_id
            FROM project_paper_links
            WHERE paper_id=$1
            ORDER BY pinned_at DESC
            """,
            paper_id,
        )
        return [str(row["project_id"]) for row in rows if row.get("project_id")]


def get_project_repository() -> ProjectRepository:
    """返回 project 仓储单例。"""
    global _repository
    if _repository is None:
        _repository = ProjectRepository()
    return _repository
