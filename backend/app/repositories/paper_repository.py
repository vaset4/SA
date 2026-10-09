"""论文主数据仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.models.library_models import DEFAULT_LIBRARY_ID
from app.repositories.common import dump_json, load_json, normalize_record, utc_now
from app.repositories.library_repository import get_library_repository
from app.repositories.upload_schema import UPLOAD_SCHEMA_SQL
from app.services.database import execute, fetch, fetchrow, fetchval

_repository: "PaperRepository | None" = None
_UPSERT_PAPER_SQL = """
INSERT INTO papers (
    user_id, library_id, paper_id, arxiv_id, doi, title, authors, abstract, pdf_url, pdf_path,
    categories, published_date, source, analysis, analysis_status, analysis_error, analysis_updated_at, tags
) VALUES (
    $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11::jsonb, $12, $13, $14::jsonb, $15, $16, $17, $18::jsonb
)
ON CONFLICT (paper_id) DO UPDATE SET
    pdf_url=COALESCE(NULLIF(EXCLUDED.pdf_url, ''), papers.pdf_url),
    pdf_path=COALESCE(EXCLUDED.pdf_path, papers.pdf_path),
    analysis=COALESCE(EXCLUDED.analysis, papers.analysis),
    last_accessed=$19,
    library_id=COALESCE(EXCLUDED.library_id, papers.library_id),
    analysis_status=COALESCE(EXCLUDED.analysis_status, papers.analysis_status),
    analysis_error=COALESCE(NULLIF(EXCLUDED.analysis_error, ''), papers.analysis_error),
    analysis_updated_at=COALESCE(EXCLUDED.analysis_updated_at, papers.analysis_updated_at)
WHERE papers.user_id=EXCLUDED.user_id
RETURNING paper_id
"""
# 这里的 SCHEMA_SQL 不只创建 papers，还顺带兼容了项目历史遗留的 notes / sessions /
# logs / rag_traces / task_runs 等表。也就是说，这个仓储文件实际上承担了一部分
# “历史 schema 兜底入口”的角色。
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    library_id TEXT NOT NULL DEFAULT 'default',
    paper_id TEXT PRIMARY KEY,
    arxiv_id TEXT,
    doi TEXT,
    title TEXT NOT NULL,
    authors JSONB NOT NULL DEFAULT '[]'::jsonb,
    abstract TEXT NOT NULL,
    pdf_url TEXT,
    pdf_path TEXT,
    categories JSONB NOT NULL DEFAULT '[]'::jsonb,
    published_date TEXT,
    source TEXT DEFAULT 'arxiv',
    analysis JSONB,
    analysis_status TEXT NOT NULL DEFAULT 'not_started',
    analysis_error TEXT NOT NULL DEFAULT '',
    analysis_updated_at TIMESTAMPTZ,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    deletion_status TEXT NOT NULL DEFAULT 'active',
    deletion_task_id TEXT,
    deletion_error TEXT NOT NULL DEFAULT '',
    deletion_failed_step TEXT,
    deletion_requested_at TIMESTAMPTZ,
    is_vectorized BOOLEAN DEFAULT FALSE,
    read_count INTEGER DEFAULT 0,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ
);
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'anonymous';
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS library_id TEXT DEFAULT 'default';
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS analysis_status TEXT DEFAULT 'not_started';
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS analysis_error TEXT DEFAULT '';
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS analysis_updated_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS deletion_status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS deletion_task_id TEXT;
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS deletion_error TEXT NOT NULL DEFAULT '';
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS deletion_failed_step TEXT;
ALTER TABLE IF EXISTS papers ADD COLUMN IF NOT EXISTS deletion_requested_at TIMESTAMPTZ;
UPDATE papers SET user_id = 'anonymous' WHERE user_id IS NULL OR user_id = '';
UPDATE papers SET library_id = 'default' WHERE library_id IS NULL OR library_id = '';
UPDATE papers SET analysis_status = 'analysis_ready'
WHERE (analysis_status IS NULL OR analysis_status = '') AND analysis IS NOT NULL;
UPDATE papers SET analysis_status = 'not_started'
WHERE (analysis_status IS NULL OR analysis_status = '') AND analysis IS NULL;
UPDATE papers SET analysis_error = '' WHERE analysis_error IS NULL;
UPDATE papers SET deletion_status = 'active' WHERE deletion_status IS NULL OR deletion_status = '';
UPDATE papers SET deletion_error = '' WHERE deletion_error IS NULL;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'papers_deletion_status_check'
    ) THEN
        ALTER TABLE papers ADD CONSTRAINT papers_deletion_status_check
            CHECK (deletion_status IN ('active', 'deleting', 'delete_failed'));
    END IF;
END $$;
ALTER TABLE IF EXISTS papers DROP CONSTRAINT IF EXISTS papers_arxiv_id_key;
CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    paper_id TEXT REFERENCES papers(paper_id) ON DELETE CASCADE,
    project_id TEXT,
    conversation_id TEXT,
    session_id TEXT,
    note_type TEXT NOT NULL DEFAULT 'reading_note',
    source_type TEXT NOT NULL DEFAULT 'manual',
    citation TEXT NOT NULL DEFAULT '',
    excerpt TEXT NOT NULL DEFAULT '',
    evidence_reference JSONB NOT NULL DEFAULT '{}'::jsonb,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'anonymous';
ALTER TABLE IF EXISTS notes ALTER COLUMN paper_id DROP NOT NULL;
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS project_id TEXT;
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS conversation_id TEXT;
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS note_type TEXT DEFAULT 'reading_note';
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'manual';
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS citation TEXT DEFAULT '';
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS excerpt TEXT DEFAULT '';
ALTER TABLE IF EXISTS notes ADD COLUMN IF NOT EXISTS evidence_reference JSONB DEFAULT '{}'::jsonb;
UPDATE notes SET user_id = 'anonymous' WHERE user_id IS NULL OR user_id = '';
UPDATE notes SET note_type = 'reading_note' WHERE note_type IS NULL OR note_type = '';
UPDATE notes SET source_type = 'manual' WHERE source_type IS NULL OR source_type = '';
UPDATE notes SET citation = '' WHERE citation IS NULL;
UPDATE notes SET excerpt = '' WHERE excerpt IS NULL;
UPDATE notes SET evidence_reference = '{}'::jsonb WHERE evidence_reference IS NULL;
CREATE INDEX IF NOT EXISTS idx_notes_user_created_at ON notes(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_paper_id ON notes(paper_id);
CREATE INDEX IF NOT EXISTS idx_notes_project_id ON notes(project_id);
CREATE INDEX IF NOT EXISTS idx_notes_conversation_id ON notes(conversation_id);
CREATE INDEX IF NOT EXISTS idx_notes_session_id ON notes(session_id);
CREATE INDEX IF NOT EXISTS idx_notes_type_source ON notes(note_type, source_type);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    topic TEXT,
    conversation_id TEXT,
    project_id TEXT,
    thread_id TEXT,
    research_mode TEXT,
    status TEXT NOT NULL DEFAULT 'created',
    last_error TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    paper_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    chat_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    state_json TEXT,
    step_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'anonymous';
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS conversation_id TEXT;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS project_id TEXT;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS thread_id TEXT;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS research_mode TEXT;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS last_error TEXT;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS attempt_count INTEGER DEFAULT 0;
ALTER TABLE IF EXISTS sessions ALTER COLUMN status SET DEFAULT 'created';
ALTER TABLE IF EXISTS sessions ALTER COLUMN attempt_count SET DEFAULT 0;
UPDATE sessions SET user_id = 'anonymous' WHERE user_id IS NULL OR user_id = '';
UPDATE sessions SET attempt_count = 0 WHERE attempt_count IS NULL;
CREATE TABLE IF NOT EXISTS logs (
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    event_type TEXT,
    node_name TEXT,
    query TEXT,
    result_count INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_read BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS rag_traces (
    request_id TEXT PRIMARY KEY,
    session_id TEXT,
    query TEXT,
    trace_data JSONB,
    user_id TEXT NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS task_runs (
    task_id TEXT PRIMARY KEY,
    user_id TEXT,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    worker_id TEXT,
    lease_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS task_events (
    event_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task_runs(task_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE IF EXISTS task_runs ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE IF EXISTS logs ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_task_runs_user_created ON task_runs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_runs_user_status_created ON task_runs(user_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_user_created ON logs(user_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_user_arxiv ON papers(user_id, arxiv_id) WHERE arxiv_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_user_doi ON papers(user_id, doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_user_library_added ON papers(user_id, library_id, added_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_user_paper ON notes(user_id, paper_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_user_status_updated ON sessions(user_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_thread_id ON sessions(thread_id) WHERE thread_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON rag_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_created ON rag_traces(created_at);
CREATE INDEX IF NOT EXISTS idx_task_runs_status_created ON task_runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_task_runs_type_created ON task_runs(task_type, created_at);
CREATE INDEX IF NOT EXISTS idx_task_events_task_created ON task_events(task_id, created_at);
""" + UPLOAD_SCHEMA_SQL


def _paper_data(row: Any) -> dict[str, Any] | None:
    """把数据库记录规范化为上层 service 可直接消费的 paper 字典。"""
    data = normalize_record(row)
    if data is None:
        return None
    data["library_id"] = data.get("library_id") or DEFAULT_LIBRARY_ID
    for key, default in (("authors", []), ("categories", []), ("analysis", None), ("tags", []), ("metadata_missing", [])):
        data[key] = load_json(data.get(key), default)
    data["analysis_status"] = str(data.get("analysis_status") or ("analysis_ready" if data.get("analysis") else "not_started")).strip()
    data["analysis_error"] = str(data.get("analysis_error") or "").strip()
    data["deletion_status"] = str(data.get("deletion_status") or "active").strip()
    data["deletion_error"] = str(data.get("deletion_error") or "").strip()
    return data


def _membership(column: str, values: list[str] | None, index: int) -> tuple[list[str], list[str], int]:
    """构建 JSONB 数组包含条件，供 tags/categories 过滤复用。"""
    clauses: list[str] = []
    params: list[str] = []
    for value in values or []:
        clauses.append(f"{column} @> ${index}::jsonb")
        params.append(dump_json([value]))
        index += 1
    return clauses, params, index


def _paper_filters(
    user_id: str | None,
    tags: list[str] | None,
    categories: list[str] | None,
    library_id: str | None,
) -> tuple[str, list[str], int]:
    """
    统一构建论文查询 where 条件。

    这里把 user/library/tags/categories 的组合过滤集中起来，避免 count/list 两套 SQL
    分别维护同样的条件拼装逻辑。
    """
    clauses: list[str] = ["deletion_status = 'active'"]
    params: list[str] = []
    index = 1
    if user_id:
        clauses.append(f"user_id = ${index}")
        params.append(user_id)
        index += 1
    if library_id:
        clauses.append(f"library_id = ${index}")
        params.append(library_id)
        index += 1
    for column, values in (("tags", tags), ("categories", categories)):
        current, extra, index = _membership(column, values, index)
        if current:
            clauses.append(f"({' OR '.join(current)})")
            params.extend(extra)
    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, params, index


class PaperRepository:
    """封装论文表与基础 schema 初始化。"""

    async def ensure_schema(self) -> None:
        """
        初始化论文及其历史兼容相关表。

        这里先确保 library schema 存在，是因为 papers 现在已经带上 `library_id` 作用域。
        """
        await get_library_repository().ensure_schema()
        await get_library_repository().ensure_default_library()
        await execute(SCHEMA_SQL)
        await self._migrate_rag_traces_user_id()

    async def _migrate_rag_traces_user_id(self) -> None:
        """为已存在的 rag_traces 表补齐 user_id 列（幂等）。"""
        has_column = await fetchval(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.columns "
            "  WHERE table_name = 'rag_traces' AND column_name = 'user_id'"
            ")"
        )
        if not has_column:
            await execute("ALTER TABLE rag_traces ADD COLUMN user_id TEXT NOT NULL DEFAULT 'unknown'")
            await execute("CREATE INDEX IF NOT EXISTS idx_traces_user_created ON rag_traces(user_id, created_at DESC)")

    async def add_or_update_paper(self, paper: dict[str, Any]) -> str:
        """新增或更新论文；显式 paper_id 原子 upsert，未显式传入时再按 arXiv ID 或 DOI 复用。
        Args:
            paper: 待保存的论文元数据，显式 paper_id 表示可重试的固定业务身份
        Returns:
            实际写入或更新的 paper_id
        """
        user_id = str(paper.get("user_id") or "anonymous").strip() or "anonymous"
        requested_paper_id = str(paper.get("paper_id") or "").strip()
        existing_id = None
        if not requested_paper_id and paper.get("arxiv_id"):
            existing_id = await fetchval(
                "SELECT paper_id FROM papers WHERE user_id=$1 AND arxiv_id=$2",
                user_id,
                paper["arxiv_id"],
            )
        if not requested_paper_id and existing_id is None and paper.get("doi"):
            existing_id = await fetchval(
                "SELECT paper_id FROM papers WHERE user_id=$1 AND doi=$2",
                user_id,
                paper["doi"],
            )
        analysis = dump_json(paper["analysis"]) if paper.get("analysis") else None
        analysis_status = str(paper.get("analysis_status") or ("analysis_ready" if paper.get("analysis") else "not_started")).strip()
        analysis_error = str(paper.get("analysis_error") or "").strip()
        analysis_updated_at = utc_now() if analysis_status in {"analysis_ready", "failed", "running", "queued"} else None
        paper_id = requested_paper_id or existing_id or str(uuid.uuid4())
        saved_id = await fetchval(
            _UPSERT_PAPER_SQL,
            user_id,
            paper.get("library_id") or DEFAULT_LIBRARY_ID,
            paper_id,
            paper.get("arxiv_id"),
            paper.get("doi"),
            paper["title"],
            dump_json(paper.get("authors", [])),
            paper["abstract"],
            paper.get("pdf_url", ""),
            paper.get("pdf_path"),
            dump_json(paper.get("categories", [])),
            paper.get("published_date", ""),
            paper.get("source", "arxiv"),
            analysis,
            analysis_status,
            analysis_error,
            analysis_updated_at,
            dump_json(paper.get("tags", [])),
            utc_now(),
        )
        if saved_id is None:
            raise ValueError(f"paper_id 已属于其他用户，拒绝覆盖: {paper_id}")
        return str(saved_id)

    async def get_paper(self, paper_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        if user_id:
            row = await fetchrow(
                "SELECT * FROM papers WHERE user_id=$1 AND paper_id=$2 AND deletion_status='active'",
                user_id,
                paper_id,
            )
        else:
            row = await fetchrow("SELECT * FROM papers WHERE paper_id=$1 AND deletion_status='active'", paper_id)
        return _paper_data(row)

    async def get_papers_by_ids(
        self,
        paper_ids: list[str],
        *,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """一次读取当前用户拥有的 active 论文，并保持请求 ID 顺序。"""
        if not paper_ids:
            return []
        rows = await fetch(
            """
            SELECT * FROM papers
            WHERE user_id=$1
              AND paper_id=ANY($2::text[])
              AND deletion_status='active'
            ORDER BY array_position($2::text[], paper_id)
            """,
            user_id,
            paper_ids,
        )
        return [data for row in rows if (data := _paper_data(row)) is not None]

    async def get_by_arxiv_id(self, arxiv_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        if user_id:
            row = await fetchrow(
                "SELECT * FROM papers WHERE user_id=$1 AND arxiv_id=$2 AND deletion_status='active'",
                user_id,
                arxiv_id,
            )
        else:
            row = await fetchrow("SELECT * FROM papers WHERE arxiv_id=$1 AND deletion_status='active'", arxiv_id)
        return _paper_data(row)

    async def search_by_exact_title(
        self,
        title: str,
        library_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        按 title 做大小写不敏感的精确匹配。

        live ragas 评估需要把数据集里的 paper_title 稳定解析成 paper_id，这里故意不用
        ILIKE 模糊搜，避免把作用域错误地落到相似标题上。
        """
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return []

        if library_id and user_id:
            rows = await fetch(
                """
                SELECT * FROM papers
                WHERE user_id=$1 AND library_id=$2 AND deletion_status='active'
                  AND LOWER(BTRIM(title)) = LOWER(BTRIM($3))
                ORDER BY added_at DESC
                LIMIT 10
                """,
                user_id,
                library_id,
                normalized_title,
            )
        elif library_id:
            rows = await fetch(
                """
                SELECT * FROM papers
                WHERE library_id=$1 AND deletion_status='active'
                  AND LOWER(BTRIM(title)) = LOWER(BTRIM($2))
                ORDER BY added_at DESC
                LIMIT 10
                """,
                library_id,
                normalized_title,
            )
        elif user_id:
            rows = await fetch(
                """
                SELECT * FROM papers
                WHERE user_id=$1 AND deletion_status='active'
                  AND LOWER(BTRIM(title)) = LOWER(BTRIM($2))
                ORDER BY added_at DESC
                LIMIT 10
                """,
                user_id,
                normalized_title,
            )
        else:
            rows = await fetch(
                """
                SELECT * FROM papers
                WHERE deletion_status='active' AND LOWER(BTRIM(title)) = LOWER(BTRIM($1))
                ORDER BY added_at DESC
                LIMIT 10
                """,
                normalized_title,
            )
        return [_paper_data(row) or {} for row in rows]

    async def exists(self, arxiv_id: str | None, doi: str | None, *, user_id: str | None = None) -> bool:
        if arxiv_id and await fetchval(
            "SELECT 1 FROM papers WHERE user_id=COALESCE($1, user_id) AND arxiv_id=$2 AND deletion_status='active' LIMIT 1",
            user_id,
            arxiv_id,
        ):
            return True
        if doi and await fetchval(
            "SELECT 1 FROM papers WHERE user_id=COALESCE($1, user_id) AND doi=$2 AND deletion_status='active' LIMIT 1",
            user_id,
            doi,
        ):
            return True
        return False

    async def update_analysis(self, paper_id: str, analysis: dict[str, Any], *, user_id: str | None = None) -> None:
        if user_id:
            await execute(
                "UPDATE papers SET analysis=$1::jsonb, analysis_status='analysis_ready', analysis_error='', analysis_updated_at=$2 WHERE user_id=$3 AND paper_id=$4",
                dump_json(analysis),
                utc_now(),
                user_id,
                paper_id,
            )
            return
        await execute(
            "UPDATE papers SET analysis=$1::jsonb, analysis_status='analysis_ready', analysis_error='', analysis_updated_at=$2 WHERE paper_id=$3",
            dump_json(analysis),
            utc_now(),
            paper_id,
        )

    async def update_analysis_status(
        self,
        paper_id: str,
        status: str,
        *,
        error: str = "",
        clear_analysis: bool = False,
        user_id: str | None = None,
    ) -> None:
        analysis_assignment = "analysis=NULL, " if clear_analysis else ""
        if user_id:
            await execute(
                (
                    f"UPDATE papers SET {analysis_assignment}"
                    "analysis_status=$1, analysis_error=$2, analysis_updated_at=$3 "
                    "WHERE user_id=$4 AND paper_id=$5"
                ),
                status,
                error,
                utc_now(),
                user_id,
                paper_id,
            )
            return
        await execute(
            (
                f"UPDATE papers SET {analysis_assignment}"
                "analysis_status=$1, analysis_error=$2, analysis_updated_at=$3 WHERE paper_id=$4"
            ),
            status,
            error,
            utc_now(),
            paper_id,
        )

    async def update_paper_metadata(
        self,
        paper_id: str,
        patch: dict[str, Any],
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """按白名单字段更新论文元数据，避免任意 patch 直接落库。"""
        if not patch:
            return await self.get_paper(paper_id, user_id)

        assignments: list[str] = []
        params: list[Any] = []
        index = 1
        json_defaults = {"authors": [], "categories": [], "analysis": {}}
        allowed_fields = ("title", "authors", "abstract", "categories", "published_date", "doi", "arxiv_id", "analysis")

        for field in allowed_fields:
            if field not in patch:
                continue
            if field in json_defaults:
                assignments.append(f"{field} = ${index}::jsonb")
                params.append(dump_json(patch.get(field) if patch.get(field) is not None else json_defaults[field]))
            else:
                assignments.append(f"{field} = ${index}")
                params.append(patch.get(field))
            index += 1

        assignments.append(f"last_accessed = ${index}")
        params.append(utc_now())
        index += 1

        if user_id:
            params.extend([user_id, paper_id])
            row = await fetchrow(
                f"""
                UPDATE papers
                SET {", ".join(assignments)}
                WHERE user_id=${index} AND paper_id=${index + 1}
                RETURNING *
                """,
                *params,
            )
        else:
            params.append(paper_id)
            row = await fetchrow(
                f"""
                UPDATE papers
                SET {", ".join(assignments)}
                WHERE paper_id=${index}
                RETURNING *
                """,
                *params,
            )
        return _paper_data(row)

    async def mark_vectorized(self, paper_id: str, *, user_id: str | None = None) -> None:
        """标记论文已完成向量化；上传/索引链会依赖这个状态向前推进。"""
        if user_id:
            await execute("UPDATE papers SET is_vectorized=TRUE WHERE user_id=$1 AND paper_id=$2", user_id, paper_id)
            return
        await execute("UPDATE papers SET is_vectorized=TRUE WHERE paper_id=$1", paper_id)

    async def count_papers(self, tags: list[str] | None, categories: list[str] | None, *, user_id: str | None = None) -> int:
        where_sql, params, _ = _paper_filters(user_id, tags, categories, None)
        return int(await fetchval(f"SELECT COUNT(*) FROM papers{where_sql}", *params) or 0)

    async def count_papers_by_library(
        self,
        library_id: str | None,
        tags: list[str] | None,
        categories: list[str] | None,
        *,
        user_id: str | None = None,
    ) -> int:
        where_sql, params, _ = _paper_filters(user_id, tags, categories, library_id or DEFAULT_LIBRARY_ID)
        return int(await fetchval(f"SELECT COUNT(*) FROM papers{where_sql}", *params) or 0)

    async def list_papers(
        self,
        tags: list[str] | None,
        categories: list[str] | None,
        limit: int,
        offset: int,
        library_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where_sql, params, index = _paper_filters(user_id, tags, categories, library_id)
        rows = await fetch(
            f"SELECT * FROM papers{where_sql} ORDER BY added_at DESC LIMIT ${index} OFFSET ${index + 1}",
            *params,
            limit,
            offset,
        )
        return [_paper_data(row) or {} for row in rows]

    async def search_by_keyword(
        self,
        keyword: str,
        library_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        按 title/abstract 做 ILIKE 搜索。

        这属于轻量元数据搜索，不走 Milvus，也不保证和向量检索结果一致。
        """
        if library_id and user_id:
            rows = await fetch(
                """
                SELECT * FROM papers
                WHERE user_id=$1 AND library_id=$2 AND deletion_status='active'
                  AND (title ILIKE $3 OR abstract ILIKE $3)
                LIMIT 50
                """,
                user_id,
                library_id,
                f"%{keyword}%",
            )
        elif library_id:
            rows = await fetch(
                "SELECT * FROM papers WHERE library_id=$1 AND deletion_status='active' AND (title ILIKE $2 OR abstract ILIKE $2) LIMIT 50",
                library_id,
                f"%{keyword}%",
            )
        elif user_id:
            rows = await fetch(
                "SELECT * FROM papers WHERE user_id=$1 AND deletion_status='active' AND (title ILIKE $2 OR abstract ILIKE $2) LIMIT 50",
                user_id,
                f"%{keyword}%",
            )
        else:
            rows = await fetch(
                "SELECT * FROM papers WHERE deletion_status='active' AND (title ILIKE $1 OR abstract ILIKE $1) LIMIT 50",
                f"%{keyword}%",
            )
        return [_paper_data(row) or {} for row in rows]

    async def list_paper_ids_by_library(self, library_id: str | None = None, *, user_id: str | None = None) -> list[str]:
        if library_id and user_id:
            rows = await fetch(
                "SELECT paper_id FROM papers WHERE user_id=$1 AND library_id=$2 AND deletion_status='active' ORDER BY added_at DESC",
                user_id,
                library_id,
            )
        elif library_id:
            rows = await fetch(
                "SELECT paper_id FROM papers WHERE library_id=$1 AND deletion_status='active' ORDER BY added_at DESC",
                library_id,
            )
        elif user_id:
            rows = await fetch(
                "SELECT paper_id FROM papers WHERE user_id=$1 AND deletion_status='active' ORDER BY added_at DESC",
                user_id,
            )
        else:
            rows = await fetch("SELECT paper_id FROM papers WHERE deletion_status='active' ORDER BY added_at DESC")
        return [str(row["paper_id"]) for row in rows]

    async def list_reconciliation_records(self) -> list[dict[str, Any]]:
        """
        列出孤儿对账使用的全量论文生命周期记录，不按用户或 active 状态过滤。

        Returns:
            包含论文身份、文件路径、删除状态和删除任务状态的记录
        """
        rows = await fetch(
            """
            SELECT p.paper_id, p.user_id, p.library_id, p.pdf_path,
                   p.deletion_status, p.deletion_task_id,
                   task.status AS deletion_task_status
            FROM papers AS p
            LEFT JOIN task_runs AS task
              ON task.task_id=p.deletion_task_id AND task.task_type='paper_delete'
            ORDER BY p.paper_id
            """
        )
        return [normalize_record(row) or {} for row in rows]

    async def delete_paper(self, paper_id: str, *, user_id: str | None = None) -> bool:
        """删除论文主记录；notes 等级联删除由 schema 外键承担。"""
        if user_id:
            return bool(
                await fetchval("DELETE FROM papers WHERE user_id=$1 AND paper_id=$2 RETURNING 1", user_id, paper_id)
            )
        return bool(await fetchval("DELETE FROM papers WHERE paper_id=$1 RETURNING 1", paper_id))


def get_paper_repository() -> PaperRepository:
    """返回论文仓储单例。"""
    global _repository
    if _repository is None:
        _repository = PaperRepository()
    return _repository
