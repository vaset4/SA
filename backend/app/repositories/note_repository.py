"""论文笔记仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.repositories.common import dump_json, load_json, normalize_record
from app.services.database import execute, fetch

_repository: "NoteRepository | None" = None

# notes 是和 papers 强关联的轻量附属实体：
#   - 不参与检索
#   - 不进入向量库
#   - 更多服务于用户在阅读论文时的手工备注


def _normalize_note(row: Any) -> dict[str, Any] | None:
    note = normalize_record(row)
    if note is None:
        return None
    note["evidence_reference"] = load_json(note.get("evidence_reference"), {})
    return note


class NoteRepository:
    """封装论文笔记持久化。"""

    async def add_note(
        self,
        user_id: str,
        paper_id: str | None,
        content: str,
        *,
        project_id: str | None = None,
        conversation_id: str | None = None,
        session_id: str | None = None,
        note_type: str = "reading_note",
        source_type: str = "manual",
        citation: str = "",
        excerpt: str = "",
        evidence_reference: dict[str, Any] | None = None,
    ) -> str:
        """新增一条论文笔记。"""
        note_id = str(uuid.uuid4())
        await execute(
            """
            INSERT INTO notes (
                note_id,
                user_id,
                paper_id,
                project_id,
                conversation_id,
                session_id,
                note_type,
                source_type,
                citation,
                excerpt,
                evidence_reference,
                content
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12)
            """,
            note_id,
            user_id,
            (paper_id or "").strip() or None,
            (project_id or "").strip() or None,
            (conversation_id or "").strip() or None,
            (session_id or "").strip() or None,
            note_type,
            source_type,
            citation,
            excerpt,
            dump_json(evidence_reference or {}),
            content,
        )
        return note_id

    async def list_notes(
        self,
        user_id: str,
        paper_id: str | None = None,
        *,
        project_id: str | None = None,
        conversation_id: str | None = None,
        session_id: str | None = None,
        note_type: str | None = None,
        source_type: str | None = None,
        include_project_papers: bool = False,
        include_project_sessions: bool = False,
        limit: int = 200,
        order: str = "DESC",
    ) -> list[dict[str, Any]]:
        """按关联对象和类型查询笔记。"""
        params: list[Any] = [user_id]
        conditions = ["n.user_id=$1"]

        def push(value: Any) -> str:
            params.append(value)
            return f"${len(params)}"

        normalized_paper_id = (paper_id or "").strip()
        if normalized_paper_id:
            conditions.append(f"n.paper_id={push(normalized_paper_id)}")

        normalized_conversation_id = (conversation_id or "").strip()
        if normalized_conversation_id:
            conditions.append(f"n.conversation_id={push(normalized_conversation_id)}")

        normalized_session_id = (session_id or "").strip()
        if normalized_session_id:
            conditions.append(f"n.session_id={push(normalized_session_id)}")

        normalized_note_type = (note_type or "").strip()
        if normalized_note_type:
            conditions.append(f"n.note_type={push(normalized_note_type)}")

        normalized_source_type = (source_type or "").strip()
        if normalized_source_type:
            conditions.append(f"n.source_type={push(normalized_source_type)}")

        normalized_project_id = (project_id or "").strip()
        if normalized_project_id:
            project_param = push(normalized_project_id)
            project_conditions = [f"n.project_id={project_param}"]
            if include_project_papers:
                project_conditions.append(
                    f"n.paper_id IN (SELECT paper_id FROM project_paper_links WHERE project_id={project_param})"
                )
            if include_project_sessions:
                project_conditions.append(
                    f"n.session_id IN (SELECT session_id FROM project_session_links WHERE project_id={project_param})"
                )
            conditions.append(f"({' OR '.join(project_conditions)})")

        safe_limit = max(1, min(int(limit or 200), 500))
        direction = "ASC" if str(order or "").strip().upper() == "ASC" else "DESC"
        rows = await fetch(
            f"""
            SELECT n.*
            FROM notes n
            WHERE {' AND '.join(conditions)}
            ORDER BY n.created_at {direction}, n.note_id {direction}
            LIMIT {safe_limit}
            """,
            *params,
        )
        return [_normalize_note(row) or {} for row in rows]

    async def delete_notes_by_paper(self, paper_id: str, *, user_id: str | None = None) -> None:
        """删除某篇论文关联笔记；删论文时会先走这里做显式清理。"""
        if user_id:
            await execute("DELETE FROM notes WHERE user_id=$1 AND paper_id=$2", user_id, paper_id)
            return
        await execute("DELETE FROM notes WHERE paper_id=$1", paper_id)


def get_note_repository() -> NoteRepository:
    """返回笔记仓储单例。"""
    global _repository
    if _repository is None:
        _repository = NoteRepository()
    return _repository
