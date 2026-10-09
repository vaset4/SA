"""项目活动流仓储。"""

from __future__ import annotations

import uuid
from typing import Any

from app.repositories.common import dump_json, load_json, normalize_record
from app.services.database import execute, fetch

_repository: "ProjectActivityRepository | None" = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_activities (
    activity_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_project_activities_project_created
    ON project_activities(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_project_activities_event_type
    ON project_activities(event_type, created_at DESC);
"""


def _activity_data(row: Any) -> dict[str, Any] | None:
    data = normalize_record(row)
    if data is None:
        return None
    data["payload"] = load_json(data.get("payload"), {})
    return data


class ProjectActivityRepository:
    async def ensure_schema(self) -> None:
        await execute(SCHEMA_SQL)

    async def append_activity(
        self,
        project_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        activity_id = f"act_{uuid.uuid4().hex[:12]}"
        await execute(
            """
            INSERT INTO project_activities (activity_id, project_id, event_type, payload)
            VALUES ($1, $2, $3, $4::jsonb)
            """,
            activity_id,
            project_id,
            event_type,
            dump_json(payload or {}),
        )
        rows = await fetch(
            "SELECT * FROM project_activities WHERE activity_id=$1",
            activity_id,
        )
        return _activity_data(rows[0]) or {}

    async def list_activities(self, project_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = await fetch(
            """
            SELECT *
            FROM project_activities
            WHERE project_id=$1
            ORDER BY created_at DESC, activity_id DESC
            LIMIT $2
            """,
            project_id,
            limit,
        )
        return [_activity_data(row) or {} for row in rows]


def get_project_activity_repository() -> ProjectActivityRepository:
    global _repository
    if _repository is None:
        _repository = ProjectActivityRepository()
    return _repository
