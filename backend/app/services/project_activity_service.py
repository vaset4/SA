"""项目活动流服务。"""

from __future__ import annotations

from typing import Any

from app.repositories.project_activity_repository import get_project_activity_repository


async def ensure_project_activity_schema() -> None:
    await get_project_activity_repository().ensure_schema()


async def append_activity(project_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    await ensure_project_activity_schema()
    return await get_project_activity_repository().append_activity(project_id, event_type, payload or {})


def _scope_text(payload: dict[str, Any]) -> str:
    paper_ids = payload.get("paper_ids")
    if isinstance(paper_ids, list) and paper_ids:
        return f"基于 {len([item for item in paper_ids if item])} 篇论文继续推进。"
    return "当前会话还没有明确挂到论文范围。"


def _describe_activity(event_type: str, payload: dict[str, Any], created_at: str) -> dict[str, Any]:
    if event_type == "paper_added":
        return {
            "id": payload.get("activity_id") or f"paper-added-{payload.get('paper_id') or created_at}",
            "type": "paper_added",
            "label": "论文加入项目",
            "title": payload.get("title") or payload.get("paper_id") or "未命名论文",
            "description": payload.get("description") or "已纳入当前项目材料区，后续可继续补分析和研究笔记。",
            "at": created_at,
        }
    if event_type == "analysis_completed":
        return {
            "id": payload.get("activity_id") or f"analysis-{payload.get('paper_id') or created_at}",
            "type": "analysis_completed",
            "label": "AI 分析完成",
            "title": payload.get("title") or payload.get("paper_id") or "未命名论文",
            "description": payload.get("summary") or "论文已具备进入正式研究范围的基础材料。",
            "at": created_at,
        }
    if event_type == "internal_research_started":
        return {
            "id": payload.get("activity_id") or f"internal-research-{payload.get('session_id') or created_at}",
            "type": "internal_research_started",
            "label": "内部研究启动",
            "title": payload.get("topic") or payload.get("session_id") or "未命名研究",
            "description": f"{_scope_text(payload)} {str(payload.get('summary') or '研究已发起，等待进一步沉淀。').strip()}".strip(),
            "at": created_at,
        }
    return {
        "id": payload.get("activity_id") or f"{event_type}-{created_at}",
        "type": event_type,
        "label": event_type,
        "title": payload.get("title") or "项目动态",
        "description": str(payload.get("description") or "").strip(),
        "at": created_at,
    }


async def list_activities(project_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    await ensure_project_activity_schema()
    rows = await get_project_activity_repository().list_activities(project_id, limit=limit)
    visible: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        payload = {**payload, "activity_id": row.get("activity_id")}
        visible.append(_describe_activity(str(row.get("event_type") or ""), payload, str(row.get("created_at") or "")))
    return visible
