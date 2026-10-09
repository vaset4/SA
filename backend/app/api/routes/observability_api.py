"""可观测性 API — 面向 developer 角色的 RAG trace 查询。"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query

from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.repositories.observability_repository import get_observability_repository
from app.services.auth_roles import can_view_observability

router = APIRouter(prefix="/observability", tags=["observability"])


def _require_developer(user: dict) -> str:
    if not can_view_observability(user):
        raise HTTPException(status_code=403, detail="developer or admin role required")
    return get_user_id(user)


@router.get("/rag-traces")
async def list_rag_traces(
    limit: int = Query(default=50, ge=1, le=200),
    request_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    current_user_id = _require_developer(user)
    repo = get_observability_repository()
    rows = await repo.list_rag_traces(
        user_id=current_user_id,
        limit=limit,
        request_id=request_id or None,
        session_id=session_id or None,
        q=q or None,
    )
    results = []
    for row in rows:
        summary = row.get("summary")
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except (json.JSONDecodeError, TypeError):
                pass
        results.append({
            "request_id": row.get("request_id"),
            "session_id": row.get("session_id"),
            "query": row.get("query"),
            "created_at": str(row.get("created_at") or ""),
            "summary": summary,
        })
    return {"traces": results, "count": len(results)}


@router.get("/rag-traces/{request_id}")
async def get_rag_trace(
    request_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    current_user_id = _require_developer(user)
    repo = get_observability_repository()
    row = await repo.get_rag_trace(request_id)
    if not row:
        raise HTTPException(status_code=404, detail="trace not found")
    trace_user_id = str(row.get("user_id") or "unknown")
    # AC10/AC12: ownership check
    if trace_user_id != current_user_id:
        # Historical trace with unknown user_id: verify via session ownership
        if trace_user_id == "unknown":
            from app.services import paper_db
            trace_session_id = str(row.get("session_id") or "").strip()
            owned_session = (
                await asyncio.to_thread(
                    paper_db.get_session,
                    trace_session_id,
                    user_id=current_user_id,
                )
                if trace_session_id
                else None
            )
            if owned_session is None:
                raise HTTPException(status_code=404, detail="trace not found")
        else:
            raise HTTPException(status_code=404, detail="trace not found")
    trace_data = row.get("trace_data")
    if isinstance(trace_data, str):
        try:
            trace_data = json.loads(trace_data)
        except (json.JSONDecodeError, TypeError):
            trace_data = {}
    return {
        "request_id": row.get("request_id"),
        "session_id": row.get("session_id"),
        "query": row.get("query"),
        "user_id": trace_user_id,
        "created_at": str(row.get("created_at") or ""),
        "trace": trace_data or {},
    }
