"""Guardrails short-circuit trace persistence."""

from __future__ import annotations

import time
from typing import Any

from app.middleware.trace_context import (
    TraceContext,
    build_full_trace,
    build_trace_summary,
    safe_write_trace,
)

ZERO_RETRIEVAL_METRICS = {
    "hybrid_results": 0,
    "reranked": 0,
    "graded_relevant": 0,
    "graded_irrelevant": 0,
}
ZERO_LATENCY_METRICS = {
    "retrieval": 0,
    "rerank": 0,
    "grading": 0,
    "generation": 0,
    "evaluation": 0,
}
GUARDRAILS_REJECT_MODE = "guardrails_reject"
SKIPPED_SYNTHESIS_STATUS = "skipped"


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _scope_from_state(state: dict[str, Any]) -> dict[str, Any]:
    structured_query = _dict_value(state, "structured_query")
    project_scope = _dict_value(state, "project_scope")
    paper_focus = _dict_value(state, "paper_focus")
    paper_ids = structured_query.get("paper_ids") or project_scope.get("paper_ids") or []
    if not isinstance(paper_ids, list):
        paper_ids = []
    return {
        "project_id": structured_query.get("project_id") or project_scope.get("project_id") or "",
        "paper_ids": [str(paper_id) for paper_id in paper_ids if str(paper_id or "").strip()],
        "paper_focus_mode": paper_focus.get("mode") or "",
    }


def _guardrails_trace_dict(
    *,
    ctx: TraceContext,
    state: dict[str, Any],
    result: dict[str, Any],
    started_at: float,
    user_query: str,
) -> dict[str, Any]:
    source_status = _dict_value(result, "source_status")
    latency = {"total": max(int((time.monotonic() - started_at) * 1000), 0), **ZERO_LATENCY_METRICS}
    return {
        "request_id": ctx.request_id,
        "session_id": ctx.session_id,
        "query": user_query,
        "rewritten_queries": [],
        "mode": GUARDRAILS_REJECT_MODE,
        "dialogue_mode": state.get("dialogue_mode") or "",
        "graph_mode": "",
        "retrieval_reasoning": result.get("guardrails_reason") or "",
        "source_status": dict(source_status),
        "outcome": GUARDRAILS_REJECT_MODE,
        "scope": _scope_from_state(state),
        "synthesis": {"status": SKIPPED_SYNTHESIS_STATUS, "evidence_layers": {}},
        "retrieval": dict(ZERO_RETRIEVAL_METRICS),
        "generation": {"self_rag_iterations": 0, "final_quality_score": None, "citations_count": 0},
        "latency_ms": latency,
    }


async def attach_short_circuit_trace(
    result: dict[str, Any],
    *,
    ctx: TraceContext | None,
    state: dict[str, Any],
    started_at: float,
    user_query: str,
) -> dict[str, Any]:
    """Persist a trace for RAG requests rejected before the RAG subgraph."""
    if ctx is None or result.get("guardrails_action") != "reject":
        return result
    if result.get("_rag_trace_dict") or result.get("trace_summary"):
        return result
    trace_dict = _guardrails_trace_dict(
        ctx=ctx,
        state=state,
        result=result,
        started_at=started_at,
        user_query=user_query,
    )
    await safe_write_trace(
        trace_data=build_full_trace(ctx, trace_dict),
        request_id=ctx.request_id,
        session_id=ctx.session_id,
        query=user_query,
        user_id=ctx.user_id,
    )
    return {
        **result,
        "request_id": ctx.request_id,
        "trace_summary": build_trace_summary(ctx, trace_dict),
        "_rag_trace_dict": trace_dict,
    }
