"""Runtime evidence-layer participation summary for RAG traces."""

from __future__ import annotations

from typing import Any

_NOT_CALLED_STATUSES = {"", "skipped", "unused", "disabled", "not_called"}


def build_evidence_layer_status(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Summarize available evidence layers and whether this request used them."""
    source_status = state.get("source_status") if isinstance(state.get("source_status"), dict) else {}
    reasons = state.get("source_status_reasons") if isinstance(state.get("source_status_reasons"), dict) else {}
    record_sources = _record_sources(state)
    return {
        "planner": _layer_entry(
            source_key="planner",
            status=str(source_status.get("planner") or ""),
            reason=str(reasons.get("planner") or ""),
            has_evidence=str(state.get("runtime_chain") or "") == "planner",
        ),
        "milvus": _layer_entry(
            source_key="milvus",
            status=str(source_status.get("milvus") or ""),
            reason=str(reasons.get("milvus") or ""),
            has_evidence="milvus" in record_sources or _has_selected_text_evidence(state),
        ),
        "lightrag": _layer_entry(
            source_key="graph",
            status=str(source_status.get("graph") or ""),
            reason=str(reasons.get("graph") or ""),
            has_evidence="lightrag" in record_sources,
        ),
        "neo4j": _layer_entry(
            source_key="neo4j",
            status=str(source_status.get("neo4j") or ""),
            reason=str(reasons.get("neo4j") or ""),
            has_evidence="neo4j" in record_sources or bool(state.get("graph_evidence")),
        ),
        "inventory": _layer_entry(
            source_key="inventory",
            status=str(source_status.get("inventory") or ""),
            reason=str(reasons.get("inventory") or ""),
            has_evidence="inventory" in record_sources or bool(state.get("inventory_contexts")),
        ),
    }


def _layer_entry(
    *,
    source_key: str,
    status: str,
    reason: str,
    has_evidence: bool,
) -> dict[str, Any]:
    normalized_status = status or ("ok" if has_evidence else "not_called")
    return {
        "available": True,
        "called": _status_called(normalized_status, has_evidence=has_evidence),
        "status": normalized_status,
        "reason": reason,
        "source_key": source_key,
    }


def _status_called(status: str, *, has_evidence: bool) -> bool:
    return has_evidence or status not in _NOT_CALLED_STATUSES


def _record_sources(state: dict[str, Any]) -> set[str]:
    sources: set[str] = set()
    for item in state.get("evidence_records") or []:
        if isinstance(item, dict) and item.get("source"):
            sources.add(str(item["source"]))
    return sources


def _has_selected_text_evidence(state: dict[str, Any]) -> bool:
    return any(isinstance(item, dict) for item in state.get("selected_evidence") or [])

