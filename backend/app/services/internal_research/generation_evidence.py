"""Map canonical evidence contracts into generation-facing evidence."""

from __future__ import annotations

from typing import Any

from app.models import evidence_layers as layers


def evidence_text(item: dict[str, Any]) -> str:
    return str(item.get("expanded_text") or item.get("excerpt") or "").strip()


def from_evidence_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in state.get("evidence_records") or []:
        if not _is_citable_direct_text(item):
            continue
        mapped = _map_record(item, index=len(selected) + 1)
        if mapped:
            selected.append(mapped)
    return selected


def from_legacy_selected_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in state.get("selected_evidence") or []:
        if not isinstance(item, dict):
            continue
        if _has_generation_identity(item):
            selected.append(dict(item))
    return selected


def from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_records = state.get("evidence_records")
    if isinstance(evidence_records, list) and evidence_records:
        return from_evidence_records(state)
    return from_legacy_selected_evidence(state)


def _is_citable_direct_text(item: object) -> bool:
    if not isinstance(item, dict) or not item.get("usable_as_citation"):
        return False
    return item.get("layer") == layers.EVIDENCE_LAYER_DIRECT_TEXT and _has_generation_identity(item)


def _has_generation_identity(item: dict[str, Any]) -> bool:
    paper_id = str(item.get("paper_id") or "").strip()
    chunk_id = str(item.get("chunk_id") or "").strip()
    return bool(paper_id and chunk_id and evidence_text(item))


def _map_record(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "evidence_id": str(item.get("evidence_id") or f"E{index}").strip(),
        "paper_id": str(item.get("paper_id") or "").strip(),
        "chunk_id": str(item.get("chunk_id") or "").strip(),
        "parent_id": str(item.get("parent_id") or "").strip(),
        "section": str(item.get("section") or "").strip(),
        "page_num": int(item.get("page_num") or 0),
        "title": str(item.get("title") or "").strip(),
        "excerpt": str(item.get("excerpt") or "").strip(),
        "expanded_text": str(item.get("expanded_text") or "").strip(),
        "source_channel": "evidence_records",
        "score": float(item.get("confidence") or 0.0),
    }
