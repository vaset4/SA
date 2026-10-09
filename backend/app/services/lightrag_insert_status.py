"""LightRAG insert status validation helpers."""

from __future__ import annotations

from typing import Any

PROCESSED_DOC_STATUSES = {"processed", "preprocessed"}


class LightRAGInsertFailure(RuntimeError):
    """Raised when LightRAG returns but its document status is not processed."""


async def validate_lightrag_doc_processed(runtime: Any, paper_id: str, track_id: str) -> None:
    status_payload = await _read_doc_status(runtime, paper_id)
    status = _doc_status_value(status_payload)
    if status in PROCESSED_DOC_STATUSES:
        return
    error_msg = _doc_status_error(status_payload)
    raise LightRAGInsertFailure(
        "LightRAG insert incomplete: "
        f"library={runtime.library_id} paper_id={paper_id} "
        f"status={status or 'missing'} track_id={track_id or '-'} "
        f"error={error_msg or 'no error_msg'}"
    )


async def _read_doc_status(runtime: Any, paper_id: str) -> Any:
    doc_status = getattr(getattr(runtime, "rag", None), "doc_status", None)
    if doc_status is None:
        raise LightRAGInsertFailure(f"LightRAG doc_status unavailable: library={runtime.library_id} paper_id={paper_id}")
    if hasattr(doc_status, "get_by_id"):
        return await doc_status.get_by_id(paper_id)
    if hasattr(doc_status, "get_by_ids"):
        records = await doc_status.get_by_ids([paper_id])
        return records[0] if records else None
    raise LightRAGInsertFailure(f"LightRAG doc_status unreadable: library={runtime.library_id} paper_id={paper_id}")


def _doc_status_value(status_payload: Any) -> str:
    if not status_payload:
        return ""
    raw_status = status_payload.get("status") if isinstance(status_payload, dict) else getattr(status_payload, "status", "")
    return str(getattr(raw_status, "value", raw_status) or "")


def _doc_status_error(status_payload: Any) -> str:
    if not status_payload:
        return ""
    raw_error = status_payload.get("error_msg") if isinstance(status_payload, dict) else getattr(status_payload, "error_msg", "")
    return str(raw_error or "")
