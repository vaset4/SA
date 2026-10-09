"""Read-only chunk/vector inspection service."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Literal, Protocol

from app.models.knowledge_inspector_contracts import (
    ChunkAuditIssue,
    ChunkAuditSummary,
    ChunkDetailResponse,
    ChunkMapChild,
    ChunkMapParentGroup,
    ChunkMapSectionGroup,
    ChunkMetadata,
    PaperChunkAuditResponse,
    PaperChunkMapResponse,
    PaperSummary,
    TextFieldView,
)
from app.models.models import Paper
from app.services import paper_db
from app.services.milvus_store import get_milvus_store


IncludeTextMode = Literal["preview", "full"]
PREVIEW_CHARS = 280
DEFAULT_MAX_TEXT_CHARS = 4000
TOP_ISSUE_LIMIT = 5
VERY_LARGE_PARENT_CHARS = 60000
TEXT_DIFF_RATIO = 0.5
CRITICAL_FIELDS = ("chunk_id", "paper_id", "parent_id", "full_text")
WARNING_FIELDS = ("embedding_text", "display_text", "section_path", "section", "block_type")


class InspectorStore(Protocol):
    """Milvus read methods used by the inspector."""

    def get_chunk_row(self, chunk_id: str) -> dict | None: ...

    def list_chunk_rows_by_paper(self, paper_id: str) -> list[dict]: ...

    def read_chunk(self, chunk_id: str, already_read: set[str]) -> dict: ...


PaperProvider = Callable[[str], Paper | None]


def get_chunk_detail(
    chunk_id: str,
    *,
    include_text: IncludeTextMode = "preview",
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    store: InspectorStore | None = None,
    paper_provider: PaperProvider | None = None,
) -> ChunkDetailResponse:
    """Build a chunk detail response from the current Milvus row."""
    row = _store(store).get_chunk_row(chunk_id)
    if row is None:
        return ChunkDetailResponse(
            chunk_id=chunk_id,
            vector_row_found=False,
            metadata=ChunkMetadata(),
            missing_fields=list(CRITICAL_FIELDS),
        )
    paper = _paper_summary(_str(row, "paper_id"), paper_provider)
    return ChunkDetailResponse(
        chunk_id=_str(row, "chunk_id") or chunk_id,
        paper_id=_str(row, "paper_id"),
        parent_id=_str(row, "parent_id"),
        vector_row_found=True,
        paper=paper,
        metadata=_metadata(row),
        text_fields=_text_fields(row, include_text, max_text_chars),
        token_counts=_token_counts(row),
        missing_fields=_missing_fields(row),
    )


def get_paper_chunk_map(
    paper_id: str,
    *,
    store: InspectorStore | None = None,
    paper_provider: PaperProvider | None = None,
) -> PaperChunkMapResponse:
    """Build a metadata-grouped chunk map for one paper."""
    rows = _store(store).list_chunk_rows_by_paper(paper_id)
    return PaperChunkMapResponse(
        paper_id=paper_id,
        paper=_paper_summary(paper_id, paper_provider),
        vector_row_count=len(rows),
        sections=_section_groups(rows),
    )


def get_paper_chunk_audit(
    paper_id: str,
    *,
    store: InspectorStore | None = None,
    paper_provider: PaperProvider | None = None,
) -> PaperChunkAuditResponse:
    """Run structural lint over current Milvus rows and paper metadata."""
    current_store = _store(store)
    rows = current_store.list_chunk_rows_by_paper(paper_id)
    paper = _paper_summary(paper_id, paper_provider)
    issues = _audit_issues(paper_id, rows, paper, current_store)
    stats = _audit_stats(rows)
    summary = _audit_summary(issues)
    return PaperChunkAuditResponse(
        paper_id=paper_id,
        paper=paper,
        vector_row_count=len(rows),
        summary=summary,
        issues=issues,
        stats=stats,
    )


def build_upload_chunk_audit_summary(paper_id: str) -> dict[str, Any]:
    """Return upload-facing audit summary fields for a saved paper."""
    audit = get_paper_chunk_audit(paper_id)
    return {
        "chunk_audit_status": audit.summary.status,
        "chunk_audit_error_count": audit.summary.error_count,
        "chunk_audit_warning_count": audit.summary.warning_count,
        "chunk_audit_top_issues": [
            issue.model_dump() for issue in audit.summary.top_issues
        ],
    }


def _store(store: InspectorStore | None) -> InspectorStore:
    return store or get_milvus_store()


def _paper_summary(paper_id: str, provider: PaperProvider | None) -> PaperSummary:
    if not paper_id:
        return PaperSummary(paper_id="", exists=False)
    paper = (provider or paper_db.get_paper)(paper_id)
    if not paper:
        return PaperSummary(paper_id=paper_id, exists=False)
    return PaperSummary(
        paper_id=paper_id,
        exists=True,
        title=str(paper.get("title") or ""),
        is_vectorized=bool(paper.get("is_vectorized")),
    )


def _metadata(row: dict) -> ChunkMetadata:
    return ChunkMetadata(
        section=_str(row, "section"),
        section_path=_str(row, "section_path"),
        page_num=_int(row, "page_num"),
        block_type=_str(row, "block_type"),
        asset_ref=_str(row, "asset_ref"),
        parser_mode=_str(row, "parser_mode"),
    )


def _text_fields(row: dict, include_text: IncludeTextMode, max_chars: int) -> list[TextFieldView]:
    fields = ("embedding_text", "display_text", "full_text")
    return [_text_field(name, _str(row, name), include_text, max_chars) for name in fields]


def _text_field(name: str, text: str, include_text: IncludeTextMode, max_chars: int) -> TextFieldView:
    limit = max(1, max_chars)
    returned = text[:limit] if include_text == "full" else ""
    return TextFieldView(
        name=name,
        length=len(text),
        preview=_preview(text),
        text=returned if include_text == "full" else None,
        returned_chars=len(returned),
        total_chars=len(text),
        truncated=include_text == "full" and len(text) > len(returned),
    )


def _token_counts(row: dict) -> dict[str, int]:
    keys = ("token_count", "parent_token_count", "child_token_count")
    return {key: _int(row, key) for key in keys if _int(row, key) > 0}


def _missing_fields(row: dict) -> list[str]:
    missing = [field for field in (*CRITICAL_FIELDS, *WARNING_FIELDS) if not _str(row, field)]
    if _int(row, "page_num") <= 0:
        missing.append("page_num")
    return missing


def _section_groups(rows: list[dict]) -> list[ChunkMapSectionGroup]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_str(row, "section_path")].append(row)
    return [_section_group(section_path, grouped[section_path]) for section_path in sorted(grouped)]


def _section_group(section_path: str, rows: list[dict]) -> ChunkMapSectionGroup:
    parents = _parent_groups(rows)
    return ChunkMapSectionGroup(
        section_path=section_path,
        section=_str(rows[0], "section") if rows else "",
        parent_count=len(parents),
        child_count=len(rows),
        parents=parents,
    )


def _parent_groups(rows: list[dict]) -> list[ChunkMapParentGroup]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_str(row, "parent_id")].append(row)
    return [_parent_group(parent_id, grouped[parent_id]) for parent_id in sorted(grouped)]


def _parent_group(parent_id: str, rows: list[dict]) -> ChunkMapParentGroup:
    full_texts = {_str(row, "full_text") for row in rows}
    children = [_child(row) for row in sorted(rows, key=_chunk_sort_key)]
    return ChunkMapParentGroup(
        parent_id=parent_id,
        child_count=len(children),
        full_text_length=len(_str(rows[0], "full_text")) if rows else 0,
        full_text_consistent=len(full_texts) <= 1,
        chunks=children,
    )


def _child(row: dict) -> ChunkMapChild:
    return ChunkMapChild(
        chunk_id=_str(row, "chunk_id"),
        parent_id=_str(row, "parent_id"),
        section=_str(row, "section"),
        section_path=_str(row, "section_path"),
        page_num=_int(row, "page_num"),
        block_type=_str(row, "block_type"),
        parser_mode=_str(row, "parser_mode"),
        display_preview=_preview(_str(row, "display_text")),
        embedding_length=len(_str(row, "embedding_text")),
        display_length=len(_str(row, "display_text")),
        full_text_length=len(_str(row, "full_text")),
        missing_fields=_missing_fields(row),
    )


def _audit_issues(
    paper_id: str,
    rows: list[dict],
    paper: PaperSummary,
    store: InspectorStore,
) -> list[ChunkAuditIssue]:
    issues = []
    issues.extend(_paper_mismatch_issues(paper_id, rows, paper))
    issues.extend(_row_field_issues(rows))
    issues.extend(_duplicate_chunk_issues(rows))
    issues.extend(_parent_consistency_issues(rows))
    issues.extend(_readability_issues(rows, store))
    issues.extend(_info_issues(paper_id, rows))
    return _with_issue_ids(issues)


def _paper_mismatch_issues(paper_id: str, rows: list[dict], paper: PaperSummary) -> list[ChunkAuditIssue]:
    if rows and not paper.exists:
        return [_issue("ERROR", "paper_missing_in_db", "Milvus rows exist but paper_db has no paper", paper_id)]
    if paper.exists and paper.is_vectorized and not rows:
        return [_issue("ERROR", "vectorized_paper_without_rows", "paper_db says vectorized but Milvus has no rows", paper_id)]
    return []


def _row_field_issues(rows: list[dict]) -> list[ChunkAuditIssue]:
    issues: list[ChunkAuditIssue] = []
    for row in rows:
        issues.extend(_missing_field_issues(row, CRITICAL_FIELDS, "ERROR"))
        issues.extend(_missing_field_issues(row, WARNING_FIELDS, "WARNING"))
        if _int(row, "page_num") <= 0:
            issues.append(_row_issue(row, "WARNING", "invalid_page_num", "page_num must be positive"))
        if _texts_differ_too_much(row):
            issues.append(_row_issue(row, "WARNING", "embedding_display_text_drift", "embedding_text and display_text differ significantly"))
    return issues


def _missing_field_issues(rows: dict, fields: Iterable[str], severity: str) -> list[ChunkAuditIssue]:
    return [
        _row_issue(rows, severity, f"missing_{field}", f"Missing {field}")
        for field in fields
        if not _str(rows, field)
    ]


def _duplicate_chunk_issues(rows: list[dict]) -> list[ChunkAuditIssue]:
    counts = Counter(_str(row, "chunk_id") for row in rows if _str(row, "chunk_id"))
    return [
        _issue("ERROR", "duplicate_chunk_id", "Duplicate chunk_id in Milvus rows", "", chunk_id=chunk_id)
        for chunk_id, count in counts.items()
        if count > 1
    ]


def _parent_consistency_issues(rows: list[dict]) -> list[ChunkAuditIssue]:
    issues = []
    for parent_id, parent_rows in _rows_by(rows, "parent_id").items():
        full_texts = {_str(row, "full_text") for row in parent_rows}
        if len(full_texts) > 1:
            issues.append(_issue("WARNING", "inconsistent_parent_full_text", "same parent_id has inconsistent full_text", "", parent_id=parent_id))
        if any(len(text) > VERY_LARGE_PARENT_CHARS for text in full_texts):
            issues.append(_issue("WARNING", "parent_full_text_very_large", "parent full_text is very large", "", parent_id=parent_id))
    return issues


def _readability_issues(rows: list[dict], store: InspectorStore) -> list[ChunkAuditIssue]:
    issues = []
    already_read: set[str] = set()
    for row in rows:
        chunk_id = _str(row, "chunk_id")
        if chunk_id and _read_chunk_missing(store, chunk_id, already_read):
            issues.append(_row_issue(row, "ERROR", "read_chunk_missing", "read_chunk cannot read existing row"))
    return issues


def _info_issues(paper_id: str, rows: list[dict]) -> list[ChunkAuditIssue]:
    stats = _audit_stats(rows)
    return [
        _issue("INFO", "block_type_distribution", "Block type distribution", paper_id, details=stats.get("block_types", {})),
        _issue("INFO", "parser_mode_distribution", "Parser mode distribution", paper_id, details=stats.get("parser_modes", {})),
        _issue("INFO", "parent_child_counts", "Parent and child counts", paper_id, details=stats.get("counts", {})),
    ]


def _audit_summary(issues: list[ChunkAuditIssue]) -> ChunkAuditSummary:
    errors = [issue for issue in issues if issue.severity == "ERROR"]
    warnings = [issue for issue in issues if issue.severity == "WARNING"]
    infos = [issue for issue in issues if issue.severity == "INFO"]
    status = "failed" if errors else "warning" if warnings else "ok"
    return ChunkAuditSummary(
        status=status,
        error_count=len(errors),
        warning_count=len(warnings),
        info_count=len(infos),
        top_issues=[*errors, *warnings][:TOP_ISSUE_LIMIT],
    )


def _audit_stats(rows: list[dict]) -> dict[str, Any]:
    pages = [_int(row, "page_num") for row in rows if _int(row, "page_num") > 0]
    parents = {_str(row, "parent_id") for row in rows if _str(row, "parent_id")}
    return {
        "counts": {"chunks": len(rows), "parents": len(parents)},
        "page_range": {"min": min(pages) if pages else 0, "max": max(pages) if pages else 0},
        "block_types": dict(Counter(_str(row, "block_type") or "(missing)" for row in rows)),
        "parser_modes": dict(Counter(_str(row, "parser_mode") or "(missing)" for row in rows)),
        "average_lengths": _average_lengths(rows),
    }


def _average_lengths(rows: list[dict]) -> dict[str, int]:
    if not rows:
        return {"embedding_text": 0, "display_text": 0, "full_text": 0}
    return {
        field: sum(len(_str(row, field)) for row in rows) // len(rows)
        for field in ("embedding_text", "display_text", "full_text")
    }


def _read_chunk_missing(store: InspectorStore, chunk_id: str, already_read: set[str]) -> bool:
    result = store.read_chunk(chunk_id, already_read)
    return bool(result.get("skipped")) and "not found" in str(result.get("full_text") or "")


def _texts_differ_too_much(row: dict) -> bool:
    embedding = _normalized_embedding_body(row)
    display = _normalized_display_text(row)
    if not embedding or not display:
        return False
    if embedding == display:
        return False
    length_gap = abs(len(embedding) - len(display)) / max(len(embedding), len(display), 1)
    return length_gap > TEXT_DIFF_RATIO and embedding[:80] != display[:80]


def _normalized_embedding_body(row: dict) -> str:
    embedding = _normalize_text_for_drift(_str(row, "embedding_text"))
    header = _normalize_text_for_drift(_str(row, "section_path"))
    if header and embedding.startswith(header):
        embedding = embedding[len(header):]
    return embedding.strip()


def _normalized_display_text(row: dict) -> str:
    return _normalize_text_for_drift(_str(row, "display_text"))


def _normalize_text_for_drift(text: str) -> str:
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    cleaned = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"[*_~`#>|]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().casefold()


def _rows_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = _str(row, key)
        if value:
            grouped[value].append(row)
    return grouped


def _with_issue_ids(issues: list[ChunkAuditIssue]) -> list[ChunkAuditIssue]:
    return [
        issue.model_copy(update={"issue_id": f"{index + 1:04d}-{issue.code}"})
        for index, issue in enumerate(issues)
    ]


def _issue(
    severity: str,
    code: str,
    message: str,
    paper_id: str,
    *,
    chunk_id: str = "",
    parent_id: str = "",
    details: dict[str, Any] | None = None,
) -> ChunkAuditIssue:
    return ChunkAuditIssue(
        issue_id="",
        severity=severity,
        code=code,
        message=message,
        paper_id=paper_id,
        chunk_id=chunk_id,
        parent_id=parent_id,
        details=details or {},
    )


def _row_issue(row: dict, severity: str, code: str, message: str) -> ChunkAuditIssue:
    return ChunkAuditIssue(
        issue_id="",
        severity=severity,
        code=code,
        message=message,
        paper_id=_str(row, "paper_id"),
        chunk_id=_str(row, "chunk_id"),
        parent_id=_str(row, "parent_id"),
        section_path=_str(row, "section_path"),
        page_num=_int(row, "page_num"),
    )


def _chunk_sort_key(row: dict) -> tuple[int, str]:
    return (_int(row, "page_num"), _str(row, "chunk_id"))


def _preview(text: str) -> str:
    return text if len(text) <= PREVIEW_CHARS else f"{text[:PREVIEW_CHARS]}..."


def _str(row: dict, key: str) -> str:
    return str(row.get(key) or "").strip()


def _int(row: dict, key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0
