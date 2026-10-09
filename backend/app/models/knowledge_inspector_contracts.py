"""Contracts for developer-facing chunk/vector inspection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeInspectorModel(BaseModel):
    """Base model for inspector payloads crossing API boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class TextFieldView(KnowledgeInspectorModel):
    name: str
    length: int = Field(ge=0)
    preview: str = ""
    text: str | None = None
    returned_chars: int = Field(default=0, ge=0)
    total_chars: int = Field(default=0, ge=0)
    truncated: bool = False


class PaperSummary(KnowledgeInspectorModel):
    paper_id: str
    exists: bool
    title: str = ""
    is_vectorized: bool = False


class ChunkMetadata(KnowledgeInspectorModel):
    section: str = ""
    section_path: str = ""
    page_num: int = 0
    block_type: str = ""
    asset_ref: str = ""
    parser_mode: str = ""


class ChunkDetailResponse(KnowledgeInspectorModel):
    chunk_id: str
    paper_id: str = ""
    parent_id: str = ""
    vector_row_found: bool
    paper: PaperSummary | None = None
    metadata: ChunkMetadata
    text_fields: list[TextFieldView] = Field(default_factory=list)
    token_counts: dict[str, int] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)


class ChunkMapChild(KnowledgeInspectorModel):
    chunk_id: str
    parent_id: str
    section: str = ""
    section_path: str = ""
    page_num: int = 0
    block_type: str = ""
    parser_mode: str = ""
    display_preview: str = ""
    embedding_length: int = Field(default=0, ge=0)
    display_length: int = Field(default=0, ge=0)
    full_text_length: int = Field(default=0, ge=0)
    missing_fields: list[str] = Field(default_factory=list)


class ChunkMapParentGroup(KnowledgeInspectorModel):
    parent_id: str
    child_count: int = Field(ge=0)
    full_text_length: int = Field(default=0, ge=0)
    full_text_consistent: bool = True
    chunks: list[ChunkMapChild] = Field(default_factory=list)


class ChunkMapSectionGroup(KnowledgeInspectorModel):
    section_path: str
    section: str = ""
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)
    parents: list[ChunkMapParentGroup] = Field(default_factory=list)


class PaperChunkMapResponse(KnowledgeInspectorModel):
    paper_id: str
    paper: PaperSummary
    vector_row_count: int = Field(ge=0)
    map_type: Literal["metadata_grouped"] = "metadata_grouped"
    tree_quality: Literal["derived_from_section_path"] = "derived_from_section_path"
    sections: list[ChunkMapSectionGroup] = Field(default_factory=list)


AuditSeverity = Literal["ERROR", "WARNING", "INFO"]
AuditStatus = Literal["ok", "warning", "failed"]


class ChunkAuditIssue(KnowledgeInspectorModel):
    issue_id: str
    severity: AuditSeverity
    code: str
    message: str
    paper_id: str = ""
    chunk_id: str = ""
    parent_id: str = ""
    section_path: str = ""
    page_num: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ChunkAuditSummary(KnowledgeInspectorModel):
    status: AuditStatus
    error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    info_count: int = Field(ge=0)
    top_issues: list[ChunkAuditIssue] = Field(default_factory=list)


class PaperChunkAuditResponse(KnowledgeInspectorModel):
    paper_id: str
    paper: PaperSummary
    vector_row_count: int = Field(ge=0)
    summary: ChunkAuditSummary
    issues: list[ChunkAuditIssue] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
