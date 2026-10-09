"""论文与笔记相关 schema。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchPapersRequest(BaseModel):
    query: str
    library_id: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    category: Optional[str] = None
    authors: Optional[str] = None
    top_k: int = 10


class NoteRequest(BaseModel):
    paper_id: Optional[str] = None
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    content: str
    note_type: str = "reading_note"
    source_type: str = "manual"
    citation: str = ""
    excerpt: str = ""
    evidence_reference: dict[str, Any] = Field(default_factory=dict)


class PaperMetadataUpdateRequest(BaseModel):
    title: Optional[str] = None
    authors: Optional[list[str] | str] = None
    abstract: Optional[str] = None
    year: Optional[str | int] = None
    published_date: Optional[str] = None
    category: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    keywords: Optional[list[str] | str] = None
    problem: Optional[str] = None
    method: Optional[str] = None
    contribution: Optional[list[str] | str] = None
    limitation: Optional[str] = None
    tldr: Optional[str] = None
    methodology_type: Optional[str] = None
    methodology_detail: Optional[str] = None
    related_work: Optional[list[str] | str] = None
