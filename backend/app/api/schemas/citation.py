"""引用相关 schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CitationFormatRequest(BaseModel):
    paper_id: str = ""
    doi: str = ""
    arxiv_id: str = ""
    style: str = "apa"


class CitationBatchRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    style: str = "apa"
