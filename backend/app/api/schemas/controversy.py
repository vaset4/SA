"""争议与 gap schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaimTripleSchema(BaseModel):
    paper_id: str
    task: str = ""
    method: str = ""
    dataset: str = ""
    metric: str = ""
    claim_direction: str = "uncertain"
    claim_text: str = ""
    evidence_excerpt: str = ""


class ConflictClusterSchema(BaseModel):
    cluster_id: str
    topic_key: str
    papers: list[str] = Field(default_factory=list)
    conflict_reason: str = ""
    conflict_type: str = ""
    support_claims: list[ClaimTripleSchema] = Field(default_factory=list)
    summary: str = ""


class GapCandidateSchema(BaseModel):
    gap_id: str
    gap_type: str
    topic_key: str
    support_paper_ids: list[str] = Field(default_factory=list)
    evidence_items: list[str] = Field(default_factory=list)
    severity: float = 0
    summary: str = ""
    next_research_question: str = ""


class ControversyMapRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    topic: str = ""
    include_gaps: bool = True
    include_conflicts: bool = True
    project_id: str | None = None


class ControversyMapResponse(BaseModel):
    conflicts: list[ConflictClusterSchema] = Field(default_factory=list)
    gaps: list[GapCandidateSchema] = Field(default_factory=list)
    overview: str = ""
    scope: dict = Field(default_factory=dict)
