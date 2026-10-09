"""Graph evidence API schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GraphEvidenceNodeSchema(BaseModel):
    id: str
    label: str
    node_type: str
    paper_id: str | None = None
    entity_key: str | None = None
    display_meta: dict[str, object] = Field(default_factory=dict)


class GraphEvidenceEdgeSchema(BaseModel):
    source: str
    target: str
    relation: str
    paper_id: str | None = None
    weight: float | None = None
    evidence_role: str | None = None


class GraphEvidencePathSchema(BaseModel):
    path_id: str
    path_type: str
    summary: str
    support_paper_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    nodes: list[GraphEvidenceNodeSchema] = Field(default_factory=list)
    edges: list[GraphEvidenceEdgeSchema] = Field(default_factory=list)


class EvidenceBundleSchema(BaseModel):
    paper_sources: list[dict[str, str]] = Field(default_factory=list)
    chunk_citations: list[dict[str, object]] = Field(default_factory=list)
    graph_evidence: list[GraphEvidencePathSchema] = Field(default_factory=list)
    evidence_summary: str = ""


class EvidencePathRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    query: str = ""
    path_type: str = "support"
    max_hops: int = Field(default=3, ge=1, le=4)
