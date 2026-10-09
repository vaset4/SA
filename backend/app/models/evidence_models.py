"""证据路径与可解释回答模型。"""

from __future__ import annotations

from typing import Literal, TypedDict


class PaperSourceRef(TypedDict):
    paper_id: str


class ChunkCitationRef(TypedDict, total=False):
    paper_id: str
    chunk_id: str
    section: str
    page_num: int
    excerpt: str


class GraphEvidenceNode(TypedDict, total=False):
    id: str
    label: str
    node_type: str
    paper_id: str
    entity_key: str
    display_meta: dict[str, object]


class GraphEvidenceEdge(TypedDict, total=False):
    source: str
    target: str
    relation: str
    paper_id: str
    weight: float
    evidence_role: str


GraphEvidencePathType = Literal["support", "comparison", "citation_bridge"]


class GraphEvidencePath(TypedDict):
    path_id: str
    path_type: GraphEvidencePathType
    summary: str
    support_paper_ids: list[str]
    confidence: float
    nodes: list[GraphEvidenceNode]
    edges: list[GraphEvidenceEdge]


class EvidenceBundle(TypedDict):
    paper_sources: list[PaperSourceRef]
    chunk_citations: list[ChunkCitationRef]
    graph_evidence: list[GraphEvidencePath]
    evidence_summary: str
