"""Graph API schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GraphNodeResponse(BaseModel):
    id: str
    label: str
    name: str


class GraphEdgeResponse(BaseModel):
    source: str
    target: str
    type: str


class GraphMetaResponse(BaseModel):
    library_id: str | None = None
    node_count: int = 0
    edge_count: int = 0


class PaperSubgraphResponse(BaseModel):
    paper: dict[str, str]
    nodes: list[GraphNodeResponse] = Field(default_factory=list)
    edges: list[GraphEdgeResponse] = Field(default_factory=list)
    meta: GraphMetaResponse


class PaperGraphNodeResponse(BaseModel):
    id: str
    paper_id: str
    title: str = ""
    year: int | None = None
    library_id: str | None = None


class PaperGraphEdgeResponse(BaseModel):
    source: str
    target: str
    type: str
    weight: int = 1
    shared_items: list[str] = Field(default_factory=list)


class PaperGraphMetaResponse(BaseModel):
    project_id: str = ""
    library_id: str = ""
    node_count: int = 0
    edge_count: int = 0
    scoped_paper_count: int = 0
    relation_counts: dict[str, int] = Field(default_factory=dict)


class PaperGraphResponse(BaseModel):
    nodes: list[PaperGraphNodeResponse] = Field(default_factory=list)
    edges: list[PaperGraphEdgeResponse] = Field(default_factory=list)
    meta: PaperGraphMetaResponse
