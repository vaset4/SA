"""研究演化地图 schema。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class EvolutionNodeSchema(BaseModel):
    id: str
    label: str
    node_type: str
    year: Optional[int] = None
    paper_id: str = ""
    importance_score: float = 0
    group: str = ""


class EvolutionEdgeSchema(BaseModel):
    source: str
    target: str
    relation: str
    year: Optional[int] = None
    weight: float = 1


class TimelineEventSchema(BaseModel):
    year: int
    title: str
    event_type: str
    paper_id: str
    summary: str = ""


class EvolutionFocusSchema(BaseModel):
    key: str = ""
    label: str = ""
    node_type: str = ""
    view_mode: str = "paper"


class EvolutionGraphResponse(BaseModel):
    focus: EvolutionFocusSchema
    timeline: list[TimelineEventSchema] = Field(default_factory=list)
    nodes: list[EvolutionNodeSchema] = Field(default_factory=list)
    edges: list[EvolutionEdgeSchema] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    filters: dict = Field(default_factory=dict)


class EvolutionEntryResponse(BaseModel):
    focus: EvolutionFocusSchema
    recommended_filters: dict = Field(default_factory=dict)
    target_url: str = ""
