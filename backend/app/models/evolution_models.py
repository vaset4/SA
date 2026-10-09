"""研究演化地图模型。"""

from __future__ import annotations

from typing import Literal, TypedDict

from app.models.project_models import ResolvedProjectScope

EvolutionViewMode = Literal["paper", "method", "task"]
EvolutionNodeType = Literal["paper", "method", "task", "dataset"]

PAPER_RELATIONS = {"CITES", "EXTENDS"}
OBJECT_RELATIONS = {"USES_METHOD", "STUDIES_TASK", "USES_DATASET"}
ALLOWED_EVOLUTION_RELATIONS = PAPER_RELATIONS | OBJECT_RELATIONS


class EvolutionQuery(TypedDict):
    view_mode: EvolutionViewMode
    paper_id: str
    method: str
    task: str
    dataset: str
    from_year: int | None
    to_year: int | None
    depth: int
    project_id: str
    resolved_scope: ResolvedProjectScope | None


class EvolutionNode(TypedDict):
    id: str
    label: str
    node_type: EvolutionNodeType
    year: int | None
    paper_id: str
    importance_score: float
    group: str


class EvolutionEdge(TypedDict):
    source: str
    target: str
    relation: str
    year: int | None
    weight: float


class TimelineEvent(TypedDict):
    year: int
    title: str
    event_type: str
    paper_id: str
    summary: str


class EvolutionFocus(TypedDict):
    key: str
    label: str
    node_type: str
    view_mode: EvolutionViewMode


class EvolutionGraphPayload(TypedDict):
    focus: EvolutionFocus
    timeline: list[TimelineEvent]
    nodes: list[EvolutionNode]
    edges: list[EvolutionEdge]
    stats: dict[str, object]
    filters: dict[str, object]
