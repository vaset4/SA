"""Neo4j 研究对象图谱相关模型。"""

from __future__ import annotations

from typing import Literal, TypedDict

GraphNodeLabel = Literal[
    "Paper",
    "Author",
    "Venue",
    "Method",
    "Dataset",
    "Metric",
    "Claim",
    "Limitation",
    "Task",
]
GraphRelationType = Literal[
    "AUTHORED_BY",
    "PUBLISHED_IN",
    "USES_METHOD",
    "USES_DATASET",
    "REPORTS_METRIC",
    "MAKES_CLAIM",
    "HAS_LIMITATION",
    "CITES",
    "COMPARES_WITH",
    "STUDIES_TASK",
    "EXTENDS",
    "ADDRESSES_LIMITATION",
]

GRAPH_NODE_LABELS: tuple[GraphNodeLabel, ...] = (
    "Paper",
    "Author",
    "Venue",
    "Method",
    "Dataset",
    "Metric",
    "Claim",
    "Limitation",
    "Task",
)
GRAPH_RELATION_TYPES: tuple[GraphRelationType, ...] = (
    "AUTHORED_BY",
    "PUBLISHED_IN",
    "USES_METHOD",
    "USES_DATASET",
    "REPORTS_METRIC",
    "MAKES_CLAIM",
    "HAS_LIMITATION",
    "CITES",
    "COMPARES_WITH",
    "STUDIES_TASK",
    "EXTENDS",
    "ADDRESSES_LIMITATION",
)


class GraphNodePayload(TypedDict, total=False):
    id: str
    label: GraphNodeLabel
    name: str
    library_id: str
    properties: dict[str, object]


class GraphEdgePayload(TypedDict, total=False):
    source: str
    target: str
    type: GraphRelationType
    paper_id: str
    library_id: str
    source_type: str
    properties: dict[str, object]


class ResearchObjectExtraction(TypedDict):
    authors: list[str]
    venue: str
    methods: list[str]
    datasets: list[str]
    metrics: list[str]
    claims: list[str]
    limitations: list[str]
    tasks: list[str]


class ResearchGraphPayload(TypedDict):
    paper: GraphNodePayload
    nodes: list[GraphNodePayload]
    edges: list[GraphEdgePayload]
