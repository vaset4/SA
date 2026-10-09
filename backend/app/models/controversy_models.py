"""争议与 gap 结构化模型。"""

from __future__ import annotations

from typing import Literal, TypedDict

from app.models.project_models import ResolvedProjectScope

ClaimDirection = Literal["better", "worse", "similar", "uncertain"]
ConflictType = Literal["metric_direction_conflict", "method_tradeoff_conflict", "replication_mismatch"]
GapType = Literal["dataset_gap", "evaluation_gap", "limitation_unresolved", "benchmark_concentration"]


class ClaimTriple(TypedDict):
    paper_id: str
    task: str
    method: str
    dataset: str
    metric: str
    claim_direction: ClaimDirection
    claim_text: str
    evidence_excerpt: str


class ConflictCluster(TypedDict):
    cluster_id: str
    topic_key: str
    papers: list[str]
    conflict_reason: str
    conflict_type: ConflictType
    support_claims: list[ClaimTriple]
    summary: str


class GapCandidate(TypedDict):
    gap_id: str
    gap_type: GapType
    topic_key: str
    support_paper_ids: list[str]
    evidence_items: list[str]
    severity: float
    summary: str
    next_research_question: str


class ControversyMapPayload(TypedDict):
    conflicts: list[ConflictCluster]
    gaps: list[GapCandidate]
    overview: str
    scope: dict[str, object]


class ControversyMapQuery(TypedDict):
    paper_ids: list[str]
    topic: str
    include_gaps: bool
    include_conflicts: bool
    project_id: str
    resolved_scope: ResolvedProjectScope | None
