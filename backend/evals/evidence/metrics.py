"""只基于冻结真值和真实阶段观察的确定性检索指标。作者：xxx。"""

from __future__ import annotations

import math
from collections.abc import Mapping

from evals.contracts import MetricResult
from evals.evidence.alignment import (
    EvidenceDataError, align_observed_item, covered_span_ids, validate_benchmark,
    validate_observed_child, validate_paper_requirements, validate_ranking_judgments,
)
from evals.evidence.contracts import (
    EffectiveScope, EvidenceBenchmark, EvidenceGradingPolicy, EvidenceGroup, IndexSnapshot, MappedRange, ObservedItem,
    RankedUnit, RankingPolicy, StageGrade, StageObservation, ranking_unit_key,
)

RANKED_STAGES = frozenset(("query_raw", "query_filtered", "merged_hits", "ranked_hits"))
CONTEXT_STAGES = frozenset(("claim_input", "final_input"))
RANKING_METRICS = ("mrr", "ndcg_at_k")
COVERAGE_METRICS = ("critical_evidence_recall", "section_recall", "cross_paper_coverage")


def group_hits(groups: tuple[EvidenceGroup, ...], covered: frozenset[str]) -> frozenset[str]:
    """参数为证据组和完整命中片段；返回至少一份完整证明满足的独立组。"""
    return frozenset(group.group_id for group in groups
                     if any(set(proof.span_ids).issubset(covered) for proof in group.proofs))


def ratio_metric(name: str, numerator: int, denominator: int) -> MetricResult:
    """参数为指标名称与真实计数；返回比例，零目标返回不适用而非满分。"""
    if type(numerator) is not int or type(denominator) is not int or not 0 <= numerator <= denominator:
        raise EvidenceDataError("invalid_input", "metric_counts_invalid")
    if denominator == 0:
        return MetricResult(name=name, status="not_measured", reason="no_applicable_targets")
    return MetricResult(
        name=name, status="measured", reason="verified_observation",
        value=numerator / denominator, numerator=numerator, denominator=denominator,
    )


def evidence_group_metrics(
    groups: tuple[EvidenceGroup, ...], covered: frozenset[str], *,
    name: str = "evidence_group_recall_at_k",
) -> tuple[MetricResult, MetricResult]:
    """参数为目标组、已覆盖片段及阶段指标名；独立返回总体和关键证据覆盖。"""
    hits = group_hits(groups, covered)
    critical = {group.group_id for group in groups if group.importance == "critical"}
    return (
        ratio_metric(name, len(hits), len(groups)),
        ratio_metric("critical_evidence_recall", len(hits & critical), len(critical)),
    )


def scope_metric(actual_ids: tuple[str | None, ...] | None, scope: EffectiveScope) -> MetricResult:
    """参数为所有实际论文身份及生效范围；缺失、不适用和任一确定越界分别报告。"""
    if scope.status == "missing":
        return MetricResult(name="scope_adherence", status="missing", reason="effective_scope_missing")
    if scope.status == "unconstrained":
        return MetricResult(name="scope_adherence", status="not_measured", reason="no_scope_constraint")
    known_ids = {identity for identity in actual_ids or () if isinstance(identity, str) and identity.strip()}
    violations = known_ids - set(scope.allowed_paper_ids)
    if violations:
        return MetricResult(name="scope_adherence", status="measured", reason="out_of_scope_evidence", value=0.0)
    if any(identity is not None and (not isinstance(identity, str) or not identity.strip()) for identity in actual_ids or ()):
        return MetricResult(name="scope_adherence", status="invalid_input", reason="observed_paper_identity_invalid")
    if actual_ids is None or any(identity is None for identity in actual_ids):
        return MetricResult(name="scope_adherence", status="missing", reason="observed_paper_identity_missing")
    return MetricResult(name="scope_adherence", status="measured", reason="within_effective_scope", value=1.0)


def section_metric(benchmark: EvidenceBenchmark, ranges: tuple[MappedRange, ...]) -> MetricResult:
    """参数为显式章节目标与真实映射；返回粗粒度章节召回，不替代原文证据覆盖。"""
    if benchmark.target_sections is None:
        return MetricResult(name="section_recall", status="missing", reason="section_targets_missing")
    sources = {source.snapshot_id: source for source in benchmark.sources}
    blocks = {(source.snapshot_id, block.block_id): block for source in benchmark.sources for block in source.blocks}
    available = {(source.paper_id, block.section_path) for source in benchmark.sources for block in source.blocks}
    targets = {(target.paper_id, target.section_path) for target in benchmark.target_sections}
    if {paper_id for paper_id, _ in targets} - {source.paper_id for source in benchmark.sources}:
        return MetricResult(name="section_recall", status="missing", reason="section_source_snapshot_missing")
    if not targets.issubset(available):
        return MetricResult(name="section_recall", status="invalid_input", reason="section_target_not_in_source")
    actual = {(sources[item.source_snapshot_id].paper_id, blocks[(item.source_snapshot_id, item.block_id)].section_path)
              for item in ranges}
    return ratio_metric("section_recall", len(targets & actual), len(targets))


def cross_paper_metric(benchmark: EvidenceBenchmark, covered: frozenset[str]) -> MetricResult:
    """参数为各论文显式证明要求及命中片段；每篇全部要求满足后才计入覆盖。"""
    if len(benchmark.target_paper_ids) <= 1:
        return MetricResult(name="cross_paper_coverage", status="not_measured", reason="not_a_cross_paper_case")
    try:
        validate_paper_requirements(benchmark)
    except EvidenceDataError as exc:
        return MetricResult(name="cross_paper_coverage", status=exc.status, reason=exc.reason)
    satisfied = sum(len(group_hits(requirement.groups, covered)) == len(requirement.groups)
                    for requirement in benchmark.paper_requirements or ())
    return ratio_metric("cross_paper_coverage", satisfied, len(benchmark.target_paper_ids))


def _validate_ranking_input(
    ranked_units: tuple[RankedUnit, ...], judgments: Mapping[str, int] | None, *,
    policy: RankingPolicy, universe: tuple[str, ...] | None,
) -> Mapping[str, int]:
    """参数为实际排名、标注、固定策略及索引全集；验证完整性后返回原标注。"""
    if judgments is None:
        raise EvidenceDataError("missing", "ranking_judgments_missing")
    if universe is None:
        raise EvidenceDataError("missing", "ranking_universe_missing")
    if len(universe) != len(set(universe)):
        raise EvidenceDataError("invalid_input", "ranking_universe_has_duplicates")
    if set(judgments) - set(universe):
        raise EvidenceDataError("invalid_input", "ranking_judgment_unit_not_in_index")
    if set(universe) - set(judgments):
        raise EvidenceDataError("missing", "ranking_judgments_incomplete")
    if any(type(value) is not int or not 0 <= value < len(policy.gains) for value in judgments.values()):
        raise EvidenceDataError("invalid_input", "ranking_relevance_out_of_range")
    ranks = tuple(item.rank for item in ranked_units)
    if any(type(rank) is not int or rank <= 0 for rank in ranks) or any(right <= left for left, right in zip(ranks, ranks[1:])):
        raise EvidenceDataError("invalid_input", "ranking_positions_invalid")
    if any(item.unit_key not in judgments for item in ranked_units):
        raise EvidenceDataError("missing", "ranking_contains_unjudged_units")
    return judgments


def ranking_metrics(
    ranked_units: tuple[RankedUnit, ...], judgments: Mapping[str, int] | None, *,
    policy: RankingPolicy, universe: tuple[str, ...] | None = None,
) -> tuple[MetricResult, MetricResult]:
    """参数为真实排名、独立完整标注及固定策略；返回 top-k 内的 MRR 和 nDCG。"""
    labels = _validate_ranking_input(ranked_units, judgments, policy=policy, universe=universe)
    gain_scale = policy.gains[-1]
    ideal_gains = sorted((policy.gains[level] / gain_scale for level in labels.values()), reverse=True)
    ideal = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains[:policy.top_k], start=1))
    if ideal == 0.0:
        return (
            MetricResult(name="mrr", status="not_measured", reason="no_relevant_ranking_targets"),
            MetricResult(name="ndcg_at_k", status="not_measured", reason="no_relevant_ranking_targets"),
        )
    seen: set[str] = set()
    reciprocal_rank = 0.0
    dcg = 0.0
    for item in ranked_units:
        if item.rank > policy.top_k or item.unit_key in seen:
            continue
        seen.add(item.unit_key)
        level = labels[item.unit_key]
        if level > 0 and reciprocal_rank == 0.0:
            reciprocal_rank = 1.0 / item.rank
        dcg += (policy.gains[level] / gain_scale) / math.log2(item.rank + 1)
    return (
        MetricResult(name="mrr", status="measured", reason="declared_ranking_policy", value=reciprocal_rank),
        MetricResult(name="ndcg_at_k", status="measured", reason="declared_ranking_policy", value=dcg / ideal),
    )


def _group_metric_name(observation: StageObservation) -> str:
    """参数为真实阶段；返回与排名或无排名上下文匹配的指标名称。"""
    if observation.stage in RANKED_STAGES:
        return "evidence_group_recall_at_k"
    if observation.stage in CONTEXT_STAGES:
        return "evidence_group_context_coverage"
    return "evidence_group_pool_coverage"


def _selected_items(benchmark: EvidenceBenchmark, observation: StageObservation) -> tuple[ObservedItem, ...]:
    """参数为真值和真实阶段；检查完整观察，排名阶段按预声明 k 选取实际位置。"""
    if observation.status != "complete":
        raise EvidenceDataError("missing", observation.reason or f"stage_observation_{observation.status}")
    if observation.items is None:
        raise EvidenceDataError("missing", "stage_items_missing")
    if observation.stage not in RANKED_STAGES:
        return observation.items
    if benchmark.ranking_policy is None:
        raise EvidenceDataError("missing", "ranking_policy_missing")
    ranks = tuple(item.rank for item in observation.items)
    if any(rank is None for rank in ranks):
        raise EvidenceDataError("missing", "ranking_positions_missing")
    if any(right <= left for left, right in zip(ranks, ranks[1:]) if right is not None and left is not None):
        raise EvidenceDataError("invalid_input", "ranking_positions_invalid")
    return tuple(item for item in observation.items if item.rank is not None and item.rank <= benchmark.ranking_policy.top_k)


def _ranked_observation_units(observation: StageObservation, index: IndexSnapshot) -> tuple[RankedUnit, ...]:
    """参数为真实阶段和索引全集；验证各 child 正文及身份，再保留真实排名键。"""
    items = observation.items or ()
    if any(item.paper_id is None or item.chunk_id is None or item.rank is None for item in items):
        raise EvidenceDataError("missing", "ranking_item_identity_missing")
    for item in items:
        validate_observed_child(item, index)
    return tuple(RankedUnit(rank=item.rank, unit_key=ranking_unit_key(item.paper_id, item.chunk_id))
                 for item in items if item.paper_id is not None and item.chunk_id is not None and item.rank is not None)


def _ranking_stage_metrics(benchmark: EvidenceBenchmark, observation: StageObservation) -> tuple[MetricResult, ...]:
    """参数为真值和单次阶段；仅使用匹配查询、索引版本与排序单位的完整标注。"""
    if observation.stage not in RANKED_STAGES:
        return tuple(MetricResult(name=name, status="not_measured", reason="stage_has_no_ranking") for name in RANKING_METRICS)
    policy = benchmark.ranking_policy
    if policy is None:
        raise EvidenceDataError("missing", "ranking_policy_missing")
    if policy.stage != observation.stage:
        return tuple(MetricResult(name=name, status="not_measured", reason="ranking_policy_stage_not_applicable") for name in RANKING_METRICS)
    if observation.query is None:
        raise EvidenceDataError("missing", "ranking_query_missing")
    index = next((index for index in benchmark.index_snapshots if index.index_snapshot_id == observation.index_snapshot_id), None)
    if index is None:
        raise EvidenceDataError("missing", "ranking_index_snapshot_missing")
    if index.retrieval_backend != observation.retrieval_backend:
        raise EvidenceDataError("invalid_input", "ranking_backend_mismatch")
    candidates = tuple(judgment for judgment in benchmark.ranking_judgments if
                       (judgment.query, judgment.stage, judgment.index_snapshot_id)
                       == (observation.query, observation.stage, index.index_snapshot_id))
    if not candidates:
        raise EvidenceDataError("missing", "ranking_judgments_missing")
    if len(candidates) != 1:
        raise EvidenceDataError("invalid_input", "ranking_judgments_ambiguous")
    judgments = candidates[0]
    validate_ranking_judgments(judgments, index, policy=policy)
    ranked = _ranked_observation_units(observation, index)
    universe = tuple(ranking_unit_key(item.paper_id, item.chunk_id) for item in index.units)
    return ranking_metrics(ranked, {item.unit_key: item.relevance for item in judgments.labels}, policy=policy, universe=universe)


def _stage_scope(observation: StageObservation) -> tuple[MetricResult, tuple[str, ...]]:
    """参数为全阶段观察；范围检查包含 k 以外实际命中，确定越界不能被缺失或均值隐藏。"""
    actual = tuple(item.paper_id for item in observation.items) if observation.items is not None else None
    metric = scope_metric(actual, observation.scope)
    violations = tuple(sorted({identity for identity in actual or () if identity is not None}
                              - set(observation.scope.allowed_paper_ids))) if observation.scope.status == "constrained" else ()
    if observation.status != "complete" and metric.value != 0.0 and metric.status == "measured":
        metric = MetricResult(name="scope_adherence", status="missing", reason="scope_observation_incomplete")
    return metric, violations


def _coverage_stage_metrics(
    benchmark: EvidenceBenchmark, observation: StageObservation, *, items: tuple[ObservedItem, ...],
) -> tuple[tuple[MetricResult, ...], frozenset[str]]:
    """参数为已校验真值、阶段和实际选中正文；返回完整证明覆盖及独立章节/跨论文诊断。"""
    indexes = {index.index_snapshot_id: index for index in benchmark.index_snapshots}
    identity = observation.index_snapshot_id
    if observation.stage in RANKED_STAGES and identity in indexes and indexes[identity].retrieval_backend != observation.retrieval_backend:
        raise EvidenceDataError("invalid_input", "observed_backend_mismatch")
    ranges = tuple(interval for item in items for interval in align_observed_item(
        item, benchmark, index_snapshot_id=identity,
    ))
    covered = covered_span_ids(benchmark.spans, ranges)
    group_metrics = evidence_group_metrics(benchmark.groups, covered, name=_group_metric_name(observation))
    return (*group_metrics, section_metric(benchmark, ranges), cross_paper_metric(benchmark, covered)), covered


def grade_stage(
    benchmark: EvidenceBenchmark, observation: StageObservation, *,
    policy: EvidenceGradingPolicy = "reviewed",
) -> StageGrade:
    """参数为真值、实际阶段及审核策略；只测事前启用项，保留原匹配公式和缺失状态。"""
    if policy not in {"reviewed", "candidate_diagnostic"}:
        raise ValueError(f"unsupported evidence grading policy: {policy}")
    grade = _grade_stage(benchmark, observation, policy=policy)
    selection = benchmark.measurement_policy
    if selection is None:
        return grade
    enabled = frozenset(selection.enabled_metrics)
    measurements = tuple(
        metric if metric.name in enabled else MetricResult(
            name=metric.name, status="not_measured", reason="metric_not_enabled",
        ) for metric in grade.measurements
    )
    return grade.model_copy(update={"measurements": measurements})


def _grade_stage(
    benchmark: EvidenceBenchmark, observation: StageObservation, *, policy: EvidenceGradingPolicy,
) -> StageGrade:
    """参数为冻结真值和一次真实阶段；返回分项四状态评分，其他程序异常继续抛出。"""
    scope, violations = _stage_scope(observation)
    group_name = _group_metric_name(observation)
    names = (group_name, *COVERAGE_METRICS, *RANKING_METRICS)
    try:
        validate_benchmark(benchmark, include_ranking=False)
        items = _selected_items(benchmark, observation)
    except EvidenceDataError as exc:
        measurements = tuple(MetricResult(name=name, status=exc.status, reason=exc.reason) for name in names)
        return StageGrade(measurements=(*measurements, scope), scope_violations=violations)
    expected_review = "reviewed" if policy == "reviewed" else "candidate"
    if benchmark.review_status != expected_review:
        reason = f"evidence_case_not_{expected_review}"
        measurements = tuple(MetricResult(name=name, status="not_measured", reason=reason) for name in names)
        return StageGrade(measurements=(*measurements, scope), scope_violations=violations)
    covered: frozenset[str] = frozenset()
    try:
        coverage, covered = _coverage_stage_metrics(benchmark, observation, items=items)
    except EvidenceDataError as exc:
        coverage = tuple(MetricResult(name=name, status=exc.status, reason=exc.reason) for name in (group_name, *COVERAGE_METRICS))
    try:
        ranking = _ranking_stage_metrics(benchmark, observation)
    except EvidenceDataError as exc:
        ranking = tuple(MetricResult(name=name, status=exc.status, reason=exc.reason) for name in RANKING_METRICS)
    return StageGrade(
        measurements=(*coverage, *ranking, scope), covered_span_ids=tuple(sorted(covered)),
        hit_group_ids=tuple(sorted(group_hits(benchmark.groups, covered))), scope_violations=violations,
    )
