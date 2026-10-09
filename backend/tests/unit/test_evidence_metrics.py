"""验证完整证明、独立排序、范围和四种测量状态。作者：xxx。"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from evals.contracts import MetricResult, snapshot_hash
from evals.evidence.alignment import EvidenceDataError, build_index_mapping, locate_span
from evals.evidence.contracts import (
    EffectiveScope, EvidenceBenchmark, EvidenceGroup, EvidenceProof, EvidenceSpan,
    IndexUnit, ObservedItem, PaperRequirement, RankedUnit, RankingJudgments,
    RankingPolicy, RelevanceJudgment, SectionTarget, SourceBlock, StageObservation,
    make_index_snapshot, make_source_snapshot, ranking_unit_key,
)
from evals.evidence.metrics import (
    cross_paper_metric, evidence_group_metrics, grade_stage, group_hits,
    ranking_metrics, ratio_metric, scope_metric,
)


def _group(group_id: str, *, spans: tuple[str, ...], importance: str = "critical") -> EvidenceGroup:
    """参数为组标识、完整证明片段及等级；返回测试证据组。"""
    return EvidenceGroup.model_validate({"group_id": group_id, "importance": importance, "proofs": [{"span_ids": spans}]})


@pytest.fixture
def benchmark() -> EvidenceBenchmark:
    """无参数；返回两篇论文、三条真实测试索引及完整独立排序标注。"""
    sources = tuple(make_source_snapshot(
        paper_id=paper, pdf_sha256=snapshot_hash(paper), artifact_manifest_snapshot="{}",
        blocks=(SourceBlock(block_id="body", page_num=1, section_path="Methods", text=text),),
    ) for paper, text in (("a", "critical evidence; background evidence; unrelated material"), ("b", "complementary evidence")))
    entries = ((sources[0], "critical", "critical evidence"), (sources[0], "background", "background evidence"),
               (sources[0], "unrelated", "unrelated material"), (sources[1], "other", "complementary evidence"))
    mappings = tuple(build_index_mapping(source, text, index_snapshot_id="pending", paper_id=source.paper_id, chunk_id=chunk)
                     for source, chunk, text in entries)
    units = tuple(IndexUnit(paper_id=source.paper_id, chunk_id=chunk, text_snapshot=text, text_sha256=snapshot_hash(text))
                  for source, chunk, text in entries)
    index = make_index_snapshot(source_snapshot_ids=tuple(source.snapshot_id for source in sources), retrieval_backend="fixture", units=units, mappings=mappings)
    spans = []
    for source, span_id, quote in (entries[0], entries[1], entries[3]):
        block_id, start, end = locate_span(source, quote)
        spans.append(EvidenceSpan(span_id=span_id, source_snapshot_id=source.snapshot_id, block_id=block_id, start=start, end=end, quote=quote, quote_sha256=snapshot_hash(quote)))
    groups = (_group("key", spans=("critical",)), _group("background", spans=("background",), importance="supporting"))
    policy = RankingPolicy(stage="query_raw", top_k=2)
    judgments = RankingJudgments(
        judgment_id="relevance-v1", index_snapshot_id=index.index_snapshot_id, stage="query_raw", query="test query",
        labels=tuple(RelevanceJudgment(unit_key=ranking_unit_key(unit.paper_id, unit.chunk_id), relevance=level)
                     for unit, level in zip(units, (3, 1, 0, 0))),
    )
    return EvidenceBenchmark(
        dataset_version="fixture-v1", task_family="single-paper", review_status="reviewed",
        sources=sources, spans=tuple(spans), groups=groups, target_paper_ids=("a",),
        target_sections=(SectionTarget(paper_id="a", section_path="Methods"),),
        index_snapshots=(index,), ranking_policy=policy, ranking_judgments=(judgments,),
    )


def _observation(benchmark: EvidenceBenchmark, chunks: tuple[str, ...], *, stage: str = "query_raw") -> StageObservation:
    """参数为测试真值、实际返回 chunk 顺序和阶段；返回独立冻结的真实观察。"""
    index = benchmark.index_snapshots[0]
    by_id = {unit.chunk_id: unit for unit in index.units}
    items = tuple(ObservedItem(
        paper_id=by_id[chunk].paper_id, chunk_id=chunk, payload_kind="child",
        text_snapshot=by_id[chunk].text_snapshot, rank=rank,
    ) for rank, chunk in enumerate(chunks, start=1))
    return StageObservation.model_validate({
        "stage": stage, "call_id": "call-a", "scope_id": "scope-a", "query": "test query",
        "retrieval_backend": "fixture", "status": "complete", "items": items,
        "scope": EffectiveScope(status="constrained", allowed_paper_ids=("a",)), "index_snapshot_id": index.index_snapshot_id,
    })


def _scores(benchmark: EvidenceBenchmark, observation: StageObservation) -> dict[str, MetricResult]:
    """参数为真值和阶段观察；返回按名称索引的测量结果，供测试检查分项状态。"""
    return {metric.name: metric for metric in grade_stage(benchmark, observation).measurements}


def test_partial_proof_and_duplicate_spans_do_not_complete_group() -> None:
    """无参数；验证多片段证明缺一不满足，重复片段不能补足证明。"""
    group = _group("method", spans=("a", "b"))
    assert group_hits((group,), frozenset(("a", "a"))) == frozenset()
    assert group_hits((group,), frozenset(("a", "b"))) == {"method"}


def test_candidate_diagnostic_preserves_review_and_existing_formula(benchmark: EvidenceBenchmark) -> None:
    """参数为测试原文；候选显式诊断沿用召回公式，默认评分和原审核状态不改变。"""
    candidate = benchmark.model_copy(update={"review_status": "candidate"})
    observed = _observation(candidate, ("critical",))
    default = grade_stage(candidate, observed)
    diagnostic = grade_stage(candidate, observed, policy="candidate_diagnostic")
    by_name = {metric.name: metric for metric in diagnostic.measurements}
    assert by_name["evidence_group_recall_at_k"].value == 0.5
    assert candidate.review_status == "candidate"
    assert next(metric for metric in default.measurements if metric.name == "evidence_group_recall_at_k").status == "not_measured"


def test_enabled_missing_and_disabled_metrics_are_distinct(benchmark: EvidenceBenchmark) -> None:
    """参数为测试原文；事前未启用排序时不打分，已启用但缺标签必须显示 missing。"""
    from evals.evidence.contracts import CORE_EVIDENCE_METRICS, EvidenceMeasurementPolicy

    candidate = benchmark.model_copy(update={
        "review_status": "candidate", "ranking_judgments": (),
        "measurement_policy": EvidenceMeasurementPolicy(enabled_metrics=CORE_EVIDENCE_METRICS),
    })
    observation = _observation(candidate, ("critical",))
    scores = {metric.name: metric for metric in grade_stage(candidate, observation, policy="candidate_diagnostic").measurements}
    assert scores["mrr"].status == "not_measured"
    assert scores["mrr"].reason == "metric_not_enabled"
    enabled = candidate.model_copy(update={"measurement_policy": EvidenceMeasurementPolicy(
        enabled_metrics=(*CORE_EVIDENCE_METRICS, "mrr"),
    )})
    scores = {metric.name: metric for metric in grade_stage(enabled, observation, policy="candidate_diagnostic").measurements}
    assert scores["mrr"].status == "missing"
    assert scores["mrr"].value is None


def test_alternative_complete_proofs_count_group_once() -> None:
    """无参数；验证等价完整证明各自满足，但同时满足仍只计一个组。"""
    group = EvidenceGroup(group_id="either", importance="critical", proofs=(EvidenceProof(span_ids=("a", "b")), EvidenceProof(span_ids=("c",))))
    assert group_hits((group,), frozenset(("a",))) == frozenset()
    assert group_hits((group,), frozenset(("c",))) == {"either"}
    assert group_hits((group,), frozenset(("a", "b", "c"))) == {"either"}


def test_unrelated_same_section_retrieval_has_zero_group_recall(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证同篇同章的无关正文不产生证据组命中。"""
    scores = _scores(benchmark, _observation(benchmark, ("unrelated",)))
    assert scores["evidence_group_recall_at_k"].value == 0.0
    assert scores["evidence_group_recall_at_k"].denominator == 2
    assert scores["critical_evidence_recall"].value == 0.0
    assert scores["section_recall"].value == 1.0
    assert scores["mrr"].value == 0.0
    assert scores["ndcg_at_k"].value == 0.0


def test_background_does_not_hide_critical_miss_and_duplicates_do_not_add_hits(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证背景证据可部分得分，重复证据不能掩盖关键漏召回。"""
    scores = _scores(benchmark, _observation(benchmark, ("background", "background")))
    assert scores["evidence_group_recall_at_k"].value == 0.5
    assert scores["evidence_group_recall_at_k"].numerator == 1
    assert scores["critical_evidence_recall"].value == 0.0
    assert scores["critical_evidence_recall"].denominator == 1


def test_empty_success_and_missing_or_failed_observations_are_distinct(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证成功空结果是零分，缺失或失败没有伪造数值。"""
    complete = _observation(benchmark, ())
    assert _scores(benchmark, complete)["evidence_group_recall_at_k"].value == 0.0
    for observation in (complete.model_copy(update={"items": None}), complete.model_copy(update={"status": "failed"})):
        metric = _scores(benchmark, observation)["evidence_group_recall_at_k"]
        assert metric.status == "missing"
        assert metric.value is None


def test_invalid_truth_does_not_shrink_denominator(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证坏证明使指标 invalid_input，不只保留容易命中的组。"""
    changed = benchmark.model_copy(update={"groups": (*benchmark.groups, _group("bad", spans=("absent",)))})
    scores = _scores(changed, _observation(benchmark, ("critical", "background")))
    assert scores["evidence_group_recall_at_k"].status == "invalid_input"
    assert scores["evidence_group_recall_at_k"].denominator is None
    assert scores["scope_adherence"].value == 1.0


def test_no_critical_or_group_targets_are_not_applicable() -> None:
    """无参数；验证没有目标时不能使用默认满分或零分代替不适用。"""
    overall, critical = evidence_group_metrics((), frozenset())
    assert overall.status == critical.status == "not_measured"
    assert overall.value is critical.value is None
    overall, critical = evidence_group_metrics((_group("background", spans=("x",), importance="supporting"),), frozenset(("x",)))
    assert overall.value == 1.0
    assert critical.status == "not_measured"


@pytest.mark.parametrize("numerator,denominator", [(True, 1), (1, False), (1.0, 2), (-1, 2), (2, 1)])
def test_ratio_rejects_invalid_counts(numerator: object, denominator: object) -> None:
    """参数为非法计数；验证布尔、浮点、负数和超过分母的计数被拒绝。"""
    with pytest.raises(EvidenceDataError, match="metric_counts_invalid"):
        ratio_metric("test", numerator, denominator)


@pytest.mark.parametrize("gains", [(0.0, float("nan")), (0.0, float("inf")), (0.0, 1.0, 1.0), (0.0, True), (1.0, 2.0)])
def test_ranking_policy_rejects_invalid_gain_values(gains: tuple[object, ...]) -> None:
    """参数为非法增益；验证 NaN、无穷、布尔和非递增等级不会进入计算。"""
    with pytest.raises(ValidationError):
        RankingPolicy(stage="query_raw", top_k=2, gains=gains)


@pytest.mark.parametrize("rank", [True, 1.5, "1", 0, -1])
def test_rank_requires_a_positive_integer(rank: object) -> None:
    """参数为非法排名；验证真实位置不会从布尔、字符串或小数转换而来。"""
    with pytest.raises(ValidationError):
        RankedUnit(rank=rank, unit_key="a")


def test_ranking_duplicate_units_keep_original_positions_without_gain() -> None:
    """无参数；验证重复单位占原排名但不重复增益，首个相关单位沿用真实位置。"""
    policy = RankingPolicy(stage="query_raw", top_k=4, gains=(0.0, 1.0, 3.0))
    units = tuple(RankedUnit(rank=rank, unit_key=key) for rank, key in enumerate(("a", "a", "b", "c"), start=1))
    mrr, ndcg = ranking_metrics(units, {"a": 0, "b": 1, "c": 2}, policy=policy, universe=("a", "b", "c"))
    assert mrr.value == pytest.approx(1 / 3)
    assert ndcg.value == pytest.approx((1 / math.log2(4) + 3 / math.log2(5)) / (3 + 1 / math.log2(3)))
    assert ndcg.numerator is ndcg.denominator is None
    repeated = (RankedUnit(rank=1, unit_key="c"), RankedUnit(rank=2, unit_key="c"))
    assert ranking_metrics(repeated, {"c": 2}, policy=policy, universe=("c",))[1].value == 1.0


def test_ranking_ideal_uses_unretrieved_judged_units_and_fixed_k() -> None:
    """无参数；验证理想排序包含未召回相关项，不能用当前结果伪造满分。"""
    policy = RankingPolicy(stage="query_raw", top_k=1)
    ranked = (RankedUnit(rank=1, unit_key="weak"), RankedUnit(rank=2, unit_key="best"))
    mrr, ndcg = ranking_metrics(ranked, {"weak": 1, "best": 3}, policy=policy, universe=("weak", "best"))
    assert mrr.value == 1.0
    assert ndcg.value == pytest.approx(1 / 7)


def test_incomplete_judgments_do_not_suppress_coverage_metrics(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证完整性不足只影响排序分数，其他有效覆盖仍可测量。"""
    judgments = benchmark.ranking_judgments[0]
    incomplete = judgments.model_copy(update={"labels": judgments.labels[:1]})
    scores = _scores(benchmark.model_copy(update={"ranking_judgments": (incomplete,)}), _observation(benchmark, ("critical",)))
    assert scores["mrr"].status == scores["ndcg_at_k"].status == "missing"
    assert scores["evidence_group_recall_at_k"].value == 0.5
    assert scores["critical_evidence_recall"].value == scores["scope_adherence"].value == 1.0


def test_ranking_requires_query_version_and_complete_universe(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证查询不匹配、缺索引全集及非法等级均不能冒充相关性标注。"""
    changed = _observation(benchmark, ("critical",)).model_copy(update={"query": "another query"})
    scores = _scores(benchmark, changed)
    assert scores["mrr"].reason == "ranking_judgments_missing"
    assert scores["critical_evidence_recall"].value == 1.0
    policy = RankingPolicy(stage="query_raw", top_k=2)
    ranked = (RankedUnit(rank=1, unit_key="a"),)
    for labels, universe, reason in ((None, ("a",), "judgments_missing"), ({"a": 1}, None, "universe_missing"),
                                     ({"a": True}, ("a",), "relevance_out_of_range"), ({"a": 1}, ("a", "b"), "judgments_incomplete")):
        with pytest.raises(EvidenceDataError, match=reason):
            ranking_metrics(ranked, labels, policy=policy, universe=universe)


def test_nonincreasing_real_ranks_are_invalid(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证逆序或相同 rank 不能被排序整理后作为真实观察评分。"""
    observation = _observation(benchmark, ("critical", "background"))
    changed = observation.model_copy(update={"items": (observation.items[1], observation.items[0])})
    scores = _scores(benchmark, changed)
    assert scores["mrr"].status == scores["evidence_group_recall_at_k"].status == "invalid_input"


def test_ranking_zero_relevance_and_extreme_finite_gains() -> None:
    """无参数；验证无相关目标不适用，合法大增益不会因求和溢出形成 NaN。"""
    ranked = (RankedUnit(rank=1, unit_key="a"), RankedUnit(rank=2, unit_key="b"))
    policy = RankingPolicy(stage="query_raw", top_k=2, gains=(0.0, 1e308))
    mrr, ndcg = ranking_metrics(ranked, {"a": 1, "b": 1}, policy=policy, universe=("a", "b"))
    assert mrr.value == ndcg.value == 1.0
    assert all(metric.status == "not_measured" for metric in ranking_metrics(ranked, {"a": 0, "b": 0}, policy=policy, universe=("a", "b")))


def test_scope_distinguishes_unknown_unconstrained_and_empty_allowed_set() -> None:
    """无参数；验证三种范围语义和空授权集合保持独立。"""
    assert scope_metric(("a",), EffectiveScope(status="missing")).status == "missing"
    assert scope_metric(("a",), EffectiveScope(status="unconstrained")).status == "not_measured"
    assert scope_metric(("a",), EffectiveScope(status="constrained")).value == 0.0
    assert scope_metric((), EffectiveScope(status="constrained")).value == 1.0
    assert scope_metric((None,), EffectiveScope(status="constrained", allowed_paper_ids=("a",))).status == "missing"


def test_scope_violation_survives_missing_hit_identity_and_is_not_limited_to_k(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证一个确定越界不会被其他缺失命中或 top-k 筛选隐藏。"""
    observation = _observation(benchmark, ("critical", "background", "other"))
    missing = ObservedItem(payload_kind="child", text_snapshot="unknown", rank=4)
    changed = observation.model_copy(update={"items": (*observation.items, missing)})
    result = grade_stage(benchmark, changed)
    assert result.scope_violations == ("b",)
    scores = {metric.name: metric for metric in result.measurements}
    assert scores["scope_adherence"].value == 0.0
    assert scores["evidence_group_recall_at_k"].value == 1.0


def test_cross_paper_coverage_requires_each_papers_own_complete_proof(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证论文 A 命中不可复制成论文 B 已满足证据要求。"""
    requirements = (PaperRequirement(paper_id="a", groups=(_group("a-proof", spans=("critical",)),)),
                    PaperRequirement(paper_id="b", groups=(_group("b-proof", spans=("other",)),)))
    changed = benchmark.model_copy(update={"target_paper_ids": ("a", "b"), "paper_requirements": requirements})
    metric = cross_paper_metric(changed, frozenset(("critical", "background")))
    assert metric.value == 0.5
    assert metric.numerator == 1 and metric.denominator == 2
    assert cross_paper_metric(changed, frozenset(("critical", "other"))).value == 1.0
    wrong = PaperRequirement(paper_id="b", groups=(_group("wrong", spans=("critical",)),))
    invalid = cross_paper_metric(changed.model_copy(update={"paper_requirements": (requirements[0], wrong)}), frozenset(("critical",)))
    assert invalid.status == "invalid_input"
    assert invalid.reason == "cross_paper_proof_source_mismatch"


def test_cross_paper_missing_requirements_and_single_paper_are_separate(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证单篇不适用，跨篇缺少要求时明确 missing。"""
    assert cross_paper_metric(benchmark, frozenset()).status == "not_measured"
    changed = benchmark.model_copy(update={"target_paper_ids": ("a", "b")})
    assert cross_paper_metric(changed, frozenset(("critical",))).status == "missing"


@pytest.mark.parametrize("stage", ["claim_input", "final_input", "evidence_pool"])
def test_unranked_context_has_stage_specific_coverage_name(benchmark: EvidenceBenchmark, stage: str) -> None:
    """参数为真值和上下文阶段；验证无排名阶段不冠以 Recall@k 或返回排序分数。"""
    observation = _observation(benchmark, ("critical",), stage=stage)
    scores = _scores(benchmark, observation)
    name = "evidence_group_pool_coverage" if stage == "evidence_pool" else "evidence_group_context_coverage"
    assert scores[name].value == 0.5
    assert "evidence_group_recall_at_k" not in scores
    assert scores["mrr"].status == scores["ndcg_at_k"].status == "not_measured"


def test_unreviewed_candidates_cannot_enter_gold_denominators(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证未审核候选无质量分母，但真实范围违规仍可判定。"""
    candidate = benchmark.model_copy(update={"review_status": "candidate"})
    scores = _scores(candidate, _observation(benchmark, ("other",)))
    assert scores["evidence_group_recall_at_k"].status == "not_measured"
    assert scores["evidence_group_recall_at_k"].denominator is None
    assert scores["scope_adherence"].value == 0.0


def test_missing_mapping_and_changed_index_text_do_not_become_false_zeros(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证无法复核的映射是 missing，实际正文篡改是 invalid_input。"""
    observation = _observation(benchmark, ("critical",))
    absent = observation.model_copy(update={"index_snapshot_id": "unavailable"})
    assert _scores(benchmark, absent)["evidence_group_recall_at_k"].status == "missing"
    changed_item = observation.items[0].model_copy(update={"text_snapshot": "different text"})
    changed = observation.model_copy(update={"items": (changed_item,)})
    assert _scores(benchmark, changed)["evidence_group_recall_at_k"].status == "invalid_input"
    assert _scores(benchmark, changed)["mrr"].status == "invalid_input"


def test_ranked_identity_change_and_missing_source_mapping_are_independent(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证排序可使用完整真实索引，但不能用错误的 index 身份冒充同一单位。"""
    index = benchmark.index_snapshots[0]
    unmapped = make_index_snapshot(source_snapshot_ids=index.source_snapshot_ids, retrieval_backend=index.retrieval_backend, units=index.units)
    judgments = benchmark.ranking_judgments[0].model_copy(update={"index_snapshot_id": unmapped.index_snapshot_id})
    changed_benchmark = benchmark.model_copy(update={"index_snapshots": (unmapped,), "ranking_judgments": (judgments,)})
    observation = _observation(changed_benchmark, ("critical",))
    scores = _scores(changed_benchmark, observation)
    assert scores["evidence_group_recall_at_k"].status == "missing"
    assert scores["mrr"].value == 1.0
    changed_item = observation.items[0].model_copy(update={"index_snapshot_id": "old-index"})
    invalid = observation.model_copy(update={"items": (changed_item,)})
    assert _scores(changed_benchmark, invalid)["mrr"].status == "invalid_input"


def test_model_source_label_does_not_need_to_equal_retrieval_backend(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证模型输入可独立命名数据层，仍须通过原文与索引正文校验。"""
    observation = _observation(benchmark, ("critical",), stage="final_input")
    changed = observation.model_copy(update={"retrieval_backend": "model_context"})
    assert _scores(benchmark, changed)["evidence_group_context_coverage"].value == 0.5


def test_scope_invalid_identity_never_becomes_a_valid_scope_member() -> None:
    """无参数；验证布尔或空身份无有效数值，但实际已知越界仍然明确失败。"""
    scope = EffectiveScope(status="constrained", allowed_paper_ids=("a",))
    assert scope_metric((True,), scope).status == "invalid_input"
    assert scope_metric(("",), scope).status == "invalid_input"
    assert scope_metric((True, "b"), scope).value == 0.0


def test_merged_stage_keeps_identity_without_borrowing_raw_query_policy(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证合并阶段保持独立，不能冒充原始查询的排序结果。"""
    scores = _scores(benchmark, _observation(benchmark, ("critical",), stage="merged_hits"))
    assert scores["evidence_group_recall_at_k"].value == 0.5
    assert scores["mrr"].status == "not_measured"
    assert scores["mrr"].reason == "ranking_policy_stage_not_applicable"
