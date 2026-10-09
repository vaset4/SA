"""从实际原文观察投影 Agent 任务事实，不把输入或预期当作命中。作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from evals.contracts import MetricResult, read_snapshot
from evals.evidence.alignment import (
    EvidenceDataError, align_observed_item, covered_span_ids, validate_benchmark,
)
from evals.evidence.contracts import (
    EffectiveScope, EvidenceBenchmark, ObservedItem, StageObservation,
)
from evals.evidence.metrics import grade_stage, group_hits, scope_metric
from evals.evidence.report import extract_stage_observations

from .contracts import AgentCase, AgentRun

_INCOMPLETE_STATES = frozenset({
    "failed", "incomplete", "terminate_fail", "terminate_honest",
    "planner_graph_timeout", "planner_graph_recursion_limit",
    "failed_honestly_without_evidence",
})
_COMPLETED_STATES = frozenset({"completed", "converged", "completed_after_clarification"})
_FIRST_ANALYSIS_FAILURE_STATES = frozenset({
    "planner_depth_assessment_failed", "planner_context_load_failed",
    "memory_read_failed", "planner_structured_output_failed",
})


@dataclass(frozen=True, kw_only=True)
class CapturedTurn:
    """保留一轮的实际输入、最终输出与独立请求级观察。"""

    number: int
    input: dict[str, Any]
    values: dict[str, Any]
    observation: dict[str, Any]
    terminal: str | None
    error: Any


@dataclass(frozen=True, kw_only=True)
class DeliveredEvidence:
    """最终出处的真实原文关联及缺失、错配原因。"""

    metric: MetricResult
    span_ids: frozenset[str] = frozenset()
    paper_ids: frozenset[str] = frozenset()
    reference_count: int = 0
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class EpisodeFacts:
    """评分派生事实；不回写为运行器观察，不传入被测图。"""

    turns: tuple[CapturedTurn, ...]
    evidence: MetricResult
    delivered: DeliveredEvidence
    scope_adherence: MetricResult
    covered_spans: frozenset[str] = frozenset()
    missing_spans: frozenset[str] = frozenset()
    selected_papers: frozenset[str] = frozenset()
    target_papers: frozenset[str] = frozenset()
    fault_observed: bool = False
    recovery_observed: bool = False
    recoverable_fault: bool = False


def load_captured_turns(run: AgentRun) -> tuple[CapturedTurn, ...]:
    """参数为冻结运行；返回实际轮次，畸形快照明确报错。"""
    raw = read_snapshot(run.observation_snapshot).get("turns")
    if not isinstance(raw, list):
        raise EvidenceDataError("missing", "turn_observations_missing")
    turns: list[CapturedTurn] = []
    for item in raw:
        if not isinstance(item, dict):
            raise EvidenceDataError("invalid_input", "turn_observation_shape_invalid")
        if not isinstance(item.get("input_snapshot"), str) or not isinstance(item.get("values_snapshot"), str):
            raise EvidenceDataError("missing", "captured_turn_input_or_values_missing")
        turns.append(CapturedTurn(
            number=item["turn"], input=read_snapshot(item["input_snapshot"]),
            values=read_snapshot(item["values_snapshot"]),
            observation=read_snapshot(item.get("evidence_observation_snapshot", "{}")),
            terminal=item.get("terminal_state"), error=item.get("error"),
        ))
    return tuple(turns)


def _scope(turn: CapturedTurn) -> EffectiveScope:
    """参数为实际轮次；读取执行输入的已绑定范围，缺失身份保持 missing。"""
    scope = turn.input.get("project_scope")
    if not isinstance(scope, dict):
        return EffectiveScope(status="missing")
    if not scope.get("user_id") or not scope.get("local_retrieval_allowed"):
        return EffectiveScope(status="missing")
    return EffectiveScope(status="constrained", allowed_paper_ids=tuple(scope["paper_ids"]))


def _index_id(turn: CapturedTurn) -> str | None:
    """参数为独立请求；只读取实际采集器记录的执行前索引声明。"""
    events = turn.observation.get("evidence_observations", [])
    declarations = [event for event in events if event.get("stage") == "index_snapshot_declared"]
    if len(declarations) != 1:
        return None
    identity = declarations[0]["payload"].get("index_snapshot_id")
    return identity if isinstance(identity, str) else None


def _record_item(record: Mapping[str, Any]) -> ObservedItem:
    """参数为真实 evidence_records 载荷；仅提取实际身份和保留的正文。"""
    text = record.get("expanded_text") or record.get("excerpt") or record.get("summary")
    return ObservedItem.model_validate({
        **{key: record.get(key) for key in (
            "paper_id", "chunk_id", "parent_id", "evidence_id", "index_snapshot_id",
        )},
        "payload_kind": "model_excerpt", "text_snapshot": text,
    })


def _turn_stages(turn: CapturedTurn) -> tuple[StageObservation, ...]:
    """参数为实际轮次；提取请求事件及明确证据字段，不递归搜索任意对象。"""
    stages = extract_stage_observations(turn.observation) if turn.observation else ()
    records = turn.values.get("evidence_records")
    if records is None:
        return stages
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise EvidenceDataError("invalid_input", "evidence_records_shape_invalid")
    return (*stages, StageObservation(
        stage="evidence_pool", call_id=f"turn-{turn.number}-evidence-records",
        retrieval_backend="actual_evidence_records", status="complete", scope=_scope(turn),
        index_snapshot_id=_index_id(turn), items=tuple(_record_item(item) for item in records),
    ))


def _references(turn: CapturedTurn) -> tuple[dict[str, Any], ...]:
    """参数为实际最终输出；返回引用 ID 及声明的论文、片段身份，不读取计划。"""
    references: list[dict[str, Any]] = []
    blocks = turn.values.get("planner_final_blocks", [])
    outcome = turn.values.get("turn_outcome", {})
    for container in [*blocks, outcome]:
        for identity in container.get("evidence_ids", []):
            references.append({"evidence_id": identity})
    for citation in turn.values.get("citations", []):
        if not isinstance(citation, dict):
            raise EvidenceDataError("invalid_input", "citation_shape_invalid")
        references.append(dict(citation))
    return tuple(references)


def _observed_registry(turns: tuple[CapturedTurn, ...]) -> dict[str, tuple[ObservedItem, str | None]]:
    """参数为实际轮次；建立真实证据 ID 到载荷的关联，冲突身份拒绝评分。"""
    registry: dict[str, tuple[ObservedItem, str | None]] = {}
    stages = tuple(stage for turn in turns for stage in _turn_stages(turn))
    for stage in stages:
        # 1. 【Agent评测】【出处关联】原始证据池建立出处身份，模型输入中的截短展示不覆盖原记录
        if stage.stage != "evidence_pool":
            continue
        for item in stage.items or ():
            if item.evidence_id is None:
                continue
            value = (item, stage.index_snapshot_id)
            prior = registry.get(item.evidence_id)
            if prior is not None and prior[0].model_dump(exclude={"rank", "score"}) != item.model_dump(exclude={"rank", "score"}):
                raise EvidenceDataError("invalid_input", f"conflicting_evidence_id:{item.evidence_id}")
            registry[item.evidence_id] = value
    return registry


def resolve_delivered_evidence(
    turns: tuple[CapturedTurn, ...], benchmark: EvidenceBenchmark,
) -> DeliveredEvidence:
    """参数为实际轮次和原文合同；逐一将最终出处连回真实证据及原文坐标。"""
    registry = _observed_registry(turns)
    references = _references(turns[-1]) if turns else ()
    if not references:
        return DeliveredEvidence(metric=MetricResult(
            name="citation_verifiability_rate", status="not_measured", reason="no_returned_citations",
        ))
    spans: set[str] = set()
    papers: set[str] = set()
    issues: list[str] = []
    for reference in references:
        identity = reference.get("evidence_id")
        binding = registry.get(identity) if isinstance(identity, str) else None
        if binding is None or any(reference.get(key) not in (None, getattr(binding[0], key)) for key in ("paper_id", "chunk_id", "parent_id")):
            issues.append(f"unresolved_or_mismatched_citation:{identity}")
            continue
        item, index_id = binding
        ranges = align_observed_item(item, benchmark, index_snapshot_id=index_id)
        if not ranges:
            issues.append(f"citation_has_no_original_text:{identity}")
            continue
        pages = {block.page_num for source in benchmark.sources for block in source.blocks
                 if any(interval.source_snapshot_id == source.snapshot_id and interval.block_id == block.block_id for interval in ranges)}
        if reference.get("page_num") is not None and reference["page_num"] not in pages:
            issues.append(f"citation_page_mismatch:{identity}")
            continue
        spans.update(covered_span_ids(benchmark.spans, ranges))
        if item.paper_id is not None:
            papers.add(item.paper_id)
    count = len(references)
    valid = count - len(issues)
    return DeliveredEvidence(
        metric=MetricResult(name="citation_verifiability_rate", status="measured", value=valid / count,
                            numerator=float(valid), denominator=float(count), reason="original_text_reference_bindings"),
        span_ids=frozenset(spans), paper_ids=frozenset(papers), reference_count=count, issues=tuple(issues),
    )


def _load_targets(case: AgentCase) -> tuple[EvidenceBenchmark, dict[str, Any]]:
    """参数为独立案例真值；校验原文和目标绑定后返回，不从输入补标签。"""
    truth = read_snapshot(case.truth_snapshot)
    if not isinstance(truth.get("evidence_benchmark"), dict) or not isinstance(truth.get("evidence_targets"), dict):
        raise EvidenceDataError("missing", "bound_evidence_targets_missing")
    benchmark = EvidenceBenchmark.model_validate(truth["evidence_benchmark"])
    validate_benchmark(benchmark, include_ranking=False)
    targets = dict(truth["evidence_targets"])
    known = {
        "required_span_ids": {span.span_id for span in benchmark.spans},
        "required_group_ids": {group.group_id for group in benchmark.groups},
        "required_paper_ids": {source.paper_id for source in benchmark.sources},
    }
    negative = truth.get("answerability") == "insufficient_evidence"
    for key, identities in known.items():
        selected = targets.get(key)
        if not isinstance(selected, list) or (not selected and not negative) or not set(selected) <= identities:
            raise EvidenceDataError("invalid_input", f"evidence_target_binding_invalid:{key}")
    return benchmark, targets


def _coverage_facts(
    turns: tuple[CapturedTurn, ...], benchmark: EvidenceBenchmark, *, targets: dict[str, Any],
) -> tuple[MetricResult, frozenset[str], frozenset[str], MetricResult]:
    """参数为真实阶段和固定目标；返回完整证明覆盖、选证论文及授权范围检查。"""
    covered: set[str] = set()
    selected: set[str] = set()
    checks: list[MetricResult] = []
    gaps: list[MetricResult] = []
    stages_seen = 0
    for turn in turns:
        for stage in _turn_stages(turn):
            stages_seen += 1
            grade = grade_stage(benchmark, stage, policy="candidate_diagnostic" if benchmark.review_status == "candidate" else "reviewed")
            covered.update(grade.covered_span_ids)
            items = stage.items
            checks.append(scope_metric(tuple(item.paper_id for item in items) if items is not None else None, _scope(turn)))
            gaps.extend(metric for metric in grade.measurements if metric.name.startswith("evidence_group_") and metric.status != "measured")
            if stage.stage in {"evidence_pool", "claim_input", "final_input"}:
                selected.update(item.paper_id for item in items or () if item.paper_id)
    if not stages_seen:
        raise EvidenceDataError("missing", "evidence_stages_not_observed")
    span_ids = frozenset(covered)
    passed = set(targets["required_span_ids"]) <= span_ids and set(targets["required_group_ids"]) <= group_hits(benchmark.groups, span_ids)
    metric = _predicate("all_required_evidence_observed", passed, "required_original_spans_and_groups")
    # 1. 【Agent评测】【资料不足】没有正例证明目标时不制造覆盖率，范围与返回引用仍按实际观察核验
    if not any(targets.values()):
        metric = MetricResult(name=metric.name, status="not_measured", reason="no_positive_evidence_targets")
    elif not passed and gaps:
        metric = gaps[0].model_copy(update={"name": metric.name})
    scope = _combine_scope(checks)
    return metric, span_ids, frozenset(selected), scope


def _combine_scope(checks: list[MetricResult]) -> MetricResult:
    """参数为实际访问的范围检查；任何越界优先失败，缺失不能被平均隐藏。"""
    failed = next((item for item in checks if item.value == 0.0), None)
    if failed is not None:
        return failed
    gap = next((item for item in checks if item.status in {"missing", "invalid_input"}), None)
    if gap is not None:
        return gap
    return _predicate("scope_adherence", bool(checks), "all_observed_access_within_authorized_scope")


def _failure_facts(turns: tuple[CapturedTurn, ...]) -> tuple[bool, bool]:
    """参数为实际错误和事件；只承认关联失败单元的补查与真实重执行。"""
    fault = False
    recovered = False
    for turn in turns:
        failed_units: set[str] = set()
        repaired_units: set[str] = set()
        fault = fault or bool(turn.error)
        for event in turn.observation.get("evidence_observations", []):
            stage, payload = event.get("stage"), event.get("payload", {})
            scope = event.get("scope") or {}
            unit = scope.get("unit_id")
            fault = fault or stage in {"scope_failed", "source_port_failed"}
            if stage == "scope_failed" and unit:
                failed_units.add(unit)
            if stage == "repair_evidence_pool":
                repaired_units.update(set(payload.get("linked_unit_ids", [])) & failed_units)
            if stage == "scope_completed" and scope.get("stage") == "repair_rerun" and unit in repaired_units:
                recovered = True
        fault = fault or bool(turn.values.get("error_code") or turn.values.get("failure_reason"))
    return fault, recovered


def collect_episode_facts(case: AgentCase, run: AgentRun) -> EpisodeFacts:
    """参数为独立真值和真实运行；返回可审计的原文、出处、范围及故障事实。"""
    try:
        turns = load_captured_turns(run)
    except EvidenceDataError as exc:
        return _unavailable_facts(exc)
    fault, recovery = _failure_facts(turns)
    declared_fault = read_snapshot(case.truth_snapshot).get("fault_injection")
    recoverable = isinstance(declared_fault, dict) and declared_fault.get("recoverable") is True
    try:
        benchmark, targets = _load_targets(case)
        evidence, covered, selected, scope = _coverage_facts(turns, benchmark, targets=targets)
        try:
            delivered = resolve_delivered_evidence(turns, benchmark)
        except EvidenceDataError as exc:
            delivered = DeliveredEvidence(metric=MetricResult(name="citation_verifiability_rate", status=exc.status, reason=exc.reason), issues=(exc.reason,))
        return EpisodeFacts(
            turns=turns, fault_observed=fault, recovery_observed=recovery, recoverable_fault=recoverable and fault,
            evidence=evidence, delivered=delivered, scope_adherence=scope,
            covered_spans=covered, missing_spans=frozenset(targets["required_span_ids"]) - covered,
            selected_papers=selected, target_papers=frozenset(targets["required_paper_ids"]),
        )
    except (EvidenceDataError, ValidationError) as exc:
        status = exc.status if isinstance(exc, EvidenceDataError) else "invalid_input"
        reason = exc.reason if isinstance(exc, EvidenceDataError) else str(exc)
        return EpisodeFacts(
            turns=turns, fault_observed=fault, recovery_observed=recovery, recoverable_fault=recoverable and fault,
            evidence=MetricResult(name="all_required_evidence_observed", status=status, reason=reason),
            scope_adherence=MetricResult(name="scope_adherence", status=status, reason=reason),
            delivered=DeliveredEvidence(metric=MetricResult(name="citation_verifiability_rate", status=status, reason=reason)),
        )


def _unavailable_facts(error: EvidenceDataError) -> EpisodeFacts:
    """参数为实际轮次数据缺口；返回明确缺失事实，不用案例输入伪造已捕获输入。"""
    return EpisodeFacts(
        turns=(), evidence=MetricResult(name="all_required_evidence_observed", status=error.status, reason=error.reason),
        scope_adherence=MetricResult(name="scope_adherence", status=error.status, reason=error.reason),
        delivered=DeliveredEvidence(metric=MetricResult(name="citation_verifiability_rate", status=error.status, reason=error.reason)),
    )


def _predicate(name: str, passed: bool, reason: str) -> MetricResult:
    """参数为既有谓词、判断和事实原因；返回内部派生判断，不新增主指标。"""
    value = float(passed)
    return MetricResult(name=name, status="measured", value=value, numerator=value, denominator=1.0, reason=reason)


def _delivery_predicate(name: str, facts: EpisodeFacts) -> MetricResult:
    """参数为出处要求和实际事实；完整目标、有效出处及片段交付必须同时满足。"""
    for metric in (facts.evidence, facts.delivered.metric):
        if metric.status in {"missing", "invalid_input"}:
            return metric.model_copy(update={"name": name})
    expected = facts.covered_spans | facts.missing_spans
    passed = facts.evidence.value == 1.0 and facts.delivered.metric.value == 1.0 and expected <= facts.delivered.span_ids
    return _predicate(name, passed, "required_spans_delivered_with_resolvable_sources")


def honest_failure(facts: EpisodeFacts) -> bool:
    """参数为实际事实；要求对用户显式保留失败和未完成，不能仅凭进程异常判诚实。"""
    if not facts.turns or not facts.fault_observed:
        return False
    last = facts.turns[-1]
    outcome = last.values.get("turn_outcome", {})
    answer, summary = last.values.get("answer"), last.values.get("error_summary")
    # 1. 【Agent评测】【首次分析失败】尚无turn_outcome时，失败码、终止决策及实际展示的错误摘要共同证明失败已告知
    early_failure = (
        last.terminal in _FIRST_ANALYSIS_FAILURE_STATES
        and last.values.get("error_code") == last.terminal
        and last.values.get("planner_loop_decision") == "terminate_fail"
        and isinstance(summary, str) and bool(summary.strip())
        and isinstance(answer, str) and summary in answer
    )
    visible = bool(answer) and bool(early_failure or outcome.get("termination_reason") or outcome.get("error_kind") or outcome.get("missing_evidence") or last.values.get("failure_reason"))
    has_claims = bool(_references(last)) or bool(last.values.get("evidence_records"))
    return (last.terminal in _INCOMPLETE_STATES or early_failure) and visible and not has_claims


def completed_terminal(case: AgentCase, run: AgentRun, facts: EpisodeFacts) -> bool:
    """参数为案例、运行和真实终态；只按允许的完成状态认定完成。"""
    if run.status != "completed" or not facts.turns:
        return False
    allowed = read_snapshot(case.truth_snapshot).get("allowed_terminal_states", [])
    states = {item["state"] for item in allowed if item.get("task_completed") is True}
    return bool(states) and facts.turns[-1].terminal in states | _COMPLETED_STATES


def _clarification_predicate(facts: EpisodeFacts) -> MetricResult:
    """参数为多轮事实；首轮须明确澄清且未抢先给出论文出处，后续必须真实存在。"""
    name = "ambiguity_resolved_before_final_attribution"
    if len(facts.turns) < 2:
        return _predicate(name, False, "followup_turn_missing")
    first = facts.turns[0]
    assessment = first.values.get("turn_depth_assessment", {})
    explicit = assessment.get("needs_clarification") or first.values.get("needs_clarification") or first.values.get("ask_clarification")
    if explicit is None:
        return MetricResult(name=name, status="missing", reason="structured_clarification_observation_missing")
    return _predicate(name, bool(explicit) and not _references(first) and first.terminal not in _COMPLETED_STATES, "clarification_before_attribution")


def evaluate_predicates(case: AgentCase, facts: EpisodeFacts) -> tuple[MetricResult, ...]:
    """参数为案例要求和实际事实；逐项返回原有任务要求的判断及缺失原因。"""
    required = read_snapshot(case.truth_snapshot).get("required_outcomes")
    if not isinstance(required, list) or not required or any(not isinstance(item, dict) for item in required):
        return (MetricResult(name="required_outcomes", status="invalid_input", reason="required_outcomes_missing_or_invalid"),)
    results = []
    for item in required:
        name = item.get("predicate")
        if not isinstance(name, str):
            results.append(MetricResult(name="required_outcomes", status="invalid_input", reason="predicate_name_missing"))
            continue
        results.append(_evaluate_predicate(name, facts))
    return tuple(results)


def _evaluate_predicate(name: str, facts: EpisodeFacts) -> MetricResult:
    """参数为单项既有要求和实际事实；集中解释范围、故障和出处语义。"""
    direct = {
        "all_required_evidence_observed": facts.evidence,
        "requested_paper_evidence_observed": facts.evidence,
        "authorized_scope_preserved": facts.scope_adherence,
    }
    if name in direct:
        return direct[name].model_copy(update={"name": name})
    if name in {"returned_sources_resolve_to_observed_evidence", "all_required_evidence_and_sources_delivered"}:
        return _delivery_predicate(name, facts)
    if name == "ambiguity_resolved_before_final_attribution":
        return _clarification_predicate(facts)
    if name in {"followup_uses_confirmed_scope", "selected_and_returned_sources_within_requested_scope"}:
        delivery = _delivery_predicate(name, facts)
        within = bool(facts.target_papers) and facts.selected_papers <= facts.target_papers and facts.delivered.paper_ids <= facts.target_papers
        followup = name != "followup_uses_confirmed_scope" or len(facts.turns) >= 2
        return delivery if delivery.value != 1.0 else _predicate(name, within and followup, "actual_selected_and_delivered_paper_scope")
    if name == "actual_source_failure_observed":
        return _predicate(name, facts.fault_observed, "actual_failure_observation")
    if name in {"no_unobserved_evidence_claimed", "failure_terminal_preserves_uncompleted_goal"}:
        return _predicate(name, honest_failure(facts), "visible_failure_without_invented_sources")
    return MetricResult(name=name, status="invalid_input", reason="unsupported_required_outcome")
