"""分层汇总原评分、重复运行和确定性检查。作者：xxx。"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

from evals.contracts import MetricResult, read_snapshot, snapshot, validate_execution_input

from .contracts import (
    METRIC_STATUSES, CaseMeasurement, CheckName, CheckOutcome, ImportedRun,
    MetricGroup, RepetitionResult, RunMeasurement, RunSlot, ScoreSlice, StateCount,
    SuiteInput, SuiteSummary,
)

ScoredMetric = tuple[ScoreSlice, MetricResult]
CheckStatus = Literal["passed", "failed", "missing", "not_applicable"]
_CHECK_METRICS: dict[CheckName, tuple[tuple[str, ...], str, float]] = {
    "scope": (("evidence-benchmark", "agent_outcome"), "scope_adherence", 1.0),
    "citation": (("research_planner", "agent_outcome"), "citation_verifiability_rate", 1.0),
    "stale_memory": (("research_planner",), "stale_memory_leak_rate", 0.0),
}


def score_group_key(score: ScoreSlice, run: ImportedRun, inputs: SuiteInput) -> str:
    """参数为原评分、原运行和套件输入；返回不会混合来源或评分语义的分组键。"""
    return snapshot({
        "purpose": inputs.manifest.purpose, "kind": score.kind,
        "source_type": run.case.source_type, "run_mode": run.run.run_mode,
        "review_status": run.review_status, "grader": score.grading.grader,
        "grader_version": score.grading.grader_version,
        "projection_version": score.grading.projection_version,
        "stage": score.stage, "backend": score.backend,
        "task_family": read_snapshot(run.topology_snapshot).get("task_family"),
    })


def partition_score_slices(inputs: SuiteInput) -> dict[tuple[str, str], tuple[ScoredMetric, ...]]:
    """参数为冻结输入；返回独立指标分组，线上发现和 Oracle 不参与质量均值。"""
    if inputs.manifest.purpose == "online_discovery":
        return {}
    runs = {run.run.run_id: run for run in inputs.runs}
    buckets: dict[tuple[str, str], list[ScoredMetric]] = defaultdict(list)
    for score in inputs.scores:
        run = runs[score.run_id]
        if run.run.run_mode == "oracle":
            continue
        key = score_group_key(score, run, inputs)
        for metric in score.grading.measurements:
            buckets[(key, metric.name)].append((score, metric))
    return {key: tuple(values) for key, values in buckets.items()}


def _run_measurement(slot: RunSlot, measurements: tuple[MetricResult, ...]) -> RunMeasurement:
    """参数为重复槽位和实际测量；只有完整测量才生成该次运行的均值。"""
    if not measurements:
        return RunMeasurement(repeat_id=slot.repeat_id, run_id=slot.run_id, measurements=(),
                              value=None, status="missing", reason="run_or_metric_not_recorded")
    statuses = {metric.status for metric in measurements}
    values = [metric.value for metric in measurements if metric.value is not None]
    if statuses == {"measured"}:
        return RunMeasurement(repeat_id=slot.repeat_id, run_id=slot.run_id, measurements=measurements,
                              value=sum(values) / len(values), status="measured", reason="original_measurements")
    status = "invalid_input" if "invalid_input" in statuses else "missing"
    if statuses == {"not_measured"}:
        status = "not_measured"
    return RunMeasurement.model_validate({
        "repeat_id": slot.repeat_id, "run_id": slot.run_id, "measurements": measurements,
        "value": None, "status": status, "reason": "original_measurement_coverage_incomplete",
    })


def _case_measurement(slots: tuple[RunSlot, ...], rows: tuple[ScoredMetric, ...], metric: str) -> CaseMeasurement:
    """参数为同题槽位、评分行和指标名；先按运行汇总，明确保留部分覆盖及范围违规。"""
    by_run: dict[str, list[MetricResult]] = defaultdict(list)
    for score, measurement in rows:
        by_run[score.run_id].append(measurement)
    runs = tuple(_run_measurement(slot, tuple(by_run.get(slot.run_id or "", ()))) for slot in slots)
    applicable = [run for run in runs if run.status != "not_measured"]
    complete = bool(applicable) and all(run.status == "measured" for run in applicable)
    values = [run.value for run in applicable if run.value is not None]
    value = sum(values) / len(values) if complete and values else None
    # 1. 【评测套件】【范围检查】已观察到的越界不会因另一调用缺数据而从报告消失
    if metric == "scope_adherence" and any(item.value == 0 for _score, item in rows):
        value = 0.0
    return CaseMeasurement(case_id=slots[0].case_id, case_version=slots[0].case_version,
                           value=value, complete=complete, runs=runs)


def _group_slots(inputs: SuiteInput, key: str) -> dict[tuple[str, str], tuple[RunSlot, ...]]:
    """参数为输入和分组键；选择该来源的预期重复槽位，未知槽位保留为缺口。"""
    runs = {run.run.run_id: run for run in inputs.runs}
    members = {score.run_id for score in inputs.scores if score_group_key(score, runs[score.run_id], inputs) == key}
    cases = {(runs[run_id].case.case_id, runs[run_id].case.case_version) for run_id in members}
    result: dict[tuple[str, str], tuple[RunSlot, ...]] = {}
    for case in sorted(cases):
        result[case] = tuple(slot for slot in inputs.manifest.slots
                             if (slot.case_id, slot.case_version) == case and (slot.run_id is None or slot.run_id in members))
    return result


def summarize_group(inputs: SuiteInput, group: tuple[str, str], rows: tuple[ScoredMetric, ...]) -> MetricGroup:
    """参数为输入、指标分组及原测量；返回题级均值和原子四态计数。"""
    key, metric = group
    slots = _group_slots(inputs, key)
    runs = {run.run.run_id: run for run in inputs.runs}
    cases = []
    for case, members in slots.items():
        selected = tuple(row for row in rows if (runs[row[0].run_id].case.case_id, runs[row[0].run_id].case.case_version) == case)
        cases.append(_case_measurement(members, selected, metric))
    counts: Counter[str] = Counter()
    for case_result in cases:
        for run in case_result.runs:
            counts.update(item.status for item in run.measurements)
            if not run.measurements:
                counts["missing"] += 1
    values = [case.value for case in cases if case.complete and case.value is not None]
    mean = sum(values) / len(values) if values else None
    if metric == "scope_adherence" and any(case.value == 0 for case in cases):
        mean = 0.0
    return MetricGroup(
        key_snapshot=key, metric=metric, mean=mean, cases=tuple(cases),
        complete_case_count=sum(case.complete for case in cases),
        states=tuple(StateCount(status=status, count=counts[status]) for status in METRIC_STATUSES),
    )


def _repetition(group: MetricGroup, case: CaseMeasurement) -> RepetitionResult:
    """参数为任务成功分组及题级结果；区分单次、至少一次与全部成功。"""
    measured = [run for run in case.runs if run.status == "measured"]
    if any(run.value not in (0.0, 1.0) or len(run.measurements) != 1 for run in measured):
        raise ValueError("task_success must be one binary outcome per run")
    successful = sum(run.value == 1.0 for run in measured)
    failed = sum(run.value == 0.0 for run in measured)
    complete = len(measured) == len(case.runs)
    any_success = True if successful else (False if complete else None)
    all_success = False if failed else (True if complete else None)
    return RepetitionResult(
        key_snapshot=group.key_snapshot, case_id=case.case_id, case_version=case.case_version, metric=group.metric,
        expected_runs=len(case.runs), measured_runs=len(measured), successful_runs=successful,
        single_run_success_rate=successful / len(measured) if complete else None,
        any_success=any_success, all_success=all_success, runs=case.runs,
    )


def summarize_repetitions(groups: tuple[MetricGroup, ...]) -> tuple[RepetitionResult, ...]:
    """参数为已分层指标；旧任务取证与新流程完成各自统计，保留独立指标名称。"""
    return tuple(_repetition(group, case) for group in groups
                 if group.metric in {"task_success", "workflow_completion"} for case in group.cases)


def _metric_check(metrics: tuple[MetricResult, ...], expected: float) -> tuple[CheckStatus, str]:
    """参数为已有确定性指标和通过值；真实违规优先，缺数据不能通过。"""
    if not metrics:
        return "missing", "required_assertion_not_recorded"
    if any(metric.status == "measured" and metric.value != expected for metric in metrics):
        return "failed", "original_deterministic_assertion_failed"
    if any(metric.status in {"missing", "invalid_input"} for metric in metrics):
        return "missing", "required_assertion_incomplete"
    if all(metric.status == "not_measured" for metric in metrics):
        return "not_applicable", "original_assertions_not_applicable"
    return "passed", "original_deterministic_assertions_passed"


def _trace_check(run: ImportedRun) -> tuple[CheckStatus, str]:
    """参数为原运行；核对实际 trace 载体和已记录缺口，不把回答当轨迹。"""
    observation = read_snapshot(run.run.observation_snapshot)
    trace_fields = {"planner_trace", "trace", "evidence_observations", "turn_observations"}
    if trace_fields.intersection(run.run.missing_fields):
        return "missing", "required_trace_reported_missing"
    if run.run.run_mode in {"agent_live", "agent_fixture"}:
        turns = observation.get("turns")
        present = isinstance(turns, list) and bool(turns) and all(
            isinstance(turn, dict) and isinstance(turn.get("events"), list)
            and isinstance(turn.get("values_snapshot"), str) for turn in turns
        )
    elif run.run.run_mode.startswith("planner"):
        present = isinstance(observation.get("planner_trace"), dict)
    else:
        present = any(isinstance(observation.get(field), (dict, list)) for field in trace_fields)
    if present:
        return "passed", "original_trace_recorded"
    return "missing", "required_trace_not_recorded"


def _false_success_check(run: ImportedRun, scores: tuple[ScoreSlice, ...]) -> tuple[CheckStatus, str]:
    """参数为实际运行和原评分；核对失败终态是否被任务成功评分掩盖。"""
    status = read_snapshot(run.run.observation_snapshot).get("status")
    workflow_scores = tuple(score for score in scores if score.grading.grader == "agent_workflow")
    if any(score.grading.grader_version == "agent-workflow-v2" for score in workflow_scores):
        if any(score.grading.grader_version != "agent-workflow-v2" for score in workflow_scores):
            return "missing", "mixed_workflow_grader_versions"
        metrics = tuple(metric for score in workflow_scores for metric in score.grading.measurements if metric.name == "false_success")
        return _metric_check(metrics, 0.0)
    # 1. 【评测套件】【历史口径】旧V1报告仍按原合同展示；V2不能借旧错误处理分数填缺失值
    workflow = tuple(metric for score in workflow_scores if score.grading.grader_version == "agent-workflow-v1"
                     for metric in score.grading.measurements if metric.name == "error_handling_correctness")
    if workflow:
        return _metric_check(workflow, 1.0)
    if read_snapshot(run.case.truth_snapshot).get("answerability") == "insufficient_evidence":
        return "not_applicable", "insufficient_evidence_case_has_no_positive_task_success_claim"
    metrics = tuple(metric for score in scores for metric in score.grading.measurements if metric.name == "task_success")
    if status not in {"completed", "failed", "cancelled", "timed_out"} or not metrics:
        return "missing", "terminal_state_or_task_outcome_not_recorded"
    if any(metric.status != "measured" for metric in metrics):
        return "missing", "task_outcome_not_measured"
    if status != "completed" and any(metric.value == 1 for metric in metrics):
        return "failed", "failed_terminal_reported_as_success"
    return "passed", "terminal_and_reported_outcome_consistent"


def _check_run(name: CheckName, run: ImportedRun, scores: tuple[ScoreSlice, ...]) -> tuple[CheckStatus, str]:
    """参数为检查名、原运行及原评分；只使用结构化事实和现有确定性判断。"""
    if run.run.run_mode == "oracle":
        return "not_applicable", "oracle_is_diagnostic_only"
    workflow = tuple(score for score in scores if score.grading.grader == "agent_workflow")
    if workflow and name in {"scope", "citation"}:
        metric_name = "scope_adherence" if name == "scope" else "citation_reference_validity"
        metrics = tuple(metric for score in workflow for metric in score.grading.measurements if metric.name == metric_name)
        return _metric_check(metrics, 1.0)
    if name in _CHECK_METRICS:
        graders, metric_name, expected = _CHECK_METRICS[name]
        metrics = tuple(metric for score in scores if score.grading.grader in graders
                        for metric in score.grading.measurements if metric.name == metric_name)
        return _metric_check(metrics, expected)
    if name == "truth_leakage":
        try:
            validate_execution_input(read_snapshot(run.run.input_snapshot))
            for turn in read_snapshot(run.run.observation_snapshot).get("turns", []):
                validate_execution_input(read_snapshot(turn["input_snapshot"]))
        except ValueError as exc:
            return "failed", str(exc)
        return "passed", "frozen_execution_input_has_no_truth_fields"
    if name == "required_trace":
        return _trace_check(run)
    if name == "false_success":
        return _false_success_check(run, scores)
    return "missing", "structured_forbidden_fact_assertion_not_recorded"


def summarize_required_checks(inputs: SuiteInput) -> tuple[CheckOutcome, ...]:
    """参数为套件输入；逐槽位保留 P0 判断，线上发现仅提供明细而不设门禁。"""
    if inputs.manifest.purpose == "online_discovery":
        return ()
    runs = {run.run.run_id: run for run in inputs.runs}
    result = []
    for slot in inputs.manifest.slots:
        scores = tuple(score for score in inputs.scores if score.run_id == slot.run_id)
        for name in slot.required_checks:
            status, reason = ("missing", "run_not_recorded") if slot.run_id is None else _check_run(name, runs[slot.run_id], scores)
            result.append(CheckOutcome.model_validate({
                "case_id": slot.case_id, "repeat_id": slot.repeat_id, "run_id": slot.run_id,
                "name": name, "status": status, "reason": reason,
            }))
    return tuple(result)


def aggregate_suite(inputs: SuiteInput) -> SuiteSummary:
    """参数为已校验冻结输入；返回来源分层、重复运行与逐项检查结果。"""
    partitions = partition_score_slices(inputs)
    groups = tuple(summarize_group(inputs, key, rows) for key, rows in sorted(partitions.items()))
    return SuiteSummary(inputs=inputs, groups=groups, repetitions=summarize_repetitions(groups),
                        checks=summarize_required_checks(inputs))
