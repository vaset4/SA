"""先核对实验条件，再比较相同题目和分母下的原始指标。作者：xxx。"""

from __future__ import annotations

from typing import Any

from evals.contracts import read_snapshot, snapshot

from .contracts import (
    ComparisonDecision, ComparisonRequest, ComparisonResult, ImportedRun,
    MetricDelta, MetricGroup, RunMeasurement, RunSlot, SuiteSummary,
)

SlotKey = tuple[str, str, str]


def _slots(summary: SuiteSummary) -> dict[SlotKey, RunSlot]:
    """参数为汇总；按题、版本和预期重复编号索引，运行 ID 可以不同。"""
    return {(slot.case_id, slot.case_version, slot.repeat_id): slot for slot in summary.inputs.manifest.slots}


def _grading_identities(summary: SuiteSummary, run_id: str) -> tuple[tuple[str, ...], ...]:
    """参数为汇总和原运行 ID；返回评分器语义身份，不把调用顺序当标准计划。"""
    identities = set()
    for score in summary.inputs.scores:
        if score.run_id != run_id:
            continue
        detail = read_snapshot(score.detail_snapshot)
        attempt = detail.get("attempt", {})
        # 1. 【评测比较】【评分身份】新attempt显式冻结judge配置，初始化失败的错误对象不冒充配置变化
        judge = attempt["identity_before"] if "identity_before" in attempt else detail.get("judge", {})
        identities.add((score.kind, score.grading.grader, score.grading.grader_version,
                        score.grading.projection_version, snapshot(judge) if score.grading.grader == "ragas_frozen" else ""))
    return tuple(sorted(identities))


def _run_reasons(current: ImportedRun, baseline: ImportedRun, request: ComparisonRequest) -> list[str]:
    """参数为配对运行及变化项；返回固定条件缺失、来源变化或额外变量变化的原因。"""
    reasons = []
    if (current.case.source_type, current.review_status, current.run.run_mode) != (
        baseline.case.source_type, baseline.review_status, baseline.run.run_mode,
    ):
        reasons.append("source_review_or_run_mode_changed")
    if current.run.run_mode == "oracle" or baseline.run.run_mode == "oracle":
        reasons.append("oracle_is_diagnostic_only")
    if current.topology_snapshot != baseline.topology_snapshot:
        reasons.append("case_topology_changed")
    if any(run.run.provenance != "captured" or "captured_execution_input" in run.run.missing_fields
           for run in (current, baseline)):
        reasons.append("captured_execution_input_missing")
    conditions = {condition.name: condition for condition in baseline.conditions}
    for condition in current.conditions:
        previous = conditions.get(condition.name)
        if previous is None or condition.status == "missing" or previous.status == "missing":
            reasons.append(f"identity_missing:{condition.name}")
        elif condition.status != previous.status:
            reasons.append(f"identity_applicability_changed:{condition.name}")
        elif condition.name == "retrieval" and "retrieval_method" in request.changed_factors:
            now = read_snapshot(condition.snapshot or "{}")
            old = read_snapshot(previous.snapshot or "{}")
            if now.get("method") not in {"dense", "hybrid"} or old.get("method") not in {"dense", "hybrid"}:
                reasons.append("retrieval_method_not_explicit")
            if {key: value for key, value in now.items() if key != "method"} != {key: value for key, value in old.items() if key != "method"}:
                reasons.append("undeclared_change:retrieval_configuration")
        elif condition.name not in request.changed_factors and condition.snapshot != previous.snapshot:
            reasons.append(f"undeclared_change:{condition.name}")
    return reasons


def assess_comparability(current: SuiteSummary, baseline: SuiteSummary, request: ComparisonRequest) -> ComparisonDecision:
    """参数为两次汇总与变化项；在计算任何分数差之前核对全部运行条件。"""
    reasons: list[str] = []
    current_manifest, baseline_manifest = current.inputs.manifest, baseline.inputs.manifest
    if current_manifest.purpose != baseline_manifest.purpose:
        reasons.append("suite_purpose_changed")
    if "online_discovery" in (current_manifest.purpose, baseline_manifest.purpose):
        reasons.append("online_discovery_has_no_benchmark_delta")
    if current_manifest.dataset_version != baseline_manifest.dataset_version:
        reasons.append("dataset_version_changed")
    if current_manifest.measurement_policy_snapshot != baseline_manifest.measurement_policy_snapshot:
        reasons.append("measurement_policy_changed")
    if current_manifest.grading_timing != baseline_manifest.grading_timing:
        reasons.append("grading_timing_changed")
    if any(read_snapshot(score.detail_snapshot).get("attempt", {}).get("identity_drift")
           for summary in (current, baseline) for score in summary.inputs.scores):
        reasons.append("grading_identity_drift")
    left, right = _slots(current), _slots(baseline)
    if left.keys() != right.keys():
        reasons.append("case_or_repeat_cohort_changed")
    current_runs = {run.run.run_id: run for run in current.inputs.runs}
    baseline_runs = {run.run.run_id: run for run in baseline.inputs.runs}
    for key in sorted(left.keys() & right.keys()):
        now, before = left[key], right[key]
        prefix = "/".join(key)
        if set(now.required_checks) != set(before.required_checks):
            reasons.append(f"{prefix}:required_checks_changed")
        if now.run_id is None or before.run_id is None:
            reasons.append(f"{prefix}:repeat_not_run")
            continue
        issues = _run_reasons(current_runs[now.run_id], baseline_runs[before.run_id], request)
        if _grading_identities(current, now.run_id) != _grading_identities(baseline, before.run_id):
            issues.append("grader_or_projection_changed")
        reasons.extend(f"{prefix}:{issue}" for issue in issues)
    return ComparisonDecision(status="incomparable" if reasons else "comparable", reasons=tuple(reasons))


def _measurement_shape(run: RunMeasurement) -> tuple[str, ...]:
    """参数为一次运行的原子测量；保留适用状态及分母，分子变化属于结果。"""
    return tuple(sorted(snapshot({"status": metric.status, "denominator": metric.denominator}) for metric in run.measurements))


def _case_pairs(current: MetricGroup, baseline: MetricGroup) -> tuple[list[dict[str, Any]], list[str]]:
    """参数为同一指标的两组题级结果；返回逐题配对及缺少观测的原因。"""
    left = {(case.case_id, case.case_version): case for case in current.cases}
    right = {(case.case_id, case.case_version): case for case in baseline.cases}
    if left.keys() != right.keys():
        return [], ["metric_case_cohort_changed"]
    pairs: list[dict[str, Any]] = []
    reasons: list[str] = []
    for key in sorted(left):
        now, before = left[key], right[key]
        current_runs = {run.repeat_id: run for run in now.runs}
        baseline_runs = {run.repeat_id: run for run in before.runs}
        if current_runs.keys() != baseline_runs.keys():
            reasons.append(f"{key[0]}:metric_repeat_cohort_changed")
            continue
        for repeat_id in current_runs:
            new_run, old_run = current_runs[repeat_id], baseline_runs[repeat_id]
            if new_run.status in {"missing", "invalid_input"} or old_run.status in {"missing", "invalid_input"}:
                reasons.append(f"{key[0]}/{repeat_id}:metric_coverage_incomplete")
            elif new_run.status != old_run.status or _measurement_shape(new_run) != _measurement_shape(old_run):
                reasons.append(f"{key[0]}/{repeat_id}:metric_applicability_or_denominator_changed")
        if now.value is not None and before.value is not None:
            pairs.append({"case_id": key[0], "case_version": key[1], "current": now.value,
                          "baseline": before.value, "delta": now.value - before.value,
                          "current_runs": [run.run_id for run in now.runs],
                          "baseline_runs": [run.run_id for run in before.runs]})
    return pairs, reasons


def _metric_configuration(summary: SuiteSummary, group: MetricGroup) -> tuple[str, ...]:
    """参数为汇总和指标分组；保留原报告的 k，排序和调用阶段不能隐式变化。"""
    key = read_snapshot(group.key_snapshot)
    return tuple(sorted({snapshot({"top_k": read_snapshot(score.detail_snapshot).get("top_k")})
                         for score in summary.inputs.scores
                         if score.kind == key["kind"] and score.stage == key["stage"]
                         and score.grading.grader == key["grader"]
                         and any(metric.name == group.metric for metric in score.grading.measurements)}))


def _compare_group(current: MetricGroup, baseline: MetricGroup, summaries: tuple[SuiteSummary, SuiteSummary]) -> MetricDelta:
    """参数为两个同义指标分组和汇总；仅在相同测量资格及分母下计算差值。"""
    pairs, reasons = _case_pairs(current, baseline)
    if _metric_configuration(summaries[0], current) != _metric_configuration(summaries[1], baseline):
        reasons.append("metric_configuration_changed")
    if not pairs and not reasons:
        reasons.append("metric_not_applicable_on_both_sides")
    if reasons:
        return MetricDelta(key_snapshot=current.key_snapshot, metric=current.metric, status="incomparable",
                           reasons=tuple(reasons), pairs_snapshot=snapshot({"pairs": []}))
    if current.mean is None or baseline.mean is None:
        raise ValueError("complete paired measurements require both means")
    return MetricDelta(key_snapshot=current.key_snapshot, metric=current.metric, status="comparable", reasons=(),
                       current=current.mean, baseline=baseline.mean, delta=current.mean - baseline.mean,
                       pairs_snapshot=snapshot({"pairs": pairs}))


def _compare_method_group(current: MetricGroup, baseline: MetricGroup) -> MetricDelta:
    """参数为两方法同指标组；仅配对双方已测的同题重复，并完整保留排除原因和预定分母。"""
    left = {(case.case_id, case.case_version): case for case in current.cases}
    right = {(case.case_id, case.case_version): case for case in baseline.cases}
    pairs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right:
            excluded.append({"case_id": key[0], "reason": "metric_case_missing"})
            continue
        now = {run.repeat_id: run for run in left[key].runs}
        old = {run.repeat_id: run for run in right[key].runs}
        values = []
        for repeat in sorted(now.keys() | old.keys()):
            new, before = now.get(repeat), old.get(repeat)
            if new is None or before is None or new.status != "measured" or before.status != "measured":
                excluded.append({"case_id": key[0], "repeat_id": repeat, "reason": "paired_metric_not_measured",
                                 "current": new.model_dump(mode="json") if new else None,
                                 "baseline": before.model_dump(mode="json") if before else None})
            elif _measurement_shape(new) != _measurement_shape(before):
                excluded.append({"case_id": key[0], "repeat_id": repeat, "reason": "metric_denominator_changed"})
            else:
                assert new.value is not None and before.value is not None
                values.append((new.value, before.value, new.run_id, before.run_id))
        if values:
            new_mean = sum(value[0] for value in values) / len(values)
            old_mean = sum(value[1] for value in values) / len(values)
            pairs.append({"case_id": key[0], "case_version": key[1], "current": new_mean, "baseline": old_mean,
                          "delta": new_mean - old_mean, "current_runs": [value[2] for value in values],
                          "baseline_runs": [value[3] for value in values]})
    current_mean = sum(pair["current"] for pair in pairs) / len(pairs) if pairs else None
    baseline_mean = sum(pair["baseline"] for pair in pairs) / len(pairs) if pairs else None
    return MetricDelta(key_snapshot=current.key_snapshot, metric=current.metric,
                       status="comparable" if pairs else "incomparable",
                       reasons=("paired_coverage_partial",) if pairs and excluded else (() if pairs else ("no_paired_measurements",)),
                       current=current_mean, baseline=baseline_mean,
                       delta=current_mean - baseline_mean if current_mean is not None and baseline_mean is not None else None,
                       pairs_snapshot=snapshot({"pairs": pairs, "excluded": excluded,
                                                "expected_cases": len(left.keys() | right.keys()), "paired_cases": len(pairs)}))


def compare_matched_measurements(current: SuiteSummary, baseline: SuiteSummary, request: ComparisonRequest) -> ComparisonResult:
    """参数为已具备身份资格的汇总及变化项；逐指标保留未测量原因，不取有分交集。"""
    left = {(group.key_snapshot, group.metric): group for group in current.groups}
    right = {(group.key_snapshot, group.metric): group for group in baseline.groups}
    deltas = []
    reasons = []
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right:
            delta = MetricDelta(key_snapshot=key[0], metric=key[1], status="incomparable",
                                reasons=("metric_or_group_missing_on_one_side",), pairs_snapshot=snapshot({"pairs": []}))
        else:
            if request.changed_factors == ("retrieval_method",) and _metric_configuration(current, left[key]) == _metric_configuration(baseline, right[key]):
                delta = _compare_method_group(left[key], right[key])
            else:
                delta = _compare_group(left[key], right[key], (current, baseline))
        deltas.append(delta)
        if delta.status == "incomparable" and delta.reasons != ("metric_not_applicable_on_both_sides",):
            reasons.append(f"{delta.metric}:metric_comparison_incomplete")
    if not any(delta.status == "comparable" for delta in deltas):
        reasons.append("no_comparable_measured_metrics")
    return ComparisonResult(status="incomparable" if reasons else "comparable", reasons=tuple(reasons),
                            request=request, deltas=tuple(deltas))


def compare_suites(current: SuiteSummary, baseline: SuiteSummary, request: ComparisonRequest) -> ComparisonResult:
    """参数为当前结果、基线和变化项；返回逐题差值或明确的不可比原因。"""
    decision = assess_comparability(current, baseline, request)
    if decision.status == "incomparable":
        return ComparisonResult(status="incomparable", reasons=decision.reasons, request=request, deltas=())
    return compare_matched_measurements(current, baseline, request)
