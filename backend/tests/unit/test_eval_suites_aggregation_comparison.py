"""验证题级分母、重复成功、来源分离及比较前的实验条件。作者：xxx。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.agent_outcome.report import build_agent_report
from evals.contracts import MetricResult, build_grading, read_snapshot, snapshot
from evals.suites.adapters import load_suite_sources
from evals.suites.aggregate import aggregate_suite
from evals.suites.compare import compare_suites
from evals.suites.contracts import ComparisonRequest, ReportKind, RunSlot, SuiteSummary
from evals.suites.report import build_suite_report, report_exit_code
from test_eval_suites_contracts_adapters import (
    make_agent_record, make_evidence_record, make_planner_record, make_slots,
    measured, write_manifest, write_source,
)


def make_summary(root: Path, records: tuple[dict[str, Any], ...], *, kind: ReportKind = "planner",
                 identity_overrides: dict[str, str] | None = None, slots: tuple[RunSlot, ...] | None = None) -> SuiteSummary:
    """参数为原记录和来源选项；通过实际文件导入、校验及聚合返回测试汇总。"""
    source = write_source(root, kind, records, with_identity=True, identity_overrides=identity_overrides)
    path = write_manifest(root, (source,), make_slots(records) if slots is None else slots)
    return aggregate_suite(load_suite_sources(path))


def evidence_calls(case_id: str, *, count: int, hit: bool) -> dict[str, Any]:
    """参数为案例及实际查询次数；构造独立事件，再由原 Evidence 评分器计算结果。"""
    record = make_evidence_record(case_id)
    observed = read_snapshot(record["run_record"]["observation_snapshot"])
    original = observed["evidence_observations"][0]
    events = []
    for number in range(count):
        event = json.loads(snapshot(original))
        event["event_id"] = f"event-{number}"
        event["scope"]["scope_id"] = f"scope-{number}"
        if not hit:
            event["payload"]["hits"] = []
        events.append(event)
    changed = {**observed, "evidence_observations": events}
    return {**record, "run_record": {**record["run_record"], "observation_snapshot": snapshot(changed)}}


def test_evidence_call_count_does_not_increase_case_weight(tmp_path: Path) -> None:
    """参数为目录；一题零分一次、另一题满分十次，题级均值仍为二分之一。"""
    records = (evidence_calls("one", count=1, hit=False), evidence_calls("many", count=10, hit=True))
    summary = make_summary(tmp_path, records, kind="evidence")
    recall = next(group for group in summary.groups if group.metric == "evidence_group_recall_at_k")
    assert recall.mean == 0.5
    assert recall.complete_case_count == 2
    assert next(state.count for state in recall.states if state.status == "measured") == 11


def test_modes_and_grader_versions_are_not_averaged_together(tmp_path: Path) -> None:
    """参数为目录；不同运行方式或评分器版本保持独立分组。"""
    first = make_planner_record("first", metrics=(measured("plan_valid_rate", 0),))
    second = make_planner_record("second", metrics=(measured("plan_valid_rate", 1),))
    second = {**second, "run_record": {**second["run_record"], "run_mode": "planner_live_fixture"}}
    second["grading_result"] = build_grading(second, [measured("plan_valid_rate", 1)], grader="research_planner", grader_version="grader-v2")
    summary = make_summary(tmp_path, (first, second))
    assert len(summary.groups) == 2
    assert sorted(group.mean for group in summary.groups) == [0, 1]
    assert {read_snapshot(group.key_snapshot)["grader_version"] for group in summary.groups} == {"eval-contract-v1", "grader-v2"}


def test_three_runs_with_one_success_are_not_all_successful(tmp_path: Path) -> None:
    """参数为目录；三次中一次成功只能增加机会指标，不能成为全部成功。"""
    records = tuple(make_agent_record(success=success) for success in (True, False, False))
    summary = make_summary(tmp_path, records, kind="agent")
    result = summary.repetitions[0]
    assert result.expected_runs == result.measured_runs == 3
    assert result.successful_runs == 1
    assert result.single_run_success_rate == pytest.approx(1 / 3)
    assert result.any_success is True
    assert result.all_success is False


def test_unexecuted_repeat_keeps_all_success_unknown(tmp_path: Path) -> None:
    """参数为目录；漏跑第三次时不能用两次成功宣称三次全部成功。"""
    records = (make_agent_record(), make_agent_record())
    slots = (*make_slots(records), RunSlot(case_id="agent-a", case_version="case-v1", repeat_id="repeat-3",
                                           run_id=None, required_checks=("truth_leakage",)))
    summary = make_summary(tmp_path, records, kind="agent", slots=slots)
    result = summary.repetitions[0]
    assert result.measured_runs == 2 and result.expected_runs == 3
    assert result.any_success is True and result.all_success is None
    assert result.single_run_success_rate is None
    assert report_exit_code(build_suite_report(summary.inputs, summary)) == 1


def test_declared_model_change_compares_same_context(tmp_path: Path) -> None:
    """参数为目录；模型是唯一声明变化项时，固定上下文允许逐题比较。"""
    baseline = make_summary(tmp_path / "before", (make_planner_record(metrics=(measured("plan_valid_rate", 0),)),),
                            identity_overrides={"model": "model-a"})
    current = make_summary(tmp_path / "after", (make_planner_record(metrics=(measured("plan_valid_rate", 1),)),),
                           identity_overrides={"model": "model-b"})
    result = compare_suites(current, baseline, ComparisonRequest(changed_factors=("model",)))
    assert result.status == "comparable"
    assert result.deltas[0].delta == 1
    pairs = read_snapshot(result.deltas[0].pairs_snapshot)["pairs"]
    assert pairs[0]["current_runs"] != pairs[0]["baseline_runs"]


@pytest.mark.parametrize("factor", ("model", "code", "prompt", "evaluator", "environment"))
def test_undeclared_condition_change_blocks_all_delta(tmp_path: Path, factor: str) -> None:
    """参数为目录及变化条件；相同分数不能掩盖未经声明的实验变化。"""
    records = (make_planner_record(metrics=(measured("plan_valid_rate", 1),)),)
    baseline = make_summary(tmp_path / "before", records)
    current = make_summary(tmp_path / "after", records, identity_overrides={factor: "changed"})
    result = compare_suites(current, baseline, ComparisonRequest(changed_factors=()))
    assert result.status == "incomparable" and result.deltas == ()
    assert any(f"undeclared_change:{factor}" in reason for reason in result.reasons)


@pytest.mark.parametrize("region", ("input", "truth"))
def test_input_or_truth_change_cannot_be_allowed_as_model_change(tmp_path: Path, region: str) -> None:
    """参数为目录和变更区域；模型对比不能同时换上下文或评分标准。"""
    original = make_planner_record(metrics=(measured("plan_valid_rate", 1),))
    changed = json.loads(snapshot(original))
    if region == "input":
        changed["run_record"]["input_snapshot"] = snapshot({"question": "不同问题"})
    else:
        changed["eval_case"]["truth_snapshot"] = snapshot({"expected": {"min_units": 2}})
    changed["grading_result"] = build_grading(changed, [measured("plan_valid_rate", 1)], grader="research_planner")
    baseline = make_summary(tmp_path / "before", (original,))
    current = make_summary(tmp_path / "after", (changed,), identity_overrides={"model": "other-model"})
    result = compare_suites(current, baseline, ComparisonRequest(changed_factors=("model",)))
    assert result.status == "incomparable"
    assert any(f"undeclared_change:{region}" in reason for reason in result.reasons)


def test_case_cohort_change_is_not_reduced_to_intersection(tmp_path: Path) -> None:
    """参数为目录；新增或删除题目时禁止只取共同题目后生成总体改善值。"""
    first = make_planner_record("one", metrics=(measured("plan_valid_rate", 1),))
    second = make_planner_record("two", metrics=(measured("plan_valid_rate", 1),))
    baseline = make_summary(tmp_path / "before", (first,))
    current = make_summary(tmp_path / "after", (first, second))
    result = compare_suites(current, baseline, ComparisonRequest(changed_factors=()))
    assert result.status == "incomparable" and result.deltas == ()
    assert "case_or_repeat_cohort_changed" in result.reasons


def test_missing_metric_has_no_numeric_delta(tmp_path: Path) -> None:
    """参数为目录；两边身份相同但当前指标缺数据时，不补零也不当通过。"""
    baseline = make_summary(tmp_path / "before", (make_planner_record(metrics=(measured("plan_valid_rate", 1),)),))
    missing = MetricResult(name="plan_valid_rate", status="missing", reason="original_observation_missing")
    current = make_summary(tmp_path / "after", (make_planner_record(metrics=(missing,)),))
    result = compare_suites(current, baseline, ComparisonRequest(changed_factors=()))
    assert result.status == "incomparable"
    assert result.deltas[0].delta is None
    assert any("coverage_incomplete" in reason for reason in result.deltas[0].reasons)


def test_changed_atomic_denominator_prevents_delta(tmp_path: Path) -> None:
    """参数为目录；比例相同但实际分母不同，也不能假定是同一测量范围。"""
    baseline = make_summary(tmp_path / "before", (make_planner_record(metrics=(measured("plan_valid_rate", 1, denominator=1),)),))
    current = make_summary(tmp_path / "after", (make_planner_record(metrics=(measured("plan_valid_rate", 1, denominator=2),)),))
    result = compare_suites(current, baseline, ComparisonRequest(changed_factors=()))
    assert result.status == "incomparable" and result.deltas[0].delta is None
    assert any("denominator_changed" in reason for reason in result.deltas[0].reasons)


def test_not_applicable_metrics_do_not_invent_scores(tmp_path: Path) -> None:
    """参数为目录；两边均不适用的指标保留原因，其他可测指标仍可比较。"""
    absent = MetricResult(name="optional", status="not_measured", reason="not_applicable")
    records = (make_planner_record(metrics=(measured("plan_valid_rate", 1), absent)),)
    baseline, current = make_summary(tmp_path / "before", records), make_summary(tmp_path / "after", records)
    result = compare_suites(current, baseline, ComparisonRequest(changed_factors=()))
    assert result.status == "comparable"
    optional = next(delta for delta in result.deltas if delta.metric == "optional")
    assert optional.delta is None and optional.reasons == ("metric_not_applicable_on_both_sides",)


def test_low_task_score_does_not_become_new_p0(tmp_path: Path) -> None:
    """参数为目录；任务未完成是质量结果，不自动升级为新的确定性违规。"""
    records = (make_agent_record(success=False),)
    summary = make_summary(tmp_path, records, kind="agent")
    assert next(group.mean for group in summary.groups if group.metric == "task_success") == 0
    assert report_exit_code(build_suite_report(summary.inputs, summary)) == 0


def test_missing_required_assertion_cannot_pass(tmp_path: Path) -> None:
    """参数为目录；尚未提供明确禁止事实断言时，适用检查保持 missing。"""
    records = (make_agent_record(),)
    summary = make_summary(tmp_path, records, kind="agent", slots=make_slots(records, required=("forbidden_fact",)))
    assert summary.checks[0].status == "missing"
    assert report_exit_code(build_suite_report(summary.inputs, summary)) == 1


def test_failed_terminal_reported_as_success_is_p0(tmp_path: Path) -> None:
    """参数为目录；原记录失败而评分声称成功时，汇总检查必须指出矛盾。"""
    records = (make_agent_record(success=False),)
    report = build_agent_report(records)
    report["rows"][0]["grading_result"] = build_grading(records[0], [measured("task_success", 1)], grader="agent_outcome",
                                                         grader_version="agent-outcome-v1")
    source = write_source(tmp_path, "agent", records, report=report)
    path = write_manifest(tmp_path, (source,), make_slots(records, required=("false_success",)))
    summary = aggregate_suite(load_suite_sources(path))
    assert summary.checks[0].status == "failed"
    assert report_exit_code(build_suite_report(summary.inputs, summary)) == 1


@pytest.mark.parametrize("purpose,oracle", (("online_discovery", False), ("regression", True)))
def test_online_and_oracle_do_not_enter_benchmark_mean(tmp_path: Path, purpose: str, oracle: bool) -> None:
    """参数为用途和诊断模式；线上发现和 Oracle 保留明细但不进入默认质量均值。"""
    record = make_evidence_record()
    if oracle:
        record = {**record, "run_record": {**record["run_record"], "run_mode": "oracle"}}
    records = (record,)
    source = write_source(tmp_path, "evidence", records)
    path = write_manifest(tmp_path, (source,), make_slots(records), purpose=purpose)
    summary = aggregate_suite(load_suite_sources(path))
    assert summary.groups == () and summary.repetitions == ()
    assert summary.inputs.scores
    assert not any(check.status == "failed" for check in summary.checks)
