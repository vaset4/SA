"""校准报告和命令入口回归。作者：xxx。"""

from __future__ import annotations

import json

from evals.claim_calibration.annotation import create_annotation
from evals.claim_calibration.contracts import CalibrationRecord, ClaimExtraction, ClaimMetricReport, DIMENSIONS
from evals.claim_calibration.report import build_calibration_report, write_calibration_report
from evals.claim_calibration.run_eval import run_evaluation
from evals.contracts import MetricResult


def _record(*, eligible: bool) -> CalibrationRecord:
    """参数为人工资格；返回最小完整校准报告行。"""
    extraction = ClaimExtraction(
        case_id="case-live" if eligible else "case-fixture", run_id="run-1" if eligible else "run-2",
        run_mode="agent_live" if eligible else "agent_fixture", review_status="reviewed",
        human_quality_eligible=eligible, status="measured", reason="ok", items=(),
        projection_hash="0" * 64,
    )
    metrics = ClaimMetricReport(
        status="measured", reason="ok", metrics=(MetricResult(
            name="citation_completeness", status="measured", reason="ok",
            value=1.0, numerator=1.0, denominator=1.0,
        ),),
    )
    annotations = ()
    if eligible:
        annotations = (create_annotation(
            extraction, annotation_id="a1", annotator_id="u1",
            dimensions={dimension: 2 for dimension in DIMENSIONS},
            reasons={dimension: f"{dimension} 已核对" for dimension in DIMENSIONS},
        ),)
    return CalibrationRecord(extraction=extraction, claim_metrics=metrics, annotations=annotations)


def test_report_excludes_fixture_from_human_quality_denominator() -> None:
    """无参数；验证报告分母只包含真实已审核 agent_live 回答。"""
    report = build_calibration_report([_record(eligible=True), _record(eligible=False)])
    assert report["record_count"] == 2
    assert report["human_quality_denominator"] == 1
    assert report["rubric_distributions"]["dimensions"]["task_completion"]["2"] == 1


def test_report_write_refuses_to_overwrite_different_history(tmp_path) -> None:
    """参数为临时目录；验证历史报告不会被不同版本结果覆盖。"""
    path = tmp_path / "report.json"
    write_calibration_report(path, {"version": "v1"})
    write_calibration_report(path, {"version": "v1"})
    try:
        write_calibration_report(path, {"version": "v2"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("different historical report must not be overwritten")


def test_run_eval_validates_and_reports_versioned_records(tmp_path) -> None:
    """参数为临时目录；验证 JSONL 入口能校验并重放分层报告。"""
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(_record(eligible=True).model_dump(mode="json"), ensure_ascii=False) + "\n", encoding="utf-8")
    validation = run_evaluation(path, adapter="validate")
    report = run_evaluation(path, adapter="replay")
    assert validation["human_quality_denominator"] == 1
    assert report["metric_coverage"]["citation_completeness"]["measured"] == 1
