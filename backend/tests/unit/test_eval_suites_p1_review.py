"""P1 独立复核发现的负例与原始评分导出回归。作者：xxx。"""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from evals.contracts import read_snapshot, snapshot


@pytest.mark.parametrize("allowed,expected_scope", [("lsttn", 1.0), ("other-paper", 0.0)])
def test_negative_case_keeps_actual_scope_and_citation_checks(allowed: str, expected_scope: float) -> None:
    """参数为实际授权范围与预期；无正例证明仍校验实际取证范围及返回引用。"""
    from evals.agent_outcome.scoring import grade_agent_record
    from test_agent_outcome_evidence import _captured

    record = deepcopy(_captured())
    truth = read_snapshot(record["agent_case"]["truth_snapshot"])
    truth.update(answerability="insufficient_evidence", required_outcomes=[], evidence_targets={
        "required_span_ids": [], "required_group_ids": [], "required_paper_ids": [],
    })
    truth["evidence_benchmark"].update(spans=[], groups=[], target_paper_ids=[], paper_requirements=[])
    for key in ("agent_case", "eval_case"):
        record[key]["truth_snapshot"] = snapshot(truth)
    observed = read_snapshot(record["agent_run"]["observation_snapshot"])
    inputs = read_snapshot(observed["turns"][0]["input_snapshot"])
    inputs["project_scope"]["paper_ids"] = [allowed]
    observed["turns"][0]["input_snapshot"] = snapshot(inputs)
    for key in ("agent_run", "run_record"):
        record[key]["observation_snapshot"] = snapshot(observed)
    metrics = {item["name"]: item for item in grade_agent_record(record)["measurements"]}
    assert metrics["task_success"]["status"] == "not_measured"
    assert metrics["scope_adherence"]["status"] == "measured"
    assert metrics["scope_adherence"]["value"] == expected_scope
    assert metrics["citation_verifiability_rate"]["value"] == 1.0


def test_csv_preserves_original_measurements_and_missing_slots(tmp_path: Path) -> None:
    """参数为临时目录；CSV原样保留分数/分母和缺失状态，不按成功行删掉未运行槽位。"""
    from evals.suites.adapters import load_suite_sources
    from evals.suites.aggregate import aggregate_suite
    from evals.suites.contracts import RunSlot
    from evals.suites.report import build_suite_report, write_suite_report
    from test_eval_suites_contracts_adapters import make_evidence_record, make_slots, write_manifest, write_source

    records = (make_evidence_record(),)
    source = write_source(tmp_path, "evidence", records)
    missing = RunSlot(case_id="not-run", case_version="v1", repeat_id="1", run_id=None, required_checks=())
    path = write_manifest(tmp_path, (source,), (*make_slots(records), missing))
    inputs = load_suite_sources(path)
    report = build_suite_report(inputs, aggregate_suite(inputs))
    write_suite_report(tmp_path / "report", report)
    with (tmp_path / "report/results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    originals = [metric for score in inputs.scores for metric in score.grading.measurements]
    assert len(rows) == len(originals) + 1
    for row, metric in zip(rows, originals):
        assert row["metric"] == metric.name and row["status"] == metric.status
        assert row["value"] == ("" if metric.value is None else str(metric.value))
        assert row["denominator"] == ("" if metric.denominator is None else str(metric.denominator))
    assert rows[-1]["case_id"] == "not-run" and rows[-1]["status"] == "missing"
    assert rows[-1]["value"] == "" and rows[-1]["run_id"] == ""
