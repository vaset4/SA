"""Claim/Citation、人工作答和 judge 的分层报告。作者：xxx。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .agreement import compare_annotations
from .contracts import CalibrationRecord, DIMENSIONS
from .judge import compare_judge_to_human

_STATUSES = ("measured", "not_measured", "missing", "invalid_input")


def build_calibration_report(records: Sequence[CalibrationRecord]) -> dict[str, Any]:
    """参数为校准记录；按来源、版本和人工资格分层生成报告。"""
    rows = [_record_row(record) for record in records]
    eligible = [row for row in rows if row["human_quality_eligible"]]
    return {
        "schema_version": "claim-calibration-report-v1",
        "record_count": len(rows), "human_quality_denominator": len(eligible),
        "rows": rows, "metric_coverage": _metric_coverage(rows),
        "breakdowns": _breakdowns(rows), "rubric_distributions": _rubric_distributions(records),
        "version_summary": _version_summary(records),
        "error_labels": dict(Counter(label for row in rows for label in row["error_labels"])),
    }


def _record_row(record: CalibrationRecord) -> dict[str, Any]:
    """参数为单条校准记录；返回保留各层版本和原始结果的报告行。"""
    extraction = record.extraction
    annotations = list(record.annotations)
    agreement = compare_annotations(annotations[0], annotations[1]).model_dump(mode="json") if len(annotations) == 2 else None
    judge_comparison = compare_judge_to_human(record.judge, annotations[0]) if record.judge and annotations else None
    human_errors = [label for item in annotations for claim in item.claim_annotations for label in claim.error_labels]
    arbitration_errors = [label for claim in (record.arbitration.resolved_claims if record.arbitration else ())
                          for label in claim.error_labels]
    error_labels = [*human_errors, *arbitration_errors, *(record.judge.error_labels if record.judge else ())]
    return {
        "case_id": extraction.case_id, "run_id": extraction.run_id,
        "run_mode": extraction.run_mode, "review_status": extraction.review_status,
        "human_quality_eligible": extraction.human_quality_eligible,
        "extraction_version": extraction.extraction_version,
        "projection_version": extraction.projection_version,
        "projection_hash": extraction.projection_hash,
        "claim_metrics": record.claim_metrics.model_dump(mode="json"),
        "annotations": [item.model_dump(mode="json") for item in annotations],
        "agreement": agreement,
        "arbitration": record.arbitration.model_dump(mode="json") if record.arbitration else None,
        "judge": record.judge.model_dump(mode="json") if record.judge else None,
        "judge_comparison": judge_comparison, "error_labels": error_labels,
    }


def _metric_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """参数为报告行；按四态统计确定性指标覆盖。"""
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for metric in row["claim_metrics"]["metrics"]:
            counters[metric["name"]][metric["status"]] += 1
    return {name: {status: counts[status] for status in _STATUSES} for name, counts in sorted(counters.items())}


def _breakdowns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """参数为报告行；按运行、审核和抽取版本分开计数。"""
    groups = Counter((row["run_mode"], row["review_status"], row["extraction_version"], row["projection_version"])
                     for row in rows)
    return [{"run_mode": key[0], "review_status": key[1], "extraction_version": key[2],
             "projection_version": key[3], "count": count}
            for key, count in sorted(groups.items())]


def _version_summary(records: Sequence[CalibrationRecord]) -> dict[str, dict[str, int]]:
    """参数为校准记录；分别统计抽取、投影、rubric 和 judge 版本。"""
    extraction = Counter(item.extraction.extraction_version for item in records)
    projection = Counter(item.extraction.projection_version for item in records)
    rubric = Counter(annotation.rubric_version for item in records for annotation in item.annotations)
    judge = Counter(item.judge.judge_version for item in records if item.judge is not None)
    return {
        "extraction": dict(extraction), "projection": dict(projection),
        "rubric": dict(rubric), "judge": dict(judge),
    }


def _rubric_distributions(records: Sequence[CalibrationRecord]) -> dict[str, Any]:
    """参数为校准记录；仅汇总真实合格回答的人工六维分布。"""
    result: dict[str, Counter[int]] = {dimension: Counter() for dimension in DIMENSIONS}
    versions: Counter[str] = Counter()
    for record in records:
        if not record.extraction.human_quality_eligible:
            continue
        for annotation in record.annotations:
            versions[annotation.rubric_version] += 1
            for dimension in DIMENSIONS:
                result[dimension][annotation.dimensions[dimension]] += 1
    return {
        "rubric_versions": dict(versions),
        "dimensions": {dimension: {str(score): result[dimension][score] for score in (0, 1, 2)} for dimension in DIMENSIONS},
    }


def write_calibration_report(path: Path, report: Mapping[str, Any]) -> None:
    """参数为目标路径和报告；首次写入或相同内容复用，拒绝覆盖不同历史结果。"""
    content = json.dumps(dict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"claim calibration report differs: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8", newline="\n")
