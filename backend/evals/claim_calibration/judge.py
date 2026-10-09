"""自动 judge 的诊断结果与人工对照。作者：xxx。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evals.contracts import snapshot

from .contracts import (
    AnnotationRecord, DiagnosticJudgeResult, HumanDimension, HumanScore, JUDGE_VERSION,
    SemanticErrorLabel,
)


def record_judge_result(
    *, judge_id: str, case_id: str, run_id: str, request: Mapping[str, Any],
    response: Mapping[str, Any], dimensions: Mapping[HumanDimension, HumanScore] | None = None,
    claim_labels: Mapping[str, str] | None = None, error_labels: tuple[SemanticErrorLabel, ...] = (),
    judge_version: str = JUDGE_VERSION,
) -> DiagnosticJudgeResult:
    """参数为真实 judge 请求/响应及解析结论；返回只用于诊断的版本化记录。"""
    return DiagnosticJudgeResult(
        judge_id=judge_id, case_id=case_id, run_id=run_id, judge_version=judge_version,
        dimensions=dict(dimensions or {}), claim_labels=dict(claim_labels or {}),
        request_snapshot=snapshot(dict(request)), response_snapshot=snapshot(dict(response)),
        error_labels=error_labels,
    )


def compare_judge_to_human(
    judge: DiagnosticJudgeResult, annotation: AnnotationRecord,
) -> dict[str, Any]:
    """参数为诊断 judge 和人工标注；返回逐维一致性及具体错误标签。"""
    if (judge.case_id, judge.run_id) != (annotation.case_id, annotation.run_id):
        raise ValueError("judge and annotation identity mismatch")
    comparable = [dimension for dimension in annotation.dimensions if dimension in judge.dimensions]
    agreement = {
        dimension: judge.dimensions[dimension] == annotation.dimensions[dimension]
        for dimension in comparable
    }
    return {
        "judge_id": judge.judge_id, "judge_version": judge.judge_version,
        "rubric_version": annotation.rubric_version, "status": "diagnostic",
        "comparable_dimensions": len(comparable), "dimension_agreement": agreement,
        "agreement_rate": sum(agreement.values()) / len(agreement) if agreement else None,
        "error_labels": list(judge.error_labels), "hard_gate_eligible": False,
    }
