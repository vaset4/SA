"""RAGAS 与确定性评分的测量状态和覆盖投影。作者：xxx。"""

from __future__ import annotations

import math
from typing import Any

from evals.contracts import MetricResult
from evals.ragas.eval_contracts import contract_measurements

RAGAS_METRICS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


def ragas_eligible(record: dict[str, Any]) -> bool:
    """参数为评分投影；是否具备当前四项 RAGAS 组合需要的实际输入。"""
    return all(bool(record.get(key)) for key in ("question", "response", "retrieved_contexts", "reference_answer"))


def build_measurements(
    record: dict[str, Any], scores: dict[str, Any], raw_grader_scores: dict[str, Any],
) -> list[MetricResult]:
    """参数为观察投影、项目分数和评分器原始分数；返回可审计的逐指标状态。"""
    results = contract_measurements(record)
    for name, score in scores.items():
        if name not in results:
            results[name] = _project_measurement(name, score, record)
    for name in RAGAS_METRICS:
        results[name] = _ragas_measurement(name, raw_grader_scores, record)
    return list(results.values())


def _ragas_measurement(name: str, raw: dict[str, Any], record: dict[str, Any]) -> MetricResult:
    """参数为 RAGAS 指标、原始返回和观察；区分未返回、数据不足与无效数字。"""
    value = raw.get(name)
    if not ragas_eligible(record):
        return MetricResult(name=name, status="missing", reason="ragas_input_incomplete")
    if name not in raw or value is None:
        return MetricResult(name=name, status="missing", reason="grader_did_not_return_sample_score")
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        return MetricResult(name=name, status="invalid_input", reason="grader_score_not_finite_number")
    return MetricResult(name=name, status="measured", value=value, reason="external_grader_score")


def _project_measurement(name: str, score: Any, record: dict[str, Any]) -> MetricResult:
    """参数为项目指标、结果和投影；为既有评分补充适用条件及缺口原因。"""
    metadata = record.get("metadata", {})
    if name == "synthesis_payload_present" and "synthesis_state" not in metadata and metadata.get("dialogue_mode") == "chat":
        return MetricResult(name=name, status="not_measured", reason="chat_has_no_synthesis_requirement")
    if name == "scope_adherence" and score is None:
        return MetricResult(name=name, status="not_measured", reason="no_scope_constraint")
    if name == "citation_presence_rate" and score is None:
        return MetricResult(name=name, status="not_measured", reason="citation_not_required")
    required = {
        "route_correctness": ("requested_mode", "resolved_mode"),
        "synthesis_payload_present": ("synthesis_state",),
        "dialogue_mode_alignment": ("dialogue_mode",),
    }.get(name, ())
    if required and not all(key in metadata for key in required):
        return MetricResult(name=name, status="missing", reason="runtime_metadata_incomplete")
    if "response" not in record and name in {"citation_presence_rate", "placeholder_citation_rate", "answer_payload_present"}:
        return MetricResult(name=name, status="missing", reason="response_missing")
    if score is None:
        return MetricResult(name=name, status="missing", reason="metric_input_incomplete")
    counts = {"numerator": int(score), "denominator": 1} if name == "citation_presence_rate" else {}
    return MetricResult(name=name, status="measured", value=score, reason="legacy_rule", **counts)


def integrity_checks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """参数为已评分记录；仅检查适用的合同缺口和已知无效输入，不放宽现有门禁。"""
    failures = []
    for record in records:
        truth = record["eval_case"]["truth_snapshot"]
        for metric in record["grading_result"]["measurements"]:
            if _blocks_integrity(metric, truth):
                failures.append({"run_id": record["run_record"]["run_id"], **metric})
    return [{"name": "evaluation_input_integrity", "passed": not failures, "details": {"failures": failures}}]


def _blocks_integrity(metric: dict[str, Any], truth_snapshot: str) -> bool:
    """参数为指标和冻结真值；返回确有适用条件的缺数据或无效输入是否阻断。"""
    import json

    truth = json.loads(truth_snapshot)
    if metric["status"] == "invalid_input":
        return True
    if metric["status"] != "missing":
        return False
    truth_field = {"golden_evidence_presence_rate": "golden_evidence", "critical_fact_coverage": "critical_facts",
                   "critical_fact_error_rate": "critical_facts", "fact_contract_score": "critical_facts"}.get(metric["name"])
    return bool(truth_field and truth.get(truth_field))
