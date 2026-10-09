"""【流程评测】【范围与引用】核对实际返回身份，不依赖原文精确映射。作者：xxx。"""

from __future__ import annotations

from typing import Any

from evals.contracts import MetricResult

from .workflow_execution import reference_checks
from .workflow_facts import WorkflowTurn, boolean_metric


def _observed_items(turn: WorkflowTurn) -> tuple[list[dict[str, Any]], bool]:
    """参数为实际轮次；仅读取真实证据/检索命中，不使用计划或标准答案造命中。"""
    records = turn.values.get("evidence_records")
    complete = isinstance(records, list)
    items = list(records) if isinstance(records, list) else []
    for event in turn.events:
        if event.get("stage") in {"query_raw", "query_filtered", "retrieval_merged", "retrieval_ranked"}:
            hits = event.get("payload", {}).get("hits")
            if isinstance(hits, list):
                items.extend(hits)
                complete = True
    return items, complete


def grade_scope(turns: tuple[WorkflowTurn, ...]) -> tuple[list[MetricResult], dict[str, Any]]:
    """参数为冻结轮次；返回实际范围符合和引用结构合法两项独立过程测量。"""
    scope_rows: list[dict[str, Any]] = []
    citation_rows: list[dict[str, Any]] = []
    for turn in turns:
        items, complete = _observed_items(turn)
        scope = turn.inputs.get("project_scope")
        allowed = scope.get("paper_ids") if isinstance(scope, dict) else None
        known_scope = isinstance(allowed, list)
        allowed_ids = set(allowed) if isinstance(allowed, list) else set()
        outside = [row.get("paper_id") for row in items if row.get("paper_id") and known_scope and row["paper_id"] not in allowed_ids]
        missing_ids = any(not row.get("paper_id") for row in items)
        passed = False if outside else None if not complete or not known_scope or missing_ids else True
        scope_rows.append({"turn": turn.number, "passed": passed, "outside_paper_ids": outside,
                           "observed_item_count": len(items), "missing_identity": missing_ids})
        issues, gaps = reference_checks(turn)
        reference_issues = [row for row in issues if row["code"] in {
            "unknown_evidence_reference", "citation_identity_mismatch", "citation_identity_missing",
        }]
        citation_missing = gaps or not turn.values
        citation_rows.append({"turn": turn.number, "passed": False if reference_issues else None if citation_missing else True,
                              "issues": reference_issues, "missing": gaps})
    metrics = [_all_observed("scope_adherence", scope_rows), _all_observed("citation_reference_validity", citation_rows)]
    return metrics, {"scope": scope_rows, "citations": citation_rows, "citation_semantics": "identity_binding_only"}


def _all_observed(name: str, rows: list[dict[str, Any]]) -> MetricResult:
    """参数为指标和各轮判断；明确越界优先失败，缺观察不能被其他轮次成功覆盖。"""
    states = [row["passed"] for row in rows]
    passed = False if False in states else None if not states or None in states else True
    return boolean_metric(name, passed, reason="all_observed_turn_identity_bindings_without_semantic_grading")
