"""【流程评测】【结构化】首次响应、缓存、回修和业务校验分开计数。作者：xxx。"""

from __future__ import annotations

from typing import Any

from evals.contracts import MetricResult

from .workflow_facts import WorkflowTurn, coverage, event_rows, ratio
from .workflow_faults import api_failure_observed


def _operation_result(events: list[dict[str, Any]]) -> dict[str, Any]:
    """参数为同一真实结构操作的有序事件；返回完整响应条件结果和所有原始尝试。"""
    starts = [row for row in events if row["stage"] == "structured_operation_start"]
    ends = [row for row in events if row["stage"] == "structured_operation_end"]
    responses = [row for row in events if row["stage"] == "structured_response_validation"]
    generated = [row for row in responses if row.get("cache_hit") is False]
    cached = [row for row in responses if row.get("cache_hit") is True]
    http = [row for row in events if row["stage"] == "model_response_usage"]
    first = generated[0] if generated else None
    end = ends[-1] if ends else {}
    api_failed = bool(http and (http[-1].get("status") == "failed" or http[-1].get("protocol_status") == "failed"))
    api_boundary = bool(http and api_failure_observed(http[-1]))
    api_blocked = api_failed and api_boundary and end.get("status") == "failed"
    first_status = "missing" if first is None else _response_state(first)
    # 1. 【流程评测】【结构化】缓存对象可合法，但不构成一次新的模型生成
    if first is None and cached and len(cached) == len(responses):
        first_status = "cache"
    final_status = _final_status(responses, ended=end, complete=bool(starts and ends), api_blocked=api_blocked)
    return {
        "operation_id": events[0]["operation_id"], "turn": events[0]["turn"],
        "schema": starts[0].get("schema") if starts else events[0].get("schema"), "first_status": first_status,
        "final_status": final_status, "api_blocked": api_blocked,
        "repair_count": end.get("repair_count"), "cache_responses": len(cached),
        "generated_responses": len(generated), "responses": responses, "http_attempts": http,
        "paths": [row for row in events if row["stage"] == "structured_path"],
        "observation_complete": bool(starts and ends),
    }


def _final_status(responses: list[dict[str, Any]], *, ended: dict[str, Any], complete: bool, api_blocked: bool) -> str:
    """参数为完整响应、操作终态和API证据；区分缓存、生成失败、外部阻塞及观察缺口。"""
    if not complete:
        return "missing"
    if api_blocked:
        return "api_blocked"
    if not responses:
        return "missing"
    last = responses[-1]
    if not isinstance(last.get("cache_hit"), bool):
        return "missing"
    if last.get("schema_valid") is False:
        return "cache_invalid" if last["cache_hit"] else "failed"
    if ended.get("status") != "success" or last.get("schema_valid") is not True:
        return "missing"
    if last["cache_hit"]:
        return "cache_valid"
    return "repaired" if ended.get("repair_succeeded") is True else "direct"


def _response_state(response: dict[str, Any]) -> str:
    """参数为确已返回的响应验证；只接受明确布尔，不把空字段当合规。"""
    value = response.get("schema_valid")
    return "passed" if value is True else "failed" if value is False else "missing"


def grade_structure(turns: tuple[WorkflowTurn, ...]) -> tuple[list[MetricResult], dict[str, Any]]:
    """参数为冻结轮次；返回条件指标、分母覆盖以及每操作的首次与最终状态。"""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    unlinked: list[dict[str, Any]] = []
    stages = {"structured_operation_start", "structured_operation_end", "structured_path",
              "structured_response_validation", "structured_response_unavailable", "model_response_usage"}
    for turn in turns:
        for event in turn.events:
            if event.get("stage") not in stages:
                continue
            payload = {**event["payload"], "stage": event["stage"], "turn": turn.number,
                       "event_id": event.get("event_id")}
            identity = payload.get("operation_id")
            if not identity:
                if event["stage"] != "model_response_usage":
                    unlinked.append(payload)
                continue
            grouped.setdefault((turn.number, identity), []).append(payload)
    operations = [_operation_result(rows) for rows in grouped.values()]
    first = coverage(operations, "first_status")
    final = coverage(operations, "final_status")
    business = event_rows(turns, "structured_business_validation")
    valid_business = [row for row in business if isinstance(row.get("valid"), bool)]
    business_schemas = {"ResearchPlanDraft", "ResearchClaimDraftBatch", "PlannerFinalAnswerDraft",
                        "ResearchPlanStrategyDraft", "ResearchReplanStrategyDraft", "ResearchClaimStrategyDraftBatch"}
    requiring_business = [row for row in operations if row.get("schema") in business_schemas]
    linked = {(row["turn"], row.get("operation_id")) for row in valid_business if row.get("operation_id")}
    business_missing = [row["operation_id"] for row in requiring_business if (row["turn"], row["operation_id"]) not in linked]
    metrics = [
        ratio("structured_observation_coverage", sum(row["observation_complete"] for row in operations),
              len(operations) + len(unlinked), reason="started_and_ended_operations_over_all_observed_operations"),
        ratio("structured_first_response_compliance", first.get("passed", 0),
              first.get("passed", 0) + first.get("failed", 0), reason="first_complete_generated_response_only"),
        ratio("structured_final_compliance", final.get("direct", 0) + final.get("repaired", 0),
              sum(final.get(key, 0) for key in ("direct", "repaired", "failed")),
              reason="generated_content_outcome_excludes_api_blocked_cache_and_missing"),
        ratio("structured_business_compliance", sum(row["valid"] for row in valid_business),
              len(valid_business), reason="actual_business_validation_attempts_only"),
        ratio("structured_business_observation_coverage", len(requiring_business) - len(business_missing),
              len(requiring_business), reason="observed_business_validation_linked_to_applicable_operation"),
    ]
    return metrics, {"operations": operations, "first_distribution": first, "final_distribution": final,
                     "business_validations": business, "unlinked_observations": unlinked,
                     "operation_count": len(operations), "business_validation_count": len(valid_business),
                     "business_validation_missing_operation_ids": business_missing,
                     "business_missing_association_count": sum(not row.get("operation_id") for row in business)}
