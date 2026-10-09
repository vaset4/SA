"""【流程评测】【故障归因】保留证据与并存责任，不按报错文案猜供应商。作者：xxx。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .workflow_facts import WorkflowTurn, event_rows

_TRANSPORT_ERRORS = frozenset({
    "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "ChunkedEncodingError",
    "RemoteProtocolError", "ConnectError", "ReadError", "WriteError",
})
_SYSTEM_ERRORS = frozenset({"KeyError", "TypeError", "AttributeError", "AssertionError", "NameError"})
_REQUEST_ERRORS = frozenset({"InvalidURL", "MissingSchema", "InvalidSchema", "InvalidHeader", "URLRequired"})


def api_failure_observed(row: dict[str, Any]) -> bool:
    """参数为真实尝试；本地URL/鉴权头构造错误归系统，不能因捕获位置在transport就免责。"""
    if row.get("error_type") in _REQUEST_ERRORS or (row.get("error_boundary") == "transport" and row.get("error_type") in _SYSTEM_ERRORS):
        return False
    return (row.get("error_boundary") in {"http_status", "transport", "api_envelope"}
            or isinstance(row.get("http_status"), int) or row.get("error_type") in _TRANSPORT_ERRORS)


def _api_fault(row: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """参数为失败HTTP事实及全部尝试；按实际边界分类并关联同调用恢复结果。"""
    is_api = api_failure_observed(row)
    actor = row.get("actor", "unknown")
    category = "product_api" if actor == "product" and is_api else "unknown"
    if actor in {"judge", "harness"}:
        category = f"{actor}_api" if is_api else actor
    if actor == "product" and not is_api and row.get("error_type") in (_REQUEST_ERRORS | _SYSTEM_ERRORS):
        category = "system"
    attempt = row.get("http_attempt")
    recovered = bool(row.get("call_id")) and isinstance(attempt, int) and any(
        item.get("turn") == row.get("turn") and item.get("call_id") == row["call_id"]
        and item.get("status") == "completed" and item.get("protocol_status") != "failed"
        and isinstance(item.get("http_attempt"), int) and item["http_attempt"] > attempt
        for item in attempts
    )
    return {**row, "category": category, "actor": actor, "recovered": recovered,
            "responsibility": "unknown", "source": "model_response_usage"}


def collect_faults(turns: tuple[WorkflowTurn, ...], *, violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """参数为实际观察与独立不变量违规；返回允许API与系统缺陷同时存在的故障表。"""
    attempts = event_rows(turns, "model_response_usage")
    faults = [_api_fault(row, attempts) for row in attempts if row.get("status") == "failed" or row.get("protocol_status") == "failed"]
    for row in attempts:
        if row.get("request_valid") is False:
            faults.append({**row, "category": "system", "actor": "product", "source": "invalid_request"})
    for row in event_rows(turns, "structured_response_validation"):
        if row.get("schema_valid") is False:
            category = "system" if row.get("cache_hit") is True else "model_output"
            faults.append({**row, "category": category, "actor": "product", "source": "structured_response_validation"})
    for row in event_rows(turns, "structured_business_validation"):
        if row.get("valid") is False:
            faults.append({**row, "category": "model_output", "actor": "product", "source": "business_validation"})
    for turn in turns:
        faults.extend(_turn_faults(turn))
    faults.extend({**row, "category": "system", "actor": "product", "source": "workflow_invariant"} for row in violations)
    return faults


def _turn_faults(turn: WorkflowTurn) -> list[dict[str, Any]]:
    """参数为单轮观察；保留采集故障和异常类型，缺因果证据时标未知。"""
    faults = [{**row, "turn": turn.number, "category": "harness", "actor": "harness",
               "source": "evidence_capture_errors"} for row in turn.capture.get("evidence_capture_errors", [])]
    error = turn.raw.get("error")
    if error:
        kind = error.get("type") if isinstance(error, dict) else None
        category = "system" if kind in _SYSTEM_ERRORS else "unknown"
        faults.append({"turn": turn.number, "category": category, "actor": "product",
                       "source": "turn_error", "error": error, "error_type": kind})
    if turn.values.get("error_code"):
        faults.append({"turn": turn.number, "category": "unknown", "actor": "product",
                       "source": "product_terminal", "error_code": turn.values["error_code"]})
    tools = event_rows((turn,), "tool_execution_result")
    for row in tools:
        if row.get("status") in {"failed", "cancelled"}:
            category = "system" if row.get("error_type") in _SYSTEM_ERRORS else "unknown"
            if api_failure_observed(row):
                category = "product_api"
            recovered = bool(row.get("tool_call_id")) and any(
                later.get("tool_call_id") == row["tool_call_id"] and later.get("status") in {"success", "empty"}
                and later.get("tool_attempt", 0) > row.get("tool_attempt", 0) for later in tools
            )
            faults.append({**row, "category": category, "actor": "product", "source": "tool_execution_result", "recovered": recovered})
    return faults


def efficiency_facts(turns: tuple[WorkflowTurn, ...]) -> dict[str, Any]:
    """参数为冻结轮次；返回真实逻辑调用、HTTP尝试和用量，缺usage不估算。"""
    attempts = event_rows(turns, "model_response_usage")
    calls = [{**row, "turn": turn.number} for turn in turns for row in turn.capture.get("model_call_observations", [])]
    by_actor = dict(Counter(str(row.get("actor", "unknown")) for row in attempts))
    usage = [row for row in attempts if isinstance(row.get("usage"), dict)]
    return {
        "model_calls": calls, "logical_model_call_count": len(calls), "http_attempts": attempts,
        "http_attempt_count": len(attempts), "http_attempts_by_actor": by_actor,
        "http_retry_count": sum(row.get("http_attempt", 0) > 1 for row in attempts if isinstance(row.get("http_attempt"), int)),
        "provider_usage": usage, "provider_usage_missing_count": len(attempts) - len(usage),
        "model_call_capture_missing_turns": [turn.number for turn in turns if "model_call_observations" not in turn.capture],
        "turn_timing": [{"turn": turn.number, "duration_seconds": turn.raw.get("duration_seconds")} for turn in turns],
    }
