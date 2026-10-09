"""Planner-native observability projections."""

from __future__ import annotations

from typing import Any, Mapping

from app.services.internal_research.research_planner_loop_state import (
    PROBE_RESULTS_STATE_KEY,
)
from app.services.internal_research.research_planner_final_delivery import project_final_delivery

_SUCCESS_STATUSES = frozenset({"evidence_found", "inferred"})
_BLOCKED_STATUSES = frozenset({"blocked"})
_INSUFFICIENT_STATUSES = frozenset({"insufficient_evidence", "external_needed"})
_FINAL_ERROR_KEYS = ("code", "block_index", "block_id", "block_role", "field", "claim_ids", "evidence_ids", "message")


def build_planner_trace_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return planner payload fields safe for trace summary/stage derivation."""
    if not _has_planner_payload(state):
        return {}
    plan = _dict_or_empty(state.get("research_plan"))
    results = _list_of_dicts(state.get("research_unit_results"))
    records = _list_of_dicts(state.get("evidence_records"))
    source_status = _dict_or_empty(state.get("source_status"))
    blocks = _list_of_dicts(state.get("planner_final_blocks"))
    errors = compact_planner_final_errors(state.get("planner_final_validation_errors"))
    repair_trace = _dict_or_empty(state.get("planner_repair_trace"))
    repair_plans = _list_of_dicts(state.get("repair_query_plans"))
    return {
        "research_plan": plan,
        "research_unit_results": results,
        "evidence_records": records,
        "run_id": str(state.get("run_id") or ""),
        "planner_read_receipts": _list_of_dicts(state.get("planner_read_receipts")),
        "planner_execution_history": _list_of_dicts(state.get("planner_execution_history")),
        "source_status_reasons": _dict_or_empty(state.get("source_status_reasons")),
        "planner_trace": _dict_or_empty(state.get("planner_trace")),
        "evidence_sufficiency_report": _dict_or_empty(state.get("evidence_sufficiency_report")),
        "repair_query_plans": repair_plans,
        "planner_repair_trace": repair_trace,
        "planner_repair_merge_report": _dict_or_empty(state.get("planner_repair_merge_report")),
        "final_answer_eligibility_report": _dict_or_empty(state.get("final_answer_eligibility_report")),
        "planner_final_blocks": blocks,
        "planner_final_validation_errors": errors,
        "final_delivery_report": state.get("final_delivery_report"),
        "final_citation_bindings": state.get("final_citation_bindings"),
        "final_generation_attempt": state.get("final_generation_attempt"),
        "final_delivery_applicability": state.get("final_delivery_applicability"),
        # 出计划前的探测：申请（用途、筛选条件、模型理由）与结果四态逐条进 trace，
        # 排查时能还原「模型问了什么、为什么问、拿到了什么」；回边留痕带出探测那几轮
        "planner_probe_results": _list_of_dicts(state.get(PROBE_RESULTS_STATE_KEY)),
        "planner_recovery_trace": _list_of_dicts(state.get("planner_recovery_trace")),
        "planner_metrics": derive_planner_metrics(
            research_plan=plan,
            research_unit_results=results,
            evidence_records=records,
            source_status=source_status,
            planner_final_blocks=blocks,
            planner_final_validation_errors=errors,
            final_delivery_report=state.get("final_delivery_report"),
            planner_repair_trace=repair_trace,
            repair_query_plans=repair_plans,
        ),
    }


def merge_planner_trace_payload(
    rag_trace_dict: dict[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Add planner observability payload to a serialized RAG trace dict."""
    payload = build_planner_trace_payload(state)
    if not payload:
        return rag_trace_dict
    return {**rag_trace_dict, **payload}


def derive_planner_metrics(
    *,
    research_plan: Any,
    research_unit_results: Any,
    evidence_records: Any,
    source_status: Any,
    planner_final_blocks: Any = None,
    planner_final_validation_errors: Any = None,
    final_delivery_report: Any = None,
    planner_repair_trace: Any = None,
    repair_query_plans: Any = None,
) -> dict[str, Any]:
    """参数为研究及同次交付事实；返回真实产物数量，块数不以错误条数替代。"""
    units = _list_of_dicts(_dict_or_empty(research_plan).get("units"))
    results = _list_of_dicts(research_unit_results)
    errors = compact_planner_final_errors(planner_final_validation_errors)
    final_blocks = _list_of_dicts(planner_final_blocks)
    status = _dict_or_empty(source_status)
    repair_trace = _dict_or_empty(planner_repair_trace)
    delivery = _dict_or_empty(final_delivery_report)
    final_status = project_final_delivery(delivery)["planner_final_status"] if delivery else str(status.get("planner_final") or "")
    accepted = len(delivery.get("valid_business_block_ids", [])) if delivery else len(final_blocks)
    return {
        "planner_units_total": len(units),
        "planner_units_success": _count_statuses(results, _SUCCESS_STATUSES),
        "planner_units_blocked": _count_statuses(results, _BLOCKED_STATUSES),
        "planner_units_insufficient": _count_statuses(results, _INSUFFICIENT_STATUSES),
        "planner_evidence_count": len(_list_of_dicts(evidence_records)),
        "planner_claim_count": _claim_count(results),
        "planner_status": str(status.get("planner") or ""),
        "planner_repair_status": str(status.get("planner_repair") or ""),
        "planner_repair_query_count": len(_repair_queries(repair_query_plans)),
        "planner_repair_remaining_gap_count": _remaining_gap_count(repair_trace),
        "planner_final_status": final_status,
        "planner_final_accepted_block_count": accepted,
        "planner_final_rejected_block_count": count_rejected_final_blocks(errors) if delivery else len(errors),
        "planner_final_block_roles": _block_roles(final_blocks),
        "planner_final_error_codes": _error_codes(errors),
    }


def count_rejected_final_blocks(errors: list[dict[str, Any]]) -> int:
    """参数为新版最终校验错误；按原列表位置统计坏块，报告级错误不算正文块。"""
    return len({error["block_index"] for error in errors if type(error.get("block_index")) is int})


def planner_metrics_from_trace(rag_trace_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Read or derive planner metrics from a serialized trace dict."""
    existing = rag_trace_dict.get("planner_metrics")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    source_status = _dict_or_empty(rag_trace_dict.get("source_status"))
    return derive_planner_metrics(
        research_plan=rag_trace_dict.get("research_plan"),
        research_unit_results=rag_trace_dict.get("research_unit_results"),
        evidence_records=rag_trace_dict.get("evidence_records"),
        source_status=source_status,
        planner_final_blocks=rag_trace_dict.get("planner_final_blocks"),
        planner_final_validation_errors=rag_trace_dict.get("planner_final_validation_errors"),
        final_delivery_report=rag_trace_dict.get("final_delivery_report"),
        planner_repair_trace=rag_trace_dict.get("planner_repair_trace"),
        repair_query_plans=rag_trace_dict.get("repair_query_plans"),
    )


def compact_planner_final_errors(raw: Any) -> list[dict[str, Any]]:
    """Keep only message/API-safe final-answer validation error fields."""
    errors = _list_of_dicts(raw)
    return [
        {key: error[key] for key in _FINAL_ERROR_KEYS if key in error}
        for error in errors
    ]


def compact_planner_final_blocks(raw: Any) -> list[dict[str, Any]]:
    """Return accepted block diagnostics without final-answer text."""
    blocks = _list_of_dicts(raw)
    return [
        {
            "role": str(block.get("role") or ""),
            "block_id": str(block.get("block_id") or ""),
            "content_kind": str(block.get("content_kind") or ""),
            "claim_ids": _string_list(block.get("claim_ids")),
            "evidence_ids": _string_list(block.get("evidence_ids")),
        }
        for block in blocks if block.get("valid", True)
    ]


def build_planner_stage(rag_trace_dict: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a full-trace planner stage with trace-safe diagnostics."""
    metrics = planner_metrics_from_trace(rag_trace_dict)
    if not _has_trace_planner_payload(rag_trace_dict, metrics):
        return None
    return {
        "name": "planner",
        "source_status": _dict_or_empty(rag_trace_dict.get("source_status")),
        "source_status_reasons": _dict_or_empty(rag_trace_dict.get("source_status_reasons")),
        "metrics": metrics,
        "planner_final": {
            "status": metrics.get("planner_final_status", ""),
            "delivery_report": rag_trace_dict.get("final_delivery_report"),
            "citation_bindings": rag_trace_dict.get("final_citation_bindings"),
            "generation_attempt": rag_trace_dict.get("final_generation_attempt"),
            "applicability": rag_trace_dict.get("final_delivery_applicability"),
            "accepted_blocks": compact_planner_final_blocks(rag_trace_dict.get("planner_final_blocks")),
            "validation_errors": compact_planner_final_errors(
                rag_trace_dict.get("planner_final_validation_errors")
            ),
        },
        "evidence_repair": {
            "status": metrics.get("planner_repair_status", ""),
            "query_count": metrics.get("planner_repair_query_count", 0),
            "remaining_gap_count": metrics.get("planner_repair_remaining_gap_count", 0),
            "trace": _dict_or_empty(rag_trace_dict.get("planner_repair_trace")),
        },
        # 出计划前的探测轮（AC 13）：申请（用途/筛选条件/模型理由）与结果四态逐条可见。
        # 必须挂进本函数才会进落库投影——落库走 build_full_trace 的显式白名单，
        # 只进 build_planner_trace_payload 的键到不了任何被持久化或被接口返回的 trace
        "context_probe": _list_of_dicts(rag_trace_dict.get("planner_probe_results")),
        "execution": {"run_id": str(rag_trace_dict.get("run_id") or ""),
                      "read_receipts": _list_of_dicts(rag_trace_dict.get("planner_read_receipts")),
                      "history": _list_of_dicts(rag_trace_dict.get("planner_execution_history"))},
        # 1. 【研究失败】【读取事实】保存与失败正文相同的派生事实，不在诊断层重新推断
        "read_facts": _dict_or_empty(_dict_or_empty(rag_trace_dict.get("planner_trace")).get("read_facts")),
        # 全循环留痕：每步 stage/action/reason，能看出回边这几轮里哪几轮是探测
        "loop_trace": _list_of_dicts(rag_trace_dict.get("planner_recovery_trace")),
    }


def planner_result_count(state: Mapping[str, Any], fallback: int = 0) -> int:
    """Prefer planner evidence count for completion logs in planner mode."""
    payload = build_planner_trace_payload(state)
    metrics = payload.get("planner_metrics") if payload else {}
    if isinstance(metrics, dict):
        count = int(metrics.get("planner_evidence_count") or 0)
        if count:
            return count
    return fallback


def _has_planner_payload(state: Mapping[str, Any]) -> bool:
    return str(state.get("runtime_chain") or "") == "planner" or bool(state.get("research_plan"))


def _has_trace_planner_payload(trace: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool:
    if str(trace.get("runtime_chain") or "") == "planner":
        return True
    return any(int(metrics.get(key) or 0) for key in ("planner_units_total", "planner_evidence_count"))


def _dict_or_empty(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, dict) else {}


def _list_of_dicts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _count_statuses(results: list[dict[str, Any]], statuses: frozenset[str]) -> int:
    return sum(1 for result in results if str(result.get("status") or "") in statuses)


def _claim_count(results: list[dict[str, Any]]) -> int:
    return sum(len(_list_of_dicts(result.get("claims"))) for result in results)


def _block_roles(blocks: list[dict[str, Any]]) -> list[str]:
    roles = [str(block.get("role") or "").strip() for block in blocks]
    return sorted({role for role in roles if role})


def _error_codes(errors: list[dict[str, Any]]) -> list[str]:
    codes = [str(error.get("code") or "").strip() for error in errors]
    return sorted({code for code in codes if code})


def _repair_queries(raw: Any) -> list[dict[str, Any]]:
    return [
        query
        for plan in _list_of_dicts(raw)
        for query in _list_of_dicts(plan.get("queries"))
    ]


def _remaining_gap_count(repair_trace: Mapping[str, Any]) -> int:
    final_pass = repair_trace.get("final_pass")
    if not isinstance(final_pass, dict):
        return 0
    return int(final_pass.get("hard_gap_count") or 0)


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item or "").strip()]
