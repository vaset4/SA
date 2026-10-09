"""RAG write-back payload helpers."""

from __future__ import annotations

from typing import Any

# 项目科研回合的回合级字段：本轮深度、深度判断、深度变更轨迹、记忆读取终态、
# 回合终态、候选写回终态。终结节点按这份清单把回合级字段同时送进 state 与 trace，
# 与 trace_builders._TURN_LEVEL_TRACE_KEYS 必须一致（漂了 trace 列表接口就取不到数）。
TURN_LEVEL_STATE_KEYS: tuple[str, ...] = (
    "turn_depth",
    "turn_depth_assessment",
    "depth_transitions",
    "memory_read_result",
    "turn_outcome",
    "memory_writeback",
    "final_delivery_report",
    "final_citation_bindings",
    "final_generation_attempt",
    "final_delivery_applicability",
)


def count_selected_evidence(state: dict[str, Any]) -> int:
    selected = state.get("selected_evidence")
    return len(selected) if isinstance(selected, list) else 0


def generation_status(state: dict[str, Any]) -> str:
    """参数为最终状态；返回同次交付的生成投影，历史路径沿既有生成状态。"""
    from app.services.internal_research.research_planner_final_delivery import (
        delivery_report_from_state,
        project_final_delivery,
    )

    delivery = delivery_report_from_state(state)
    if delivery is not None:
        return project_final_delivery(delivery)["planner_final_status"]
    source_status = state.get("source_status")
    if isinstance(source_status, dict):
        status = str(source_status.get("generation") or "").strip()
        if status:
            return status
    return "ok" if state.get("answer") else "skipped"


def generation_reason(state: dict[str, Any]) -> str:
    reasons = state.get("source_status_reasons")
    if isinstance(reasons, dict):
        return str(reasons.get("generation") or "").strip()
    return ""


def build_write_back_payload(
    state: dict[str, Any],
    *,
    trace: Any,
    trace_summary: dict[str, Any] | None,
    rag_trace_dict: dict[str, Any] | None,
    turn_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """【研究交付】【终结回写】将同次正文、交付与观察投影为图状态增量。

    参数为当前状态、跟踪对象、跟踪摘要/明细及实际回合字段；返回完整终结节点载荷。
    回合字段只投影已产生的值，保留原始正文及其引用位置。
    """
    return {
        "answer": state.get("answer"),
        "citations": state.get("citations"),
        "research_scope": state.get("research_scope", {}),
        "research_objects": state.get("research_objects", []),
        "note_actions": state.get("note_actions", []),
        "paper_sources": state.get("paper_sources", []),
        "chunk_citations": state.get("chunk_citations", []),
        "graph_evidence": state.get("graph_evidence", []),
        "evidence_summary": state.get("evidence_summary", ""),
        "synthesis_state": state.get("synthesis_state", {}),
        "dialogue_mode": state.get("dialogue_mode"),
        "runtime_chain": state.get("runtime_chain"),
        "source_status": state.get("source_status", {}),
        "source_status_reasons": state.get("source_status_reasons", {}),
        "evidence_records": state.get("evidence_records", []),
        "missing_evidence": state.get("missing_evidence", []),
        "answer_brief": state.get("answer_brief", {}),
        "research_plan": state.get("research_plan"),
        "research_unit_results": state.get("research_unit_results", []),
        "coverage_review": state.get("coverage_review"),
        "planner_trace": state.get("planner_trace"),
        "evidence_sufficiency_report": state.get("evidence_sufficiency_report", {}),
        "repair_query_plans": state.get("repair_query_plans", []),
        "planner_repair_trace": state.get("planner_repair_trace", {}),
        "planner_repair_merge_report": state.get("planner_repair_merge_report", {}),
        "final_answer_eligibility_report": state.get("final_answer_eligibility_report", {}),
        "planner_final_blocks": state.get("planner_final_blocks", []),
        "planner_final_validation_errors": state.get("planner_final_validation_errors", []),
        "final_delivery_report": state.get("final_delivery_report"),
        "final_citation_bindings": state.get("final_citation_bindings"),
        "final_generation_attempt": state.get("final_generation_attempt"),
        "final_delivery_applicability": state.get("final_delivery_applicability"),
        "answer_confidence": state.get("answer_confidence"),
        "request_id": getattr(trace, "request_id", None) if trace is not None else None,
        "trace_summary": trace_summary or None,
        "_rag_trace_dict": rag_trace_dict or None,
        # 回合级字段只回写本轮真实有的键：给缺失键补 None 会把 planner 已判好的深度
        # 与记忆读取结果覆盖成空值（本 payload 就是终结节点的 state 增量）
        **(turn_fields or {}),
    }
