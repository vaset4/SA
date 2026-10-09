"""【研究交付】【保存合同】同次产物与提交身份，作者：xxx，2026-10-04 13:00:00。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DELIVERY_METADATA_FIELDS = (
    "request_id", "trace_summary", "runtime_chain", "source_status", "source_status_reasons",
    "citations", "chunk_citations", "paper_sources", "graph_evidence", "graph_hint_section", "graph_hint_full",
    "evidence_records", "answer_brief", "missing_evidence", "research_plan", "research_unit_results",
    "coverage_review", "planner_trace", "planner_execution_history", "evidence_sufficiency_report", "repair_query_plans", "planner_repair_trace",
    "planner_repair_merge_report", "final_answer_eligibility_report", "planner_final_blocks",
    "planner_final_validation_errors", "final_delivery_report", "final_citation_bindings", "final_generation_attempt",
    "final_delivery_applicability", "error_code", "error_summary", "failure_reason", "turn_depth", "depth_transitions",
    "memory_writeback", "turn_outcome", "memory_read_result", "research_scope", "research_objects", "note_actions",
    "synthesis_state", "dialogue_mode", "answer_confidence", "status", "clarification_question", "report",
    "project_id", "paper_ids", "project_scope", "structured_query", "user_query",
)

class DeliveryArtifact(BaseModel):
    """整体保存服务端产物；正文不做裁剪，支持字段不从历史结果补齐。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    answer: str
    metadata: dict[str, Any]
    response: dict[str, Any]
    session_status: str | None = None
    attempt_summary: dict[str, Any] | None = None


class PendingDelivery(BaseModel):
    """保存产物生产身份及期望版本，供新补存操作重新提交同一份产物。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    target_run_id: str
    producer_run_id: str
    expected_delivery_revision: int = Field(ge=0)
    session_id: str | None = None
    session_run_id: str | None = None
    artifact: DeliveryArtifact


class DeliveryConflict(ValueError):
    """当前操作失去写入权、保存版本变化或仍有待处理产物。"""
