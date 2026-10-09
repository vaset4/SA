"""Trace summary and full-trace builders."""

from __future__ import annotations

import json
from typing import Any, Callable

from app.services.internal_research.research_planner_observability import (
    build_planner_stage,
)

# 项目科研回合的轮级字段：本轮生效深度、深度与记忆读取判断、深度变更轨迹、记忆读取终态、回合终态与候选写回终态。
# 这些键只能进 summary：trace 列表接口只投影 trace_data->'summary'，
# 而 trace 超过 OBS_TRACE_MAX_BYTES 时整个 debug 块会被替换掉，放 debug 等于大 trace 上直接消失。
_TURN_LEVEL_TRACE_KEYS = (
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


def _http_attempt_count(call: Any) -> int:
    """参数为调用观察；当前明确记录的零次发送仍是零，仅旧对象缺字段时读取历史计数。"""
    attempts = getattr(call, "http_attempt_count", None)
    if attempts is not None:
        return int(attempts)
    if getattr(call, "cache_hit", False):
        return 0
    return 1 + int(getattr(call, "retry_count", 0) or 0)


def _turn_level_trace_fields(rag_trace_dict: dict[str, Any]) -> dict[str, Any]:
    """挑出本回合真实产出的轮级字段，供 trace summary 投影。

    深度判断准确率、深度升级召回率与记忆读取时机准确率都从这几个键取数，
    所以缺失的键保持缺失：补 0 / 空数组会让「本轮没跑过这个能力」和
    「跑过且结果为空」在评测里分不开。

    参数:
        rag_trace_dict: 序列化后的 RAG trace，已并入 planner 与回合级 state 字段
    返回:
        dict: 只含输入里真实存在的轮级键，不为缺失键编造默认值
    """
    return {key: rag_trace_dict[key] for key in _TURN_LEVEL_TRACE_KEYS if key in rag_trace_dict}


def build_trace_summary_payload(
    trace_ctx: Any,
    rag_trace_dict: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact trace summary returned with assistant messages."""
    retrieval = rag_trace_dict.get("retrieval", {})
    generation = rag_trace_dict.get("generation", {})
    synthesis = rag_trace_dict.get("synthesis", {})
    selected_count = int(retrieval.get("selected", 0) or 0)
    citation_count = int(generation.get("citations_count", 0) or 0)
    summary = {
        "request_id": trace_ctx.request_id,
        "session_id": trace_ctx.session_id,
        "query": rag_trace_dict.get("query", ""),
        "mode": rag_trace_dict.get("mode", ""),
        "graph_mode": rag_trace_dict.get("graph_mode", ""),
        "dialogue_mode": rag_trace_dict.get("dialogue_mode", ""),
        "runtime_chain": rag_trace_dict.get("runtime_chain", ""),
        "source_status": rag_trace_dict.get("source_status", {}),
        "evidence_layer_status": synthesis.get("evidence_layer_status", {}),
        "retrieval_reasoning": rag_trace_dict.get("retrieval_reasoning", ""),
        "latency_ms": rag_trace_dict.get("latency_ms", {}),
        "retrieved_count": retrieval.get("hybrid_results", 0),
        "reranked_count": retrieval.get("reranked", 0),
        "selected_count": selected_count,
        "cited_count": citation_count,
        "evidence_count": selected_count,
        "citation_count": citation_count,
        "evidence_consistent": selected_count == citation_count,
        "generation_status": retrieval.get("generation_status", ""),
        "generation_error_kind": retrieval.get("generation_error_kind", ""),
        "generation_reason": retrieval.get("generation_reason", ""),
        "model_call_count": len(trace_ctx.model_calls),
        "model_http_attempt_count": sum(_http_attempt_count(call) for call in trace_ctx.model_calls),
        "evidence_observation_count": len(trace_ctx.evidence_observations),
        "evidence_capture_error_count": len(trace_ctx.evidence_capture_errors),
    }
    # 回合级字段整体入 summary，contract 对象保持原状：depth 与原因码、能力四态结果、
    # 记忆命中条目、Evidence 与缺口、循环步数与终止原因都在这几个对象里
    summary.update(_turn_level_trace_fields(rag_trace_dict))
    planner_stage = build_planner_stage(rag_trace_dict)
    if planner_stage is not None:
        summary.update(planner_stage["metrics"])
        summary["source_status_reasons"] = planner_stage["source_status_reasons"]
    return summary


def build_full_trace_payload(
    trace_ctx: Any,
    rag_trace_dict: dict[str, Any],
    *,
    call_summary_fn: Callable[[Any], dict[str, Any]],
    call_debug_fn: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """参数为请求 trace、图状态及投影函数；返回保留完整模型响应和校验事件的持久化数据。"""
    observations = [json.loads(item) for item in trace_ctx.evidence_observations]
    return {
        "request_id": trace_ctx.request_id,
        "session_id": trace_ctx.session_id,
        "user_id": trace_ctx.user_id,
        "query": rag_trace_dict.get("query", ""),
        "summary": build_trace_summary_payload(trace_ctx, rag_trace_dict),
        "stages": build_trace_stages_payload(trace_ctx, rag_trace_dict, call_summary_fn),
        "model_calls": [call_summary_fn(call) for call in trace_ctx.model_calls],
        "latency_ms": rag_trace_dict.get("latency_ms", {}),
        # 1. 【研究观察】【响应证据】大 trace 可裁剪调试副本，原始模型输出与失败字段单独留存
        "model_responses": [
            {"call_id": call.call_id, "structured_operation_id": call.structured_operation_id,
             "output_text": call.output_text, "output_chars": call.output_chars, "cache_hit": call.cache_hit}
            for call in trace_ctx.model_calls
        ],
        "model_response_observations": [item for item in observations if item["stage"] == "model_response_content"],
        "structured_output_observations": [
            item for item in observations if item["stage"].startswith("structured_")
        ],
        "debug": {
            "full_model_inputs_outputs": [call_debug_fn(call) for call in trace_ctx.model_calls],
            "evidence_observations": observations,
            "evidence_capture_errors": [json.loads(item) for item in trace_ctx.evidence_capture_errors],
        },
    }


def build_trace_stages_payload(
    trace_ctx: Any,
    rag_trace_dict: dict[str, Any],
    call_summary_fn: Callable[[Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build ordered full-trace stages from serialized trace data."""
    latency = rag_trace_dict.get("latency_ms", {})
    scope = rag_trace_dict.get("scope", {})
    retrieval = rag_trace_dict.get("retrieval", {})
    generation = rag_trace_dict.get("generation", {})
    synthesis = rag_trace_dict.get("synthesis", {})

    stages = [
        _request_stage(trace_ctx, rag_trace_dict, scope, latency),
        _routing_stage(rag_trace_dict, latency),
        _retrieval_stage(rag_trace_dict, retrieval, latency),
        _evidence_stage(retrieval, synthesis, latency),
        _model_calls_stage(trace_ctx, call_summary_fn),
        _generation_stage(retrieval, generation, latency),
    ]
    planner_stage = build_planner_stage(rag_trace_dict)
    if planner_stage is not None:
        stages.insert(4, planner_stage)
    return stages


def _request_stage(trace_ctx: Any, trace: dict[str, Any], scope: dict, latency: dict) -> dict[str, Any]:
    return {
        "name": "request_overview",
        "request_id": trace_ctx.request_id,
        "session_id": trace_ctx.session_id,
        "user_id": trace_ctx.user_id,
        "query": trace.get("query", ""),
        "mode": trace.get("mode", ""),
        "dialogue_mode": trace.get("dialogue_mode", ""),
        "runtime_chain": trace.get("runtime_chain", ""),
        "scope": scope,
        "latency_ms": latency.get("total", 0),
    }


def _routing_stage(trace: dict[str, Any], latency: dict) -> dict[str, Any]:
    return {
        "name": "routing",
        "graph_mode": trace.get("graph_mode", ""),
        "runtime_chain": trace.get("runtime_chain", ""),
        "retrieval_reasoning": trace.get("retrieval_reasoning", ""),
        "source_status": trace.get("source_status", {}),
        "latency_ms": latency.get("retrieval", 0),
    }


def _retrieval_stage(trace: dict[str, Any], retrieval: dict, latency: dict) -> dict[str, Any]:
    return {
        "name": "retrieval",
        "rewritten_queries": trace.get("rewritten_queries", []),
        "vector_results": {"count": retrieval.get("hybrid_results", 0)},
        "graph_calls": retrieval.get("graph_calls", 0),
        "reranked": retrieval.get("reranked", 0),
        "latency_ms": latency.get("rerank", 0),
    }


def _evidence_stage(retrieval: dict, synthesis: dict, latency: dict) -> dict[str, Any]:
    return {
        "name": "evidence",
        "graded_relevant": retrieval.get("graded_relevant", 0),
        "graded_irrelevant": retrieval.get("graded_irrelevant", 0),
        "selected_count": retrieval.get("selected", 0),
        "selection_stats": retrieval.get("selection_stats", {}),
        "synthesis_status": synthesis.get("status", ""),
        "evidence_layers": synthesis.get("evidence_layers", {}),
        "evidence_layer_status": synthesis.get("evidence_layer_status", {}),
        "latency_ms": latency.get("grading", 0),
    }


def _model_calls_stage(trace_ctx: Any, call_summary_fn: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "model_calls",
        "count": len(trace_ctx.model_calls),
        "calls": [call_summary_fn(call) for call in trace_ctx.model_calls],
    }


def _generation_stage(retrieval: dict, generation: dict, latency: dict) -> dict[str, Any]:
    return {
        "name": "generation",
        "self_rag_iterations": generation.get("self_rag_iterations", 0),
        "selected_count": retrieval.get("selected", 0),
        "citation_count": generation.get("citations_count", 0),
        "generation_status": retrieval.get("generation_status", ""),
        "error_kind": retrieval.get("generation_error_kind", ""),
        "reason": retrieval.get("generation_reason", ""),
        "latency_ms": latency.get("generation", 0),
    }
