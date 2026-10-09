"""RAG trace 初始化、更新与收尾服务。"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.services.internal_research.research_planner_final_delivery import (
    delivery_report_from_state,
    delivery_state_fields,
    project_final_delivery,
)

from app.services.internal_research.research_planner_observability import (
    merge_planner_trace_payload,
    planner_result_count,
)
from app.services.rag_trace_layers import build_evidence_layer_status
from app.services.rag_trace_payload import (
    TURN_LEVEL_STATE_KEYS,
    build_write_back_payload,
    count_selected_evidence,
    generation_reason,
    generation_status,
)

logger = logging.getLogger(__name__)


def _normalize_dialogue_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"chat", "lookup", "research"} else ""


def _apply_trace_scope(trace: Any, state: dict) -> None:
    if trace is None:
        return
    structured_query = state.get("structured_query") if isinstance(state.get("structured_query"), dict) else {}
    synthesis_state = state.get("synthesis_state") if isinstance(state.get("synthesis_state"), dict) else {}
    resolved_scope = structured_query.get("resolved_scope") if isinstance(structured_query.get("resolved_scope"), dict) else {}
    project_scope = synthesis_state.get("project_scope") if isinstance(synthesis_state.get("project_scope"), dict) else {}
    paper_focus = synthesis_state.get("paper_focus") if isinstance(synthesis_state.get("paper_focus"), dict) else {}
    paper_ids = []
    for group in (
        structured_query.get("paper_ids"),
        resolved_scope.get("paper_ids"),
        project_scope.get("paper_ids"),
    ):
        if not isinstance(group, list):
            continue
        for item in group:
            paper_id = str(item or "").strip()
            if paper_id and paper_id not in paper_ids:
                paper_ids.append(paper_id)
    trace.dialogue_mode = (
        _normalize_dialogue_mode(state.get("dialogue_mode"))
        or _normalize_dialogue_mode(structured_query.get("dialogue_mode"))
        or _normalize_dialogue_mode(synthesis_state.get("dialogue_mode"))
    )
    trace.project_id = str(
        structured_query.get("project_id")
        or resolved_scope.get("project_id")
        or project_scope.get("project_id")
        or trace.project_id
        or ""
    ).strip()
    trace.paper_ids = paper_ids or list(getattr(trace, "paper_ids", []) or [])
    trace.paper_focus_mode = str(paper_focus.get("mode") or trace.paper_focus_mode or "").strip()


def _apply_trace_synthesis(trace: Any, synthesis_state: dict[str, Any] | None) -> None:
    if trace is None or not isinstance(synthesis_state, dict):
        return
    trace.dialogue_mode = _normalize_dialogue_mode(synthesis_state.get("dialogue_mode")) or trace.dialogue_mode
    trace.synthesis_status = str(synthesis_state.get("status") or trace.synthesis_status or "").strip()
    if isinstance(synthesis_state.get("paper_focus"), dict):
        trace.paper_focus_mode = str(synthesis_state["paper_focus"].get("mode") or trace.paper_focus_mode or "").strip()
    if isinstance(synthesis_state.get("project_scope"), dict):
        trace.project_id = str(synthesis_state["project_scope"].get("project_id") or trace.project_id or "").strip()
    if isinstance(synthesis_state.get("evidence_layers"), dict):
        trace.evidence_layers = dict(synthesis_state["evidence_layers"])


def _apply_trace_research_metrics(trace: Any, state: dict) -> None:
    """research 流的 agentic 指标回写到 trace，让 routing/retrieval/evidence 阶段不为空。"""
    if trace is None:
        return
    source_status = state.get("source_status")
    if isinstance(source_status, dict) and source_status:
        trace.source_status = dict(source_status)
    runtime_chain = str(state.get("runtime_chain") or "").strip()
    if runtime_chain:
        trace.runtime_chain = runtime_chain
    trace.evidence_layer_status = build_evidence_layer_status(state)
    vector_calls = int(state.get("vector_call_count", 0) or 0)
    graph_calls = int(state.get("graph_call_count", 0) or 0)
    if vector_calls or graph_calls:
        if not isinstance(trace.retrieval, dict):
            trace.retrieval = {}
        trace.retrieval["hybrid_results"] = max(trace.retrieval.get("hybrid_results", 0), vector_calls)
        trace.retrieval["graph_calls"] = graph_calls
    evidence_brief = state.get("evidence_brief") if isinstance(state.get("evidence_brief"), dict) else {}
    available_evidence = int(evidence_brief.get("available_count", 0) or 0)
    if available_evidence:
        if not isinstance(trace.retrieval, dict):
            trace.retrieval = {}
        trace.retrieval["graded_relevant"] = max(trace.retrieval.get("graded_relevant", 0), available_evidence)
    # 05-24: graph_prefetch / sufficiency_grader 执行标记
    if graph_calls > 0:
        trace.graph_prefetched = True
    if state.get("sufficiency_result"):
        trace.sufficiency_graded = True


def trace_enabled(settings: Any) -> bool:
    """判断当前是否开启 trace。"""
    return bool(getattr(settings, "tracing_enabled", True))


def ensure_trace(state: dict, *, settings: Any, observability_module: Any):
    """
    按需初始化请求级 RAG trace。

    trace 挂在 state 上随节点流转，因此它更像一次请求的运行时观察对象，而不是全局单例。
    """
    if not trace_enabled(settings):
        return None
    trace = state.get("trace")
    if trace is not None:
        return trace
    from app.middleware.trace_context import get_trace_ctx
    ctx = get_trace_ctx()
    init_kwargs: dict[str, Any] = {
        "session_id": state.get("session_id", "") or "",
        "query": state.get("user_query", "") or "",
    }
    if ctx is not None and ctx.request_id:
        init_kwargs["request_id"] = ctx.request_id
    trace = observability_module.RAGTrace(**init_kwargs)
    _apply_trace_scope(trace, state)
    return trace


def with_trace(payload: dict, trace: Any) -> dict:
    """把 trace 以最小侵入方式并回节点返回值。"""
    if trace is None:
        return payload
    return {**payload, "trace": trace}


def update_trace_generation(
    trace: Any,
    *,
    citations: list[dict] | None = None,
    selected_count: int | None = None,
    evidence_selection_stats: dict[str, Any] | None = None,
    generation_status: str | None = None,
    error_kind: str | None = None,
    reason: str | None = None,
    quality_score: float | None = None,
    iteration: int | None = None,
) -> None:
    """统一更新生成与收敛阶段的 trace 指标。"""
    if trace is None:
        return
    if citations is not None:
        trace.citations_count = len(citations)
        trace.retrieval["cited"] = len(citations)
    if selected_count is not None:
        trace.retrieval["selected"] = max(0, int(selected_count))
    if evidence_selection_stats:
        trace.retrieval["selection_stats"] = dict(evidence_selection_stats)
    if generation_status:
        trace.retrieval["generation_status"] = str(generation_status)
    if error_kind:
        trace.retrieval["generation_error_kind"] = str(error_kind)
    if reason:
        trace.retrieval["generation_reason"] = str(reason)
    if quality_score is not None:
        trace.final_quality_score = quality_score
    if iteration is not None:
        trace.self_rag_iterations = iteration


async def execute_write_back(
    state: dict,
    *,
    paper_db: Any,
    settings: Any,
    observability_module: Any,
    with_trace_fn: Any,
    update_trace_generation_fn: Any,
) -> dict:
    """
    执行写回后处理与 trace 持久化。

    这里做三类收尾动作：
      1. 基于最终命中的论文尝试推送推荐通知
      2. 评估并写回项目研究记忆候选，组装回合终态
      3. 把整次 RAG trace 与计时事件落到观测层
    """
    started_at = time.monotonic()
    await _push_recommendations(state, paper_db=paper_db, settings=settings)
    # 记忆写回挂在这一步：答案已定稿、run 终态还没落库，同步与流式两条路径都必过此处
    turn_fields = await build_turn_level_fields(state)
    state = {**state, **turn_fields}
    trace = state.get("trace")
    rag_trace_dict, trace_summary = await _finalize_trace(
        state,
        turn_fields=turn_fields,
        started_at=started_at,
        observability_module=observability_module,
        update_trace_generation_fn=update_trace_generation_fn,
    )
    return with_trace_fn(
        build_write_back_payload(
            state,
            trace=trace,
            trace_summary=trace_summary,
            rag_trace_dict=rag_trace_dict,
            turn_fields=turn_fields,
        ),
        trace,
    )


async def _finalize_trace(
    state: dict,
    *,
    turn_fields: dict[str, Any],
    started_at: float,
    observability_module: Any,
    update_trace_generation_fn: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    收尾 trace：回写终局指标、并入回合级字段、落库并记一次完成事件。

    参数:
        state: 终结节点看到的图状态
        turn_fields: 本回合的回合级字段（深度/记忆读取/回合终态/写回终态）
        started_at: 写回开始时刻，用于记录本步耗时
        observability_module: 观测层模块
        update_trace_generation_fn: 生成阶段指标更新函数
    返回:
        (序列化后的 rag trace 字典, trace summary)；trace 未开启时两者都是空字典
    """
    trace = state.get("trace")
    if trace is None:
        return {}, {}
    _apply_final_trace_metrics(trace, state, update_trace_generation_fn=update_trace_generation_fn)
    # 回合级字段不在 RAGTrace 上，planner 合并也不带，这里显式并进去才能进 trace summary
    rag_trace_dict = {**merge_planner_trace_payload(trace.to_dict(), state), **turn_fields}
    trace_summary = await _persist_trace(trace, rag_trace_dict, observability_module=observability_module)
    observability_module.record_timed_event(
        "rag_answer_completed",
        "rag_write_back",
        started_at,
        query=trace.query,
        result_count=planner_result_count(state, fallback=len(state.get("citations", []))),
        session_id=trace.session_id or None,
    )
    return rag_trace_dict, trace_summary


async def build_turn_level_fields(state: dict) -> dict[str, Any]:
    """
    评估记忆写回、组装回合终态，返回本回合真实存在的回合级字段。

    交付合同按实际研究入口检查；只有带 turn_depth 的项目科研回合才评估记忆和回合终态，
    避免把没有项目科研深度的问答误记为已完成项目科研回合。

    参数:
        state: 终结节点看到的图状态
    返回:
        回合级字段字典，含本轮已有的深度/深度判断/记忆读取结果，加新算出的
        memory_writeback 与 turn_outcome；无深度时仅返回实际适用的交付字段
    """
    from app.services.internal_research.research_memory_writeback import (
        build_turn_outcome,
    )

    # 1. 【研究交付】【记忆前置】应有报告的研究出口缺失报告时，先记录合同失败再评估写回
    delivery = delivery_report_from_state(state)
    delivery_delta = {}
    if delivery is not None:
        projection = project_final_delivery(delivery)
        delivery_delta = {**delivery_state_fields(delivery),
                          "source_status": {**state.get("source_status", {}),
                                            "planner_final": projection["planner_final_status"]}}
        if any(reason.code in {"missing_delivery_report", "invalid_delivery_report"} for reason in delivery.reasons):
            delivery_delta.update({"citations": [], "chunk_citations": [], "planner_final_blocks": [],
                                   "source_status_reasons": {**state.get("source_status_reasons", {}),
                                                             "planner_final": delivery.reasons[0].message}})
        state = {**state, **delivery_delta}
    # 2. 【研究交付】【入口适用】交付适用性独立于深度字段，缺失深度不能跳过研究合同检查
    if not str(state.get("turn_depth") or "").strip():
        return delivery_delta
    # 3. 【研究记忆】【提交前留痕】图内只记录尚未写回，候选必须等同次答案提交成功后生成
    memory_writeback = {"status": "skipped", "candidate_ids": [], "reason_codes": ["awaiting_result_commit"],
                        "retryable": False, "error_kind": ""}
    # 4. 本轮真实存在的回合级字段原样带出，缺的键保持缺失不补默认值
    fields = {key: state[key] for key in TURN_LEVEL_STATE_KEYS if key in state}
    fields["memory_writeback"] = memory_writeback
    fields["turn_outcome"] = build_turn_outcome(state, memory_writeback=memory_writeback)
    return {**fields, **delivery_delta}


async def _push_recommendations(state: dict, *, paper_db: Any, settings: Any) -> None:
    """
    按本轮命中的论文尝试推送推荐通知。

    参数:
        state: 当前图状态，命中论文取自 rag_result.sources
        paper_db: 论文库访问对象
        settings: 全局配置，取推送分数阈值
    """
    from app.services.recommendation import build_profile, push_notification, score_paper

    rag_result = state.get("rag_result")
    if not rag_result or not rag_result.get("sources"):
        return
    # 这属于“回答完成后的副作用”，失败时不应该影响主回答返回。
    try:
        profile = build_profile()
        seen = set()
        for paper_id in rag_result["sources"]:
            if paper_id in seen:
                continue
            seen.add(paper_id)
            paper = await asyncio.to_thread(paper_db.get_paper, paper_id)
            if paper:
                score, reason = score_paper(paper, profile)
                if score >= settings.push_score_threshold:
                    push_notification(paper, score, reason)
    except Exception as exc:
        logger.error(
            "recommendation push side channel failed session_id=%s: %s",
            state.get("session_id") or getattr(state.get("trace"), "session_id", ""),
            exc,
            exc_info=True,
        )


def _apply_final_trace_metrics(trace: Any, state: dict, *, update_trace_generation_fn: Any) -> None:
    """
    收尾阶段把 scope/synthesis/research 指标与生成指标全部回写到 trace。

    参数:
        trace: 本次请求的 RAGTrace
        state: 当前图状态
        update_trace_generation_fn: 生成阶段指标更新函数
    """
    _apply_trace_scope(trace, state)
    _apply_trace_synthesis(
        trace,
        state.get("synthesis_state") if isinstance(state.get("synthesis_state"), dict) else None,
    )
    _apply_trace_research_metrics(trace, state)
    trace.graph_mode = str(state.get("graph_mode") or trace.graph_mode or "").strip()
    trace.retrieval_reasoning = str(state.get("retrieval_reasoning") or trace.retrieval_reasoning or "").strip()
    trace.checkpoint("evaluation_end")
    update_trace_generation_fn(
        trace,
        citations=state.get("citations", []),
        selected_count=count_selected_evidence(state),
        evidence_selection_stats=state.get("evidence_selection_stats")
        if isinstance(state.get("evidence_selection_stats"), dict)
        else None,
        generation_status=generation_status(state),
        error_kind=state.get("error_kind"),
        reason=generation_reason(state),
        quality_score=(state.get("evaluation") or {}).get("quality_score"),
        iteration=state.get("self_rag_iteration"),
    )


async def _persist_trace(
    trace: Any,
    rag_trace_dict: dict[str, Any],
    *,
    observability_module: Any,
) -> dict[str, Any]:
    """
    落一次完整 trace，并返回随消息返回的精简 summary。

    参数:
        trace: 本次请求的 RAGTrace
        rag_trace_dict: 已并入 planner 与回合级字段的序列化 trace
        observability_module: 观测层模块，无请求上下文时退回旧落库路径
    返回:
        trace summary；无请求上下文时为空字典
    """
    from app.middleware.trace_context import (
        build_full_trace,
        build_trace_summary,
        get_trace_ctx,
        safe_write_trace,
    )

    ctx = get_trace_ctx()
    if ctx is None:
        await observability_module.persist_rag_trace(trace)
        return {}
    await safe_write_trace(
        trace_data=build_full_trace(ctx, rag_trace_dict),
        request_id=ctx.request_id,
        session_id=ctx.session_id,
        query=rag_trace_dict.get("query", ""),
        user_id=ctx.user_id,
    )
    return build_trace_summary(ctx, rag_trace_dict)
