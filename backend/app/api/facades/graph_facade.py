"""
Graph 调用与流式桥接 facade。

这是 API 层和 LangGraph 主图之间最关键的“桥接层”。
route 不应该直接知道：
- graph 输入 state 该怎么组装
- session_id / conversation_id 如何映射到 thread_id
- 是普通同步调用还是流式调用
- 中断恢复（interrupt/resume）应该怎么处理

这些细节统一收口在这里，避免每个 API 路由都重复写一遍图调用样板代码。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import HTTPException
from langgraph.types import Command
from app.models.result_delivery_contracts import DELIVERY_METADATA_FIELDS

from app.api import agent_singleton as agent_singleton_module
from app.config import get_settings
import app.middleware.observability as observability
from app.middleware.trace_context import (
    TraceContext,
    get_trace_ctx,
    reset_trace_ctx,
    set_trace_ctx,
)
from app.services.checkpoint_service import build_checkpoint_config
from app.services.rag_run_service import execution_timeout_seconds
from app.services.session_mapping_service import (
    build_session_identity,
    ensure_session_id,
    resolve_thread_id,
)


async def get_shared_agent():
    """返回共享 graph 单例；封装一层便于测试替换。"""
    return await agent_singleton_module.get_shared_agent()


async def get_shared_agent_state(
    session_id: str,
    conversation_id: str | None = None,
    *,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """读取共享 graph 状态快照。"""
    return await agent_singleton_module.get_shared_agent_state(
        session_id,
        conversation_id,
        thread_id=thread_id,
    )


def _resolve_session_id(session_id: str | None, conversation_id: str | None) -> str:
    """
    兼容旧测试替身和新签名的 session_id 解析入口。

    生产代码走 `ensure_session_id(session_id, conversation_id)`；
    但部分 direct-call 测试仍注入旧的一参 stub，这里保持兼容。
    """
    try:
        parameters = inspect.signature(ensure_session_id).parameters
    except (TypeError, ValueError):
        parameters = {}
    if len(parameters) >= 2:
        return ensure_session_id(session_id, conversation_id)
    return ensure_session_id(session_id)


def _build_agent_state(
    user_query: str,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str | None]:
    """
    构造一次 graph 调用的初始状态。

    这里负责把 API 侧零散参数收拢成主图能理解的 `AgentState` 初值。
    一个重要约束是：即使某些字段本轮没用，也尽量补齐默认结构，
    这样后续节点和快照序列化就不必到处判空。
    """
    payload = dict(extra or {})
    conversation_id = str(payload.get("conversation_id") or "").strip() or None
    session_id = _resolve_session_id(payload.get("session_id"), conversation_id)
    thread_id = resolve_thread_id(
        session_id,
        conversation_id,
        thread_id=payload.get("thread_id"),
    )
    state = {
        "user_query": user_query,
        # 【研究执行】【运行边界】新 graph 调用独立运行；durable 入口随后透传其既有 run_id
        "run_id": str(payload.get("run_id") or uuid.uuid4()),
        "research_plan": None,
        "research_unit_results": [],
        "final_delivery_report": None, "final_citation_bindings": [],
        "final_generation_attempt": None, "final_delivery_applicability": None,
        "planner_final_blocks": [], "planner_final_validation_errors": [],
        "planner_final_candidate": {},
        "turn_outcome": None, "memory_writeback": None,
        "evidence_records": [],
        "planner_read_receipts": [],
        "planner_execution_history": [],
        "planner_locator_facts": [],
        "planner_budget": {},
        "planner_probe_results": [],
        "planner_draft": None, "planner_observation": {}, "planner_loop_decision": "",
        "planner_recovery_trace": [], "coverage_review": None, "evidence_sufficiency_report": None,
        "turn_depth": None, "turn_depth_assessment": None, "memory_read_result": None,
        "_paper_list_lines": [], "_section_structure_lines": [], "_graph_status_lines": [],
        "intent": None,
        "core_intent": None,
        "structured_query": None,
        "dialogue_mode": None,
        "project_scope": None,
        "paper_focus": None,
        "synthesis_state": None,
        "report": None,
        # 【研究交付】【新问题初始化】清空同线程旧正文；显式恢复仍使用原 checkpoint
        "answer": None,
        "research_mode": None,
        "research_brief": None,
        "research_notes": [],
        "raw_notes": [],
        "selected_papers": [],
        "comparison_matrix": [],
        "evidence_cards": [],
        "source_records": [],
        "reasoning_summaries": [],
        "key_findings": [],
        "uncertainties": [],
        "idea_candidates": [],
        "external_gaps": [],
        "follow_up_questions": [],
        "candidate_papers_to_import": [],
        "papers": [],
        "filtered_papers": [],
        "rag_result": None,
        "chat_history": [],
        "session_id": session_id,
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "context_summary": None,
    }
    state.update(payload)
    state["session_id"] = session_id
    state["conversation_id"] = conversation_id
    state["thread_id"] = thread_id
    return state, session_id, conversation_id


def _agent_config(
    session_id: str,
    conversation_id: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    把项目侧 session_id 映射到 LangGraph configurable.thread_id。

    这层转换很关键，因为业务侧暴露的是 session/conversation 语义，
    但 LangGraph 只认识 checkpoint config 里的 thread_id。
    """
    try:
        parameters = inspect.signature(build_checkpoint_config).parameters
    except (TypeError, ValueError):
        parameters = {}
    if len(parameters) >= 3:
        return build_checkpoint_config(
            session_id,
            conversation_id,
            thread_id=thread_id,
        )
    if len(parameters) >= 2:
        return build_checkpoint_config(session_id, conversation_id)
    return build_checkpoint_config(session_id)


def snapshot_has_pending_interrupt(snapshot: Any) -> bool:
    """判断当前 thread 是否停在等待用户澄清/恢复的中断点上。"""
    if snapshot is None:
        return False
    if isinstance(snapshot, dict):
        interrupts = snapshot.get("interrupts")
        return isinstance(interrupts, list) and bool(interrupts)
    interrupts = getattr(snapshot, "interrupts", ())
    return bool(interrupts)


def _interrupt_values(payload: dict[str, Any]) -> list[Any]:
    raw_interrupts = payload.get("__interrupt__")
    if not isinstance(raw_interrupts, list):
        return []
    values: list[Any] = []
    for item in raw_interrupts:
        if isinstance(item, dict):
            values.append(item.get("value"))
            continue
        values.append(getattr(item, "value", None))
    return values


def _clarification_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    从 graph 输出里统一提取“等待澄清”的前端友好结构。

    澄清信息可能来自显式字段，也可能藏在 LangGraph interrupt 里；
    facade 层把这两种来源收敛成同一套返回格式。
    """
    question = str(payload.get("clarification_question") or "").strip()
    if payload.get("awaiting_clarification"):
        report = str(payload.get("report") or question).strip()
        question = question or report
        if question:
            return {
                "awaiting_clarification": True,
                "clarification_question": question,
                "report": report or question,
            }
    for value in _interrupt_values(payload):
        if isinstance(value, dict):
            question = str(
                value.get("clarification_question")
                or value.get("question")
                or value.get("report")
                or ""
            ).strip()
        else:
            question = str(value or "").strip()
        if question:
            return {
                "awaiting_clarification": True,
                "clarification_question": question,
                "report": question,
            }
    return None


async def _graph_input(
    *,
    agent: Any,
    user_query: str,
    state: dict[str, Any],
    session_id: str,
    conversation_id: str | None,
    thread_id: str | None,
    execution_kind: str = "new",
) -> Any:
    """
    根据当前 thread 快照决定本轮 graph 的真实输入。

    Args:
        agent: 共享 LangGraph 实例
        user_query: 本轮用户输入
        state: route/facade 已组装好的初始 AgentState
        session_id: LangGraph 会话 ID
        conversation_id: 前端对话 ID，用于补齐 checkpoint config
        thread_id: sessions 表已持久化的 checkpoint key

    Returns:
        Any: 首次执行时返回完整 state；中断恢复时返回 `Command(resume=...)`
    """
    # 1. 【研究执行】【命令语义】新问题从新输入开始，只有显式继续才消费澄清答案
    if execution_kind == "new":
        return state
    if execution_kind not in {"resume", "retry"}:
        raise ValueError("未知的研究执行命令")
    snapshot = None
    if callable(getattr(agent, "aget_state", None)):
        snapshot = await agent.aget_state(
            _agent_config(session_id, conversation_id, thread_id)
        )
    if execution_kind == "resume":
        if not snapshot_has_pending_interrupt(snapshot) or not user_query.strip():
            raise ValueError("继续研究需要已保存的澄清中断和用户回答")
        return Command(resume=user_query, update=_resume_scope_update(state, snapshot))
    if snapshot_has_pending_interrupt(snapshot):
        raise ValueError("研究仍在等待澄清，请先回答澄清问题")
    if snapshot is None:
        raise ValueError("没有可恢复的研究 checkpoint")
    # 2. 【研究执行】【失败恢复】官方空输入恢复原执行，更新当前授权而不伪造 resume 答案
    scope_update = _resume_scope_update(state, snapshot)
    if scope_update:
        await agent.aupdate_state(_agent_config(session_id, conversation_id, thread_id), scope_update)
    return None


def _resume_scope_update(state: dict[str, Any], snapshot: Any) -> dict[str, Any] | None:
    """【检索】【恢复版本】保留已完成研究状态，只替换本轮重新授权的范围和发布快照。"""
    scope = state.get("project_scope")
    if not isinstance(scope, dict):
        return None
    previous = snapshot if isinstance(snapshot, dict) else snapshot.values
    structured = {**(previous.get("structured_query") or {}), "project_scope": scope,
        "paper_ids": scope.get("paper_ids", []), "filter_expr": scope.get("filter_expr", "")}
    return {"project_scope": scope, "filter_expr": structured["filter_expr"], "structured_query": structured}


def _graph_timeout_seconds(settings: Any, intent: Any) -> float:
    if intent in ("deep_research", "external_research"):
        return float(settings.deep_research_graph_timeout_seconds)
    return float(settings.graph_timeout_seconds)


def _begin_trace_context(
    *,
    session_id: str,
    extra: dict[str, Any] | None,
) -> Any:
    """
    为请求建立 request-scoped trace 上下文。

    收敛后所有入口跑的都是同一条主链，不再按 preset intent 关门；唯一的门槛是
    本请求已经自建过 trace（外层流式协程或中间件已建立），此时不重复覆盖。

    Args:
        session_id: LangGraph 会话 ID
        extra: facade 上层传入的请求级扩展字段

    Returns:
        Any: trace token；若当前请求已有 trace，则返回 None
    """
    if get_trace_ctx() is not None:
        return None
    payload = extra or {}
    return set_trace_ctx(TraceContext(
        request_id=str(payload.get("request_id") or "") or str(uuid.uuid4()),
        session_id=session_id or "",
        user_id=str(payload.get("user_id") or "unknown"),
    ))


def begin_stream_trace_context(*, extra: dict[str, Any] | None, session_id: str) -> Any:
    """
    为流式端点在“真正迭代 stream 的协程”里建立 RAG trace 上下文。

    流式路径与同步端点的关键差异：`open_graph_stream` 返回的是惰性 async 生成器，
    图节点（含写 trace 的 execute_write_back）在外层 `async for` 迭代时才执行。
    contextvar 必须在那个迭代协程里 set，才能被图节点通过 get_trace_ctx() 取到；
    在 open_graph_stream 内部 set 会随其栈帧销毁而失效。

    这里复用同步端点的 `_begin_trace_context`（含 ctx 已存在的门槛）。

    Args:
        extra: 主图 extra，需含 request_id / user_id
        session_id: open_graph_stream 返回的实际 session_id

    Returns:
        Any: trace token（供调用方在 finally reset_trace_ctx）；不需建立时返回 None
    """
    payload = extra or {}
    return _begin_trace_context(
        session_id=session_id,
        extra=payload,
    )


async def _finalize_graph_result(
    result: dict[str, Any],
    *,
    state: dict[str, Any],
    started_at: float,
    user_query: str,
) -> dict[str, Any]:
    """
    把 graph 原始输出整理成 API 层可稳定消费的最终结果。

    Args:
        result: graph 原始返回状态
        state: 本轮执行前的初始状态
        started_at: facade 开始调用 graph 的时间戳
        user_query: 用户原始问题

    Returns:
        dict[str, Any]: 已补齐 clarification、trace 和短路观测信息的结果
    """
    # 1. 先把 graph 输出里“等待澄清”这类内部信号规范化成前端可识别字段
    clarification = _clarification_payload(result)
    if clarification:
        result = {**result, **clarification}
    ctx = get_trace_ctx()
    from app.services.guardrails_trace_service import attach_short_circuit_trace
    # 2. 再为 guardrails reject 等短路路径补齐本轮应有的 trace 结果
    result = await attach_short_circuit_trace(
        result,
        ctx=ctx,
        state=state,
        started_at=started_at,
        user_query=user_query,
    )
    # 3. 若本轮已有 trace 但尚未生成摘要，则从 trace payload 派生前端展示摘要
    if ctx is not None and result.get("trace_summary") is None:
        from app.middleware.trace_context import build_trace_summary
        trace_dict = result.get("_rag_trace_dict") or {}
        if trace_dict:
            result["trace_summary"] = build_trace_summary(ctx, trace_dict)
    return result


async def invoke_graph(user_query: str, extra: dict[str, Any] | None = None, *, execution_kind: str = "new") -> tuple[dict[str, Any], str]:
    """【检索】【问答版本】在整次图调用内固定发布版本，成功、失败和取消后均释放占用。"""
    from app.services.upload_read_service import publication_scope_lifetime

    async with publication_scope_lifetime((extra or {}).get("project_scope")):
        return await _invoke_graph(user_query, extra, execution_kind=execution_kind)


async def _invoke_graph(user_query: str, extra: dict[str, Any] | None = None, *, execution_kind: str = "new") -> tuple[dict[str, Any], str]:
    """
    执行一次普通 graph 调用。

    这是同步请求最常走的入口，核心流程是：
    1. 组装初始状态
    2. 获取共享 graph
    3. 判断是新执行还是 interrupt 恢复
    4. 依据 intent 选取合适的超时时间
    5. 统一记录 observability 事件并标准化输出
    """
    started_at = time.monotonic()
    state, session_id, conversation_id = _build_agent_state(user_query, extra)
    agent = await get_shared_agent()
    graph_input = await _graph_input(
        agent=agent,
        user_query=user_query,
        state=state,
        session_id=session_id,
        conversation_id=conversation_id,
        thread_id=state.get("thread_id"),
        execution_kind=execution_kind,
    )
    settings = get_settings()
    intent = state.get("intent")
    graph_timeout = _graph_timeout_seconds(settings, intent)
    trace_token = _begin_trace_context(session_id=session_id, extra=extra)
    try:
        result = await _invoke_agent(agent, graph_input,
            config=_agent_config(session_id, conversation_id, state.get("thread_id")),
            timeout=graph_timeout, intent=intent, started_at=started_at, query=user_query, session_id=session_id)
        observability.record_timed_event(
            "research_completed",
            "graph_facade",
            started_at,
            query=user_query,
            result_count=1 if result.get("report") else 0,
            session_id=session_id,
        )
        result = await _finalize_graph_result(
            result,
            state=state,
            started_at=started_at,
            user_query=user_query,
        )
        return result, session_id
    finally:
        if trace_token is not None:
            reset_trace_ctx(trace_token)


async def _invoke_agent(agent: Any, graph_input: Any, *, config: dict, timeout: float,
                        intent: str | None, started_at: float, query: str, session_id: str) -> dict:
    """参数为输入与运行上下文；按配置执行图，0 不限总时长，返回真实产物或保留原错误。"""
    effective_timeout = execution_timeout_seconds(timeout)
    try:
        return await asyncio.wait_for(agent.ainvoke(graph_input, config=config), timeout=effective_timeout)
    except asyncio.TimeoutError as exc:
        if effective_timeout is None:
            observability.record_timed_event("research_failed", "graph_facade", started_at, query=query, session_id=session_id)
            raise
        observability.record_timed_event("research_timeout", "graph_facade", started_at, query=query, session_id=session_id)
        detail = (f"Deep Research 超时（{timeout:.0f}s），请缩小研究范围或稍后重试" if intent == "deep_research"
            else f"RAG 问答超时（{timeout:.0f}s），请简化问题或稍后重试")
        raise HTTPException(status_code=504, detail=detail) from exc
    except Exception:
        observability.record_timed_event("research_failed", "graph_facade", started_at, query=query, session_id=session_id)
        raise


async def open_graph_stream(user_query: str, extra: dict[str, Any] | None = None, *, execution_kind: str = "new"):
    """
    打开 graph 流式输出。

    这里兼容两种 LangGraph 流模式：
        - `astream_events`: 更细粒度的事件流
        - `astream`: 更粗粒度的 values 流

        facade 层负责把两种底层形式统一包装成上层可消费的 stream kind。

        Args:
            user_query: 用户本轮输入
            extra: route 层附带的 session / scope / intent 等上下文

        Returns:
            tuple: `(stream_kind, graph_stream, session_id)`，供上层继续封装 SSE
    """
    started_at = time.monotonic()
    state, session_id, conversation_id = _build_agent_state(user_query, extra)
    agent = await get_shared_agent()
    config = _agent_config(
        session_id,
        conversation_id,
        state.get("thread_id"),
    )
    graph_input = await _graph_input(
        agent=agent,
        user_query=user_query,
        state=state,
        session_id=session_id,
        conversation_id=conversation_id,
        thread_id=state.get("thread_id"),
        execution_kind=execution_kind,
    )
    try:
        if callable(getattr(agent, "astream_events", None)):
            stream = agent.astream_events(
                graph_input,
                config=config,
                version="v2",
                stream_mode=["custom", "values"],
            )
            observability.record_timed_event(
                "research_stream_opened",
                "graph_facade",
                started_at,
                query=user_query,
                session_id=session_id,
            )
            return "events", stream, session_id
        stream = agent.astream(graph_input, config=config, stream_mode="values")
        observability.record_timed_event(
            "research_stream_opened",
            "graph_facade",
            started_at,
            query=user_query,
            session_id=session_id,
        )
        return "values", stream, session_id
    except Exception:
        observability.record_timed_event(
            "research_stream_failed",
            "graph_facade",
            started_at,
            query=user_query,
            session_id=session_id,
        )
        raise


def research_response_payload(
    result: dict[str, Any],
    session_id: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """统一整理 `/research` 的同步返回结构。"""
    clarification = _clarification_payload(result)
    report = clarification["report"] if clarification else (
        result.get("answer") if result.get("final_delivery_report") else result.get("report", "")
    )
    session_identity = build_session_identity(session_id, conversation_id)
    payload = {
        **{key: result[key] for key in DELIVERY_METADATA_FIELDS if key in result},
        "report": report,
        "intent": result.get("intent"),
        "core_intent": result.get("core_intent"),
        **session_identity,
        "research_mode": result.get("research_mode"),
        "selected_papers": result.get("selected_papers", []),
        "comparison_matrix": result.get("comparison_matrix", []),
        "evidence_cards": result.get("evidence_cards", []),
        "source_records": result.get("source_records", []),
        "reasoning_summaries": result.get("reasoning_summaries", []),
        "key_findings": result.get("key_findings", []),
        "uncertainties": result.get("uncertainties", []),
        "idea_candidates": result.get("idea_candidates", []),
        "external_gaps": result.get("external_gaps", []),
        "follow_up_questions": result.get("follow_up_questions", []),
        "candidate_papers_to_import": result.get("candidate_papers_to_import", []),
    }
    if clarification:
        payload["awaiting_clarification"] = True
        payload["clarification_question"] = clarification["clarification_question"]
    return payload


def _extract_stream_report(state: dict[str, Any]) -> tuple[str, Optional[dict[str, Any]]]:
    """提取当前应展示给前端的 report 文本。"""
    clarification = _clarification_payload(state)
    if clarification:
        return clarification["report"], clarification
    if state.get("final_delivery_report"):
        return state.get("answer", ""), None
    return state.get("report") or "", None


def _report_delta(previous: str, current: str) -> tuple[str, str]:
    """
    仅输出 report 新增尾部片段。

    这样前端收到的是“增量 token”，而不是每次都重复收到完整 report。
    """
    if not current or current == previous:
        return "", previous
    if current.startswith(previous):
        return current[len(previous):], current
    return current, current


def _stream_done_payload(state: dict[str, Any], session_id: str) -> dict[str, Any]:
    """统一封装流结束事件。"""
    clarification = _clarification_payload(state)
    payload = {
        **research_response_payload(state, session_id),
        "type": "done",
        "intent": state.get("intent", "unknown"),
        "session_id": session_id,
        "research_mode": state.get("research_mode"),
        "key_findings": state.get("key_findings", []),
        "uncertainties": state.get("uncertainties", []),
        "idea_candidates": state.get("idea_candidates", []),
        "external_gaps": state.get("external_gaps", []),
        "follow_up_questions": state.get("follow_up_questions", []),
        "candidate_papers_to_import": state.get("candidate_papers_to_import", []),
        "evidence_count": len(state.get("evidence_cards", []) or []),
    }
    if clarification:
        payload["awaiting_clarification"] = True
        payload["clarification_question"] = clarification["clarification_question"]
    return payload


def _stream_done_event_payload(state: dict[str, Any], session_id: str) -> dict[str, Any]:
    """把最终完成态包装成 stage=done 事件。"""
    payload = _stream_done_payload(state, session_id)
    payload["type"] = "event"
    payload["stage"] = "done"
    return payload


def _route_event_payload(event: dict[str, Any]) -> Optional[dict[str, Any]]:
    """把 router 节点结束事件翻译成前端可消费结构。"""
    if event.get("event") != "on_chain_end" or event.get("name") != "router":
        return None
    output = event.get("data", {}).get("output", {})
    intent = output.get("intent")
    if not intent:
        return None
    return {"type": "event", "stage": "route", "node": "router", "intent": intent}


def _graph_stream_chunk(event: dict[str, Any]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """从 LangGraph 事件流中提取有效 chunk。"""
    if event.get("event") != "on_chain_stream" or event.get("name") != "LangGraph":
        return None, None
    chunk = event.get("data", {}).get("chunk")
    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return None, None
    mode, payload = chunk
    if not isinstance(mode, str) or not isinstance(payload, dict):
        return None, None
    return mode, payload


async def _iter_value_payloads(graph_stream, session_id: str) -> AsyncIterator[dict[str, Any]]:
    """把 values 模式输出转换成前端统一 payload。"""
    sent_report = ""
    final_state: dict[str, Any] = {}
    async for state in graph_stream:
        if not isinstance(state, dict):
            continue
        final_state = state
        report_text, _ = _extract_stream_report(state)
        delta, sent_report = _report_delta(sent_report, report_text)
        if delta:
            yield {"type": "token", "content": delta}
    yield _stream_done_payload(final_state, session_id)


async def _iter_event_payloads(event_stream, session_id: str) -> AsyncIterator[dict[str, Any]]:
    """
    把 C2 事件流转换成前端统一 payload。

    这里会同时透传：
    - router 等关键阶段事件
    - 自定义 stage 事件
    - report 的增量文本
    - 最终 done 事件
    """
    sent_report = ""
    final_state: dict[str, Any] = {}
    async for event in event_stream:
        route_payload = _route_event_payload(event)
        if route_payload:
            yield route_payload
        mode, payload = _graph_stream_chunk(event)
        if mode == "custom":
            yield {"type": "event", **payload}
        elif mode == "values":
            final_state = payload
            report_text, _ = _extract_stream_report(payload)
            delta, sent_report = _report_delta(sent_report, report_text)
            if delta:
                yield {"type": "token", "content": delta}
    yield _stream_done_event_payload(final_state, session_id)
    yield _stream_done_payload(final_state, session_id)


def _encode_sse(payload: dict[str, Any]) -> str:
    """把 JSON payload 编码为 SSE 文本帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def encode_sse(payload: dict[str, Any]) -> str:
    """对外暴露的 SSE 帧编码入口，供 route 层自行拼装流式响应。"""
    return _encode_sse(payload)


def _cross_paper_done_payload(state: dict[str, Any], session_id: str) -> dict[str, Any]:
    """
    组装跨论文问答流式结束事件。

    最终答案由 planner 综合一次性产出，这里在流末尾把答案与证据、planner 元数据
    一起下发，供前端把消息整块落地并渲染证据/计划面板。
    """
    answer = state.get("answer")
    if answer is None and not state.get("final_delivery_report"):
        answer = state.get("report", "")
    return {
        **{key: state[key] for key in DELIVERY_METADATA_FIELDS if key in state},
        "type": "done",
        "session_id": session_id,
        "answer": answer,
        "citations": state.get("citations", []),
        "paper_sources": state.get("paper_sources", []),
        "chunk_citations": state.get("chunk_citations", []),
        "graph_evidence": state.get("graph_evidence", []),
        "graph_hint_section": state.get("graph_hint_section", ""),
        "graph_hint_full": state.get("graph_hint_full", ""),
        "evidence_summary": state.get("evidence_summary", ""),
        "evidence_records": state.get("evidence_records", []),
        "answer_brief": state.get("answer_brief", {}),
        "missing_evidence": state.get("missing_evidence", []),
        "research_scope": state.get("research_scope", {}),
        "research_objects": state.get("research_objects", []),
        "note_actions": state.get("note_actions", []),
        "synthesis_state": state.get("synthesis_state", {}),
        "research_plan": state.get("research_plan"),
        "research_unit_results": state.get("research_unit_results", []),
        "coverage_review": state.get("coverage_review"),
        "planner_trace": state.get("planner_trace"),
        "evidence_sufficiency_report": state.get("evidence_sufficiency_report"),
        "repair_query_plans": state.get("repair_query_plans"),
        "planner_repair_trace": state.get("planner_repair_trace"),
        "planner_repair_merge_report": state.get("planner_repair_merge_report"),
        "final_answer_eligibility_report": state.get("final_answer_eligibility_report"),
        "planner_final_blocks": state.get("planner_final_blocks", []),
        "planner_final_validation_errors": state.get("planner_final_validation_errors", []),
        "final_delivery_report": state.get("final_delivery_report"),
        "final_citation_bindings": state.get("final_citation_bindings"),
        "final_generation_attempt": state.get("final_generation_attempt"),
        "final_delivery_applicability": state.get("final_delivery_applicability"),
        "trace_summary": state.get("trace_summary"),
        "source_status": state.get("source_status", {}),
        "source_status_reasons": state.get("source_status_reasons", {}),
        "runtime_chain": state.get("runtime_chain"),
        "request_id": state.get("request_id"),
        "confidence": state.get("answer_confidence"),
        "dialogue_mode": state.get("dialogue_mode"),
        # 项目科研回合的轮级字段：本轮生效深度、深度变更轨迹与记忆候选写回终态。
        # 这里是手列白名单，不在表里的 state 键不会下发；缺失保持 None，不补默认值
        "turn_depth": state.get("turn_depth"),
        "turn_outcome": state.get("turn_outcome"),
        "depth_transitions": state.get("depth_transitions"),
        "memory_writeback": state.get("memory_writeback"),
        # 记忆读取终态随回答一起下发：回答卡要能显示「本轮用了哪几条项目历史判断、
        # 来源是谁」，以及未用记忆 / 读取失败这两种真实结果。四态不归并靠这个键传下去
        "memory_read_result": state.get("memory_read_result"),
        # 与非流式失败契约同构：稳定错误码 + 用户摘要
        "error_code": state.get("error_code"),
        "error_summary": state.get("error_summary"),
    }


async def iter_cross_paper_payloads(
    graph_stream,
    *,
    stream_kind: str,
    session_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """
    把跨论文问答的 graph 流转换成前端统一 payload 序列。

    与深度研究流复用同一套阶段事件/增量文本提取逻辑，区别只在流结束事件：
    这里给出携带最终答案、证据与 planner 元数据的跨论文 done 载荷。

    Args:
        graph_stream: LangGraph 原始异步流
        stream_kind: 底层流模式，取值为 `events` 或 `values`
        session_id: 当前流式请求绑定的 session_id

    Yields:
        dict: 阶段事件、增量 token 与最终 done 载荷
    """
    sent_report = ""
    final_state: dict[str, Any] = {}
    if stream_kind == "events":
        async for event in graph_stream:
            route_payload = _route_event_payload(event)
            if route_payload:
                yield route_payload
            mode, payload = _graph_stream_chunk(event)
            if mode == "custom":
                yield {"type": "event", **payload}
            elif mode == "values":
                final_state = payload
                delta, sent_report = _report_delta(sent_report, payload.get("report") or "")
                if delta:
                    yield {"type": "token", "content": delta}
    else:
        async for state in graph_stream:
            if not isinstance(state, dict):
                continue
            final_state = state
            delta, sent_report = _report_delta(sent_report, state.get("report") or "")
            if delta:
                yield {"type": "token", "content": delta}
    yield _cross_paper_done_payload(final_state, session_id)


async def iter_research_payloads(
    graph_stream,
    *,
    stream_kind: str,
    session_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """
    把 graph 流式结果转换为稳定研究 payload。

    该边界保留结构化 payload，让 route 能在编码 SSE 前根据最终 done/error
    写 durable session 状态，而不需要反向解析已经编码的字符串。

    Args:
        graph_stream: LangGraph 原始异步流
        stream_kind: 底层流模式，取值为 `events` 或 `values`
        session_id: 当前流式请求绑定的 session_id

    Yields:
        dict[str, Any]: session、阶段、token 与最终 done payload
    """
    yield {"type": "event", "stage": "session", "session_id": session_id}
    iterator = (
        _iter_event_payloads(graph_stream, session_id)
        if stream_kind == "events"
        else _iter_value_payloads(graph_stream, session_id)
    )
    async for payload in iterator:
        yield payload


async def stream_research_sse(
    graph_stream,
    *,
    stream_kind: str,
    session_id: str,
) -> AsyncIterator[str]:
    """
    把结构化研究 payload 编码为浏览器可消费的 SSE 文本帧。

    Args:
        graph_stream: LangGraph 原始异步流
        stream_kind: 底层流模式，取值为 `events` 或 `values`
        session_id: 当前流式请求绑定的 session_id

    Yields:
        str: 编码好的 SSE 文本帧
    """
    try:
        async for payload in iter_research_payloads(
            graph_stream,
            stream_kind=stream_kind,
            session_id=session_id,
        ):
            yield _encode_sse(payload)
    except Exception as exc:
        yield _encode_sse({"type": "error", "detail": str(exc)})
