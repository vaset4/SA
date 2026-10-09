"""
对话、会话与恢复路由。

这是“用户发问 -> 进入 Agent/Graph -> 拿到回答”的最关键 API 入口之一。
阅读这个文件时，建议重点区分三类 ID：

1. `session_id`
   对应 LangGraph checkpoint 维度，用来恢复图状态
2. `conversation_id`
   对应前端展示库 conversations/messages，用来展示历史消息
3. `project_id`
   对应研究空间，用来约束本次请求允许访问的论文范围

这三个概念会在同一次请求里同时出现，但它们分别服务于状态恢复、UI 展示、检索范围控制。
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field, replace

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.api.facades.graph_facade import (
    begin_stream_trace_context,
    get_shared_agent_state,
    invoke_graph,
    iter_cross_paper_payloads,
    open_graph_stream,
)
from app.api.facades.research_contract_facade import (
    build_dialogue_structured_query,
    build_paper_focus_contract,
    build_project_scope_contract,
    resolve_dialogue_mode,
)
from app.api.facades.rag_run_stream_facade import build_run_event_response
from app.api.facades.session_identity_facade import resolve_conversation_identity
from app.api.schemas.chat import ChatRequest
from app.api.schemas.research import CrossPaperQARequest, SessionCreateRequest
from app.config import get_settings
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.models.research_contracts import ScopeOwnershipError
from app.models.result_delivery_contracts import DELIVERY_METADATA_FIELDS, DeliveryConflict
from app.middleware.observability import log_structured_event
from app.middleware.trace_context import reset_trace_ctx
from app.services import conversation_service, rag_run_service
from app.services.result_delivery_service import deliver_result, read_delivery_response
from app.services.llm_service import classify_llm_upstream_error
from app.services.project_scope_service import build_project_context
from app.services.rag_run_executor_service import get_rag_run_executor_registry
from app.services.research_session_lifecycle import (
    LegacyResearchSessionNotRecoverable,
    ResearchSessionNotFound,
    allowed_actions_for_status,
    get_research_session_lifecycle,
)
from app.services.session_mapping_service import ensure_session_id

router = APIRouter()
logger = logging.getLogger(__name__)


def _completed_qa_outcome(result: dict) -> str:
    """判定 cross-paper 问答完成后的 outcome 标签（四态可分，纯函数便于测试）。

    规划终态按 error_code 与 source_status.planner 取值区分，四种情形互不合并：
    - evidence_not_found_in_scope（或 planner=no_evidence）：范围内检索过但无证据，正常收尾
    - planner=partial：一部分子问题拿到了证据、另一部分没查清，交付的是部分完成，
      不能记成完整成功，否则「只答了一部分」在回合统计里与全答对无法区分
    - 其余非空 error_code / planner==failed：系统性硬失败
    - 空答案：fallback_empty_answer；guardrails 拒绝：guardrails_reject

    参数:
        result: 主图产物字典（含 answer / error_code / source_status / guardrails_action）
    返回:
        outcome 标签字符串
    """
    raw_answer = (result.get("answer") or result.get("report") or "").strip()
    if result.get("guardrails_action") == "reject":
        return "guardrails_reject"
    # 1. 【研究交付】【聊天投影】同次报告优先于正文非空、单元状态和旧错误标签
    from app.services.internal_research.research_planner_final_delivery import (
        delivery_report_from_state,
        project_final_delivery,
    )

    delivery = delivery_report_from_state(result)
    if delivery is not None:
        return project_final_delivery(delivery)["chat_result_status"]
    source_status = result.get("source_status")
    if not isinstance(source_status, dict):
        source_status = {}
    planner_status = source_status.get("planner")
    error_code = result.get("error_code")
    # 诚实收尾：范围内查过但没找到，区别于系统失败
    if error_code == "evidence_not_found_in_scope" or planner_status == "no_evidence":
        return "evidence_not_found"
    # 部分完成：有已查清的部分也有未查清的部分，自成一档
    if planner_status == "partial":
        return "partial_answer"
    # 系统性规划失败
    if planner_status == "failed" or error_code:
        return "planner_failed"
    # 正常有答案 / 空答案兜底
    return "fallback_empty_answer" if not raw_answer else "answered"


def _map_cross_paper_http_error(exc: Exception) -> HTTPException | None:
    """把底层检索异常映射成更明确的 HTTP 错误。"""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, DeliveryConflict):
        return HTTPException(status_code=409, detail=str(exc))
    message = str(exc)
    if "Milvus query timed out" in message or "LightRAG query timed out" in message:
        return HTTPException(status_code=504, detail=f"cross-paper retrieval timed out: {message}")
    if "milvus failed:" in message or "graph failed:" in message:
        return HTTPException(status_code=503, detail=f"cross-paper retrieval failed: {message}")
    # 上游模型服务繁忙/超时/鉴权类错误：还原为可读原因，让前端提示"稍后重试"而非笼统 500
    upstream = classify_llm_upstream_error(exc)
    if upstream is not None:
        status_code, reason = upstream
        return HTTPException(status_code=status_code, detail=reason)
    return None


def _completed_run_metadata(result: dict, response: dict) -> dict:
    """【研究交付】【运行元数据】组装同步与流式共用的持久化消息字段。

    参数为本次图结果和 API 响应；返回保留研究输入及同次交付报告/引用的运行元数据。
    """
    metadata_sources = (result, response)
    delivery_source = result if result.get("final_generation_attempt") else response
    return {
        "run_id": response.get("run_id"),
        "run_status": response.get("run_status"),
        "request_id": result.get("request_id"),
        "trace_summary": result.get("trace_summary"),
        "runtime_chain": result.get("runtime_chain"),
        "source_status": result.get("source_status"),
        "source_status_reasons": result.get("source_status_reasons"),
        "citations": response.get("citations", []),
        "paper_sources": response.get("paper_sources", []),
        "chunk_citations": response.get("chunk_citations", []),
        "graph_evidence": response.get("graph_evidence", []),
        "graph_hint_section": response.get("graph_hint_section", ""),
        "graph_hint_full": response.get("graph_hint_full", ""),
        "evidence_records": _metadata_value(metadata_sources, "evidence_records", []),
        "answer_brief": _metadata_value(metadata_sources, "answer_brief", {}),
        "missing_evidence": _metadata_value(metadata_sources, "missing_evidence", []),
        "research_plan": _metadata_value(metadata_sources, "research_plan"),
        "research_unit_results": _metadata_value(metadata_sources, "research_unit_results", []),
        "coverage_review": _metadata_value(metadata_sources, "coverage_review"),
        "planner_trace": _metadata_value(metadata_sources, "planner_trace"),
        "evidence_sufficiency_report": _metadata_value(metadata_sources, "evidence_sufficiency_report"),
        "repair_query_plans": _metadata_value(metadata_sources, "repair_query_plans", []),
        "planner_repair_trace": _metadata_value(metadata_sources, "planner_repair_trace", {}),
        "planner_repair_merge_report": _metadata_value(metadata_sources, "planner_repair_merge_report", {}),
        "final_answer_eligibility_report": _metadata_value(metadata_sources, "final_answer_eligibility_report", {}),
        "planner_final_blocks": _metadata_value(metadata_sources, "planner_final_blocks", []),
        "planner_final_validation_errors": _metadata_value(metadata_sources, "planner_final_validation_errors", []),
        "final_delivery_report": delivery_source.get("final_delivery_report"),
        "final_citation_bindings": delivery_source.get("final_citation_bindings"),
        "final_generation_attempt": delivery_source.get("final_generation_attempt"),
        "final_delivery_applicability": delivery_source.get("final_delivery_applicability"),
        "error_code": _metadata_value(metadata_sources, "error_code"),
        "error_summary": _metadata_value(metadata_sources, "error_summary"),
        # 项目科研回合的轮级字段：本轮生效深度、深度变更轨迹与记忆候选写回终态。
        # 同步与流式两条路径共用这一处，缺失的键保持 None，不补默认值冒充「本轮没变化」
        "turn_depth": _metadata_value(metadata_sources, "turn_depth"),
        "depth_transitions": _metadata_value(metadata_sources, "depth_transitions"),
        "memory_writeback": _metadata_value(metadata_sources, "memory_writeback"),
        "turn_outcome": _metadata_value(metadata_sources, "turn_outcome"),
        # 记忆读取终态进 run metadata：历史消息重新打开时，回答卡仍要能显示
        # 本轮引用了哪几条项目历史判断，以及未用记忆 / 读取失败的真实结果
        "memory_read_result": _metadata_value(metadata_sources, "memory_read_result"),
    }


def _metadata_value(sources: tuple[dict, dict], key: str, default=None):
    result, response = sources
    value = result.get(key)
    if value is not None:
        return value
    return response.get(key, default)


async def _previous_rag_context(user_id: str, conversation_id: str | None) -> dict:
    if not conversation_id:
        return {}
    run = await conversation_service.get_previous_delivery(user_id=user_id, conversation_id=conversation_id)
    if not run:
        return {}
    metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
    return {
        "query": str(run.get("query") or ""),
        "answer": str(run.get("answer") or ""),
        "evidence_records": metadata.get("evidence_records") or [],
        "answer_brief": metadata.get("answer_brief") or {},
    }


@dataclass(frozen=True)
class _CrossPaperContext:
    """一次跨论文问答已构建好的执行上下文，供同步与流式端点共享，避免两处 scope 逻辑漂移。"""

    conversation_id: str | None
    session_id: str | None
    project_id: str | None
    scoped_ids: list[str]
    filter_expr: str
    dialogue_mode: str
    run: dict | None
    graph_extra: dict = field(default_factory=dict)


async def _build_cross_paper_context(
    *,
    req: ChatRequest | CrossPaperQARequest,
    user_id: str,
) -> _CrossPaperContext:
    """【聊天】【范围构建】参数为普通/跨论文聊天请求及用户；返回可信范围和图上下文。

    同步、流式和自由聊天共用身份及项目校验；运行领取与结果提交由受管执行入口负责。
    """
    # 1. 先确认用户能否访问这次请求关联的 conversation/session
    conversation_id = await _resolve_identity_or_404(user_id, req.conversation_id, req.session_id)
    paper_ids = [paper_id.strip() for paper_id in req.paper_ids if paper_id.strip()]
    try:
        # 2. 把请求范围收敛成实际允许访问的项目论文集合
        project_context = await build_project_context(
            req.project_id,
            requested_paper_ids=paper_ids or None,
            user_id=user_id,
        )
    except ScopeOwnershipError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    scoped_ids = project_context["paper_ids"]
    rejected_ids = project_context.get("rejected_paper_ids", [])
    logger.info(
        "cross_paper_qa scope: requested=%d scoped=%d rejected=%d "
        "project_id=%s project_paper_count=%d requested_ids=%s scoped_ids=%s",
        len(paper_ids),
        len(scoped_ids),
        len(rejected_ids),
        req.project_id or "-",
        len((project_context.get("resolved_scope") or {}).get("paper_ids", [])),
        paper_ids,
        scoped_ids,
    )
    # 3. 把合法论文范围转换成后续检索链可直接执行的 filter_expr
    #    范围为空是合法状态：项目还没有论文时仍要能谈研究问题与已确认记忆，
    #    「这一问要不要论文证据」由 planner 的处理深度判断，不在入口层提前拒绝
    escaped_ids = [paper_id.replace("\\", "\\\\").replace('"', '\\"') for paper_id in scoped_ids]
    filter_expr = " or ".join(f'paper_id == "{paper_id}"' for paper_id in escaped_ids)
    dialogue_mode = resolve_dialogue_mode(
        req.query,
        explicit_mode=req.dialogue_mode,
        project_id=req.project_id,
        paper_ids=scoped_ids,
    )
    structured_query = build_dialogue_structured_query(
        query=req.query,
        project_context=project_context,
        project_id=req.project_id,
        explicit_paper_ids=scoped_ids,
        dialogue_mode=dialogue_mode,
    )
    previous_rag_context = await _previous_rag_context(user_id, conversation_id)
    graph_extra = {
        "intent": "rag_qa",
        "filter_expr": filter_expr,
        "chat_history": req.chat_history,
        "session_id": req.session_id,
        "conversation_id": conversation_id,
        "dialogue_mode": dialogue_mode,
        "project_scope": build_project_scope_contract(project_context),
        "paper_focus": build_paper_focus_contract(project_context, explicit_paper_ids=scoped_ids),
        "structured_query": structured_query,
        "previous_rag_context": previous_rag_context,
        "user_id": user_id,
    }
    return _CrossPaperContext(
        conversation_id=conversation_id,
        session_id=req.session_id,
        project_id=project_context["project_id"] or req.project_id,
        scoped_ids=scoped_ids,
        filter_expr=filter_expr,
        dialogue_mode=dialogue_mode,
        run=None,
        graph_extra=graph_extra,
    )


def _empty_session_payload(session_id: str, session_row: dict) -> dict:
    """
    为已创建但尚无 checkpoint 的 session 返回空快照。

    这能让前端把“空会话”和“会话不存在”区分开：
    前者是合法状态，后者才应该报 404。
    """
    topic = session_row.get("topic") or "新对话"
    return {
        "session_id": session_id,
        "conversation_id": session_row.get("conversation_id"),
        "project_id": session_row.get("project_id"),
        "thread_id": session_row.get("thread_id"),
        "paper_ids": session_row.get("paper_ids") or [],
        "research_mode": session_row.get("research_mode"),
        "status": session_row.get("status") or "created",
        "allowed_actions": session_row.get("allowed_actions")
        or allowed_actions_for_status(session_row.get("status") or "created"),
        "last_error": session_row.get("last_error"),
        "attempt_count": int(session_row.get("attempt_count") or 0),
        "topic": topic,
        "chat_history": [],
        "values": {
            "session_id": session_id,
            "thread_id": session_row.get("thread_id"),
            "chat_history": [],
        },
        "next": [],
        "metadata": {},
        "tasks": [],
        "interrupts": [],
    }


async def _resolve_identity_or_404(
    user_id: str,
    conversation_id: str | None,
    session_id: str | None,
    allow_missing_conversation: bool = False,
) -> str | None:
    """
    统一解析对话/会话身份，并把归属错误转换为 HTTP 404。

    Args:
        user_id: 当前登录用户 ID
        conversation_id: 前端展示层对话 ID
        session_id: LangGraph 会话 ID
        allow_missing_conversation: 是否允许“session 存在但尚未绑定对话”的合法空态

    Returns:
        str | None: 解析后的 conversation_id；允许缺失时可能为 None
    """
    try:
        identity = await resolve_conversation_identity(
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            allow_missing_conversation=allow_missing_conversation,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return identity.conversation_id


@router.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user_with_rate_limit),
               idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=8, max_length=128)):
    """【自由对话】【受管执行】参数为用户问题及动作标识；返回统一保存事实和原有聊天上下文。"""
    user_id = get_user_id(user)
    ctx = await _build_cross_paper_context(req=req, user_id=user_id)
    ctx = replace(ctx, graph_extra={**ctx.graph_extra, "intent": "free_chat",
                                   "context_summary": req.context_summary or ""})
    return await _run_chat_sync(req=req, ctx=ctx, user_id=user_id, idempotency_key=idempotency_key)


@router.post("/qa/cross-paper")
async def cross_paper_qa(req: CrossPaperQARequest, user: dict = Depends(get_current_user_with_rate_limit),
                         idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=8, max_length=128)):
    """【跨论文问答】【受管执行】参数为用户问题、身份和动作标识；返回完整结果及真实保存状态。"""
    user_id = get_user_id(user)
    ctx = await _build_cross_paper_context(req=req, user_id=user_id)
    return await _run_chat_sync(req=req, ctx=ctx, user_id=user_id, idempotency_key=idempotency_key)


async def _run_chat_sync(*, req: ChatRequest | CrossPaperQARequest, ctx: _CrossPaperContext,
                         user_id: str, idempotency_key: str | None) -> dict:
    """【聊天】【命令受理】参数为可信上下文和动作标识；共享幂等执行与同次结果提交。"""
    session_id = ensure_session_id(req.session_id)
    ctx = replace(ctx, session_id=session_id,
                  graph_extra={**ctx.graph_extra, "session_id": session_id, "thread_id": session_id})
    timeout = float(get_settings().graph_timeout_seconds)
    try:
        run, created = await rag_run_service.create_operation(
            user_id=user_id, conversation_id=ctx.conversation_id, session_id=session_id, thread_id=session_id,
            idempotency_key=idempotency_key if isinstance(idempotency_key, str) else str(uuid.uuid4()),
            query=req.query, deadline_at=rag_run_service.deadline_after(timeout), operation_kind="chat",
            request_payload=req.model_dump(mode="json"), metadata={"project_id": ctx.project_id,
                "paper_ids": ctx.scoped_ids, "explicit_paper_ids": req.paper_ids},
        )
    except rag_run_service.ActiveRAGRunConflict as exc:
        raise HTTPException(status_code=409, detail={"detail": str(exc), "run_id": exc.run["run_id"]}) from exc
    except rag_run_service.IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not created:
        return read_delivery_response(run)
    ctx = replace(ctx, run=run)
    try:
        return await get_rag_run_executor_registry().execute(
            user_id=user_id, run_id=run["run_id"], timeout_seconds=timeout,
            runner_factory=lambda: _execute_cross_paper_sync(req=req, ctx=ctx, user_id=user_id),
        )
    except Exception as exc:
        mapped = _map_cross_paper_http_error(exc)
        status_code = mapped.status_code if mapped else 500
        detail = mapped.detail if mapped else "cross-paper request failed"
        raise HTTPException(status_code=status_code, detail={"detail": detail, "run_id": run["run_id"],
            "run_status": "timed_out" if status_code == 504 else "failed"}) from exc


async def _execute_cross_paper_sync(*, req: ChatRequest | CrossPaperQARequest, ctx: _CrossPaperContext, user_id: str) -> dict:
    """参数为已取得执行权的上下文；图及来源占用结束后一次提交结果，保存错误如实返回。"""
    if ctx.run is None:
        raise RuntimeError("跨论文问答缺少操作身份")
    try:
        result, session_id = await invoke_graph(req.query, {**ctx.graph_extra, "run_id": ctx.run["run_id"]})
        response = _cross_paper_response(result, session_id, ctx)
        artifact_result = {**result, **_completed_run_metadata(result, response), "answer": response["answer"]}
        committed = await deliver_result(user_id=user_id, run=ctx.run, result=artifact_result, response=response)
        return committed["response"]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        mapped = _map_cross_paper_http_error(exc)
        if mapped is not None:
            raise mapped from exc
        raise


def _cross_paper_response(result: dict, session_id: str, ctx: _CrossPaperContext) -> dict:
    """参数为同次图结果和范围；返回同步响应，不改变原正文与引用坐标。"""
    if ctx.run is None:
        raise RuntimeError("聊天响应缺少已领取的运行身份")
    answer = result.get("answer")
    if answer is None and not result.get("final_delivery_report"):
        answer = result.get("report", "")
    if not isinstance(answer, str):
        raise ValueError("最终交付报告缺少同次正文")
    dialogue_mode, filter_expr = ctx.dialogue_mode, ctx.filter_expr
    _log_cross_paper_result(result, session_id, ctx)
    response = {
        "answer": answer,
        "chat_history": result.get("chat_history", []),
        "context_summary": result.get("context_summary", ""),
        "citations": result.get("citations", []),
        "paper_sources": result.get("paper_sources", []),
        "chunk_citations": result.get("chunk_citations", []),
        "graph_evidence": result.get("graph_evidence", []),
        "graph_hint_section": result.get("graph_hint_section", ""),
        "graph_hint_full": result.get("graph_hint_full", ""),
        "evidence_summary": result.get("evidence_summary", ""),
        "research_scope": result.get("research_scope", {}),
        "research_objects": result.get("research_objects", []),
        "note_actions": result.get("note_actions", []),
        "synthesis_state": result.get("synthesis_state", {}),
        "dialogue_mode": result.get("dialogue_mode") or dialogue_mode,
        "confidence": result.get("answer_confidence"),
        "session_id": session_id,
        "filter_expr": filter_expr,
        "project_id": ctx.project_id,
        "request_id": result.get("request_id"),
        "trace_summary": result.get("trace_summary"),
        "runtime_chain": result.get("runtime_chain"),
        "source_status": result.get("source_status", {}),
        "research_plan": result.get("research_plan"),
        "research_unit_results": result.get("research_unit_results", []),
        "coverage_review": result.get("coverage_review"),
        "planner_trace": result.get("planner_trace"),
        "evidence_sufficiency_report": result.get("evidence_sufficiency_report"),
        "repair_query_plans": result.get("repair_query_plans", []),
        "planner_repair_trace": result.get("planner_repair_trace", {}),
        "planner_repair_merge_report": result.get("planner_repair_merge_report", {}),
        "final_answer_eligibility_report": result.get("final_answer_eligibility_report", {}),
        "planner_final_blocks": result.get("planner_final_blocks", []),
        "planner_final_validation_errors": result.get("planner_final_validation_errors", []),
        "final_delivery_report": result.get("final_delivery_report"),
        "final_citation_bindings": result.get("final_citation_bindings"),
        "final_generation_attempt": result.get("final_generation_attempt"),
        "final_delivery_applicability": result.get("final_delivery_applicability"),
        "source_status_reasons": result.get("source_status_reasons", {}),
        # 项目科研回合的轮级字段：本轮生效深度、深度变更轨迹与记忆候选写回终态
        "turn_outcome": result.get("turn_outcome"),
        "turn_depth": result.get("turn_depth"),
        "depth_transitions": result.get("depth_transitions"),
        "memory_writeback": result.get("memory_writeback"),
        # 记忆读取终态与流式 done 载荷同构，两条路径的回答卡才能显示同一份记忆使用情况
        "memory_read_result": result.get("memory_read_result"),
        "run_id": ctx.run["run_id"], "run_status": "completed",
    }
    # 规划失败时透传稳定 error_code / error_summary，供前端与流式终态同构消费
    if result.get("error_code"):
        response["error_code"] = result.get("error_code")
    if result.get("error_summary"):
        response["error_summary"] = result.get("error_summary")
    if get_settings().rag_expose_unified_contract:
        # 1. 【研究交付】【来源保存】显示投影不能裁剪保存原文，否则引用身份及最终重试凭据会失配
        response["evidence_records"] = result.get("evidence_records") or []
        response["answer_brief"] = result.get("answer_brief") or {}
        response["missing_evidence"] = result.get("missing_evidence") or []
    return response


def _log_cross_paper_result(result: dict, session_id: str, ctx: _CrossPaperContext) -> None:
    """参数为图结果与范围；记录实际取证数量和交付状态，无返回值。"""
    scoped_ids, dialogue_mode = ctx.scoped_ids, ctx.dialogue_mode
    result_count = max(
        len(result.get("citations", [])),
        len(result.get("chunk_citations", [])),
        len(result.get("graph_evidence", [])),
        len(result.get("evidence_records", [])),
        len((result.get("rag_result") or {}).get("sources", []))
        if isinstance(result.get("rag_result"), dict)
        else 0,
        len(result.get("paper_sources", [])),
    )
    outcome = _completed_qa_outcome(result)
    log_structured_event(
        logger,
        "cross_paper_qa_completed",
        request_id=result.get("request_id"),
        session_id=session_id,
        project_id=ctx.project_id,
        mode=result.get("retrieval_mode"),
        dialogue_mode=result.get("dialogue_mode") or dialogue_mode,
        synthesis_status=(result.get("synthesis_state") or {}).get("status")
        if isinstance(result.get("synthesis_state"), dict)
        else None,
        paper_focus_mode=((result.get("synthesis_state") or {}).get("paper_focus") or {}).get("mode")
        if isinstance((result.get("synthesis_state") or {}).get("paper_focus"), dict)
        else None,
        evidence_layers=(result.get("synthesis_state") or {}).get("evidence_layers")
        if isinstance(result.get("synthesis_state"), dict)
        else None,
        source_status=result.get("source_status"),
        retrieval_reasoning=result.get("retrieval_reasoning"),
        outcome=outcome,
        result_count=result_count,
        paper_count=len(scoped_ids),
        error_code=result.get("error_code"),
        error_summary=result.get("error_summary"),
    )

async def _finalize_cross_paper_stream(*, done_payload: dict, ctx: _CrossPaperContext,
                                       user_id: str, query: str, session_id: str) -> dict:
    """【研究交付】【流式提交】参数为真实最终事件；原样保存正文及引用，返回提交回执。"""
    if ctx.run is None:
        raise RuntimeError("流式跨论文研究缺少 durable run")
    final_payload = {**done_payload, "session_id": session_id, "run_id": ctx.run["run_id"]}
    artifact_result = {**done_payload, **_completed_run_metadata(done_payload, final_payload), "answer": done_payload.get("answer")}
    return await deliver_result(user_id=user_id, run=ctx.run, result=artifact_result, response=final_payload)


async def _execute_cross_paper_stream_run(*, req: CrossPaperQARequest, ctx: _CrossPaperContext, user_id: str):
    """【跨论文问答】【流式执行】参数为已领取的操作；排空图和读取占用后提交最终结果。"""
    from app.services.upload_read_service import publication_scope_lifetime

    if ctx.run is None:
        raise RuntimeError("流式跨论文研究缺少 durable run")
    trace_token = None
    try:
        async with publication_scope_lifetime(ctx.graph_extra.get("project_scope")):
            stream_kind, graph_stream, session_id = await open_graph_stream(
                req.query, {**ctx.graph_extra, "run_id": ctx.run["run_id"]},
            )
            if session_id != ctx.session_id:
                raise RuntimeError("graph 返回的 session_id 与 durable run 不一致")
            trace_token = begin_stream_trace_context(extra=ctx.graph_extra, session_id=session_id)
            await rag_run_service.append_event(user_id=user_id, run_id=ctx.run["run_id"], event_type="event",
                payload={"type": "event", "stage": "session", "session_id": session_id})
            done = await _collect_cross_paper_stream(graph_stream, stream_kind=stream_kind,
                session_id=session_id, user_id=user_id, run_id=ctx.run["run_id"])
        return await _finalize_cross_paper_stream(done_payload=done, ctx=ctx, user_id=user_id,
                                                  query=req.query, session_id=session_id)
    finally:
        if trace_token is not None:
            reset_trace_ctx(trace_token)


async def _collect_cross_paper_stream(graph_stream, *, stream_kind: str, session_id: str, user_id: str, run_id: str) -> dict:
    """参数为图流与操作身份；消费并关闭完整流后返回真实终态，不能提前放开执行权。"""
    done = None
    try:
        async for payload in iter_cross_paper_payloads(graph_stream, stream_kind=stream_kind, session_id=session_id):
            if payload.get("type") == "done":
                done = payload
            else:
                await rag_run_service.append_event(user_id=user_id, run_id=run_id,
                    event_type=str(payload.get("type") or "event"), payload=payload)
    finally:
        if callable(getattr(graph_stream, "aclose", None)):
            await graph_stream.aclose()
    if done is None:
        raise RuntimeError("cross-paper stream ended without a done payload")
    return done


@router.post("/qa/cross-paper/stream")
async def cross_paper_qa_stream(
    req: CrossPaperQARequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """
    跨论文问答流式入口。

    与同步 `/qa/cross-paper` 共享同一套范围构建逻辑，区别只在输出形态：
    这里以 SSE 实时下发 planner 各阶段进度与关键细节，最终答案由主图综合后整块下发，
    让用户在长链路执行期间看到"正在做什么"，而不是干等整条链跑完。

    Args:
        req: 跨论文问答请求体，必须提供 query，并可附带 project_id/paper_ids
        user: 当前登录用户上下文

    Returns:
        StreamingResponse: text/event-stream，逐帧下发阶段事件、增量答案与最终 done 载荷
    """
    current_user_id = get_user_id(user)
    # 身份校验与范围收敛失败应直接返回 HTTP 错误码，不能塞进 SSE 流里
    ctx = await _build_cross_paper_context(
        req=req,
        user_id=current_user_id,
    )
    session_id = ensure_session_id(req.session_id)
    request = req.model_copy(update={"session_id": session_id})
    ctx = replace(
        ctx,
        session_id=session_id,
        graph_extra={**ctx.graph_extra, "session_id": session_id, "thread_id": session_id},
    )
    timeout_seconds = float(get_settings().graph_timeout_seconds)
    try:
        run, created = await rag_run_service.create_operation(
            user_id=current_user_id,
            conversation_id=ctx.conversation_id,
            session_id=session_id,
            thread_id=session_id,
            idempotency_key=idempotency_key,
            query=req.query,
            deadline_at=rag_run_service.deadline_after(timeout_seconds),
            metadata={"project_id": ctx.project_id, "paper_ids": ctx.scoped_ids, "explicit_paper_ids": req.paper_ids},
            operation_kind="chat", request_payload=req.model_dump(mode="json"),
        )
    except rag_run_service.ActiveRAGRunConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "detail": str(exc),
                "run_id": exc.run.get("run_id"),
                "run_status": exc.run.get("status"),
            },
        ) from exc
    ctx = replace(ctx, run=run)
    if created:
        try:
            get_rag_run_executor_registry().start(
                user_id=current_user_id,
                run_id=str(run["run_id"]),
                timeout_seconds=timeout_seconds,
                runner_factory=lambda: _execute_cross_paper_stream_run(
                    req=request,
                    ctx=ctx,
                    user_id=current_user_id,
                ),
            )
        except RuntimeError as exc:
            error = "应用关闭期间未启动跨论文研究"
            await rag_run_service.finish_run(
                user_id=current_user_id,
                run_id=str(run["run_id"]),
                status=rag_run_service.STATUS_CANCELLED,
                error=error,
            )
            raise HTTPException(status_code=503, detail=error) from exc
    return build_run_event_response(
        user_id=current_user_id,
        run_id=str(run["run_id"]),
        after_sequence=0,
    )


@router.post("/sessions")
async def create_session(req: SessionCreateRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    """
    显式创建一个空研究会话。

    Args:
        req: 会话创建请求，当前只使用 topic 作为前端展示标题
        user: 当前登录用户上下文

    Returns:
        dict: 仅包含新创建的真实 session_id
    """
    session = await get_research_session_lifecycle().ensure_session(
        user_id=get_user_id(user),
        topic=(req.topic or "新对话").strip() or "新对话",
    )
    return {
        "session_id": session["session_id"],
        "conversation_id": session.get("conversation_id"),
        "thread_id": session["thread_id"],
        "status": session["status"],
        "allowed_actions": session.get("allowed_actions")
        or allowed_actions_for_status(session.get("status")),
    }


async def _bind_session_thread_or_409(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str | None,
) -> dict:
    """
    为 GET 恢复请求读取或一次性绑定 durable thread_id。

    Args:
        user_id: 当前登录用户 ID
        session_id: durable session ID
        conversation_id: owner 校验后的前端对话 ID

    Returns:
        dict: 已具备 thread_id 的 durable session
    """
    lifecycle = get_research_session_lifecycle()

    async def checkpoint_exists(thread_id: str) -> bool:
        """判断指定候选 thread 是否存在官方 checkpoint。"""
        snapshot = await get_shared_agent_state(
            session_id,
            conversation_id,
            thread_id=thread_id,
        )
        return snapshot is not None

    try:
        return await lifecycle.bind_legacy_thread(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            checkpoint_exists=checkpoint_exists,
        )
    except ResearchSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LegacyResearchSessionNotRecoverable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _session_snapshot_with_metadata(snapshot: dict, session_row: dict) -> dict:
    """【研究交付】【会话回读】参数为 checkpoint 与会话；新版答案仅返回已提交整体投影。"""
    values = dict(snapshot.get("values") or {})
    committed = session_row.get("current_result")
    if session_row.get("current_run_id") or committed is not None:
        # 1. 【研究交付】【保存边界】checkpoint 中尚未提交的正文和引用不能冒充当前答案
        for key in (*DELIVERY_METADATA_FIELDS, "answer"):
            values.pop(key, None)
        values.update(committed or {})
    return {
        **snapshot,
        "values": values,
        "current_result": committed,
        "current_run_id": session_row.get("current_run_id"),
        "result_run_id": session_row.get("result_run_id"),
        "delivery_revision": session_row.get("delivery_revision"),
        "project_id": session_row.get("project_id"),
        "paper_ids": session_row.get("paper_ids") or [],
        "research_mode": session_row.get("research_mode"),
        "status": session_row.get("status"),
        "allowed_actions": session_row.get("allowed_actions")
        or allowed_actions_for_status(session_row.get("status")),
        "last_error": session_row.get("last_error"),
        "attempt_count": int(session_row.get("attempt_count") or 0),
    }


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    conversation_id: str | None = Query(default=None),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """返回官方 checkpoint 快照，并附带 durable session scope/status。"""
    current_user_id = get_user_id(user)
    try:
        await get_research_session_lifecycle().get_session(
            user_id=current_user_id,
            session_id=session_id,
        )
    except ResearchSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    resolved_conversation_id = await _resolve_identity_or_404(
        current_user_id,
        conversation_id,
        session_id,
        allow_missing_conversation=True,
    )
    session_row = await _bind_session_thread_or_409(
        user_id=current_user_id,
        session_id=session_id,
        conversation_id=resolved_conversation_id,
    )
    snapshot = await get_shared_agent_state(
        session_id,
        resolved_conversation_id,
        thread_id=session_row["thread_id"],
    )
    # 1. 【研究交付】【当前结果】同一次查询读取目标完整版本和实际执行停止事实
    if session_row.get("result_run_id"):
        from app.services.result_delivery_service import read_current_delivery_response

        result_run = await rag_run_service.get_run(user_id=current_user_id, run_id=session_row["result_run_id"])
        if result_run is None:
            raise HTTPException(status_code=409, detail="会话已提交结果不存在")
        session_row = {**session_row, "current_result": await read_current_delivery_response(result_run),
                       "delivery_revision": result_run["delivery_revision"]}

    if snapshot:
        return _session_snapshot_with_metadata(snapshot, session_row)
    payload = _empty_session_payload(session_id, session_row)
    if resolved_conversation_id:
        payload["conversation_id"] = resolved_conversation_id
        payload["values"]["conversation_id"] = resolved_conversation_id
    return _session_snapshot_with_metadata(payload, session_row)
