"""
研究、搜索与报告类路由。

这个文件管理的是“研究任务型”的 API，而不是普通聊天。

当前产品语义里：
- `/research` 是外部研究入口
- `/search/arxiv` 是外部论文搜索辅助入口

这些入口和 `chat_api.py` 的共同点是都复用了主图；
不同点在于这里通常会显式指定 intent，从而绕过主图的自动意图识别。
"""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.agents.agent import INTENT_DEEP_RESEARCH, INTENT_INTERNAL_RESEARCH
from app.api.facades.graph_facade import (
    get_shared_agent_state,
    invoke_graph,
    research_response_payload,
    snapshot_has_pending_interrupt,
)
from app.api.facades.research_contract_facade import (
    EXTERNAL_RESEARCH_MODE,
    build_project_scope_contract,
    build_research_structured_query,
    resolve_core_research_intent,
)
from app.api.facades.session_identity_facade import resolve_conversation_identity
from app.api.schemas.research import ResearchRequest, ResearchResumeRequest
from app.config import get_settings
from app.repositories.rag_run_repository import get_rag_run_repository
from app.services import rag_run_service
from app.services.rag_run_executor_service import get_rag_run_executor_registry
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.models.research_contracts import RetrievalScopeError, ScopeOwnershipError
from app.nodes.arxiv_search import run_arxiv_search
from app.nodes.guardrails_node import guardrails_node
from app.services.llm_service import classify_llm_upstream_error
from app.services.project_scope_service import build_project_context
from app.services.session_mapping_service import ensure_session_id
from app.services.research_session_lifecycle import (
    ACTION_RESUME,
    ACTION_RETRY,
    InvalidResearchSessionTransition,
    LegacyResearchSessionNotRecoverable,
    ResearchSessionAssociationError,
    ResearchSessionLifecycle,
    ResearchSessionNotFound,
    ResearchSessionScopeConflict,
    allowed_actions_for_status,
    get_research_session_lifecycle,
    persist_research_session_links,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_session_state(raw: object) -> dict:
    """把 sessions 表里存储的 state_json 安全反序列化成 dict。"""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _legacy_research_mode(row: dict, state: dict) -> str:
    """
    从 durable 字段或旧 structured_query 读取研究模式。

    Args:
        row: sessions 表返回的 durable 行
        state: 已反序列化的旧研究状态

    Returns:
        str: 归一化后的研究模式；无法确认时返回空字符串
    """
    structured_query = state.get("structured_query")
    structured_mode = (
        structured_query.get("research_mode") if isinstance(structured_query, dict) else ""
    )
    return str(row.get("research_mode") or state.get("research_mode") or structured_mode or "").strip()


def _research_session_summary(row: dict) -> dict | None:
    """
    从 session 行中提取“研究任务摘要”。

    Args:
        row: sessions 表返回的 durable 行

    Returns:
        dict | None: deep research 摘要；普通聊天 session 返回 None
    """
    state = row.get("current_result") or _parse_session_state(row.get("state_json"))
    topic = str(row.get("topic") or "").strip()
    report = str(state.get("report") or "").strip()
    clarification = str(state.get("clarification_question") or "").strip()
    intent = str(state.get("intent") or "").strip()
    # 研究判定只认两条真有数据的依据：/research 入口写 topic 时强制加的「深度研究」
    # 前缀，以及 /research 与 /research/stream 经 ensure_session 落库的 research_mode 列
    # （值 external）。state_json 里的 intent / research_brief 等字段生产上无写入方
    # （save_checkpoint 无生产调用点），不能当判定依据
    research_mode = _legacy_research_mode(row, state)
    looks_like_research = bool(
        research_mode == EXTERNAL_RESEARCH_MODE
        or topic.startswith("深度研究")
        or topic.lower().startswith("research")
    )
    if not looks_like_research:
        return None
    status = str(row.get("status") or "").strip()
    if not status:
        return None
    preview = clarification or report
    counts = _research_state_counts(state)
    return {
        "session_id": row.get("session_id", ""),
        "conversation_id": row.get("conversation_id"),
        "project_id": row.get("project_id"),
        "thread_id": row.get("thread_id"),
        "current_run_id": row.get("current_run_id"),
        "result_run_id": row.get("result_run_id"),
        "delivery_revision": row.get("delivery_revision", 0),
        "topic": topic or "未命名研究任务",
        "status": status,
        "allowed_actions": row.get("allowed_actions") or allowed_actions_for_status(status),
        "last_error": row.get("last_error"),
        "attempt_count": int(row.get("attempt_count") or 0),
        "updated_at": row.get("updated_at"),
        "created_at": row.get("created_at"),
        "step_name": row.get("step_name") or "",
        "paper_ids": row.get("paper_ids") or [],
        "preview": preview[:280],
        "intent": intent or INTENT_DEEP_RESEARCH,
        "research_mode": research_mode or EXTERNAL_RESEARCH_MODE,
        **counts,
    }


def _state_list(state: dict, key: str) -> list[object]:
    """
    从旧研究 state 中读取列表字段，无效类型按空列表处理。

    Args:
        state: 已反序列化的研究状态
        key: 要读取的状态字段名

    Returns:
        list[object]: 可安全统计和切片的列表值
    """
    value = state.get(key)
    return value if isinstance(value, list) else []


def _research_state_counts(state: dict) -> dict:
    """提取研究列表需要的轻量统计，避免把完整 state 暴露给前端。"""
    evidence_cards = _state_list(state, "evidence_cards")
    idea_candidates = _state_list(state, "idea_candidates")
    key_findings = _state_list(state, "key_findings")
    candidate_papers = _state_list(state, "candidate_papers_to_import")
    external_gaps = _state_list(state, "external_gaps")
    follow_up_questions = _state_list(state, "follow_up_questions")
    return {
        "evidence_count": len(evidence_cards),
        "idea_count": len(idea_candidates),
        "candidate_paper_count": len(candidate_papers),
        "external_gap_count": len(external_gaps),
        "follow_up_count": len(follow_up_questions),
        "key_findings": key_findings[:3],
    }
async def _resolve_identity_or_404(
    user_id: str,
    conversation_id: str | None,
    session_id: str | None,
) -> str | None:
    try:
        identity = await resolve_conversation_identity(
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return identity.conversation_id


def _research_topic(query: str) -> str:
    """从研究问题生成简短 session 标题。"""
    normalized = str(query or "").strip()
    return f"深度研究：{normalized[:80]}" if normalized else "深度研究"


async def _checkpoint_exists(
    session_id: str,
    conversation_id: str | None,
    thread_id: str,
) -> bool:
    """
    判断 legacy thread 候选是否已经存在官方 LangGraph checkpoint。

    Args:
        session_id: 对外 durable session ID
        conversation_id: 已完成 owner 校验的前端对话 ID
        thread_id: 待核验的 checkpoint key

    Returns:
        bool: 候选 thread 已存在有效 snapshot 时为 True
    """
    snapshot = await get_shared_agent_state(
        session_id,
        conversation_id,
        thread_id=thread_id,
    )
    return snapshot is not None


async def _prepare_research_session(
    *,
    lifecycle: ResearchSessionLifecycle,
    user_id: str,
    query: str,
    session_id: str | None,
    conversation_id: str | None,
    project_context: dict,
    research_mode: str,
) -> dict:
    """
    在 graph 前创建/绑定 durable session，并迁移到 running。

    Args:
        lifecycle: 研究 session 生命周期服务
        user_id: 当前登录用户 ID
        query: 研究问题
        session_id: 可选已有 session ID
        conversation_id: 可选前端对话 ID
        project_context: owner 校验后的项目与论文范围
        research_mode: internal 或 external

    Returns:
        dict: 已持久化且状态为 running 的 session
    """
    try:
        return await lifecycle.prepare_command_session(
            user_id=user_id,
            topic=_research_topic(query),
            session_id=session_id,
            conversation_id=conversation_id,
            project_id=project_context.get("project_id"),
            paper_ids=project_context.get("paper_ids"),
            research_mode=research_mode,
            checkpoint_exists=lambda candidate: _checkpoint_exists(
                str(session_id or candidate),
                conversation_id,
                candidate,
            ),
            scope_linker=partial(persist_research_session_links, user_id=user_id),
        )
    except ResearchSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        InvalidResearchSessionTransition,
        LegacyResearchSessionNotRecoverable,
        ResearchSessionScopeConflict,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ResearchSessionAssociationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _record_operation_failure(status: str, error: str, *, user_id: str, run_id: str) -> None:
    """参数为已退出的操作和失败事实；同库提交 session 与运行终态，无返回值。"""
    await get_rag_run_repository().settle_session_operation(
        user_id=user_id, run_id=run_id, status=status, error=error,
    )


def _raise_graph_error(exc: Exception) -> NoReturn:
    """参数为受管执行已收尾的异常；保留既有 HTTP 与上游错误映射。"""
    if isinstance(exc, HTTPException):
        raise exc
    upstream = classify_llm_upstream_error(exc)
    if upstream is not None:
        status_code, reason = upstream
        raise HTTPException(status_code=status_code, detail=reason) from exc
    if isinstance(exc, TimeoutError):
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=f"deep research failed: {exc}") from exc


async def _existing_operation(*, user_id: str, idempotency_key: str,
                              operation_kind: str, request_payload: dict) -> dict | None:
    """参数为重发命令；返回同一操作，异输入明确冲突，不在重放时再次执行图。"""
    run = await rag_run_service.get_by_idempotency_key(user_id=user_id, idempotency_key=idempotency_key)
    if run is None:
        return None
    try:
        rag_run_service.assert_same_operation(run, operation_kind=operation_kind, request_payload=request_payload)
    except rag_run_service.IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return run


def _operation_response(run: dict) -> dict:
    """参数为持久运行；整体读取已保存响应并附当前操作事实，不拼接不同交付版本。"""
    from app.services.result_delivery_service import read_delivery_response

    return {**read_delivery_response(run), "session_id": run.get("session_id"),
            "cancel_requested": run.get("cancel_requested", False), "operation_replayed": True}


def _research_graph_extra(
    req: ResearchRequest,
    *,
    user_id: str,
    conversation_id: str | None,
    project_context: dict,
    core_intent: str,
    session: dict,
) -> dict:
    """
    组装已绑定 durable identity 的深度研究 graph 输入。

    Args:
        req: 研究请求
        user_id: 当前登录用户 ID，供 trace 归属使用
        conversation_id: owner 校验后的前端对话 ID
        project_context: owner 校验后的项目范围
        core_intent: internal_research 或 external_research
        session: 已进入 running 的 durable session

    Returns:
        dict: graph facade 可直接消费的 extra
    """
    return {
        "intent": INTENT_DEEP_RESEARCH,
        "core_intent": core_intent,
        "user_id": user_id,
        "session_id": session["session_id"],
        "conversation_id": conversation_id,
        "thread_id": session["thread_id"],
        "filter_expr": project_context["filter_expr"],
        "project_scope": build_project_scope_contract(project_context),
        "structured_query": build_research_structured_query(
            project_context,
            project_id=project_context.get("project_id"),
            query=req.query,
            paper_ids=req.paper_ids,
            dialogue_mode=req.dialogue_mode or "research",
        ),
        "dialogue_mode": req.dialogue_mode or "research",
    }


async def _invoke_durable_research(
    req: ResearchRequest, *, lifecycle: ResearchSessionLifecycle, user_id: str,
    session: dict, graph_extra: dict, run: dict, execution_kind: str = "new",
    clarification_result: dict | None = None,
) -> dict:
    """参数为已领取操作与图输入；受管执行后提交完整产物，返回可区分保存状态的响应。"""
    factory = partial(_generate_and_deliver_research, req=req, lifecycle=lifecycle,
                      user_id=user_id, session=session, graph_extra=graph_extra, run=run,
                      execution_kind=execution_kind, clarification_result=clarification_result)
    try:
        return await get_rag_run_executor_registry().execute(
            user_id=user_id, run_id=run["run_id"],
            timeout_seconds=float(get_settings().deep_research_graph_timeout_seconds),
            runner_factory=factory,
            terminal_callback=partial(_record_operation_failure, user_id=user_id, run_id=run["run_id"]),
        )
    except Exception as exc:
        _raise_graph_error(exc)


async def _generate_and_deliver_research(
    *, req: ResearchRequest, lifecycle: ResearchSessionLifecycle, user_id: str,
    session: dict, graph_extra: dict, run: dict, execution_kind: str,
    clarification_result: dict | None = None,
) -> dict:
    """参数为当前操作与可选真实澄清快照；图及读取占用退出后一次暂存/提交整个结果。"""
    from app.services.result_delivery_service import deliver_result

    session_id = str(session["session_id"])
    if clarification_result is None:
        result, resolved_session_id = await invoke_graph(
            req.query, graph_extra, execution_kind=execution_kind,
        )
        if resolved_session_id != session_id:
            raise RuntimeError("graph 返回的 session_id 与 durable session 不一致")
    else:
        # 1. 【研究命令】【恢复澄清】重试不能把原问题当答案；仅恢复已保存的澄清问题
        result = clarification_result
    status = "awaiting_clarification" if result.get("awaiting_clarification") else "completed"
    response = _research_result_payload(result, {**session, "status": status,
        "allowed_actions": allowed_actions_for_status(status)})
    delivery = await deliver_result(user_id=user_id, run=run, result=result,
                                     response=response, session_status=status)
    return {**delivery["response"], "run_id": run["run_id"],
            "run_status": delivery["run"]["status"],
            **{key: delivery.get(key) for key in ("delivery_revision", "persistence_status",
                                                "persistence_error", "can_retry_save")}}


def _research_result_payload(result: dict, durable: dict) -> dict:
    """把 graph 结果与 durable session 命令能力合并为统一研究响应。"""
    payload = research_response_payload(
        result,
        str(durable["session_id"]),
        durable.get("conversation_id"),
    )
    payload.update(
        {
            "conversation_id": durable.get("conversation_id"),
            "thread_id": durable.get("thread_id"),
            "status": durable.get("status"),
            "allowed_actions": durable.get("allowed_actions")
            or allowed_actions_for_status(durable.get("status")),
            "attempt_count": int(durable.get("attempt_count") or 0),
        }
    )
    return payload


async def _run_research_request(
    *,
    req: ResearchRequest,
    user_id: str,
    conversation_id: str | None,
    project_context: dict,
    core_intent: str,
    lifecycle: ResearchSessionLifecycle,
    idempotency_key: str,
) -> dict:
    """创建 durable session、运行 graph 并组装研究响应。"""
    research_mode = "internal" if core_intent == INTENT_INTERNAL_RESEARCH else "external"
    session = await _prepare_research_session(
        lifecycle=lifecycle,
        user_id=user_id,
        query=req.query,
        session_id=req.session_id,
        conversation_id=conversation_id,
        project_context=project_context,
        research_mode=research_mode,
    )
    run, session, created = await _accept_research_operation(
        lifecycle=lifecycle, user_id=user_id, session=session, operation_kind="research",
        idempotency_key=idempotency_key, query=req.query, request_payload=req.model_dump(mode="json"),
    )
    if not created:
        return _operation_response(run)
    return await _invoke_durable_research(
        req,
        lifecycle=lifecycle,
        user_id=user_id,
        session=session,
        run=run,
        graph_extra=_research_graph_extra(
            req,
            user_id=user_id,
            conversation_id=conversation_id,
            project_context=project_context,
            core_intent=core_intent,
            session=session,
        ),
    )


async def _session_for_action(
    *,
    lifecycle: ResearchSessionLifecycle,
    user_id: str,
    session_id: str,
    action: str,
    idempotency_key: str,
    request_payload: dict,
) -> tuple[dict, dict | None]:
    """参数为 owner、session 与命令身份；返回会话和并发重放，或明确拒绝非法动作。"""
    try:
        session = await lifecycle.get_session(user_id=user_id, session_id=session_id)
    except ResearchSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    actions = session.get("allowed_actions") or allowed_actions_for_status(session.get("status"))
    if action not in actions:
        # 1. 【研究命令】【并发重放】首次 key 查询后可能已有相同命令领取，先回查再裁定状态冲突
        existing = await _existing_operation(user_id=user_id, idempotency_key=idempotency_key,
                                             operation_kind=action, request_payload=request_payload)
        if existing is not None:
            return session, existing
        raise HTTPException(status_code=409, detail={"detail": f"当前研究会话不允许执行 {action}",
            "run_id": session.get("current_run_id"), "run_status": session.get("status")})
    return session, None


async def _project_context_or_http(
    *,
    project_id: str | None,
    paper_ids: list[str],
    user_id: str,
) -> dict:
    """构造服务端可信研究范围，并统一映射 owner/scope 错误。"""
    try:
        return await build_project_context(
            project_id,
            requested_paper_ids=paper_ids,
            user_id=user_id,
        )
    except ScopeOwnershipError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _effective_scope_or_http(
    *,
    lifecycle: ResearchSessionLifecycle,
    user_id: str,
    session_id: str | None,
    project_id: str | None,
    paper_ids: list[str],
) -> dict:
    """空请求值回读 durable scope，并把身份或范围冲突映射为 HTTP。"""
    try:
        return await lifecycle.resolve_scope_request(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            paper_ids=paper_ids,
        )
    except ResearchSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ResearchSessionScopeConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def _command_context(session: dict, *, user_id: str) -> tuple[dict, str]:
    """只用 durable session 的 project/paper scope 构造命令执行上下文。"""
    context = await _project_context_or_http(
        project_id=session.get("project_id"),
        paper_ids=list(session.get("paper_ids") or []),
        user_id=user_id,
    )
    core_intent = resolve_core_research_intent(
        context.get("project_id"),
        paper_ids=context.get("paper_ids"),
    )
    if core_intent == INTENT_INTERNAL_RESEARCH and not context["local_retrieval_allowed"]:
        raise HTTPException(status_code=422, detail=str(RetrievalScopeError()))
    return context, core_intent


async def _command_snapshot(session: dict) -> dict | None:
    """按 durable session/thread identity 读取命令所需 checkpoint。"""
    return await get_shared_agent_state(
        str(session["session_id"]),
        session.get("conversation_id"),
        thread_id=session.get("thread_id"),
    )


def _retry_query(snapshot: dict | None) -> str:
    """从 checkpoint 读取原始研究问题；缺失时拒绝伪造重试输入。"""
    values = snapshot.get("values") if isinstance(snapshot, dict) else None
    query = str(values.get("user_query") or "").strip() if isinstance(values, dict) else ""
    if not query:
        raise HTTPException(status_code=409, detail="研究 checkpoint 缺少原始问题，无法重试")
    return query


async def _accept_research_operation(*, lifecycle: ResearchSessionLifecycle, user_id: str,
                                     session: dict, operation_kind: str, idempotency_key: str,
                                     query: str, request_payload: dict,
                                     clarification_pending: bool = False) -> tuple[dict, dict, bool]:
    """参数为命令及范围；返回仓储原子领取结果，并把冲突映射为稳定 HTTP。"""
    try:
        return await lifecycle.accept_operation(
            user_id=user_id, session=session, operation_kind=operation_kind,
            idempotency_key=idempotency_key, query=query, request_payload=request_payload,
            deadline_at=rag_run_service.deadline_after(float(get_settings().deep_research_graph_timeout_seconds)),
            clarification_pending=clarification_pending,
        )
    except rag_run_service.ActiveRAGRunConflict as exc:
        raise HTTPException(status_code=409, detail={"detail": str(exc),
            "run_id": exc.run["run_id"], "run_status": exc.run["status"]}) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def _execute_research_command(
    *, query: str, session: dict, project_context: dict, core_intent: str,
    lifecycle: ResearchSessionLifecycle, user_id: str, idempotency_key: str,
    operation_kind: str, request_payload: dict, clarification_result: dict | None = None,
) -> dict:
    """参数为澄清或重试命令；仅新领取者可以进入受管图，重放直接查回原操作。"""
    await persist_research_session_links(session, user_id=user_id)
    run, session, created = await _accept_research_operation(
        lifecycle=lifecycle, user_id=user_id, session=session, operation_kind=operation_kind,
        idempotency_key=idempotency_key, query=query, request_payload=request_payload,
        clarification_pending=clarification_result is not None,
    )
    if not created:
        return _operation_response(run)
    request = ResearchRequest(query=query, session_id=session["session_id"],
        conversation_id=session.get("conversation_id"), project_id=session.get("project_id"),
        paper_ids=list(session.get("paper_ids") or []), dialogue_mode="research")
    return await _invoke_durable_research(
        request, lifecycle=lifecycle, user_id=user_id, session=session, run=run,
        execution_kind=operation_kind, clarification_result=clarification_result,
        graph_extra=_research_graph_extra(request, user_id=user_id,
            conversation_id=session.get("conversation_id"), project_context=project_context,
            core_intent=core_intent, session=session),
    )


def _raise_command_transition(exc: Exception) -> NoReturn:
    """把生命周期命令错误转换为稳定 HTTP 语义。"""
    if isinstance(exc, ResearchSessionNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ResearchSessionAssociationError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/research/sessions")
async def list_research_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """列出当前用户的研究 session 摘要。"""
    rows = await get_research_session_lifecycle().list_sessions(
        user_id=get_user_id(user),
        limit=limit,
        offset=offset,
    )
    sessions = [item for item in (_research_session_summary(row) for row in rows) if item]
    return {"sessions": sessions}


@router.post("/research/sessions/{session_id}/resume")
async def resume_research_session(
    session_id: str, req: ResearchResumeRequest,
    user: dict = Depends(get_current_user_with_rate_limit),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
):
    """参数为 session、用户澄清及请求标识；只消费实际 interrupt，返回同一命令结果。"""
    user_id = get_user_id(user)
    payload = {"session_id": session_id, "answer": req.answer}
    existing = await _existing_operation(user_id=user_id, idempotency_key=idempotency_key,
                                         operation_kind="resume", request_payload=payload)
    if existing is not None:
        return _operation_response(existing)
    lifecycle = get_research_session_lifecycle()
    session, existing = await _session_for_action(lifecycle=lifecycle, user_id=user_id,
        session_id=session_id, action=ACTION_RESUME, idempotency_key=idempotency_key, request_payload=payload)
    if existing is not None:
        return _operation_response(existing)
    if not snapshot_has_pending_interrupt(await _command_snapshot(session)):
        raise HTTPException(status_code=409, detail="研究 checkpoint 不在等待澄清状态")
    context, core_intent = await _command_context(session, user_id=user_id)
    return await _execute_research_command(
        query=req.answer, session=session, project_context=context, core_intent=core_intent,
        lifecycle=lifecycle, user_id=user_id, idempotency_key=idempotency_key,
        operation_kind="resume", request_payload=payload,
    )


@router.post("/research/sessions/{session_id}/retry")
async def retry_research_session(
    session_id: str, user: dict = Depends(get_current_user_with_rate_limit),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
):
    """参数为 session 与请求标识；重试保留原目标，未回答的澄清只恢复问题而不调用图。"""
    user_id = get_user_id(user)
    payload = {"session_id": session_id}
    existing = await _existing_operation(user_id=user_id, idempotency_key=idempotency_key,
                                         operation_kind="retry", request_payload=payload)
    if existing is not None:
        return _operation_response(existing)
    lifecycle = get_research_session_lifecycle()
    session, existing = await _session_for_action(lifecycle=lifecycle, user_id=user_id,
        session_id=session_id, action=ACTION_RETRY, idempotency_key=idempotency_key, request_payload=payload)
    if existing is not None:
        return _operation_response(existing)
    snapshot = await _command_snapshot(session)
    query = _retry_query(snapshot)
    clarification = None
    if snapshot is not None and snapshot_has_pending_interrupt(snapshot):
        values = snapshot["values"]
        clarification = {**values, "awaiting_clarification": True}
    context, core_intent = await _command_context(session, user_id=user_id)
    return await _execute_research_command(
        query=query, session=session, project_context=context, core_intent=core_intent,
        lifecycle=lifecycle, user_id=user_id, idempotency_key=idempotency_key,
        operation_kind="retry", request_payload=payload, clarification_result=clarification,
    )


@router.post("/research/sessions/{session_id}/cancel")
async def cancel_research_session(
    session_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """参数为 session 与 owner；等待本地实际停止，跨进程返回正在停止事实。"""
    user_id = get_user_id(user)
    lifecycle = get_research_session_lifecycle()
    try:
        return await lifecycle.cancel(user_id=user_id, session_id=session_id)
    except (ResearchSessionNotFound, InvalidResearchSessionTransition) as exc:
        _raise_command_transition(exc)


@router.post("/research")
async def research(req: ResearchRequest, user: dict = Depends(get_current_user_with_rate_limit),
                   idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128)):
    """
    深度研究主入口。

    这条路由会显式把 `intent` 固定为 `deep_research`，
    并把 project scope 透传给主图，确保研究任务在正确论文范围内运行。
    """
    current_user_id = get_user_id(user)
    existing = await _existing_operation(user_id=current_user_id, idempotency_key=idempotency_key,
        operation_kind="research", request_payload=req.model_dump(mode="json"))
    if existing is not None:
        return _operation_response(existing)
    conversation_id = await _resolve_identity_or_404(current_user_id, req.conversation_id, req.session_id)
    lifecycle = get_research_session_lifecycle()
    effective_scope = await _effective_scope_or_http(
        lifecycle=lifecycle,
        user_id=current_user_id,
        session_id=req.session_id,
        project_id=req.project_id,
        paper_ids=req.paper_ids,
    )
    project_context = await _project_context_or_http(
        project_id=effective_scope.get("project_id"),
        paper_ids=list(effective_scope.get("paper_ids") or []),
        user_id=current_user_id,
    )
    core_intent = resolve_core_research_intent(
        project_context["project_id"],
        paper_ids=project_context["paper_ids"],
    )
    if core_intent == INTENT_INTERNAL_RESEARCH and not project_context["local_retrieval_allowed"]:
        raise HTTPException(status_code=422, detail=str(RetrievalScopeError()))
    return await _run_research_request(
        req=req,
        user_id=current_user_id,
        conversation_id=conversation_id,
        project_context=project_context,
        core_intent=core_intent,
        lifecycle=lifecycle,
        idempotency_key=idempotency_key,
    )


@router.post("/search/arxiv")
async def search_arxiv(req: ResearchRequest, user: dict = Depends(get_current_user_with_rate_limit)):
    """
    触发 arXiv 论文检索能力。

    主链收敛后 arXiv 检索已移出顶层图，这里直接调用能力入口（搜索 → 过滤 → 报告），
    返回体结构与收敛前保持一致。conversation_id 仍先做 owner 校验，不存在时返回 404。
    离开主图后不再有顶层入口把关，所以这里自己跑一遍安全底线（空输入 / 超长 / 敏感词），
    命中时与主图一样返回拒绝报告，而不是继续去查 arXiv。
    """
    current_user_id = get_user_id(user)
    conversation_id = await _resolve_identity_or_404(current_user_id, req.conversation_id, req.session_id)
    session_id = ensure_session_id(req.session_id, conversation_id)
    # 1. 先过安全底线，拒绝时不发起任何 arXiv 检索
    guardrail_result = await guardrails_node({"user_query": req.query})
    if guardrail_result.get("guardrails_action") == "reject":
        logger.info(
            "【arXiv检索】【安全护栏】拒绝查询，原因: %s",
            guardrail_result.get("guardrails_reason"),
        )
        return {
            "report": guardrail_result.get("report", ""),
            "papers": [],
            "session_id": session_id,
        }
    # 2. 放行后执行检索能力，返回体结构与收敛前一致
    result = await run_arxiv_search(
        {
            "user_query": req.query,
            "session_id": session_id,
            "conversation_id": conversation_id,
        }
    )
    return {
        "report": result.get("report", ""),
        "papers": result.get("filtered_papers") or result.get("papers", []),
        "session_id": session_id,
    }
