"""
SSE 流式研究路由。

这是 graph/research 链路的流式 HTTP 入口。它不直接生成 token，
而是负责把 graph facade 暴露出的事件/值流封装成标准 SSE 响应。
"""

from __future__ import annotations

from functools import partial

from fastapi import APIRouter, Depends, Header, HTTPException

from app.agents.agent import INTENT_DEEP_RESEARCH, INTENT_INTERNAL_RESEARCH
from app.api.facades.graph_facade import (
    get_shared_agent_state,
    iter_research_payloads,
    open_graph_stream,
)
from app.api.facades.rag_run_stream_facade import build_run_event_response
from app.api.facades.research_contract_facade import (
    build_project_scope_contract,
    build_research_structured_query,
    resolve_core_research_intent,
)
from app.api.facades.session_identity_facade import resolve_conversation_identity
from app.api.schemas.research import ResearchStreamRequest
from app.config import get_settings
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.models.research_contracts import RetrievalScopeError, ScopeOwnershipError
from app.services.project_scope_service import build_project_context
from app.services import rag_run_service
from app.services.rag_run_executor_service import get_rag_run_executor_registry
from app.services.research_session_lifecycle import (
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
from app.api.routes.research_api import _accept_research_operation, _record_operation_failure

router = APIRouter()


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


async def _stream_checkpoint_exists(
    session_id: str,
    conversation_id: str | None,
    thread_id: str,
) -> bool:
    """
    判断流式研究 legacy thread 候选是否存在 checkpoint。

    Args:
        session_id: durable session ID
        conversation_id: owner 校验后的前端对话 ID
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


async def _prepare_stream_session(
    *,
    lifecycle: ResearchSessionLifecycle,
    req: ResearchStreamRequest,
    user_id: str,
    conversation_id: str | None,
    project_context: dict,
    research_mode: str,
) -> dict:
    """
    在打开惰性 graph stream 前创建 durable session 并写 running。

    Args:
        lifecycle: 研究 session 生命周期服务
        req: 流式研究请求
        user_id: 当前登录用户 ID
        conversation_id: owner 校验后的前端对话 ID
        project_context: owner 校验后的项目范围
        research_mode: internal 或 external

    Returns:
        dict: 已进入 running 的 durable session
    """
    topic = f"深度研究：{req.query.strip()[:80]}" if req.query.strip() else "深度研究"
    try:
        return await lifecycle.prepare_command_session(
            user_id=user_id,
            topic=topic,
            session_id=req.session_id,
            conversation_id=conversation_id,
            project_id=project_context.get("project_id"),
            paper_ids=project_context.get("paper_ids"),
            research_mode=research_mode,
            checkpoint_exists=lambda candidate: _stream_checkpoint_exists(
                str(req.session_id or candidate),
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


def _stream_done_with_status(payload: dict, durable: dict) -> dict:
    """
    把 durable 终态与允许命令合并到流式 done payload。

    Args:
        payload: graph 产生的结构化 done 数据
        durable: 已完成终态写入的 session 行

    Returns:
        dict: 可直接编码为 SSE 的最终数据
    """
    return {
        **payload,
        "status": durable.get("status"),
        "allowed_actions": durable.get("allowed_actions")
        or allowed_actions_for_status(durable.get("status")),
        "attempt_count": int(durable.get("attempt_count") or 0),
    }


async def _execute_durable_stream(
    *, req: ResearchStreamRequest, conversation_id: str | None, project_context: dict,
    core_intent: str, session: dict, lifecycle: ResearchSessionLifecycle, user_id: str, run: dict,
) -> None:
    """参数为已领取后台流；图和读取占用退出后提交同次正文、引用与 session 终态。"""
    from app.services.upload_read_service import publication_scope_lifetime
    from app.services.result_delivery_service import deliver_result

    async with publication_scope_lifetime(project_context):
        payload = await _consume_durable_stream(req=req, conversation_id=conversation_id,
            project_context=project_context, core_intent=core_intent, session=session,
            lifecycle=lifecycle, user_id=user_id, run=run)
    status = "awaiting_clarification" if payload.get("awaiting_clarification") else "completed"
    response = _stream_done_with_status(payload, {**session, "status": status,
        "allowed_actions": allowed_actions_for_status(status)})
    await deliver_result(user_id=user_id, run=run, result=payload,
                         response=response, session_status=status)


async def _consume_durable_stream(
    *,
    req: ResearchStreamRequest,
    conversation_id: str | None,
    project_context: dict,
    core_intent: str,
    session: dict,
    lifecycle: ResearchSessionLifecycle,
    user_id: str,
    run: dict,
) -> dict:
    """执行深度研究 graph，并把中间事件和最终结果写入 durable run。"""
    session_id = str(session["session_id"])
    run_id = str(run["run_id"])
    graph_extra = _stream_graph_extra(
        req,
        user_id=user_id,
        conversation_id=conversation_id,
        project_context=project_context,
        core_intent=core_intent,
        session=session,
    )
    stream_kind, graph_stream, resolved_session_id = await open_graph_stream(req.query, graph_extra, execution_kind="new")
    done = None
    try:
        if resolved_session_id != session_id:
            raise RuntimeError("graph 返回的 session_id 与 durable session 不一致")
        async for payload in iter_research_payloads(
            graph_stream, stream_kind=stream_kind, session_id=session_id,
        ):
            if payload.get("type") == "done":
                done = payload
            else:
                await rag_run_service.append_event(
                    user_id=user_id, run_id=run_id,
                    event_type=str(payload.get("type") or "event"), payload=payload,
                )
    finally:
        # 1. 【研究执行】【流式排空】done 只代表产物已产生，必须等待图与 checkpoint 清理后才能提交
        if callable(getattr(graph_stream, "aclose", None)):
            await graph_stream.aclose()
    if done is None:
        raise RuntimeError("research stream ended without a done payload")
    return done


def _stream_graph_extra(
    req: ResearchStreamRequest,
    *,
    user_id: str,
    conversation_id: str | None,
    project_context: dict,
    core_intent: str,
    session: dict,
) -> dict:
    """组装固定 durable identity 的流式 graph 输入；user_id 供 trace 归属使用。"""
    return {
        "intent": req.intent or INTENT_DEEP_RESEARCH,
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
    }


async def _stream_project_context_or_http(
    *,
    lifecycle: ResearchSessionLifecycle,
    user_id: str,
    req: ResearchStreamRequest,
) -> dict:
    """
    回读 durable scope 并构造 owner-scoped 流式研究上下文。

    Args:
        lifecycle: 研究 session 生命周期服务
        user_id: 当前登录用户 ID
        req: 流式研究请求

    Returns:
        dict: 已完成 owner 校验的 project/paper context
    """
    try:
        effective_scope = await lifecycle.resolve_scope_request(
            user_id=user_id,
            session_id=req.session_id,
            project_id=req.project_id,
            paper_ids=req.paper_ids,
        )
        return await build_project_context(
            effective_scope.get("project_id"),
            requested_paper_ids=effective_scope.get("paper_ids"),
            user_id=user_id,
        )
    except ResearchSessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ResearchSessionScopeConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ScopeOwnershipError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/research/stream")
async def stream(
    req: ResearchStreamRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """参数为问题与稳定请求标识；只有原子新领取者启动后台图，其余订阅原操作。"""
    current_user_id = get_user_id(user)
    existing = await rag_run_service.get_by_idempotency_key(
        user_id=current_user_id, idempotency_key=idempotency_key,
    )
    if existing is not None:
        try:
            rag_run_service.assert_same_operation(existing, operation_kind="research",
                                                   request_payload=req.model_dump(mode="json"))
        except rag_run_service.IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return build_run_event_response(user_id=current_user_id, run_id=existing["run_id"], after_sequence=0)
    conversation_id = await _resolve_identity_or_404(current_user_id, req.conversation_id, req.session_id)
    lifecycle = get_research_session_lifecycle()
    project_context = await _stream_project_context_or_http(lifecycle=lifecycle, user_id=current_user_id, req=req)
    core_intent = resolve_core_research_intent(project_context["project_id"], paper_ids=project_context["paper_ids"])
    if core_intent == INTENT_INTERNAL_RESEARCH and not project_context["local_retrieval_allowed"]:
        raise HTTPException(status_code=422, detail=str(RetrievalScopeError()))
    research_mode = "internal" if core_intent == INTENT_INTERNAL_RESEARCH else "external"
    session = await _prepare_stream_session(lifecycle=lifecycle, req=req, user_id=current_user_id,
        conversation_id=conversation_id, project_context=project_context, research_mode=research_mode)
    run, session, created = await _accept_research_operation(
        lifecycle=lifecycle, user_id=current_user_id, session=session, operation_kind="research",
        idempotency_key=idempotency_key, query=req.query, request_payload=req.model_dump(mode="json"),
    )
    if created:
        await _start_research_stream(req=req, conversation_id=conversation_id, project_context=project_context,
            core_intent=core_intent, session=session, lifecycle=lifecycle, user_id=current_user_id, run=run)
    return build_run_event_response(user_id=current_user_id, run_id=run["run_id"], after_sequence=0)


async def _start_research_stream(*, req: ResearchStreamRequest, conversation_id: str | None,
                                 project_context: dict, core_intent: str, session: dict,
                                 lifecycle: ResearchSessionLifecycle, user_id: str, run: dict) -> None:
    """参数为新领取操作；启动受管流，应用已关闭时原子记录确定未启动的终态。"""
    try:
        get_rag_run_executor_registry().start(
            user_id=user_id, run_id=run["run_id"],
            timeout_seconds=float(get_settings().deep_research_graph_timeout_seconds),
            runner_factory=partial(_execute_durable_stream, req=req, conversation_id=conversation_id,
                project_context=project_context, core_intent=core_intent, session=session,
                lifecycle=lifecycle, user_id=user_id, run=run),
            terminal_callback=partial(_record_operation_failure, user_id=user_id, run_id=run["run_id"]),
        )
    except RuntimeError as exc:
        await _record_operation_failure(user_id=user_id, run_id=run["run_id"],
            status="cancelled", error="应用关闭期间未启动研究运行")
        raise HTTPException(status_code=503, detail="应用关闭期间未启动研究运行") from exc
