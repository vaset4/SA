"""User-visible RAG run status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.api.facades.rag_run_stream_facade import build_run_event_response
from app.middleware.auth import get_current_user_with_rate_limit, get_user_id
from app.services import rag_run_service
from app.services.rag_run_executor_service import get_rag_run_executor_registry
from app.models.result_delivery_contracts import DeliveryConflict
from app.services.result_delivery_service import retry_saved_result, read_current_delivery_response
from app.services.internal_research.research_planner_final_retry import (
    retry_planner_final_answer,
)

router = APIRouter()


@router.get("/rag-runs/{run_id}")
async def get_rag_run(
    run_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    run = await rag_run_service.get_run(user_id=get_user_id(user), run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAG run 不存在")
    return {**run, "current_result": await read_current_delivery_response(run)}


@router.get("/rag-runs/{run_id}/events")
async def get_rag_run_events(
    run_id: str,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """按 owner 返回可重放 run 事件；用于诊断和非 SSE 恢复。"""
    user_id = get_user_id(user)
    if await rag_run_service.get_run(user_id=user_id, run_id=run_id) is None:
        raise HTTPException(status_code=404, detail="RAG run 不存在")
    return {
        "events": await rag_run_service.list_events(
            user_id=user_id,
            run_id=run_id,
            after_sequence=after_sequence,
            limit=limit,
        )
    }


@router.get("/rag-runs/{run_id}/events/stream")
async def subscribe_rag_run_events(
    run_id: str,
    after_sequence: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """重连时只订阅已有 run，不重新调用 graph。"""
    user_id = get_user_id(user)
    if await rag_run_service.get_run(user_id=user_id, run_id=run_id) is None:
        raise HTTPException(status_code=404, detail="RAG run 不存在")
    return build_run_event_response(
        user_id=user_id,
        run_id=run_id,
        after_sequence=after_sequence,
    )


@router.post("/rag-runs/{run_id}/cancel")
async def cancel_rag_run(
    run_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    """取消 durable run 及本进程持有的真实执行任务。"""
    try:
        return await get_rag_run_executor_registry().cancel(
            user_id=get_user_id(user),
            run_id=run_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="RAG run 不存在") from exc


@router.get("/conversations/{conversation_id}/rag-runs/latest")
async def get_latest_conversation_rag_run(
    conversation_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
):
    run = await rag_run_service.get_latest_for_conversation(
        user_id=get_user_id(user),
        conversation_id=conversation_id,
    )
    if run is None:
        return {"run": None}
    return {"run": run}


@router.post("/conversations/{conversation_id}/rag-runs/{run_id}/retry-final-answer")
async def retry_conversation_rag_run_final_answer(
    conversation_id: str,
    run_id: str,
    user: dict = Depends(get_current_user_with_rate_limit),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
):
    """【研究交付】【最终重试入口】参数为原答案与动作标识；返回新操作和目标结果的同版本回执。"""
    try:
        return await retry_planner_final_answer(
            user_id=get_user_id(user),
            conversation_id=conversation_id,
            run_id=run_id,
            idempotency_key=idempotency_key,
        )
    except (rag_run_service.ActiveRAGRunConflict, rag_run_service.IdempotencyConflict, DeliveryConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/rag-runs/{run_id}/retry-save")
async def retry_rag_run_save(run_id: str,
                             idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
                             user: dict = Depends(get_current_user_with_rate_limit)):
    """【研究交付】【补存入口】参数为原目标与动作标识；仅提交服务器暂存产物，返回真实保存结果。"""
    try:
        return await retry_saved_result(user_id=get_user_id(user), target_run_id=run_id, idempotency_key=idempotency_key)
    except (rag_run_service.ActiveRAGRunConflict, rag_run_service.IdempotencyConflict, DeliveryConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/rag-runs/{run_id}/retry-memory-writeback")
async def retry_rag_run_memory(run_id: str, user: dict = Depends(get_current_user_with_rate_limit)):
    """【研究记忆】【候选重试】参数为已交付运行；仅恢复该版本候选，不重新生成答案。"""
    from app.services.internal_research.research_memory_writeback import after_result_committed

    run = await rag_run_service.get_run(user_id=get_user_id(user), run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAG run 不存在")
    metadata = run.get("metadata") or {}
    if metadata.get("persistence_status") != "saved" or not run.get("delivery_revision"):
        raise HTTPException(status_code=409, detail="请先保存答案，再重试候选处理")
    try:
        outcome = await after_result_committed({"run": run, "delivery_revision": run["delivery_revision"]}, processing_source="http_retry")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run_id, "delivery_revision": run["delivery_revision"], "memory_writeback": outcome}
