"""
Durable RAG run 事件到 SSE 响应的协议适配。

作者: xxx
时间: 2026-08-19 18:35:00
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from app.services import rag_run_service
from app.services.rag_run_executor_service import get_rag_run_executor_registry

EVENT_POLL_INTERVAL_SECONDS = 0.2
EVENT_BATCH_LIMIT = 100


def _encode_event(payload: dict[str, Any]) -> str:
    """把持久化 run 事件编码为标准 SSE 帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_run_event_frames(
    *,
    user_id: str,
    run_id: str,
    after_sequence: int,
    cancel_on_disconnect: bool = True,
) -> AsyncIterator[str]:
    """从数据库按 sequence 重放并持续订阅 run 事件。"""
    sequence = after_sequence
    terminal_delivered = False
    try:
        while True:
            events = await rag_run_service.list_events(
                user_id=user_id,
                run_id=run_id,
                after_sequence=sequence,
                limit=EVENT_BATCH_LIMIT,
            )
            for event in events:
                sequence = int(event["sequence"])
                payload = {**event["payload"], "sequence": sequence}
                # 1. 【研究交付】【订阅关闭】终态送出前记录事实，客户端正常关流不能取消提交后候选
                terminal_delivered = event.get("event_type") == "terminal"
                yield _encode_event(payload)
                if event.get("event_type") == "terminal":
                    return
            if not events:
                run = await rag_run_service.get_run(user_id=user_id, run_id=run_id)
                if run is None or run.get("status") in rag_run_service.TERMINAL_STATUSES:
                    return
            await asyncio.sleep(EVENT_POLL_INTERVAL_SECONDS)
    except (asyncio.CancelledError, GeneratorExit):
        if cancel_on_disconnect and not terminal_delivered:
            await get_rag_run_executor_registry().cancel(user_id=user_id, run_id=run_id)
        raise


def build_run_event_response(*, user_id: str, run_id: str, after_sequence: int) -> StreamingResponse:
    """构造禁缓存的 durable run 事件订阅响应。"""
    response = StreamingResponse(
        stream_run_event_frames(
            user_id=user_id,
            run_id=run_id,
            after_sequence=after_sequence,
        ),
        media_type="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["X-RAG-Run-ID"] = run_id
    return response
