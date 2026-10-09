from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from fastapi import HTTPException

import app.api.routes.chat_api as chat_api
import app.api.routes.research_api as research_api
import app.api.routes.stream_api as stream_api
from app.api.schemas.chat import ChatRequest
from app.api.schemas.research import ResearchRequest, ResearchStreamRequest


@pytest.fixture(autouse=True)
def isolate_operation_lookup(monkeypatch):
    """参数为补丁工具；隔离前置幂等读取，只验证真正的身份错误映射。"""
    monkeypatch.setattr(research_api.rag_run_service, "get_by_idempotency_key", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_chat_maps_identity_lookup_failure_to_404(monkeypatch) -> None:
    async def fake_resolve_identity_or_404(*args, **kwargs):
        raise HTTPException(status_code=404, detail="会话未关联对话")

    monkeypatch.setattr(chat_api, "_resolve_identity_or_404", fake_resolve_identity_or_404)

    with pytest.raises(HTTPException) as exc_info:
        await chat_api.chat(
            ChatRequest(query="hello", session_id="session-1"),
            user={"user_id": "user-1"},
            idempotency_key="identity-research-key",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "会话未关联对话"


@pytest.mark.asyncio
async def test_research_maps_identity_lookup_failure_to_404(monkeypatch) -> None:
    async def fake_resolve_conversation_identity(**kwargs):
        raise LookupError("会话未关联对话")

    monkeypatch.setattr(research_api, "resolve_conversation_identity", fake_resolve_conversation_identity)

    with pytest.raises(HTTPException) as exc_info:
        await research_api.research(
            ResearchRequest(query="research", session_id="session-1"),
            user={"user_id": "user-1"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "会话未关联对话"


@pytest.mark.asyncio
async def test_stream_maps_identity_lookup_failure_to_404(monkeypatch) -> None:
    async def fake_resolve_conversation_identity(**kwargs):
        raise LookupError("会话未关联对话")

    monkeypatch.setattr(stream_api, "resolve_conversation_identity", fake_resolve_conversation_identity)

    with pytest.raises(HTTPException) as exc_info:
        await stream_api.stream(
            ResearchStreamRequest(query="stream", session_id="session-1"),
            # 流式入口要求携带幂等键（Header 必填），缺了会在建连前就因参数类型炸掉
            idempotency_key="test-idempotency-key-0001",
            user={"user_id": "user-1"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "会话未关联对话"
