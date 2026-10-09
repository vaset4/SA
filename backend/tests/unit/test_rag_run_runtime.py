"""
durable RAG run 幂等、事件和终态合同测试。

作者: xxx
时间: 2026-08-19 13:30:55
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.api.facades import rag_run_stream_facade
from app.api.schemas.chat import ChatRequest
from app.api.schemas.request_limits import (
    MAX_CHAT_HISTORY_ITEMS,
    MAX_CHAT_MESSAGE_LENGTH,
    MAX_PAPER_IDS,
    MAX_QUERY_LENGTH,
)
from app.api.schemas.research import CrossPaperQARequest, ResearchStreamRequest
from app.services import rag_run_executor_service, rag_run_service


class _CreateRepository:
    """模拟数据库唯一约束对同 key 和同 thread 的裁决。"""

    def __init__(self) -> None:
        self.by_key: dict[str, dict] = {}
        self.active_by_thread: dict[str, dict] = {}

    async def create_or_get_run(self, **payload) -> tuple[dict | None, bool]:
        key = payload["idempotency_key"]
        thread_id = payload["thread_id"]
        if key in self.by_key:
            return self.by_key[key], False
        if thread_id in self.active_by_thread:
            return None, False
        run = {**payload, "status": rag_run_service.STATUS_QUEUED}
        self.by_key[key] = run
        self.active_by_thread[thread_id] = run
        return run, True

    async def get_by_idempotency_key(self, *, user_id: str, idempotency_key: str) -> dict | None:
        return self.by_key.get(idempotency_key)

    async def get_active_for_thread(self, *, user_id: str, thread_id: str) -> dict | None:
        return self.active_by_thread.get(thread_id)


class _TerminalRepository:
    """记录终态 CAS 是否被重复调用。"""

    def __init__(self) -> None:
        self.run = {"run_id": "run-1", "status": rag_run_service.STATUS_COMPLETED}
        self.finish_calls = 0

    async def get_run(self, *, user_id: str, run_id: str) -> dict:
        return self.run

    async def finish_run(self, **payload) -> dict:
        self.finish_calls += 1
        return self.run


@pytest.mark.asyncio
async def test_same_key_reuses_run_and_same_thread_rejects_second_action() -> None:
    """同 key 返回原 run；不同 key 不得绕过同 thread 活跃约束。"""
    repository = _CreateRepository()
    deadline = datetime.now(timezone.utc)
    common = {
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "session_id": "session-1",
        "thread_id": "session-1",
        "query": "compare papers",
        "deadline_at": deadline,
        "repository": repository,
    }

    first, created = await rag_run_service.create_or_get_stream_run(
        **common,
        idempotency_key="action-key-1",
    )
    repeated, repeated_created = await rag_run_service.create_or_get_stream_run(
        **common,
        idempotency_key="action-key-1",
    )

    assert created is True
    assert repeated_created is False
    assert repeated["run_id"] == first["run_id"]
    with pytest.raises(rag_run_service.ActiveRAGRunConflict):
        await rag_run_service.create_or_get_stream_run(
            **common,
            idempotency_key="action-key-2",
        )


@pytest.mark.asyncio
async def test_terminal_run_is_irreversible_without_second_terminal_write() -> None:
    """已经 completed 的 run 不再执行第二次终态 CAS。"""
    repository = _TerminalRepository()

    result = await rag_run_service.finish_run(
        user_id="user-1",
        run_id="run-1",
        status=rag_run_service.STATUS_FAILED,
        repository=repository,
    )

    assert result["status"] == rag_run_service.STATUS_COMPLETED
    assert repository.finish_calls == 0


@pytest.mark.asyncio
async def test_event_stream_replays_only_after_requested_sequence(monkeypatch) -> None:
    """重连从 after_sequence 后重放，并在唯一 terminal 事件处结束。"""
    calls: list[int] = []

    async def fake_list_events(**payload) -> list[dict]:
        calls.append(payload["after_sequence"])
        return [{
            "sequence": 3,
            "event_type": "terminal",
            "payload": {"type": "done", "answer": "complete"},
        }]

    monkeypatch.setattr(rag_run_stream_facade.rag_run_service, "list_events", fake_list_events)
    stream = rag_run_stream_facade.stream_run_event_frames(
        user_id="user-1",
        run_id="run-1",
        after_sequence=2,
        cancel_on_disconnect=False,
    )
    frames = [frame async for frame in stream]
    payload = json.loads(frames[0].removeprefix("data: ").strip())

    assert calls == [2]
    assert payload["sequence"] == 3
    assert payload["type"] == "done"


@pytest.mark.asyncio
async def test_executor_timeout_and_shutdown_leave_terminal_runs(monkeypatch) -> None:
    """超时和 shutdown 取消都写真实终态，并清空进程任务注册表。"""
    terminal_statuses: list[str] = []

    async def fake_finish_run(**payload) -> dict:
        terminal_statuses.append(payload["status"])
        return {"run_id": payload["run_id"], "status": payload["status"]}

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(rag_run_executor_service.rag_run_service, "finish_run", fake_finish_run)
    repository = SimpleNamespace(
        claim_execution=AsyncMock(return_value={"status": "running"}),
        record_execution_stopped=AsyncMock(return_value={"status": "running"}),
    )
    monkeypatch.setattr(rag_run_executor_service, "get_rag_run_repository", lambda: repository)
    run_state = {"status": "running", "cancel_requested": False}

    async def request_cancel(**payload):
        """参数为取消身份；更新隔离运行信号，供正式监测协程消费。"""
        run_state["cancel_requested"] = True
        return dict(run_state)

    monkeypatch.setattr(rag_run_service, "request_cancel", request_cancel)
    monkeypatch.setattr(rag_run_service, "get_run", AsyncMock(side_effect=lambda **payload: dict(run_state)))
    registry = rag_run_executor_service.RAGRunExecutorRegistry()
    registry.start(
        user_id="user-1",
        run_id="run-timeout",
        timeout_seconds=0.01,
        runner_factory=wait_forever,
    )
    await asyncio.sleep(0.05)
    registry.start(
        user_id="user-1",
        run_id="run-shutdown",
        timeout_seconds=10,
        runner_factory=wait_forever,
    )
    await asyncio.sleep(0)
    await registry.close()

    assert terminal_statuses == [
        rag_run_service.STATUS_TIMED_OUT,
        rag_run_service.STATUS_CANCELLED,
    ]
    assert registry.active_count() == 0


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ResearchStreamRequest, {"query": "q" * (MAX_QUERY_LENGTH + 1)}),
        (ResearchStreamRequest, {"query": "q", "paper_ids": [f"paper-{index}" for index in range(MAX_PAPER_IDS + 1)]}),
        (ChatRequest, {"query": "q", "chat_history": [{"content": "x"}] * (MAX_CHAT_HISTORY_ITEMS + 1)}),
        (CrossPaperQARequest, {"query": "q", "chat_history": [{"content": "x" * (MAX_CHAT_MESSAGE_LENGTH + 1)}]}),
    ],
)
def test_request_models_reject_unbounded_inputs(model, payload) -> None:
    """超长文本和无界集合在 graph/run 创建前由 Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        model.model_validate(payload)
