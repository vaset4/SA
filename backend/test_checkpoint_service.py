from __future__ import annotations

import asyncio
from types import SimpleNamespace

import app.services.checkpoint_service as checkpoint_service


def test_build_checkpoint_config_maps_session_to_thread() -> None:
    assert checkpoint_service.build_checkpoint_config("sess-1") == {
        "configurable": {"thread_id": "sess-1"}
    }


def test_build_checkpoint_config_does_not_use_conversation_as_thread() -> None:
    """conversation_id 只保留展示语义，不能隐式覆盖 checkpoint key。"""
    assert checkpoint_service.build_checkpoint_config("sess-1", "conv-1") == {
        "configurable": {"thread_id": "sess-1"}
    }


def test_build_checkpoint_config_uses_persisted_thread_when_provided() -> None:
    """legacy session 已持久化的 thread_id 必须优先用于 checkpoint。"""
    assert checkpoint_service.build_checkpoint_config(
        "sess-1",
        "conv-1",
        thread_id="legacy-thread",
    ) == {
        "configurable": {"thread_id": "legacy-thread"}
    }


def test_is_empty_snapshot_detects_blank_checkpoint() -> None:
    snapshot = SimpleNamespace(
        created_at=None,
        values={},
        next=(),
        tasks=(),
        interrupts=(),
    )

    assert checkpoint_service.is_empty_snapshot(snapshot) is True


def test_serialize_snapshot_preserves_session_and_conversation_identity() -> None:
    """快照同时返回 durable session、展示 conversation 和显式 thread 身份。"""
    interrupt = SimpleNamespace(id="interrupt-1", value={"approve": True})
    task = SimpleNamespace(
        id="task-1",
        name="router",
        path=("router",),
        error=None,
        interrupts=(interrupt,),
        state={"intent": "rag_qa"},
        result={"intent": "rag_qa"},
    )
    snapshot = SimpleNamespace(
        values={"report": "ok"},
        next=("done",),
        config={"configurable": {"thread_id": "sess-1"}},
        metadata={"source": "unit-test"},
        created_at="2026-03-23T21:00:00+08:00",
        parent_config=None,
        tasks=(task,),
        interrupts=(interrupt,),
    )

    payload = checkpoint_service.serialize_snapshot(
        session_id="sess-1",
        snapshot=snapshot,
        conversation_id="conv-1",
    )

    assert payload["session_id"] == "sess-1"
    assert payload["thread_id"] == "sess-1"
    assert payload["conversation_id"] == "conv-1"
    assert payload["values"]["report"] == "ok"
    assert payload["tasks"][0]["name"] == "router"
    assert payload["interrupts"][0]["id"] == "interrupt-1"


def test_read_graph_snapshot_returns_none_for_empty_checkpoint() -> None:
    class FakeAgent:
        async def aget_state(self, config):  # noqa: ANN001
            assert config == {"configurable": {"thread_id": "sess-empty"}}
            return SimpleNamespace(
                created_at=None,
                values={},
                next=(),
                tasks=(),
                interrupts=(),
            )

    result = asyncio.run(
        checkpoint_service.read_graph_snapshot(FakeAgent(), "sess-empty")
    )

    assert result is None
