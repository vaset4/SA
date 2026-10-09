"""
LangGraph checkpoint 读取与序列化辅助。

这个文件的作用是把 LangGraph 原生 snapshot / interrupt / task 结构，
转换成 API 层更稳定、更容易序列化的 JSON 结构。

换句话说：
- graph 内部状态由 LangGraph 管
- 对外暴露格式由这里统一定义
"""

from __future__ import annotations

from typing import Any

from fastapi.encoders import jsonable_encoder

from app.services.session_mapping_service import build_session_identity, resolve_thread_id


def build_checkpoint_config(
    session_id: str,
    conversation_id: str | None = None,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    构造 LangGraph 读取/恢复所需的 thread config。

    所有读写 checkpoint 的地方都应该通过这层生成 config，
    避免 thread_id 拼接规则散落在多个文件里。
    """
    return {
        "configurable": {
            "thread_id": resolve_thread_id(
                session_id,
                conversation_id,
                thread_id=thread_id,
            )
        }
    }


def serialize_interrupt(interrupt: Any) -> dict[str, Any]:
    """把 LangGraph Interrupt 转成稳定 JSON。"""
    return {
        "id": getattr(interrupt, "id", ""),
        "value": jsonable_encoder(getattr(interrupt, "value", None)),
    }


def serialize_task(task: Any) -> dict[str, Any]:
    """
    提取任务快照中的核心字段。

    task 结构常带有较多内部实现细节，这里只保留 API 和调试真正关心的字段。
    """
    return {
        "id": getattr(task, "id", ""),
        "name": getattr(task, "name", ""),
        "path": list(getattr(task, "path", ())),
        "error": str(task.error) if getattr(task, "error", None) else None,
        "interrupts": [
            serialize_interrupt(interrupt)
            for interrupt in getattr(task, "interrupts", ())
        ],
        "state": jsonable_encoder(getattr(task, "state", None)),
        "result": jsonable_encoder(getattr(task, "result", None)),
    }


def is_empty_snapshot(snapshot: Any) -> bool:
    """
    判断 thread_id 是否尚未产生有效 checkpoint。

    “空快照”不代表报错，而通常意味着：
    session 已存在，但 graph 还没有真正跑出任何状态。
    """
    return (
        getattr(snapshot, "created_at", None) is None
        and not getattr(snapshot, "values", {})
        and not getattr(snapshot, "next", ())
        and not getattr(snapshot, "tasks", ())
        and not getattr(snapshot, "interrupts", ())
    )


def serialize_snapshot(
    session_id: str,
    snapshot: Any,
    conversation_id: str | None = None,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """
    把 LangGraph StateSnapshot 转成 API 友好结构。

    这里顺便把 session/conversation 身份信息合并进来，
    让前端拿到的是一份完整的“可展示快照对象”。
    """
    payload = build_session_identity(
        session_id,
        conversation_id,
        thread_id=thread_id,
    )
    payload.update(
        {
            "values": jsonable_encoder(snapshot.values),
            "next": list(snapshot.next),
            "config": jsonable_encoder(snapshot.config),
            "metadata": jsonable_encoder(snapshot.metadata),
            "created_at": snapshot.created_at,
            "parent_config": jsonable_encoder(snapshot.parent_config),
            "tasks": [serialize_task(task) for task in snapshot.tasks],
            "interrupts": [
                serialize_interrupt(interrupt)
                for interrupt in snapshot.interrupts
            ],
        }
    )
    return payload


async def read_graph_snapshot(
    agent: Any,
    session_id: str,
    conversation_id: str | None = None,
    *,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """
    读取指定 session 的官方 checkpoint 快照。

    若 snapshot 为空，则返回 `None` 而不是空壳对象，
    由上层决定是回退成空会话还是直接返回 404。
    """
    snapshot = await agent.aget_state(
        build_checkpoint_config(
            session_id,
            conversation_id,
            thread_id=thread_id,
        )
    )
    if is_empty_snapshot(snapshot):
        return None
    return serialize_snapshot(
        session_id=session_id,
        snapshot=snapshot,
        conversation_id=conversation_id,
        thread_id=thread_id,
    )
