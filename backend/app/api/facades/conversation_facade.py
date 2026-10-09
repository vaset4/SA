"""
前端会话管理 facade。

这层主要做两件事：
1. 把 route 层的 HTTP 语义转换成更靠近业务的调用
2. 补齐 facade 层特有的小规则，例如生成 conversation_id / message_id

它本身仍然很薄，但把这些“边界杂务”从 route 和 service 中间剥离出来后，
接口职责会更清楚。
"""

from __future__ import annotations

import uuid
from typing import Any

from app.services import conversation_service


async def list_conversations(user_id: str, limit: int, offset: int, sort: str) -> dict[str, Any]:
    """获取前端会话库中的对话列表。"""
    conversations = await conversation_service.list_conversations(
        user_id=user_id,
        limit=limit,
        offset=offset,
        sort_by=sort,
    )
    return {"conversations": conversations, "total": len(conversations)}


async def create_conversation(
    user_id: str,
    title: str | None,
    associated_papers: list[Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    创建前端会话记录。

    conversation_id 在 facade 层生成，而不是交给 route 或 repository，
    这样 ID 生成策略集中在一处，便于后续统一替换。
    """
    conversation_id = f"conv-{uuid.uuid4().hex[:12]}"
    return await conversation_service.create_conversation(
        conversation_id,
        title=title,
        associated_papers=associated_papers,
        user_id=user_id,
        session_id=session_id,
    )


async def get_conversation_or_none(user_id: str, conversation_id: str):
    """获取单个前端会话。"""
    return await conversation_service.get_conversation(conversation_id, user_id=user_id)


async def get_conversation_by_session_id_or_none(user_id: str, session_id: str):
    """按 session_id 反查前端会话。"""
    return await conversation_service.find_conversation_by_session_id(session_id, user_id=user_id)


async def update_conversation_or_raise(
    user_id: str,
    conversation_id: str,
    title: str | None,
    is_pinned: bool | None,
    tags: list[Any] | None,
    associated_papers: list[Any] | None,
    session_id: str | None,
) -> dict[str, Any]:
    """更新对话元数据；若目标不存在则抛出统一的 LookupError。"""
    success = await conversation_service.update_conversation(
        conversation_id,
        user_id=user_id,
        title=title,
        is_pinned=is_pinned,
        tags=tags,
        associated_papers=associated_papers,
        session_id=session_id,
    )
    if not success:
        raise LookupError("对话不存在")
    return {"success": True, "message": "对话已更新"}


async def delete_conversation_or_raise(user_id: str, conversation_id: str) -> dict[str, Any]:
    """删除前端会话记录。"""
    success = await conversation_service.delete_conversation(conversation_id, user_id=user_id)
    if not success:
        raise LookupError("对话不存在")
    return {"success": True, "message": "对话已删除"}


async def add_message(
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    title: str | None,
    tone: str | None,
    render_mode: str | None,
    metadata: dict[str, Any] | None = None,
):
    """向前端会话库追加消息。"""
    # message_id 同样由 facade 统一生成，避免上层重复关心 ID 细节。
    message_id = f"msg-{uuid.uuid4().hex[:12]}"
    return await conversation_service.add_message(
        conversation_id,
        message_id=message_id,
        role=role,
        content=content,
        user_id=user_id,
        title=title,
        tone=tone,
        render_mode=render_mode,
        metadata=metadata,
    )


async def set_messages_or_raise(
    user_id: str,
    conversation_id: str,
    messages: list[Any],
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """批量覆盖某个对话的消息列表。"""
    success = await conversation_service.set_messages(
        conversation_id,
        messages,
        user_id=user_id,
        session_id=session_id,
    )
    if not success:
        raise LookupError("对话不存在")
    return {"success": True, "message": "消息已更新"}


async def add_paper_to_conversation_or_raise(user_id: str, conversation_id: str, paper_id: str) -> dict[str, Any]:
    """
    给对话增加关联论文。

    facade 层在成功写入后会再回查一次 conversation，
    这样前端可以直接拿到最新的 `associated_papers`，无需再发一次 GET。
    """
    success = await conversation_service.add_paper_to_conversation(
        conversation_id,
        paper_id,
        user_id=user_id,
    )
    if not success:
        raise LookupError("对话不存在")
    conversation = await conversation_service.get_conversation(conversation_id, user_id=user_id)
    return {
        "success": True,
        "message": "论文已添加到对话",
        "associated_papers": conversation["associated_papers"],
    }


async def remove_paper_from_conversation_or_raise(user_id: str, conversation_id: str, paper_id: str) -> dict[str, Any]:
    """从对话中移除关联论文。"""
    success = await conversation_service.remove_paper_from_conversation(
        conversation_id,
        paper_id,
        user_id=user_id,
    )
    if not success:
        raise LookupError("对话不存在")
    conversation = await conversation_service.get_conversation(conversation_id, user_id=user_id)
    return {
        "success": True,
        "message": "论文已从对话移除",
        "associated_papers": conversation["associated_papers"],
    }
