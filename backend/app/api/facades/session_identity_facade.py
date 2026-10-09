"""
会话身份解析 facade。

这层统一处理 chat / research / stream 路由里最容易混淆的两个概念：
1. `conversation_id`：前端展示层对话 ID
2. `session_id`：LangGraph checkpoint / 恢复维度
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.api.facades import conversation_facade
from app.repositories.session_repository import get_session_repository


@dataclass(frozen=True)
class ConversationIdentity:
    conversation_id: str | None
    session_id: str | None


def _normalize(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


async def resolve_conversation_identity(
    *,
    user_id: str,
    conversation_id: str | None,
    session_id: str | None,
    allow_missing_conversation: bool = False,
) -> ConversationIdentity:
    """
    统一解析 conversation_id / session_id 对应的合法身份。

    Args:
        user_id: 当前登录用户 ID
        conversation_id: 前端展示层对话 ID
        session_id: LangGraph 会话 ID
        allow_missing_conversation: 是否允许 session 尚未绑定 conversation 的合法空态

    Returns:
        ConversationIdentity: 已完成归属校验后的身份结果
    """
    normalized_conversation_id = _normalize(conversation_id)
    normalized_session_id = _normalize(session_id)
    # 1. 优先按显式 conversation_id 解析，因为这是前端最明确的用户意图
    if normalized_conversation_id:
        return await _resolve_explicit_conversation(
            user_id=user_id,
            conversation_id=normalized_conversation_id,
            session_id=normalized_session_id,
        )
    # 2. 没有 conversation_id 时，再按 session_id 反查对应对话
    if normalized_session_id:
        return await _resolve_from_session(
            user_id=user_id,
            session_id=normalized_session_id,
            allow_missing_conversation=allow_missing_conversation,
        )
    # 3. 两者都没有时，返回一个空身份，让上层自己决定是否允许无状态请求
    return ConversationIdentity(conversation_id=None, session_id=None)


async def _resolve_explicit_conversation(
    *,
    user_id: str,
    conversation_id: str,
    session_id: str | None,
) -> ConversationIdentity:
    """
    按显式 conversation_id 解析并校验归属。

    Args:
        user_id: 当前登录用户 ID
        conversation_id: 显式传入的前端对话 ID
        session_id: 可选 LangGraph 会话 ID

    Returns:
        ConversationIdentity: 归属合法的 conversation / session 组合
    """
    conversation = await conversation_facade.get_conversation_or_none(user_id, conversation_id)
    if conversation is None:
        raise LookupError("对话不存在")
    if session_id is not None:
        session = await _ensure_owned_session(user_id=user_id, session_id=session_id)
        persisted_conversation_id = _normalize(session.get("conversation_id"))
        if persisted_conversation_id and persisted_conversation_id != conversation_id:
            raise LookupError("会话与对话身份不一致")
    return ConversationIdentity(conversation_id=conversation_id, session_id=session_id)


async def _resolve_from_session(
    *,
    user_id: str,
    session_id: str,
    allow_missing_conversation: bool,
) -> ConversationIdentity:
    """
    按 session_id 反查 conversation，并处理“尚未绑定对话”的空态。

    Args:
        user_id: 当前登录用户 ID
        session_id: LangGraph 会话 ID
        allow_missing_conversation: 是否允许 session 存在但 conversation 尚未建立

    Returns:
        ConversationIdentity: 由 session 反查出的 conversation / session 组合
    """
    session = await _ensure_owned_session(user_id=user_id, session_id=session_id)
    persisted_conversation_id = _normalize(session.get("conversation_id"))
    if persisted_conversation_id:
        conversation = await conversation_facade.get_conversation_or_none(
            user_id,
            persisted_conversation_id,
        )
        if conversation is None:
            raise LookupError("对话不存在")
    else:
        # legacy session 没有显式 conversation_id 时，才允许按旧关联表反查一次
        conversation = await conversation_facade.get_conversation_by_session_id_or_none(user_id, session_id)
    if conversation is None:
        if allow_missing_conversation:
            return ConversationIdentity(conversation_id=None, session_id=session_id)
        raise LookupError("会话未关联对话")
    conversation_id = _normalize(conversation.get("id"))
    if conversation_id is None:
        raise LookupError("会话未关联对话")
    return ConversationIdentity(conversation_id=conversation_id, session_id=session_id)


async def _ensure_owned_session(*, user_id: str, session_id: str) -> dict[str, Any]:
    """通过异步仓储校验 session 归属并返回 durable identity。"""
    session = await get_session_repository().get_session(session_id, user_id=user_id)
    if session is None:
        raise LookupError("会话不存在")
    return session
