"""
统一定义 session / thread / conversation 的语义映射。

这是理解本项目状态持久化的关键文件之一。
项目里同时存在三种看起来相似、但职责不同的标识：

1. `session_id`
   业务侧“这次对话/研究任务”的标识
2. `conversation_id`
   前端会话展示库中的对话标识
3. `thread_id`
   LangGraph checkpoint 真正使用的线程标识

这里的职责就是把三者关系固定下来，避免不同模块各自发明映射规则。
"""

from __future__ import annotations

import uuid
from typing import Any


def _normalize(value: str | None) -> str:
    """统一做字符串化和去空白，减少各处重复处理。"""
    return str(value or "").strip()


def ensure_session_id(session_id: str | None, conversation_id: str | None = None) -> str:
    """
    确保每次图调用都拿到稳定的 session_id。

    新 session 只接受显式 session_id 或生成随机 UUID。
    `conversation_id` 参数暂时保留调用兼容，但不再参与身份兜底，
    避免 UI 对话身份继续变成 durable session 主键。
    """
    normalized_session_id = _normalize(session_id)
    if normalized_session_id:
        return normalized_session_id
    return str(uuid.uuid4())


def session_id_to_thread_id(session_id: str) -> str:
    """兼容旧路径：仅按 session_id 推导 thread_id。"""
    return _normalize(session_id)


def resolve_thread_id(
    session_id: str,
    conversation_id: str | None = None,
    *,
    thread_id: str | None = None,
) -> str:
    """
    统一计算 LangGraph thread_id。

    新 session 固定 `thread_id == session_id`；历史 session 只有在生命周期服务
    已根据 checkpoint/conversation 证据持久化 `thread_id` 后，才允许使用该显式值。
    `conversation_id` 仅保留调用兼容，不再隐式覆盖 checkpoint key。
    """
    normalized_thread_id = _normalize(thread_id)
    if normalized_thread_id:
        return normalized_thread_id
    return session_id_to_thread_id(session_id)


def build_session_identity(
    session_id: str,
    conversation_id: str | None = None,
    *,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """显式返回当前请求的会话语义边界。"""
    payload = {
        "session_id": session_id,
        "thread_id": resolve_thread_id(session_id, conversation_id, thread_id=thread_id),
    }
    if _normalize(conversation_id):
        payload["conversation_id"] = _normalize(conversation_id)
    return payload
