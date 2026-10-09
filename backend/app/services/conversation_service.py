"""
前端会话展示层服务。

这一层夹在 route 和 repository 之间，职责偏“薄 service”：
- 提供统一的 async API
- 屏蔽 repository 单例获取细节
- 兼容旧测试桩或同步旧调用方式

也就是说，它更多是在整理边界，而不是承载复杂业务规则。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from app.repositories.conversation_repository import ConversationRepository, get_conversation_repository


def _supported_kwargs(method, payload: dict[str, Any]) -> dict[str, Any]:
    """
    按方法签名裁剪参数，兼容旧测试桩。

    一些旧测试替身方法参数不完整，这里按签名裁剪后再调用，
    可以避免 service 层因为“多传了一个新参数”而把兼容性打断。
    """
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return payload
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return payload
    return {key: value for key, value in payload.items() if key in parameters}


async def _invoke_repository(method, **payload):
    """统一调用 conversation repository。"""
    return await method(**_supported_kwargs(method, payload))


def _repository(repository: ConversationRepository | None = None) -> ConversationRepository:
    # 允许显式注入 repository，主要用于测试；默认走单例。
    return repository or get_conversation_repository()


async def ensure_conversation_schema(repository: ConversationRepository | None = None) -> None:
    """初始化 conversations/messages 表。"""
    await _repository(repository).ensure_schema()


async def get_previous_delivery(*, user_id: str, conversation_id: str,
                                repository: ConversationRepository | None = None) -> dict | None:
    """【研究交付】【上一轮】参数为用户和对话；返回按提问顺序选取的已保存结果。"""
    return await _repository(repository).get_previous_delivery(user_id=user_id, conversation_id=conversation_id)


async def list_conversations(
    *,
    user_id: str = "anonymous",
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "updated_at",
    repository: ConversationRepository | None = None,
) -> list[dict[str, Any]]:
    """列出会话。"""
    return await _invoke_repository(
        _repository(repository).list_conversations,
        user_id=user_id,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
    )


async def get_conversation(
    conversation_id: str,
    *,
    user_id: str = "anonymous",
    repository: ConversationRepository | None = None,
) -> dict[str, Any] | None:
    """读取单个会话详情。"""
    return await _invoke_repository(
        _repository(repository).get_conversation,
        user_id=user_id,
        conversation_id=conversation_id,
    )


async def find_conversation_by_session_id(
    session_id: str,
    *,
    user_id: str = "anonymous",
    repository: ConversationRepository | None = None,
) -> dict[str, Any] | None:
    """按 session_id 反查会话元数据。"""
    return await _invoke_repository(
        _repository(repository).find_conversation_by_session_id,
        user_id=user_id,
        session_id=session_id,
    )


async def create_conversation(
    conversation_id: str,
    title: str = "新对话",
    associated_papers: list[str] | None = None,
    *,
    user_id: str = "anonymous",
    session_id: str | None = None,
    repository: ConversationRepository | None = None,
) -> dict[str, Any]:
    """创建会话记录。"""
    return await _invoke_repository(
        _repository(repository).create_conversation,
        user_id=user_id,
        conversation_id=conversation_id,
        title=title,
        associated_papers=associated_papers,
        session_id=session_id,
    )


async def update_conversation(
    conversation_id: str,
    *,
    user_id: str = "anonymous",
    title: str | None = None,
    is_pinned: bool | None = None,
    tags: list[str] | None = None,
    associated_papers: list[str] | None = None,
    session_id: str | None = None,
    repository: ConversationRepository | None = None,
) -> bool:
    """更新会话元数据。"""
    return bool(
        await _invoke_repository(
            _repository(repository).update_conversation,
            user_id=user_id,
            conversation_id=conversation_id,
            title=title,
            is_pinned=is_pinned,
            tags=tags,
            associated_papers=associated_papers,
            session_id=session_id,
        )
    )


async def delete_conversation(
    conversation_id: str,
    *,
    user_id: str = "anonymous",
    repository: ConversationRepository | None = None,
) -> bool:
    """删除会话。"""
    return bool(
        await _invoke_repository(
            _repository(repository).delete_conversation,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    )


async def add_message(
    conversation_id: str,
    message_id: str,
    role: str,
    content: str,
    *,
    user_id: str = "anonymous",
    title: str | None = None,
    tone: str | None = None,
    render_mode: str | None = None,
    metadata: dict[str, Any] | None = None,
    repository: ConversationRepository | None = None,
) -> dict[str, Any]:
    """向会话追加消息。"""
    return await _invoke_repository(
        _repository(repository).add_message,
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        role=role,
        content=content,
        title=title,
        tone=tone,
        render_mode=render_mode,
        metadata=metadata,
    )


async def get_assistant_message_by_run_id(
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    repository: ConversationRepository | None = None,
) -> dict[str, Any] | None:
    return await _invoke_repository(
        _repository(repository).get_assistant_message_by_run_id,
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
    )


async def update_assistant_message_by_run_id(
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    content: str | None = None,
    render_mode: str | None = None,
    metadata: dict[str, Any] | None = None,
    repository: ConversationRepository | None = None,
) -> dict[str, Any] | None:
    return await _invoke_repository(
        _repository(repository).update_assistant_message_by_run_id,
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        content=content,
        render_mode=render_mode,
        metadata=metadata,
    )


async def set_messages(
    conversation_id: str,
    messages: list[dict[str, Any]],
    *,
    user_id: str = "anonymous",
    session_id: str | None = None,
    repository: ConversationRepository | None = None,
) -> bool:
    """批量覆盖会话消息。"""
    return bool(
        await _invoke_repository(
            _repository(repository).set_messages,
            user_id=user_id,
            conversation_id=conversation_id,
            messages=messages,
            session_id=session_id,
        )
    )


async def add_paper_to_conversation(
    conversation_id: str,
    paper_id: str,
    *,
    user_id: str = "anonymous",
    repository: ConversationRepository | None = None,
) -> bool:
    """为会话绑定论文。"""
    return bool(
        await _invoke_repository(
            _repository(repository).add_paper_to_conversation,
            user_id=user_id,
            conversation_id=conversation_id,
            paper_id=paper_id,
        )
    )


async def set_associated_papers(
    conversation_id: str,
    paper_ids: list[str],
    *,
    user_id: str = "anonymous",
    repository: ConversationRepository | None = None,
) -> bool:
    """
    一次性写入 conversation 的完整论文集合。

    Args:
        conversation_id: 前端展示层对话 ID
        paper_ids: 已完成 owner 校验的完整论文 ID 集合
        user_id: 当前登录用户 ID
        repository: 可替换的 conversation 仓储

    Returns:
        bool: owner 匹配的 conversation 更新成功时为 True
    """
    normalized = list(dict.fromkeys(
        str(paper_id or "").strip()
        for paper_id in paper_ids
        if str(paper_id or "").strip()
    ))
    return bool(
        await _invoke_repository(
            _repository(repository).update_conversation,
            user_id=user_id,
            conversation_id=conversation_id,
            title=None,
            is_pinned=None,
            tags=None,
            associated_papers=normalized,
            session_id=None,
        )
    )


async def remove_paper_from_conversation(
    conversation_id: str,
    paper_id: str,
    *,
    user_id: str = "anonymous",
    repository: ConversationRepository | None = None,
) -> bool:
    """从会话解绑论文。"""
    return bool(
        await _invoke_repository(
            _repository(repository).remove_paper_from_conversation,
            user_id=user_id,
            conversation_id=conversation_id,
            paper_id=paper_id,
        )
    )


async def remove_paper_references(
    paper_id: str,
    *,
    user_id: str = "anonymous",
    repository: ConversationRepository | None = None,
) -> int:
    """批量清理所有会话里的 `associated_papers` 引用。"""
    return int(
        await _invoke_repository(
            _repository(repository).remove_paper_references,
            user_id=user_id,
            paper_id=paper_id,
        )
    )


class ConversationDB:
    """
    仅供测试和旧调用方使用的同步兼容包装层。

    新代码应优先使用 async service 接口；这个类主要用于平滑过渡，
    避免旧同步代码一次性全部迁移。
    """

    def __init__(self, repository: ConversationRepository | None = None) -> None:
        self._repository = repository or get_conversation_repository()
        asyncio.run(ensure_conversation_schema(self._repository))

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated_at",
        user_id: str = "anonymous",
    ) -> list[dict[str, Any]]:
        return asyncio.run(
            list_conversations(
                user_id=user_id,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                repository=self._repository,
            )
        )

    def get_conversation(self, conversation_id: str, user_id: str = "anonymous") -> dict[str, Any] | None:
        return asyncio.run(get_conversation(conversation_id, user_id=user_id, repository=self._repository))

    def create_conversation(
        self,
        conversation_id: str,
        title: str = "新对话",
        associated_papers: list[str] | None = None,
        user_id: str = "anonymous",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            create_conversation(
                conversation_id,
                title=title,
                associated_papers=associated_papers,
                user_id=user_id,
                session_id=session_id,
                repository=self._repository,
            )
        )

    def update_conversation(
        self,
        conversation_id: str,
        *,
        user_id: str = "anonymous",
        title: str | None = None,
        is_pinned: bool | None = None,
        tags: list[str] | None = None,
        associated_papers: list[str] | None = None,
        session_id: str | None = None,
    ) -> bool:
        return asyncio.run(
            update_conversation(
                conversation_id,
                user_id=user_id,
                title=title,
                is_pinned=is_pinned,
                tags=tags,
                associated_papers=associated_papers,
                session_id=session_id,
                repository=self._repository,
            )
        )

    def delete_conversation(self, conversation_id: str, user_id: str = "anonymous") -> bool:
        return asyncio.run(delete_conversation(conversation_id, user_id=user_id, repository=self._repository))

    def add_message(
        self,
        conversation_id: str,
        message_id: str,
        role: str,
        content: str,
        title: str | None = None,
        tone: str | None = None,
        render_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str = "anonymous",
    ) -> dict[str, Any]:
        return asyncio.run(
            add_message(
                conversation_id,
                message_id,
                role,
                content,
                user_id=user_id,
                title=title,
                tone=tone,
                render_mode=render_mode,
                metadata=metadata,
                repository=self._repository,
            )
        )

    def set_messages(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        *,
        user_id: str = "anonymous",
        session_id: str | None = None,
    ) -> bool:
        return asyncio.run(
            set_messages(
                conversation_id,
                messages,
                user_id=user_id,
                session_id=session_id,
                repository=self._repository,
            )
        )

    def add_paper_to_conversation(self, conversation_id: str, paper_id: str, user_id: str = "anonymous") -> bool:
        return asyncio.run(
            add_paper_to_conversation(
                conversation_id,
                paper_id,
                user_id=user_id,
                repository=self._repository,
            )
        )

    def remove_paper_from_conversation(self, conversation_id: str, paper_id: str, user_id: str = "anonymous") -> bool:
        return asyncio.run(
            remove_paper_from_conversation(
                conversation_id,
                paper_id,
                user_id=user_id,
                repository=self._repository,
            )
        )

    def remove_paper_references(self, paper_id: str, user_id: str = "anonymous") -> int:
        return asyncio.run(remove_paper_references(paper_id, user_id=user_id, repository=self._repository))


def get_conversation_db() -> ConversationDB:
    """返回同步兼容包装层。"""
    return ConversationDB()


def close_conversation_db() -> None:
    """兼容旧接口；async 主链不再维护后台线程。"""
    return None
