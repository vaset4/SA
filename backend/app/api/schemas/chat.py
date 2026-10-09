"""对话相关 schema。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.api.schemas.request_limits import (
    ChatHistory,
    ContextSummary,
    PaperIdList,
    UserQuery,
    validate_chat_history_content,
)


class ChatRequest(BaseModel):
    query: UserQuery
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    paper_ids: PaperIdList = Field(default_factory=list)
    dialogue_mode: Optional[Literal["chat", "lookup", "research"]] = None
    chat_history: ChatHistory = Field(default_factory=list)
    context_summary: Optional[ContextSummary] = None

    @field_validator("chat_history")
    @classmethod
    def validate_history_content(cls, value: ChatHistory) -> ChatHistory:
        """在进入 graph 前拒绝单条内容过长的历史消息。"""
        return validate_chat_history_content(value)
