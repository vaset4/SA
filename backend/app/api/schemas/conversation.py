"""前端会话管理相关 schema。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "新对话"
    associated_papers: list[Any] = Field(default_factory=list)
    session_id: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    is_pinned: Optional[bool] = None
    tags: Optional[list[Any]] = None
    associated_papers: Optional[list[Any]] = None
    session_id: Optional[str] = None


class AddMessageRequest(BaseModel):
    role: str
    content: str
    title: Optional[str] = None
    tone: Optional[str] = None
    render_mode: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SetMessagesRequest(BaseModel):
    messages: list[Any] = Field(default_factory=list)
    session_id: Optional[str] = None


class AddPaperRequest(BaseModel):
    paper_id: str
