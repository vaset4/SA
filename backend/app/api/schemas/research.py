"""研究与图检索相关 schema。"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator

from app.api.schemas.request_limits import (
    ChatHistory,
    PaperIdList,
    UserQuery,
    validate_chat_history_content,
)


class ResearchRequest(BaseModel):
    query: UserQuery
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    research_mode: Optional[str] = None
    dialogue_mode: Optional[Literal["chat", "lookup", "research"]] = None
    paper_ids: PaperIdList = Field(default_factory=list)


class ResearchStreamRequest(BaseModel):
    query: UserQuery
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    intent: Optional[Literal["deep_research", "external_research"]] = None
    research_mode: Optional[str] = None
    dialogue_mode: Optional[Literal["chat", "lookup", "research"]] = None
    paper_ids: PaperIdList = Field(default_factory=list)


class ResearchResumeRequest(BaseModel):
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GraphQueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = 5
    library_id: Optional[str] = None
    project_id: Optional[str] = None

class CrossPaperQARequest(BaseModel):
    query: UserQuery
    paper_ids: PaperIdList = Field(default_factory=list)
    chat_history: ChatHistory = Field(default_factory=list)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    dialogue_mode: Optional[Literal["chat", "lookup", "research"]] = None

    @field_validator("chat_history")
    @classmethod
    def validate_history_content(cls, value: ChatHistory) -> ChatHistory:
        """在创建 run 前拒绝单条内容过长的历史消息。"""
        return validate_chat_history_content(value)


class SessionCreateRequest(BaseModel):
    topic: Optional[str] = None
