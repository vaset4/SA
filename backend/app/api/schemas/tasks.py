"""后台任务相关 schema。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ResearchTaskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class LightRAGTaskRequest(BaseModel):
    """图谱只消费服务端核验原文，调用方不能替换论文内容或指定另一文献库。"""

    model_config = ConfigDict(extra="forbid")
    paper_id: str


class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    progress: int = 0
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    cancel_requested: bool = False
    worker_id: Optional[str] = None
    lease_until: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> "TaskResponse":
        return cls(**task)


class TaskEventResponse(BaseModel):
    event_id: str
    task_id: str
    event_type: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "TaskEventResponse":
        return cls(**event)
