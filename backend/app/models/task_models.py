"""后台任务与事件模型。"""

from __future__ import annotations

from typing import Any, Dict, NotRequired, TypedDict


class TaskRun(TypedDict):
    """后台任务持久化视图。"""

    task_id: str
    user_id: NotRequired[str | None]
    task_type: str
    status: str
    progress: int
    message: str
    payload: Dict[str, Any]
    result: NotRequired[Any]
    error: NotRequired[str | None]
    cancel_requested: bool
    worker_id: NotRequired[str | None]
    lease_until: NotRequired[str | None]
    created_at: str
    updated_at: str
    started_at: NotRequired[str | None]
    finished_at: NotRequired[str | None]


class TaskEvent(TypedDict):
    """后台任务事件视图。"""

    event_id: str
    task_id: str
    event_type: str
    message: str
    payload: NotRequired[Dict[str, Any]]
    created_at: str
