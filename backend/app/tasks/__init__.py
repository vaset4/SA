"""
后台任务 handler 注册表。

worker_main 通过这里把 `task_type` 分发到具体 handler，是后台任务体系里的核心路由表。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.tasks.graph_task import run_lightrag_build_task
from app.tasks.paper_analysis_task import run_paper_analysis_task
from app.tasks.paper_delete_task import run_paper_delete_task
from app.tasks.research_task import run_deep_research_task
from app.tasks.upload_task import run_upload_ingest_task

TaskHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

TASK_HANDLERS: dict[str, TaskHandler] = {
    "upload_ingest": run_upload_ingest_task,
    "paper_analysis": run_paper_analysis_task,
    "paper_delete": run_paper_delete_task,
    "lightrag_build": run_lightrag_build_task,
    "deep_research": run_deep_research_task,
}


def get_task_handler(task_type: str) -> TaskHandler:
    """返回任务类型对应的 handler。"""
    try:
        return TASK_HANDLERS[task_type]
    except KeyError as exc:
        raise LookupError(f"不支持的任务类型: {task_type}") from exc


def supported_task_types() -> list[str]:
    """返回已注册的任务类型列表。"""
    return sorted(TASK_HANDLERS)
