"""
研究类后台任务 handler。

这类 handler 本身很薄：只负责把后台任务语义桥接到主图调用。
"""

from __future__ import annotations

from app.agents.agent import INTENT_DEEP_RESEARCH
from app.api.facades.graph_facade import invoke_graph, research_response_payload
from app.services import task_service


async def run_deep_research_task(task_run: dict) -> dict:
    """执行 deep research 任务。"""
    task_id = task_run["task_id"]
    payload = task_run.get("payload", {})
    query = (payload.get("query") or "").strip()
    session_id = payload.get("session_id")
    if not query:
        raise ValueError("deep research 任务缺少 query")

    await task_service.report_progress(task_id, 10, "开始执行 deep research")
    result, resolved_session_id = await invoke_graph(
        query,
        {"intent": INTENT_DEEP_RESEARCH, "session_id": session_id},
    )
    await task_service.report_progress(task_id, 100, "deep research 执行完成")
    return research_response_payload(result, resolved_session_id)
