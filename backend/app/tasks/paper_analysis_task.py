"""手动论文分析后台任务 handler。"""

from __future__ import annotations

from app.services import task_service
from app.services.paper_analysis_service import analyze_paper_by_id


async def run_paper_analysis_task(task_run: dict) -> dict:
    task_id = task_run["task_id"]
    payload = task_run.get("payload", {})
    paper_id = str(payload.get("paper_id") or "").strip()
    user_id = str(payload.get("user_id") or "anonymous").strip() or "anonymous"
    if not paper_id:
        raise ValueError("paper_id 不能为空")
    await task_service.report_progress(task_id, 15, "论文 AI 分析任务已开始", {"paper_id": paper_id})
    result = await analyze_paper_by_id(paper_id, user_id=user_id)
    await task_service.report_progress(
        task_id,
        100,
        "论文 AI 分析任务执行完成",
        {"paper_id": paper_id, "analysis_status": result.get("analysis_status")},
    )
    return result
