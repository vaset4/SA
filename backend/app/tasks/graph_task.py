"""旧图谱任务的显式迁移边界，作者：xxx。"""
from __future__ import annotations

from app.services.task_service import get_upload_task_owner


async def run_lightrag_build_task(task_run: dict) -> dict:
    """拒绝缺少发布版本的旧任务；参数为历史任务，无成功返回，不删除已有图谱。"""
    get_upload_task_owner(task_run)
    raise ValueError("旧图谱任务没有可核验的发布版本，请在论文详情重新发起可选图谱构建")
