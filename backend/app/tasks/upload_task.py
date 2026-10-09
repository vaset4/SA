"""持久上传任务入口：归属核对后交给恢复流水线，作者：xxx。"""
from __future__ import annotations

from app.services import task_service
from app.services.ingest.upload_pipeline import prepare_upload_task


def validate_upload_task(task_run: dict) -> str:
    """【上传】【任务归属】核对任务表 owner 和稳定论文身份，拒绝无法追溯的旧载荷。"""
    owner = task_service.get_upload_task_owner(task_run)
    payload = task_run.get("payload", {})
    paper_id = str(payload.get("paper_id") or "").strip()
    if not paper_id:
        raise ValueError(f"上传任务缺少稳定 paper_id，拒绝重试旧 payload: {task_run['task_id']}")
    if payload.get("staging_file_path"):
        raise ValueError("此旧上传任务没有持久恢复记录，请使用原文件重新上传；已保留旧文件供核对")
    return owner


async def run_upload_ingest_task(task_run: dict) -> dict:
    """消费持久上传任务，返回由 worker 事务发布的真实结果。"""
    owner = validate_upload_task(task_run)
    return await prepare_upload_task({**task_run, "user_id": owner})
