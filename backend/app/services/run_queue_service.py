"""
后台 worker 队列服务。

这是 worker 侧与任务存储交互的统一入口：claim、heartbeat、cancel 检查、进度回写都从这里过。
"""

from __future__ import annotations

import asyncio
from typing import Any

import app.services.paper_db as paper_db
from app.services import task_service
from app.models.upload_models import UploadPausedError

DEFAULT_LEASE_SECONDS = 300


class RunQueueService:
    """封装 worker 侧的任务领取、租约和完成语义。"""

    def __init__(self, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> None:
        self.lease_seconds = lease_seconds

    async def claim_next_task(
        self,
        worker_id: str,
        task_types: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """领取下一条可执行任务。"""
        task = await asyncio.to_thread(
            paper_db.claim_next_task,
            worker_id,
            self.lease_seconds,
            task_types,
        )
        if task is not None:
            # 成功领取后追加结构化事件，便于追踪任务究竟被哪个 worker 处理。
            await task_service.append_event(
                task["task_id"],
                "claimed",
                f"worker {worker_id} 已领取任务",
                {"worker_id": worker_id},
            )
        return task

    async def heartbeat(self, task_id: str, worker_id: str) -> dict[str, Any] | None:
        """续租任务执行 lease。"""
        return await asyncio.to_thread(
            paper_db.heartbeat_task,
            task_id,
            worker_id,
            self.lease_seconds,
        )

    async def is_cancel_requested(self, task_id: str) -> bool:
        """检查任务是否已经收到取消请求。"""
        return bool(await asyncio.to_thread(paper_db.is_task_cancel_requested, task_id))

    async def update_progress(
        self,
        task_id: str,
        progress: int,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """更新任务进度。"""
        return await task_service.report_progress(task_id, progress, message, payload)

    async def mark_completed(
        self,
        task_id: str,
        result: Any,
        *,
        worker_id: str,
        message: str = "任务执行完成",
    ) -> dict[str, Any] | None:
        """由仍持租的 worker 标记任务完成；失租时返回 None。"""
        return await task_service.complete_task(task_id, result, worker_id=worker_id, message=message)

    async def mark_failed(
        self,
        task_id: str,
        error: str,
        *,
        worker_id: str,
        message: str = "任务执行失败",
    ) -> dict[str, Any] | None:
        """由仍持租的 worker 标记任务失败；失租时返回 None。"""
        return await task_service.fail_task(task_id, error, worker_id=worker_id, message=message)

    async def mark_paused(self, error: UploadPausedError) -> dict[str, Any]:
        """把上传暂停与原阶段原因持久保存，worker 不再自动领取该任务。"""
        from app.repositories.upload_repository import get_upload_repository

        if error.execution is None:
            raise ValueError("上传暂停缺少持租执行身份")
        return await get_upload_repository().pause(error.execution, error.failure)

    async def mark_cancelled(
        self,
        task_id: str,
        *,
        worker_id: str,
        message: str = "任务已取消",
    ) -> dict[str, Any] | None:
        """由仍持租的 worker 标记任务取消；失租时返回 None。"""
        return await task_service.mark_cancelled(task_id, worker_id=worker_id, message=message)
