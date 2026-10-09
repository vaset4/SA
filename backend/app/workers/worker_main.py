"""
后台 worker 主循环。

这是后台任务体系的通用执行外壳：claim -> dispatch handler -> complete/fail/cancel。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import app.middleware.observability as observability
from app.services.run_queue_service import RunQueueService
from app.tasks import get_task_handler
from app.models.upload_models import UploadCancelledError, UploadLeaseLostError, UploadPausedError

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
HEARTBEAT_INTERVAL_DIVISOR = 3

logger = logging.getLogger(__name__)


class LeaseLostError(RuntimeError):
    """表示当前 worker 已无法证明自己仍持有任务租约。"""


async def _heartbeat_while_running(
    queue: RunQueueService,
    *,
    task_id: str,
    worker_id: str,
) -> None:
    """按租约时长三分之一续租；heartbeat 返回空时抛出失租错误。"""
    heartbeat_interval = queue.lease_seconds / HEARTBEAT_INTERVAL_DIVISOR
    while True:
        await asyncio.sleep(heartbeat_interval)
        lease = await queue.heartbeat(task_id, worker_id)
        if lease is None:
            raise LeaseLostError(f"任务租约已失效: {task_id}")


async def _run_with_heartbeat(
    handler: Callable[[dict], Awaitable[Any]],
    task_run: dict,
    *,
    queue: RunQueueService,
    worker_id: str,
) -> tuple[Any | None, Exception | None]:
    """并发管理 handler 与 heartbeat；返回 handler 结果或导致失租的异常。"""
    handler_task = asyncio.create_task(handler(task_run))
    heartbeat_task = asyncio.create_task(
        _heartbeat_while_running(queue, task_id=task_run["task_id"], worker_id=worker_id)
    )
    try:
        done, _ = await asyncio.wait(
            {handler_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            try:
                await heartbeat_task
            except Exception as exc:
                return None, exc
            return None, LeaseLostError(f"任务 heartbeat 意外结束: {task_run['task_id']}")
        return await handler_task, None
    finally:
        for managed_task in (handler_task, heartbeat_task):
            if not managed_task.done():
                managed_task.cancel()
        await asyncio.gather(handler_task, heartbeat_task, return_exceptions=True)


def _record_execution_event(
    event_type: str,
    *,
    started_at: float,
    task_type: str,
    task_id: str,
    result_count: int = 0,
) -> None:
    """记录 worker 执行观测事件；参数标识事件、任务与结果数，无返回值。"""
    observability.record_timed_event(
        event_type,
        "worker_main",
        started_at,
        query=f"{task_type}:{task_id}",
        result_count=result_count,
    )


def _lease_lost_result(
    task_id: str,
    *,
    worker_id: str,
    task_type: str,
    started_at: float,
    error: Exception,
) -> dict:
    """记录失租并返回非终态 lease_lost 结果，不写任务终态。"""
    logger.error(
        "【后台任务】【租约丢失】停止任务执行，task_id=%s worker_id=%s error=%s",
        task_id,
        worker_id,
        error,
    )
    _record_execution_event(
        "task_execution_lease_lost",
        started_at=started_at,
        task_type=task_type,
        task_id=task_id,
    )
    return {"task_id": task_id, "status": "lease_lost", "error": str(error)}


def _terminal_result(
    result: dict | None,
    *,
    event_type: str,
    terminal_name: str,
    task_id: str,
    worker_id: str,
    task_type: str,
    started_at: float,
    result_count: int = 0,
) -> dict:
    """返回 CAS 终态并记录事件；CAS 失败时转换成可见的 lease_lost 结果。"""
    if result is None:
        error = LeaseLostError(f"{terminal_name}终态 CAS 失败: {task_id}")
        return _lease_lost_result(
            task_id,
            worker_id=worker_id,
            task_type=task_type,
            started_at=started_at,
            error=error,
        )
    _record_execution_event(
        event_type,
        started_at=started_at,
        task_type=task_type,
        task_id=task_id,
        result_count=result_count,
    )
    return result


async def _execute_claimed_task(queue: RunQueueService, task: dict, *, worker_id: str) -> dict | None:
    """在通用租约保护下执行并提交，所有上传发布仍经过完成边界。"""
    task_id = task["task_id"]
    if task.get("cancel_requested"):
        return await queue.mark_cancelled(task_id, worker_id=worker_id, message="任务在执行前已取消")
    handler = get_task_handler(task["task_type"])
    result, lease_error = await _run_with_heartbeat(handler, task, queue=queue, worker_id=worker_id)
    if lease_error is not None:
        raise LeaseLostError(str(lease_error)) from lease_error
    if await queue.is_cancel_requested(task_id):
        return await queue.mark_cancelled(task_id, worker_id=worker_id, message="任务执行后检测到取消请求")
    return await queue.mark_completed(task_id, result, worker_id=worker_id)


async def execute_task(task_run: dict, *, worker_id: str, queue_service: RunQueueService | None = None) -> dict:
    """执行已领取任务；暂停、取消、失租和完成分别落入对应的持久边界。"""
    queue = queue_service or RunQueueService()
    task_id = task_run["task_id"]
    details = {"task_id": task_id, "worker_id": worker_id, "task_type": task_run["task_type"], "started_at": time.monotonic()}
    try:
        terminal = await _execute_claimed_task(queue, task_run, worker_id=worker_id)
    except (LeaseLostError, UploadLeaseLostError) as error:
        return _lease_lost_result(**details, error=error)
    except UploadPausedError as error:
        try:
            terminal = await queue.mark_paused(error)
        except UploadLeaseLostError as lease_error:
            return _lease_lost_result(**details, error=lease_error)
        except UploadCancelledError:
            terminal = await queue.mark_cancelled(task_id, worker_id=worker_id, message="暂停前检测到取消请求")
    except UploadCancelledError:
        terminal = await queue.mark_cancelled(task_id, worker_id=worker_id, message="上传已取消，保留已完成进度")
    except Exception as error:
        terminal = await queue.mark_failed(task_id, str(error), worker_id=worker_id)
    status = str(terminal.get("status")) if terminal is not None else "lease_lost"
    return _terminal_result(terminal, **details, event_type=f"task_execution_{status}",
        terminal_name=status, result_count=int(status == "completed"))


async def run_worker_loop(
    *,
    worker_id: str | None = None,
    queue_service: RunQueueService | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    stop_when_idle: bool = False,
    max_jobs: int | None = None,
    task_types: list[str] | None = None,
    stop_event: asyncio.Event | None = None,
) -> int:
    """
    轮询并执行后台任务。

    `stop_when_idle` 常用于内嵌 worker；独立 worker 则通常持续轮询。
    """
    queue = queue_service or RunQueueService()
    resolved_worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    processed = 0

    while True:
        if stop_event is not None and stop_event.is_set():
            return processed
        # 1. 【后台任务】【独立执行】每次领取使用独立身份，旧线程不能借同一 daemon 名称重新获得租约
        execution_worker_id = f"{resolved_worker_id}:{uuid.uuid4().hex}"
        task_run = await queue.claim_next_task(execution_worker_id, task_types)
        if task_run is None:
            if stop_when_idle:
                return processed
            await asyncio.sleep(poll_interval)
            continue

        await execute_task(task_run, worker_id=execution_worker_id, queue_service=queue)
        processed += 1
        if max_jobs is not None and processed >= max_jobs:
            return processed


async def main_async() -> int:
    """默认 worker 入口。"""
    return await run_worker_loop()


if __name__ == "__main__":
    asyncio.run(main_async())
