"""
轻量内嵌 worker 调度器。

当系统没有独立常驻 worker 时，这里会在当前事件循环里按需拉起一个轻量 worker。
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from app.workers.worker_main import run_worker_loop

_running_workers: dict[tuple[str, ...], asyncio.Task[int]] = {}
_stop_events: dict[tuple[str, ...], asyncio.Event] = {}
_wake_events: dict[tuple[str, ...], asyncio.Event] = {}
logger = logging.getLogger(__name__)


def _task_key(task_types: list[str] | None) -> tuple[str, ...]:
    """将任务类型列表转为注册键；返回稳定元组。"""
    return tuple(sorted(task_types or ()))


async def _consume_pending(key: tuple[str, ...], stop_event: asyncio.Event, wake_event: asyncio.Event,
                           *, stop_when_idle: bool) -> int:
    """【后台任务】【接续消费】每轮消费后检查新入队信号；返回本 worker 完成的任务数。"""
    processed = 0
    worker_name = "embedded-worker" if not key else f"embedded-{'-'.join(key)}"
    while not stop_event.is_set():
        wake_event.clear()
        processed += await run_worker_loop(worker_id=worker_name, stop_when_idle=stop_when_idle,
            task_types=list(key) or None, stop_event=stop_event)
        # 1. 【后台任务】【空闲退出】判空期间若有新受理，必须再领一轮，不能丢掉最后一次唤醒
        if not wake_event.is_set():
            break
    return processed


def _cleanup(done_task: asyncio.Task[int], *, key: tuple[str, ...]) -> None:
    """回收已结束 worker 的注册和异常；旧回调不移除后来启动的 worker。"""
    if _running_workers.get(key) is done_task:
        _running_workers.pop(key, None)
        _stop_events.pop(key, None)
        _wake_events.pop(key, None)
    if done_task.cancelled():
        return
    exc = done_task.exception()
    if exc is not None:
        logger.error("【后台任务】【内嵌 Worker】异常退出，task_types=%s", list(key),
            exc_info=(type(exc), exc, exc.__traceback__))


def ensure_embedded_worker_running(*, task_types: list[str] | None = None,
                                   stop_when_idle: bool = True) -> asyncio.Task[int]:
    """确保指定任务类型至少有一个内嵌 worker 在运行。"""
    key = _task_key(task_types)
    current = _running_workers.get(key)
    if current is not None and not current.done():
        _wake_events[key].set()
        return current

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    wake_event = asyncio.Event()
    task = loop.create_task(_consume_pending(key, stop_event, wake_event, stop_when_idle=stop_when_idle))
    _running_workers[key] = task
    _stop_events[key] = stop_event
    _wake_events[key] = wake_event
    task.add_done_callback(partial(_cleanup, key=key))
    return task


async def close_embedded_workers() -> None:
    """停止领取新任务，并在数据库关闭前等待当前持租任务完成。"""
    for stop_event in list(_stop_events.values()):
        stop_event.set()
    pending = [task for task in _running_workers.values() if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    _running_workers.clear()
    _stop_events.clear()
    _wake_events.clear()


def active_worker_count() -> int:
    """返回仍在执行的内嵌 worker 数量，供生命周期验证。"""
    return sum(not task.done() for task in _running_workers.values())
