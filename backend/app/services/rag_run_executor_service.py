"""受管研究执行、跨进程停止信号及真实退出事实。作者：xxx，2026-10-04 12:00:00。"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from functools import partial
from typing import Any

from app.repositories.rag_run_repository import get_rag_run_repository
from app.services import rag_run_service
from app.services.provider_request_queue import provider_request_context

logger = logging.getLogger(__name__)
RunFactory = Callable[[], Awaitable[Any]]
TerminalCallback = Callable[[str, str], Awaitable[None]]
STOP_SIGNAL_POLL_SECONDS = 0.2


class RAGRunExecutorRegistry:
    """持有本进程同步和流式工作；数据库独占归属才是图执行许可。"""

    def __init__(self) -> None:
        """无参数；建立进程实例身份和任务表，不推断其他进程是否存活。"""
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._owners: dict[str, str] = {}
        self._instance_id = f"rag-executor-{uuid.uuid4().hex}"
        self._closing = False

    def start(self, *, user_id: str, run_id: str, timeout_seconds: float,
              runner_factory: RunFactory, terminal_callback: TerminalCallback | None = None) -> bool:
        """参数为已领取操作与工作工厂；返回是否新建本地任务，持久领取仍由任务核对。"""
        current = self._tasks.get(run_id)
        if current is not None and not current.done():
            return False
        if self._closing:
            raise RuntimeError("应用正在关闭，不能启动新的 RAG run")
        timeout = rag_run_service.execution_timeout_seconds(timeout_seconds)
        task = asyncio.create_task(self._run_guarded(
            user_id=user_id, run_id=run_id, timeout_seconds=timeout,
            runner_factory=runner_factory, terminal_callback=terminal_callback,
        ), name=f"rag-run:{run_id}")
        self._tasks[run_id] = task
        self._owners[run_id] = user_id
        task.add_done_callback(partial(self._discard, run_id))
        return True

    async def execute(self, *, user_id: str, run_id: str, timeout_seconds: float,
                      runner_factory: RunFactory, terminal_callback: TerminalCallback | None = None) -> Any:
        """参数为同步操作与工作工厂；等待受管结果，HTTP 任务取消时排空已拥有工作。"""
        if self._closing:
            await get_rag_run_repository().settle_session_operation(
                user_id=user_id, run_id=run_id, status="cancelled", error="应用关闭期间未启动研究运行",
            )
            raise RuntimeError("应用正在关闭，不能启动新的 RAG run")
        if not self.start(user_id=user_id, run_id=run_id, timeout_seconds=timeout_seconds,
                          runner_factory=runner_factory, terminal_callback=terminal_callback):
            raise ValueError("该操作已由本进程执行，请查询原运行")
        task = self._tasks[run_id]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await self.cancel(user_id=user_id, run_id=run_id)
            raise

    async def cancel(self, *, user_id: str, run_id: str) -> dict[str, Any]:
        """参数为 owner 和操作；本地等待退出，远端仅返回正在停止的持久信号。"""
        run = await rag_run_service.request_cancel(user_id=user_id, run_id=run_id)
        task = self._tasks.get(run_id)
        if task is not None and not task.done() and self._owners.get(run_id) == user_id:
            # 1. 【研究执行】【取消排空】持久信号由唯一监测任务消费，重复请求不再打断清理
            result = (await asyncio.shield(asyncio.gather(task, return_exceptions=True)))[0]
            if isinstance(result, Exception):
                raise result
            return await rag_run_service.get_run(user_id=user_id, run_id=run_id) or run
        return run

    async def close(self) -> None:
        """无参数；关闭前取消并排空本进程任务，停止事实落库异常向生命周期上抛。"""
        self._closing = True
        pending = [self.cancel(user_id=self._owners[run_id], run_id=run_id)
                   for run_id, task in self._tasks.items() if not task.done()]
        results = await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._owners.clear()
        for result in results:
            if isinstance(result, Exception):
                raise result

    def active_count(self) -> int:
        """无参数；返回本进程仍在运行或收尾的任务数。"""
        return sum(not task.done() for task in self._tasks.values())

    async def _run_guarded(self, *, user_id: str, run_id: str, timeout_seconds: float | None,
                           runner_factory: RunFactory, terminal_callback: TerminalCallback | None) -> Any:
        """参数为受管执行上下文；领取、监督、排空后记录停止并收敛真实终态。"""
        repo = get_rag_run_repository()
        claim = asyncio.create_task(repo.claim_execution(
            user_id=user_id, run_id=run_id, owner_instance_id=self._instance_id,
        ))
        failure: BaseException | None = None
        result: Any = None
        try:
            owned = await asyncio.shield(claim)
        except asyncio.CancelledError as exc:
            owned = await claim
            failure = exc
        if owned is None:
            raise RuntimeError("研究执行归属未取得，不能启动或终结其他执行器的任务")
        try:
            if failure is not None or owned.get("cancel_requested"):
                raise asyncio.CancelledError("研究运行已请求停止")
            deadline = owned.get("deadline_at")
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline)
            if deadline is not None:
                remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                timeout_seconds = remaining if timeout_seconds is None else min(timeout_seconds, remaining)
            if timeout_seconds is not None and timeout_seconds <= 0:
                raise TimeoutError("研究运行超过执行期限")
            result = await self._supervise(user_id=user_id, run_id=run_id,
                                           timeout_seconds=timeout_seconds, runner_factory=runner_factory)
        except (Exception, asyncio.CancelledError) as exc:
            failure = exc
        finally:
            # 2. 【研究执行】【退出确认】graph、文件读取与请求占用退出后才能释放 thread
            stopped = await repo.record_execution_stopped(
                user_id=user_id, run_id=run_id, owner_instance_id=self._instance_id,
            )
            if stopped is None:
                raise RuntimeError("研究执行停止事实保存失败，当前线程不可重入")
        if failure is not None:
            status = self._failure_status(failure)
            error = str(failure) or {"cancelled": "研究运行已取消", "timed_out": "研究运行超过执行期限"}.get(
                status, "研究运行失败",
            )
            await self._notify_terminal(callback=terminal_callback, run_id=run_id, status=status, error=error)
            await rag_run_service.finish_run(user_id=user_id, run_id=run_id, status=status, error=error)
            raise failure
        return result

    async def _supervise(self, *, user_id: str, run_id: str, timeout_seconds: float | None,
                         runner_factory: RunFactory) -> Any:
        """参数为工作和期限；同时监听持久停止信号，任何退出都等待被拥有的子任务。"""
        cancel_event = threading.Event()
        loop = asyncio.get_running_loop()
        on_event = partial(self._provider_event, loop=loop, user_id=user_id, run_id=run_id)
        # 1. 【研究执行】【期限传递】无总时限同时传给监督器和请求线程，取消仍独立生效
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        with provider_request_context(cancel_event=cancel_event, deadline=deadline,
                                      run_id=run_id, on_event=on_event):
            work = asyncio.ensure_future(runner_factory())
        monitor = asyncio.create_task(self._monitor_stop(user_id=user_id, run_id=run_id),
                                      name=f"rag-stop:{run_id}")
        try:
            done, _ = await asyncio.wait({work, monitor}, timeout=timeout_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise TimeoutError("研究运行超过执行期限")
            if monitor in done:
                await monitor
            return await work
        finally:
            # 2. 【研究执行】【排队取消】先唤醒发送线程，再取消并排空异步工作
            cancel_event.set()
            for task in (work, monitor):
                if not task.done():
                    task.cancel()
            settled = await asyncio.gather(work, monitor, return_exceptions=True)
            for failure in settled:
                if isinstance(failure, Exception):
                    raise failure

    @staticmethod
    def _provider_event(event: str, data: dict[str, Any], *, loop: asyncio.AbstractEventLoop,
                        user_id: str, run_id: str) -> None:
        """参数为发送线程事件和运行归属；等待持久化，保证退出前无遗留事件任务。"""
        if event == "provider_queue_disabled":
            return
        pending = asyncio.run_coroutine_threadsafe(rag_run_service.append_event(
            user_id=user_id, run_id=run_id, event_type="event",
            payload={"type": "event", "stage": event, **data},
        ), loop)
        pending.result()

    async def _monitor_stop(self, *, user_id: str, run_id: str) -> None:
        """参数为受管操作；即使模型仍在等待也能响应其他进程写入的取消与期限。"""
        while True:
            run = await rag_run_service.get_run(user_id=user_id, run_id=run_id)
            if run is None:
                raise LookupError("受管研究操作已不存在")
            deadline = run.get("deadline_at")
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline)
            if deadline is not None and deadline <= datetime.now(timezone.utc):
                raise TimeoutError("研究运行超过执行期限")
            if run.get("cancel_requested"):
                raise asyncio.CancelledError("研究运行已取消")
            await asyncio.sleep(STOP_SIGNAL_POLL_SECONDS)

    @staticmethod
    def _failure_status(error: BaseException) -> str:
        """参数为实际工作失败；返回既有运行词表，保留来源权限等真实错误为 failed。"""
        if isinstance(error, asyncio.CancelledError):
            return rag_run_service.STATUS_CANCELLED
        if isinstance(error, TimeoutError) or getattr(error, "status_code", None) == 504:
            return rag_run_service.STATUS_TIMED_OUT
        return rag_run_service.STATUS_FAILED

    @staticmethod
    async def _notify_terminal(*, callback: TerminalCallback | None, run_id: str,
                               status: str, error: str) -> None:
        """参数为业务收尾端口与真实失败；事务失败直接传播，不再吞错后宣告成功。"""
        if callback is not None:
            await callback(status, error)

    def _discard(self, run_id: str, task: asyncio.Task[Any]) -> None:
        """参数为完成任务；检取异常并移除本地引用，不将缺少本地任务当作远端停止。"""
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)
            self._owners.pop(run_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("【研究执行】【受管退出】运行失败，run_id=%s", run_id,
                         exc_info=(type(exc), exc, exc.__traceback__))


_registry = RAGRunExecutorRegistry()


def get_rag_run_executor_registry() -> RAGRunExecutorRegistry:
    """无参数；返回本进程的受管研究执行器。"""
    return _registry
