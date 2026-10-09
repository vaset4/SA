"""【模型服务】【共享排队】实际 HTTP 的取消、额度和冷却边界。作者：xxx。"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any, Callable, Iterator
from uuid import uuid4

from app.repositories.provider_queue_repository import ProviderQueueRepository

logger = logging.getLogger(__name__)
POLL_SECONDS = 0.25


class ProviderRequestCancelled(RuntimeError):
    """运行取消，尚未发送的模型请求必须终止。"""


@dataclass(frozen=True)
class ProviderRequestContext:
    """线程共享取消信号，截止时间使用本进程 monotonic 时钟。"""

    cancel_event: threading.Event
    deadline: float | None = None
    run_id: str = ""
    on_event: Callable[[str, dict], None] | None = None


_context: ContextVar[ProviderRequestContext | None] = ContextVar("provider_request", default=None)


@contextmanager
def provider_request_context(*, cancel_event: threading.Event, deadline: float | None = None,
                             run_id: str = "", on_event=None) -> Iterator[ProviderRequestContext]:
    """参数为运行信号、截止时间、身份和事件回调；返回可传播至工作线程的上下文。"""
    context = ProviderRequestContext(cancel_event, deadline, run_id, on_event)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def check_provider_request_active() -> None:
    """无参数；取消或总期限届满时抛出明确异常，不发送后续请求。"""
    context = _context.get()
    if context is None:
        return
    if context.cancel_event.is_set():
        raise ProviderRequestCancelled("模型请求已取消")
    if context.deadline is not None and time.monotonic() >= context.deadline:
        raise TimeoutError("模型请求超过运行总期限")


def provider_request_timeout(timeout: float, *, check_active: bool = True) -> float:
    """参数为传输超时；返回不超过运行剩余期限的 HTTP 超时。"""
    if check_active:
        check_provider_request_active()
    context = _context.get()
    if context is None or context.deadline is None:
        return timeout
    return min(timeout, max(context.deadline - time.monotonic(), 0.001))


def wait_for_provider_retry(seconds: float) -> None:
    """参数为实际退避秒数；等待可由取消唤醒，且不延长运行总期限。"""
    check_provider_request_active()
    context = _context.get()
    if context is None:
        time.sleep(seconds)
        return
    if context.deadline is not None and time.monotonic() + seconds >= context.deadline:
        raise TimeoutError("模型服务等待超过运行剩余期限")
    context.cancel_event.wait(seconds)
    check_provider_request_active()


async def run_provider_thread(function, *args, **kwargs):
    """参数为同步调用及参数；返回结果，异步取消先唤醒并排空实际工作线程。"""
    context = _context.get()
    if context is None:
        with provider_request_context(cancel_event=threading.Event()):
            return await run_provider_thread(function, *args, **kwargs)
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        context.cancel_event.set()
        # 1. 【模型服务】【停止确认】已发送的同步 HTTP 仍归当前执行者所有
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    """参数为 Retry-After 和可选 UTC 时钟；返回合法秒数或 None，支持 HTTP 日期。"""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                return None
            seconds = (target - (now or datetime.now(timezone.utc))).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) and seconds >= 0 else None


def queue_policy_identity(settings: Any) -> dict:
    """参数为配置；返回可冻结的非敏感策略，不包含协调库地址或供应商凭证。"""
    return {
        "enabled": settings.llm_queue_enabled, "group": settings.llm_queue_group,
        "requests_per_window": settings.llm_queue_requests_per_window,
        "window_seconds": settings.llm_queue_window_seconds,
        "min_interval_seconds": max(settings.llm_queue_min_interval_seconds,
                                    settings.llm_queue_window_seconds / settings.llm_queue_requests_per_window),
        "cooldown_seconds": settings.llm_queue_cooldown_seconds,
        "http_max_retries": settings.llm_http_max_retries,
        "retry_backoff_seconds": settings.llm_retry_backoff_seconds,
        "retry_delay_cap_seconds": settings.llm_retry_delay_cap_seconds,
    }


def _emit(event: str, data: dict) -> None:
    """参数为事件和非敏感事实；向既有观察与运行状态回调发布同一事实。"""
    from app.middleware.trace_context import record_evidence_observation

    record_evidence_observation(event, data)
    logger.info("【模型服务】【共享排队】event=%s fields=%s", event, data)
    context = _context.get()
    if context is not None and context.on_event is not None:
        context.on_event(event, data)


class ProviderRequestQueue:
    """一个明确额度组的共享准入端口，仓储可替换用于确定性验证。"""

    def __init__(self, policy: dict, repository: ProviderQueueRepository | None) -> None:
        """参数为非敏感策略和协调仓储；只保存当前组配置。"""
        self.policy = dict(policy)
        self.repository = repository

    def acquire(self, identity: dict) -> dict:
        """参数为逻辑调用/尝试身份；FIFO 等待并返回实际准入事实，未发送取消不计 HTTP。"""
        check_provider_request_active()
        context = _context.get()
        facts = {**identity, "group": self.policy["group"],
                 "run_id": context.run_id if context else "", "request_id": str(uuid4())}
        if not self.policy["enabled"]:
            _emit("provider_queue_disabled", facts)
            return {**facts, "queue_wait_seconds": 0.0}
        assert self.repository is not None, "启用模型共享队列必须提供协调仓储"
        started = time.monotonic()
        queued = False
        request = {**facts, "call_id": str(facts.get("call_id") or ""),
                   "attempt": int(facts.get("attempt") or 1)}
        self.repository.enqueue(self.policy["group"], self._quota_policy(), request)
        try:
            while True:
                check_provider_request_active()
                result = self.repository.poll(
                    self.policy["group"], self._quota_policy(), facts["request_id"],
                    check_provider_request_active,
                )
                connection = result.pop("_connection", None)
                facts = {**facts, **result, "queue_wait_seconds": time.monotonic() - started}
                if result["admitted"]:
                    ticket = {**facts, "_connection": connection}
                    try:
                        _emit("provider_admitted", facts)
                    except BaseException:
                        self.release(ticket)
                        raise
                    return ticket
                if context and context.deadline is not None:
                    if time.monotonic() + result["estimated_wait_seconds"] >= context.deadline:
                        raise TimeoutError("模型服务额度等待超过运行剩余期限")
                if not queued:
                    _emit("provider_queued", facts)
                    queued = True
                wait_for_provider_retry(POLL_SECONDS)
        except BaseException as error:
            self.repository.cancel(facts["request_id"])
            _emit("provider_cancelled" if isinstance(error, ProviderRequestCancelled) else "provider_queue_failed",
                  {**facts, "error_type": type(error).__name__})
            raise

    def mark_started(self, ticket: dict) -> None:
        """参数为发送资格；在最后取消检查后持久计费，不在这个边界发布耗时回调。"""
        check_provider_request_active()
        if self.policy["enabled"]:
            assert self.repository is not None, "启用模型共享队列必须提供协调仓储"
            self.repository.mark_started(ticket["_connection"], ticket["request_id"])

    def release(self, ticket: dict, *, error: BaseException | None = None) -> None:
        """参数为发送资格；HTTP返回或未发取消均关闭资格连接，未发不计费。"""
        if self.policy["enabled"]:
            assert self.repository is not None, "启用模型共享队列必须提供协调仓储"
            try:
                self.repository.release(ticket["_connection"], ticket["request_id"], self._quota_policy())
            except Exception as release_error:
                if error is None:
                    raise
                logger.error("【模型服务】【资格释放】请求失败后的协调收尾失败", exc_info=True)
                error.add_note(f"模型发送资格收尾失败: {type(release_error).__name__}")

    def _quota_policy(self) -> dict:
        """无参数；返回必须在同组所有进程一致的额度策略，不包含每客户端重试预算。"""
        return {key: self.policy[key] for key in (
            "requests_per_window", "window_seconds", "min_interval_seconds", "cooldown_seconds",
        )}

    def cooldown(self, retry_after: str | None, identity: dict) -> float:
        """参数为供应商等待头和尝试身份；共享冷却并返回秒数，不额外创建重试预算。"""
        parsed = retry_after_seconds(retry_after)
        seconds = self.policy["cooldown_seconds"] if parsed is None else parsed
        if self.policy["enabled"]:
            assert self.repository is not None, "启用模型共享队列必须提供协调仓储"
            self.repository.cooldown(self.policy["group"], seconds)
        _emit("provider_cooldown", {**identity, "group": self.policy["group"],
              "cooldown_seconds": seconds, "source": "configured" if parsed is None else "retry_after"})
        return seconds


@lru_cache(maxsize=8)
def _configured_queue(database_url: str, policy_items: tuple) -> ProviderRequestQueue:
    """参数为显式协调库和冻结配置；返回初始化后的准入端口，不回退进程内队列。"""
    policy = dict(policy_items)
    repository = None
    if policy["enabled"]:
        if not database_url or not policy["group"]:
            raise ValueError("启用模型共享队列必须显式配置协调数据库和额度组")
        repository = ProviderQueueRepository(database_url)
        repository.initialize()
    return ProviderRequestQueue(policy, repository)


def get_provider_queue() -> ProviderRequestQueue:
    """无参数；使用明确共享配置装配队列，隔离业务数据库不会形成新额度桶。"""
    from app.config import get_settings

    settings = get_settings()
    return _configured_queue(settings.llm_queue_database_url, tuple(queue_policy_identity(settings).items()))
