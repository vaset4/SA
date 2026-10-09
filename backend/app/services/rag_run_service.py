"""Service layer for durable user-visible RAG run lifecycle state."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.repositories.rag_run_repository import RAGRunRepository, get_rag_run_repository

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_TIMED_OUT = "timed_out"

TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_TIMED_OUT}
ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_RUNNING}
VALID_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
OPERATION_KINDS = frozenset({"research", "resume", "retry", "chat", "final_only", "persist_only"})

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    STATUS_QUEUED: TERMINAL_STATUSES | {STATUS_RUNNING},
    STATUS_RUNNING: TERMINAL_STATUSES,
    STATUS_TIMED_OUT: set(),
    STATUS_CANCELLED: set(),
    STATUS_FAILED: set(),
    STATUS_COMPLETED: set(),
}


class ActiveRAGRunConflict(ValueError):
    """表示同一 owner/thread 已有另一个 active run。"""

    def __init__(self, run: dict[str, Any]) -> None:
        """保存冲突 run，供 HTTP 边界返回稳定 run_id。"""
        self.run = run
        super().__init__("当前研究线程已有运行中的任务")


class IdempotencyConflict(ValueError):
    """同一用户动作标识被用于不同输入，不能重放另一项操作。"""


def operation_fingerprint(*, operation_kind: str, request_payload: dict, target_run_id: str | None = None) -> str:
    """参数为命令类型、原请求与目标；返回稳定摘要，不包括服务端生成的执行身份。"""
    canonical = json.dumps(
        {"version": 1, "operation_kind": operation_kind, "target_run_id": target_run_id,
         "request": request_payload}, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_same_operation(run: dict, *, operation_kind: str, request_payload: dict,
                          target_run_id: str | None = None) -> None:
    """参数为已有操作和重发请求；只有相同合同摘要允许重放，旧记录不冒充新合同。"""
    expected = operation_fingerprint(
        operation_kind=operation_kind, request_payload=request_payload, target_run_id=target_run_id,
    )
    if run.get("request_fingerprint") != expected:
        raise IdempotencyConflict("该请求标识已用于不同内容或历史操作，请勿复用")


async def create_operation(
    *, user_id: str, conversation_id: str | None, session_id: str | None,
    thread_id: str | None, idempotency_key: str, query: str, deadline_at: datetime | None,
    request_payload: dict, operation_kind: str = "chat", target_run_id: str | None = None,
    expected_delivery_revision: int | None = None, metadata: dict | None = None,
    session_command: dict | None = None, repository: RAGRunRepository | None = None,
) -> tuple[dict[str, Any], bool]:
    """参数为完整用户操作与可选 session 前提；返回唯一操作和是否取得新执行权。"""
    if operation_kind not in OPERATION_KINDS:
        raise ValueError("未知研究操作类型")
    if not str(user_id).strip() or not 8 <= len(idempotency_key) <= 128:
        raise ValueError("用户身份或请求标识无效")
    final_operation = operation_kind in {"final_only", "persist_only"}
    if final_operation != bool(target_run_id) or (final_operation and thread_id is not None):
        raise ValueError("最终结果操作必须指向目标，且不能占用 checkpoint thread")
    if not final_operation and not thread_id:
        raise ValueError("图执行必须绑定 checkpoint thread")
    if deadline_at is not None and deadline_at.tzinfo is None:
        raise ValueError("操作的执行期限必须带时区；无总时限请传 None")
    repo = _repository(repository)
    operation = {
        "run_id": _new_run_id(), "user_id": user_id, "conversation_id": conversation_id,
        "session_id": session_id, "thread_id": thread_id, "idempotency_key": idempotency_key,
        "query": query, "deadline_at": deadline_at, "operation_kind": operation_kind,
        "target_run_id": target_run_id, "expected_delivery_revision": expected_delivery_revision,
        "metadata": metadata or {}, "request_fingerprint": operation_fingerprint(
            operation_kind=operation_kind, request_payload=request_payload, target_run_id=target_run_id,
        ),
    }
    try:
        run, created = await repo.create_operation(operation=operation, session_command=session_command)
    except ValueError as exc:
        active = await repo.get_active_for_thread(user_id=user_id, thread_id=thread_id) if thread_id else None
        if active is not None:
            raise ActiveRAGRunConflict(active) from exc
        raise
    if run is not None:
        assert_same_operation(run, operation_kind=operation_kind,
                              request_payload=request_payload, target_run_id=target_run_id)
        return run, created
    active = None
    if target_run_id is not None:
        active = await repo.get_active_for_target(user_id=user_id, target_run_id=target_run_id)
    elif thread_id is not None:
        active = await repo.get_active_for_thread(user_id=user_id, thread_id=thread_id)
    if active is not None:
        raise ActiveRAGRunConflict(active)
    raise RuntimeError("操作领取冲突后未找到对应记录")


def _repository(repository: RAGRunRepository | None = None) -> RAGRunRepository:
    return repository or get_rag_run_repository()


def _new_run_id() -> str:
    return f"ragrun-{uuid.uuid4().hex[:12]}"


def _assert_transition(current: str, target: str) -> None:
    if target not in VALID_STATUSES:
        raise ValueError(f"invalid RAG run status: {target}")
    if current == target:
        return
    allowed = _ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(f"invalid RAG run transition: {current} -> {target}")


def execution_timeout_seconds(seconds: float) -> float | None:
    """参数为总时限秒数；返回正值或表示禁用的 None，负数明确拒绝。"""
    if seconds < 0:
        raise ValueError("RAG run timeout 不能小于 0")
    return None if seconds == 0 else seconds


def deadline_after(seconds: float) -> datetime | None:
    """参数为总时限秒数；返回 UTC 截止时间，0 返回 None，不制造远期截止日期。"""
    timeout = execution_timeout_seconds(seconds)
    return None if timeout is None else datetime.now(timezone.utc) + timedelta(seconds=timeout)


async def ensure_rag_run_schema(repository: RAGRunRepository | None = None) -> None:
    await _repository(repository).ensure_schema()


async def create_run(
    *,
    user_id: str,
    conversation_id: str | None,
    session_id: str | None,
    query: str,
    metadata: dict[str, Any] | None = None,
    thread_id: str | None = None,
    deadline_at: datetime | None = None,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any]:
    """创建普通 run；没有幂等键时保留既有同步入口语义。"""
    return await _repository(repository).create_run(
        run_id=_new_run_id(),
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
        query=query,
        metadata=metadata,
        thread_id=thread_id,
        deadline_at=deadline_at,
    )


async def create_or_get_stream_run(
    *,
    user_id: str,
    conversation_id: str | None,
    session_id: str | None,
    thread_id: str,
    idempotency_key: str,
    query: str,
    deadline_at: datetime | None,
    metadata: dict[str, Any] | None = None,
    repository: RAGRunRepository | None = None,
) -> tuple[dict[str, Any], bool]:
    """按幂等键创建/复用流式 run，并拒绝同 thread 的第二个 active run。"""
    normalized_key = str(idempotency_key or "").strip()
    normalized_thread = str(thread_id or "").strip()
    if not normalized_key or not normalized_thread:
        raise ValueError("idempotency_key 和 thread_id 不能为空")
    repo = _repository(repository)
    run, created = await repo.create_or_get_run(
        run_id=_new_run_id(),
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
        thread_id=normalized_thread,
        idempotency_key=normalized_key,
        query=query,
        deadline_at=deadline_at,
        metadata=metadata,
    )
    if run is not None:
        return run, created
    existing = await repo.get_by_idempotency_key(
        user_id=user_id,
        idempotency_key=normalized_key,
    )
    if existing is not None:
        return existing, False
    active = await repo.get_active_for_thread(user_id=user_id, thread_id=normalized_thread)
    if active is not None:
        raise ActiveRAGRunConflict(active)
    raise RuntimeError("RAG run 创建失败且未找到冲突记录")


async def update_status(
    *,
    user_id: str,
    run_id: str,
    status: str,
    request_id: str | None = None,
    session_id: str | None = None,
    answer: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any]:
    """迁移 run 状态；终态统一走带 terminal 事件的 CAS 写入。"""
    repo = _repository(repository)
    current = await repo.get_run(user_id=user_id, run_id=run_id)
    if current is None:
        raise LookupError("RAG run not found")
    _assert_transition(str(current.get("status") or ""), status)
    if status in TERMINAL_STATUSES:
        return await finish_run(
            user_id=user_id,
            run_id=run_id,
            status=status,
            request_id=request_id,
            session_id=session_id,
            answer=answer,
            error=error,
            metadata=metadata,
            repository=repo,
        )
    return await repo.update_run(
        run_id=run_id,
        user_id=user_id,
        status=status,
        request_id=request_id,
        session_id=session_id,
        answer=answer,
        error=error,
        metadata=metadata,
        completed=False,
        expected_status=str(current.get("status") or ""),
    )


async def append_event(
    *,
    user_id: str,
    run_id: str,
    event_type: str,
    payload: dict[str, Any],
    repository: RAGRunRepository | None = None,
) -> dict[str, Any]:
    """持久化一个可重放流事件，并返回数据库分配的 sequence。"""
    return await _repository(repository).append_event(
        user_id=user_id,
        run_id=run_id,
        event_type=event_type,
        payload=payload,
    )


async def finish_run(
    *,
    user_id: str,
    run_id: str,
    status: str,
    request_id: str | None = None,
    session_id: str | None = None,
    answer: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    terminal_payload: dict[str, Any] | None = None,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any]:
    """用 CAS 写唯一终态；并发收尾时保留最先落库的真实终态。"""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"RAG run 终态无效: {status}")
    repo = _repository(repository)
    current = await repo.get_run(user_id=user_id, run_id=run_id)
    if current is None:
        raise LookupError("RAG run not found")
    current_status = str(current.get("status") or "")
    if current_status in TERMINAL_STATUSES:
        return current
    _assert_transition(current_status, status)
    payload = {
        **(terminal_payload or {"type": "terminal"}),
        "run_id": run_id,
        "run_status": status,
        **({"detail": error} if error else {}),
    }
    updated = await repo.finish_run(
        user_id=user_id,
        run_id=run_id,
        expected_status=current_status,
        status=status,
        payload=payload,
        request_id=request_id,
        session_id=session_id,
        answer=answer,
        error=error,
        metadata=metadata,
    )
    if updated is not None:
        return updated
    latest = await repo.get_run(user_id=user_id, run_id=run_id)
    if latest is None:
        raise LookupError("RAG run not found")
    return latest


async def amend_run_metadata(
    *,
    user_id: str,
    run_id: str,
    metadata: dict[str, Any],
    answer: str | None = None,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any]:
    """
    把补丁写回 run 的元数据与正文，不迁移状态、不新增 terminal 事件。

    只重试最终答案这条路跑在 run 已落终态之后：它要回写的是本轮的真实状态
    （source_status / trace_summary / 重试记录）与重试产出的正文，状态本身不变。
    所以不能走 update_status —— 终态会转到 finish_run，而 finish_run 在 run 已是终态时
    原样返回，补丁一个字都写不进去；也不能再走一次 finish_run，那会插出第二个 terminal 事件。
    """
    repo = _repository(repository)
    current = await repo.get_run(user_id=user_id, run_id=run_id)
    if current is None:
        raise LookupError("RAG run not found")
    current_status = str(current.get("status") or "")
    updated = await repo.update_run(
        run_id=run_id,
        user_id=user_id,
        status=current_status,
        answer=answer,
        metadata=metadata,
        # completed=False 保住原有的 completed_at：这是事后补写，不是又一次收尾
        completed=False,
        expected_status=current_status,
    )
    if updated is None:
        raise LookupError("RAG run not found")
    return updated


async def request_cancel(
    *,
    user_id: str,
    run_id: str,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any]:
    """写 cancel_requested；已经终止的 run 原样返回。"""
    repo = _repository(repository)
    updated = await repo.request_cancel(user_id=user_id, run_id=run_id)
    if updated is not None:
        return updated
    current = await repo.get_run(user_id=user_id, run_id=run_id)
    if current is None:
        raise LookupError("RAG run not found")
    return current


async def list_events(
    *,
    user_id: str,
    run_id: str,
    after_sequence: int,
    limit: int,
    repository: RAGRunRepository | None = None,
) -> list[dict[str, Any]]:
    """返回指定 run 在 sequence 之后的持久化事件。"""
    return await _repository(repository).list_events(
        user_id=user_id,
        run_id=run_id,
        after_sequence=after_sequence,
        limit=limit,
    )


async def expire_overdue_runs(repository: RAGRunRepository | None = None) -> int:
    """参数为仓储；对过期操作发停止信号，返回数量，实际所有者退出前保持排他。"""
    repo = _repository(repository)
    expired = await repo.list_expired_active_runs()
    for run in expired:
        await repo.request_cancel(
            user_id=str(run["user_id"]),
            run_id=str(run["run_id"]),
        )
    return len(expired)


async def get_run(
    *,
    user_id: str,
    run_id: str,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any] | None:
    return await _repository(repository).get_run(user_id=user_id, run_id=run_id)


async def get_by_idempotency_key(
    *,
    user_id: str,
    idempotency_key: str,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any] | None:
    """按 owner 和幂等键读取 run，供重复 POST 在副作用前短路。"""
    return await _repository(repository).get_by_idempotency_key(
        user_id=user_id,
        idempotency_key=idempotency_key,
    )


async def get_latest_for_conversation(
    *,
    user_id: str,
    conversation_id: str,
    repository: RAGRunRepository | None = None,
) -> dict[str, Any] | None:
    return await _repository(repository).get_latest_for_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )
