"""
研究会话 durable lifecycle 领域服务。

作者: xxx
时间: 2026-08-18 16:00:00
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any
from datetime import datetime

from app.repositories.session_repository import SessionRepository, get_session_repository
from app.services.session_mapping_service import ensure_session_id
from app.services import rag_run_service

STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_AWAITING_CLARIFICATION = "awaiting_clarification"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_TIMED_OUT = "timed_out"

ACTION_RESUME = "resume"
ACTION_CANCEL = "cancel"
ACTION_RETRY = "retry"

VALID_STATUSES = {
    STATUS_CREATED,
    STATUS_RUNNING,
    STATUS_AWAITING_CLARIFICATION,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_TIMED_OUT,
}

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    STATUS_CREATED: {STATUS_RUNNING, STATUS_CANCELLED},
    STATUS_RUNNING: {
        STATUS_AWAITING_CLARIFICATION,
        STATUS_COMPLETED,
        STATUS_FAILED,
        STATUS_CANCELLED,
        STATUS_TIMED_OUT,
    },
    STATUS_AWAITING_CLARIFICATION: {STATUS_RUNNING, STATUS_CANCELLED},
    STATUS_FAILED: {STATUS_RUNNING},
    STATUS_CANCELLED: {STATUS_RUNNING},
    STATUS_TIMED_OUT: {STATUS_RUNNING},
    STATUS_COMPLETED: set(),
}

_ALLOWED_ACTIONS: dict[str, tuple[str, ...]] = {
    STATUS_CREATED: (ACTION_CANCEL,),
    STATUS_RUNNING: (ACTION_CANCEL,),
    STATUS_AWAITING_CLARIFICATION: (ACTION_RESUME, ACTION_CANCEL),
    STATUS_FAILED: (ACTION_RETRY,),
    STATUS_CANCELLED: (ACTION_RETRY,),
    STATUS_TIMED_OUT: (ACTION_RETRY,),
    STATUS_COMPLETED: (),
}

SessionScopeLinker = Callable[[dict[str, Any]], Awaitable[None]]


class ResearchSessionNotFound(LookupError):
    """表示当前 owner 无法读取指定研究 session。"""


class ResearchSessionScopeConflict(ValueError):
    """表示请求范围与 session 已持久化范围冲突。"""


class InvalidResearchSessionTransition(ValueError):
    """表示研究 session 状态机拒绝本次迁移。"""


class LegacyResearchSessionNotRecoverable(ValueError):
    """表示历史 session 的 checkpoint thread 无法唯一确认。"""


class ResearchSessionAssociationError(RuntimeError):
    """表示 durable session 已创建，但项目或对话范围关联失败。"""


def allowed_actions_for_status(status: str | None) -> list[str]:
    """
    返回指定 durable status 当前允许执行的用户命令。

    Args:
        status: 研究 session 当前状态

    Returns:
        list[str]: 顺序稳定的 resume/cancel/retry 命令列表
    """
    return list(_ALLOWED_ACTIONS.get(str(status or "").strip(), ()))


def _with_allowed_actions(row: dict[str, Any]) -> dict[str, Any]:
    """复制 session 行并附加由服务端状态机计算的 allowed_actions。"""
    return {
        **row,
        "allowed_actions": allowed_actions_for_status(row.get("status")),
    }


def _normalize(value: Any) -> str | None:
    """把可选标识归一化为空值或去空白字符串。"""
    normalized = str(value or "").strip()
    return normalized or None


def _normalize_paper_ids(paper_ids: list[str] | None) -> list[str]:
    """去空白、去重并保持论文 ID 首次出现顺序。"""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_paper_id in paper_ids or []:
        paper_id = str(raw_paper_id or "").strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        normalized.append(paper_id)
    return normalized


def _legacy_state(row: dict[str, Any]) -> dict[str, Any]:
    """读取 legacy state_json；无效 JSON 明确视为空状态。"""
    raw = row.get("state_json")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _derive_legacy_status(row: dict[str, Any]) -> str:
    """按旧快照中明确存在的信号推导一次性 durable status。"""
    state = _legacy_state(row)
    if state.get("awaiting_clarification") or state.get("clarification_question"):
        return STATUS_AWAITING_CLARIFICATION
    if state.get("cancelled") or state.get("status") == STATUS_CANCELLED:
        return STATUS_CANCELLED
    if state.get("error") or state.get("status") == STATUS_FAILED:
        return STATUS_FAILED
    if state.get("report"):
        return STATUS_COMPLETED
    return STATUS_RUNNING if row.get("step_name") else STATUS_CREATED


def _assert_scope_compatible(row: dict[str, Any], requested: dict[str, Any]) -> None:
    """拒绝用新请求覆盖 session 已固定的身份或研究范围。"""
    for key in ("conversation_id", "project_id", "research_mode"):
        stored_value = _normalize(row.get(key))
        requested_value = _normalize(requested.get(key))
        if stored_value and requested_value and stored_value != requested_value:
            raise ResearchSessionScopeConflict(f"研究会话的 {key} 与已持久化范围冲突")
    stored_papers = set(_normalize_paper_ids(row.get("paper_ids")))
    requested_papers = set(_normalize_paper_ids(requested.get("paper_ids")))
    if stored_papers and requested_papers and stored_papers != requested_papers:
        raise ResearchSessionScopeConflict("研究会话的 paper_ids 与已持久化范围冲突")


class ResearchSessionLifecycle:
    """统一承担研究 session 的创建、范围绑定和显式状态迁移。"""

    def __init__(self, repository: SessionRepository | None = None) -> None:
        """
        创建生命周期服务。

        Args:
            repository: 可替换的 session 仓储；默认使用项目单例
        """
        self._repository = repository or get_session_repository()

    async def get_session(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        """
        按 owner 读取 durable session，并首次固化 legacy status。

        Args:
            user_id: 当前登录用户 ID
            session_id: 研究 session ID

        Returns:
            dict[str, Any]: owner-scoped durable session
        """
        row = await self._repository.get_session(session_id, user_id=user_id)
        if row is None:
            raise ResearchSessionNotFound("会话不存在")
        if _normalize(row.get("status")):
            return _with_allowed_actions(row)
        initialized = await self._repository.initialize_legacy_status(
            session_id,
            user_id=user_id,
            status=_derive_legacy_status(row),
        )
        if initialized is not None:
            return _with_allowed_actions(initialized)
        latest = await self._repository.get_session(session_id, user_id=user_id)
        if latest is None:
            raise ResearchSessionNotFound("会话不存在")
        return _with_allowed_actions(latest)

    async def list_sessions(
        self,
        *,
        user_id: str,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        """
        按 owner 列出 session，并把命中的 legacy status 一次性固化。

        Args:
            user_id: 当前登录用户 ID
            limit: 最大返回数量
            offset: 分页偏移量

        Returns:
            list[dict[str, Any]]: durable session 列表
        """
        rows = await self._repository.list_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if _normalize(row.get("status")):
                normalized.append(_with_allowed_actions(row))
                continue
            normalized.append(
                await self.get_session(
                    user_id=user_id,
                    session_id=str(row["session_id"]),
                )
            )
        return normalized

    async def ensure_session(
        self,
        *,
        user_id: str,
        topic: str,
        session_id: str | None = None,
        conversation_id: str | None = None,
        project_id: str | None = None,
        paper_ids: list[str] | None = None,
        research_mode: str | None = None,
    ) -> dict[str, Any]:
        """
        在 graph 前创建 session，或只补齐已有 session 的空 scope。

        Args:
            user_id: 当前登录用户 ID
            topic: 会话展示标题
            session_id: 可选既有 session ID；缺失时生成新 ID
            conversation_id: 可选前端对话 ID
            project_id: 可选项目 ID
            paper_ids: 已完成 owner 校验的论文范围
            research_mode: internal 或 external

        Returns:
            dict[str, Any]: 已持久化并可立即读取的 durable session
        """
        normalized_user_id = _normalize(user_id)
        if normalized_user_id is None:
            raise ValueError("user_id 不能为空")
        resolved_session_id = ensure_session_id(session_id)
        requested = self._scope_payload(
            conversation_id=conversation_id,
            project_id=project_id,
            paper_ids=paper_ids,
            research_mode=research_mode,
        )
        existing = await self._repository.get_session(resolved_session_id, user_id=normalized_user_id)
        if existing is not None:
            return await self._bind_existing(existing, normalized_user_id, requested)
        return await self._create_new(
            user_id=normalized_user_id,
            session_id=resolved_session_id,
            topic=topic,
            requested=requested,
        )

    async def resolve_scope_request(
        self,
        *,
        user_id: str,
        session_id: str | None,
        project_id: str | None,
        paper_ids: list[str] | None,
    ) -> dict[str, Any]:
        """
        用非空请求值覆盖空位，否则回读 session 已持久化的研究范围。

        Args:
            user_id: 当前登录用户 ID
            session_id: 可选既有研究 session ID
            project_id: 请求显式携带的项目 ID
            paper_ids: 请求显式携带的论文 ID 集合

        Returns:
            dict[str, Any]: owner 校验后的有效 project/paper/mode 范围
        """
        requested = self._scope_payload(
            conversation_id=None,
            project_id=project_id,
            paper_ids=paper_ids,
            research_mode=None,
        )
        normalized_session_id = _normalize(session_id)
        if normalized_session_id is None:
            return requested
        row = await self.get_session(user_id=user_id, session_id=normalized_session_id)
        _assert_scope_compatible(row, requested)
        return {
            "project_id": requested["project_id"] or _normalize(row.get("project_id")),
            "paper_ids": requested["paper_ids"] or _normalize_paper_ids(row.get("paper_ids")),
            "research_mode": _normalize(row.get("research_mode")),
        }

    async def transition(
        self,
        *,
        user_id: str,
        session_id: str,
        target_status: str,
        last_error: str | None = None,
        expected_run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        校验状态机后使用 CAS 持久化一次状态迁移。

        Args:
            user_id: 当前登录用户 ID
            session_id: 研究 session ID
            target_status: 目标状态
            last_error: failed/timed_out 对应的可见错误

        Returns:
            dict[str, Any]: 迁移后的 durable session
        """
        row = await self.get_session(user_id=user_id, session_id=session_id)
        current_status = str(row.get("status") or "")
        if row.get("current_run_id") != expected_run_id:
            raise InvalidResearchSessionTransition("研究执行身份已变化，拒绝旧回调写入")
        self._assert_transition(current_status, target_status)
        if current_status == target_status:
            if target_status == STATUS_RUNNING:
                raise InvalidResearchSessionTransition("研究已在运行，未取得新的执行权")
            return row
        updated = await self._repository.transition_status(
            session_id,
            user_id=user_id,
            expected_status=current_status,
            target_status=target_status,
            last_error=last_error,
            increment_attempt=self._increments_attempt(current_status, target_status),
            expected_run_id=expected_run_id,
        )
        if updated is not None:
            return _with_allowed_actions(updated)
        latest = await self.get_session(user_id=user_id, session_id=session_id)
        if target_status != STATUS_RUNNING and latest.get("status") == target_status and latest.get("current_run_id") == expected_run_id:
            return latest
        raise InvalidResearchSessionTransition(
            f"研究会话状态已变化，无法执行 {current_status} -> {target_status}"
        )

    async def prepare_command_session(
        self, *, user_id: str, topic: str, session_id: str | None,
        conversation_id: str | None, project_id: str | None, paper_ids: list[str] | None,
        research_mode: str, checkpoint_exists: Callable[[str], Awaitable[bool]],
        scope_linker: SessionScopeLinker | None = None,
    ) -> dict[str, Any]:
        """参数为可信身份范围和关联端口；准备 durable session，执行权仅由操作事务授予。"""
        session = await self.ensure_session(
            user_id=user_id, topic=topic, session_id=session_id, conversation_id=conversation_id,
            project_id=project_id, paper_ids=paper_ids, research_mode=research_mode,
        )
        session = await self.bind_legacy_thread(
            user_id=user_id, session_id=str(session["session_id"]),
            conversation_id=conversation_id, checkpoint_exists=checkpoint_exists,
        )
        if scope_linker is not None:
            try:
                await scope_linker(session)
            except Exception as exc:
                raise ResearchSessionAssociationError(f"研究会话范围关联失败: {exc}") from exc
        return session

    async def accept_operation(
        self, *, user_id: str, session: dict, operation_kind: str,
        idempotency_key: str, query: str, request_payload: dict, deadline_at: datetime | None,
        clarification_pending: bool = False,
    ) -> tuple[dict, dict, bool]:
        """参数为用户命令与已核对 session；原子领取运行并返回操作、session 和新领取标记。"""
        existing = await rag_run_service.get_by_idempotency_key(
            user_id=user_id, idempotency_key=idempotency_key,
        )
        if existing is not None:
            rag_run_service.assert_same_operation(
                existing, operation_kind=operation_kind, request_payload=request_payload,
            )
            durable = await self.get_session(user_id=user_id, session_id=existing["session_id"])
            return existing, durable, False
        allowed = {"research": {STATUS_CREATED}, "resume": {STATUS_AWAITING_CLARIFICATION},
                   "retry": {STATUS_FAILED, STATUS_CANCELLED, STATUS_TIMED_OUT}}
        if session.get("status") not in allowed.get(operation_kind, set()):
            if session.get("current_run_id"):
                active = await rag_run_service.get_run(user_id=user_id, run_id=session["current_run_id"])
                if active is not None and active["status"] in rag_run_service.ACTIVE_STATUSES:
                    if active.get("idempotency_key") == idempotency_key:
                        rag_run_service.assert_same_operation(
                            active, operation_kind=operation_kind, request_payload=request_payload,
                        )
                        return active, session, False
                    raise rag_run_service.ActiveRAGRunConflict(active)
            raise InvalidResearchSessionTransition("当前研究会话不允许该操作")
        command = {key: session.get(key) for key in
                   ("project_id", "paper_ids", "research_mode", "conversation_id")}
        command.update(expected_status=session["status"],
                       increment_attempt=operation_kind != "resume" and not clarification_pending)
        run, created = await rag_run_service.create_operation(
            user_id=user_id, conversation_id=session.get("conversation_id"),
            session_id=session["session_id"], thread_id=session["thread_id"],
            idempotency_key=idempotency_key, query=query, deadline_at=deadline_at,
            operation_kind=operation_kind, request_payload=request_payload, session_command=command,
            metadata={"kind": "deep_research", "project_id": session.get("project_id"),
                      "paper_ids": session.get("paper_ids") or []},
        )
        durable = await self.get_session(user_id=user_id, session_id=session["session_id"])
        return run, durable, created

    async def prepare_for_run(
        self, *,
        user_id: str,
        topic: str,
        session_id: str | None,
        conversation_id: str | None,
        project_id: str | None,
        paper_ids: list[str] | None,
        research_mode: str,
        checkpoint_exists: Callable[[str], Awaitable[bool]],
        scope_linker: SessionScopeLinker | None = None,
    ) -> dict[str, Any]:
        """
        组合 create-before-run、legacy thread 绑定与 running CAS。

        Args:
            user_id: 当前登录用户 ID
            topic: session 展示标题
            session_id: 可选已有 session ID
            conversation_id: 可选前端对话 ID
            project_id: owner 校验后的项目 ID
            paper_ids: owner 校验后的论文范围
            research_mode: internal 或 external
            checkpoint_exists: 判断候选 legacy thread 是否有 checkpoint 的端口
            scope_linker: 将 session 幂等关联到项目和对话范围的异步端口

        Returns:
            dict[str, Any]: 已持久化且进入 running 的 session
        """
        # 1. graph 启动前先持久化 session 与 scope
        session = await self.ensure_session(
            user_id=user_id,
            topic=topic,
            session_id=session_id,
            conversation_id=conversation_id,
            project_id=project_id,
            paper_ids=paper_ids,
            research_mode=research_mode,
        )
        # 2. legacy 行只允许根据既有证据一次性补齐 thread_id
        session = await self.bind_legacy_thread(
            user_id=user_id,
            session_id=str(session["session_id"]),
            conversation_id=conversation_id,
            checkpoint_exists=checkpoint_exists,
        )
        # 3. 最后迁移到 running 并完成范围关联，graph 不会先于 durable 关联执行
        return await self._start_with_scope_linker(
            user_id=user_id,
            session_id=str(session["session_id"]),
            scope_linker=scope_linker,
        )

    async def start(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        """把 created/retry/resume session 迁移到 running 并按规则增加 attempt。"""
        return await self.transition(
            user_id=user_id,
            session_id=session_id,
            target_status=STATUS_RUNNING,
        )

    async def resume(
        self,
        *,
        user_id: str,
        session_id: str,
        scope_linker: SessionScopeLinker | None = None,
    ) -> dict[str, Any]:
        """
        将 awaiting_clarification session 恢复为 running。

        Args:
            user_id: 当前登录用户 ID
            session_id: 要恢复的研究 session ID
            scope_linker: 恢复执行前重放项目与对话关联的异步端口

        Returns:
            dict[str, Any]: 复用原 session/thread 且保持 attempt 的 running session
        """
        row = await self.get_session(user_id=user_id, session_id=session_id)
        if row.get("status") != STATUS_AWAITING_CLARIFICATION:
            raise InvalidResearchSessionTransition("当前研究会话不允许执行 resume")
        return await self._start_with_scope_linker(
            user_id=user_id,
            session_id=session_id,
            scope_linker=scope_linker,
        )

    async def retry(
        self,
        *,
        user_id: str,
        session_id: str,
        scope_linker: SessionScopeLinker | None = None,
    ) -> dict[str, Any]:
        """
        将失败、取消或超时 session 以同一身份重新进入 running。

        Args:
            user_id: 当前登录用户 ID
            session_id: 要重试的研究 session ID
            scope_linker: 重试执行前重放项目与对话关联的异步端口

        Returns:
            dict[str, Any]: attempt 已增加且复用原 scope/thread 的 session
        """
        row = await self.get_session(user_id=user_id, session_id=session_id)
        retryable = {STATUS_FAILED, STATUS_CANCELLED, STATUS_TIMED_OUT}
        if row.get("status") not in retryable:
            raise InvalidResearchSessionTransition("当前研究会话不允许执行 retry")
        return await self._start_with_scope_linker(
            user_id=user_id,
            session_id=session_id,
            scope_linker=scope_linker,
        )

    async def cancel(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        """
        请求受管执行停止；本进程排空，其他进程仅返回正在停止。

        Args:
            user_id: 当前登录用户 ID
            session_id: 要取消的研究 session ID

        Returns:
            dict[str, Any]: 状态为 cancelled 的 durable session
        """
        row = await self.get_session(user_id=user_id, session_id=session_id)
        if row.get("current_run_id"):
            from app.services.rag_run_executor_service import get_rag_run_executor_registry

            run = await get_rag_run_executor_registry().cancel(
                user_id=user_id, run_id=row["current_run_id"],
            )
            latest = await self.get_session(user_id=user_id, session_id=session_id)
            if latest.get("status") == STATUS_AWAITING_CLARIFICATION:
                if run.get("execution_stopped_at") is None:
                    raise InvalidResearchSessionTransition("研究执行尚未确认退出，当前澄清任务仍在停止")
                latest = await self.transition(
                    user_id=user_id, session_id=session_id, target_status=STATUS_CANCELLED,
                    expected_run_id=run["run_id"],
                )
            return {**latest, "run_id": run["run_id"], "run_status": run["status"],
                    "cancel_requested": run.get("cancel_requested", False)}
        if row.get("status") == STATUS_RUNNING:
            raise InvalidResearchSessionTransition("历史研究缺少执行归属，无法确认停止，当前线程不可重入")
        cancellable = {STATUS_CREATED, STATUS_RUNNING, STATUS_AWAITING_CLARIFICATION}
        if row.get("status") not in cancellable:
            raise InvalidResearchSessionTransition("当前研究会话不允许执行 cancel")
        return await self.transition(
            user_id=user_id,
            session_id=session_id,
            target_status=STATUS_CANCELLED,
        )

    async def record_result(
        self,
        *,
        user_id: str,
        session_id: str,
        result: dict[str, Any],
        expected_run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        把 graph 最终产物映射为澄清、取消或完成状态。

        Args:
            user_id: 当前登录用户 ID
            session_id: 研究 session ID
            result: graph 最终状态或 done payload

        Returns:
            dict[str, Any]: 写入显式终态后的 durable session
        """
        result_status = _normalize(result.get("status"))
        if result.get("awaiting_clarification"):
            target_status = STATUS_AWAITING_CLARIFICATION
        elif result.get("cancelled") or result_status == STATUS_CANCELLED:
            target_status = STATUS_CANCELLED
        else:
            target_status = STATUS_COMPLETED
        return await self.transition(
            user_id=user_id,
            session_id=session_id,
            target_status=target_status,
            expected_run_id=expected_run_id,
        )

    async def record_failure(
        self,
        *,
        user_id: str,
        session_id: str,
        error: BaseException,
        timed_out: bool = False,
        cancelled: bool = False,
        expected_run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        把 graph 异常持久化为 failed、timed_out 或 cancelled。

        Args:
            user_id: 当前登录用户 ID
            session_id: 研究 session ID
            error: 原始异常
            timed_out: 是否属于超时
            cancelled: 是否属于取消

        Returns:
            dict[str, Any]: 写入失败终态后的 durable session
        """
        if cancelled:
            target_status = STATUS_CANCELLED
        elif timed_out:
            target_status = STATUS_TIMED_OUT
        else:
            target_status = STATUS_FAILED
        return await self.transition(
            user_id=user_id,
            session_id=session_id,
            target_status=target_status,
            last_error=str(error) or type(error).__name__,
            expected_run_id=expected_run_id,
        )

    async def bind_legacy_thread(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_id: str | None,
        checkpoint_exists: Callable[[str], Awaitable[bool]],
    ) -> dict[str, Any]:
        """
        根据 checkpoint 与已绑定 conversation 证据一次性固化 legacy thread。

        Args:
            user_id: 当前登录用户 ID
            session_id: 研究 session ID
            conversation_id: 已完成 owner 校验的前端对话 ID
            checkpoint_exists: 判断候选 thread 是否存在 checkpoint 的异步端口

        Returns:
            dict[str, Any]: 已具备显式 thread_id 的 durable session
        """
        row = await self.get_session(user_id=user_id, session_id=session_id)
        if _normalize(row.get("thread_id")):
            return row
        thread_id = await self._legacy_thread_candidate(
            session_id=session_id,
            conversation_id=conversation_id,
            checkpoint_exists=checkpoint_exists,
        )
        bound = await self._repository.bind_thread_id(
            session_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        if bound is not None:
            return _with_allowed_actions(bound)
        latest = await self.get_session(user_id=user_id, session_id=session_id)
        if _normalize(latest.get("thread_id")):
            return latest
        raise LegacyResearchSessionNotRecoverable("旧会话无法恢复：thread_id 绑定失败")

    @staticmethod
    def _scope_payload(
        *,
        conversation_id: str | None,
        project_id: str | None,
        paper_ids: list[str] | None,
        research_mode: str | None,
    ) -> dict[str, Any]:
        """归一化一次 session 创建/绑定所需的可持久化 scope。"""
        return {
            "conversation_id": _normalize(conversation_id),
            "project_id": _normalize(project_id),
            "paper_ids": _normalize_paper_ids(paper_ids),
            "research_mode": _normalize(research_mode),
        }

    async def _bind_existing(
        self,
        row: dict[str, Any],
        user_id: str,
        requested: dict[str, Any],
    ) -> dict[str, Any]:
        """校验已有范围后，仅补齐 session 当前仍为空的字段。"""
        _assert_scope_compatible(row, requested)
        updated = await self._repository.bind_session_scope(
            str(row["session_id"]),
            user_id=user_id,
            conversation_id=requested["conversation_id"],
            project_id=requested["project_id"],
            research_mode=requested["research_mode"],
            paper_ids=requested["paper_ids"],
        )
        if updated is None:
            raise ResearchSessionNotFound("会话不存在")
        if _normalize(updated.get("status")):
            return _with_allowed_actions(updated)
        return await self.get_session(user_id=user_id, session_id=str(row["session_id"]))

    async def _create_new(
        self,
        *,
        user_id: str,
        session_id: str,
        topic: str,
        requested: dict[str, Any],
    ) -> dict[str, Any]:
        """创建新 session，并用同一个 session_id 固定 checkpoint thread。"""
        created = await self._repository.create_session(
            user_id=user_id,
            session_id=session_id,
            topic=str(topic or "").strip() or "新对话",
            conversation_id=requested["conversation_id"],
            project_id=requested["project_id"],
            thread_id=session_id,
            research_mode=requested["research_mode"],
            paper_ids=requested["paper_ids"],
        )
        if created is not None:
            return _with_allowed_actions(created)
        existing = await self._repository.get_session(session_id, user_id=user_id)
        if existing is None:
            raise ResearchSessionNotFound("会话不存在或无权访问")
        return await self._bind_existing(existing, user_id, requested)

    async def _start_with_scope_linker(
        self,
        *,
        user_id: str,
        session_id: str,
        scope_linker: SessionScopeLinker | None,
    ) -> dict[str, Any]:
        """进入 running 后执行幂等范围关联，关联失败时写入可恢复 failed。"""
        running = await self.start(user_id=user_id, session_id=session_id)
        if scope_linker is None:
            return running
        try:
            await scope_linker(running)
        except Exception as exc:
            message = f"研究会话范围关联失败: {exc}"
            await self.record_failure(
                user_id=user_id,
                session_id=session_id,
                error=ResearchSessionAssociationError(message),
            )
            raise ResearchSessionAssociationError(message) from exc
        return running

    @staticmethod
    def _assert_transition(current_status: str, target_status: str) -> None:
        """校验目标状态和当前状态是否满足冻结状态机。"""
        if target_status not in VALID_STATUSES:
            raise InvalidResearchSessionTransition(f"未知研究会话状态: {target_status}")
        if current_status == target_status:
            return
        if target_status not in _ALLOWED_TRANSITIONS.get(current_status, set()):
            raise InvalidResearchSessionTransition(
                f"非法研究会话状态迁移: {current_status} -> {target_status}"
            )

    @staticmethod
    def _increments_attempt(current_status: str, target_status: str) -> bool:
        """首次运行和 retry 增加 attempt，clarification resume 保持同一 attempt。"""
        return target_status == STATUS_RUNNING and current_status in {
            STATUS_CREATED,
            STATUS_FAILED,
            STATUS_CANCELLED,
            STATUS_TIMED_OUT,
        }

    @staticmethod
    async def _legacy_thread_candidate(
        *,
        session_id: str,
        conversation_id: str | None,
        checkpoint_exists: Callable[[str], Awaitable[bool]],
    ) -> str:
        """用旧映射规则与 checkpoint 存在性确认唯一 thread 候选。"""
        normalized_conversation_id = _normalize(conversation_id)
        if normalized_conversation_id is None or normalized_conversation_id == session_id:
            return session_id
        session_checkpoint = await checkpoint_exists(session_id)
        conversation_checkpoint = await checkpoint_exists(normalized_conversation_id)
        if session_checkpoint and conversation_checkpoint:
            raise LegacyResearchSessionNotRecoverable("旧会话无法恢复：存在两个 checkpoint thread")
        if session_checkpoint:
            return session_id
        return normalized_conversation_id


async def persist_research_session_links(
    session: dict[str, Any],
    *,
    user_id: str,
) -> None:
    """
    将 durable session 幂等关联到项目，并一次性写入对话论文集合。

    Args:
        session: 已按 owner 读取且包含固定 project/paper/conversation 范围的 session
        user_id: 当前登录用户 ID

    Returns:
        None: 所有关联成功后返回；任一主关联失败时直接抛错
    """
    from app.services import conversation_service, project_service

    session_id = str(session.get("session_id") or "").strip()
    project_id = _normalize(session.get("project_id"))
    conversation_id = _normalize(session.get("conversation_id"))
    paper_ids = _normalize_paper_ids(session.get("paper_ids"))
    # 1. 项目关联使用 repository 的 ON CONFLICT 语义，重复执行不会生成重复关系
    if project_id:
        await project_service.attach_session(
            project_id,
            session_id,
            user_id=user_id,
            session_row=session,
        )
    # 2. 对话论文范围一次传入完整集合，避免逐篇更新读取旧闭包并互相覆盖
    if conversation_id and paper_ids:
        updated = await conversation_service.set_associated_papers(
            conversation_id,
            paper_ids,
            user_id=user_id,
        )
        if not updated:
            raise LookupError("会话关联的对话不存在，无法写入论文范围")


_lifecycle: ResearchSessionLifecycle | None = None


def get_research_session_lifecycle() -> ResearchSessionLifecycle:
    """返回研究 session 生命周期服务单例。"""
    global _lifecycle
    if _lifecycle is None:
        _lifecycle = ResearchSessionLifecycle()
    return _lifecycle
