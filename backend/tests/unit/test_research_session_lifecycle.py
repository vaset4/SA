"""
研究会话生命周期回归测试。

作者: xxx
时间: 2026-08-18 16:00:00
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.research_session_lifecycle import (
    ACTION_CANCEL,
    ACTION_RESUME,
    ACTION_RETRY,
    STATUS_AWAITING_CLARIFICATION,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_CREATED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_TIMED_OUT,
    InvalidResearchSessionTransition,
    LegacyResearchSessionNotRecoverable,
    ResearchSessionAssociationError,
    ResearchSessionLifecycle,
    ResearchSessionNotFound,
    ResearchSessionScopeConflict,
    allowed_actions_for_status,
    persist_research_session_links,
)
from app.services.session_mapping_service import build_session_identity


class InMemorySessionRepository:
    """保存测试 session，并按 owner 模拟真实仓储隔离。"""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def create_session(self, **payload: Any) -> dict[str, Any] | None:
        """创建 session；冲突时模拟 `ON CONFLICT DO NOTHING`。"""
        session_id = str(payload["session_id"])
        if session_id in self.rows:
            return None
        row = {
            **payload,
            "status": payload.get("status", STATUS_CREATED),
            "last_error": None,
            "attempt_count": 0,
            "state_json": None,
            "step_name": None,
        }
        self.rows[session_id] = row
        return dict(row)

    async def get_session(self, session_id: str, *, user_id: str) -> dict[str, Any] | None:
        """仅返回属于指定用户的 session。"""
        row = self.rows.get(session_id)
        if row is None or row["user_id"] != user_id:
            return None
        return dict(row)

    async def bind_session_scope(self, session_id: str, *, user_id: str, **payload: Any) -> dict[str, Any] | None:
        """只补齐尚未持久化的 scope 字段。"""
        row = await self.get_session(session_id, user_id=user_id)
        if row is None:
            return None
        stored = self.rows[session_id]
        for key in ("conversation_id", "project_id", "thread_id", "research_mode"):
            if not stored.get(key) and payload.get(key):
                stored[key] = payload[key]
        if not stored.get("paper_ids") and payload.get("paper_ids"):
            stored["paper_ids"] = list(payload["paper_ids"])
        return dict(stored)

    async def bind_thread_id(
        self,
        session_id: str,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        """只为 owner 匹配且尚未绑定的 legacy session 写入 thread_id。"""
        row = self.rows.get(session_id)
        if row is None or row["user_id"] != user_id or row.get("thread_id"):
            return None
        row["thread_id"] = thread_id
        return dict(row)

    async def transition_status(
        self,
        session_id: str,
        *,
        user_id: str,
        expected_status: str,
        target_status: str,
        last_error: str | None,
        increment_attempt: bool,
        expected_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """按当前状态执行原子状态迁移。"""
        row = self.rows.get(session_id)
        if row is None or row["user_id"] != user_id or row["status"] != expected_status:
            return None
        if row.get("current_run_id") != expected_run_id:
            return None
        row["status"] = target_status
        row["last_error"] = last_error
        if increment_attempt:
            row["attempt_count"] += 1
        return dict(row)


@pytest.mark.asyncio
async def test_create_session_is_immediately_readable_with_durable_identity() -> None:
    """创建返回后必须已能按 owner 读取同一份 session 身份与范围。"""
    repository = InMemorySessionRepository()
    lifecycle = ResearchSessionLifecycle(repository=repository)

    created = await lifecycle.ensure_session(
        user_id="user-1",
        topic="图神经网络研究",
        session_id="session-1",
        conversation_id="conv-1",
        project_id="project-1",
        paper_ids=["paper-1", "paper-2"],
        research_mode="internal",
    )
    loaded = await lifecycle.get_session(user_id="user-1", session_id="session-1")

    assert loaded == created
    assert loaded["thread_id"] == "session-1"
    assert loaded["conversation_id"] == "conv-1"
    assert loaded["project_id"] == "project-1"
    assert loaded["paper_ids"] == ["paper-1", "paper-2"]
    assert loaded["status"] == STATUS_CREATED


@pytest.mark.asyncio
async def test_session_read_is_owner_aware() -> None:
    """其他用户读取同一 session_id 时必须得到不存在，而不是跨 owner 数据。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    await lifecycle.ensure_session(
        user_id="user-1",
        topic="owner isolation",
        session_id="session-owner",
    )

    with pytest.raises(ResearchSessionNotFound, match="会话不存在"):
        await lifecycle.get_session(user_id="user-2", session_id="session-owner")


def test_new_session_thread_id_never_uses_conversation_id() -> None:
    """新 session 的 checkpoint key 固定为 session_id，conversation 只保留展示语义。"""
    identity = build_session_identity("session-1", "conv-1")

    assert identity == {
        "session_id": "session-1",
        "conversation_id": "conv-1",
        "thread_id": "session-1",
    }


@pytest.mark.asyncio
async def test_invalid_status_transition_is_rejected_before_repository_write() -> None:
    """已完成 session 不得重新进入 running；后续 route 会把该错误映射为 HTTP 409。"""
    repository = InMemorySessionRepository()
    lifecycle = ResearchSessionLifecycle(repository=repository)
    await lifecycle.ensure_session(
        user_id="user-1",
        topic="terminal session",
        session_id="session-terminal",
    )
    repository.rows["session-terminal"]["status"] = STATUS_COMPLETED

    with pytest.raises(InvalidResearchSessionTransition, match="completed -> running"):
        await lifecycle.transition(
            user_id="user-1",
            session_id="session-terminal",
            target_status=STATUS_RUNNING,
        )

    assert repository.rows["session-terminal"]["status"] == STATUS_COMPLETED


@pytest.mark.asyncio
async def test_graph_result_persists_clarification_then_completion() -> None:
    """graph 的澄清与完成结果必须写入显式 durable status。"""
    repository = InMemorySessionRepository()
    lifecycle = ResearchSessionLifecycle(repository=repository)
    await lifecycle.ensure_session(
        user_id="user-1",
        topic="clarification session",
        session_id="session-clarification",
    )
    running = await lifecycle.start(user_id="user-1", session_id="session-clarification")
    awaiting = await lifecycle.record_result(
        user_id="user-1",
        session_id="session-clarification",
        result={"awaiting_clarification": True},
    )
    resumed = await lifecycle.start(user_id="user-1", session_id="session-clarification")
    completed = await lifecycle.record_result(
        user_id="user-1",
        session_id="session-clarification",
        result={"report": "done"},
    )

    assert running["attempt_count"] == 1
    assert awaiting["status"] == STATUS_AWAITING_CLARIFICATION
    assert resumed["attempt_count"] == 1
    assert completed["status"] == STATUS_COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timed_out", "cancelled", "expected_status"),
    [
        (False, False, STATUS_FAILED),
        (True, False, STATUS_TIMED_OUT),
        (False, True, STATUS_CANCELLED),
    ],
)
async def test_graph_failure_persists_explicit_terminal_status(
    timed_out: bool,
    cancelled: bool,
    expected_status: str,
) -> None:
    """graph 的普通失败、超时和取消必须分别写入可恢复终态。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    session = await lifecycle.ensure_session(
        user_id="user-1",
        topic="terminal outcome",
    )
    await lifecycle.start(user_id="user-1", session_id=str(session["session_id"]))

    terminal = await lifecycle.record_failure(
        user_id="user-1",
        session_id=str(session["session_id"]),
        error=RuntimeError("graph stopped"),
        timed_out=timed_out,
        cancelled=cancelled,
    )

    assert terminal["status"] == expected_status
    assert terminal["last_error"] == "graph stopped"


@pytest.mark.asyncio
async def test_existing_scope_cannot_be_overwritten() -> None:
    """已有 project、paper 和 mode 不能被后续请求改写。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    await lifecycle.ensure_session(
        user_id="user-1",
        topic="fixed scope",
        session_id="session-scope",
        project_id="project-1",
        paper_ids=["paper-1"],
        research_mode="internal",
    )

    with pytest.raises(ResearchSessionScopeConflict, match="project_id"):
        await lifecycle.ensure_session(
            user_id="user-1",
            topic="changed scope",
            session_id="session-scope",
            project_id="project-2",
            paper_ids=["paper-1"],
            research_mode="internal",
        )


@pytest.mark.asyncio
async def test_legacy_thread_is_bound_once_from_checkpoint_evidence() -> None:
    """legacy session 只按 checkpoint 证据绑定一次 conversation thread。"""
    repository = InMemorySessionRepository()
    lifecycle = ResearchSessionLifecycle(repository=repository)
    session = await lifecycle.ensure_session(
        user_id="user-1",
        topic="legacy",
        session_id="legacy-session",
        conversation_id="conv-legacy",
    )
    repository.rows[str(session["session_id"])]["thread_id"] = None

    async def checkpoint_exists(thread_id: str) -> bool:
        """仅 legacy conversation thread 存在 checkpoint。"""
        return thread_id == "conv-legacy"

    bound = await lifecycle.bind_legacy_thread(
        user_id="user-1",
        session_id="legacy-session",
        conversation_id="conv-legacy",
        checkpoint_exists=checkpoint_exists,
    )

    assert bound["thread_id"] == "conv-legacy"


@pytest.mark.asyncio
async def test_legacy_thread_rejects_ambiguous_checkpoint_evidence() -> None:
    """session 与 conversation 均有 checkpoint 时不得猜测 legacy thread。"""
    repository = InMemorySessionRepository()
    lifecycle = ResearchSessionLifecycle(repository=repository)
    session = await lifecycle.ensure_session(
        user_id="user-1",
        topic="ambiguous legacy",
        session_id="legacy-ambiguous",
        conversation_id="conv-ambiguous",
    )
    repository.rows[str(session["session_id"])]["thread_id"] = None

    async def checkpoint_exists(thread_id: str) -> bool:
        """模拟两个候选 thread 均已存在 checkpoint。"""
        return True

    with pytest.raises(LegacyResearchSessionNotRecoverable, match="两个 checkpoint"):
        await lifecycle.bind_legacy_thread(
            user_id="user-1",
            session_id="legacy-ambiguous",
            conversation_id="conv-ambiguous",
            checkpoint_exists=checkpoint_exists,
        )


@pytest.mark.parametrize(
    ("status", "expected_actions"),
    [
        (STATUS_CREATED, [ACTION_CANCEL]),
        (STATUS_RUNNING, [ACTION_CANCEL]),
        (STATUS_AWAITING_CLARIFICATION, [ACTION_RESUME, ACTION_CANCEL]),
        (STATUS_FAILED, [ACTION_RETRY]),
        (STATUS_CANCELLED, [ACTION_RETRY]),
        (STATUS_TIMED_OUT, [ACTION_RETRY]),
        (STATUS_COMPLETED, []),
    ],
)
def test_allowed_actions_follow_durable_status(
    status: str,
    expected_actions: list[str],
) -> None:
    """服务端状态机必须给前端稳定的命令集合，前端不再自行推断。"""
    assert allowed_actions_for_status(status) == expected_actions


@pytest.mark.asyncio
async def test_resume_keeps_session_thread_and_attempt() -> None:
    """澄清恢复必须复用同一 session/thread，且不增加执行 attempt。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    created = await lifecycle.ensure_session(
        user_id="user-1",
        topic="clarification",
        session_id="session-resume",
        project_id="project-1",
        paper_ids=["paper-1", "paper-2"],
        research_mode="internal",
    )
    await lifecycle.start(user_id="user-1", session_id="session-resume")
    await lifecycle.transition(
        user_id="user-1",
        session_id="session-resume",
        target_status=STATUS_AWAITING_CLARIFICATION,
    )

    resumed = await lifecycle.resume(user_id="user-1", session_id="session-resume")

    assert resumed["session_id"] == created["session_id"]
    assert resumed["thread_id"] == created["thread_id"]
    assert resumed["paper_ids"] == ["paper-1", "paper-2"]
    assert resumed["attempt_count"] == 1
    assert resumed["allowed_actions"] == [ACTION_CANCEL]


@pytest.mark.asyncio
async def test_retry_keeps_scope_and_increments_attempt() -> None:
    """失败重试必须复用已持久化范围，并为同一 session 增加一次 attempt。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    created = await lifecycle.ensure_session(
        user_id="user-1",
        topic="retry",
        session_id="session-retry",
        conversation_id="conv-1",
        project_id="project-1",
        paper_ids=["paper-1", "paper-2"],
        research_mode="internal",
    )
    await lifecycle.start(user_id="user-1", session_id="session-retry")
    await lifecycle.record_failure(
        user_id="user-1",
        session_id="session-retry",
        error=RuntimeError("provider failed"),
    )

    retried = await lifecycle.retry(user_id="user-1", session_id="session-retry")

    assert retried["session_id"] == created["session_id"]
    assert retried["thread_id"] == created["thread_id"]
    assert retried["project_id"] == "project-1"
    assert retried["paper_ids"] == ["paper-1", "paper-2"]
    assert retried["attempt_count"] == 2
    assert retried["last_error"] is None


@pytest.mark.asyncio
async def test_cancel_only_accepts_cancellable_status() -> None:
    """取消只允许 created/running/awaiting，cancelled 再次取消必须明确拒绝。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    created = await lifecycle.ensure_session(
        user_id="user-1",
        topic="cancel",
        session_id="session-cancel",
    )

    cancelled = await lifecycle.cancel(user_id="user-1", session_id="session-cancel")
    assert cancelled["status"] == STATUS_CANCELLED
    assert cancelled["allowed_actions"] == [ACTION_RETRY]

    with pytest.raises(InvalidResearchSessionTransition, match="cancel"):
        await lifecycle.cancel(user_id="user-1", session_id="session-cancel")

    assert created["session_id"] == cancelled["session_id"]


@pytest.mark.asyncio
async def test_scope_link_failure_is_visible_and_retry_is_idempotent() -> None:
    """项目关联失败必须写 failed；同 session 重试成功后再进入 running。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    link_calls: list[str] = []

    async def failing_linker(session: dict[str, Any]) -> None:
        """模拟 project-session 关联写失败。"""
        link_calls.append(str(session["session_id"]))
        raise RuntimeError("project link unavailable")

    async def successful_linker(session: dict[str, Any]) -> None:
        """模拟相同 project-session 关联的幂等重试成功。"""
        link_calls.append(str(session["session_id"]))

    async def checkpoint_exists(thread_id: str) -> bool:
        """新 session 不需要 legacy checkpoint 证据。"""
        return False

    with pytest.raises(ResearchSessionAssociationError, match="project link unavailable"):
        await lifecycle.prepare_for_run(
            user_id="user-1",
            topic="linked research",
            session_id="session-link",
            conversation_id="conv-1",
            project_id="project-1",
            paper_ids=["paper-1", "paper-2"],
            research_mode="internal",
            checkpoint_exists=checkpoint_exists,
            scope_linker=failing_linker,
        )

    failed = await lifecycle.get_session(user_id="user-1", session_id="session-link")
    retried = await lifecycle.retry(
        user_id="user-1",
        session_id="session-link",
        scope_linker=successful_linker,
    )

    assert failed["status"] == STATUS_FAILED
    assert "project link unavailable" in failed["last_error"]
    assert retried["status"] == STATUS_RUNNING
    assert retried["attempt_count"] == 2
    assert retried["project_id"] == "project-1"
    assert retried["paper_ids"] == ["paper-1", "paper-2"]
    assert link_calls == ["session-link", "session-link"]


@pytest.mark.asyncio
async def test_empty_scope_request_reuses_persisted_scope() -> None:
    """恢复请求未传 project/papers 时必须回读 session 范围，不能降级为 external。"""
    lifecycle = ResearchSessionLifecycle(repository=InMemorySessionRepository())
    await lifecycle.ensure_session(
        user_id="user-1",
        topic="persisted scope",
        session_id="session-scope-reuse",
        project_id="project-1",
        paper_ids=["paper-1", "paper-2"],
        research_mode="internal",
    )

    scope = await lifecycle.resolve_scope_request(
        user_id="user-1",
        session_id="session-scope-reuse",
        project_id=None,
        paper_ids=[],
    )

    assert scope["project_id"] == "project-1"
    assert scope["paper_ids"] == ["paper-1", "paper-2"]


@pytest.mark.asyncio
async def test_persisted_links_write_multi_paper_collection_once(monkeypatch) -> None:
    """项目研究关联必须幂等挂 session，并将两篇论文作为一个集合写入 conversation。"""
    from app.services import conversation_service, project_service

    calls: list[tuple[str, object]] = []

    async def fake_attach(project_id: str, session_id: str, **payload: Any) -> dict:
        """记录 project-session 关联调用。"""
        calls.append(("project", (project_id, session_id, payload["user_id"])))
        return {"project_id": project_id, "session_id": session_id}

    async def fake_set(conversation_id: str, paper_ids: list[str], **payload: Any) -> bool:
        """记录 conversation 论文集合单次写入。"""
        calls.append(("papers", (conversation_id, paper_ids, payload["user_id"])))
        return True

    monkeypatch.setattr(project_service, "attach_session", fake_attach)
    monkeypatch.setattr(conversation_service, "set_associated_papers", fake_set)

    await persist_research_session_links(
        {
            "session_id": "session-1",
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "project_id": "project-1",
            "paper_ids": ["paper-1", "paper-2", "paper-1"],
        },
        user_id="user-1",
    )

    assert calls == [
        ("project", ("project-1", "session-1", "user-1")),
        ("papers", ("conv-1", ["paper-1", "paper-2"], "user-1")),
    ]


@pytest.mark.asyncio
async def test_set_associated_papers_updates_complete_collection_once() -> None:
    """conversation service 必须去重整组论文并只执行一次 repository 更新。"""
    from app.services import conversation_service

    class RecordingConversationRepository:
        """记录 associated_papers 更新参数的测试仓储。"""

        def __init__(self) -> None:
            self.updates: list[dict[str, Any]] = []

        async def update_conversation(self, **payload: Any) -> bool:
            """记录一次 conversation 更新并模拟 owner 命中。"""
            self.updates.append(payload)
            return True

    repository = RecordingConversationRepository()
    updated = await conversation_service.set_associated_papers(
        "conv-1",
        ["paper-1", "paper-2", "paper-1", ""],
        user_id="user-1",
        repository=repository,  # type: ignore[arg-type]
    )

    assert updated is True
    assert len(repository.updates) == 1
    assert repository.updates[0]["associated_papers"] == ["paper-1", "paper-2"]
