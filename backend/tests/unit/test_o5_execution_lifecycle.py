"""O5 执行归属与停止事实回归。作者：xxx，2026-10-04 12:00:00。"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock
from datetime import datetime, timedelta, timezone
import threading

import httpx
import pytest
import requests

from app.services import database, rag_run_executor_service, rag_run_service
from app.services.provider_request_queue import _context, run_provider_thread


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [0, 10])
async def test_provider_cancel_drains_actual_http_thread(execution_store, timeout_seconds):
    """参数为隔离仓储和总时限；有无总时限都须排空已发 HTTP 才能释放归属，无返回值。"""
    registry = rag_run_executor_service.RAGRunExecutorRegistry()
    execution_store.run["deadline_at"] = rag_run_service.deadline_after(timeout_seconds)
    entered, release = threading.Event(), threading.Event()
    contexts = []

    def request():
        """无参数；模拟已发请求，记录传入的运行上下文并等待真实结束。"""
        contexts.append(_context.get())
        entered.set()
        release.wait(2)

    async def work():
        """无参数；通过正式异步线程端口执行可观测的同步请求。"""
        return await run_provider_thread(request)

    task = asyncio.create_task(registry.execute(user_id="owner", run_id="run",
        timeout_seconds=timeout_seconds, runner_factory=work))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    cancel = asyncio.create_task(registry.cancel(user_id="owner", run_id="run"))
    while not contexts[0].cancel_event.is_set():
        await asyncio.sleep(0.01)
    assert contexts[0].run_id == "run"
    assert (contexts[0].deadline is None) is (timeout_seconds == 0)
    assert execution_store.run["execution_stopped_at"] is None
    release.set()
    await cancel
    with pytest.raises(asyncio.CancelledError):
        await task
    assert execution_store.terminals == ["cancelled"]


@pytest.mark.asyncio
async def test_provider_event_is_persisted_before_work_exits(execution_store, monkeypatch):
    """参数为运行仓储及事件端口；工作线程等候持久事件完成，终态不能先于等待事件。"""
    events = []

    async def append(**payload):
        """参数为真实事件载荷；模拟异步写入并记录收到的运行与阶段。"""
        await asyncio.sleep(0)
        events.append(payload)

    def request():
        """无参数；从发送线程发布等待事件，返回时事件已持久化。"""
        context = _context.get()
        context.on_event("provider_queued", {"estimated_wait_seconds": 12})
        assert len(events) == 1

    async def work():
        """无参数；通过正式发送线程端口传播事件上下文。"""
        await run_provider_thread(request)

    monkeypatch.setattr(rag_run_service, "append_event", append)
    await rag_run_executor_service.RAGRunExecutorRegistry().execute(user_id="owner", run_id="run",
        timeout_seconds=10, runner_factory=work)
    assert events[0]["payload"]["stage"] == "provider_queued"
    assert events[0]["run_id"] == "run"


@pytest.mark.asyncio
async def test_provider_wait_timeout_keeps_timeout_terminal(execution_store):
    """参数为运行仓储；总期限包括队列等待，唤醒线程后仍保存timed_out而不是failed。"""
    contexts = []

    def request():
        """无参数；模拟排队中线程，被截止时间取消后抛出明确队列取消。"""
        from app.services.provider_request_queue import check_provider_request_active

        context = _context.get()
        contexts.append(context)
        context.cancel_event.wait(2)
        check_provider_request_active()

    async def work():
        """无参数；使用真实线程排空端口等待准入。"""
        await run_provider_thread(request)

    with pytest.raises(TimeoutError):
        await rag_run_executor_service.RAGRunExecutorRegistry().execute(user_id="owner", run_id="run",
            timeout_seconds=0.05, runner_factory=work)
    assert contexts[0].cancel_event.is_set()
    assert execution_store.terminals == ["timed_out"]


@pytest.fixture(autouse=True)
def isolated_io(monkeypatch):
    """参数为补丁工具；阻断所有真实外联，返回计数并在结束时核对零调用。"""
    import socket

    calls = []

    def blocked(*args, **kwargs):
        """参数为任意外联请求；记录并立即失败，无成功返回。"""
        calls.append((args, kwargs))
        raise AssertionError("O5 offline test attempted external IO")

    for target, name in ((httpx.Client, "send"), (httpx.AsyncClient, "send"),
                         (requests.Session, "request"), (database, "get_pool"),
                         (socket, "create_connection")):
        monkeypatch.setattr(target, name, blocked)
    yield calls
    assert calls == []


@pytest.mark.asyncio
async def test_remote_cancel_does_not_claim_execution_stopped(monkeypatch):
    """参数为补丁工具；另一进程仍持有任务时只写信号，不能返回假终态。"""
    remote = {"run_id": "remote", "status": "running", "cancel_requested": True}
    finish = AsyncMock(return_value={**remote, "status": "cancelled"})
    monkeypatch.setattr(rag_run_service, "request_cancel", AsyncMock(return_value=remote))
    monkeypatch.setattr(rag_run_service, "get_run", AsyncMock(return_value=remote))
    monkeypatch.setattr(rag_run_service, "finish_run", finish)
    result = await rag_run_executor_service.RAGRunExecutorRegistry().cancel(
        user_id="owner", run_id="remote",
    )
    assert result["status"] == "running"
    assert result["cancel_requested"] is True
    finish.assert_not_awaited()


@pytest.mark.asyncio
async def test_deadline_scanner_requests_stop_without_unlocking_thread():
    """无参数；过期只要求所有者停止，不能扫描即写 terminal。"""
    repository = AsyncMock()
    repository.list_expired_active_runs.return_value = [{
        "user_id": "owner", "run_id": "remote", "status": "running",
    }]
    repository.get_run.return_value = {"run_id": "remote", "status": "running"}
    await rag_run_service.expire_overdue_runs(repository=repository)
    repository.finish_run.assert_not_awaited()
    repository.request_cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_terminal_callback_failure_remains_visible():
    """无参数；业务终态事务失败必须传播，不允许继续宣告运行保存成功。"""
    callback = AsyncMock(side_effect=OSError("session transaction failed"))
    with pytest.raises(OSError, match="session transaction failed"):
        await rag_run_executor_service.RAGRunExecutorRegistry._notify_terminal(
            callback=callback, run_id="run", status="failed", error="provider failed",
        )


class ExecutionStore:
    """保留跨执行器共享运行事实的隔离存储；不替换被测执行器或业务服务。"""

    def __init__(self):
        """无参数；建立一个具有期限的待运行操作。"""
        self.run = {"run_id": "run", "user_id": "owner", "status": "queued", "cancel_requested": False,
                    "owner_instance_id": None, "execution_stopped_at": None,
                    "deadline_at": datetime.now(timezone.utc) + timedelta(seconds=10)}
        self.terminals = []

    async def claim_execution(self, **payload):
        """参数为执行器身份；模拟持久 owner CAS，返回新取得身份的运行。"""
        if self.run["owner_instance_id"] is not None:
            return None
        self.run.update(owner_instance_id=payload["owner_instance_id"], status="running")
        return dict(self.run)

    async def record_execution_stopped(self, **payload):
        """参数为真正退出的执行器身份；只允许所属进程记录停止事实。"""
        assert payload["owner_instance_id"] == self.run["owner_instance_id"]
        self.run["execution_stopped_at"] = datetime.now(timezone.utc)
        return dict(self.run)

    async def get_run(self, **payload):
        """参数为查询条件；返回当前事实副本。"""
        return dict(self.run)

    async def request_cancel(self, **payload):
        """参数为用户操作；活动运行仅写信号，终态保持不变。"""
        if self.run["status"] in {"queued", "running"}:
            self.run["cancel_requested"] = True
        return dict(self.run)

    async def finish_run(self, **payload):
        """参数为终态；断言实际停止已发生，记录唯一结束事实。"""
        assert self.run["execution_stopped_at"] is not None
        if self.run["status"] not in {"queued", "running"}:
            return dict(self.run)
        self.terminals.append(payload["status"])
        self.run["status"] = payload["status"]
        return dict(self.run)


@pytest.fixture
def execution_store(monkeypatch):
    """参数为补丁工具；只替换数据库边界，返回供跨执行器共用的真实事实替身。"""
    store = ExecutionStore()
    monkeypatch.setattr(rag_run_executor_service, "get_rag_run_repository", lambda: store)
    monkeypatch.setattr(rag_run_service, "get_rag_run_repository", lambda: store)
    return store


@pytest.mark.asyncio
async def test_unbounded_execution_preserves_network_timeout(execution_store):
    """参数为隔离仓储；无总期限工作可完成，网络单次超时仍原值透传，无返回值。"""
    from app.services.provider_request_queue import provider_request_timeout

    execution_store.run["deadline_at"] = None
    registry = rag_run_executor_service.RAGRunExecutorRegistry()

    async def work():
        """无参数；跨异步调度后读取真实请求上下文，返回完成结果。"""
        await asyncio.sleep(0)
        assert _context.get().deadline is None
        assert provider_request_timeout(240) == 240
        return "finished"

    result = await registry.execute(user_id="owner", run_id="run", timeout_seconds=0, runner_factory=work)
    assert result == "finished"
    assert execution_store.run["execution_stopped_at"] is not None
    assert execution_store.terminals == []
    assert registry.active_count() == 0


@pytest.mark.asyncio
async def test_disabled_current_timeout_keeps_saved_deadline(execution_store):
    """参数为隔离仓储；当前配置无总时限也不能延长已保存操作的截止时间，无返回值。"""
    execution_store.run["deadline_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    work = AsyncMock()
    with pytest.raises(TimeoutError):
        await rag_run_executor_service.RAGRunExecutorRegistry().execute(
            user_id="owner", run_id="run", timeout_seconds=0, runner_factory=work,
        )
    work.assert_not_awaited()
    assert execution_store.terminals == ["timed_out"]


@pytest.mark.asyncio
async def test_remote_stop_waits_for_owner_cleanup(execution_store):
    """参数为隔离存储；远端取消先回信号，只有实际文件清理退出后才能变终态。"""
    owner = rag_run_executor_service.RAGRunExecutorRegistry()
    remote = rag_run_executor_service.RAGRunExecutorRegistry()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def work():
        """无参数；模拟已经进入的模型等待及必须等待完成的文件清理。"""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    task = asyncio.create_task(owner.execute(user_id="owner", run_id="run", timeout_seconds=10,
                                            runner_factory=work))
    await entered.wait()
    response = await remote.cancel(user_id="owner", run_id="run")
    assert response["status"] == "running"
    await asyncio.wait_for(cleaning.wait(), timeout=1)
    assert execution_store.terminals == []
    assert execution_store.run["execution_stopped_at"] is None
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert execution_store.terminals == ["cancelled"]


@pytest.mark.asyncio
async def test_two_executors_cannot_invoke_same_operation(execution_store):
    """参数为共享存储；两个进程注册同一操作，仅 owner CAS 胜者进入图。"""
    first, second = rag_run_executor_service.RAGRunExecutorRegistry(), rag_run_executor_service.RAGRunExecutorRegistry()
    started = asyncio.Event()
    calls = []

    async def work():
        """无参数；记录真正进入图的次数，等待停止。"""
        calls.append("graph")
        started.set()
        await asyncio.Event().wait()

    first.start(user_id="owner", run_id="run", timeout_seconds=10, runner_factory=work)
    await started.wait()
    with pytest.raises(RuntimeError, match="归属未取得"):
        await second.execute(user_id="owner", run_id="run", timeout_seconds=10, runner_factory=work)
    assert calls == ["graph"]
    await first.cancel(user_id="owner", run_id="run")
    assert execution_store.terminals == ["cancelled"]


@pytest.mark.asyncio
async def test_sync_request_cancel_and_deadline_have_real_stop(execution_store):
    """参数为隔离存储；同步 HTTP task 取消也排空受管工作并完成唯一终态。"""
    registry = rag_run_executor_service.RAGRunExecutorRegistry()
    started, stopped = asyncio.Event(), asyncio.Event()

    async def work():
        """无参数；记录进入及实际退出，供终态顺序断言。"""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    request = asyncio.create_task(registry.execute(user_id="owner", run_id="run", timeout_seconds=10,
                                                 runner_factory=work))
    await started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert stopped.is_set()
    assert execution_store.terminals == ["cancelled"]
    assert registry.active_count() == 0


@pytest.mark.asyncio
async def test_prestart_shutdown_still_records_stopped_owner(execution_store):
    """参数为隔离存储；注册后尚未进入图就关闭，仍记录确定退出且不调用图。"""
    registry = rag_run_executor_service.RAGRunExecutorRegistry()
    work = AsyncMock()
    registry.start(user_id="owner", run_id="run", timeout_seconds=10, runner_factory=work)
    await registry.close()
    work.assert_not_awaited()
    assert execution_store.terminals == ["cancelled"]


@pytest.mark.asyncio
async def test_deadline_finishes_only_after_work_finally(execution_store):
    """参数为隔离存储；执行期限取消工作并等待清理，之后生成 timed_out 事实。"""
    registry = rag_run_executor_service.RAGRunExecutorRegistry()
    cleaned = asyncio.Event()

    async def work():
        """无参数；模拟可取消的外部等待和 finally 清理。"""
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with pytest.raises(TimeoutError):
        await registry.execute(user_id="owner", run_id="run", timeout_seconds=0.01, runner_factory=work)
    assert cleaned.is_set()
    assert execution_store.terminals == ["timed_out"]


class TransactionStore:
    """用可回滚连接执行真实仓储代码；仅模拟数据库原语，不连接 PostgreSQL。"""

    def __init__(self):
        """无参数；建立固定范围 session 和跨连接事务锁。"""
        self.runs = {}
        self.session = {"session_id": "session", "user_id": "owner", "thread_id": "session",
                        "status": "created", "project_id": "project", "conversation_id": None,
                        "research_mode": "internal", "paper_ids": ["paper"], "attempt_count": 0,
                        "current_run_id": None}
        self.lock = asyncio.Lock()
        self.fail_binding = False

    @asynccontextmanager
    async def acquire(self):
        """无参数；返回隔离连接，事务语义由 transaction 提供。"""
        yield self

    @asynccontextmanager
    async def transaction(self):
        """无参数；串行化事务并在绑定失败时回滚操作和 session 两类记录。"""
        async with self.lock:
            before = deepcopy((self.runs, self.session))
            try:
                yield self
            except BaseException:
                self.runs, self.session = before
                raise

    async def fetchrow(self, sql, *args):
        """参数为仓储 SQL 及绑定值；实现领取需要的查询/约束，未知 SQL 立即失败。"""
        if sql.startswith("SELECT * FROM sessions"):
            return deepcopy(self.session) if args == ("session", "owner") else None
        if "INSERT INTO rag_runs" in sql:
            return self._insert(args)
        if "idempotency_key=$2" in sql:
            return next((deepcopy(row) for row in self.runs.values()
                         if row["user_id"] == args[0] and row["idempotency_key"] == args[1]), None)
        if "thread_id=$2" in sql:
            return next((deepcopy(row) for row in self.runs.values()
                         if row["user_id"] == args[0] and row["thread_id"] == args[1]
                         and row["status"] in {"queued", "running"}), None)
        if "run_id=$1 AND user_id=$2" in sql:
            return deepcopy(self.runs.get(args[0]))
        raise AssertionError(f"unexpected SQL: {sql}")

    def _insert(self, args):
        """参数为真实 INSERT 绑定值；模拟同 key 和活动 thread 唯一约束。"""
        if any(row["idempotency_key"] == args[5] or
               (row["thread_id"] == args[4] and row["status"] in {"queued", "running"})
               for row in self.runs.values()):
            return None
        keys = ("run_id", "user_id", "conversation_id", "session_id", "thread_id", "idempotency_key",
                "query", "metadata", "deadline_at", "created_at", "operation_kind", "target_run_id",
                "request_fingerprint", "expected_delivery_revision")
        row = {**dict(zip(keys, args)), "status": "queued", "owner_instance_id": None,
               "execution_stopped_at": None, "pending_artifact": None}
        row["metadata"] = json.loads(row["metadata"])
        self.runs[row["run_id"]] = row
        return deepcopy(row)

    async def execute(self, sql, *args):
        """参数为会话绑定 SQL；允许注入事务失败以验证 run 不会单独提交。"""
        assert "UPDATE sessions SET current_run_id=$1" in sql
        self.session.update(current_run_id=args[0], status="running",
                            attempt_count=self.session["attempt_count"] + args[1])
        if self.fail_binding:
            raise OSError("session bind failed")


@pytest.fixture
def operation_store(monkeypatch):
    """参数为补丁工具；只替换连接池，所有领取 SQL/业务判断仍使用正式代码。"""
    from app.repositories.rag_run_repository import RAGRunRepository

    store = TransactionStore()
    monkeypatch.setattr(database, "get_pool", AsyncMock(return_value=store))
    repository = RAGRunRepository()
    return store, repository


async def claim_test_operation(repository, *, key="request-key", query="研究问题", command=None):
    """参数为真实仓储及测试命令；走正式服务创建一项具有明确范围的研究操作。"""
    return await rag_run_service.create_operation(
        user_id="owner", conversation_id=None, session_id="session", thread_id="session",
        idempotency_key=key, query=query, deadline_at=rag_run_service.deadline_after(10),
        request_payload={"query": query, "session_id": "session"}, operation_kind="research",
        session_command=command or {"expected_status": "created", "increment_attempt": True,
            "project_id": "project", "conversation_id": None, "research_mode": "internal", "paper_ids": ["paper"]},
        repository=repository,
    )


@pytest.mark.asyncio
async def test_atomic_command_replay_and_changed_body_conflict(operation_store):
    """参数为隔离事务库；同 key 并发只增长一次 attempt，异正文明确拒绝。"""
    store, repository = operation_store
    first, repeated = await asyncio.gather(claim_test_operation(repository), claim_test_operation(repository))
    assert first[0]["run_id"] == repeated[0]["run_id"]
    assert [first[1], repeated[1]].count(True) == 1
    assert store.session["attempt_count"] == 1
    assert store.session["current_run_id"] == first[0]["run_id"]
    with pytest.raises(rag_run_service.IdempotencyConflict):
        await claim_test_operation(repository, query="不同问题")


@pytest.mark.asyncio
async def test_atomic_command_other_key_returns_active_identity(operation_store):
    """参数为隔离事务库；不同命令竞争同 thread 时返回当前执行身份且不增长次数。"""
    store, repository = operation_store
    first, _ = await claim_test_operation(repository)
    with pytest.raises(rag_run_service.ActiveRAGRunConflict) as caught:
        await claim_test_operation(repository, key="request-other")
    assert caught.value.run["run_id"] == first["run_id"]
    assert store.session["attempt_count"] == 1


@pytest.mark.asyncio
async def test_session_binding_failure_rolls_back_operation(operation_store):
    """参数为隔离事务库；session 绑定失败时 run 创建也回滚，不留下半领取状态。"""
    store, repository = operation_store
    store.fail_binding = True
    with pytest.raises(OSError, match="session bind failed"):
        await claim_test_operation(repository)
    assert store.runs == {}
    assert store.session["current_run_id"] is None
    assert store.session["status"] == "created"


@pytest.mark.asyncio
async def test_stopped_fact_required_before_retry(operation_store):
    """参数为隔离事务库；旧执行终态但没有实际停止事实时，官方 thread 不可重入。"""
    store, repository = operation_store
    first, _ = await claim_test_operation(repository)
    store.runs[first["run_id"]]["status"] = "failed"
    store.session["status"] = "failed"
    command = {"expected_status": "failed", "increment_attempt": True, "project_id": "project",
               "conversation_id": None, "research_mode": "internal", "paper_ids": ["paper"]}
    with pytest.raises(ValueError, match="无法确认上次研究执行已经退出"):
        await claim_test_operation(repository, key="retry-key", command=command)
    assert store.session["attempt_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["resume", "retry"])
async def test_real_command_route_separates_clarification_from_retry(operation_store, monkeypatch, command):
    """参数为隔离事务和命令；正式路由只能将用户答案用于 resume，retry 未答澄清不跑图。"""
    from app.api.routes import research_api
    from app.api.schemas.research import ResearchResumeRequest
    from app.services import result_delivery_service
    from app.services.research_session_lifecycle import ResearchSessionLifecycle

    store, repository = operation_store
    store.session.update(status="awaiting_clarification" if command == "resume" else "cancelled", attempt_count=1)
    sessions = AsyncMock()
    sessions.get_session.side_effect = lambda *args, **kwargs: deepcopy(store.session)
    lifecycle = ResearchSessionLifecycle(repository=sessions)
    monkeypatch.setattr(research_api, "get_research_session_lifecycle", lambda: lifecycle)
    monkeypatch.setattr(rag_run_service, "get_rag_run_repository", lambda: repository)
    monkeypatch.setattr(research_api, "persist_research_session_links", AsyncMock())
    monkeypatch.setattr(research_api, "get_shared_agent_state", AsyncMock(return_value={
        "values": {"user_query": "原目标", "clarification_question": "请指定比较对象"},
        "interrupts": [{"value": "请指定比较对象"}],
    }))
    monkeypatch.setattr(research_api, "build_project_context", AsyncMock(return_value={
        "user_id": "owner", "project_id": "project", "paper_ids": ["paper"], "library_ids": [],
        "filter_expr": "", "local_retrieval_allowed": True, "resolved_scope": None,
    }))
    graph = AsyncMock(return_value=({"report": "研究结果"}, "session"))
    monkeypatch.setattr(research_api, "invoke_graph", graph)

    async def execute(**payload):
        """参数为受管任务入口；路由测试仅替换调度端口，工作工厂照常执行。"""
        return await payload["runner_factory"]()

    async def deliver(**payload):
        """参数为真实路由产物；记录交付内容，不访问结果数据库。"""
        return {"run": {**payload["run"], "status": "completed"}, "response": payload["response"],
                "delivery_revision": 1, "persistence_status": "saved"}

    registry = AsyncMock()
    registry.execute.side_effect = execute
    monkeypatch.setattr(research_api, "get_rag_run_executor_registry", lambda: registry)
    monkeypatch.setattr(result_delivery_service, "deliver_result", deliver)
    if command == "resume":
        result = await research_api.resume_research_session("session", ResearchResumeRequest(answer="比较A和B"),
            user={"user_id": "owner"}, idempotency_key="resume-key")
        assert graph.await_args.args[0] == "比较A和B"
        assert graph.await_args.kwargs["execution_kind"] == "resume"
        assert result["status"] == "completed"
    else:
        result = await research_api.retry_research_session("session", user={"user_id": "owner"},
                                                          idempotency_key="retry-key")
        graph.assert_not_awaited()
        assert result["status"] == "awaiting_clarification"
        assert result["clarification_question"] == "请指定比较对象"
    assert store.session["attempt_count"] == 1
    assert len(store.runs) == 1


@pytest.mark.asyncio
async def test_old_session_callback_cannot_write_new_attempt():
    """无参数；旧操作即使看到相同 running 状态，也不得写入已改归属的新尝试。"""
    from app.services.research_session_lifecycle import InvalidResearchSessionTransition, ResearchSessionLifecycle

    repository = AsyncMock()
    repository.get_session.return_value = {"status": "running", "current_run_id": "new-run"}
    lifecycle = ResearchSessionLifecycle(repository=repository)
    with pytest.raises(InvalidResearchSessionTransition, match="拒绝旧回调"):
        await lifecycle.record_result(user_id="owner", session_id="session", result={"report": "旧答案"},
                                      expected_run_id="old-run")
    repository.transition_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_local_cancel_waits_for_same_cleanup(execution_store):
    """参数为隔离存储；重复取消不能打断第一次取消启动的清理或提前解锁 thread。"""
    registry = rag_run_executor_service.RAGRunExecutorRegistry()
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def work():
        """无参数；模拟必须完成后才能记录停止的 checkpoint/文件资源清理。"""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    registry.start(user_id="owner", run_id="run", timeout_seconds=10, runner_factory=work)
    await started.wait()
    first = asyncio.create_task(registry.cancel(user_id="owner", run_id="run"))
    await cleaning.wait()
    second = asyncio.create_task(registry.cancel(user_id="owner", run_id="run"))
    await asyncio.sleep(0.02)
    premature_stop = execution_store.run["execution_stopped_at"]
    release.set()
    await asyncio.gather(first, second)
    assert premature_stop is None
    assert execution_store.terminals == ["cancelled"]


@pytest.mark.asyncio
async def test_command_replay_precedes_session_status_validation(operation_store, monkeypatch):
    """参数为隔离事务；相同命令在 session 已 running 后回查原 run，不因状态变化冲突。"""
    from app.services.research_session_lifecycle import ResearchSessionLifecycle

    store, repository = operation_store
    run, _ = await claim_test_operation(repository)
    monkeypatch.setattr(rag_run_service, "get_rag_run_repository", lambda: repository)
    sessions = AsyncMock()
    sessions.get_session.return_value = deepcopy(store.session)
    lifecycle = ResearchSessionLifecycle(repository=sessions)
    replay, durable, created = await lifecycle.accept_operation(
        user_id="owner", session=deepcopy(store.session), operation_kind="research",
        idempotency_key="request-key", query="研究问题",
        request_payload={"query": "研究问题", "session_id": "session"},
        deadline_at=rag_run_service.deadline_after(10),
    )
    assert replay["run_id"] == run["run_id"]
    assert not created
    assert durable["attempt_count"] == 1


@pytest.mark.asyncio
async def test_research_stream_drains_graph_after_done(monkeypatch):
    """参数为补丁工具；真实研究消费入口在 done 后继续等待图 finally，不能提前提交。"""
    from app.api.routes import stream_api
    from app.api.schemas.research import ResearchStreamRequest

    cleaning, release = asyncio.Event(), asyncio.Event()
    closed = []

    async def graph():
        """无参数；输出结果后模拟仍在完成的 checkpoint 清理。"""
        try:
            yield {"type": "done", "report": "同次正文"}
        finally:
            cleaning.set()
            await release.wait()
            closed.append(True)

    async def payloads(source, **kwargs):
        """参数为隔离图；保留生成器逐项读取与尾部清理语义。"""
        async for item in source:
            yield item

    monkeypatch.setattr(stream_api, "_stream_graph_extra", lambda *args, **kwargs: {})
    monkeypatch.setattr(stream_api, "open_graph_stream", AsyncMock(return_value=("updates", graph(), "session")))
    monkeypatch.setattr(stream_api, "iter_research_payloads", payloads)
    task = asyncio.create_task(stream_api._consume_durable_stream(
        req=ResearchStreamRequest(query="问题"), conversation_id=None, project_context={},
        core_intent="internal_research", session={"session_id": "session"},
        lifecycle=AsyncMock(), user_id="owner", run={"run_id": "run"},
    ))
    await asyncio.wait_for(cleaning.wait(), timeout=1)
    assert not task.done()
    release.set()
    result = await task
    assert result["report"] == "同次正文"
    assert closed == [True]


@pytest.mark.asyncio
async def test_resume_route_rechecks_key_when_concurrent_command_changed_status(monkeypatch):
    """参数为补丁工具；同 key 首查落空后另一请求已领取，正式路由仍返回原运行。"""
    from app.api.routes import research_api
    from app.api.schemas.research import ResearchResumeRequest

    payload = {"session_id": "session", "answer": "用户答案"}
    existing = {"run_id": "same-run", "session_id": "session", "status": "running",
                "request_fingerprint": rag_run_service.operation_fingerprint(
                    operation_kind="resume", request_payload=payload)}
    monkeypatch.setattr(rag_run_service, "get_by_idempotency_key", AsyncMock(side_effect=[None, existing]))
    lifecycle = AsyncMock()
    lifecycle.get_session.return_value = {"session_id": "session", "status": "running",
                                          "current_run_id": "same-run", "allowed_actions": ["cancel"]}
    monkeypatch.setattr(research_api, "get_research_session_lifecycle", lambda: lifecycle)
    graph = AsyncMock()
    monkeypatch.setattr(research_api, "invoke_graph", graph)
    result = await research_api.resume_research_session("session", ResearchResumeRequest(answer="用户答案"),
        user={"user_id": "owner"}, idempotency_key="same-command-key")
    assert result["run_id"] == "same-run"
    assert result["operation_replayed"] is True
    graph.assert_not_awaited()
