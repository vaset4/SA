"""查询批次与适配器退出的生命周期回归，线程顺序由事件控制。

作者：xxx
创建时间：2026-09-24 10:36:32
"""

from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import pytest

from app.config import get_settings
from app.models.research_planner_contracts import ResearchUnit
from app.nodes import rag_node
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError,
    PlannerExecutionContext,
    _call_adapter,
)
from app.services.internal_research.research_planner_lifecycle import settle_owned_work
from research_planner_test_utils import make_unit
from o2_evidence_test_utils import bind_publication, scope
from app.services.knowledge_access_service import read_knowledge

_SIGNAL_TIMEOUT_SECONDS = 5.0
_PENDING_WINDOW_SECONDS = 0.05
_ADAPTER_TIMEOUT_SECONDS = 0.01
_NO_TIMEOUT_SECONDS = 30.0
_PAPER = "paper-a"


@dataclass
class _QueryGate:
    """单次同步查询的返回值、进入通知及线程释放信号。"""

    outcome: Any
    started: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    release: threading.Event = field(default_factory=threading.Event)
    calls: int = 0


class _ControlledRead:
    """保留真实线程执行，只用事件替代外部检索和阻塞时间。"""

    def __init__(self, outcomes: dict[str, Any]) -> None:
        """参数为按定位词配置的结果或异常；建立当前事件循环的线程通知端口。"""
        self.loop = asyncio.get_running_loop()
        self.gates = {query: _QueryGate(outcome) for query, outcome in outcomes.items()}
        self.lock = threading.Lock()

    def __call__(self, _store: Any, query: str, *_args: Any) -> Any:
        """参数沿用同步检索调用，返回指定命中或抛出原异常；等待测试显式放行线程。"""
        gate = self.gates[query]
        with self.lock:
            gate.calls += 1
        self.loop.call_soon_threadsafe(gate.started.set)
        try:
            assert gate.release.wait(_SIGNAL_TIMEOUT_SECONDS), f"查询 {query} 未被测试释放"
            if isinstance(gate.outcome, BaseException):
                raise gate.outcome
            return deepcopy(gate.outcome)
        finally:
            self.loop.call_soon_threadsafe(gate.finished.set)

    def release_all(self) -> None:
        """无参数、无返回值；失败断言之后也释放所有测试线程，避免阻塞测试进程退出。"""
        for gate in self.gates.values():
            gate.release.set()


@dataclass
class _DirectAdapter:
    """记录完整直接正文适配器调用次数，保持实际批次和父块调用链。"""

    state: dict[str, Any] = field(default_factory=lambda: {"project_scope": scope()})
    task: asyncio.Task[Any] | None = None
    calls: int = 0

    async def __call__(self, unit: ResearchUnit, context: PlannerExecutionContext) -> list[dict[str, Any]]:
        """参数为研究单元与上下文；返回真实直接正文适配器的证据结果。"""
        self.calls += 1
        self.task = asyncio.current_task()
        return await rag_node._planner_direct_text_records(self.state, unit, context)


@dataclass
class _CleanupAdapter:
    """进入取消清理后由事件控制结束，暴露 wait_for 清理期间再次取消的竞态。"""

    cleanup_error: Exception | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    cleaning: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[Any] | None = None
    calls: int = 0

    async def __call__(self, _unit: ResearchUnit, _context: PlannerExecutionContext) -> list[dict[str, Any]]:
        """参数为适配器输入；取消后等待放行，再保留取消或抛出指定清理异常。"""
        self.calls += 1
        self.task = asyncio.current_task()
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cleaning.set()
            await self.release.wait()
            if self.cleanup_error is not None:
                raise self.cleanup_error
            raise


def _configure(monkeypatch: pytest.MonkeyPatch, *, retries: int = 0, timeout: float = _NO_TIMEOUT_SECONDS) -> None:
    """参数为配置夹具、额外重试次数与超时；无返回值，关闭真实查询扩展并清零退避。"""
    cfg = get_settings()
    monkeypatch.setattr(cfg, "rag_planner_evidence_max_retries", retries)
    monkeypatch.setattr(cfg, "rag_planner_evidence_retry_backoff_seconds", 0.0)
    monkeypatch.setattr(cfg, "rag_planner_evidence_timeout_seconds", timeout)
    monkeypatch.setattr(rag_node.settings, "rag_planner_lightrag_locator_enabled", False)


def _prepare_search(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[str, list[dict[str, Any]] | BaseException],
    *,
    parent: bool = False,
) -> tuple[_ControlledRead, ResearchUnit]:
    """参数为查询结果和父块开关；返回线程控制器与按声明顺序构造的研究单元。"""
    unique = {}
    for outcome in outcomes.values():
        if isinstance(outcome, list):
            for row in outcome:
                unique.setdefault(row["chunk_id"], row)
    bind_publication(monkeypatch, list(unique.values()))
    search = _ControlledRead(outcomes)
    monkeypatch.setattr(rag_node, "_search_chunks", search)
    monkeypatch.setattr(rag_node, "get_vector_store", Mock(return_value=object()))
    unit = ResearchUnit.model_validate(make_unit("unit-a", paper_ids=[_PAPER]))
    policy = unit.evidence_policy.model_copy(update={
        "locator_queries": list(outcomes),
        "evidence_granularity": "parent_chunk" if parent else "child_chunk",
    })
    return search, unit.model_copy(update={"evidence_policy": policy})


def _hit(chunk_id: str, text: str) -> dict[str, Any]:
    """参数为块身份及片段，返回等分数、等章节的检索命中以便验证稳定顺序。"""
    ordinal = {"a": 0, "b": 1, "shared": 2, "last": 3}[chunk_id]
    return {
        "paper_id": _PAPER, "chunk_id": f"v3:{'b' * 32}:{ordinal}", "parent_id": f"parent-{chunk_id}",
        "preview": text, "text": text, "full_text": text, "section": "Method", "score": 0.5,
    }


async def _wait_signal(event: asyncio.Event) -> None:
    """参数为执行事件；无返回值，缺失通知以明确超时失败结束测试。"""
    await asyncio.wait_for(event.wait(), timeout=_SIGNAL_TIMEOUT_SECONDS)


async def _assert_pending(task: asyncio.Task[Any]) -> None:
    """参数为被测请求任务；断言阻塞工作未放行时请求不能退出，且不取消被测任务。"""
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=_PENDING_WINDOW_SECONDS)
    assert not task.done()


async def _request_with_release(adapter: Any, unit: ResearchUnit, released: asyncio.Event) -> list[dict[str, Any]]:
    """参数为适配器、单元和读取占用释放通知；返回证据，finally 模拟请求释放索引占用。"""
    try:
        return await _call_adapter(adapter, unit, PlannerExecutionContext(question="query"))
    finally:
        released.set()


def test_failed_query_finishes_batch_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为打补丁夹具；验证快失败加慢线程不能启动重叠重试，最终保留原失败原因。"""
    _configure(monkeypatch, retries=1)

    async def scenario() -> None:
        """无参数、无返回值；先结束失败查询，确认兄弟仍占用时整批不可退出。"""
        failure = OSError("query connection lost")
        search, unit = _prepare_search(monkeypatch, {"q1": failure, "q2": []})
        adapter = _DirectAdapter()
        task = asyncio.create_task(_call_adapter(adapter, unit, PlannerExecutionContext()))
        try:
            await asyncio.gather(*(_wait_signal(gate.started) for gate in search.gates.values()))
            search.gates["q1"].release.set()
            await _wait_signal(search.gates["q1"].finished)
            await _assert_pending(task)
            assert adapter.calls == 1
            assert search.gates["q2"].calls == 1
            search.gates["q2"].release.set()
            with pytest.raises(EvidenceCapabilityError) as caught:
                await task
            assert caught.value.__cause__ is failure
            assert adapter.calls == 2
            assert all(gate.calls == 2 for gate in search.gates.values())
        finally:
            search.release_all()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_multiple_failures_use_query_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为夹具；验证后声明查询先失败时，仍按声明顺序抛出首条查询的异常。"""
    _configure(monkeypatch)

    async def scenario() -> None:
        """无参数、无返回值；反向释放两条失败查询并检查异常对象身份。"""
        first_failure = ValueError("first declared failure")
        search, unit = _prepare_search(monkeypatch, {"q1": first_failure, "q2": OSError("finished first")})
        task = asyncio.create_task(rag_node._planner_direct_text_records({"project_scope": scope()}, unit, None))
        try:
            await asyncio.gather(*(_wait_signal(gate.started) for gate in search.gates.values()))
            search.gates["q2"].release.set()
            await _wait_signal(search.gates["q2"].finished)
            await _assert_pending(task)
            search.gates["q1"].release.set()
            with pytest.raises(ValueError) as caught:
                await task
            assert caught.value is first_failure
        finally:
            search.release_all()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_successful_queries_keep_merge_and_deduplication_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为夹具；验证完成顺序反转仍保留首条查询的重复块片段与稳定记录顺序。"""
    _configure(monkeypatch)

    async def scenario() -> None:
        """无参数、无返回值；后声明查询先完成，检查真实转换结果的块身份与片段。"""
        search, unit = _prepare_search(monkeypatch, {
            "q1": [_hit("shared", "first version"), _hit("a", "second block")],
            "q2": [_hit("shared", "later version"), _hit("b", "third block")],
        })
        task = asyncio.create_task(rag_node._planner_direct_text_records({"project_scope": scope()}, unit, None))
        try:
            await asyncio.gather(*(_wait_signal(gate.started) for gate in search.gates.values()))
            search.gates["q2"].release.set()
            await _wait_signal(search.gates["q2"].finished)
            search.gates["q1"].release.set()
            records = (await task).records
            assert [record["chunk_id"] for record in records] == [_hit(key, "")["chunk_id"] for key in ("shared", "a", "b")]
            assert records[0]["excerpt"] == "first version"
            assert len({record["evidence_id"] for record in records}) == 3
        finally:
            search.release_all()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_repeated_cancellation_waits_for_queries_and_skips_parent_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为夹具；验证重复取消下线程结束先于占用释放，不启动重试或后续父块回读。"""
    _configure(monkeypatch, retries=2)

    async def scenario() -> None:
        """无参数、无返回值；取消后让一条查询失败，仍保持取消身份和完整线程收尾。"""
        failure = OSError("query failed while cancelling")
        search, unit = _prepare_search(monkeypatch, {"q1": failure, "q2": []}, parent=True)
        parent_read = Mock(side_effect=AssertionError("取消后不应回读父块"))
        monkeypatch.setattr(rag_node, "_planner_read_direct_materials", parent_read)
        adapter = _DirectAdapter()
        released = asyncio.Event()
        task = asyncio.create_task(_request_with_release(adapter, unit, released))
        try:
            await asyncio.gather(*(_wait_signal(gate.started) for gate in search.gates.values()))
            task.cancel("first user cancellation")
            await _assert_pending(task)
            task.cancel("second user cancellation")
            await _assert_pending(task)
            assert not released.is_set()
            assert adapter.calls == 1
            search.release_all()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.__cause__.__cause__ is failure
            assert released.is_set()
            assert all(gate.finished.is_set() for gate in search.gates.values())
            assert adapter.calls == 1
            parent_read.assert_not_called()
        finally:
            search.release_all()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("user_cancels", [0, 2])
def test_timeout_during_queries_waits_for_thread_completion(
    monkeypatch: pytest.MonkeyPatch, user_cancels: int,
) -> None:
    """参数为夹具与清理期间取消次数；验证真实查询线程超时后的占用释放和失败链。"""
    _configure(monkeypatch, timeout=_ADAPTER_TIMEOUT_SECONDS)

    async def scenario() -> None:
        """无参数、无返回值；在线程仍阻塞时让适配器到期，放行后保留最终终止身份。"""
        failure = OSError("query failed while timing out")
        search, unit = _prepare_search(monkeypatch, {"q1": failure, "q2": []})
        adapter = _DirectAdapter()
        released = asyncio.Event()
        task = asyncio.create_task(_request_with_release(adapter, unit, released))
        try:
            await asyncio.gather(*(_wait_signal(gate.started) for gate in search.gates.values()))
            await _assert_pending(task)
            assert adapter.task is not None and adapter.task.cancelling() == 1
            for index in range(user_cancels):
                task.cancel(f"user cancellation {index}")
                await _assert_pending(task)
            assert not released.is_set()
            search.release_all()
            if user_cancels:
                with pytest.raises(asyncio.CancelledError) as caught:
                    await task
                termination = caught.value
            else:
                with pytest.raises(EvidenceCapabilityError) as caught_timeout:
                    await task
                termination = caught_timeout.value.__cause__
                assert isinstance(termination, TimeoutError)
                assert str(termination) == f"取证调用超过 {_ADAPTER_TIMEOUT_SECONDS:g} 秒未返回"
            assert isinstance(termination.__cause__, asyncio.CancelledError)
            assert termination.__cause__.__cause__ is failure
            assert all(gate.finished.is_set() for gate in search.gates.values())
            assert adapter.calls == 1
            assert released.is_set()
        finally:
            search.release_all()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("user_cancels", [0, 1, 2])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_timeout_cleanup_keeps_ownership_and_cancellation_priority(
    monkeypatch: pytest.MonkeyPatch, user_cancels: int, cleanup_fails: bool,
) -> None:
    """参数为夹具、清理期间取消次数与清理故障开关；验证退出身份、异常链和适配器所有权。"""
    _configure(monkeypatch, timeout=_ADAPTER_TIMEOUT_SECONDS)

    async def scenario() -> None:
        """无参数、无返回值；真实超时进入清理后再取消请求，直到显式放行才准退出。"""
        error = ValueError("adapter cleanup failed") if cleanup_fails else None
        adapter = _CleanupAdapter(cleanup_error=error)
        unit = ResearchUnit.model_validate(make_unit("unit-a"))
        released = asyncio.Event()
        task = asyncio.create_task(_request_with_release(adapter, unit, released))
        try:
            await _wait_signal(adapter.cleaning)
            for index in range(user_cancels):
                task.cancel(f"user cancellation {index}")
                await _assert_pending(task)
            assert not released.is_set()
            assert adapter.task is not None and not adapter.task.done()
            adapter.release.set()
            if user_cancels:
                with pytest.raises(asyncio.CancelledError) as caught:
                    await task
                termination = caught.value
            else:
                with pytest.raises(EvidenceCapabilityError) as caught_timeout:
                    await task
                termination = caught_timeout.value.__cause__
                assert isinstance(termination, TimeoutError)
                assert str(termination) == f"取证调用超过 {_ADAPTER_TIMEOUT_SECONDS:g} 秒未返回"
            if cleanup_fails:
                assert termination.__cause__ is error
            assert adapter.calls == 1
            assert adapter.task.done()
            assert released.is_set()
        finally:
            adapter.release.set()
            await asyncio.gather(task, return_exceptions=True)
            if adapter.task is not None:
                await asyncio.gather(adapter.task, return_exceptions=True)

    asyncio.run(scenario())


def test_user_cancellation_during_cleanup_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为夹具；验证首次用户取消进入适配器清理后，再次取消不会遗留工作或触发重试。"""
    _configure(monkeypatch, retries=2)

    async def scenario() -> None:
        """无参数、无返回值；两次用户取消跨过清理边界，最后按取消退出。"""
        adapter = _CleanupAdapter()
        unit = ResearchUnit.model_validate(make_unit("unit-a"))
        released = asyncio.Event()
        task = asyncio.create_task(_request_with_release(adapter, unit, released))
        try:
            await _wait_signal(adapter.entered)
            task.cancel("first cancellation")
            await _wait_signal(adapter.cleaning)
            task.cancel("second cancellation")
            await _assert_pending(task)
            assert not released.is_set()
            adapter.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert adapter.task is not None and adapter.task.done()
            assert adapter.calls == 1
            assert released.is_set()
        finally:
            adapter.release.set()
            await asyncio.gather(task, return_exceptions=True)
            if adapter.task is not None:
                await asyncio.gather(adapter.task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_first", [False, True])
def test_last_completion_and_cancellation_in_same_loop_turn(cancel_first: bool) -> None:
    """参数控制同轮完成与取消的入队顺序；验证结果已就绪也不能抹掉调用方取消。"""
    async def scenario() -> None:
        """无参数、无返回值；用同轮回调复现最后一个查询完成与取消同时抵达。"""
        loop = asyncio.get_running_loop()
        work = loop.create_future()
        task = asyncio.create_task(settle_owned_work(work))
        if cancel_first:
            loop.call_soon(task.cancel, "same-turn cancellation")
        loop.call_soon(work.set_result, [[_hit("last", "final block")]])
        if not cancel_first:
            loop.call_soon(task.cancel, "same-turn cancellation")
        outcome, cancellation = await task
        assert outcome == [[_hit("last", "final block")]]
        assert isinstance(cancellation, asyncio.CancelledError)
        assert str(cancellation) == "same-turn cancellation"
        assert work.done() and not work.cancelled()

    asyncio.run(scenario())


def test_cancellation_during_publication_read_waits_for_owned_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为夹具；权威正文批量读取中取消，必须等待其线程结束才释放发布占用。"""
    _configure(monkeypatch, retries=2)

    async def scenario() -> None:
        """无参数、无返回值；查询完成后在发布 IO 内重复取消请求。"""
        rows = [_hit("a", "first"), _hit("b", "second")]
        search, unit = _prepare_search(monkeypatch, {"q1": rows}, parent=True)
        batch_id = "b" * 32
        publication = _ControlledRead({"batch": {batch_id: {"paper_id": _PAPER, "batch_id": batch_id, "chunks": rows}}})
        def reader(batch_ids, **kwargs):
            """参数为批次和授权；等待测试放行后返回完整发布清单。"""
            return publication(None, "batch")
        async def read(request):
            """参数为正文请求；运行正式知识模块，只替换发布存储。"""
            return await read_knowledge(request, publication_reader=reader)
        monkeypatch.setattr(rag_node, "read_knowledge", read)
        monkeypatch.setattr(rag_node.settings, "rag_auto_expand_parent_limit", 2)
        search.release_all()
        released = asyncio.Event()
        adapter = _DirectAdapter()
        task = asyncio.create_task(_request_with_release(adapter, unit, released))
        try:
            await _wait_signal(publication.gates["batch"].started)
            task.cancel("publication read cancellation")
            await _assert_pending(task)
            task.cancel("repeat publication read cancellation")
            await _assert_pending(task)
            assert not released.is_set()
            publication.release_all()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert publication.gates["batch"].finished.is_set()
            assert publication.gates["batch"].calls == 1
            assert adapter.calls == 1
            assert released.is_set()
        finally:
            search.release_all()
            publication.release_all()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
