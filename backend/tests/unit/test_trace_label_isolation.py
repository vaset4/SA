"""覆盖并发、嵌套和线程传播中的模型调用标签隔离。

作者：xxx
创建时间：2026-09-24 10:28:59（北京时间）
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from app.middleware import trace_context as tracing

_ASYNC_TEST_TIMEOUT_SECONDS = 5.0


@pytest.fixture
def trace() -> Iterator[tracing.TraceContext]:
    """无参数；提供带默认节点和阶段的请求上下文，结束时恢复原上下文。"""
    ctx = tracing.TraceContext(
        request_id="label-request", node="request_node", stage="request_stage",
    )
    token = tracing.set_trace_ctx(ctx)
    try:
        yield ctx
    finally:
        tracing.reset_trace_ctx(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("second_labels", [
    ("planner_execute", "claim_draft"),
    ("other_node", "other_stage"),
])
async def test_concurrent_repair_call_keeps_own_labels(
    trace: tracing.TraceContext, second_labels: tuple[str, str],
) -> None:
    """参数为请求和第二单元标签；首单元先退出时，另一单元回修仍保留自身标签和证据身份。"""
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    first_finished = asyncio.Event()

    async def first_unit() -> tracing.ModelCallRecord:
        """无参数；等待兄弟进入标签作用域后结束首单元，返回首条调用记录。"""
        with tracing.trace_label("planner_execute", "claim_draft"):
            with tracing.evidence_scope("claim_context", unit_id="unit-a"):
                call = trace.begin_call()
                first_entered.set()
                await second_entered.wait()
        first_finished.set()
        return call

    async def second_unit() -> tuple[tracing.ModelCallRecord, tracing.ModelCallRecord]:
        """无参数；在首单元退出后模拟结构化输出回修，返回初次和回修调用记录。"""
        await first_entered.wait()
        with tracing.trace_label(*second_labels):
            with tracing.evidence_scope("claim_context", unit_id="unit-b"):
                initial = trace.begin_call()
                second_entered.set()
                await first_finished.wait()
                return initial, trace.begin_call()

    # 1. 【调用跟踪】【并发回修】事件确定先进入后退出的交错顺序，不依赖等待时长碰运气
    with tracing.trace_label("rag_answer", "planner_execute"):
        first, (second, repaired) = await asyncio.wait_for(
            asyncio.gather(first_unit(), second_unit()), timeout=_ASYNC_TEST_TIMEOUT_SECONDS,
        )
        parent_call = trace.begin_call()
    assert (first.node, first.stage) == ("planner_execute", "claim_draft")
    assert (second.node, second.stage) == second_labels
    assert (repaired.node, repaired.stage) == second_labels
    assert (parent_call.node, parent_call.stage) == ("rag_answer", "planner_execute")
    assert first.evidence_scope_id is not None
    assert second.evidence_scope_id is not None
    assert first.evidence_scope_id != second.evidence_scope_id == repaired.evidence_scope_id
    assert parent_call.evidence_scope_id is None
    assert trace.model_calls == [first, second, repaired, parent_call]
    assert [call.call_id for call in trace.model_calls] == [
        f"{trace.request_id}:call-{index}" for index in range(1, len(trace.model_calls) + 1)
    ]
    assert (trace.node, trace.stage) == ("request_node", "request_stage")


def test_nested_partial_labels_restore_after_exception(trace: tracing.TraceContext) -> None:
    """参数为请求；部分标签逐层继承，异常退出恢复父标签且不改变请求默认值。"""
    failure = ValueError("claim repair failed")
    with tracing.trace_label(node="planner_execute") as active:
        assert active is trace
        with tracing.trace_label(stage="claim_draft"):
            child = trace.begin_call()
            with pytest.raises(ValueError) as caught:
                with tracing.trace_label(node="repair_node"):
                    failed = trace.begin_call()
                    raise failure
            restored_child = trace.begin_call()
        restored_parent = trace.begin_call()
    base = trace.begin_call()
    assert caught.value is failure
    assert (child.node, child.stage) == ("planner_execute", "claim_draft")
    assert (failed.node, failed.stage) == ("repair_node", "claim_draft")
    assert (restored_child.node, restored_child.stage) == (child.node, child.stage)
    assert (restored_parent.node, restored_parent.stage) == ("planner_execute", "request_stage")
    assert (base.node, base.stage) == ("request_node", "request_stage")


def _record_thread_calls() -> tuple[tracing.ModelCallRecord, ...]:
    """无参数；记录线程继承、临时切换与恢复后的调用，返回三条请求内记录。"""
    ctx = tracing.get_trace_ctx()
    assert ctx is not None
    inherited = ctx.begin_call()
    with tracing.trace_label("thread_worker", "thread_repair"):
        nested = ctx.begin_call()
    return inherited, nested, ctx.begin_call()


@pytest.mark.asyncio
async def test_thread_labels_propagate_without_changing_parent(trace: tracing.TraceContext) -> None:
    """参数为请求；线程继承当前标签，线程内切换不影响父任务，调用仍汇总到同一请求。"""
    with tracing.trace_label("planner_execute", "claim_draft"):
        thread_calls = await asyncio.to_thread(_record_thread_calls)
        parent_call = trace.begin_call()
    assert [(call.node, call.stage) for call in thread_calls] == [
        ("planner_execute", "claim_draft"),
        ("thread_worker", "thread_repair"),
        ("planner_execute", "claim_draft"),
    ]
    assert (parent_call.node, parent_call.stage) == ("planner_execute", "claim_draft")
    assert trace.model_calls == [*thread_calls, parent_call]
    assert len({call.call_id for call in trace.model_calls}) == len(trace.model_calls)
    assert (trace.node, trace.stage) == ("request_node", "request_stage")


def test_replaced_context_with_same_request_id_has_independent_labels(trace: tracing.TraceContext) -> None:
    """参数为请求；同名新请求使用自己的默认标签，恢复原请求后仍能读取原作用域标签。"""
    other = tracing.TraceContext(
        request_id=trace.request_id, node="new_request_node", stage="new_request_stage",
    )
    with tracing.trace_label("planner_execute", "claim_draft"):
        token = tracing.set_trace_ctx(other)
        try:
            independent = other.begin_call()
            with tracing.trace_label(stage="new_claim") as active:
                assert active is other
                new_child = other.begin_call()
            new_parent = other.begin_call()
        finally:
            tracing.reset_trace_ctx(token)
        restored = trace.begin_call()
    assert (independent.node, independent.stage) == ("new_request_node", "new_request_stage")
    assert (new_child.node, new_child.stage) == ("new_request_node", "new_claim")
    assert (new_parent.node, new_parent.stage) == (independent.node, independent.stage)
    assert (restored.node, restored.stage) == ("planner_execute", "claim_draft")
    assert other.model_calls == [independent, new_child, new_parent]
    assert trace.model_calls == [restored]


def test_label_without_request_does_not_attach_to_later_context() -> None:
    """无参数；无请求时标签作用域返回 None，也不向作用域内后建的请求泄漏标签。"""
    assert tracing.get_trace_ctx() is None
    with tracing.trace_label("orphan_node", "orphan_stage") as active:
        assert active is None
        ctx = tracing.TraceContext(request_id="new-request")
        token = tracing.set_trace_ctx(ctx)
        try:
            call = ctx.begin_call()
        finally:
            tracing.reset_trace_ctx(token)
    assert (call.node, call.stage) == ("unknown", "unknown")
