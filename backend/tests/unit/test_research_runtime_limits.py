"""研究无总时限及配置边界回归。作者：xxx，2026-10-07 15:00:00。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.services import rag_run_service


@pytest.mark.parametrize("seconds", [0, 1, 300])
def test_execution_settings_accept_explicit_unbounded_seconds(seconds):
    """参数为总秒数；0 和正数可配置，关闭总时限不会改变单请求超时，无返回值。"""
    settings = Settings(_env_file=None, graph_timeout_seconds=seconds,
        deep_research_graph_timeout_seconds=seconds, rag_planner_max_parallel_units=0)
    assert settings.graph_timeout_seconds == settings.deep_research_graph_timeout_seconds == seconds
    assert settings.rag_planner_max_parallel_units == 0
    assert settings.llm_timeout > 0


@pytest.mark.parametrize("field", ["graph_timeout_seconds", "deep_research_graph_timeout_seconds",
                                   "rag_planner_max_parallel_units", "rag_research_planner_max_units"])
def test_execution_settings_reject_negative_limits(field):
    """参数为配置名；负数不是禁用标识，应在配置加载时明确失败，无返回值。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: -1})


@pytest.mark.parametrize("limit", [0, 8, 20])
def test_action_batch_settings_accept_zero_and_preserve_positive_limits(limit):
    """参数为每批动作上限；0关闭数量门槛，显式正值原样保留，无返回值。"""
    settings = Settings(_env_file=None, rag_research_planner_max_units=limit)
    assert settings.rag_research_planner_max_units == limit


def test_deadline_conversion_has_no_artificial_far_future():
    """无参数；0 返回空期限，正值生成真实 UTC 期限，负值拒绝，无返回值。"""
    assert rag_run_service.deadline_after(0) is None
    before = datetime.now(timezone.utc)
    deadline = rag_run_service.deadline_after(10)
    assert deadline.tzinfo is not None
    assert 0 < (deadline - before).total_seconds() <= 11
    with pytest.raises(ValueError):
        rag_run_service.deadline_after(-1)


@pytest.mark.asyncio
async def test_operation_accepts_null_deadline_without_changing_identity():
    """无参数；无总期限仍经原操作领取和幂等校验进入持久层，无返回值。"""
    async def save(*, operation, session_command):
        """参数为正式操作与会话命令；返回原操作，保留所有幂等身份。"""
        return operation, True

    repo = SimpleNamespace(create_operation=AsyncMock(side_effect=save))
    run, created = await rag_run_service.create_operation(user_id="owner", conversation_id="conversation",
        session_id="session", thread_id="thread", idempotency_key="unique-operation", query="question",
        deadline_at=None, request_payload={"query": "question"}, repository=repo)
    assert created and run["deadline_at"] is None
    assert run["thread_id"] == "thread" and run["request_fingerprint"]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, 1])
async def test_graph_invocation_accepts_disabled_deadline(monkeypatch, timeout):
    """参数为补丁和总时限；正式图适配器允许异步完成且不把 0 当即刻超时，无返回值。"""
    from app.api.facades import graph_facade

    async def invoke(*args, **kwargs):
        """参数为图输入；经过真实事件循环调度后返回产物。"""
        await asyncio.sleep(0)
        return {"answer": "complete"}

    monkeypatch.setattr(graph_facade.observability, "record_timed_event", lambda *args, **kwargs: None)
    result = await graph_facade._invoke_agent(SimpleNamespace(ainvoke=invoke), {}, config={},
        timeout=timeout, intent=None, started_at=0, query="question", session_id="session")
    assert result["answer"] == "complete"


@pytest.mark.asyncio
async def test_graph_invocation_keeps_positive_deadline(monkeypatch):
    """参数为补丁；正值总期限仍中止图工作并返回 504，清理已完成，无返回值。"""
    from app.api.facades import graph_facade

    stopped = asyncio.Event()

    async def invoke(*args, **kwargs):
        """参数为图输入；阻塞直至取消，并记录 finally 已执行，无返回值。"""
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(graph_facade.observability, "record_timed_event", lambda *args, **kwargs: None)
    with pytest.raises(HTTPException) as error:
        await graph_facade._invoke_agent(SimpleNamespace(ainvoke=invoke), {}, config={},
            timeout=0.01, intent=None, started_at=0, query="question", session_id="session")
    assert error.value.status_code == 504
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_graph_without_deadline_preserves_inner_timeout(monkeypatch):
    """参数为补丁；无总期限时内部 IO 超时保留来源，不谎报为总运行 0 秒超时，无返回值。"""
    from app.api.facades import graph_facade

    failure = TimeoutError("single request read timed out")
    graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=failure))
    monkeypatch.setattr(graph_facade.observability, "record_timed_event", lambda *args, **kwargs: None)
    with pytest.raises(TimeoutError) as error:
        await graph_facade._invoke_agent(graph, {}, config={}, timeout=0,
            intent=None, started_at=0, query="question", session_id="session")
    assert error.value is failure
