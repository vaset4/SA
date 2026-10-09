"""
研究路由 durable session 回归测试。

作者: xxx
时间: 2026-08-18 16:00:00
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.agents.agent import INTENT_INTERNAL_RESEARCH
import app.api.routes.research_api as research_api
import app.api.routes.stream_api as stream_api
from app.api.schemas.research import ResearchRequest, ResearchResumeRequest, ResearchStreamRequest
from app.services.research_session_lifecycle import (
    ACTION_RETRY,
    InvalidResearchSessionTransition,
)


def _external_project_context() -> dict:
    """返回无需本地论文范围的外部研究上下文。"""
    return {
        "user_id": "user-1",
        "project_id": "",
        "library_ids": [],
        "paper_ids": [],
        "filter_expr": "",
        "resolved_scope": None,
        "local_retrieval_allowed": False,
    }


class RecordingLifecycle:
    """记录 route 对生命周期服务的调用顺序与终态。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def prepare_command_session(self, **payload):
        """模拟 create-before-run 并返回 running session。"""
        self.calls.append(("prepare", str(payload.get("session_id") or "new")))
        return {
            "session_id": "session-durable",
            "conversation_id": payload.get("conversation_id"),
            "thread_id": "session-durable",
            "status": "running",
            "attempt_count": 1,
        }

    async def accept_operation(self, **payload):
        """参数为用户命令；模拟原子领取端口，已有执行与冲突由独立仓储回归覆盖。"""
        session = {**payload["session"], "status": "running", "current_run_id": "run-1"}
        if payload["operation_kind"] in {"resume", "retry"}:
            action = getattr(self, payload["operation_kind"])
            session = await action(session_id=session["session_id"])
        return {"run_id": "run-1", "status": "queued", "session_id": session["session_id"]}, session, True

    async def resolve_scope_request(self, **payload):
        """模拟新请求直接采用显式 scope。"""
        return {
            "project_id": payload.get("project_id"),
            "paper_ids": payload.get("paper_ids") or [],
            "research_mode": None,
        }

    async def record_result(self, **payload):
        """记录 graph 成功产物并返回 completed session。"""
        self.calls.append(("result", str(payload["session_id"])))
        return {
            "session_id": payload["session_id"],
            "conversation_id": None,
            "thread_id": "session-durable",
            "status": "completed",
            "attempt_count": 1,
        }

    async def record_failure(self, **payload):
        """记录 graph 失败与目标错误类别。"""
        status = "timed_out" if payload.get("timed_out") else "failed"
        self.calls.append(("failure", status))
        return {"session_id": payload["session_id"], "status": status}


class PersistedScopeLifecycle(RecordingLifecycle):
    """返回 durable session 范围并记录入口是否发送了空 scope。"""

    def __init__(self, scope: dict) -> None:
        super().__init__()
        self.scope = scope
        self.scope_calls: list[dict] = []

    async def resolve_scope_request(self, **payload):
        """模拟从 durable session 回读 project/paper scope。"""
        self.scope_calls.append(dict(payload))
        return dict(self.scope)


@pytest.fixture(autouse=True)
def isolated_operation_boundaries(monkeypatch):
    """参数为补丁工具；仅替换新持久/调度边界，旧路由断言仍调用实际路由与图工厂。"""
    from app.services import result_delivery_service

    async def execute(**payload):
        """参数为受管任务；执行真实路由工作并把异常送往其收尾端口。"""
        try:
            return await payload["runner_factory"]()
        except Exception as exc:
            status = "timed_out" if getattr(exc, "status_code", None) == 504 else "failed"
            await payload["terminal_callback"](status, str(exc))
            raise

    async def settle(**payload):
        """参数为运行终态；记录被测路由所使用生命周期的失败端口。"""
        return await research_api.get_research_session_lifecycle().record_failure(
            session_id="session-durable", error=RuntimeError(payload["error"]),
            timed_out=payload["status"] == "timed_out")

    async def deliver(**payload):
        """参数为原样产物；模拟唯一交付事务，保留既有流式完成顺序断言。"""
        lifecycle = research_api.get_research_session_lifecycle()
        durable = await lifecycle.record_result(session_id=payload["run"].get("session_id", "session-durable"),
                                                 result=payload["result"])
        response = {**payload["response"], **durable}
        await stream_api.rag_run_service.finish_run(status="completed", terminal_payload=response)
        return {"run": {**payload["run"], "status": "completed"}, "response": response,
                "delivery_revision": 1, "persistence_status": "saved"}

    monkeypatch.setattr(research_api.rag_run_service, "get_by_idempotency_key", AsyncMock(return_value=None))
    monkeypatch.setattr(research_api, "get_rag_run_executor_registry", lambda: SimpleNamespace(execute=execute))
    monkeypatch.setattr(research_api, "get_rag_run_repository", lambda: SimpleNamespace(settle_session_operation=settle))
    monkeypatch.setattr(research_api, "persist_research_session_links", AsyncMock())
    monkeypatch.setattr(result_delivery_service, "deliver_result", deliver)
    monkeypatch.setattr(stream_api.rag_run_service, "finish_run", AsyncMock())


async def _patch_external_research(monkeypatch, lifecycle: RecordingLifecycle) -> None:
    """替换 owner/scope/lifecycle 边界，保留真实 route 编排。"""
    monkeypatch.setattr(research_api, "_resolve_identity_or_404", lambda *args: _async_value(None))
    monkeypatch.setattr(
        research_api,
        "build_project_context",
        lambda *args, **kwargs: _async_value(_external_project_context()),
    )
    monkeypatch.setattr(research_api, "get_research_session_lifecycle", lambda: lifecycle)


async def _async_value(value):
    """把测试值包装为可 await 返回值。"""
    return value


@pytest.mark.asyncio
async def test_research_creates_durable_session_before_graph(monkeypatch) -> None:
    """无 session 请求必须先持久化，再用同一 session/thread 调用 graph。"""
    lifecycle = RecordingLifecycle()
    await _patch_external_research(monkeypatch, lifecycle)

    async def fake_invoke(query: str, extra: dict, *, execution_kind: str):
        """断言 graph 只收到已持久化的 durable identity。"""
        lifecycle.calls.append(("graph", str(extra["session_id"])))
        assert extra["thread_id"] == "session-durable"
        return {"report": f"done:{query}"}, "session-durable"

    monkeypatch.setattr(research_api, "invoke_graph", fake_invoke)

    response = await research_api.research(
        ResearchRequest(query="研究问题"),
        user={"user_id": "user-1"},
    )

    assert lifecycle.calls == [
        ("prepare", "new"),
        ("graph", "session-durable"),
        ("result", "session-durable"),
    ]
    assert response["session_id"] == "session-durable"
    assert response["thread_id"] == "session-durable"
    assert response["status"] == "completed"


@pytest.mark.asyncio
async def test_research_maps_invalid_transition_to_409(monkeypatch) -> None:
    """completed session 重新运行时必须返回 409，且不得调用 graph。"""
    lifecycle = RecordingLifecycle()
    await _patch_external_research(monkeypatch, lifecycle)

    async def reject_prepare(**payload):
        """模拟 completed -> running 被状态机拒绝。"""
        raise InvalidResearchSessionTransition("非法研究会话状态迁移: completed -> running")

    lifecycle.prepare_command_session = reject_prepare

    with pytest.raises(HTTPException) as exc_info:
        await research_api.research(
            ResearchRequest(query="再次运行", session_id="session-completed"),
            user={"user_id": "user-1"},
        )

    assert exc_info.value.status_code == 409
    assert "completed -> running" in exc_info.value.detail


@pytest.mark.asyncio
async def test_research_timeout_persists_timed_out_status(monkeypatch) -> None:
    """graph 504 必须先写 timed_out，再保留原 HTTP 语义。"""
    lifecycle = RecordingLifecycle()
    await _patch_external_research(monkeypatch, lifecycle)

    async def fake_invoke(query: str, extra: dict, *, execution_kind: str):
        """模拟 graph facade 超时。"""
        raise HTTPException(status_code=504, detail="research timed out")

    monkeypatch.setattr(research_api, "invoke_graph", fake_invoke)

    with pytest.raises(HTTPException) as exc_info:
        await research_api.research(
            ResearchRequest(query="超时研究"),
            user={"user_id": "user-1"},
        )

    assert exc_info.value.status_code == 504
    assert lifecycle.calls[-1] == ("failure", "timed_out")


@pytest.mark.asyncio
async def test_research_session_identity_mismatch_persists_failed_status(monkeypatch) -> None:
    """graph 返回其他 session_id 时必须先写 failed，不能遗留 running。"""
    lifecycle = RecordingLifecycle()
    await _patch_external_research(monkeypatch, lifecycle)

    async def fake_invoke(query: str, extra: dict, *, execution_kind: str):
        """模拟 graph 破坏 durable session identity。"""
        return {"report": f"done:{query}"}, "other-session"

    monkeypatch.setattr(research_api, "invoke_graph", fake_invoke)

    with pytest.raises(HTTPException) as exc_info:
        await research_api.research(
            ResearchRequest(query="身份漂移研究"),
            user={"user_id": "user-1"},
        )

    assert exc_info.value.status_code == 500
    assert lifecycle.calls[-1] == ("failure", "failed")


@pytest.mark.asyncio
async def test_research_reads_persisted_scope_for_empty_request(monkeypatch) -> None:
    """普通研究入口的空 scope 必须回读 durable project 与多论文范围。"""
    lifecycle = PersistedScopeLifecycle(
        {"project_id": "project-1", "paper_ids": ["paper-1", "paper-2"], "research_mode": "internal"}
    )
    await _patch_external_research(monkeypatch, lifecycle)
    monkeypatch.setattr(
        research_api,
        "build_project_context",
        lambda *args, **kwargs: _async_value(_internal_project_context()),
    )

    async def fake_invoke(query: str, extra: dict, *, execution_kind: str):
        """断言 graph 使用服务端回读的完整范围。"""
        assert extra["core_intent"] == INTENT_INTERNAL_RESEARCH
        assert extra["project_scope"]["paper_ids"] == ["paper-1", "paper-2"]
        assert extra["structured_query"]["paper_ids"] == ["paper-1", "paper-2"]
        return {"report": "范围恢复完成"}, "session-durable"

    monkeypatch.setattr(research_api, "invoke_graph", fake_invoke)
    response = await research_api.research(
        ResearchRequest(query="继续研究", session_id="session-durable"),
        user={"user_id": "user-1"},
    )

    assert lifecycle.scope_calls[0]["project_id"] is None
    assert lifecycle.scope_calls[0]["paper_ids"] == []
    assert response["status"] == "completed"


@pytest.mark.asyncio
async def test_research_with_persisted_papers_without_project_stays_internal(monkeypatch) -> None:
    """无 project 但 durable papers 存在时仍必须保留 internal research 语义。"""
    lifecycle = PersistedScopeLifecycle(
        {"project_id": "", "paper_ids": ["paper-1"], "research_mode": "internal"}
    )
    await _patch_external_research(monkeypatch, lifecycle)
    paper_only_context = {
        **_internal_project_context(),
        "project_id": "",
        "paper_ids": ["paper-1"],
        "resolved_scope": {"project_id": "", "paper_ids": ["paper-1"]},
    }
    monkeypatch.setattr(
        research_api,
        "build_project_context",
        lambda *args, **kwargs: _async_value(paper_only_context),
    )

    async def fake_invoke(query: str, extra: dict, *, execution_kind: str):
        """断言论文范围足以选择内部研究主链。"""
        assert extra["core_intent"] == INTENT_INTERNAL_RESEARCH
        assert extra["project_scope"]["project_id"] == ""
        assert extra["project_scope"]["paper_ids"] == ["paper-1"]
        return {"report": "论文范围研究完成"}, "session-durable"

    monkeypatch.setattr(research_api, "invoke_graph", fake_invoke)
    response = await research_api.research(
        ResearchRequest(query="继续论文研究", session_id="session-durable"),
        user={"user_id": "user-1"},
    )

    assert response["status"] == "completed"


@pytest.mark.asyncio
async def test_stream_reads_persisted_scope_for_empty_request(monkeypatch) -> None:
    """流式入口也必须在打开 graph 前回读 durable project/paper scope。"""
    lifecycle = PersistedScopeLifecycle(
        {"project_id": "project-1", "paper_ids": ["paper-1", "paper-2"], "research_mode": "internal"}
    )
    monkeypatch.setattr(stream_api, "_resolve_identity_or_404", lambda *args: _async_value(None))
    monkeypatch.setattr(stream_api, "get_research_session_lifecycle", lambda: lifecycle)
    monkeypatch.setattr(
        stream_api,
        "build_project_context",
        lambda *args, **kwargs: _async_value(_internal_project_context()),
    )
    captured: dict = {}

    async def fake_prepare(**payload):
        """记录流式入口传给 durable session 的服务端范围。"""
        captured["project_context"] = payload["project_context"]
        return {"session_id": "session-durable", "thread_id": "session-durable"}

    async def fake_execute(**payload):
        """记录 durable executor 收到的 internal intent。"""
        captured["core_intent"] = payload["core_intent"]

    async def fake_create_run(**payload):
        """返回本次新建的 durable run。"""
        return {"run_id": "run-1", "status": "queued"}, True

    class FakeRegistry:
        """记录 route 登记的 runner，测试随后显式执行。"""

        def start(self, **payload):
            captured["runner_factory"] = payload["runner_factory"]
            return True

    monkeypatch.setattr(stream_api.rag_run_service, "get_by_idempotency_key", lambda **kwargs: _async_value(None))
    monkeypatch.setattr(stream_api.rag_run_service, "create_or_get_stream_run", fake_create_run)
    monkeypatch.setattr(stream_api, "get_rag_run_executor_registry", lambda: FakeRegistry())
    monkeypatch.setattr(stream_api, "build_run_event_response", lambda **kwargs: "stream-response")
    monkeypatch.setattr(stream_api, "_prepare_stream_session", fake_prepare)
    monkeypatch.setattr(stream_api, "_execute_durable_stream", fake_execute)
    response = await stream_api.stream(
        ResearchStreamRequest(query="流式继续研究", session_id="session-durable"),
        idempotency_key="idempotency-1",
        user={"user_id": "user-1"},
    )
    await captured["runner_factory"]()

    assert response == "stream-response"
    assert lifecycle.scope_calls[0]["project_id"] is None
    assert lifecycle.scope_calls[0]["paper_ids"] == []
    assert captured["project_context"]["paper_ids"] == ["paper-1", "paper-2"]
    assert captured["core_intent"] == INTENT_INTERNAL_RESEARCH


@pytest.mark.asyncio
async def test_stream_done_payload_persists_completed_status(monkeypatch) -> None:
    """流式 done 发给客户端前必须先写入 completed 状态。"""
    lifecycle = RecordingLifecycle()
    monkeypatch.setattr(research_api, "get_research_session_lifecycle", lambda: lifecycle)

    async def fake_payloads(*args, **kwargs):
        """产生一个最终 done payload。"""
        yield {"type": "done", "session_id": "session-durable", "report": "done"}

    async def fake_open_graph_stream(query: str, extra: dict, *, execution_kind: str):
        """返回与 durable session 一致的惰性 graph stream。"""
        return "values", object(), "session-durable"

    updates: list[dict] = []
    monkeypatch.setattr(stream_api, "open_graph_stream", fake_open_graph_stream)
    monkeypatch.setattr(stream_api, "iter_research_payloads", fake_payloads)
    monkeypatch.setattr(stream_api.rag_run_service, "update_status", lambda **payload: _record_async(updates, payload))
    monkeypatch.setattr(stream_api.rag_run_service, "finish_run", lambda **payload: _record_async(updates, payload))
    await stream_api._execute_durable_stream(
        req=ResearchStreamRequest(query="流式完成"),
        conversation_id=None,
        project_context=_external_project_context(),
        core_intent="external_research",
        session={"session_id": "session-durable", "thread_id": "session-durable"},
        lifecycle=lifecycle,
        user_id="user-1",
        run={"run_id": "run-1"},
    )

    assert lifecycle.calls == [("result", "session-durable")]
    assert updates[-1]["status"] == "completed"
    assert updates[-1]["terminal_payload"]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_without_done_payload_persists_failed_status(monkeypatch) -> None:
    """底层流正常结束却没有 done 时必须显式失败，不能误写 completed。"""
    lifecycle = RecordingLifecycle()

    async def fake_payloads(*args, **kwargs):
        """模拟只产生中间事件便提前结束的异常流。"""
        yield {"type": "event", "stage": "research"}

    async def fake_open_graph_stream(query: str, extra: dict, *, execution_kind: str):
        """返回与 durable session 一致但缺 done 的 graph stream。"""
        return "values", object(), "session-durable"

    events: list[dict] = []
    monkeypatch.setattr(stream_api, "open_graph_stream", fake_open_graph_stream)
    monkeypatch.setattr(stream_api, "iter_research_payloads", fake_payloads)
    monkeypatch.setattr(stream_api.rag_run_service, "update_status", lambda **payload: _async_value(payload))
    monkeypatch.setattr(stream_api.rag_run_service, "append_event", lambda **payload: _record_async(events, payload))
    with pytest.raises(RuntimeError, match="without a done payload"):
        await stream_api._execute_durable_stream(
            req=ResearchStreamRequest(query="流式提前结束"),
            conversation_id=None,
            project_context=_external_project_context(),
            core_intent="external_research",
            session={"session_id": "session-durable", "thread_id": "session-durable"},
            lifecycle=lifecycle,
            user_id="user-1",
            run={"run_id": "run-1"},
        )

    assert events[0]["payload"]["stage"] == "research"


async def _record_async(records: list[dict], payload: dict) -> dict:
    """记录异步 service 调用并原样返回载荷。"""
    records.append(payload)
    return payload


@pytest.mark.asyncio
async def test_stream_session_identity_mismatch_persists_failed_status(monkeypatch) -> None:
    """流打开后返回其他 session_id 时必须写 failed，不能遗留 running。"""
    lifecycle = RecordingLifecycle()

    async def fake_open_graph_stream(query: str, extra: dict, *, execution_kind: str):
        """模拟流入口破坏 durable session identity。"""
        return "values", object(), "other-session"

    monkeypatch.setattr(stream_api, "open_graph_stream", fake_open_graph_stream)
    monkeypatch.setattr(stream_api.rag_run_service, "update_status", lambda **payload: _async_value(payload))

    with pytest.raises(RuntimeError, match="session_id 与 durable session 不一致"):
        await stream_api._execute_durable_stream(
            req=ResearchStreamRequest(query="流式身份漂移"),
            conversation_id=None,
            project_context=_external_project_context(),
            core_intent="external_research",
            session={"session_id": "session-durable", "thread_id": "session-durable"},
            lifecycle=lifecycle,
            user_id="user-1",
            run={"run_id": "run-1"},
        )


class CommandLifecycle(RecordingLifecycle):
    """模拟命令路由需要的 owner-scoped durable session。"""

    def __init__(self, status: str) -> None:
        super().__init__()
        self.session = {
            "session_id": "session-command",
            "user_id": "user-1",
            "conversation_id": "conv-1",
            "project_id": "project-1",
            "thread_id": "session-command",
            "research_mode": "internal",
            "paper_ids": ["paper-1", "paper-2"],
            "status": status,
            "attempt_count": 1,
        }

    async def get_session(self, **payload):
        """返回命令目标 session。"""
        return dict(self.session)

    async def resume(self, **payload):
        """记录 resume 并保持 attempt。"""
        self.calls.append(("resume", str(payload["session_id"])))
        self.session.update(status="running", attempt_count=1)
        return dict(self.session)

    async def retry(self, **payload):
        """记录 retry 并增加 attempt。"""
        self.calls.append(("retry", str(payload["session_id"])))
        self.session.update(status="running", attempt_count=2)
        return dict(self.session)

    async def cancel(self, **payload):
        """记录 cancel 并返回 durable cancelled。"""
        self.calls.append(("cancel", str(payload["session_id"])))
        self.session.update(status="cancelled")
        return {**self.session, "allowed_actions": [ACTION_RETRY]}

    async def record_result(self, **payload):
        """记录命令 graph 结果并返回 completed。"""
        self.calls.append(("result", str(payload["session_id"])))
        self.session.update(status="completed")
        return {**self.session, "allowed_actions": []}


def _internal_project_context() -> dict:
    """返回包含两篇论文的内部研究上下文。"""
    return {
        "user_id": "user-1",
        "project_id": "project-1",
        "library_ids": ["default"],
        "paper_ids": ["paper-1", "paper-2"],
        "filter_expr": '(paper_id == "paper-1" or paper_id == "paper-2")',
        "resolved_scope": {
            "project_id": "project-1",
            "library_ids": ["default"],
            "paper_ids": ["paper-1", "paper-2"],
        },
        "local_retrieval_allowed": True,
    }


async def _patch_command_context(monkeypatch, lifecycle: CommandLifecycle) -> None:
    """替换命令路由的 session、scope 与 checkpoint 读取边界。"""
    monkeypatch.setattr(research_api, "get_research_session_lifecycle", lambda: lifecycle)
    monkeypatch.setattr(
        research_api,
        "build_project_context",
        lambda *args, **kwargs: _async_value(_internal_project_context()),
    )
    monkeypatch.setattr(
        research_api,
        "get_shared_agent_state",
        lambda *args, **kwargs: _async_value({
            "values": {"user_query": "原始研究问题"},
            "interrupts": [{"value": "请补充范围"}] if lifecycle.session["status"] == "awaiting_clarification" else [],
        }),
    )


@pytest.mark.asyncio
async def test_resume_command_reuses_session_thread_and_scope(monkeypatch) -> None:
    """awaiting->resume 必须用同一 session/thread 和已持久化的两篇论文。"""
    lifecycle = CommandLifecycle("awaiting_clarification")
    await _patch_command_context(monkeypatch, lifecycle)

    async def fake_invoke(query: str, extra: dict, *, execution_kind: str):
        """记录恢复输入并返回同一 durable session。"""
        lifecycle.calls.append(("graph", str(extra["session_id"])))
        assert query == "补充比较实验部分"
        assert extra["thread_id"] == "session-command"
        assert extra["project_scope"]["paper_ids"] == ["paper-1", "paper-2"]
        return {"report": "研究完成"}, "session-command"

    monkeypatch.setattr(research_api, "invoke_graph", fake_invoke)

    response = await research_api.resume_research_session(
        "session-command",
        ResearchResumeRequest(answer="补充比较实验部分"),
        user={"user_id": "user-1"},
    )

    assert lifecycle.calls == [
        ("resume", "session-command"),
        ("graph", "session-command"),
        ("result", "session-command"),
    ]
    assert response["session_id"] == "session-command"
    assert response["status"] == "completed"
    assert response["allowed_actions"] == []


@pytest.mark.asyncio
async def test_retry_command_reuses_checkpoint_query_and_increments_attempt(monkeypatch) -> None:
    """failed->retry 必须从 checkpoint 取原问题，并保持原 scope/thread。"""
    lifecycle = CommandLifecycle("failed")
    await _patch_command_context(monkeypatch, lifecycle)

    async def fake_invoke(query: str, extra: dict, *, execution_kind: str):
        """断言 retry 不要求前端重新提交项目或问题。"""
        lifecycle.calls.append(("graph", str(extra["session_id"])))
        assert query == "原始研究问题"
        assert extra["thread_id"] == "session-command"
        return {"report": "重试完成"}, "session-command"

    monkeypatch.setattr(research_api, "invoke_graph", fake_invoke)

    response = await research_api.retry_research_session(
        "session-command",
        user={"user_id": "user-1"},
    )

    assert lifecycle.calls == [
        ("retry", "session-command"),
        ("graph", "session-command"),
        ("result", "session-command"),
    ]
    assert response["attempt_count"] == 2
    assert response["status"] == "completed"


@pytest.mark.asyncio
async def test_cancel_command_delegates_to_lifecycle_stop(monkeypatch) -> None:
    """参数为补丁工具；正式 cancel 将真实停止职责委托生命周期，不在路由猜测执行事实。"""
    lifecycle = CommandLifecycle("running")
    monkeypatch.setattr(research_api, "get_research_session_lifecycle", lambda: lifecycle)

    response = await research_api.cancel_research_session(
        "session-command",
        user={"user_id": "user-1"},
    )

    assert lifecycle.calls == [("cancel", "session-command")]
    assert response["status"] == "cancelled"
    assert response["allowed_actions"] == [ACTION_RETRY]


def test_research_session_summary_exposes_allowed_actions() -> None:
    """研究列表必须返回服务端 action，而不是让前端按 report 猜测。"""
    summary = research_api._research_session_summary({
        "session_id": "session-failed",
        "topic": "深度研究：失败样本",
        "status": "failed",
        "allowed_actions": [ACTION_RETRY],
        "research_mode": "external",
        "state_json": json.dumps({"intent": "deep_research"}),
    })

    assert summary is not None
    assert summary["allowed_actions"] == [ACTION_RETRY]


def test_research_session_summary_recognizes_external_research_mode() -> None:
    """research_mode=external 是真落库的研究标识：topic 没有前缀也要算研究任务。"""
    summary = research_api._research_session_summary({
        "session_id": "session-bound-late",
        "topic": "新对话",
        "status": "completed",
        "research_mode": "external",
        "state_json": "",
    })

    assert summary is not None
    assert summary["research_mode"] == "external"
    assert summary["intent"] == "deep_research"


def test_research_session_summary_ignores_state_json_fields() -> None:
    """state_json 生产上无写入方，里面的 intent/research_brief 不能让普通会话混进研究列表。"""
    summary = research_api._research_session_summary({
        "session_id": "session-chat",
        "topic": "随便聊聊",
        "status": "completed",
        "research_mode": None,
        "state_json": json.dumps({
            "intent": "deep_research",
            "research_brief": "旧数据",
            "awaiting_clarification": True,
        }),
    })

    assert summary is None
