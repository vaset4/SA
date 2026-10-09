"""【跨论文问答】【入口回归】范围、同次保存与受管错误，作者：xxx，2026-10-04 13:00:00。"""
from __future__ import annotations

import socket
from result_delivery_test_utils import offline_memory_callback  # noqa: F401

from unittest.mock import AsyncMock

import httpx
import pytest
import requests
from fastapi import HTTPException

from app.api.routes import chat_api
from app.api.schemas.research import CrossPaperQARequest
from app.api.schemas.chat import ChatRequest
from app.services import database
from app.services.result_delivery_service import deliver_result
from result_delivery_test_utils import Connection, InlineExecutor, repository


@pytest.fixture
def route_ports(monkeypatch):
    """参数为测试替换器；隔离真实外联，保留正式路由、保存服务和事务仓储。"""
    connection = Connection({"user_id": "tester", "run_id": "ragrun-1"})
    received, external = [], []

    def reject(*_args, **_kwargs):
        """参数不使用；记录并拒绝真实 HTTP、数据库和网络访问。"""
        external.append(True)
        raise AssertionError("O5 chat test forbids external services")

    async def create_operation(**kwargs):
        """参数为正式操作合同；记录幂等身份和范围，返回离线操作。"""
        received.append(kwargs)
        connection.data["runs"]["ragrun-1"].update(query=kwargs["query"], session_id=kwargs["session_id"])
        return connection.data["runs"]["ragrun-1"], True

    async def save(**kwargs):
        """参数为同次交付；保留真实服务与同事务仓储执行。"""
        return await deliver_result(**kwargs, repository=repository(connection))

    async def scope(project_id, **_kwargs):
        """参数为项目范围请求；返回当前项目两篇论文的可信范围。"""
        return {"project_id": project_id, "resolved_scope": {"paper_ids": ["paper-a", "paper-b"]},
                "paper_ids": ["paper-a", "paper-b"]}

    for target, name in ((httpx.Client, "send"), (httpx.AsyncClient, "send"),
                         (requests.Session, "request"), (database, "get_pool"), (socket, "create_connection")):
        monkeypatch.setattr(target, name, reject)
    monkeypatch.setattr(chat_api, "_resolve_identity_or_404", AsyncMock(return_value="conversation"))
    monkeypatch.setattr(chat_api, "_previous_rag_context", AsyncMock(return_value={}))
    monkeypatch.setattr(chat_api, "build_project_context", scope)
    monkeypatch.setattr(chat_api.rag_run_service, "create_operation", create_operation)
    monkeypatch.setattr(chat_api, "get_rag_run_executor_registry", lambda: InlineExecutor())
    monkeypatch.setattr(chat_api, "deliver_result", save)
    yield connection, received
    assert external == []


@pytest.mark.asyncio
async def test_cross_paper_qa_uses_project_scope_when_request_has_no_paper_ids(monkeypatch, route_ports):
    """参数为隔离端口；未指定论文时使用项目范围并完成真实保存。"""
    graph = AsyncMock(return_value=({"report": "done", "citations": []}, "session"))
    monkeypatch.setattr(chat_api, "invoke_graph", graph)
    response = await chat_api.cross_paper_qa(
        CrossPaperQARequest(query="比较当前项目里的论文", project_id="project"), user={"user_id": "tester"},
    )
    assert graph.call_args.args[1]["structured_query"]["paper_ids"] == ["paper-a", "paper-b"]
    assert response["filter_expr"] == 'paper_id == "paper-a" or paper_id == "paper-b"'
    assert response["persistence_status"] == "saved"


@pytest.mark.asyncio
async def test_cross_paper_qa_passes_history_and_previous_rag_context(monkeypatch, route_ports):
    """参数为隔离端口；用户历史与上一轮已保存材料原样进入图。"""
    previous = {"query": "融合三篇论文", "answer": "上一轮融合架构", "evidence_records": []}
    monkeypatch.setattr(chat_api, "_previous_rag_context", AsyncMock(return_value=previous))
    graph = AsyncMock(return_value=({"report": "done"}, "session"))
    monkeypatch.setattr(chat_api, "invoke_graph", graph)
    await chat_api.cross_paper_qa(CrossPaperQARequest(query="详细展开说明", project_id="project",
        chat_history=[{"role": "user", "content": "详细展开说明"}]), user={"user_id": "tester"})
    extra = graph.call_args.args[1]
    assert extra["previous_rag_context"] == previous
    assert extra["chat_history"][-1]["content"] == "详细展开说明"


@pytest.mark.asyncio
async def test_cross_paper_qa_runs_graph_when_scope_is_empty(monkeypatch, route_ports):
    """参数为隔离端口；空项目仍能讨论研究问题，不伪造论文或提前拒绝。"""
    monkeypatch.setattr(chat_api, "build_project_context", AsyncMock(return_value={
        "project_id": "", "resolved_scope": None, "paper_ids": []}))
    graph = AsyncMock(return_value=({"report": "先把研究问题列出来"}, "session"))
    monkeypatch.setattr(chat_api, "invoke_graph", graph)
    response = await chat_api.cross_paper_qa(CrossPaperQARequest(query="先列研究问题"), user={"user_id": "tester"})
    assert graph.call_args.args[1]["structured_query"]["paper_ids"] == []
    assert response["answer"] == "先把研究问题列出来"


@pytest.mark.asyncio
async def test_cross_paper_qa_maps_rejected_explicit_paper_to_422(monkeypatch, route_ports):
    """参数为隔离端口；错误论文范围在领取操作和执行图之前拒绝。"""
    monkeypatch.setattr(chat_api, "build_project_context", AsyncMock(side_effect=ValueError("paper-private 不属于当前用户")))
    with pytest.raises(HTTPException) as error:
        await chat_api.cross_paper_qa(CrossPaperQARequest(query="比较", paper_ids=["paper-private"]),
                                    user={"user_id": "tester"})
    assert error.value.status_code == 422
    assert route_ports[1] == []


@pytest.mark.asyncio
async def test_cross_paper_qa_logs_guardrails_reject_outcome(monkeypatch, route_ports):
    """参数为隔离端口；拒绝输入的交付事实继续进入原观测字段。"""
    monkeypatch.setattr(chat_api, "invoke_graph", AsyncMock(return_value=({
        "report": "没有明确研究对象", "guardrails_action": "reject", "source_status": {"graph": "skipped"},
        "request_id": "request-guardrails"}, "session")))
    events = []
    monkeypatch.setattr(chat_api, "log_structured_event", lambda *_args, **kwargs: events.append(kwargs))
    response = await chat_api.cross_paper_qa(CrossPaperQARequest(query="19*88="), user={"user_id": "tester"})
    assert events[0]["outcome"] == "guardrails_reject"
    assert response["request_id"] == "request-guardrails"


@pytest.mark.asyncio
async def test_cross_paper_qa_marks_run_completed(monkeypatch, route_ports):
    """参数为隔离端口；完成时正文、run状态和消息状态在同一事务对齐。"""
    graph = AsyncMock(return_value=({"answer": "  原样正文\n", "request_id": "req-1",
        "graph_hint_section": "图谱线索", "graph_hint_full": "完整图谱"}, "session"))
    monkeypatch.setattr(chat_api, "invoke_graph", graph)
    result = await chat_api.cross_paper_qa(CrossPaperQARequest(query="怎么做"), user={"user_id": "tester"},
                                        idempotency_key="same-user-action")
    connection, commands = route_ports
    message = next(item for item in connection.data["messages"].values() if item["role"] == "assistant")
    assert graph.call_args.args[1]["run_id"] == "ragrun-1"
    assert result["answer"] == message["content"] == "  原样正文\n"
    assert result["run_status"] == message["metadata"]["run_status"] == "completed"
    assert message["metadata"]["graph_hint_full"] == "完整图谱"
    assert commands[0]["idempotency_key"] == "same-user-action"
    assert commands[0]["thread_id"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("error,status", [("graph failed: exploded", 503),
    ("milvus failed: Milvus query timed out", 504), ("response assembly failed", 500)])
async def test_cross_paper_qa_maps_errors_after_managed_execution(monkeypatch, route_ports, error, status):
    """参数为实际错误；HTTP 映射保留运行身份，运行收敛由受管执行器负责。"""
    monkeypatch.setattr(chat_api, "invoke_graph", AsyncMock(side_effect=RuntimeError(error)))
    with pytest.raises(HTTPException) as captured:
        await chat_api.cross_paper_qa(CrossPaperQARequest(query="怎么做"), user={"user_id": "tester"})
    assert captured.value.status_code == status
    assert captured.value.detail["run_id"] == "ragrun-1"
    assert captured.value.detail["run_status"] == ("timed_out" if status == 504 else "failed")


@pytest.mark.asyncio
async def test_cross_paper_stream_passes_run_identity_to_graph(monkeypatch, route_ports):
    """参数为隔离端口；正式stream传生产身份并在关闭图流后提交结果。"""
    connection, _ = route_ports
    ctx = chat_api._CrossPaperContext("conversation", "session", "project", [], "", "research",
                                      connection.data["runs"]["ragrun-1"])

    async def stream():
        """无参数；返回离线图values事件。"""
        yield {"answer": "offline answer"}

    opened = AsyncMock(return_value=("values", stream(), "session"))
    monkeypatch.setattr(chat_api, "open_graph_stream", opened)
    monkeypatch.setattr(chat_api, "begin_stream_trace_context", lambda **_kwargs: None)
    monkeypatch.setattr(chat_api.rag_run_service, "append_event", AsyncMock())
    result = await chat_api._execute_cross_paper_stream_run(
        req=CrossPaperQARequest(query="继续研究", session_id="session"), ctx=ctx, user_id="tester")
    assert opened.call_args.args[1]["run_id"] == "ragrun-1"
    assert result["persistence_status"] == "saved"


@pytest.mark.asyncio
async def test_free_chat_uses_managed_delivery_and_keeps_context(monkeypatch, route_ports):
    """参数为隔离端口；普通聊天同样领取运行并保存，正文及旧上下文字段保持一致。"""
    graph = AsyncMock(return_value=({"report": "  自由聊天😀\n", "context_summary": "摘要",
                                    "chat_history": [{"role": "assistant", "content": "历史"}]}, "session"))
    monkeypatch.setattr(chat_api, "invoke_graph", graph)
    response = await chat_api.chat(ChatRequest(query="你好", context_summary="原摘要"),
                                   user={"user_id": "tester"}, idempotency_key="free-chat-action")
    assert response["persistence_status"] == "saved"
    assert response["answer"] == "  自由聊天😀\n"
    assert response["context_summary"] == "摘要"
    assert graph.call_args.args[1]["intent"] == "free_chat"
    assert graph.call_args.args[1]["context_summary"] == "原摘要"
    assert route_ports[1][0]["idempotency_key"] == "free-chat-action"


def test_session_read_uses_complete_committed_projection():
    """无参数；checkpoint 未保存正文不可盖过旧提交，也不能混用新版引用。"""
    checkpoint = {"values": {"answer": "未提交", "final_generation_attempt": {"attempt_id": "new"},
                              "citations": ["new"], "user_query": "问题"}, "interrupts": []}
    committed = {"answer": "  已保存😀\n", "final_generation_attempt": {"attempt_id": "old"}}
    response = chat_api._session_snapshot_with_metadata(checkpoint, {
        "current_run_id": "run", "current_result": committed, "delivery_revision": 2})
    assert response["values"] == committed
    assert response["delivery_revision"] == 2
    empty = chat_api._session_snapshot_with_metadata(checkpoint, {"current_run_id": "run"})
    assert "answer" not in empty["values"]
    assert "final_generation_attempt" not in empty["values"]
    assert checkpoint["values"]["answer"] == "未提交"
