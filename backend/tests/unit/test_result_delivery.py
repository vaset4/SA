"""【研究交付】【原子保存回归】真实服务和仓储离线验证，作者：xxx，2026-10-04 13:00:00。"""

from result_delivery_test_utils import offline_memory_callback  # noqa: F401

from copy import deepcopy
import socket

import httpx
import pytest
import requests

from app.models.result_delivery_contracts import DeliveryConflict
from app.services import database
from app.services.result_delivery_service import assemble_delivery, deliver_result, persist_result
from result_delivery_test_utils import Connection, final_operation, repository


@pytest.fixture(autouse=True)
def external_guard(monkeypatch):
    """参数为测试替换器；同步异步 HTTP、数据库和网络边界全部拒绝并计数。"""
    calls = []

    def reject(*_args, **_kwargs):
        """参数不使用；记录意外外联并使测试失败。"""
        calls.append("external")
        raise AssertionError("O5 offline external boundary rejected")

    for target, name in ((httpx.Client, "send"), (httpx.AsyncClient, "send"),
                         (requests.Session, "request"), (database, "get_pool"), (socket, "create_connection")):
        monkeypatch.setattr(target, name, reject)
    yield
    assert calls == []


def generated():
    """无参数；返回保留前后空白及 Unicode 坐标的同次载体。"""
    return {"answer": "  😀 正文\n\n来源：[E1]\n", "evidence_records": [{"evidence_id": "E1", "chunk_id": "batch-a"}],
            "final_generation_attempt": {"attempt_id": "attempt-a"}, "final_delivery_report": {"attempt_id": "attempt-a"},
            "final_citation_bindings": [{"start": 12, "end": 16}], "final_delivery_applicability": "research"}


@pytest.mark.asyncio
async def test_commit_preserves_one_artifact_across_every_projection():
    """无参数；验证同连接事务把原正文、版本、报告同时写入所有正式投影。"""
    connection = Connection()
    result = generated()
    committed = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result,
                                     session_status="completed", repository=repository(connection))
    target = connection.data["runs"]["target"]
    message = committed["message_patch"]
    session = connection.data["sessions"]["session"]
    assert target["answer"] == message["content"] == session["current_result"]["answer"] == result["answer"]
    assert target["delivery_revision"] == message["delivery_revision"] == session["delivery_revision"] == 1
    assert target["metadata"]["final_generation_attempt"] == message["final_generation_attempt"]
    assert len(connection.data["events"]) == 1
    assert connection.term_connections == [connection]
    assert target["pending_artifact"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["target", "user", "assistant", "conversation", "session", "terminal"])
async def test_commit_failure_rolls_back_all_projections_and_retains_stage(boundary):
    """参数为各实际 SQL 失败点；验证原子回滚以及真实暂存仍可补存。"""
    connection = Connection(fail_at=boundary)
    original = deepcopy(connection.data["runs"]["target"])
    result = await deliver_result(user_id="user", run=original, result=generated(),
                                  session_status="completed", repository=repository(connection))
    target = connection.data["runs"]["target"]
    assert result["persistence_status"] == "pending" and result["can_retry_save"]
    assert target["delivery_revision"] == 0 and target.get("answer") is None
    assert target["pending_artifact"]["artifact"]["answer"] == generated()["answer"]
    assert connection.data["messages"] == {}
    assert "current_result" not in connection.data["sessions"]["session"]
    assert not any(event["payload"].get("type") == "done" for event in connection.data["events"])


@pytest.mark.asyncio
async def test_stage_failure_does_not_promise_recovery():
    """无参数；暂存本身失败时返回原正文，但明确不能关闭后补存。"""
    connection = Connection(fail_at="stage")
    result = await deliver_result(user_id="user", run=connection.data["runs"]["target"],
                                  result=generated(), repository=repository(connection))
    assert result["persistence_status"] == "not_staged" and not result["can_retry_save"]
    assert result["response"]["answer"] == generated()["answer"]
    assert connection.data["runs"]["target"]["pending_artifact"] is None


@pytest.mark.asyncio
async def test_new_save_operation_is_idempotent_without_changing_old_terminal():
    """无参数；新轻量补存复用同attempt，保留原失败事件，不新增用户问题或模型调用。"""
    connection = Connection(fail_at="assistant")
    repo = repository(connection)
    await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=generated(), repository=repo)
    previous_event = deepcopy(connection.data["events"][0])
    connection.fail_at = None
    operation = final_operation(connection)
    saved = await persist_result(user_id="user", run=operation, repository=repo)
    repeated = await persist_result(user_id="user", run=operation, repository=repo)
    assert saved["delivery_revision"] == repeated["delivery_revision"] == 1
    assert saved["run"]["status"] == "failed"
    assert connection.data["events"][0] == previous_event
    assert len(connection.data["events"]) == 2
    assert len(connection.data["messages"]) == 2
    assert "answer" not in connection.data["events"][-1]["payload"]
    assert connection.data["runs"]["save"].get("answer") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, "research-delivery-v4"])
async def test_save_only_preserves_old_or_current_report_without_model_calls(monkeypatch, policy):
    """参数为历史或当前规则版本；同体补存保留正文、引用、报告及attempt，禁止重新生成。"""
    from app.services import llm_service

    def forbidden_model(*args, **kwargs):
        """参数为意外模型配置请求；补存不能产生任何新模型调用。"""
        raise AssertionError("save-only must not create a model")

    monkeypatch.setattr(llm_service, "get_llm", forbidden_model)
    artifact = generated()
    artifact["final_delivery_report"].update(contract_version="research-planner-final-v3",
        entrypoint="normal", status="partial", generation_status="completed", semantic_validation="not_assessed")
    if policy is not None:
        artifact["final_delivery_report"]["delivery_policy_version"] = policy
    original = deepcopy(artifact)
    connection = Connection(fail_at="assistant")
    repo = repository(connection)
    await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=artifact, repository=repo)
    connection.fail_at = None
    saved = await persist_result(user_id="user", run=final_operation(connection), repository=repo)
    message = saved["message_patch"]
    assert message["content"] == original["answer"]
    for key in ("final_delivery_report", "final_citation_bindings", "final_generation_attempt"):
        assert message[key] == original[key]
    assert artifact == original


@pytest.mark.asyncio
async def test_old_session_save_does_not_overwrite_new_attempt():
    """无参数；旧产物补存可更新历史答案，但不能影响新研究会话。"""
    connection = Connection(fail_at="assistant")
    repo = repository(connection)
    await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=generated(),
                         session_status="completed", repository=repo)
    connection.fail_at = None
    connection.data["sessions"]["session"] = {"current_run_id": "new-attempt", "status": "running"}
    await persist_result(user_id="user", run=final_operation(connection), repository=repo)
    assert connection.data["sessions"]["session"] == {"current_run_id": "new-attempt", "status": "running"}


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["cancel", "revision", "pending"])
async def test_invalid_producer_cannot_replace_target(conflict):
    """参数为取消、版本或已有产物冲突；冲突时保持旧目标。"""
    connection = Connection()
    run = deepcopy(connection.data["runs"]["target"])
    if conflict == "cancel":
        connection.data["runs"]["target"]["cancel_requested"] = True
    elif conflict == "revision":
        connection.data["runs"]["target"]["delivery_revision"] = 1
    else:
        connection.data["runs"]["target"]["pending_artifact"] = {"producer_run_id": "other"}
    before = deepcopy(connection.data)
    with pytest.raises(DeliveryConflict):
        await deliver_result(user_id="user", run=run, result=generated(), repository=repository(connection))
    assert connection.data == before


@pytest.mark.asyncio
@pytest.mark.parametrize("save_failure", [None, "assistant"])
async def test_formal_stream_runner_keeps_exact_markdown(monkeypatch, save_failure):
    """参数为图、事务端口及保存故障点；正式流结束后原样提交或保留待保存产物。"""
    from app.api.routes import chat_api
    from app.api.schemas.research import CrossPaperQARequest

    connection = Connection(fail_at=save_failure)
    run = connection.data["runs"]["target"]
    ctx = chat_api._CrossPaperContext("conversation", "session", "project", [], "", "research", run)
    result = generated()
    closed = []

    async def graph_stream():
        """无参数；返回真实图values接口形态。"""
        try:
            yield result
        finally:
            closed.append(True)

    async def open_graph(*_args, **_kwargs):
        """参数为正式图请求；只替换模型入口。"""
        return "values", graph_stream(), "session"

    async def append_event(**_kwargs):
        """参数为中间事件；不连接数据库。"""

    async def save(**kwargs):
        """参数为正式产物；断言图退出后再调用真实保存服务。"""
        assert closed == [True]
        return await deliver_result(**kwargs, repository=repository(connection))

    monkeypatch.setattr(chat_api, "open_graph_stream", open_graph)
    monkeypatch.setattr(chat_api, "begin_stream_trace_context", lambda **kwargs: None)
    monkeypatch.setattr(chat_api.rag_run_service, "append_event", append_event)
    monkeypatch.setattr(chat_api, "deliver_result", save)
    saved = await chat_api._execute_cross_paper_stream_run(
        req=CrossPaperQARequest(query="原问题", session_id="session"), ctx=ctx, user_id="user",
    )
    assert closed == [True]
    target = connection.data["runs"]["target"]
    if save_failure:
        assert saved["persistence_status"] == "pending"
        assert target.get("answer") is None and connection.data["messages"] == {}
        artifact = target["pending_artifact"]["artifact"]
        assert artifact["answer"] == result["answer"]
        assert artifact["metadata"]["final_citation_bindings"] == result["final_citation_bindings"]
    else:
        assert saved["message_patch"]["content"] == target["answer"] == result["answer"]
        assert saved["message_patch"]["final_citation_bindings"] == result["final_citation_bindings"]


def test_assembly_rejects_mixed_final_attempts():
    """无参数；报告和生成尝试来自不同版本时在写入前明确拒绝。"""
    result = {**generated(), "final_generation_attempt": {"attempt_id": "other-attempt"}}
    with pytest.raises(ValueError, match="不匹配"):
        assemble_delivery(run={"run_id": "target"}, result=result)


@pytest.mark.asyncio
async def test_post_commit_candidate_runs_only_after_all_projections_are_saved():
    """无参数；候选写回必须看到已提交版本，候选错误独立于答案保存。"""
    connection = Connection()
    observed = []

    async def candidate(committed):
        """参数为已提交结果；检查版本已写成后模拟候选处理失败。"""
        observed.append(committed["delivery_revision"])
        assert connection.data["runs"]["target"]["pending_artifact"] is None
        assert len(connection.data["events"]) == 1
        raise RuntimeError("离线候选失败")

    result = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=generated(),
                                  repository=repository(connection), post_commit=candidate)
    assert observed == [1]
    assert result["persistence_status"] == "saved"
    assert result["post_commit"]["status"] == "failed"
    assert result["post_commit"]["error"] == "离线候选失败"
    assert result["post_commit"]["retryable"] is True
    assert result["post_commit"]["candidate_ids"] == []


@pytest.mark.asyncio
async def test_save_failure_never_starts_candidate_processing():
    """无参数；保存事务失败时不能让候选获得未交付答案身份。"""
    from unittest.mock import AsyncMock

    connection = Connection(fail_at="assistant")
    candidate = AsyncMock()
    result = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=generated(),
                                  repository=repository(connection), post_commit=candidate)
    assert result["persistence_status"] == "pending"
    candidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_candidate_status_updates_all_saved_views_and_rejects_late_revision():
    """无参数；候选事实共享提交版本，消息更新失败整体回滚，旧版本不能改新答案。"""
    connection = Connection()
    repo = repository(connection)
    await repo.stage(user_id="user", pending=assemble_delivery(
        run=connection.data["runs"]["target"], result=generated(), session_status="completed"))
    await repo.commit(user_id="user", operation_run_id="target", target_run_id="target")
    outcome = {"status": "success", "candidate_ids": ["candidate"]}
    await repo.record_memory_writeback(user_id="user", run_id="target", delivery_revision=1, outcome=outcome)
    message = next(row for row in connection.data["messages"].values() if row["role"] == "assistant")
    assert message["metadata"]["memory_writeback"] == outcome
    assert connection.data["sessions"]["session"]["current_result"]["memory_writeback"] == outcome
    assert not await repo.record_memory_writeback(user_id="user", run_id="target", delivery_revision=1,
                                                  outcome={"status": "pending"})
    before = deepcopy(connection.data)
    connection.fail_at = "memory_message"
    with pytest.raises(OSError):
        await repo.record_memory_writeback(user_id="user", run_id="target", delivery_revision=1, outcome={"status": "failed"})
    assert connection.data == before
    with pytest.raises(DeliveryConflict):
        await repo.record_memory_writeback(user_id="user", run_id="target", delivery_revision=0, outcome=outcome)


@pytest.mark.asyncio
async def test_memory_retry_route_uses_saved_run_and_rejects_unsaved_result(monkeypatch):
    """参数为隔离依赖；重试候选仅关联用户已提交产物，未保存答案明确拒绝。"""
    from unittest.mock import AsyncMock
    from fastapi import HTTPException
    from app.api.routes import rag_run_api
    from app.services.internal_research import research_memory_writeback

    run = {"run_id": "target", "user_id": "user", "delivery_revision": 2, "metadata": {"persistence_status": "saved"}}
    get_run = AsyncMock(return_value=run)
    candidate = AsyncMock(return_value={"status": "success", "candidate_ids": ["memory"]})
    monkeypatch.setattr(rag_run_api.rag_run_service, "get_run", get_run)
    monkeypatch.setattr(research_memory_writeback, "after_result_committed", candidate)
    result = await rag_run_api.retry_rag_run_memory("target", user={"user_id": "user"})
    assert result["delivery_revision"] == 2
    assert candidate.call_args.args[0]["run"] is run
    assert candidate.call_args.kwargs == {"processing_source": "http_retry"}
    get_run.assert_awaited_once_with(user_id="user", run_id="target")
    get_run.return_value = {**run, "metadata": {"persistence_status": "pending"}}
    with pytest.raises(HTTPException) as failure:
        await rag_run_api.retry_rag_run_memory("target", user={"user_id": "user"})
    assert failure.value.status_code == 409
    assert candidate.await_count == 1


@pytest.mark.asyncio
async def test_candidate_pending_claim_rejects_second_processor():
    """无参数；同一提交版本的候选处理只能领取一次，不能靠重复点击重复调用模型。"""
    connection = Connection()
    repo = repository(connection)
    await repo.stage(user_id="user", pending=assemble_delivery(run=connection.data["runs"]["target"], result=generated()))
    await repo.commit(user_id="user", operation_run_id="target", target_run_id="target")
    assert await repo.record_memory_writeback(user_id="user", run_id="target", delivery_revision=1, outcome={"status": "pending"})
    with pytest.raises(DeliveryConflict, match="正在处理"):
        await repo.record_memory_writeback(user_id="user", run_id="target", delivery_revision=1, outcome={"status": "pending"})


@pytest.mark.asyncio
async def test_saved_terminal_has_pending_before_candidate_callback():
    """无参数；验证已保存答案先有pending，候选回调不改历史terminal。"""
    connection = Connection()

    async def candidates(committed):
        """参数为真实仓储回执；保存事实先于候选完成。"""
        assert committed["response"]["memory_writeback"]["status"] == "pending"
        assert committed["response"]["memory_writeback"]["processing_run_id"] == "target"
        assert connection.data["runs"]["target"]["answer"] == generated()["answer"]
        return {"status": "empty", "candidate_ids": []}

    result = await deliver_result(user_id="user", run=connection.data["runs"]["target"],
        result=generated(), session_status="completed", repository=repository(connection), post_commit=candidates)
    assert result["response"]["memory_writeback"]["status"] == "empty"
    assert connection.data["events"][0]["payload"]["memory_writeback"]["status"] == "pending"


@pytest.mark.asyncio
async def test_closing_delivered_terminal_does_not_cancel_saved_execution(monkeypatch):
    """参数为替换器；正式SSE生成器在客户端收到done后关闭，不应取消候选处理。"""
    from app.api.facades import rag_run_stream_facade as facade

    async def events(**kwargs):
        """参数为查询身份；返回已持久保存的唯一终态。"""
        return [{"sequence": 1, "event_type": "terminal", "payload": {"type": "done"}}]

    def forbidden():
        """无参数；终态正常关闭不得进入取消执行器。"""
        raise AssertionError("unexpected cancel")

    monkeypatch.setattr(facade.rag_run_service, "list_events", events)
    monkeypatch.setattr(facade, "get_rag_run_executor_registry", forbidden)
    stream = facade.stream_run_event_frames(user_id="user", run_id="run", after_sequence=0)
    assert 'done' in await anext(stream)
    await stream.aclose()


@pytest.mark.asyncio
async def test_current_run_projection_uses_commit_operation_execution(monkeypatch):
    """参数为隔离端口；补存回读使用本次保存操作停止事实，不能用旧答案执行已停止冒充。"""
    from unittest.mock import AsyncMock
    from app.api.routes import rag_run_api

    target = {"run_id": "target", "user_id": "user", "status": "completed", "answer": "原正文",
        "delivery_revision": 2, "execution_stopped_at": "old-stop", "metadata": {
            "persistence_status": "saved", "producer_run_id": "target", "operation_run_id": "save-operation",
            "memory_writeback": {"status": "pending"}}}
    operation = {"run_id": "save-operation", "execution_started_at": "new-start", "execution_stopped_at": None}
    reader = AsyncMock(side_effect=[target, operation])
    monkeypatch.setattr(rag_run_api.rag_run_service, "get_run", reader)
    response = await rag_run_api.get_rag_run("target", user={"user_id": "user"})
    current = response["current_result"]
    assert current["answer"] == "原正文" and current["delivery_revision"] == 2
    assert current["execution_stopped_at"] is None and current["execution_started_at"] == "new-start"
    assert current["memory_writeback"]["status"] == "pending"
    reader.assert_any_await(user_id="user", run_id="save-operation")


@pytest.mark.asyncio
async def test_stopped_operation_refreshes_candidate_snapshot(monkeypatch):
    """参数为隔离端口；避免查询跨候选完成边界时把旧pending拼到新停止事实上。"""
    from unittest.mock import AsyncMock
    from app.services import result_delivery_service, rag_run_service

    target = {"run_id": "target", "user_id": "user", "status": "completed", "answer": "正文", "delivery_revision": 2,
        "metadata": {"persistence_status": "saved", "operation_run_id": "operation", "memory_writeback": {"status": "pending"}}}
    latest = {**target, "metadata": {**target["metadata"], "memory_writeback": {"status": "empty"}}}
    monkeypatch.setattr(rag_run_service, "get_run", AsyncMock(side_effect=[
        {"execution_stopped_at": "stopped"}, latest]))
    result = await result_delivery_service.read_current_delivery_response(target)
    assert result["memory_writeback"]["status"] == "empty"
    assert result["execution_stopped_at"] == "stopped"
