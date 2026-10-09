"""【真实验收】【合同回归】启动、归属及嵌套输出反例，作者：xxx。"""
from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.research_planner_loop_contracts import PlannerProbeDecision
from app.services.llm_service import StructuredOutputLLM
from evals.agent_outcome.runtime_storage import EvaluationStorage
from evals.contracts import read_snapshot, snapshot
from test_agent_outcome_runtime_storage import initial_snapshot


def source_pool(monkeypatch, *, batch_id):
    """参数为连接替换器和当前发布批次；仅模拟SQL返回，生产来源授权/范围/版本规则仍执行。"""
    from app.repositories import memory_source_repository

    connection = MagicMock()
    connection.fetchrow = AsyncMock(side_effect=[
        {"paper_id": "paper", "user_id": "user", "library_id": "library", "index_batch_id": batch_id},
        {"project_id": "project", "user_id": "user", "default_library_id": "library"},
        {"paper_ids": [], "library_ids": ["library"], "excluded_paper_ids": []},
        {"batch_id": batch_id, "manifest_ref": {}},
    ] * 2)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = connection
    monkeypatch.setattr(memory_source_repository, "get_pool", AsyncMock(return_value=pool))
    return connection


def test_database_runtime_initializes_projects_after_base(monkeypatch):
    """参数为依赖替换器；基础库完成后必须初始化项目，再允许会话记忆消费者启动。"""
    from app.runtime import container
    from app.repositories import project_repository

    events = []

    async def ensure_projects():
        """无参数；模拟项目仓储依赖基础用户论文表并记录顺序。"""
        assert events == ["base"]
        events.append("projects")

    monkeypatch.setattr(container.paper_db, "init_db", lambda: events.append("base"))
    monkeypatch.setattr(project_repository, "get_project_repository", lambda: SimpleNamespace(ensure_schema=ensure_projects))
    if hasattr(container, "get_project_repository"):
        monkeypatch.setattr(container, "get_project_repository", project_repository.get_project_repository)
    asyncio.run(container.DatabaseRuntime().initialize())
    assert events == ["base", "projects"]


def test_owned_storage_reads_and_relation_scope(tmp_path):
    """参数为目录；详情和问题读取保留归属，错误历史关联不能泄漏另一用户对象。"""
    state = read_snapshot(initial_snapshot())
    state["questions"] = [dict(question_id="q", user_id="user", project_id="project", review_state="accepted"),
                          dict(question_id="foreign", user_id="another", project_id="project", review_state="accepted")]
    state["question_links"] = [dict(memory_id="accepted", question_id="q"), dict(memory_id="accepted", question_id="foreign")]
    state["links"] = [dict(src_memory_id="accepted", dst_memory_id="other", project_id="project")]
    storage = EvaluationStorage(snapshot(state), output_dir=tmp_path / "storage")
    assert [q["question_id"] for q in asyncio.run(storage.list_questions("project", user_id="user"))] == ["q"]
    assert asyncio.run(storage.get_item("accepted", user_id="another", project_id="project")) is None
    assert asyncio.run(storage.get_detail("accepted", user_id="user", project_id="wrong")) is None
    detail = asyncio.run(storage.get_detail("accepted", user_id="user", project_id="project"))
    assert detail["question_ids"] == ["q"] and detail["links"] == []


def test_repair_contains_complete_native_schema(monkeypatch):
    """参数为配置替身；正式回修保留初次完整嵌套合同、上次正文和字段错误。"""
    from app.services import llm_service
    from test_llm_structured_output_resilience import _StubClient

    monkeypatch.setattr(llm_service, "_structured_output_native_enabled", lambda: False)
    monkeypatch.setattr(llm_service, "_structured_output_repair_max", lambda: 1)
    client = _StubClient(["not json", '{"needs_probe":false,"requests":[]}'])
    StructuredOutputLLM(client, PlannerProbeDecision).invoke("query")
    messages = client.calls[1]["messages"]
    expected = json.dumps(PlannerProbeDecision.model_json_schema(), ensure_ascii=False, indent=2)
    assert expected in messages[0]["content"]
    assert messages[-2] == {"role": "assistant", "content": "not json"}
    assert "json_invalid" in messages[-1]["content"]
    assert "Required JSON keys:" not in messages[0]["content"]


def test_probe_prompt_names_json_and_flat_source_contract():
    """无参数；首次探测消息直接说明请求字段，避免source和filters误解。"""
    from app.services.internal_research.research_planner_pipeline import _probe_prompt

    prompt = _probe_prompt({"user_query": "test"})
    assert "JSON" in prompt and "source_type" in prompt
    assert "flat" in prompt and "filters object" in prompt


@pytest.mark.parametrize("quote,expected", [("actual quote", "success"), ("missing quote", "empty")])
def test_episode_source_validation_and_maintenance_stay_isolated(tmp_path, monkeypatch, quote, expected):
    """参数为介质和原句场景；运行生产精确核验、装配和读取服务，仅替换来源IO，维护落在episode。"""
    from app.repositories import memory_source_repository
    from app.services import memory_evidence_service, research_memory_injector_service, research_memory_stale_service
    from evals.suites.runtime import _bind_storage

    state = read_snapshot(initial_snapshot())
    state["items"] = [{**state["items"][0], "evidence_mode": "evidence_backed"}]
    state["evidence_refs"] = [dict(ref_id="ref", memory_id="accepted", paper_id="paper", quote_text=quote,
                                    ref_role="primary_support")]
    frozen = snapshot(state)
    storage = EvaluationStorage(frozen, output_dir=tmp_path / "storage")
    source = AsyncMock(return_value={"current_batch_id": "batch"})
    monkeypatch.setattr(memory_source_repository, "read_source_state", source)
    monkeypatch.setattr(memory_evidence_service, "read_source_state", source)
    connection = source_pool(monkeypatch, batch_id="batch")
    original_io = storage.record_io

    def record_under_source_lock(action, payload):
        """参数为介质动作；成功核验的提交必须发生在来源事务释放共享锁之前。"""
        transaction = connection.transaction.return_value
        if action == "source_checked" and expected == "success":
            assert transaction.__aenter__.await_count > transaction.__aexit__.await_count
        original_io(action, payload)

    monkeypatch.setattr(storage, "record_io", record_under_source_lock)
    manifest = {"batch_id": "batch", "source_ref": {"sha256": "source"}, "document_ref": {"sha256": "doc"},
                "chunks": [{"chunk_id": "batch::chunk", "paper_id": "paper", "full_text": "actual quote"}]}
    monkeypatch.setattr(memory_evidence_service, "_manifest", AsyncMock(return_value=manifest))
    with ExitStack() as stack:
        _bind_storage(stack, storage)
        result = asyncio.run(research_memory_injector_service.load_injectable_memory("project", "user"))
        assert result["status"] == expected
        if expected == "success":
            assert result["items"][0]["version"] == 2
            asyncio.run(research_memory_stale_service.revalidate_memory_snapshot({
                "memory_read_result": result, "project_scope": {"user_id": "user", "project_id": "project"}}))
        foreign = [{**state["items"][0], "user_id": "foreign"}]
        with pytest.raises(LookupError, match="不属于"):
            asyncio.run(research_memory_stale_service.checked_memory_items(foreign, user_id="user", project_id="project"))
    current = asyncio.run(storage.get_item("accepted", user_id="user", project_id="project"))
    assert current["version"] == 2
    assert current["stale_state"] == ("fresh" if expected == "success" else "needs_rebind")
    assert len(asyncio.run(storage.list_versions("accepted"))) == 1
    assert all("FOR SHARE" in call.args[0] for call in connection.fetchrow.call_args_list)
    connection.execute.assert_not_called()
    assert snapshot(state) == frozen
    second = EvaluationStorage(frozen, output_dir=tmp_path / "second")
    assert asyncio.run(second.get_item("accepted"))["version"] == 1
    storage.persist_snapshot()


def test_episode_source_maintenance_rejects_version_scope_and_write_failure(tmp_path, monkeypatch):
    """参数为介质及IO替换；版本竞争、越权和落盘失败不改变内存状态或伪造维护成功。"""
    storage = EvaluationStorage(initial_snapshot(), output_dir=tmp_path / "storage")
    item = asyncio.run(storage.get_item("accepted"))
    asyncio.run(storage.mark_checked_stale(item, ["changed"]))
    with pytest.raises(ValueError, match="已更新"):
        asyncio.run(storage.mark_checked_stale(item, ["stale version"]))
    with pytest.raises(LookupError):
        asyncio.run(storage.mark_checked_stale({**item, "user_id": "foreign"}, ["invalid owner"]))
    current = asyncio.run(storage.get_item("accepted"))
    original = storage.record_io

    def fail_write(action, payload):
        """参数为真实写动作；仅在提交落盘处制造IO错误，读取仍可执行。"""
        if action == "source_checked":
            raise OSError("disk failure")
        return original(action, payload)

    monkeypatch.setattr(storage, "record_io", fail_write)
    with pytest.raises(OSError, match="disk failure"):
        asyncio.run(storage.mark_checked_stale(current, ["second reason"]))
    assert asyncio.run(storage.get_item("accepted")) == current


@pytest.mark.parametrize("probe_request", [
    {"source": "local_section_text", "filters": {"required_sections": ["section"]}},
    {"source_type": "invented_source"},
    {"source_type": "local_metadata", "target_paper_ids": None},
])
def test_nested_bad_repair_still_fails_after_one_attempt(monkeypatch, probe_request):
    """参数为不合约请求；字段误名、非法枚举和数组null经过一次回修仍明确失败。"""
    from app.services import llm_service
    from test_llm_structured_output_resilience import _StubClient

    monkeypatch.setattr(llm_service, "_structured_output_native_enabled", lambda: True)
    monkeypatch.setattr(llm_service, "_structured_output_repair_max", lambda: 1)
    client = _StubClient(["not json", json.dumps({"needs_probe": True, "requests": [probe_request]})])
    with pytest.raises(llm_service.StructuredOutputError) as failure:
        llm_service.StructuredOutputLLM(client, PlannerProbeDecision).invoke("query")
    assert len(client.calls) == 2 and failure.value.details["repair_count"] == 1
    assert client.calls[0]["response_format"] == client.calls[1]["response_format"]


def test_optional_object_keeps_nonnullable_child_contract(monkeypatch):
    """无参数；clarification整体允许空值，子字段仍保持字符串且完整出现在回修合同。"""
    from app.models.research_planner_contracts import TurnDepthProposal
    from pydantic import ValidationError

    contract = TurnDepthProposal.model_json_schema()
    assert {"type": "null"} in contract["properties"]["clarification"]["anyOf"]
    base = {"depth": "minimal", "reason_code": "direct", "requires_paper_evidence": False, "resolved_query": "question"}
    assert TurnDepthProposal.model_validate({**base, "clarification": None}).clarification is None
    with pytest.raises(ValidationError) as error:
        TurnDepthProposal.model_validate({**base, "clarification": {"needed": False, "question": None, "reason_code": None}})
    assert {tuple(issue["loc"]) for issue in error.value.errors()} == {
        ("clarification", "question"), ("clarification", "reason_code")}
    from app.services import llm_service
    from test_llm_structured_output_resilience import _StubClient

    monkeypatch.setattr(llm_service, "_structured_output_native_enabled", lambda: False)
    monkeypatch.setattr(llm_service, "_structured_output_repair_max", lambda: 1)
    invalid = {**base, "clarification": {"needed": False, "question": None, "reason_code": None}}
    client = _StubClient([json.dumps(invalid), json.dumps({**base, "clarification": None})])
    StructuredOutputLLM(client, TurnDepthProposal).invoke("q")
    messages = client.calls[1]["messages"]
    assert json.dumps(contract, ensure_ascii=False, indent=2) in messages[0]["content"]
    assert '"clarification", "question"' in messages[-1]["content"]



def test_source_changed_after_verification_cannot_commit(tmp_path, monkeypatch):
    """参数为介质及来源读取替换；核验后批次变化时拒绝写入，原版本与引用保持不变。"""
    state = read_snapshot(initial_snapshot())
    ref = {"ref_id": "r", "memory_id": "accepted", "paper_id": "paper", "verification": {"current_batch_id": "old"},
           "source_ref": {}, "document_ref": {}}
    state["evidence_refs"] = [ref]
    storage = EvaluationStorage(snapshot(state), output_dir=tmp_path / "storage")
    source_pool(monkeypatch, batch_id="new")
    item = asyncio.run(storage.get_item("accepted"))
    with pytest.raises(ValueError, match="来源在核验后已更新"):
        asyncio.run(storage.record_source_checks(item, [ref]))
    assert asyncio.run(storage.get_item("accepted")) == item
    assert asyncio.run(storage.list_evidence_refs("accepted")) == [ref]
    assert asyncio.run(storage.list_versions("accepted")) == []
