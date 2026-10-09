"""最终重试复用保存事实并整体更新同次交付；作者：xxx。"""
from __future__ import annotations

from result_delivery_test_utils import offline_memory_callback  # noqa: F401

from copy import deepcopy
import socket
import json
import pytest

from app.models.research_planner_contracts import ResearchClaim, ResearchPlan
from app.services.internal_research.research_planner_final_retry import retry_planner_final_answer
from result_delivery_test_utils import Connection, OperationPort, InlineExecutor, repository
from research_planner_test_utils import make_draft, make_unit, result_with_exposure, supporting_spans


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """参数为测试替换器；意外外部模型、检索或数据库连接立即失败。"""
    def reject(*_args, **_kwargs):
        """参数不使用；禁止绕过内存仓储和模型替身连接网络。"""
        raise AssertionError("O4 离线重试测试禁止真实网络连接")

    # 1. Windows asyncio 的 socketpair 使用回环连接，外联守卫放在真实客户端和数据库边界
    import httpx
    import requests
    from app.services import database

    monkeypatch.setattr(httpx.AsyncClient, "send", reject)
    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(requests.Session, "request", reject)
    monkeypatch.setattr(database, "get_pool", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


class Provider:
    """返回明确的离线最终载体并记录实际输入。"""

    def __init__(self, blocks=None, *, fail=False):
        """参数为正文块和失败开关；初始化离线模型。"""
        self.blocks = blocks if blocks is not None else [{
            "block_id": "history-1", "text": "保留的历史成果 😀", "content_kind": "history",
            "facet_ids": ["f1"], "section_ids": [], "claim_ids": ["u1.c1"], "evidence_ids": [],
        }]
        self.fail = fail
        self.inputs = []

    async def generate(self, payload):
        """参数为正式生成输入；返回原始合同，失败时保留真实异常。"""
        self.inputs.append(payload)
        if self.fail:
            raise RuntimeError("offline provider failed")
        return {"contract_version": "research-planner-final-v3", "blocks": deepcopy(self.blocks)}


class RunRepository:
    """保持原操作状态的内存运行仓储。"""

    def __init__(self, metadata, *, fail=False):
        """参数为保存事实与失败开关；建立内存运行。"""
        self.fail = fail
        self.run = {"run_id": "run-1", "user_id": "user-1", "conversation_id": "conv-1",
                    "status": "completed", "answer": "旧正文", "metadata": deepcopy(metadata)}

    async def get_run(self, **_kwargs):
        """接收真实服务的身份参数；返回当前运行副本。"""
        return deepcopy(self.run)

    async def update_run(self, **kwargs):
        """参数为真实服务写入字段；保留原状态并保存整份尝试。"""
        if self.fail:
            raise OSError("offline run save failed")
        self.run.update(status=kwargs["status"], answer=kwargs["answer"])
        self.run["metadata"].update(deepcopy(kwargs["metadata"]))
        return deepcopy(self.run)


class ConversationRepository:
    """记录实际消息补丁的内存仓储。"""

    def __init__(self, metadata, *, fail=False):
        """参数为保存元数据及失败开关；建立旧消息。"""
        self.fail = fail
        self.message = {"id": "msg-1", "role": "assistant", "content": "旧正文", **deepcopy(metadata)}

    async def get_assistant_message_by_run_id(self, **_kwargs):
        """接收消息身份参数；返回保存消息副本。"""
        return deepcopy(self.message)

    async def update_assistant_message_by_run_id(self, **kwargs):
        """参数为正文和元数据；返回同次已保存投影或显式保存错误。"""
        if self.fail:
            raise OSError("offline message save failed")
        self.message.update(content=kwargs["content"], **deepcopy(kwargs["metadata"]))
        return deepcopy(self.message)


class PaperRepository:
    """只模拟当前来源归属核验，不获取正文或执行检索。"""

    def __init__(self, *, readable=True):
        """参数为论文读取权限；初始化离线核验记录。"""
        self.readable = readable
        self.calls = []

    async def get_papers_by_ids(self, paper_ids, *, user_id):
        """参数为真实论文与用户；返回可读取身份集合。"""
        self.calls.append((paper_ids, user_id))
        return [{"paper_id": paper_id} for paper_id in paper_ids] if self.readable else []


def metadata(version="research-planner-v2", *, has_input=True):
    """参数为保存版本及成果是否存在；返回未伪造新版支持的历史输入。"""
    draft = make_draft([make_unit("u1", output_role="evidence")])
    draft["contract_version"] = version
    plan = ResearchPlan.model_validate(draft)
    result = {"unit_id": "u1", "status": "evidence_found", "evidence_ids": ["ev-saved"],
              "claims": [{"claim_id": "u1.c1", "text": "保存的主张", "claim_type": "paper_supported",
                          "validation_status": "validated", "supporting_evidence_ids": ["ev-saved"]}]}
    return {"source_status": {"planner": "ok" if has_input else "no_evidence", "planner_final": "failed"},
            "research_plan": plan.model_dump(mode="json"), "research_unit_results": [result] if has_input else [],
            "evidence_records": [{"evidence_id": "ev-saved", "paper_id": "paper-a", "layer": "direct_text",
                                  "usable_as_citation": True, "excerpt": "保存的原文", "chunk_id": "saved-chunk"}] if has_input else [],
            "turn_outcome": {"status": "completed", "evidence_ids": ["stale"]},
            "planner_final_retry_attempts": [{"status": "old", "attempted_at": str(i)} for i in range(5)]}


async def retry(saved, provider, *, run_fail=False, message_fail=False, readable=True, answer_reviewer=None):
    """参数为保存输入、模型和边界失败；返回正式重试结果以及注入仓储。"""
    run, conversation, papers = RunRepository(saved, fail=run_fail), ConversationRepository(saved, fail=message_fail), PaperRepository(readable=readable)
    connection = Connection(run.run, fail_at="target" if run_fail else "assistant" if message_fail else None)
    result = await retry_planner_final_answer(user_id="user-1", conversation_id="conv-1", run_id="run-1",
        final_provider=provider, answer_reviewer=answer_reviewer, run_repository=OperationPort(connection), conversation_repository=conversation,
        paper_repository=papers, delivery_repository=repository(connection), executor=InlineExecutor())
    run.run = connection.data["runs"]["run-1"]
    if result["message_patch"] is not None:
        conversation.message = result["message_patch"]
    return result, run, conversation, papers


@pytest.mark.asyncio
async def test_final_retry_rejects_source_removed_from_project_before_generation(monkeypatch):
    """参数为当前项目范围端口；拥有论文但已移出原项目时，禁止送入最终模型。"""
    from unittest.mock import AsyncMock
    from app.services import project_scope_service

    scope = AsyncMock(return_value={"paper_ids": []})
    monkeypatch.setattr(project_scope_service, "_build_project_context_scope", scope)
    provider = Provider()
    with pytest.raises(LookupError, match="移出当前项目"):
        await retry({**metadata(), "project_id": "project"}, provider)
    assert provider.inputs == []
    assert scope.call_args.kwargs["requested_paper_ids"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["research-planner-v1", "research-planner-v2", "research-planner-v3"])
async def test_historical_retry_preserves_original_inputs_and_is_partial(version):
    """参数为三个历史版本；合法历史成果可显示，但具体支持未知阻止完整成功。"""
    saved = metadata(version)
    original = deepcopy(saved)
    provider = Provider()
    result, run, conversation, papers = await retry(saved, provider)
    assert result["retry_status"] == "partial"
    report = result["message_patch"]["final_delivery_report"]
    assert report["status"] == "partial"
    assert report["entrypoint"] == "historical_retry"
    assert report["delivery_policy_version"] == "research-delivery-v4"
    assert "历史成果" in conversation.message["content"]
    assert run.run["answer"] == conversation.message["content"]
    for key in ("final_delivery_report", "final_citation_bindings", "final_generation_attempt"):
        assert run.run["metadata"][key] == conversation.message[key]
    assert len(run.run["metadata"]["planner_final_retry_attempts"]) == 5
    assert run.run["metadata"]["research_plan"] == original["research_plan"]
    assert run.run["metadata"]["research_unit_results"] == original["research_unit_results"]
    assert saved == original
    assert papers.calls == [(["paper-a"], "user-1")]
    assert provider.inputs[0].historical_input_context["legacy_version"] == version
    assert conversation.message["turn_outcome"]["status"] == "incomplete"
    assert conversation.message["turn_outcome"]["evidence_ids"] == []


@pytest.mark.asyncio
async def test_empty_historical_input_does_not_call_provider():
    """无参数；确无保存成果时为不足，并明确记录最终模型未调用。"""
    provider = Provider()
    result, _, _, papers = await retry(metadata(has_input=False), provider)
    assert result["message_patch"]["final_delivery_report"]["status"] == "insufficient"
    assert provider.inputs == []
    assert papers.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [Provider([], fail=False), Provider(fail=True)])
async def test_attempt_without_business_delivery_replaces_old_body_as_failed(provider):
    """参数为空输出与模型失败；当前失败说明替换旧正文，不拼接不同尝试。"""
    result, run, conversation, _ = await retry(metadata(), provider)
    assert result["retry_status"] == "failed"
    assert conversation.message["content"] != "旧正文"
    assert run.run["answer"] == conversation.message["content"]
    assert conversation.message["final_delivery_report"]["status"] == "failed"
    assert conversation.message["source_status"]["planner"] == "ok"
    assert conversation.message["final_citation_bindings"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["run_fail", "message_fail"])
async def test_save_failure_is_an_explicit_operation_error(boundary):
    """参数为两个保存边界；生成成功不能掩盖持久化异常。"""
    result, run, conversation, _ = await retry(metadata(), Provider(), **{boundary: True})
    assert result["persistence_status"] == "pending"
    assert result["can_retry_save"] is True
    assert "failed" in result["persistence_error"]
    assert run.run["answer"] == conversation.message["content"] == "旧正文"


@pytest.mark.asyncio
async def test_source_access_failure_prevents_generation():
    """无参数；保存来源已无权读取时显式失败，不发最终模型。"""
    provider = Provider()
    with pytest.raises(LookupError, match="无权读取"):
        await retry(metadata(), provider, readable=False)
    assert provider.inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_mode", ["whole_plan", "incremental"])
async def test_v4_retry_uses_saved_claim_exposure_and_persists_exact_citation_coordinates(monkeypatch, execution_mode):
    """参数为执行模式与仓储替换器；两种保存成果均复用真实支持并保留正文引用坐标。"""
    from app.repositories import conversation_repository
    from app.services import conversation_service
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewDraft

    saved = metadata("research-planner-v4")
    saved["research_plan"]["metadata"]["execution_mode"] = execution_mode
    saved["research_plan"]["coverage_review"]["explicit_facets"][0].update(
        required_claim_types=["paper_supported"], required_paper_ids=["paper-a"])
    if execution_mode == "incremental":
        # 1. 【研究交付】【增量重试】累计动作不预先承诺主张类型，交付以实际验证成果为准
        saved["research_plan"]["units"][0]["expected_output"]["required_claim_types"] = []
    plan = ResearchPlan.model_validate(saved["research_plan"])
    records = saved["evidence_records"]
    claim = ResearchClaim(claim_id="u1.c1", text="保存的主张", claim_type="paper_supported",
        validation_status="validated", supporting_evidence_ids=["ev-saved"],
        supporting_spans=supporting_spans(records), addressed_facet_ids=["f1"])
    unit_result = result_with_exposure(plan.units[0], records, [claim])
    saved["research_unit_results"] = [unit_result.model_dump(mode="json")]
    before = deepcopy(saved)
    provider = Provider([{"block_id": "b1", "text": "😀e\u0301 保存的主张\r\n后续解释", "content_kind": "fact",
        "facet_ids": ["f1"], "section_ids": [], "claim_ids": ["u1.c1"], "evidence_ids": ["ev-saved"]}])
    reviewer = SimpleNamespace(review=AsyncMock(return_value=ResearchAnswerReviewDraft.model_validate({
        "points": [{"facet_id": "f1", "verdict": "answered", "explanation": "正文交付保存主张并关联原文",
                    "block_ids": ["b1"], "claim_ids": ["u1.c1"],
                    "sources": [{"evidence_id": "ev-saved", "quote": "保存的原文"}]}],
        "issues": [], "recommendation": "deliver", "reason": "已核对本次新生成正文与原保存成果"})))
    result, run, conversation, _ = await retry(saved, provider, answer_reviewer=reviewer)
    assert result["retry_status"] == "ok"
    assert reviewer.review.await_count == (1 if execution_mode == "incremental" else 0)
    assert result["message_patch"]["turn_outcome"]["status"] == "completed"
    assert provider.inputs[0].historical_input_context == {}
    assert provider.inputs[0].exposure["prompt_version"] != unit_result.metadata["provenance"]["exposure"]["prompt_version"]
    assert saved == before
    assert run.run["metadata"]["research_unit_results"] == before["research_unit_results"]

    async def read_conversation(*_args):
        """参数为真实仓储查询；返回同一会话的数据库行形状。"""
        return {"id": "conv-1", "user_id": "user-1"}

    async def read_messages(*_args):
        """参数为真实消息查询；从 JSON 往返后的持久化载荷返回原始正文。"""
        return [{"id": "msg-1", "role": "assistant", "content": conversation.message["content"],
                 "metadata": json.dumps(run.run["metadata"], ensure_ascii=False)}]

    monkeypatch.setattr(conversation_repository, "fetchrow", read_conversation)
    monkeypatch.setattr(conversation_repository, "fetch", read_messages)
    restored = await conversation_service.get_conversation(user_id="user-1", conversation_id="conv-1",
        repository=conversation_repository.ConversationRepository())
    message = restored["messages"][0]
    assert message["content"] == run.run["answer"]
    assert message["final_delivery_report"] == run.run["metadata"]["final_delivery_report"]
    assert message["final_generation_attempt"]["attempt_id"] == message["final_delivery_report"]["attempt_id"]
    for binding in message["final_citation_bindings"]:
        assert message["content"][binding["start"]:binding["end"]] == f"[{binding['alias']}]"
        assert binding["evidence_id"] == "ev-saved"


@pytest.mark.asyncio
async def test_whole_plan_retry_still_rejects_missing_promised_claims():
    """无参数；完整计划缺少要求的产出承诺时仍拒绝重试，不调用最终模型。"""
    saved = metadata("research-planner-v4")
    saved["research_plan"]["units"][0]["expected_output"]["required_claim_types"] = []
    saved["research_plan"]["coverage_review"]["explicit_facets"][0]["required_claim_types"] = ["paper_supported"]
    provider = Provider()
    with pytest.raises(ValueError, match="claim requirements lack promised outputs"):
        await retry(saved, provider)
    assert provider.inputs == []


@pytest.mark.asyncio
async def test_incremental_retry_rejects_invalid_saved_requirement():
    """无参数；增量模式仍拒绝材料要求与主张类型矛盾的保存合同，不调用最终模型。"""
    saved = metadata("research-planner-v4")
    saved["research_plan"]["metadata"]["execution_mode"] = "incremental"
    saved["research_plan"]["units"][0]["expected_output"]["required_claim_types"] = []
    saved["research_plan"]["coverage_review"]["explicit_facets"][0].update(
        deliverable_kind="materials", required_claim_types=["paper_supported"])
    provider = Provider()
    with pytest.raises(ValueError, match="material facet f1 cannot require claims"):
        await retry(saved, provider)
    assert provider.inputs == []


@pytest.mark.asyncio
async def test_historical_answer_read_keeps_original_report_without_generation(monkeypatch):
    """参数为历史仓储端口；仅重开旧答案时保留原partial和缺失规则版本，不创建模型或新尝试。"""
    from app.repositories import conversation_repository
    from app.services import conversation_service, llm_service

    saved = metadata("research-planner-v3")
    saved["final_delivery_report"] = {"contract_version": "research-planner-final-v3", "attempt_id": "old-attempt",
        "entrypoint": "normal", "status": "partial", "generation_status": "completed",
        "semantic_validation": "not_assessed"}
    saved["final_generation_attempt"] = {"attempt_id": "old-attempt"}
    saved["final_citation_bindings"] = [{"attempt_id": "old-attempt", "start": 6, "end": 10, "alias": "E1"}]
    original = deepcopy(saved)
    body = "  😀旧文 [E1]\n"

    def forbidden_model(*args, **kwargs):
        """参数为意外模型配置请求；历史读取禁止重新生成。"""
        raise AssertionError("historical read must not create a model")

    async def read_conversation(*args):
        """参数为真实仓储查询；返回原会话所属用户。"""
        return {"id": "conv-1", "user_id": "user-1"}

    async def read_messages(*args):
        """参数为真实消息查询；返回没有新版默认值的持久化旧消息。"""
        return [{"id": "msg-1", "role": "assistant", "content": body,
                 "metadata": json.dumps(saved, ensure_ascii=False)}]

    monkeypatch.setattr(llm_service, "get_llm", forbidden_model)
    monkeypatch.setattr(conversation_repository, "fetchrow", read_conversation)
    monkeypatch.setattr(conversation_repository, "fetch", read_messages)
    restored = await conversation_service.get_conversation(user_id="user-1", conversation_id="conv-1",
        repository=conversation_repository.ConversationRepository())
    message = restored["messages"][0]
    assert message["content"] == body
    for key in ("final_delivery_report", "final_generation_attempt", "final_citation_bindings"):
        assert message[key] == original[key]
    assert saved == original
