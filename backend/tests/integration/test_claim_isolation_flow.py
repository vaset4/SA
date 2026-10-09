"""【研究主张】【错误隔离贯通】真实主图保留正确成果、反馈错误并继续最终核对。作者：xxx。"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.api.facades import graph_facade
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.nodes import rag_node
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewDraft
from app.services.result_delivery_service import deliver_result, read_delivery_response
from result_delivery_test_utils import Connection, repository
from result_delivery_test_utils import offline_memory_callback  # noqa: F401
from .test_incremental_research import state
from .test_main_graph_answer_delivery import DeliveryModel, install_workflow


class IsolationModel(DeliveryModel):
    """【离线模型】【条目隔离】一次分析提供错误前项和合法后项，不隐式重做模型调用。"""

    def __init__(self, records, *, all_rejected=False, repeat=False):
        """参数为材料及测试分支；初始化真实调用计数和当前研究动作。"""
        super().__init__(records, fail_after_reads=False)
        self.all_rejected = all_rejected
        self.claim_calls = self.review_calls = self.final_calls = 0
        self.steps[-1]["decision"]["allow_partial"] = True
        if repeat:
            self.steps[-1:] = [deepcopy(self.steps[2]), deepcopy(self.steps[2])]

    def with_structured_output(self, schema):
        """参数为正式合同；只替换分析/核对的外部响应，其余动作使用真实图流程。"""
        if schema is ResearchClaimStrategyDraftBatch:
            return SimpleNamespace(ainvoke=self.claim_output)
        if schema is ResearchAnswerReviewDraft:
            return SimpleNamespace(ainvoke=self.review_output)
        return super().with_structured_output(schema)

    async def claim_output(self, messages):
        """参数为分析提示；模型错误选择未知ID，另一条继续使用真实原文支持。"""
        self.claim_calls += 1
        batch = await super().with_structured_output(ResearchClaimStrategyDraftBatch).ainvoke(messages)
        good = batch.claims[0]
        bad = good.model_copy(update={"text": "错误来源不应被接纳", "supporting_evidence_ids": ["unknown-id"],
            "supporting_spans": [good.supporting_spans[0].model_copy(update={"evidence_id": "unknown-id"})]})
        return batch.model_copy(update={"claims": [bad] if self.all_rejected else [bad, good]})

    async def review_output(self, messages):
        """参数为完整实际终稿和材料；仅对最终保留的合法c2进行整稿核对。"""
        self.review_calls += 1
        facts = json.JSONDecoder().raw_decode(messages[0].content.split("Review facts:\n", 1)[1])[0]
        block = facts["blocks"][0]
        assert block["claim_ids"] == ["u3.c2"]
        return ResearchAnswerReviewDraft(points=[dict(facet_id=facet["facet_id"], verdict="answered",
            explanation="正文保留由原文支持的合法结论", block_ids=[block["block_id"]], claim_ids=["u3.c2"],
            sources=[dict(evidence_id=record["evidence_id"], quote="The paper uses attention.") for record in self.records])
            for facet in facts["requirements"]], issues=[], recommendation="deliver", reason="合法主张完成所问内容")

    async def ainvoke(self, messages):
        """参数为最终生成输入；只能选择原第2条的主张，不冒充原第1条错误已修正。"""
        self.final_calls += 1
        response = await super().ainvoke(messages)
        raw = json.loads(response.content)
        raw["blocks"][0]["claim_ids"] = ["u3.c2"]
        return SimpleNamespace(content=json.dumps(raw, ensure_ascii=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("all_rejected", [False, True])
async def test_actual_graph_retains_reads_and_exposes_rejections_before_finish(monkeypatch, all_rejected):
    """参数为隔离端口及全拒分支；真实研究回路看到错误，合法主张到最终审核且只保存一次。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    model = IsolationModel(observed["records"], all_rejected=all_rejected)
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    result = await workflow.ainvoke({**state(), "session_id": "session", "conversation_id": "conversation", "intent": "rag_qa"},
        config=graph_facade._agent_config("session", "conversation", "claim-isolation"))
    assert observed["reads"] == ["u1", "u2"] and len(result["evidence_records"]) == 2
    assert [item["unit_id"] for item in result["research_unit_results"]] == ["u1", "u2", "u3"]
    analysis = result["research_unit_results"][-1]
    assert "claims[1]" in analysis["metadata"]["rejected_claims"][0]
    assert "unknown-id" in analysis["metadata"]["rejected_claims"][0]
    assert any('"rejected_claims": ["claims[1]' in prompt and "unknown-id" in prompt for prompt in model.prompts)
    assert model.claim_calls == 1 and len(observed["rag"]) == 1
    if all_rejected:
        assert analysis["status"] == "blocked" and analysis["claims"] == []
        assert model.review_calls == model.final_calls == 0
        assert result["final_delivery_report"]["status"] == "insufficient"
    else:
        assert [claim["claim_id"] for claim in analysis["claims"]] == ["u3.c2"]
        assert model.review_calls == model.final_calls == 1
        assert result["final_delivery_report"]["semantic_validation"] == "passed"
        connection = Connection()
        saved = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result, repository=repository(connection))
        assert saved["persistence_status"] == "saved"
        assert read_delivery_response(connection.data["runs"]["target"])["answer"] == result["answer"]


@pytest.mark.asyncio
async def test_repeated_analysis_does_not_turn_isolation_into_hidden_retries(monkeypatch):
    """参数为隔离端口；同一输入反复提分析由既有无进展规则拒绝，合法主张及已读材料仍保留。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    model = IsolationModel(observed["records"], repeat=True)
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    result = await workflow.ainvoke({**state(), "session_id": "session", "conversation_id": "conversation", "intent": "rag_qa"},
        config=graph_facade._agent_config("session", "conversation", "claim-isolation-repeat"))
    assert model.claim_calls == 1 and observed["reads"] == ["u1", "u2"]
    assert model.final_calls == model.review_calls == 0
    assert result["planner_observation"]["error_code"] == "repeated_action"
    assert result["final_delivery_report"]["status"] == "failed"
    assert result["research_unit_results"][-1]["claims"][0]["claim_id"] == "u3.c2"
