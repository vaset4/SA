"""【研究交付】【确定性缺口闭环】真实图修复漏引，模型评审通过不覆盖实际交付缺口。作者：xxx。"""

import json
from types import SimpleNamespace

import pytest

from app.api.facades import graph_facade
from app.models.research_action_contracts import InitialResearchDecisionOutput
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.nodes import rag_node
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewDraft
from app.services.result_delivery_service import deliver_result, read_delivery_response
from result_delivery_test_utils import Connection, repository
from result_delivery_test_utils import offline_memory_callback  # noqa: F401
from .test_incremental_research import state
from .test_main_graph_answer_delivery import DeliveryModel, install_workflow


class CitationRepairModel(DeliveryModel):
    """【离线端口】首次漏推理引用、第二次补全；两个语义评审均返回通过以检验程序责任。"""

    def __init__(self, records):
        """参数为真实测试来源；初始化最终尝试与核对观察。"""
        super().__init__(records, fail_after_reads=False)
        self.final_count = 0
        self.review_inputs = []

    def with_structured_output(self, schema):
        """参数为正式模型合同；生成包含必需推理的离线研究产物。"""
        if schema is ResearchAnswerReviewDraft:
            return SimpleNamespace(ainvoke=self.review_candidate)
        base = super().with_structured_output(schema)

        async def invoke(messages):
            """参数为正常结构化输入；保留正式校验，只构造测试要求与两类主张。"""
            result = await base.ainvoke(messages)
            if schema is InitialResearchDecisionOutput:
                raw = result.model_dump(mode="json")
                raw["requested_facets"][1]["required_claim_types"] = ["paper_supported", "evidence_grounded_inference"]
                return schema.model_validate(raw)
            if schema is ResearchClaimStrategyDraftBatch:
                inference = result.claims[0].model_copy(update={"claim_type": "evidence_grounded_inference",
                    "reasoning_summary": "两份材料均描述相同方法，因此支持这一综合解释"})
                return result.model_copy(update={"claims": [*result.claims, inference]})
            return result

        return SimpleNamespace(ainvoke=invoke)

    async def review_candidate(self, messages):
        """参数为最终核对输入；故意始终认可语义，验证程序仍依据真实缺口继续修稿。"""
        self.review_inputs.append(messages[0].content)
        if len(self.review_inputs) == 1:
            self.steps.append(dict(decision=dict(kind="finish", reason="使用现有主张补全推理块所需引用")))
        return ResearchAnswerReviewDraft.model_validate({"points": [
            {"facet_id": facet, "verdict": "answered", "explanation": "已阅读有效事实块",
             "block_ids": ["answer"], "claim_ids": ["u3.c1"], "sources": [
                 {"evidence_id": self.records[0]["evidence_id"], "quote": "The paper uses attention."}]}
            for facet in ("first", "second")], "issues": [], "recommendation": "deliver", "reason": "离线模型声明可交付"})

    async def ainvoke(self, messages):
        """参数为最终生成输入；首次故意漏第二来源，下一次仅修正文引用而不重读或重分析。"""
        self.final_count += 1
        response = await super().ainvoke(messages)
        raw = json.loads(response.content)
        ids = [record["evidence_id"] for record in self.records]
        raw["blocks"].append({"block_id": "b4", "text": "两份材料均说明该方法使用注意力。", "content_kind": "inference",
            "facet_ids": ["first", "second"], "section_ids": [], "claim_ids": ["u3.c2"],
            "evidence_ids": ids[:1] if self.final_count == 1 else ids})
        return SimpleNamespace(content=json.dumps(raw, ensure_ascii=False))


@pytest.mark.asyncio
async def test_passed_review_with_missing_inference_citation_repairs_before_single_save(monkeypatch):
    """参数为依赖替换器；真实主图根据权威缺口回规划修稿，来源和主张复用，最终只保存一次。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    model = CitationRepairModel(observed["records"])
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    result = await workflow.ainvoke({**state(), "session_id": "session", "conversation_id": "conversation", "intent": "rag_qa"},
        config=graph_facade._agent_config("session", "conversation", "deterministic-delivery-gap"))
    assert model.final_count == len(model.review_inputs) == 2
    assert observed["reads"] == ["u1", "u2"] and len(model.analysis_messages) == 1
    assert "Original question: " + result["research_plan"]["question"] in model.analysis_messages[0][0].content
    assert len(observed["rag"]) == 1
    candidate = next(row for row in result["planner_execution_history"] if row["reason"] == "candidate_content_review")
    report = candidate["final_delivery_report"]
    assert report["content_review"]["status"] == "passed" and report["status"] == "partial"
    assert report["validation_errors"][0]["code"] == "missing_support_citations"
    assert next(row for row in report["requirement_results"] if row["requirement_id"] == "second")["status"] == "missing"
    assert '"delivery_findings"' in model.review_inputs[0]
    assert any("missing_support_citations" in prompt and "Candidate delivery findings" in prompt for prompt in model.prompts)
    assert any("missing_support_citations" in prompt and "Previous candidate delivery findings" in prompt for prompt in model.prompts)
    assert result["final_delivery_report"]["status"] == "complete"
    connection = Connection()
    saved = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result,
                                 repository=repository(connection))
    reopened = read_delivery_response(connection.data["runs"]["target"])
    assert saved["persistence_status"] == "saved" and reopened["answer"] == result["answer"]
    assert reopened["final_delivery_report"] == result["final_delivery_report"]
