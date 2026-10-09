"""【研究交付】【独立贯通】真实续读、材料选择、内容回边与同版保存。作者：xxx。"""

from copy import deepcopy
from functools import partial
import json
from types import SimpleNamespace

import pytest

from app.api.facades import graph_facade
from app.models.research_action_contracts import InitialResearchDecisionOutput, ResearchDecisionOutput
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.nodes import rag_node
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewDraft
from app.services.result_delivery_service import deliver_result, read_delivery_response
from o2_evidence_test_utils import bind_publication
from result_delivery_test_utils import Connection, offline_memory_callback, repository  # noqa: F401
from test_planner_section_text_tool import _METHOD, _OTHER, _row
from .test_incremental_research import initial, state
from .test_main_graph_answer_delivery import install_workflow

METHOD_TEXT = "The paper uses attention."
RESULT_TEXT = "Under condition B, error falls from 12 to 8."
FUTURE_TEXT = "Results under condition B are left for future work."


class ContinuedMaterialModel:
    """【离线模型】【材料接续】根据本次真实回执选行动，来源未提供时不生成后文结论。"""

    def __init__(self, observed, *, partial_answer):
        """参数为真实读取观察与交付分支；初始化确定响应的计数与输入记录。"""
        self.observed = observed
        self.partial_answer = partial_answer
        self.decision_count = 0
        self.analysis_messages = []
        self.review_count = 0
        self.final_count = 0
        self.patch_count = 0

    def with_structured_output(self, schema):
        """参数为正式结构化合同；返回仅在内存解析固定响应的模型端口。"""
        return SimpleNamespace(ainvoke=partial(self.structured, schema))

    async def structured(self, schema, messages):
        """参数为正式合同及实际提示；返回严格解析的决策、结论或逐点核对。"""
        prompt = messages[0].content
        if schema is InitialResearchDecisionOutput:
            action = dict(goal="read the method", covered_facet_ids=["first", "second"],
                input=dict(source_type="local_section_text", target_paper_ids=["paper-a"],
                           required_sections=[_METHOD]))
            return schema.model_validate(initial([action]), strict=True)
        if schema is ResearchDecisionOutput:
            return schema.model_validate(self.next_decision(prompt), strict=True)
        if schema is ResearchClaimStrategyDraftBatch:
            return self.analyze_material(prompt)
        assert schema is ResearchAnswerReviewDraft
        return self.review_answer(prompt)

    def next_decision(self, prompt):
        """参数为当前规划提示；仅使用真正返回的位置和材料，返回下一项当前动作。"""
        self.decision_count += 1
        if self.decision_count == 1:
            ids = [item["continuation_id"] for item in self.observed["reports"][0].remaining]
            assert ids and all(value in prompt for value in ids)
            action = dict(goal="read the returned remainder", covered_facet_ids=["first", "second"],
                input=dict(source_type="local_section_text", target_paper_ids=["paper-a"], continuation_ids=ids))
        elif self.decision_count in {2, 4}:
            index = 0 if self.decision_count == 2 else 1
            evidence_id = self.observed["records"][index]["evidence_id"]
            assert evidence_id in prompt
            action = dict(goal="explain the method" if index == 0 else "analyze the omitted result",
                covered_facet_ids=["first", "second"], input=dict(source_type="inference_from_units",
                    dependencies=[dict(unit_id=f"u{index + 1}", required_product="materials", required=True,
                                       selections=[dict(evidence_id=evidence_id)])]))
        else:
            assert self.decision_count in {3, 5}
            return dict(decision=dict(kind="finish", reason="present currently established content", allow_partial=True))
        return dict(decision=dict(kind="execute", actions=[action]))

    def analyze_material(self, prompt):
        """参数为正式分析输入；仅从真正可见范围选引用，返回本次材料中的主张。"""
        index = len(self.analysis_messages)
        self.analysis_messages.append(prompt)
        record = self.observed["records"][index]
        exposure, _ = json.JSONDecoder().raw_decode(prompt.split("Visible ranges: ", 1)[1])
        assert [item["evidence_id"] for item in exposure["evidence"]] == [record["evidence_id"]]
        quote = METHOD_TEXT if index == 0 else record["excerpt"]
        assert quote in prompt
        bounds = exposure["evidence"][0]
        span = {key: bounds[key] for key in ("evidence_id", "field", "start", "end")}
        return ResearchClaimStrategyDraftBatch.model_validate(dict(claims=[
            dict(text=quote, claim_type="paper_supported", addressed_facet_ids=["first", "second"],
                 supporting_evidence_ids=[record["evidence_id"]], supporting_spans=[{**span, "quote": quote}])
        ]), strict=True)

    def review_answer(self, prompt):
        """参数为待交付正文与真实原文；首次指出已取得后文的遗漏，第二次核对结果或明确未来工作。"""
        self.review_count += 1
        first = self.review_count == 1
        index = 0 if first else 1
        record = self.observed["records"][index]
        tail = self.observed["records"][1]["excerpt"]
        assert tail in prompt
        missing = first or self.partial_answer
        source = dict(evidence_id=record["evidence_id"], quote=METHOD_TEXT if first else tail)
        facts = json.JSONDecoder().raw_decode(prompt.split("Review facts:\n", 1)[1])[0]
        block_id = facts["blocks"][0]["block_id"]
        points = [dict(facet_id=facet, verdict="missing" if facet == "second" and missing else "answered",
            explanation="实际效果尚未回答" if facet == "second" and missing else "对应正文有实际来源",
            block_ids=[block_id], claim_ids=[f"u{3 if first else 4}.c1"], sources=[source])
            for facet in ("first", "second")]
        recommendation = "continue_research" if first else "partial" if self.partial_answer else "deliver"
        return ResearchAnswerReviewDraft.model_validate(dict(points=points, issues=[], recommendation=recommendation,
            reason="已取得后文需要分析" if first else "原文明示该结果留待未来" if self.partial_answer else "两项内容已说明",
            partial_basis=dict(kind="source_statement", explanation="原文明示条件B的结果留待未来",
                               sources=[source], read_unit_ids=[]) if not first and self.partial_answer else None))

    async def ainvoke(self, messages):
        """参数为最终组织提示；只选择本次分析已形成的主张和真实证据，返回候选正文。"""
        patch_prompt = "Patch schema:\n" in messages[0].content
        self.final_count += int(not patch_prompt)
        self.patch_count += int(patch_prompt)
        index = 1 if patch_prompt else 0
        record = self.observed["records"][index]
        text = METHOD_TEXT if index == 0 else record["excerpt"]
        block = dict(text=text, product_kind="claims", facet_ids=["first", "second"], section_ids=[],
                     claim_ids=[f"u{3 if index == 0 else 4}.c1"])
        if not patch_prompt:
            return SimpleNamespace(content=json.dumps(dict(contract_version="research-planner-final-draft-v4", blocks=[block])))
        request = json.JSONDecoder().raw_decode(messages[0].content.split("Revision request:\n", 1)[1])[0]
        base = json.JSONDecoder().raw_decode(messages[0].content.split("Base blocks:\n", 1)[1])[0]
        patch = {key: request[key] for key in ("base_attempt_id", "base_digest", "research_snapshot_digest")}
        patch.update(contract_version="research-planner-final-patch-v1", status="answered", unresolved_items=[],
            operations=[dict(operation="replace_blocks", block_ids=[base[0]["block_id"]], blocks=[block])])
        return SimpleNamespace(content=json.dumps(patch, ensure_ascii=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_answer", [False, True])
async def test_continuation_selected_tail_review_and_single_delivery(monkeypatch, partial_answer):
    """参数为替换器及有依据部分完成分支；真实主图续读、补分析并仅保存核对后的同版答案。"""
    # 1. 【研究取证】【真实续读】同节前后块使用正式发布读取，不读取无关章节
    original_resolver = rag_node._planner_read_request
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    observed.update(records=[], reports=[], reads=[])
    front = METHOD_TEXT + " method context" * 100
    tail = METHOD_TEXT + " " + (FUTURE_TEXT if partial_answer else RESULT_TEXT)
    store_calls = bind_publication(monkeypatch, [_row(0, _METHOD, front), _row(1, _METHOD, tail),
                                                _row(2, _OTHER, "Unrelated results must remain unread.")])
    monkeypatch.setattr(rag_node.settings, "rag_planner_section_text_max_chars", len(front))
    monkeypatch.setattr(rag_node.settings, "rag_planner_consumer_text_max_chars", 0)

    async def read_section(unit, context):
        """参数为正式动作及执行上下文；调用真实章节适配器并保留实际返回内容。"""
        result = await rag_node._planner_section_text_records(current, unit, context)
        observed["reads"].append(unit.unit_id)
        observed["records"].extend(result.records)
        observed["reports"].append(result.read_report)
        return result

    current = {**state(), "session_id": "session", "conversation_id": "conversation", "intent": "rag_qa"}
    model = ContinuedMaterialModel(observed, partial_answer=partial_answer)
    monkeypatch.setattr(rag_node, "_planner_read_request", original_resolver)
    monkeypatch.setattr(rag_node, "_planner_adapters", lambda current: {"local_section_text": read_section})
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    config = graph_facade._agent_config("session", "conversation", f"independent-continuation-{partial_answer}")
    result = await workflow.ainvoke(current, config=config)
    assert not result.get("error_code"), result.get("error_summary")
    assert len(store_calls) == 2 and observed["reads"] == ["u1", "u2"]
    assert observed["reports"][0].status == "partial" and observed["reports"][1].status == "complete"
    assert [record["excerpt"] for record in observed["records"]] == [front, tail]
    assert tail not in model.analysis_messages[0] and tail in model.analysis_messages[1]
    assert model.review_count == 2 and model.final_count == model.patch_count == 1
    assert len(observed["rag"]) == 1
    # 2. 【研究核对】【候选回边】首稿的部分完成提议不能跳过可补缺口，后文分析保留精确来源
    candidate = next(row for row in result["planner_execution_history"] if row["reason"] == "candidate_content_review")
    assert tail not in candidate["answer"] and candidate["final_delivery_report"]["status"] == "partial"
    expected_status = "partial" if partial_answer else "complete"
    assert result["final_delivery_report"]["status"] == expected_status
    assert tail in result["answer"]
    assert result["research_unit_results"][-1]["claims"][0]["supporting_evidence_ids"] == [observed["records"][1]["evidence_id"]]
    snapshot = await workflow.aget_state(config)
    assert snapshot.values["final_delivery_report"] == result["final_delivery_report"]
    # 3. 【研究交付】【同版保存】最终稿只提交一次，重开保留核对、候选历史及真实续读来源
    connection = Connection()
    saved = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result,
                                 repository=repository(connection))
    assert saved["persistence_status"] == "saved"
    reopened = read_delivery_response(deepcopy(connection.data["runs"]["target"]))
    assert reopened["answer"] == result["answer"]
    assert reopened["final_delivery_report"] == result["final_delivery_report"]
    assert reopened["planner_execution_history"] == result["planner_execution_history"]
    assert reopened["evidence_records"] == result["evidence_records"]
