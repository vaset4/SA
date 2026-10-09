"""【研究交付】【实质完成贯通】真实图回边与同次保存，外部模型和读取完全隔离。作者：xxx。"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.api.facades import graph_facade
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.nodes import rag_node
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewDraft
from app.services.result_delivery_service import assemble_delivery, deliver_result, read_delivery_response
from result_delivery_test_utils import Connection, repository
from o2_evidence_test_utils import direct_evidence
from result_delivery_test_utils import offline_memory_callback  # noqa: F401
from .test_incremental_research import state
from .test_main_graph_answer_delivery import DeliveryModel, install_workflow

RESULT_TEXT = "Under condition B, error falls from 12 to 8."


class CompletionModel(DeliveryModel):
    """【离线模型替身】首次最终稿遗漏结果，明确选择补分析或仅改写后完成。"""

    def __init__(self, records, *, rewrite_only=False, review_error=False):
        """参数为原文及修正路径；初始化确定的模型脚本与观察，无真实调用。"""
        super().__init__(records, fail_after_reads=False)
        self.rewrite_only, self.review_error = rewrite_only, review_error
        self.review_count = 0
        self.review_inputs = []
        self.final_count = 0
        self.patch_count = 0
        # 1. 【研究交付】【旧提议边界】此前部分结束的提议不能覆盖本次核对发现的可补缺口
        self.steps[-1]["decision"]["allow_partial"] = True

    def with_structured_output(self, schema):
        """参数为真实调用合同；保留原分析链，核对只给缺口而不凭空生成研究主张。"""
        if schema is ResearchAnswerReviewDraft:
            return SimpleNamespace(ainvoke=self.review_candidate)
        structured = super().with_structured_output(schema)
        if schema is not ResearchClaimStrategyDraftBatch:
            return structured

        async def analyze(messages):
            """参数为分析实际输入；仅在补分析或初次已充分的分支生成结果主张。"""
            draft = await structured.ainvoke(messages)
            if self.review_count or self.rewrite_only:
                claim = draft.claims[0].model_copy(update={"text": "The paper uses attention. " + RESULT_TEXT})
                draft = draft.model_copy(update={"claims": [claim]})
            return draft

        return SimpleNamespace(ainvoke=analyze)

    async def review_candidate(self, messages):
        """参数为真正最终正文与材料；第一次指出缺结果，第二次核对真实补充后的正文。"""
        self.review_count += 1
        self.review_inputs.append(messages[0].content)
        if self.review_error:
            raise RuntimeError("offline review failure")
        missing = self.review_count == 1
        if missing:
            actions = [dict(goal="extract the omitted measured result", covered_facet_ids=["first", "second"],
                input=dict(source_type="inference_from_units", dependencies=[
                    dict(unit_id=unit, required_product="materials", required=True) for unit in ("u1", "u2")]))]
            self.steps = ([] if self.rewrite_only else [dict(decision=dict(kind="execute", actions=actions))])
            self.steps.append(dict(decision=dict(kind="finish", reason="corrected the omitted result")))
        claim_id = "u3.c1" if missing or self.rewrite_only else "u4.c1"
        facts = json.JSONDecoder().raw_decode(messages[0].content.split("Review facts:\n", 1)[1])[0]
        block_id = facts["blocks"][0]["block_id"]
        points = [{"facet_id": name, "verdict": "missing" if missing and name == "second" else "answered",
                   "explanation": "缺少条件B下实际效果，请分析已取得后文" if missing and name == "second" else "正文给出了所问结论",
                   "block_ids": [block_id], "claim_ids": [claim_id],
                   "sources": [{"evidence_id": self.records[-1]["evidence_id"], "quote": "The paper uses attention."}]}
                  for name in ("first", "second")]
        return ResearchAnswerReviewDraft(points=points, issues=[],
            recommendation=("revise_answer" if self.rewrite_only else "continue_research") if missing else "deliver",
            revision_scope="local" if missing and self.rewrite_only else None,
            reason="缺少条件B的效果；材料已取得，不需要再次搜索" if missing else "方法与实际效果均已回答")

    async def ainvoke(self, messages):
        """参数为最终组织输入；首稿只有方法，修正稿选择实际存在的补充主张。"""
        if "Patch schema:\n" in messages[0].content:
            self.patch_count += 1
            request = json.JSONDecoder().raw_decode(messages[0].content.split("Revision request:\n", 1)[1])[0]
            blocks = json.JSONDecoder().raw_decode(messages[0].content.split("Base blocks:\n", 1)[1])[0]
            patch = {key: request[key] for key in ("base_attempt_id", "base_digest", "research_snapshot_digest")}
            patch.update(contract_version="research-planner-final-patch-v1", status="answered", unresolved_items=[],
                operations=[dict(operation="replace_blocks", block_ids=[blocks[0]["block_id"]], blocks=[dict(
                    product_kind="claims", text=blocks[0]["text"] + RESULT_TEXT, facet_ids=["first", "second"],
                    section_ids=[], claim_ids=["u3.c1" if self.rewrite_only else "u4.c1"])])])
            return SimpleNamespace(content=json.dumps(patch, ensure_ascii=False))
        self.final_count += 1
        response = await super().ainvoke(messages)
        raw = json.loads(response.content)
        if self.final_count > 1:
            raw["blocks"][0]["text"] += RESULT_TEXT
            raw["blocks"][0]["claim_ids"] = ["u3.c1" if self.rewrite_only else "u4.c1"]
        return SimpleNamespace(content=json.dumps(raw, ensure_ascii=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite_only", [False, True])
async def test_real_main_graph_reviews_repairs_and_only_commits_final_candidate(monkeypatch, rewrite_only):
    """参数为补分析或纯改写；实际编译主图在缺口后重入，最终仅保存通过核对的一份正文。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    observed["records"][-1] = direct_evidence("second", text="The paper uses attention. " + RESULT_TEXT)
    model = CompletionModel(observed["records"], rewrite_only=rewrite_only)
    observed["model"] = model
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    current = {**state(), "session_id": "session", "conversation_id": "conversation", "intent": "rag_qa"}
    config = graph_facade._agent_config("session", "conversation", "completion-test")
    result = await workflow.ainvoke(current, config=config)
    assert result.get("error_code", "") == "", result
    assert model.review_count == 2 and model.final_count == model.patch_count == 1, result["final_delivery_report"]
    assert observed["reads"] == ["u1", "u2"]
    assert len(model.analysis_messages) == (1 if rewrite_only else 2)
    assert len(observed["rag"]) == 1
    assert RESULT_TEXT in result["answer"]
    assert result["final_delivery_report"]["semantic_validation"] == "passed"
    history = result["planner_execution_history"]
    candidate = next(row for row in history if row["reason"] == "candidate_content_review")
    assert candidate["final_delivery_report"]["status"] == "partial"
    assert RESULT_TEXT not in candidate["answer"]
    if not rewrite_only:
        assert any("缺少条件B的效果" in prompt and "Candidate content review" in prompt for prompt in model.prompts)
    if not rewrite_only:
        assert RESULT_TEXT in result["research_unit_results"][-1]["claims"][0]["text"]
    snapshot = await workflow.aget_state(config)
    assert snapshot.values["final_delivery_report"] == result["final_delivery_report"]
    connection = Connection()
    saved = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result,
                                 repository=repository(connection))
    assert saved["persistence_status"] == "saved"
    reopened = read_delivery_response(connection.data["runs"]["target"])
    assert reopened["answer"] == result["answer"]
    assert reopened["final_delivery_report"] == result["final_delivery_report"]
    assert reopened["planner_execution_history"] == result["planner_execution_history"]
    for key in ("answer", "research_unit_results", "evidence_records"):
        changed = deepcopy(result)
        if key == "answer":
            changed[key] += "论文没有其他实验。"
        else:
            changed[key] = []
        with pytest.raises(ValueError, match="正文|输入"):
            assemble_delivery(run=connection.data["runs"]["target"], result=changed)


@pytest.mark.asyncio
async def test_review_failure_keeps_real_read_facts_and_is_saveable(monkeypatch):
    """参数为替换器；核对失败追加真实已读事实后重新绑定整稿，失败答案仍可保存重开。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    model = CompletionModel(observed["records"], review_error=True)
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    result = await workflow.ainvoke({**state(), "session_id": "session", "conversation_id": "conversation", "intent": "rag_qa"},
        config=graph_facade._agent_config("session", "conversation", "review-failure"))
    assert result["final_delivery_report"]["status"] == "failed"
    assert result["final_delivery_report"]["content_review"]["status"] == "failed"
    assert result["source_status"]["local_evidence_execution"] == "attempted"
    connection = Connection()
    saved = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result,
                                 repository=repository(connection))
    assert saved["persistence_status"] == "saved"
    assert saved["message_patch"]["content"] == result["answer"]
    assert model.review_count == model.final_count == 1


class StructureRepairModel(DeliveryModel):
    """【离线模型】【结构修复】首稿含结构错误，补丁仅删除错误块，不额外调用规划。"""

    def __init__(self, records, *, unreadable=False):
        """参数为来源记录和空白块场景；初始化实际生成/补丁计数，无外部调用。"""
        super().__init__(records, fail_after_reads=False)
        self.unreadable = unreadable
        self.generated = 0
        self.revised = 0
        self.original = None

    async def ainvoke(self, messages):
        """参数为本次实际提示；首稿加入错误块，修复时按基稿身份删除，未改正文不输出。"""
        prompt = messages[0].content
        if "Patch schema:\n" not in prompt:
            self.generated += 1
            response = await super().ainvoke(messages)
            raw = json.loads(response.content)
            error_block = ({"product_kind": "structure", "text": " \n ", "facet_ids": [], "section_ids": []}
                           if self.unreadable else {"product_kind": "claims", "text": "此段主张不存在",
                               "facet_ids": ["first"], "section_ids": [], "claim_ids": ["unknown"]})
            raw["blocks"].append(error_block)
            return SimpleNamespace(content=json.dumps(raw))
        self.revised += 1
        request = json.JSONDecoder().raw_decode(prompt.split("Revision request:\n", 1)[1])[0]
        blocks = json.JSONDecoder().raw_decode(prompt.split("Base blocks:\n", 1)[1])[0]
        self.original = blocks[0]
        patch = {key: request[key] for key in ("base_attempt_id", "base_digest", "research_snapshot_digest")}
        patch.update(contract_version="research-planner-final-patch-v1", status="answered", unresolved_items=[],
                     operations=[dict(operation="replace_blocks", block_ids=[blocks[1]["block_id"]], blocks=[])])
        return SimpleNamespace(content=json.dumps(patch))


@pytest.mark.asyncio
@pytest.mark.parametrize("unreadable", [False, True])
async def test_compiled_main_graph_repairs_structure_before_one_review_and_one_save(monkeypatch, unreadable):
    """参数为隔离端口和空白块场景；真实图结构错误直达补丁，错误稿零核对，最终只核对并保存一次。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    model = StructureRepairModel(observed["records"], unreadable=unreadable)
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    result = await workflow.ainvoke({**state(), "session_id": "session", "conversation_id": "conversation", "intent": "rag_qa"},
        config=graph_facade._agent_config("session", "conversation", "structure-repair"))
    assert result["final_delivery_report"]["status"] == "complete", result["final_delivery_report"]
    assert model.generated == model.revised == len(observed["rag"]) == 1
    assert sum("Review facts:\n" in prompt for prompt in model.prompts) == 1
    assert len(model.analysis_messages) == 1 and model.steps == []
    final_block = result["planner_final_blocks"][0]
    assert final_block["text"] == model.original["text"]
    assert final_block["block_id"] == model.original["block_id"]
    prior = next(row for row in result["planner_execution_history"] if row["reason"] == "candidate_content_review")
    assert prior["final_delivery_report"]["semantic_validation"] == "not_assessed"
    assert prior["final_delivery_report"]["content_review"] is None
    connection = Connection()
    saved = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result, repository=repository(connection))
    assert saved["persistence_status"] == "saved"
    assert read_delivery_response(connection.data["runs"]["target"])["answer"] == result["answer"]


class SourceNavigationModel(DeliveryModel):
    """【离线模型】【出处展示】有效研究正文后附结构性来源目录，不能因此触发反复改写。"""

    def __init__(self, records):
        """参数为来源记录；要求独立出处分节并记录最终模型调用数，无外部请求。"""
        super().__init__(records, fail_after_reads=False)
        self.final_calls = 0
        self.steps[0]["answer_sections"] = [{"section_id": "sources", "title": "原文出处", "source": "explicit",
            "required": True, "facet_ids": ["first", "second"], "presentation_kind": "user_section", "user_text_span": "原文出处"}]

    async def ainvoke(self, messages):
        """参数为实际生成输入；返回业务正文及同节结构目录，不提供局部补丁假成功路径。"""
        self.final_calls += 1
        response = await super().ainvoke(messages)
        raw = json.loads(response.content)
        raw["blocks"].append({"product_kind": "structure", "text": "## 原文出处\n\n本回答依据已读取的两段论文原文。",
                              "facet_ids": ["first", "second"], "section_ids": ["sources"]})
        return SimpleNamespace(content=json.dumps(raw, ensure_ascii=False))


@pytest.mark.asyncio
async def test_compiled_graph_reviews_valid_source_navigation_without_revision(monkeypatch):
    """参数为隔离端口；真实图将有效出处目录送入完整核对，一次生成一次保存，目录不获内容信用。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    model = SourceNavigationModel(observed["records"])
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    current = {**state(), "user_query": "first second，请按原文出处分节", "session_id": "session",
               "conversation_id": "conversation", "intent": "rag_qa"}
    result = await workflow.ainvoke(current, config=graph_facade._agent_config("session", "conversation", "source-navigation"))
    assert result["final_delivery_report"]["status"] == "complete", result
    assert result["final_delivery_report"]["semantic_validation"] == "passed"
    assert model.final_calls == len(observed["rag"]) == 1
    assert sum("Review facts:\n" in prompt for prompt in model.prompts) == 1
    assert not result["planner_final_blocks"][-1]["business_delivery"]
    assert result["final_delivery_report"]["section_results"][0]["status"] == "delivered"
    connection = Connection()
    saved = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=result, repository=repository(connection))
    assert saved["persistence_status"] == "saved"
    assert read_delivery_response(connection.data["runs"]["target"])["answer"] == result["answer"]
