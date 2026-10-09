"""【研究循环】【离线贯通】明确替换外部模型与读取，检查正常图的增量链路。作者：xxx。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models.research_action_contracts import InitialResearchDecisionOutput, ResearchDecisionOutput, ReadSection
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewDraft
from app.nodes import rag_node
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research.research_planner_conversion import migrate_execution_checkpoint
from o2_evidence_test_utils import direct_evidence, read_result, scope
from research_planner_test_utils import supporting_spans


def facet(name):
    """参数为用户问题标识；返回明确的论文事实要求。"""
    return dict(facet_id=name, label=name, source="explicit", user_text_span=name, required=True,
                deliverable_kind="claims", required_claim_types=["paper_supported"])


def read_action(query):
    """参数为检索目的；返回只索取材料的当前动作。"""
    return dict(goal=query, input=dict(source_type="local_direct_text", target_paper_ids=["paper-a"],
                                      locator_queries=[query]), covered_facet_ids=["first"])


def initial(actions=None):
    """参数为可选动作；返回包含两个要求但只读取一个方面的首次决策。"""
    return dict(requested_facets=[facet("first"), facet("second")],
                decision=dict(kind="execute", actions=actions or [read_action("first")]))


def state():
    """无参数；返回固定离线运行身份与可信用户问题。"""
    return dict(user_query="first second", project_scope=scope(), run_id="incremental-run", turn_depth="research",
                turn_depth_assessment={"depth": "research", "requires_paper_evidence": True})


def test_section_schema_rejects_graph_switch():
    """章节动作拒绝图专用字段，模式定义中也没有此选项。"""
    assert "verify_with_direct_text" not in ReadSection.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ReadSection(source_type="local_section_text", target_paper_ids=["paper-a"],
                    required_sections=["Method"], verify_with_direct_text=True)


def test_current_read_does_not_need_future_claim_coverage():
    """合法材料动作能够先执行，未覆盖的要求仍完整保留。"""
    current = {**state(), "planner_draft": initial()}
    delta = pipeline.validate_planner_state(current, SimpleNamespace(rag_research_planner_max_units=8))
    assert delta["planner_loop_decision"] == "continue"
    assert delta["research_plan"]["units"][0]["expected_output"]["deliverable_kind"] == "materials"
    assert delta["research_plan"]["units"][0]["claim_policy"]["allowed_claim_types"] == []
    assert len(delta["coverage_review"]["explicit_facets"]) == 2


def test_same_error_different_action_remains_correctable():
    """同错误码但输入不同不会被旧整份计划错误签名误判为重复。"""
    current = state()
    for index, paper in enumerate(("foreign-a", "foreign-b", "foreign-b")):
        action = read_action("first")
        action["input"]["target_paper_ids"] = [paper]
        raw = initial([action]) if not current.get("coverage_review") else dict(decision=dict(kind="execute", actions=[action]))
        current.update(planner_draft=raw)
        current.update(pipeline.validate_planner_state(current, SimpleNamespace(rag_research_planner_max_units=8)))
        assert current["planner_loop_decision"] == ("terminate_fail" if index == 2 else "replan")


@pytest.mark.parametrize("failure", ["scope", "too_many_actions", "action_input_unavailable"])
def test_rejected_initial_actions_resume_with_accepted_requirements_and_original_feedback(failure):
    """参数为动作准入失败类型；首轮要求已接纳时，恢复旧草稿并以严格后续合同纠错，无返回值。"""
    # 1. 【研究循环】【纠错恢复】要求合法而动作非法，真实准入保留要求但不建立执行账本
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    actions = [read_action("first")]
    if failure == "scope":
        actions[0]["input"]["target_paper_ids"] = ["foreign-paper"]
    elif failure == "too_many_actions":
        actions = [read_action(f"source-{index}") for index in range(settings.rag_research_planner_max_units + 1)]
    else:
        actions = [dict(goal="analyze", covered_facet_ids=["first"], input=dict(
            source_type="inference_from_units", dependencies=[dict(
                unit_id="u2", required_product="materials", required=True)]))]
    current = {**state(), "planner_draft": initial(actions)}
    current["planner_draft"]["answer_sections"] = []
    current.update(pipeline.validate_planner_state(current, settings))
    assert current["planner_loop_decision"] == "replan"
    assert not current.get("research_plan")
    assert len(current["coverage_review"]["explicit_facets"]) == 2
    original = deepcopy(current)

    # 2. 【研究循环】【纠错恢复】旧草稿仍严格识别为首次合同，不被当前已接纳要求误改身份
    assert migrate_execution_checkpoint(current) == {}
    assert current == original
    model = ScriptedModel([])
    model.steps = [dict(decision=dict(kind="execute", actions=[read_action("first")]))]
    delta = asyncio.run(pipeline.analyze_planner_state(current, model, settings))
    assert not delta.get("error_code"), delta
    assert "requested_facets" not in delta["planner_draft"]
    feedback = json.loads(model.prompts[0].split("Latest action feedback:\n", 1)[1])
    assert feedback == {key: original["planner_observation"][key] for key in feedback}
    assert "pending_work_keys" not in feedback
    assert feedback["error_code"] == original["planner_observation"]["error_code"]
    assert "Do not resubmit requested_facets or answer_sections" in model.prompts[0]
    current.update(delta)

    # 3. 【研究循环】【纠错恢复】新动作准入成功，原要求与失败历史保持，首个身份未跳号
    current.update(pipeline.validate_planner_state(current, settings))
    assert current["planner_loop_decision"] == "continue"
    assert current["research_plan"]["units"][0]["unit_id"] == "u1"
    assert current["coverage_review"]["explicit_facets"] == original["coverage_review"]["explicit_facets"]
    assert current["planner_budget"]["history"][0] == original["planner_budget"]["history"][0]
    assert not current.get("planner_execution_history")
    assert not model.steps


@pytest.mark.parametrize("draft", [
    {**initial(), "unexpected": True},
    dict(decision=dict(kind="execute", actions=[read_action("first")]), answer_sections=[]),
])
def test_incremental_checkpoint_still_rejects_invalid_draft_fields(draft):
    """参数为含未知字段的首轮或后续草稿；恢复不能删除字段或换合同掩盖格式错误，无返回值。"""
    original = deepcopy(draft)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        migrate_execution_checkpoint({**state(), "planner_draft": draft, "coverage_review": {"explicit_facets": []}})
    assert draft == original


def test_accepted_requirements_still_reject_resubmission_after_initial_action_failure():
    """无参数；旧草稿可恢复不代表后续模型能重交要求，准入仍严格拒绝并保留基准。"""
    action = read_action("first")
    action["input"]["target_paper_ids"] = ["foreign-paper"]
    current = {**state(), "planner_draft": initial([action])}
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    current.update(pipeline.validate_planner_state(current, settings))
    baseline = deepcopy(current["coverage_review"])
    current["planner_draft"] = initial()
    delta = pipeline.validate_planner_state(current, settings)
    assert delta["planner_observation"]["error_code"] == "invalid_action_format"
    assert current["coverage_review"] == baseline
    assert not delta.get("research_plan")


@pytest.mark.parametrize("limit,accepted", [(0, True), (10, True), (8, False), (-1, False)])
def test_action_batch_zero_limit_preserves_all_actions_and_scope_checks(limit, accepted):
    """参数为动作上限及预期准入；0完整接纳十个独立章节动作，正数和权限校验保持，无返回值。"""
    action_count = 10
    actions = [dict(goal=f"read section {index}", covered_facet_ids=["first"], input=dict(
        source_type="local_section_text", target_paper_ids=["paper-a"], required_sections=[f"Section {index}"]))
        for index in range(action_count)]
    current = {**state(), "planner_draft": initial(actions)}
    original = deepcopy(current)
    settings = SimpleNamespace(rag_research_planner_max_units=limit)
    delta = pipeline.validate_planner_state(current, settings)
    assert current == original
    if not accepted:
        assert delta["planner_observation"]["error_code"] == "too_many_actions"
        assert not delta.get("research_plan")
        return
    assert delta["planner_loop_decision"] == "continue"
    assert len(delta["research_plan"]["units"]) == action_count
    assert delta["research_plan"]["metadata"]["pending_unit_ids"] == [f"u{i}" for i in range(1, action_count + 1)]
    current["planner_draft"]["decision"]["actions"][-1]["input"]["target_paper_ids"] = ["foreign-paper"]
    rejected = pipeline.validate_planner_state(current, settings)
    assert rejected["planner_observation"]["error_code"] == "out_of_scope_paper_id"


def test_repeated_read_cannot_change_goal_or_default_to_create_progress():
    """同一读取只改业务文字或显式默认值，不能制造新动作。"""
    current = {**state(), "planner_draft": initial()}
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    current.update(pipeline.validate_planner_state(current, settings))
    async def empty(unit, context):
        return read_result([])
    current.update(asyncio.run(pipeline.execute_planner_state(current, llm=ScriptedModel([]), settings=settings,
                                                             adapters={"local_direct_text": empty})))
    action = read_action("first")
    action["goal"] = "renamed same read"
    action["input"]["evidence_granularity"] = "child_chunk"
    current["planner_draft"] = dict(decision=dict(kind="execute", actions=[action]))
    result = pipeline.validate_planner_state(current, settings)
    assert result["planner_observation"]["error_code"] == "repeated_action"
    assert result["planner_loop_decision"] == "replan"


def test_future_dependency_is_rejected_before_execution():
    """分析不能声明尚未取得的未来动作结果。"""
    action = dict(goal="analyze", input=dict(source_type="inference_from_units",
        dependencies=[dict(unit_id="u2", required_product="materials", required=True)]))
    current = {**state(), "planner_draft": initial([action])}
    result = pipeline.validate_planner_state(current, SimpleNamespace(rag_research_planner_max_units=8))
    assert result["planner_observation"]["error_code"] == "action_input_unavailable"
    assert "research_plan" not in result


def test_finish_progress_and_selected_analysis_input_keys():
    """结束失败后真实成果变化可重提；分析只看选中依赖及其实际材料版本。"""
    from app.services.internal_research.research_action_validation import action_work_keys
    from app.services.internal_research.research_planner_loop_controller import advance_budget, decide_action_admission, init_budget
    from app.models.research_planner_loop_contracts import PlannerDecision, PlannerObservation
    finish = dict(decision=dict(kind="finish", reason="ready"))
    current = {**state(), "research_unit_results": []}
    old = action_work_keys(finish, current)
    observation = PlannerObservation(phase="validate", ok=False, error_code="finish_requirements_missing", pending_work_keys=old)
    budget = advance_budget(init_budget(), PlannerDecision(action="replan", reason="missing"), stage="validate", observation=observation)
    record = direct_evidence("selected")
    current.update(research_unit_results=[dict(unit_id="u1", status="evidence_found", evidence_ids=[record["evidence_id"]], claims=[])],
                   evidence_records=[record])
    changed = observation.model_copy(update={"pending_work_keys": action_work_keys(finish, current)})
    assert decide_action_admission(changed, budget).action == "replan"
    action = dict(decision=dict(kind="execute", actions=[dict(goal="read result", input=dict(source_type="inference_from_units",
        dependencies=[dict(unit_id="u1", required_product="materials", required=True)]))]))
    before = action_work_keys(action, current)
    current["evidence_records"].append(direct_evidence("unrelated"))
    assert action_work_keys(action, current) == before
    current["research_unit_results"][0]["status"] = "blocked"
    assert action_work_keys(action, current) != before


def test_initial_explicit_insufficient_finish_has_no_fake_execution():
    """没有执行材料时只能诚实缺证结束，不能虚构动作或引用。"""
    current = {**state(), "planner_draft": dict(requested_facets=[facet("first")],
        decision=dict(kind="finish", allow_partial=True, reason="no material available"))}
    result = pipeline.validate_planner_state(current, SimpleNamespace(rag_research_planner_max_units=8))
    assert result["final_delivery_report"]["status"] == "insufficient"
    assert result["source_status"]["local_evidence_execution"] == "not_run"
    assert rag_node.route_after_planner_validate(result) == "write_back"


def test_empty_read_feedback_and_explicit_partial_finish():
    """空读取不算完成；模型必须继续研究或明确选择部分交付。"""
    async def run():
        current = {**state(), "planner_draft": initial()}
        settings = SimpleNamespace(rag_research_planner_max_units=8)
        current.update(pipeline.validate_planner_state(current, settings))
        async def empty(unit, context):
            return read_result([])
        current.update(await pipeline.execute_planner_state(current, llm=ScriptedModel([]), settings=settings,
                                                           adapters={"local_direct_text": empty}))
        assert current["planner_loop_decision"] == "replan"
        assert current["evidence_sufficiency_report"]["missing_required_facet_ids"] == ["first", "second"]
        current["planner_draft"] = dict(decision=dict(kind="finish", reason="nothing found"))
        rejected = pipeline.validate_planner_state(current, settings)
        assert rejected["planner_observation"]["error_code"] == "finish_requirements_missing"
        current.update(rejected)
        current["planner_draft"]["decision"]["allow_partial"] = True
        current.update(pipeline.validate_planner_state(current, settings))
        assert rag_node.route_after_planner_validate(current) == "planner_synthesize"
        current.update(await pipeline.synthesize_planner_state(current, ScriptedModel([])))
        assert current["final_delivery_report"]["status"] == "insufficient"
        assert current["final_citation_bindings"] == []
    asyncio.run(run())


@pytest.mark.parametrize("failure", [RuntimeError("reader failed"), asyncio.CancelledError("cancelled")])
def test_parallel_failure_preserves_completed_sibling(failure):
    """并行读取失败或取消时，已成功的兄弟材料与回执仍可追溯。"""
    from app.services.internal_research.research_planner_executor import EvidenceCapabilityError, ResearchPlannerExecutor
    async def run():
        current = {**state(), "planner_draft": initial([read_action("first"), read_action("second")])}
        current.update(pipeline.validate_planner_state(current, SimpleNamespace(rag_research_planner_max_units=8)))
        plan = pipeline._plan_from_state(current)
        context = pipeline._execution_context(current, plan)
        async def adapter(unit, ctx):
            if unit.unit_id == "u2":
                raise failure
            return read_result([direct_evidence("completed")])
        executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter})
        expected = asyncio.CancelledError if isinstance(failure, asyncio.CancelledError) else EvidenceCapabilityError
        with pytest.raises(expected):
            await executor.execute_with_evidence(plan, context)
        assert context.execution_state.snapshot().evidence_records
        assert any(receipt["outcome"] == "success" for receipt in context.execution_state.snapshot().read_receipts)
    asyncio.run(run())


class ScriptedModel:
    """【离线模型替身】按步骤返回动作及选择真实测试材料的主张。"""

    model = "offline-model"
    temperature = 0
    base_url = "https://offline.invalid/v1"

    def __init__(self, records):
        """参数为测试原文；初始化确定性的外部模型响应队列。"""
        self.records = records
        self.prompts = []
        self.steps = [initial(), dict(decision=dict(kind="execute", actions=[read_action("second")])),
            dict(decision=dict(kind="execute", actions=[dict(goal="explain both facts", covered_facet_ids=["first", "second"],
                input=dict(source_type="inference_from_units", dependencies=[
                    dict(unit_id=unit_id, required_product="materials", required=True) for unit_id in ("u1", "u2")]))])),
            dict(decision=dict(kind="finish", reason="both facts supported"))]

    def with_structured_output(self, schema):
        """参数为产品实时合同；返回依照同一合同严格解析的替身调用端口。"""
        async def invoke(messages):
            self.prompts.append(messages[0].content)
            if schema in {InitialResearchDecisionOutput, ResearchDecisionOutput}:
                return schema.model_validate(self.steps.pop(0), strict=True)
            if schema is ResearchAnswerReviewDraft:
                facts = json.JSONDecoder().raw_decode(messages[0].content.split("Review facts:\n", 1)[1])[0]
                block_id = facts["blocks"][0]["block_id"]
                return schema.model_validate({"points": [{"facet_id": facet_id, "verdict": "answered",
                    "explanation": "正文给出了由实际材料支持的结论", "block_ids": [block_id], "claim_ids": ["u3.c1"],
                    "sources": [{"evidence_id": record["evidence_id"], "quote": "The paper uses attention."}
                                for record in self.records]} for facet_id in ("first", "second")],
                    "issues": [], "recommendation": "deliver", "reason": "当前两项结论均有对应正文与原文"})
            assert schema is ResearchClaimStrategyDraftBatch
            spans = [span.model_dump(exclude={"evidence_version"}) for span in supporting_spans(self.records)]
            return schema.model_validate(dict(claims=[dict(text="The paper uses attention.", claim_type="paper_supported",
                supporting_evidence_ids=[record["evidence_id"] for record in self.records], supporting_spans=spans,
                addressed_facet_ids=["first", "second"])]), strict=True)
        return SimpleNamespace(ainvoke=invoke)

    async def ainvoke(self, messages):
        """参数为最终生成输入；返回有实际主张和出处选择的离线正文。"""
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(dict(contract_version="research-planner-final-draft-v4", blocks=[
            dict(text="The paper uses attention.", product_kind="claims", facet_ids=["first", "second"],
                 section_ids=[], claim_ids=["u3.c1"])])))


def test_normal_compiled_graph_reads_then_analyzes_and_finishes(monkeypatch):
    """正常编译图读取两轮、分析累计正文、最终生成；外部写回替身只保存同一结果。"""
    records = [direct_evidence("first"), direct_evidence("second")]
    model = ScriptedModel(records)
    reads, saved = [], []

    async def first_analysis(current, llm):
        return {}

    async def init(current):
        return {}

    async def adapter(unit, context):
        reads.append(unit.unit_id)
        return read_result([records[len(reads) - 1]])

    async def resolve(*args):
        return None

    async def write_back(current):
        saved.append(json.loads(json.dumps({key: current[key] for key in
            ("answer", "research_plan", "research_unit_results", "final_delivery_report", "final_citation_bindings")})))
        return {}

    monkeypatch.setattr(pipeline, "_first_analysis_delta", first_analysis)
    monkeypatch.setattr(rag_node, "rag_init_node", init)
    monkeypatch.setattr(rag_node, "write_back_node", write_back)
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    monkeypatch.setattr(rag_node, "_planner_adapters", lambda current: {"local_direct_text": adapter})
    monkeypatch.setattr(rag_node, "_planner_optional_adapters", lambda current: {})
    monkeypatch.setattr(rag_node, "_planner_read_request", resolve)
    result = asyncio.run(rag_node.create_rag_node().ainvoke(state(), config={"recursion_limit": 40}))
    assert result.get("error_code", "") == "", result
    assert reads == ["u1", "u2"]
    assert len(result["research_unit_results"]) == 3
    assert result["research_plan"]["metadata"]["pending_unit_ids"] == []
    executed = [row for row in result["planner_budget"]["history"] if row["stage"] == "execute"]
    assert all(row["observation"]["scoped_in_index"] is None for row in executed)
    assert result["final_delivery_report"]["status"] == "complete", result["final_delivery_report"]
    assert len(result["final_citation_bindings"]) == 2
    assert saved[0]["answer"] == result["answer"]
    assert not model.steps
    assert "u1" in model.prompts[1] and records[0]["evidence_id"] in model.prompts[1]
    asyncio.run(_check_accumulated_result_qualification(result, model))


async def _check_accumulated_result_qualification(result, model):
    """参数为正常图成果；新独立读取不污染旧结论，篡改支持版本则阻断其下游。"""
    current = json.loads(json.dumps(result))
    current["planner_draft"] = dict(decision=dict(kind="execute", actions=[read_action("third")]))
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    current.update(pipeline.validate_planner_state(current, settings))
    async def read_third(unit, context):
        return read_result([direct_evidence("third")])
    current.update(await pipeline.execute_planner_state(current, llm=model, settings=settings,
                                                       adapters={"local_direct_text": read_third}))
    assert current["research_unit_results"][2]["status"] == "inferred"
    current["research_unit_results"][2]["claims"][0]["supporting_spans"][0]["evidence_version"] = "stale"
    action = dict(goal="use old fact", covered_facet_ids=["first"], input=dict(source_type="inference_from_units",
        dependencies=[dict(unit_id="u3", required_product="claims", required=True)]))
    current["planner_draft"] = dict(decision=dict(kind="execute", actions=[action]))
    current.update(pipeline.validate_planner_state(current, settings))
    current.update(await pipeline.execute_planner_state(current, llm=model, settings=settings,
                                                       adapters={"local_direct_text": read_third}))
    assert current["research_unit_results"][-1]["status"] == "blocked"
    old = next(item for item in current["research_unit_results"] if item["unit_id"] == "u3")
    assert old["claims"][0]["supporting_spans"][0]["evidence_version"] == "stale"
