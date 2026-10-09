"""【研究决策】【历史与新链路】持久草稿恢复和实际编译主图保持业务成果。作者：xxx。"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models.research_action_contracts import ResearchDecisionOutput
from app.nodes import rag_node
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research.research_action_validation import action_work_keys
from app.services.internal_research.research_planner_conversion import migrate_execution_checkpoint
from o2_evidence_test_utils import direct_evidence, read_result
from .test_incremental_research import ScriptedModel, initial, read_action, state
from .test_main_graph_answer_delivery import install_workflow


def test_legacy_unaccepted_draft_recovers_without_reinterpreting_the_new_wire():
    """无参数；旧合法思路随原草稿保留，要求已接纳但无计划时仍请求严格后续新合同。"""
    action = read_action("first")
    action["input"]["target_paper_ids"] = ["foreign-paper"]
    current = {**state(), "planner_draft": {**initial([action]), "research_intent": "旧草稿说明"}}
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    current.update(pipeline.validate_planner_state(current, settings))
    original = deepcopy(current)
    assert not current.get("research_plan") and current["coverage_review"]
    assert migrate_execution_checkpoint(current) == {} and current == original
    model = ScriptedModel([])
    model.steps = [{"decision": {"kind": "execute", "actions": [read_action("first")]}}]
    delta = asyncio.run(pipeline.analyze_planner_state(current, model, settings))
    assert not delta.get("error_code")
    assert "requested_facets" not in delta["planner_draft"]
    assert delta["planner_draft"]["research_intent"] == ""
    assert current["planner_draft"]["research_intent"] == "旧草稿说明"
    assert current["coverage_review"] == original["coverage_review"]


@pytest.mark.asyncio
async def test_legacy_pending_execution_retains_work_keys_and_saved_reasoning():
    """无参数；旧已准入草稿继续读取，执行后记录相同工作身份且保存原说明与草稿。"""
    current = {**state(), "planner_draft": {**initial(), "research_intent": "历史未来思路"}}
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    current.update(pipeline.validate_planner_state(current, settings))
    original = deepcopy(current)
    work = action_work_keys(current["planner_draft"], current)
    calls = []
    async def adapter(unit, context):
        """参数为获准读取单元及上下文；返回人工材料，记录实际离线读取次数。"""
        calls.append(unit.unit_id)
        return read_result([direct_evidence("first")])
    async def resolve(*args):
        """参数为读取解析请求；沿既有离线端口，不访问生产库。"""
        return None
    result = await pipeline.execute_planner_state(current, llm=None, settings=settings,
        adapters={"local_direct_text": adapter}, request_resolver=resolve)
    assert calls == ["u1"] and result["research_unit_results"]
    assert result["planner_observation"]["attempted_work_keys"] == work
    assert result["research_plan"]["metadata"]["analyzer_reasoning"] == "历史未来思路"
    assert result["research_plan"]["metadata"]["pending_unit_ids"] == []
    assert current == original and migrate_execution_checkpoint(current) == {}


def test_invalid_legacy_draft_is_not_accepted_by_history_recovery():
    """无参数；历史持久合同同样严格，未知字段不能借恢复入口获得资格。"""
    draft = {**initial(), "research_intent": "历史说明", "unknown_option": True}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        migrate_execution_checkpoint({**state(), "planner_draft": draft})


@pytest.mark.asyncio
async def test_compiled_graph_rejects_bypassed_new_wire_and_keeps_prior_reads(monkeypatch):
    """参数为隔离端口；真实主图在读取后拒绝替身绕过schema的旧字段，保留已有材料。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=False)
    model = observed["model"]
    original = model.with_structured_output
    def structured(schema):
        """参数为实时合同；第二次规划返回人工非法原值，其余使用真实测试端口。"""
        if schema is ResearchDecisionOutput:
            async def invalid(messages):
                """参数为实际规划消息；返回含旧字段的字典，故意绕过替身的提前校验。"""
                return {"decision": {"kind": "finish", "reason": "仍应核对"}, "research_intent": "非法新输出"}
            return SimpleNamespace(ainvoke=invalid)
        return original(schema)
    monkeypatch.setattr(model, "with_structured_output", structured)
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    result = await workflow.ainvoke({**state(), "intent": "rag_qa"}, config={"configurable": {"thread_id": "new-wire-invalid"}})
    assert observed["reads"] == ["u1"]
    assert result["research_unit_results"] and result["evidence_records"]
    assert result["final_delivery_report"]["status"] == "failed"
    assert len(observed["rag"]) == 1
