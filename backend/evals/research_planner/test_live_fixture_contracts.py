"""【规划评测】【当前适配】离线替身使用实时合同，旧种子和旧成绩不变。作者：xxx。"""

import asyncio
import copy
import json

import pytest
from langchain_core.messages import HumanMessage

from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft, ResearchReplanStrategyDraft
from app.models.research_planner_contracts import ResearchUnit
from app.services.internal_research.research_planner_claim_provider import build_claim_prompt
from evals.research_planner import live_fixture


def _record():
    """无参数；返回明确的双取证及单主张推断种子，均为测试生成数据。"""
    return {"question": "整理材料并比较证据", "environment": {"depth": "research"}, "seed": {
        "research_plan": {"units": [
            {"unit_id": "u1", "goal": "第一份材料", "research_action": "读取材料一",
             "evidence_policy": {"source_type": "local_direct_text"}},
            {"unit_id": "u2", "goal": "第二份材料", "research_action": "读取材料二",
             "evidence_policy": {"source_type": "local_direct_text"}},
            {"unit_id": "u3", "goal": "比较证据", "research_action": "基于明确主张解释",
             "evidence_policy": {"source_type": "inference_from_units"}},
        ]}, "research_unit_results": [
            {"unit_id": "u1", "evidence_ids": ["seed-one"], "claims": [{"text": "第一项事实",
             "claim_type": "paper_supported", "supporting_evidence_ids": ["seed-one"]}]},
            {"unit_id": "u2", "evidence_ids": ["seed-two"], "claims": [{"text": "第二项事实",
             "claim_type": "paper_supported", "supporting_evidence_ids": ["seed-two"]}]},
            {"unit_id": "u3", "claims": [{"text": "根据第二项事实解释", "claim_type": "evidence_grounded_inference",
             "source_unit_ids": ["u2"]}]},
        ], "answer": "Fixture answer"}}


def test_fixture_accepts_current_plan_schema_without_mutating_seed():
    """无参数；验证当前图所请求的新策略schema可由适配器正常返回。"""
    record = _record()
    original = copy.deepcopy(record)
    output = live_fixture._FixtureLLM(record).with_structured_output(ResearchPlanStrategyDraft)
    draft = asyncio.run(output.ainvoke([HumanMessage(content="offline plan prompt")]))
    assert isinstance(draft, ResearchPlanStrategyDraft)
    assert "contract_version" not in draft.model_dump()
    assert draft.units[2].dependencies[1].required_product == "claims"
    assert record == original


def test_fixture_replan_only_returns_actions():
    """无参数；重规划适配同一行动合同，不回传另一份内容要求。"""
    output = live_fixture._FixtureLLM(_record()).with_structured_output(ResearchReplanStrategyDraft)
    draft = asyncio.run(output.ainvoke([HumanMessage(content="offline replan")]))
    assert isinstance(draft, ResearchReplanStrategyDraft)
    assert "requested_facets" not in draft.model_dump()


def test_claim_prompt_selects_second_unit_and_no_default_identity():
    """无参数；实际任务投影选中第二单元，缺失或重复匹配时不能默认第一单元。"""
    record = _record()
    raw = record["seed"]["research_plan"]["units"][1]
    unit = ResearchUnit.model_validate(live_fixture._complete_unit(raw, 2, previous_units=[]))
    prompt = [HumanMessage(content=build_claim_prompt(unit, [], []))]
    assert live_fixture._unit_id_from_prompt(record, prompt) == "u2"
    with pytest.raises(ValueError, match="one task projection"):
        live_fixture._unit_id_from_prompt(record, [HumanMessage(content="missing identity")])
    record["seed"]["research_plan"]["units"][0].update(goal=raw["goal"], research_action=raw["research_action"])
    with pytest.raises(ValueError, match="one declared action"):
        live_fixture._unit_id_from_prompt(record, prompt)


def test_legacy_support_mapping_requires_a_single_selected_claim():
    """无参数；旧单元选择只有一个具体主张时可映射，多主张或缺上游时明确失败。"""
    record = _record()
    draft = live_fixture._claim_drafts(record, "u3")[0]
    assert draft.source_claim_ids == ["u2.c1"]
    assert "source_unit_ids" not in draft.model_dump()
    original = copy.deepcopy(record)
    record["seed"]["research_unit_results"][1]["claims"].append({"text": "another", "claim_type": "paper_supported"})
    with pytest.raises(ValueError, match="unique upstream claim"):
        live_fixture._claim_drafts(record, "u3")
    record["seed"]["research_unit_results"][2]["claims"][0]["source_claim_ids"] = ["u2.c1"]
    assert live_fixture._claim_drafts(record, "u3")[0].source_claim_ids == ["u2.c1"]
    assert original["seed"]["research_unit_results"][1]["claims"][0]["text"] == "第二项事实"


def test_fixture_runs_current_nodes_and_keeps_original_seed():
    """无参数；真实分析、校验、执行节点接收新schema，保留来源选择及旧种子字节。"""
    record = _record()
    before = json.dumps(record, ensure_ascii=False, sort_keys=True)
    fixture = live_fixture._RecordFixture(record=record)
    result = asyncio.run(fixture.run(live_fixture.run_planner_nodes, initial_state=fixture._initial_state()))
    assert result.get("research_plan"), result
    assert result["research_plan"]["contract_version"] == "research-planner-v4"
    results = {item["unit_id"]: item for item in result["research_unit_results"]}
    assert results["u1"]["claims"][0]["text"] == "第一项事实"
    assert results["u2"]["claims"][0]["text"] == "第二项事实"
    assert results["u3"]["claims"][0]["source_claim_ids"] == ["u2.c1"]
    assert json.dumps(record, ensure_ascii=False, sort_keys=True) == before
