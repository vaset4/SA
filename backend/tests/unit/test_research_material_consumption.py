"""【研究取证】【选择与续读】真实知识范围和模型输入边界离线回归。作者：xxx。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.models.research_action_contracts import MaterialsDependency, ReadSection
from app.models.research_planner_contracts import ConsumerExposure, ResearchUnit, ResearchUnitResult
from app.services.internal_research.research_planner_consumer_budget import (
    CLAIM_PROMPT_VERSION, configured_consumer_budget, prepare_consumer_input,
)
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from research_planner_test_utils import make_unit
from test_knowledge_access_service import _chunk, _item, _publish, _read

pytest_plugins = ["test_knowledge_access_service"]


def test_successive_continuations_reconstruct_exact_published_text(publication):
    """参数为隔离发布端口；有限预算逐次读取真实后文，拼接与原文完全一致，无返回值。"""
    body = "甲乙丙丁戊己庚辛壬癸" * 3
    request = _publish(publication, [_chunk(0, body)], {"s0": _item("s0", body)})
    request = replace(request, operation="sections", chunk_ids=(), section_paths=("Paper / 1 Method",), max_chars=8)
    current = _read(request)
    original_first = deepcopy(current.read_report.to_dict())
    texts = [current.materials[0].text]
    spans = [current.materials[0].source_spans[0]]
    while current.read_report.remaining:
        assert all(item["continuation_id"].startswith("continue-") for item in current.read_report.remaining)
        current = _read(replace(request, continuations=current.read_report.remaining))
        texts.extend(item.text for item in current.materials)
        spans.extend(span for item in current.materials for span in item.source_spans)
    assert "".join(texts) == body
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(body)
    assert all(left["end"] == right["start"] for left, right in zip(spans, spans[1:]))
    assert original_first["status"] == "partial"
    assert original_first["remaining"][0]["start"] == 8


@pytest.mark.parametrize("mutation", ["offset", "paper", "version", "section"])
def test_continuation_rejects_tampered_or_cross_scope_locator(publication, mutation):
    """参数为发布端口及非法变更；续读原范围、目标、版本和章节不能被替换，无返回值。"""
    body = "完整来源正文" * 10
    request = _publish(publication, [_chunk(0, body)], {"s0": _item("s0", body)})
    request = replace(request, operation="sections", chunk_ids=(), section_paths=("Paper / 1 Method",), max_chars=8)
    first = _read(request)
    entries = deepcopy(first.read_report.remaining)
    if mutation == "offset":
        entries[0]["start"] += 1
    elif mutation == "paper":
        request = replace(request, paper_ids=("outside",))
    elif mutation == "version":
        scope = deepcopy(request.scope)
        scope["publication"].update(paper_batches={"paper": "b" * 32}, batch_ids=["b" * 32])
        request = replace(request, scope=scope)
    else:
        request = replace(request, section_paths=("Other section",))
    with pytest.raises(ValueError):
        _read(replace(request, continuations=entries))


def _analysis(selections=None):
    """参数为具体材料选择；返回只消费已返回材料的分析单元。"""
    raw = make_unit("analysis", source_type="inference_from_units", allowed_claims=["paper_supported"])
    raw["dependencies"] = [{"unit_id": "read", "required_product": "materials", "selections": selections}]
    return ResearchUnit.model_validate(raw)


def _records():
    """无参数；返回靠后关键实验结果及独立前置材料，不预造分析结论。"""
    return [{"evidence_id": "front", "layer": "direct_text", "usable_as_citation": True,
             "paper_id": "paper-a", "excerpt": "方法设置。" * 2000},
            {"evidence_id": "result", "layer": "direct_text", "usable_as_citation": True,
             "paper_id": "paper-a", "excerpt": "25%通道训练减少70%时间，误差增加2%；完整通道增删验证留待未来。"}]


def test_material_selection_supplies_later_results_without_changing_evidence(monkeypatch):
    """参数为配置替身；明确选择后面的实验材料后整段送模，不再被前面的材料挤掉，无返回值。"""
    monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", 8000)
    records = _records()
    original = deepcopy(records)
    result = ResearchUnitResult(unit_id="read", status="evidence_found", evidence_ids=["front", "result"])
    all_input = ResearchUnitReasoner().prepare(_analysis(), [], [result], dependency_evidence_records=records)
    assert "result" in all_input.consumer.exposure.omitted_evidence_ids
    selected = ResearchUnitReasoner().prepare(_analysis([{"evidence_id": "result"}]), [], [result],
                                             dependency_evidence_records=records)
    assert records[1]["excerpt"] in selected.consumer.evidence_text
    assert [entry.evidence_id for entry in selected.consumer.exposure.evidence] == ["result"]
    assert records[0]["excerpt"] not in selected.consumer.evidence_text
    assert records == original


def test_unprovided_range_resumes_model_input_after_previous_prefix(monkeypatch):
    """参数为配置替身；系统后文range_id指向已取得而未送模部分，选中后从真实终点继续，无返回值。"""
    monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", 900)
    records = _records()[:1]
    result = ResearchUnitResult(unit_id="read", status="evidence_found", evidence_ids=["front"])
    first = ResearchUnitReasoner().prepare(_analysis(), [], [result], dependency_evidence_records=records)
    remaining = first.consumer.exposure.unprovided_ranges[0]
    old = ResearchUnitResult(unit_id="old_analysis", status="blocked", metadata={
        "provenance": {"exposure": first.consumer.exposure.model_dump(mode="json")}})
    second = ResearchUnitReasoner().prepare(_analysis([{"evidence_id": "front", "range_id": remaining.range_id}]),
        [], [result, old], dependency_evidence_records=records)
    observed = second.consumer.exposure.evidence[0]
    assert observed.start == first.consumer.exposure.evidence[0].end
    assert observed.start > 0 and observed.end > observed.start
    assert records[0]["excerpt"][observed.start:observed.end] in second.consumer.evidence_text
    assert second.consumer.exposure.unprovided_ranges[0].start == observed.end


@pytest.mark.parametrize("selection", [{"evidence_id": "outside"}, {"evidence_id": "result", "range_id": "invented"}])
def test_unavailable_material_or_range_is_rejected(selection):
    """参数为未取得材料或伪造范围；严格失败，不自动回到全部材料，无返回值。"""
    result = ResearchUnitResult(unit_id="read", status="evidence_found", evidence_ids=["front", "result"])
    with pytest.raises(ValueError):
        ResearchUnitReasoner().prepare(_analysis([selection]), [], [result], dependency_evidence_records=_records())


def test_none_budget_and_historical_zero_are_distinct(monkeypatch):
    """参数为配置替身；配置0提供完整材料，显式0保持空输入，旧投影可原样回读，无返回值。"""
    monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", 0)
    assert configured_consumer_budget() is None
    records = _records()
    full = prepare_consumer_input(records, [], evidence_max_chars=None, prompt_version=CLAIM_PROMPT_VERSION)
    empty = prepare_consumer_input(records, [], evidence_max_chars=0, prompt_version=CLAIM_PROMPT_VERSION)
    assert all(record["excerpt"] in full.evidence_text for record in records)
    assert full.exposure.evidence_max_chars is None and not full.exposure.unprovided_ranges
    assert not empty.exposure.evidence and len(empty.exposure.unprovided_ranges) == 2
    old = empty.exposure.model_dump(mode="json", exclude={"unprovided_ranges"})
    old["schema_version"] = "research-exposure-v1"
    old["rendering_version"] = "research-consumer-v3"
    assert ConsumerExposure.model_validate(old).evidence_max_chars == 0
    assert ConsumerExposure.model_validate(old).unprovided_ranges == ()


def test_selection_and_locator_schemas_reject_ambiguous_empty_input():
    """无参数；省略选择沿用全部依赖，显式空选择与两份章节定位均拒绝，无返回值。"""
    assert MaterialsDependency(unit_id="read", required_product="materials", required=True).selections is None
    with pytest.raises(ValidationError):
        MaterialsDependency(unit_id="read", required_product="materials", required=True, selections=[])
    with pytest.raises(ValidationError):
        ReadSection(source_type="local_section_text", target_paper_ids=["paper"],
                    required_sections=["Method"], continuation_ids=["continue-unknown"])


@pytest.mark.parametrize("field", ["rag_planner_consumer_text_max_chars", "rag_planner_section_text_max_chars"])
def test_read_and_consumer_config_accept_zero_but_reject_negative(field):
    """参数为既有配置项；0可关闭限额，负数始终明确拒绝，无返回值。"""
    assert getattr(Settings(_env_file=None, **{field: 0}), field) == 0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: -1})


@pytest.mark.parametrize("budget", [0, 8])
def test_adapter_and_reuse_request_share_budget_and_continuation_identity(monkeypatch, budget):
    """参数为配置与替身；正式适配器及复用请求使用同一None/正数定位，续读不会命中原前缀缓存。"""
    from app.nodes import rag_node
    from app.services.internal_research.research_planner_execution_state import ExecutionIdentity, PlannerExecutionState
    from o2_evidence_test_utils import bind_publication, scope
    from test_planner_section_text_tool import _row, _plan, _METHOD

    body = "甲乙丙丁戊己庚辛壬癸" * 3
    bind_publication(monkeypatch, [_row(0, _METHOD, body)])
    monkeypatch.setattr(rag_node.settings, "rag_planner_section_text_max_chars", budget)
    current = {"project_scope": scope()}
    identity = ExecutionIdentity(run_id="run", user_id="u1", scope={"paper_ids": ["paper-a"]},
                                 publication_snapshot=current["project_scope"]["publication"])
    context = SimpleNamespace(execution_state=PlannerExecutionState(identity))
    unit = _plan([_METHOD]).units[0]

    async def run():
        """无参数；通过执行账本保存第一次真实读取，随后核对续读请求与实际起点。"""
        request = await rag_node._planner_read_request(current, unit, context)
        actual = rag_node._planner_section_request(current, unit, context)
        assert request["max_chars"] == actual.max_chars == (None if budget == 0 else budget)
        first = await context.execution_state.read(unit_id=unit.unit_id, source_type="local_section_text",
            request=request, invoke=lambda: rag_node._planner_section_text_records(current, unit, context))
        if budget == 0:
            assert first.records[0]["excerpt"] == body and not first.read_report.remaining
            return
        ids = [entry["continuation_id"] for entry in first.read_report.remaining]
        continued = unit.model_copy(update={"evidence_policy": unit.evidence_policy.model_copy(update={
            "required_sections": [], "continuation_ids": ids})})
        next_request = await rag_node._planner_read_request(current, continued, context)
        assert next_request != request
        assert next_request["continuations"] == list(first.read_report.remaining)
        second = await rag_node._planner_section_text_records(current, continued, context)
        assert second.records[0]["source_spans"][0]["start"] == len(first.records[0]["excerpt"])
        with pytest.raises(ValueError):
            context.execution_state.resolve_continuations(["invented"], ["paper-a"])
        with pytest.raises(ValueError):
            context.execution_state.resolve_continuations(ids, ["outside"])

    asyncio.run(run())


def test_selected_range_rejects_changed_evidence_version(monkeypatch):
    """参数为配置替身；旧未提供范围不能静默套用到变更后的证据正文，无返回值。"""
    monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", 900)
    records = _records()[:1]
    result = ResearchUnitResult(unit_id="read", status="evidence_found", evidence_ids=["front"])
    first = ResearchUnitReasoner().prepare(_analysis(), [], [result], dependency_evidence_records=records)
    remaining = first.consumer.exposure.unprovided_ranges[0]
    old = ResearchUnitResult(unit_id="old_analysis", status="blocked", metadata={
        "provenance": {"exposure": first.consumer.exposure.model_dump(mode="json")}})
    changed = [{**records[0], "excerpt": records[0]["excerpt"].replace("方法", "改变")}]
    with pytest.raises(ValueError, match="版本"):
        ResearchUnitReasoner().prepare(_analysis([{"evidence_id": "front", "range_id": remaining.range_id}]),
            [], [result, old], dependency_evidence_records=changed)


def test_failed_analysis_preserves_unprovided_ranges_for_followup_selection(monkeypatch):
    """参数为配置替身；零合格主张仍保存真实送模/未提供范围，下一步可以明确选择后文。"""
    from app.services.internal_research.research_planner_execution_state import PlannerExecutionState
    from app.services.internal_research.research_planner_executor import PlannerExecutionContext, ResearchPlannerExecutor

    monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", 900)
    records = _records()
    read = ResearchUnitResult(unit_id="read", status="evidence_found", evidence_ids=["front", "result"])
    reasoner = ResearchUnitReasoner(draft_provider=lambda *args, **kwargs: [])
    executor = ResearchPlannerExecutor(reasoner=reasoner, adapters={})
    context = PlannerExecutionContext(execution_state=PlannerExecutionState(evidence_records=records))
    blocked = asyncio.run(executor._execute_inference(_analysis(), [read], context))
    assert blocked.status == "blocked" and not blocked.claims
    exposure = blocked.metadata["provenance"]["exposure"]
    selected_range = next(item for item in exposure["unprovided_ranges"] if item["evidence_id"] == "result")
    next_input = reasoner.prepare(_analysis([{"evidence_id": "result", "range_id": selected_range["range_id"]}]),
        [], [read, blocked], dependency_evidence_records=records)
    assert records[1]["excerpt"] in next_input.consumer.evidence_text


@pytest.mark.parametrize("selection", [{"evidence_id": "outside"}, {"evidence_id": "result", "range_id": "invented"}])
def test_action_admission_reports_invalid_material_selection_before_execution(selection):
    """参数为无效材料或范围；正式动作准入反馈具体选择错误，尚未执行读取或分析。"""
    from app.models.research_planner_contracts import PlannerCoverageReview
    from app.services.internal_research.research_action_validation import validate_actions
    from app.services.internal_research.research_planner_validator import PlannerValidationOptions

    draft = {"decision": {"kind": "execute", "actions": [{"goal": "读取已有实验结果", "input": {
        "source_type": "inference_from_units", "dependencies": [{"unit_id": "read", "required": True,
        "required_product": "materials", "selections": [selection]}]}}]}}
    outcome, _ = validate_actions(draft, options=PlannerValidationOptions(question="实验结果", requirements_baseline=PlannerCoverageReview()),
        previous_plan=None, results=[{"unit_id": "read", "status": "evidence_found", "evidence_ids": ["result"]}])
    assert outcome.failure.code == "action_material_selection_invalid"
    assert outcome.plan is None


def test_material_selection_survives_action_to_execution_projection():
    """无参数；模型选择在正式动作转换中原样进入材料依赖，不丢范围、不扩大为整个上游。"""
    from app.models.research_action_contracts import ResearchAction
    from app.models.research_planner_contracts import PlannerCoverageReview, UnitDependency
    from app.services.internal_research.research_action_validation import _project_action

    selections = [{"evidence_id": "result", "range_id": "range-offered"}]
    action = ResearchAction.model_validate({"goal": "分析后文", "input": {"source_type": "inference_from_units",
        "dependencies": [{"unit_id": "read", "required": True, "required_product": "materials", "selections": selections}]}})
    projected = _project_action(action, "analysis", PlannerCoverageReview())
    assert [item.model_dump() for item in projected.dependencies[0].selections] == selections
    with pytest.raises(ValidationError, match="materials dependency"):
        UnitDependency(unit_id="read", required_product="claims", selections=selections)
