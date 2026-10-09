"""O4 明确支持与实际送模范围入口回归。作者：xxx。"""

import asyncio

import pytest

from app.models.research_planner_contracts import (
    ResearchClaim, ResearchClaimDraft, ResearchUnit, ResearchUnitResult,
)
from app.services.internal_research.research_planner_consumer_budget import prepare_consumer_input
from app.services.internal_research.research_planner_support import resolve_support
from app.services.internal_research.research_planner_fulfillment import evaluate_delivered_facet
from app.services.internal_research.research_planner_validator import PlannerValidationOptions
from research_planner_strategy_test_utils import validate_strategy_fixture, current_plan_fixture
from app.services.internal_research.research_planner_executor import ResearchPlannerExecutor, PlannerExecutionContext
from app.services.internal_research.research_planner_execution_state import PlannerExecutionState, ExecutionIdentity
from app.config import get_settings
from o2_evidence_test_utils import direct_evidence, read_result

from app.services.internal_research.research_planner_reasoner import validate_claim_drafts
from research_planner_test_utils import make_unit, make_draft, supporting_spans, result_with_exposure, explicit_claim_provider
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner


def test_direct_claim_without_actual_quote_is_rejected() -> None:
    """无参数；论文事实没有可核对原文片段时不能获得验证资格。"""
    unit = ResearchUnit.model_validate(make_unit())
    records = [{"evidence_id": "e1", "paper_id": "paper-a", "layer": "direct_text",
                "usable_as_citation": True, "excerpt": "A has accuracy 42 percent."}]
    draft = ResearchClaimDraft(text="A has accuracy 99 percent.", claim_type="paper_supported",
                               supporting_evidence_ids=["e1"])
    outcome = validate_claim_drafts(unit, records, [], [draft])
    assert not outcome.claims
    assert "supporting_spans" in ";".join(outcome.errors)


def _record(key="a", paper="paper-a"):
    """参数为材料/论文身份；返回原文固定且可核对的测试来源。"""
    return {"evidence_id": key, "paper_id": paper, "layer": "direct_text", "usable_as_citation": True,
            "excerpt": "Accuracy is 42 percent under condition X."}


def _claim(key, records, **fields):
    """参数为明确来源与主张字段；返回绑定已知原文的事实。"""
    return ResearchClaim(claim_id=key, text="Accuracy is 42 percent under condition X.",
                         claim_type="paper_supported", validation_status="validated",
                         supporting_evidence_ids=[record["evidence_id"] for record in records],
                         supporting_spans=supporting_spans(records), addressed_facet_ids=["f1"], **fields)


def _source_plan():
    """无参数；返回 A/B 取证和只选择特定事实的合成任务。"""
    a = make_unit("a", paper_ids=["paper-a", "paper-b"])
    b = make_unit("b", goal="compare", source_type="inference_from_units", allowed_claims=["evidence_grounded_inference"],
                  output_role="synthesis", deps=[{"unit_id": "a", "required_product": "claims"}])
    return current_plan_fixture(make_draft([a, b]))


def test_selected_claim_inherits_only_its_support_without_rereading() -> None:
    """无参数；选择 c1 不继承同单元 c2 或 related 材料，也不要求本次重读原文。"""
    plan = _source_plan()
    a, b = _record(), _record("b", "paper-b")
    upstream = result_with_exposure(plan.units[0], [a, b], [_claim("a.c1", [a], related_evidence_ids=["b"]), _claim("a.c2", [b])])
    inference = ResearchClaimDraft(text="仅依据 A 的分析", claim_type="evidence_grounded_inference",
                source_claim_ids=["a.c1"], reasoning_summary="选择 A 的具体事实", addressed_facet_ids=["f1"])
    outcome = validate_claim_drafts(plan.units[1], [], [upstream], [inference], dependency_evidence_records=[a, b], plan=plan)
    assert not outcome.errors
    result = ResearchUnitResult(unit_id="b", status="inferred", claims=outcome.claims, metadata={"provenance": outcome.provenance})
    final_input = prepare_consumer_input([], outcome.claims, evidence_max_chars=0, prompt_version="final-v3")
    support = resolve_support(plan, [upstream, result], [a, b], exposure=final_input.exposure)["b.c1"]
    assert support.qualified
    assert support.evidence_ids == ("a",)
    assert support.paper_ids == ("paper-a",)
    assert support.claim_ids == ("a.c1",)
    assert outcome.claims[0].source_unit_ids == ["a"]


@pytest.mark.parametrize("mutation,error", [("quote", "support_quote_mismatch"), ("version", "support_span_not_exposed"),
                                            ("unexposed", "support_span_not_exposed")])
def test_direct_support_rejects_unread_or_changed_excerpt(mutation, error) -> None:
    """参数为真实边界变更；直接事实必须绑定实际投影内的同版原文。"""
    unit = ResearchUnit.model_validate(make_unit())
    record = _record()
    span = supporting_spans([record])[0]
    prepared = prepare_consumer_input([record], [], evidence_max_chars=1000, prompt_version="claim-v3")
    if mutation == "quote":
        span = span.model_copy(update={"quote": "Accuracy is 99 percent"})
    elif mutation == "version":
        record = {**record, "excerpt": "Changed source body"}
    else:
        prepared = prepare_consumer_input([record], [], evidence_max_chars=0, prompt_version="claim-v3")
    draft = ResearchClaimDraft(text="accuracy", claim_type="paper_supported", supporting_evidence_ids=["a"], supporting_spans=[span])
    outcome = validate_claim_drafts(unit, [record], [], [draft], prepared_input=prepared)
    assert not outcome.claims
    assert error in ";".join(outcome.errors)


@pytest.mark.parametrize("source_id", ["missing.c1", "b.c1"])
def test_source_claim_must_exist_in_direct_dependency(source_id) -> None:
    """参数为不存在或自引用的来源；不能由后端猜测为整单元依赖。"""
    plan, record = _source_plan(), _record()
    upstream = result_with_exposure(plan.units[0], [record], [_claim("a.c1", [record])])
    draft = ResearchClaimDraft(text="analysis", claim_type="evidence_grounded_inference", source_claim_ids=[source_id], reasoning_summary="selected")
    outcome = validate_claim_drafts(plan.units[1], [], [upstream], [draft], dependency_evidence_records=[record], plan=plan)
    assert not outcome.claims
    assert any("source_claim" in reason for reason in outcome.errors)


def test_recommendation_credit_is_owned_selected_and_order_independent() -> None:
    """无参数；推荐信用只能来自正文选中且显式关联的真实推荐单元主张，第二条也可兑现。"""
    plan, record = _source_plan(), _record()
    unit = plan.units[1].model_copy(update={"output_role": "final_recommendation"})
    facet = plan.coverage_review.explicit_facets[0].model_copy(update={"required_output_role": "final_recommendation"})
    plan = plan.model_copy(update={"units": [plan.units[0], unit],
            "coverage_review": plan.coverage_review.model_copy(update={"explicit_facets": [facet]})})
    upstream = result_with_exposure(plan.units[0], [record], [_claim("a.c1", [record])])
    drafts = [ResearchClaimDraft(text="auxiliary" if index == 0 else "recommendation", claim_type="evidence_grounded_inference",
              source_claim_ids=["a.c1"], reasoning_summary="selected support", addressed_facet_ids=[] if index == 0 else ["f1"])
              for index in range(2)]
    outcome = validate_claim_drafts(unit, [], [upstream], drafts, dependency_evidence_records=[record], plan=plan)
    assert not outcome.errors
    result = ResearchUnitResult(unit_id="b", status="inferred", claims=outcome.claims, metadata={"provenance": outcome.provenance})
    complete = evaluate_delivered_facet(plan, facet, [upstream, result], [record], claim_ids=["b.c2"], evidence_ids=["a"])
    assert complete.satisfied and complete.claim_ids == ["b.c2"]
    missing = evaluate_delivered_facet(plan, facet, [upstream, result], [record], claim_ids=["a.c1", "b.c1"], evidence_ids=["a"])
    assert not missing.satisfied


@pytest.mark.parametrize("mutation", ["title", "remove_section", "role"])
def test_sections_and_role_survive_replanning_baseline(mutation) -> None:
    """无参数；首份计划的章节义务不能在重规划时删除或改名。"""
    draft = make_draft()
    draft["coverage_review"]["answer_sections"] = [{"section_id": "findings", "title": "结论", "facet_ids": ["f1"],
                                                  "required": True, "source": "explicit", "user_text_span": "结论", "presentation_kind": "exact_heading"}]
    draft["question"] += " 请按结论分节"
    first = validate_strategy_fixture(draft)
    assert first.plan is not None
    if mutation == "title":
        draft["coverage_review"]["answer_sections"][0]["title"] = "可选扩展"
    elif mutation == "remove_section":
        draft["coverage_review"]["answer_sections"] = []
    else:
        draft["coverage_review"]["explicit_facets"][0]["required_output_role"] = "final_recommendation"
    changed = validate_strategy_fixture(draft, options=PlannerValidationOptions(requirements_baseline=first.plan.coverage_review))
    assert changed.failure
    assert "invalid_plan_format" in {issue["code"] for issue in changed.failure.trace["validation_errors"]}


def test_historical_plan_keeps_support_unknown() -> None:
    """无参数；历史计划可以读取，但不因当前准备而获得新版支持资格。"""
    plan = current_plan_fixture(make_draft()).model_copy(update={"contract_version": "research-planner-v2"})
    record = _record()
    result = result_with_exposure(plan.units[0], [record], [_claim("a.c1", [record])])
    support = resolve_support(plan, [result], [record])["a.c1"]
    assert not support.qualified and "historical_support_unknown" in support.reasons


@pytest.mark.parametrize("input_change", ["budget", "rendering"])
def test_executor_cache_uses_prepared_budget_and_preserves_original_exposure(monkeypatch, input_change) -> None:
    """参数为配置替身；正式执行同输入可复用，预算改变必须重新草拟且旧投影不变。"""
    record = direct_evidence("cache")
    plan = validate_strategy_fixture(make_draft()).plan
    assert plan is not None
    identity = ExecutionIdentity(run_id="run", user_id="user", scope={"paper_ids": ["paper-a"]},
                                 publication_snapshot={"paper_batches": {"paper-a": record["source_spans"][0]["batch_id"]}})
    calls = []

    def provider(unit, records, dependencies, **kwargs):
        """参数为正式草拟输入；记录调用并返回带明确来源的外部替身结果。"""
        calls.append(kwargs["prepared_input"].exposure)
        return explicit_claim_provider(unit, records, dependencies, **kwargs)

    def adapter(unit, context):
        """参数为实际请求；返回固定原文与读取报告，不连接外部服务。"""
        return read_result([record])

    executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter}, reasoner=ResearchUnitReasoner(draft_provider=provider))
    context = PlannerExecutionContext(execution_state=PlannerExecutionState(identity), result_context={"model": "test", "prompt": "v3"})
    first = asyncio.run(executor.execute_with_evidence(plan, context))
    assert first.results[0].status == "evidence_found"
    original = first.results[0].metadata["provenance"]["exposure"]
    again = asyncio.run(executor.execute_with_evidence(plan, context, previous_results=first.results))
    assert len(calls) == 1 and again.results[0].metadata["result_reused"]
    if input_change == "budget":
        monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", get_settings().rag_planner_consumer_text_max_chars + 1)
    else:
        from app.services.internal_research import research_planner_consumer_budget as consumer
        original_renderer = consumer.evidence_source_prompt
        monkeypatch.setattr(consumer, "evidence_source_prompt", lambda item: original_renderer(item) + "\nOriginal source:")
    third = asyncio.run(executor.execute_with_evidence(plan, context, previous_results=again.results))
    assert len(calls) == 2 and not third.results[0].metadata["result_reused"]
    assert first.results[0].metadata["provenance"]["exposure"] == original


def test_removed_plan_owner_cannot_keep_current_support_credit() -> None:
    """无参数；已经从当前计划移除的单元不能依靠保存的旧声明继续贡献支持。"""
    plan, record = _source_plan(), _record()
    upstream = result_with_exposure(plan.units[0], [record], [_claim("a.c1", [record])])
    current = plan.model_copy(update={"units": [plan.units[1]]})
    support = resolve_support(current, [upstream], [record])["a.c1"]
    assert not support.qualified
    assert "claim_owner_unknown:a" in support.reasons


def test_prepare_rejects_conflicting_source_identity() -> None:
    """无参数；同 ID 不同正文在准备边界显式失败，不能先合并覆盖而隐藏 O2 冲突。"""
    unit = ResearchUnit.model_validate(make_unit())
    record = _record()
    with pytest.raises(ValueError, match="conflict|冲突"):
        ResearchUnitReasoner().prepare(unit, [record, {**record, "excerpt": "different body"}], [])


def test_three_level_claim_rebinding_keeps_original_exposure_and_revokes_descendants() -> None:
    """无参数；三层推断连续重编号仅重绑身份，原投影保留，前置撤销立即使下游失去信用。"""
    record = direct_evidence("chain")
    identity = ExecutionIdentity(run_id="run", user_id="user", scope={"paper_ids": ["paper-a"]},
                                 publication_snapshot={"paper_batches": {"paper-a": record["source_spans"][0]["batch_id"]}})
    calls = []

    def adapter(unit, context):
        """参数为正式读取请求；返回固定来源正文与读取事实。"""
        return read_result([record])

    def provider(unit, records, dependencies, **kwargs):
        """参数为实际草拟输入；记录真实模型替身调用并显式选择依赖主张。"""
        calls.append(unit.unit_id)
        return explicit_claim_provider(unit, records, dependencies, **kwargs)

    executor = ResearchPlannerExecutor(adapters={"local_direct_text": adapter}, reasoner=ResearchUnitReasoner(draft_provider=provider))
    context = PlannerExecutionContext(execution_state=PlannerExecutionState(identity), result_context={"model": "test", "prompt": "v3"})
    source = make_unit("source")
    synthesis = make_unit("synthesis", goal="compare", source_type="inference_from_units", output_role="synthesis",
                          allowed_claims=["evidence_grounded_inference"], deps=[{"unit_id": "source"}])
    final = make_unit("final", goal="recommend", source_type="inference_from_units", output_role="final_recommendation",
                      allowed_claims=["evidence_grounded_inference"], deps=[{"unit_id": "synthesis"}])
    plan = current_plan_fixture(make_draft([source, synthesis, final]))
    first = asyncio.run(executor.execute_with_evidence(plan, context))
    assert [result.status for result in first.results] == ["evidence_found", "inferred", "inferred"]
    original_exposures = [result.metadata["provenance"]["exposure"] for result in first.results]
    previous = first.results
    for prefix in ("renamed-", "again-"):
        mapping = {unit.unit_id: prefix + unit.unit_id for unit in plan.units}
        units = [unit.model_copy(update={"unit_id": mapping[unit.unit_id], "dependencies": [
            dependency.model_copy(update={"unit_id": mapping[dependency.unit_id]}) for dependency in unit.dependencies]})
            for unit in plan.units]
        plan = plan.model_copy(update={"units": units})
        report = asyncio.run(executor.execute_with_evidence(plan, context, previous_results=previous))
        assert all(result.metadata["result_reused"] for result in report.results)
        assert [result.metadata["provenance"]["exposure"] for result in report.results] == original_exposures
        support = resolve_support(plan, report.results, report.evidence_records)
        assert all(item.qualified for item in support.values())
        assert support[report.results[-1].claims[0].claim_id].evidence_ids == (record["evidence_id"],)
        previous = report.results
    assert calls == ["source", "synthesis", "final"]
    revoked = [previous[0].model_copy(update={"status": "blocked", "claims": []}), *previous[1:]]
    support = resolve_support(plan, revoked, report.evidence_records)
    assert not any(item.qualified for item in support.values())
    assert first.results[1].claims[0].source_claim_ids == ["source.c1"]


def test_cross_run_source_claim_is_rejected() -> None:
    """无参数；相同材料和主张编号也不能跨运行借用依赖结论。"""
    plan, record = _source_plan(), _record()
    upstream = result_with_exposure(plan.units[0], [record], [_claim("a.c1", [record])])
    metadata = {"provenance": {**upstream.metadata["provenance"], "execution_identity": {"run_id": "older"}}}
    upstream = upstream.model_copy(update={"metadata": metadata})
    draft = ResearchClaimDraft(text="analysis", claim_type="evidence_grounded_inference",
                               source_claim_ids=["a.c1"], reasoning_summary="selected support")
    outcome = validate_claim_drafts(plan.units[1], [], [upstream], [draft], dependency_evidence_records=[record],
                                    plan=plan, execution_identity={"run_id": "current"})
    assert not outcome.claims
    assert "source_claim_cross_run:a.c1" in outcome.errors


def test_related_background_does_not_complete_required_paper() -> None:
    """无参数；A 事实携带 B 背景不能完成明确要求 B 事实的任务。"""
    plan = _source_plan()
    a, b = _record(), _record("b", "paper-b")
    facet = plan.coverage_review.explicit_facets[0].model_copy(update={"required_paper_ids": ["paper-a", "paper-b"]})
    result = result_with_exposure(plan.units[0], [a, b], [_claim("a.c1", [a], related_evidence_ids=["b"])])
    report = evaluate_delivered_facet(plan, facet, [result], [a, b], claim_ids=["a.c1"], evidence_ids=["a", "b"])
    assert not report.satisfied
    assert report.missing_paper_ids == ["paper-b"]
