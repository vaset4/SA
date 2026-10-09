"""【研究交付】【内容核对回归】来源、缺口、错误边界与同次保存。作者：xxx。"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.internal_research.research_answer_review import (
    prepare_review_input, review_final_answer, review_requires_continuation, validate_review_delivery,
)
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewDraft
from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer_async
from app.services.internal_research.research_planner_final_provider import prepare_final_input
from test_research_planner_delivery_policy import case


def review_draft(*, verdict="answered", recommendation="deliver", issues=None):
    """参数为离线核对判断；返回引用真实测试原文的模型输出，无真实模型调用。"""
    return ResearchAnswerReviewDraft.model_validate({
        "points": [{"facet_id": "f1", "verdict": verdict, "explanation": "正文给出了条件A与测量值91%。",
                    "block_ids": ["b1"], "claim_ids": [],
                    "sources": [{"evidence_id": "e1", "quote": "条件A下测量值为91%。"}]}],
        "issues": issues or [], "recommendation": recommendation, "reason": "逐要点核对的具体结论",
        "partial_basis": {"kind": "current_scope", "explanation": "当前材料只包含条件A的结果，所问条件未覆盖",
            "sources": [{"evidence_id": "e1", "quote": "条件A下测量值为91%。"}], "read_unit_ids": []}
            if recommendation == "partial" else None,
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,recommendation,status", [
    ("answered", "deliver", "complete"), ("missing", "continue_research", "partial"),
    ("missing", "partial", "partial"), ("unsupported", "revise_answer", "partial"),
])
async def test_semantic_assessment_changes_structurally_complete_delivery(verdict, recommendation, status):
    """参数为真实语义判断分支；正文有合法引用不能绕过缺内容核对，原输入保持不变。"""
    payload, raw = case()
    final = validate_planner_final_answer(payload, raw)
    assert final.delivery_report.status == "complete"
    original = deepcopy(payload)
    reviewer = SimpleNamespace(review=AsyncMock(return_value=review_draft(verdict=verdict, recommendation=recommendation)))
    reviewed = await review_final_answer(payload, final, reviewer)
    assert payload == original
    assert reviewer.review.await_count == 1
    assert reviewed.delivery_report.status == status
    assert reviewed.answer == final.answer
    assert reviewed.delivery_report.requirement_results[0].status == ("delivered" if verdict == "answered" else "missing")
    received = reviewer.review.call_args.args[0]
    assert received.answer == final.answer and received.facts["question"] == payload.question
    assert received.facts["sources"][0]["text"] == payload.evidence_records[0]["excerpt"]
    assert reviewed.delivery_report.content_review.attempt_id == payload.attempt_id
    report = reviewed.delivery_report.model_dump(mode="json")
    validate_review_delivery(reviewed.answer, report)
    expected = verdict != "answered" and recommendation != "partial"
    assert review_requires_continuation(report) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["unknown_facet", "missing_facet", "unknown_block", "fake_quote", "contradiction"])
async def test_review_rejects_wrong_id_source_and_status(mutation):
    """参数为评审违约方式；未知关联、伪造原文或缺口与成功矛盾均显式失败。"""
    payload, raw = case()
    draft = review_draft().model_dump(mode="json")
    point = draft["points"][0]
    if mutation == "unknown_facet":
        point["facet_id"] = "other-question"
    elif mutation == "missing_facet":
        draft["points"] = []
    elif mutation == "unknown_block":
        point["block_ids"] = ["foreign-answer"]
    elif mutation == "fake_quote":
        point["sources"][0]["quote"] = "论文没有进行任何实验"
    else:
        point["verdict"] = "missing"
    reviewer = SimpleNamespace(review=AsyncMock(return_value=draft))
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw), reviewer)
    assert reviewed.delivery_report.status == "failed"
    assert reviewed.delivery_report.semantic_validation == "failed"
    assert reviewed.delivery_report.content_review.error
    assert raw["blocks"][0]["text"] in reviewed.answer


@pytest.mark.asyncio
async def test_unfounded_absence_statement_in_gap_block_is_reviewed():
    """无参数；未读不能证明论文未做实验，gap块的额外断言仍在实际全文核对中。"""
    payload, raw = case()
    false_statement = "论文没有单独实验。"
    raw["blocks"].append({"block_id": "gap", "text": false_statement, "content_kind": "gap",
                           "facet_ids": ["f1"], "section_ids": [], "claim_ids": [], "evidence_ids": []})
    issue = dict(kind="unsupported_statement", facet_ids=["f1"], block_ids=["gap"], answer_quote=false_statement,
                 explanation="原文只给测量结果，没有否定其他实验", suggested_action="删除无依据否定，保留已证实结果", sources=[])
    reviewer = SimpleNamespace(review=AsyncMock(return_value=review_draft(recommendation="revise_answer", issues=[issue])))
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw), reviewer)
    assert false_statement in reviewer.review.call_args.args[0].answer
    assert reviewed.delivery_report.status == "partial"
    assert review_requires_continuation(reviewed.delivery_report.model_dump(mode="json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("reviewer", [None, SimpleNamespace(review=AsyncMock(side_effect=RuntimeError("review unavailable")))])
async def test_missing_or_failed_review_is_not_assessed_as_success(reviewer):
    """参数为缺失或失败的核对端口；不把引用通过当作语义成功，不吞错。"""
    payload, raw = case()
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw), reviewer)
    assert reviewed.delivery_report.status == "failed"
    assert reviewed.delivery_report.content_review.error


@pytest.mark.asyncio
async def test_review_identity_cannot_be_reused_for_different_body_or_attempt():
    """无参数；保存阶段拒绝另一份正文或另次生成借用原核对结论。"""
    payload, raw = case()
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw),
        SimpleNamespace(review=AsyncMock(return_value=review_draft())))
    report = reviewed.delivery_report.model_dump(mode="json")
    with pytest.raises(ValueError, match="正文"):
        validate_review_delivery("改写内容" + reviewed.answer, report)
    with pytest.raises(ValueError, match="身份"):
        validate_review_delivery(reviewed.answer, {**report, "attempt_id": "another"})
    with pytest.raises(ValueError, match="状态"):
        validate_review_delivery(reviewed.answer, {**report, "semantic_validation": "not_assessed"})
    with pytest.raises(ValueError, match="完整正文"):
        validate_review_delivery(reviewed.answer + "\n论文没有其他实验。", report)


@pytest.mark.asyncio
@pytest.mark.parametrize("historical", [False, True])
async def test_final_only_retry_reviews_new_incremental_body_without_new_research(historical):
    """参数为当前或历史重试；新稿内容核对不触发取证，历史未知支持不重评。"""
    payload, raw = case(None if historical else "internal")
    plan = payload.plan.model_copy(update={"metadata": payload.plan.metadata.model_copy(update={"execution_mode": "incremental"})})
    provider = SimpleNamespace(generate=AsyncMock(return_value=raw))
    reviewer = SimpleNamespace(review=AsyncMock(return_value=review_draft(verdict="missing", recommendation="continue_research")))
    result = await synthesize_planner_answer_async(plan, payload.unit_results, project_paper_count=1,
        evidence_records=payload.evidence_records, final_provider=provider, answer_reviewer=reviewer,
        final_entrypoint="historical_retry" if historical else "final_retry",
        historical_input_context={"legacy_version": "research-planner-v3"} if historical else None)
    assert reviewer.review.await_count == (0 if historical else 1)
    assert result["final_delivery_report"]["status"] == "partial"
    assert "planner_loop_decision" not in result


def test_review_input_preserves_source_text_and_ignores_unread_as_proof(monkeypatch):
    """参数为替换器；核对有独立真实投影，没展示的正文不能用作核对依据。"""
    from app.services.internal_research import research_answer_review as module
    payload, raw = case()
    source = payload.evidence_records[0]
    limited = replace(payload, exposure={**payload.exposure, "evidence": []})
    monkeypatch.setattr(module, "configured_consumer_budget", lambda: 0)
    review_input = prepare_review_input(limited, validate_planner_final_answer(payload, raw))
    assert review_input.facts["sources"] == []
    assert source["excerpt"] not in str(review_input.facts["sources"])


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,expected", [("missing", "passed"), ("unsupported", "failed")])
async def test_optional_requirement_does_not_excuse_unsupported_assertions(verdict, expected):
    """参数为可选要点状态；可选内容可缺少，但写出的无支持断言仍不能放行。"""
    payload, raw = case()
    coverage = payload.plan.coverage_review
    optional = coverage.inferred_facets[0].model_copy(update={"required": False})
    plan = payload.plan.model_copy(update={"coverage_review": coverage.model_copy(update={"inferred_facets": [optional]})})
    payload = replace(payload, plan=plan)
    reviewer = SimpleNamespace(review=AsyncMock(return_value=review_draft(verdict=verdict)))
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw), reviewer)
    assert reviewed.delivery_report.semantic_validation == expected


@pytest.mark.asyncio
async def test_explicit_future_work_allows_source_backed_partial_without_endless_research():
    """无参数；论文明确未做的组合结果可诚实部分结束，不强制继续查找不存在的本轮结果。"""
    statement = "The combined experiment remains future work."
    payload, raw = case(text=statement)
    record = {**payload.evidence_records[0], "excerpt": statement}
    payload = prepare_final_input(replace(payload, evidence_records=[record], exposure={}))
    draft = review_draft(verdict="missing", recommendation="partial").model_dump(mode="json")
    source = {"evidence_id": "e1", "quote": statement}
    draft["points"][0]["sources"] = [source]
    draft["partial_basis"] = {"kind": "source_statement", "explanation": "原文明示该组合实验留待后续",
                              "sources": [source], "read_unit_ids": []}
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw),
        SimpleNamespace(review=AsyncMock(return_value=draft)))
    report = reviewed.delivery_report.model_dump(mode="json")
    assert report["status"] == "partial" and report["semantic_validation"] == "issues_found"
    assert not review_requires_continuation(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["source_statement", "read_outcome", "current_scope"])
async def test_partial_stop_requires_actual_basis(kind):
    """参数为停止理由；没有真实引用或读取关联时，部分结束不能作为任意退出开关。"""
    payload, raw = case()
    draft = review_draft(verdict="missing", recommendation="partial").model_dump(mode="json")
    draft["partial_basis"] = {"kind": kind, "explanation": "没有依据的结束理由", "sources": [], "read_unit_ids": []}
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw),
        SimpleNamespace(review=AsyncMock(return_value=draft)))
    assert reviewed.delivery_report.semantic_validation == "failed"


@pytest.mark.asyncio
async def test_actual_read_failure_supports_current_inability_without_paper_absence():
    """无参数；真实失败读取可解释当前拿不到答案，核对不把它改成论文没有实验。"""
    payload, raw = case()
    from app.models.knowledge_access_contracts import KnowledgeReadReport
    report = KnowledgeReadReport(status="unavailable", operation="sections",
        requested=("Results",), remaining=({"reason": "权限变更，读取失败"},))
    result = payload.unit_results[0].model_copy(update={"metadata": {"read_reports": [report.to_dict()]}})
    payload = replace(payload, unit_results=[result])
    draft = review_draft(verdict="missing", recommendation="partial").model_dump(mode="json")
    draft["partial_basis"] = {"kind": "read_outcome", "explanation": "本轮权限变化后无法取得剩余材料",
                              "sources": [], "read_unit_ids": [result.unit_id]}
    reviewed = await review_final_answer(payload, validate_planner_final_answer(payload, raw),
        SimpleNamespace(review=AsyncMock(return_value=draft)))
    assert reviewed.delivery_report.semantic_validation == "issues_found"
    assert not review_requires_continuation(reviewed.delivery_report.model_dump(mode="json"))


def test_review_sends_body_once_and_offsets_identify_exact_unchanged_text():
    """无参数；核对只发送一份完整正文，表格、代码及Unicode块通过同稿坐标准确定位。"""
    from app.services.internal_research.research_answer_review import _review_prompt

    text = "## Evidence\n\n|条件|值|\n|---|---|\n|😀A|91%|\n\n```text\n原始代码\n```"
    payload, raw = case(text=text)
    final = validate_planner_final_answer(payload, raw)
    review = prepare_review_input(payload, final)
    assert "text" not in review.facts["blocks"][0]
    block = review.facts["blocks"][0]
    assert review.answer[block["start"]:block["end"]] == text
    assert _review_prompt(review).count(text) == 1
    assert review.exposure["evidence"]


def test_review_retains_unselected_acquired_source_text():
    """无参数；未选入终稿的已取得原文仍送核对，生成投影较窄不导致遗漏不利材料。"""
    payload, raw = case()
    other = {**payload.evidence_records[0], "evidence_id": "unused", "excerpt": "独立未消费原文", "chunk_id": "unused"}
    payload = replace(payload, evidence_records=[*payload.evidence_records, other])
    final = validate_planner_final_answer(payload, raw)
    review = prepare_review_input(payload, final)
    assert any(item["evidence_id"] == "unused" and item["text"] == "独立未消费原文" for item in review.facts["sources"])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["protocol", "http", "other"])
async def test_review_failure_reuses_provider_summary_without_changing_other_errors(kind):
    """参数为协议、HTTP或程序错误；真实核对入口保留原正文与失败，既有其他错误表达不变。"""
    from app.services.llm_service import ProviderHTTPError, ProviderProtocolError

    payload, raw = case()
    final = validate_planner_final_answer(payload, raw)
    errors = {
        "protocol": ProviderProtocolError("invalid_sse_json", line_number=2, data_text="脱敏坏片段"),
        "http": ProviderHTTPError("technical details", response=None, user_summary="模型服务返回404，具体原因未明确"),
        "other": RuntimeError("existing error"),
    }
    error = errors[kind]
    expected = error.user_summary if isinstance(error, (ProviderHTTPError, ProviderProtocolError)) else f"RuntimeError: {error}"
    reviewer = SimpleNamespace(review=AsyncMock(side_effect=error))
    reviewed = await review_final_answer(payload, final, reviewer)
    assert reviewer.review.await_count == 1
    assert reviewed.delivery_report.status == "failed" and reviewed.delivery_report.content_review.status == "failed"
    assert reviewed.delivery_report.content_review.error == expected
    assert reviewed.answer == final.answer and reviewed.delivery_report.citation_bindings == final.delivery_report.citation_bindings
