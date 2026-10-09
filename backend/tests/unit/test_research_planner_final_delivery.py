"""【研究交付】【最终入口】O4 正文和交付合同回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import socket

import pytest

from app.models.research_planner_contracts import ResearchPlan, ResearchUnitResult
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer_async


@pytest.fixture(autouse=True)
def _isolate_network(monkeypatch):
    """参数为替身工具；禁止测试连接数据库、HTTP 或检索网络，返回无值。"""
    import httpx
    import requests
    from app.services import database

    def reject(*args, **kwargs):
        """参数为意外网络调用；直接失败以暴露遗漏的外部替身。"""
        raise AssertionError("offline final-delivery test attempted network IO")

    monkeypatch.setattr(httpx.AsyncClient, "send", reject)
    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(requests.Session, "request", reject)
    monkeypatch.setattr(database, "get_pool", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def _material_plan():
    """无参数；返回只要求论文元数据的实际研究计划。"""
    return ResearchPlan.model_validate({
        "question": "整理论文信息", "plan_type": "single_unit", "units": [{
            "unit_id": "u1", "goal": "整理论文信息", "research_action": "读取元数据",
            "covered_facet_ids": ["f1"], "output_role": "evidence",
            "evidence_policy": {"source_type": "local_metadata", "target_paper_ids": ["p1"]},
            "claim_policy": {"allowed_claim_types": []},
            "expected_output": {"output_goal": "论文信息", "deliverable_kind": "materials"},
        }], "coverage_review": {"explicit_facets": [{
            "facet_id": "f1", "label": "论文信息", "source": "explicit", "required": True,
            "deliverable_kind": "materials", "required_paper_ids": ["p1"], "min_material_count": 1,
        }]},
    })


class _Provider:
    """【研究交付】【隔离模型】只返回预置原始载体。"""

    def __init__(self, blocks):
        """参数为模型正文块；初始化调用记录，无返回值。"""
        self.blocks = blocks
        self.calls = []

    async def generate(self, payload):
        """参数为正式最终输入；返回预置 v3 原始 JSON 对象。"""
        self.calls.append(payload)
        return {"contract_version": "research-planner-final-v3", "blocks": self.blocks,
                "status": "answered", "unresolved_items": []}


def _material_block(**updates):
    """参数为字段覆盖；返回关联实际元数据的可读正文块。"""
    return {"block_id": "b1", "text": "论文元数据已整理。", "content_kind": "materials",
            "facet_ids": ["f1"], "section_ids": [], "claim_ids": [], "evidence_ids": ["meta1"], **updates}


def _run(provider):
    """参数为隔离模型；运行正式综合入口，返回实际交付载荷。"""
    return asyncio.run(synthesize_planner_answer_async(
        _material_plan(), [ResearchUnitResult(unit_id="u1", status="evidence_found", evidence_ids=["meta1"])],
        project_paper_count=1, final_provider=provider, evidence_records=[{
            "evidence_id": "meta1", "paper_id": "p1", "layer": "metadata", "source": "paper_metadata",
            "usable_as_citation": False, "summary": "论文元数据已整理。",
        }],
    ))


def test_materials_without_claims_call_provider_and_complete():
    """无参数；材料要求必须调用最终组织并按实际产物完成。"""
    provider = _Provider([_material_block()])
    result = _run(provider)
    assert len(provider.calls) == 1
    assert result["final_delivery_report"]["status"] == "complete"
    assert result["answer"] == "论文元数据已整理。"
    assert result["final_citation_bindings"] == []


def test_local_invalid_relation_preserves_body_and_marks_partial():
    """无参数；局部未知来源不得删正文，也不得计入有效交付。"""
    result = _run(_Provider([_material_block(), _material_block(
        block_id="b2", text="保留这段错误正文。", evidence_ids=["unknown"],
    )]))
    assert "保留这段错误正文。" in result["answer"]
    assert result["final_delivery_report"]["status"] == "partial"


def test_generated_empty_delivery_is_failed():
    """无参数；已有材料但模型空交付属于生成失败。"""
    result = _run(_Provider([]))
    assert result["final_delivery_report"]["status"] == "failed"
    assert result["final_delivery_report"]["reasons"][0]["code"] == "empty_delivery"


def test_v4_material_draft_has_server_owned_identity_and_rejects_extra_support():
    """无参数；新版材料分支由程序编号，模型不能附带另一份主张关系。"""
    from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
    from app.services.internal_research.research_planner_final_contracts import PlannerFinalAnswerInput
    from app.services.internal_research.research_planner_final_provider import prepare_final_input

    payload = prepare_final_input(PlannerFinalAnswerInput(
        question="整理论文信息", plan=_material_plan(), attempt_id="draft-a",
        unit_results=[ResearchUnitResult(unit_id="u1", status="evidence_found", evidence_ids=["meta1"])],
        evidence_records=[{"evidence_id": "meta1", "paper_id": "p1", "layer": "metadata",
                           "source": "paper_metadata", "usable_as_citation": False, "summary": "论文信息"}],
    ))
    block = {"product_kind": "materials", "text": "论文信息", "facet_ids": ["f1"],
             "section_ids": [], "evidence_ids": ["meta1"]}
    raw = {"contract_version": "research-planner-final-draft-v4", "blocks": [block]}
    final = validate_planner_final_answer(payload, raw)
    assert final.delivery_report.status == "complete"
    assert final.delivery_report.delivery_policy_version == "research-delivery-v5"
    assert final.blocks[0].block_id == "draft-a:0"
    rejected = validate_planner_final_answer(payload, {**raw, "blocks": [{**block, "claim_ids": []}]})
    assert rejected.answer == "论文信息"
    assert not rejected.blocks[0].valid


def _revision_case():
    """无参数；返回两块有效材料和一块错误正文的候选，供真实补丁反例共用。"""
    from dataclasses import replace
    from test_research_planner_delivery_policy import case
    from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
    from app.services.internal_research.research_planner_final_revision import make_candidate

    payload, raw = case()
    payload = replace(payload, attempt_id="base", delivery_policy_version="research-delivery-v5")
    raw["blocks"] += [{**raw["blocks"][0], "block_id": "b2", "text": "保留的第二段 😀\n"},
                      {**raw["blocks"][0], "block_id": "bad", "text": "待修正文", "evidence_ids": ["unknown"]}]
    final = validate_planner_final_answer(payload, raw)
    return replace(payload, attempt_id="revised"), make_candidate(payload, final, raw)


def _patch(candidate, operations):
    """参数为原候选和操作；返回绑定同一基稿及研究版本的离线补丁。"""
    return {"contract_version": "research-planner-final-patch-v1", "base_attempt_id": candidate["attempt_id"],
            "base_digest": candidate["base_digest"], "research_snapshot_digest": candidate["research_snapshot_digest"],
            "operations": operations, "status": "answered", "unresolved_items": []}


@pytest.mark.parametrize("replacement", [[], ["修复正文"], ["拆分一", "拆分二"]])
def test_atomic_patch_replaces_splits_deletes_and_retains_untouched_text(replacement):
    """参数为替换正文；拆分/删除仅影响指定块，未改身份、文字和关系完全保留。"""
    from copy import deepcopy
    from app.services.internal_research.research_planner_final_revision import apply_final_patch

    payload, candidate = _revision_case()
    original = deepcopy(candidate)
    blocks = [{"product_kind": "materials", "text": text, "facet_ids": ["f1"], "section_ids": [],
               "evidence_ids": ["e1"]} for text in replacement]
    output = apply_final_patch(payload, candidate, _patch(candidate, [
        {"operation": "replace_blocks", "block_ids": ["bad"], "blocks": blocks}]))
    assert candidate == original
    assert [block["text"] for block in output["blocks"]] == [block["text"] for block in original["blocks"][:2]] + replacement
    assert [block["block_id"] for block in output["blocks"][:2]] == ["b1", "b2"]
    assert [block["block_id"] for block in output["blocks"][2:]] == [f"revised:{i}" for i in range(len(replacement))]


@pytest.mark.parametrize("mutation", ["stale", "evidence", "duplicate", "unknown", "invalid_new", "overlap"])
def test_invalid_patch_never_partially_changes_base(mutation):
    """参数为冲突类型；旧版本、越界目标或新块不合格时整份原稿不变。"""
    from copy import deepcopy
    from dataclasses import replace
    from app.services.internal_research.research_planner_final_revision import apply_final_patch

    payload, candidate = _revision_case()
    original = deepcopy(candidate)
    patch = _patch(candidate, [{"operation": "replace_blocks", "block_ids": ["bad"], "blocks": []}])
    if mutation == "stale":
        patch["base_attempt_id"] = "stale"
    elif mutation == "evidence":
        payload = replace(payload, evidence_records=[])
    elif mutation == "duplicate":
        patch["operations"][0]["block_ids"] = ["bad", "bad"]
    elif mutation == "unknown":
        patch["operations"][0]["block_ids"] = ["b1"]
    elif mutation == "invalid_new":
        patch["operations"][0]["blocks"] = [{"text": "可读但非法"}]
    else:
        patch["operations"].append(deepcopy(patch["operations"][0]))
    with pytest.raises(ValueError):
        apply_final_patch(payload, candidate, patch)
    assert candidate == original


def test_supported_partial_may_retain_missing_content_issues():
    """无参数；有依据的缺失结果仍可诚实部分结束，缺失问题不要求机械清空。"""
    from app.services.internal_research.research_planner_final_revision import delivery_next_action

    _, candidate = _revision_case()
    report = {**candidate["report"], "validation_errors": [], "section_results": [],
              "content_review": {"status": "issues_found", "assessment": {
                  "recommendation": "partial", "issues": [{"kind": "missing_content"}],
                  "points": [{"verdict": "missing"}], "partial_basis": {"kind": "source_statement"}}}}
    assert delivery_next_action({**candidate, "report": report}) == "continue"


def test_missing_presentation_allows_insertion_and_rejects_deleted_anchor():
    """无参数；遗漏章节允许插入，新稿保留所有旧块，插入锚点同时删除则整份补丁拒绝。"""
    from copy import deepcopy
    from app.services.internal_research.research_planner_final_revision import apply_final_patch

    payload, candidate = _revision_case()
    candidate["report"]["section_results"] = [{"required": True, "status": "missing", "requirement_id": "s1",
        "block_ids": [], "reasons": [{"message": "缺少展示标题"}]}]
    inserted = {"product_kind": "structure", "text": "## 已知结论", "facet_ids": [], "section_ids": []}
    operations = [{"operation": "insert_after", "after_block_id": None, "blocks": [inserted]},
                  {"operation": "replace_blocks", "block_ids": ["bad"], "blocks": []}]
    result = apply_final_patch(payload, candidate, _patch(candidate, operations))
    assert [block["block_id"] for block in result["blocks"]] == ["revised:0", "b1", "b2"]
    conflicted = deepcopy(operations)
    conflicted[0]["after_block_id"] = "bad"
    with pytest.raises(ValueError, match="锚点"):
        apply_final_patch(payload, candidate, _patch(candidate, conflicted))


def test_patch_expansion_requires_relationship_to_the_named_issue():
    """无参数；相关同要点正文可显式扩大目标，错误的问题身份不能授权改稿。"""
    from app.services.internal_research.research_planner_final_revision import apply_final_patch

    payload, candidate = _revision_case()
    patch = _patch(candidate, [{"operation": "replace_blocks", "block_ids": ["b2", "bad"], "blocks": []}])
    with pytest.raises(ValueError, match="未授权"):
        apply_final_patch(payload, candidate, patch)
    patch["expansions"] = [{"block_id": "b2", "issue_index": 0, "reason": "两段共同解释同一研究要点"}]
    assert len(apply_final_patch(payload, candidate, patch)["blocks"]) == 1
    patch["expansions"][0]["issue_index"] = 999
    with pytest.raises(ValueError, match="关联"):
        apply_final_patch(payload, candidate, patch)


@pytest.mark.asyncio
async def test_whole_scope_is_explicit_and_final_retry_does_not_auto_patch():
    """无参数；整体重组来自明确核对范围；最终重试仅生成一次，不启动局部或研究回路。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.internal_research.research_planner_synthesis import _generate_candidate

    payload, candidate = _revision_case()
    candidate["report"]["content_review"] = {"assessment": {"revision_scope": "whole", "issues": [],
        "points": [], "reason": "论证主线必须重新组织"}}
    provider = SimpleNamespace(generate=AsyncMock(return_value={"whole": True}), revise=AsyncMock())
    assert await _generate_candidate(payload, provider, candidate) == {"whole": True}
    provider.revise.assert_not_called()
    provider.generate.return_value = {"contract_version": "research-planner-final-draft-v4", "blocks": [
        {"product_kind": "materials", "text": "不合法来源仍保留", "facet_ids": ["f1"], "section_ids": [],
         "evidence_ids": ["unknown"]}]}
    result = await synthesize_planner_answer_async(payload.plan, payload.unit_results, project_paper_count=1,
        evidence_records=payload.evidence_records, final_provider=provider, final_entrypoint="final_retry",
        repair_context={"planner_final_candidate": candidate})
    assert "不合法来源仍保留" in result["answer"]
    assert result["final_delivery_report"]["semantic_validation"] == "not_assessed"
    assert "planner_loop_decision" not in result
    provider.revise.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_patch_propagates_without_mutating_candidate():
    """无参数；取消信号不成为失败模型响应，待修原稿完整保留且不继续核对。"""
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    payload, candidate = _revision_case()
    original = deepcopy(candidate)
    provider = SimpleNamespace(revise=AsyncMock(side_effect=asyncio.CancelledError), generate=AsyncMock())
    reviewer = SimpleNamespace(review=AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        await synthesize_planner_answer_async(payload.plan, payload.unit_results, project_paper_count=1,
            evidence_records=payload.evidence_records, final_provider=provider, answer_reviewer=reviewer,
            repair_context={"planner_final_candidate": candidate})
    assert candidate == original
    reviewer.review.assert_not_called()


def test_v4_claims_derive_identity_and_citations_and_reject_mixed_types():
    """无参数；同源主张自动派生身份和引用，混合事实/推断必须拆块，未知主张不被猜测。"""
    from dataclasses import replace
    from test_research_planner_delivery_policy import _two_facet_claim_case
    from research_planner_test_utils import result_with_exposure
    from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
    from app.services.internal_research.research_planner_final_provider import prepare_final_input

    payload, _ = _two_facet_claim_case()
    claims = payload.unit_results[0].claims
    unit = payload.plan.units[0]
    policy = unit.claim_policy.model_copy(update={"allowed_claim_types": ["paper_supported", "evidence_grounded_inference"]})
    unit = unit.model_copy(update={"claim_policy": policy})
    claims = [claims[0], claims[1].model_copy(update={"claim_type": "evidence_grounded_inference",
                                                    "reasoning_summary": "依据可见结果解释适用条件"})]
    plan = payload.plan.model_copy(update={"units": [unit]})
    payload = prepare_final_input(replace(payload, plan=plan, exposure={}, unit_results=[result_with_exposure(unit, payload.evidence_records, claims)]))
    block = {"product_kind": "claims", "text": "|来源|判断|\n|---|---|\n|论文|91%|", "facet_ids": ["f1"],
             "section_ids": [], "claim_ids": ["u1.c1"]}
    raw = {"contract_version": "research-planner-final-draft-v4", "blocks": [block]}
    final = validate_planner_final_answer(payload, raw)
    assert final.blocks[0].valid and final.blocks[0].content_kind == "fact"
    assert final.blocks[0].evidence_ids == ["e1"]
    assert final.delivery_report.citation_bindings[0].evidence_id == "e1"
    mixed = {**block, "facet_ids": ["f1", "f2"], "claim_ids": ["u1.c1", "u1.c2"]}
    final = validate_planner_final_answer(payload, {**raw, "blocks": [mixed]})
    assert not final.blocks[0].valid
    assert "invalid_content_identity" in {error.code for error in final.errors}
    split = validate_planner_final_answer(payload, {**raw, "blocks": [block, {**block, "facet_ids": ["f2"], "claim_ids": ["u1.c2"]}]})
    assert [item.content_kind for item in split.blocks] == ["fact", "inference"]
    assert all(item.valid for item in split.blocks)


def test_v5_section_presentation_does_not_claim_missing_content_is_delivered():
    """无参数；真实标题和来源正文可以完成展示，模型声明缺失结果仍保持内容missing/partial。"""
    from test_research_planner_delivery_policy import case
    from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer

    payload, _ = case("exact_heading", text="## Evidence\n\n当前仅取得条件A的结果")
    raw = {"contract_version": "research-planner-final-draft-v4", "blocks": [
        {"product_kind": "materials", "text": "## Evidence\n\n当前仅取得条件A的结果", "facet_ids": ["f1"],
         "section_ids": ["s1"], "evidence_ids": ["e1"]}], "status": "incomplete",
        "unresolved_items": [{"facet_id": "f1", "section_id": "s1", "reason": "所问条件B的测量结果尚未提供"}]}
    report = validate_planner_final_answer(payload, raw).delivery_report
    assert report.section_results[0].status == "delivered"
    assert report.requirement_results[0].status == "missing" and report.status == "partial"


@pytest.mark.parametrize("unreadable", [
    {"product_kind": "structure", "text": " \n ", "facet_ids": [], "section_ids": []},
    {"product_kind": "structure", "text": None, "facet_ids": [], "section_ids": []}, None,
])
def test_unreadable_block_is_repairable_by_its_original_position(unreadable):
    """参数为不可读原块；混合稿按原位置定位并可删除，保留可读业务正文及原始错误。"""
    from dataclasses import replace
    from test_research_planner_delivery_policy import case
    from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
    from app.services.internal_research.research_planner_final_revision import (
        apply_final_patch, candidate_revision_request, make_candidate,
    )

    payload, _ = case()
    raw = {"contract_version": "research-planner-final-draft-v4", "blocks": [
        {"product_kind": "materials", "text": "保留的论文结果", "facet_ids": ["f1"],
         "section_ids": [], "evidence_ids": ["e1"]},
        unreadable,
    ]}
    final = validate_planner_final_answer(payload, raw)
    assert final.errors[0].block_index == 1 and final.errors[0].block_id == ""
    candidate = make_candidate(payload, final, raw)
    target = candidate["blocks"][1]["block_id"]
    assert candidate_revision_request(candidate)["allowed_block_ids"] == [target]
    revised_payload = replace(payload, attempt_id="repaired-empty", delivery_policy_version="research-delivery-v5")
    corrected = apply_final_patch(revised_payload, candidate, _patch(candidate, [
        {"operation": "replace_blocks", "block_ids": [target], "blocks": []}]))
    result = validate_planner_final_answer(revised_payload, corrected)
    assert not result.errors
    assert result.answer == final.answer == "保留的论文结果\n\n来源：[E1]"
    assert len(candidate["blocks"]) == 2


@pytest.mark.asyncio
async def test_all_unreadable_blocks_fail_without_review_or_repair():
    """无参数；全稿不可读时保留原始错误并明确终止，不伪造可修正文或调用核对。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from test_research_planner_delivery_policy import case

    payload, _ = case()
    plan = payload.plan.model_copy(update={"metadata": payload.plan.metadata.model_copy(update={"execution_mode": "incremental"})})
    raw = {"contract_version": "research-planner-final-draft-v4", "blocks": [None,
        {"product_kind": "structure", "text": " \n ", "facet_ids": [], "section_ids": []}]}
    provider = SimpleNamespace(generate=AsyncMock(return_value=raw), revise=AsyncMock())
    reviewer = SimpleNamespace(review=AsyncMock())
    result = await synthesize_planner_answer_async(plan, payload.unit_results, project_paper_count=1,
        evidence_records=payload.evidence_records, final_provider=provider, answer_reviewer=reviewer)
    assert result["planner_loop_decision"] == "terminate_fail"
    assert result["final_delivery_report"]["semantic_validation"] == "not_assessed"
    assert result["final_delivery_report"]["status"] == "failed"
    assert result["planner_final_candidate"]["raw_output"] == raw
    assert not any(block["text"] for block in result["planner_final_candidate"]["blocks"])
    provider.revise.assert_not_called()
    reviewer.review.assert_not_called()


@pytest.mark.asyncio
async def test_missing_section_can_reassociate_related_body_without_rewriting_unrelated_blocks():
    """无参数；漏填章节关系可按稳定要点扩展修复，其他正文不被授权且原候选保持不变。"""
    from copy import deepcopy
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from test_research_planner_delivery_policy import case
    from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
    from app.services.internal_research.research_planner_final_revision import (
        apply_final_patch, candidate_revision_request,
    )

    payload, _ = case("exact_heading", section_ids=[])
    plan = payload.plan.model_copy(update={"metadata": payload.plan.metadata.model_copy(update={"execution_mode": "incremental"})})
    payload = replace(payload, plan=plan)
    raw = {"contract_version": "research-planner-final-draft-v4", "blocks": [
        {"product_kind": "materials", "text": "已取得的测量结果", "facet_ids": ["f1"],
         "section_ids": [], "evidence_ids": ["e1"]},
        {"product_kind": "structure", "text": "其他说明", "facet_ids": [], "section_ids": []},
    ]}
    reviewer = SimpleNamespace(review=AsyncMock())
    result = await synthesize_planner_answer_async(plan, payload.unit_results, project_paper_count=1,
        evidence_records=payload.evidence_records, final_provider=SimpleNamespace(generate=AsyncMock(return_value=raw)),
        answer_reviewer=reviewer)
    assert result["planner_loop_decision"] == "revise_answer"
    reviewer.review.assert_not_called()
    candidate = result["planner_final_candidate"]
    original = deepcopy(candidate)
    request = candidate_revision_request(candidate)
    related, unrelated = [block["block_id"] for block in candidate["blocks"]]
    assert request["related_block_ids"][0] == [related]
    revision = replace(payload, attempt_id="section-repair", delivery_policy_version="research-delivery-v5")
    corrected = {**raw["blocks"][0], "text": "## Evidence\n\n已取得的测量结果", "section_ids": ["s1"]}
    patch = _patch(candidate, [{"operation": "replace_blocks", "block_ids": [related], "blocks": [corrected]}])
    patch["expansions"] = [{"block_id": related, "issue_index": 0, "reason": "本段实际回答该章节要求的测量要点"}]
    output = apply_final_patch(revision, candidate, patch)
    assert validate_planner_final_answer(revision, output).delivery_report.status == "complete"
    assert len(output["blocks"]) == 2 and output["blocks"][1]["block_id"] == unrelated
    invalid = deepcopy(patch)
    invalid["expansions"][0]["block_id"] = unrelated
    with pytest.raises(ValueError, match="关联"):
        apply_final_patch(revision, candidate, invalid)
    corrupted = deepcopy(candidate)
    corrupted["section_facets"]["s1"] = []
    with pytest.raises(ValueError, match="稳定要求"):
        apply_final_patch(revision, corrupted, patch)
    assert candidate == original
