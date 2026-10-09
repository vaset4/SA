"""【研究交付】【展示规则】02k 内容、用户格式及历史未知回归。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
import json
import socket

import pytest

from app.models.research_planner_contracts import ResearchClaim, ResearchPlan, ResearchUnitResult
from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
from app.services.internal_research.research_planner_final_contracts import (
    FinalDeliveryReport, PlannerFinalAnswerInput,
)
from app.services.internal_research.research_planner_final_provider import _final_answer_prompt, prepare_final_input
from research_planner_test_utils import result_with_exposure, supporting_spans


@pytest.fixture(autouse=True)
def no_external_io(monkeypatch):
    """参数为替换工具；禁止测试发起真实模型、HTTP、检索或数据库连接。"""
    import httpx
    import requests
    from app.services import database

    monkeypatch.setattr(httpx.AsyncClient, "send", reject_io)
    monkeypatch.setattr(httpx.Client, "send", reject_io)
    monkeypatch.setattr(requests.Session, "request", reject_io)
    monkeypatch.setattr(database, "get_pool", reject_io)
    monkeypatch.setattr(socket, "create_connection", reject_io)


def reject_io(*args, **kwargs):
    """参数为意外的外部调用；立即报错以揭示隔离缺口。"""
    raise AssertionError("02k delivery tests forbid external IO")


def case(presentation_kind="internal", *, text="条件A下测量值为91%。", section_ids=None):
    """参数为展示语义、正文及归组；返回真实材料输入和最小构造的最终块。"""
    section = {"section_id": "s1", "title": "Evidence", "facet_ids": ["f1"],
               "required": True, "source": "inferred", "presentation_kind": presentation_kind,
               "user_text_span": "" if presentation_kind in {None, "internal"} else "Evidence"}
    plan = ResearchPlan.model_validate({
        "contract_version": "research-planner-v3" if presentation_kind is None else "research-planner-v4",
        "question": "请整理测量资料，按Evidence分节", "plan_type": "single_unit", "units": [{
            "unit_id": "u1", "goal": "整理测量资料", "research_action": "读取论文正文",
            "covered_facet_ids": ["f1"], "output_role": "evidence",
            "evidence_policy": {"source_type": "local_direct_text", "target_paper_ids": ["p1"]},
            "claim_policy": {"allowed_claim_types": []},
            "expected_output": {"output_goal": "测量资料", "deliverable_kind": "materials"},
        }], "coverage_review": {"inferred_facets": [{
            "facet_id": "f1", "label": "测量条件与数值", "source": "inferred", "required": True,
            "deliverable_kind": "materials", "required_paper_ids": ["p1"], "min_material_count": 1,
        }], "answer_sections": [section]},
    })
    payload = prepare_final_input(PlannerFinalAnswerInput(question=plan.question, plan=plan,
        unit_results=[ResearchUnitResult(unit_id="u1", status="evidence_found", evidence_ids=["e1"])],
        evidence_records=[{"evidence_id": "e1", "paper_id": "p1", "layer": "direct_text",
                           "source": "paper", "usable_as_citation": True, "excerpt": "条件A下测量值为91%。"}]))
    draft = {"contract_version": "research-planner-final-v3", "status": "answered", "blocks": [{
        "block_id": "b1", "text": text, "content_kind": "materials", "facet_ids": ["f1"],
        "section_ids": ["s1"] if section_ids is None else section_ids,
        "claim_ids": [], "evidence_ids": ["e1"],
    }]}
    return payload, draft


@pytest.mark.parametrize("kind", ["structure", "gap"])
@pytest.mark.parametrize("policy", ["research-delivery-v4", "research-delivery-v5"])
def test_presentation_prose_and_content_credit_are_separate(kind, policy):
    """参数为非业务正文类型及政策；新版展示接受同节说明，缺失实验仍missing，旧版保持原判定。"""
    from dataclasses import replace

    payload, raw = case("exact_heading", text="当前仅取得条件A的结果", section_ids=[])
    raw["blocks"].append({"block_id": "navigation", "text": "## Evidence\n\n出处说明：已知结果见论文正文。",
        "content_kind": kind, "facet_ids": ["f1"], "section_ids": ["s1"], "claim_ids": [], "evidence_ids": []})
    raw["unresolved_items"] = [{"facet_id": "f1", "reason": "所问条件B的结果尚未取得"}]
    outcome = validate_planner_final_answer(replace(payload, delivery_policy_version=policy), raw)
    section = outcome.delivery_report.section_results[0]
    assert section.status == ("delivered" if policy.endswith("v5") else "missing")
    if policy.endswith("v4"):
        assert section.reasons[0].code == "user_presentation_not_delivered"
    assert outcome.delivery_report.requirement_results[0].status == "missing"
    assert outcome.delivery_report.status == "partial"
    assert not outcome.blocks[1].business_delivery


@pytest.mark.parametrize("text", ["## Evidence", "## Evidence\n\n   ", "> ## Evidence\n> 出处说明",
                                  "```md\n## Evidence\n出处说明\n```"])
def test_v5_presentation_rejects_heading_only_and_false_headings(text):
    """参数为无正文或伪标题；新版展示仍要求真实标题下的有效正文，不因放开结构块而制造完成。"""
    from dataclasses import replace

    payload, raw = case("exact_heading", text="已有资料", section_ids=[])
    raw["blocks"].append({"block_id": "navigation", "text": text, "content_kind": "structure",
        "facet_ids": [], "section_ids": ["s1"], "claim_ids": [], "evidence_ids": []})
    final = validate_planner_final_answer(replace(payload, delivery_policy_version="research-delivery-v5"), raw)
    assert final.delivery_report.section_results[0].status == "missing"


def test_v5_presentation_does_not_accept_invalid_blocks_or_make_empty_content_complete():
    """无参数；非法出处关系不获展示信用，只有结构说明也不能完成实际研究内容。"""
    from dataclasses import replace

    payload, raw = case("exact_heading")
    payload = replace(payload, delivery_policy_version="research-delivery-v5")
    block = {"block_id": "navigation", "text": "## Evidence\n\n只有出处说明", "content_kind": "structure",
             "facet_ids": [], "section_ids": ["s1"], "claim_ids": [], "evidence_ids": []}
    raw["blocks"] = [block]
    final = validate_planner_final_answer(payload, raw)
    assert final.delivery_report.section_results[0].status == "delivered"
    assert final.delivery_report.requirement_results[0].status == "missing"
    assert final.delivery_report.status == "failed" and not final.delivery_report.valid_business_block_ids
    raw["blocks"] = [{**block, "evidence_ids": ["e1"]}]
    assert validate_planner_final_answer(payload, raw).delivery_report.section_results[0].status == "missing"


@pytest.mark.parametrize("section_ids", [[], ["s1"]])
def test_internal_title_and_grouping_cannot_demote_delivered_content(section_ids):
    """参数为内部归组有无；真实内容已交付时，不因缺少自拟Evidence标题变partial。"""
    outcome = validate_planner_final_answer(*case(section_ids=section_ids))
    assert outcome.delivery_report.status == "complete"
    assert outcome.delivery_report.requirement_results[0].status == "delivered"
    assert outcome.delivery_report.section_results[0].required is False
    assert "Evidence" not in outcome.answer
    assert outcome.delivery_report.semantic_validation == "not_assessed"


@pytest.mark.parametrize("text", ["## 自然的中文标题\n\n条件A下测量值为91%。",
                                  "自然的中文标题\n----------------\n\n条件A下测量值为91%。"])
def test_user_section_accepts_real_section_without_internal_exact_title(text):
    """参数为ATX或Setext章节；用户仅要求分节时，自拟标题不需要逐字匹配。"""
    assert validate_planner_final_answer(*case("user_section", text=text)).delivery_report.status == "complete"


@pytest.mark.parametrize("kind", ["user_section", "exact_heading"])
@pytest.mark.parametrize("text", [
    "条件A下测量值为91%。",
    "Evidence\n\n条件A下测量值为91%。",
    "```markdown\n## Evidence\n```\n\n条件A下测量值为91%。",
    "~~~markdown\nEvidence\n--------\n~~~\n\n条件A下测量值为91%。",
    "    ## Evidence\n\n条件A下测量值为91%。",
    "> ## Evidence\n\n条件A下测量值为91%。",
    "> Evidence\n> --------\n\n条件A下测量值为91%。",
    "条件A下测量值为91%。\n\n## Evidence",
])
def test_fake_or_empty_user_section_remains_missing(kind, text):
    """参数为展示要求和伪标题/空章节；代码、引文及标题前的正文不能满足格式义务。"""
    outcome = validate_planner_final_answer(*case(kind, text=text))
    assert outcome.delivery_report.status == "partial"
    assert outcome.delivery_report.requirement_results[0].status == "delivered"
    assert outcome.delivery_report.section_results[0].status == "missing"


@pytest.mark.parametrize("text", ["## Evidence\n\n😀 条件A下测量值为91%。",
                                  "Evidence\n========\n\n😀 条件A下测量值为91%。",
                                  "## **Evidence**\r\n\r\n😀 条件A下测量值为91%。"])
def test_exact_heading_and_body_preserve_same_attempt_citation_offsets(text):
    """参数为实际标题和正文；用户指定标题满足后，引用仍绑定同一份最终Markdown。"""
    outcome = validate_planner_final_answer(*case("exact_heading", text=text))
    assert outcome.delivery_report.status == "complete"
    assert outcome.answer.startswith(text)
    binding = outcome.delivery_report.citation_bindings[0]
    assert outcome.answer[binding.start:binding.end] == "[E1]"
    assert binding.attempt_id == outcome.delivery_report.attempt_id


def test_exact_heading_requires_the_user_title_even_with_a_different_real_heading():
    """无参数；用户指定标题时，另一个真实章节不能替代该标题。"""
    outcome = validate_planner_final_answer(*case("exact_heading", text="## 自拟标题\n\n条件A下测量值为91%。"))
    assert outcome.delivery_report.status == "partial"


def test_exact_heading_cannot_borrow_body_from_another_section():
    """无参数；指定标题下空白时，同级另一章节的正文不能借来完成格式义务。"""
    outcome = validate_planner_final_answer(*case(
        "exact_heading", text="## Evidence\n\n## 其他内容\n\n条件A下测量值为91%。"))
    assert outcome.delivery_report.status == "partial"


def test_separate_heading_block_covers_following_associated_body():
    """无参数；结构块与后续实质正文共享章节身份时，按最终文本的真实分节验收。"""
    payload, draft = case("exact_heading")
    draft["blocks"].insert(0, {"block_id": "title", "text": "## Evidence", "content_kind": "structure",
                               "facet_ids": [], "section_ids": ["s1"], "claim_ids": [], "evidence_ids": []})
    assert validate_planner_final_answer(payload, draft).delivery_report.status == "complete"


def test_fence_opened_in_an_earlier_block_cannot_create_a_later_real_heading():
    """无参数；按完整Markdown解析，跨块代码围栏中的同名标题仍不构成用户分节。"""
    payload, draft = case("exact_heading", text="## Evidence\n\n条件A下测量值为91%。")
    draft["blocks"].insert(0, {"block_id": "code", "text": "```markdown", "content_kind": "structure",
                               "facet_ids": [], "section_ids": [], "claim_ids": [], "evidence_ids": []})
    assert validate_planner_final_answer(payload, draft).delivery_report.status == "partial"


@pytest.mark.parametrize("kind", ["internal", "user_section"])
def test_inferred_required_content_is_still_required_when_other_body_is_valid(kind):
    """参数为内部或用户分节；已有有效正文不能免除另一项必需的隐含内容。"""
    payload, draft = case(kind, text="## Evidence\n\n条件A下测量值为91%。")
    raw = payload.plan.model_dump(mode="json")
    raw["coverage_review"]["inferred_facets"].append({
        **raw["coverage_review"]["inferred_facets"][0], "facet_id": "f2", "label": "适用条件的解释"})
    raw["units"][0]["covered_facet_ids"].append("f2")
    raw["coverage_review"]["answer_sections"][0]["facet_ids"].append("f2")
    from dataclasses import replace

    outcome = validate_planner_final_answer(replace(payload, plan=ResearchPlan.model_validate(raw)), draft)
    assert outcome.delivery_report.status == "partial"
    missing = next(row for row in outcome.delivery_report.requirement_results if row.requirement_id == "f2")
    assert missing.required and missing.status == "missing"
    if kind == "user_section":
        assert outcome.delivery_report.section_results[0].reasons[0].code == "section_content_not_delivered"


def test_answer_and_patch_prompts_separate_reader_prose_from_structured_controls():
    """无参数；核对真实生成/补丁提示的正文职责独立，原问、证据与结构化支持选择仍完整保留。"""
    from app.services.internal_research.research_planner_final_provider import _revision_prompt
    from app.services.internal_research.research_planner_final_revision import make_candidate

    payload, raw = case("exact_heading")
    candidate = make_candidate(payload, validate_planner_final_answer(payload, raw), raw)
    for prompt in (_final_answer_prompt(payload), _revision_prompt(payload, candidate)):
        prose, controls = prompt.split("Reader-facing answer content (blocks[].text):", 1)[1].split(
            "Structured response controls (not reader-facing prose):", 1)
        assert "paper's section, table or figure" in prose
        assert "missing information in the user's terms" in prose
        assert "claim_ids" not in prose and "evidence_ids" not in prose
        assert "Do not embed citation markers or citation links in text values." in controls
        assert "evidence-anchor" not in prompt
        assert '"claim_ids"' in prompt and '"evidence_ids"' in prompt
        assert payload.question in prompt and payload.context_text in prompt


def test_provider_prompt_uses_model_schemas_and_distinguishes_display_obligations():
    """无参数；正式提示使用合同schema并区分内部标题、任意分节标题及用户原题。"""
    from app.services.internal_research.research_planner_final_provider import _revision_prompt
    from app.services.internal_research.research_planner_final_revision import make_candidate

    payload, raw = case("exact_heading")
    prompt = _final_answer_prompt(payload)
    assert "research-planner-final-draft-v4" in prompt
    assert '"additionalProperties": false' in prompt
    assert "Internal sections are organizational suggestions" in prompt
    assert "For user_section" in prompt and "For exact_heading" in prompt
    assert payload.exposure["prompt_version"] == "research-planner-final-prompt-v12"
    candidate = make_candidate(payload, validate_planner_final_answer(payload, raw), raw)
    identity = f"Prompt version: {payload.exposure['prompt_version']}"
    assert identity in prompt and identity in _revision_prompt(payload, candidate)


def test_final_prompt_does_not_offer_claim_support_as_material_products():
    """无参数；可见引用原文不自动成为材料产物，代码展示仍按真实主张身份校验，无返回值。"""
    # 1. 【研究交付】【产物选择】构造有合格主张和可见支持原文、但没有材料交付要求的输入
    material_payload, draft = case(text="```text\n条件A下测量值为91%。\n```")
    raw_plan = material_payload.plan.model_dump(mode="json")
    raw_plan["coverage_review"]["inferred_facets"][0]["deliverable_kind"] = "claims"
    raw_plan["units"][0]["claim_policy"]["allowed_claim_types"] = ["paper_supported"]
    raw_plan["units"][0]["expected_output"]["deliverable_kind"] = "claims"
    plan = ResearchPlan.model_validate(raw_plan)
    records = material_payload.evidence_records
    claim = ResearchClaim(claim_id="u1.c1", text="条件A下测量值为91%。", claim_type="paper_supported",
        validation_status="validated", supporting_evidence_ids=["e1"], supporting_spans=supporting_spans(records),
        addressed_facet_ids=["f1"])
    result = result_with_exposure(plan.units[0], records, [claim])
    payload = prepare_final_input(PlannerFinalAnswerInput(question=plan.question, plan=plan,
        unit_results=[result], evidence_records=records))
    assert [item["evidence_id"] for item in payload.exposure["evidence"]] == ["e1"]

    # 2. 【研究交付】【产物选择】同一代码块作为事实可交付，误标材料仍保留正文并报告真实错误
    draft["blocks"][0].update(content_kind="fact", claim_ids=["u1.c1"])
    assert validate_planner_final_answer(payload, draft).delivery_report.status == "complete"
    draft["blocks"][0]["content_kind"] = "materials"
    rejected = validate_planner_final_answer(payload, draft)
    assert {error.code for error in rejected.errors} >= {"invalid_content_identity", "unknown_evidence_id"}
    assert draft["blocks"][0]["text"] in rejected.answer

    # 3. 【研究交付】【产物选择】最终提示明确空材料目录和来源身份，不把 Markdown 外观当成产物类型
    prompt = _final_answer_prompt(payload)
    catalog = prompt.split("Allowed material products:\n\n", 1)[1].split("\n\n", 1)[0]
    assert json.loads(catalog) == []
    assert records[0]["excerpt"] in prompt
    assert "Source excerpts (for claim support or listed material products):" in prompt
    assert "Actual material excerpts:" not in prompt
    assert "the server derives content_kind and the required citation closure" in prompt
    assert "Materials must not include claim_ids" in prompt


@pytest.mark.parametrize("budget,visible", [(0, False), (10000, True)])
def test_final_material_catalog_retains_exact_facet_mapping_and_visibility(budget, visible):
    """参数为消费预算和预期可见性；材料目录映射实际允许要求，未展示材料仍不可交付，无返回值。"""
    original, draft = case()
    payload = prepare_final_input(original, evidence_max_chars=budget)
    prompt = _final_answer_prompt(payload)
    catalog = prompt.split("Allowed material products:\n\n", 1)[1].split("\n\n", 1)[0]
    assert json.loads(catalog) == [{"evidence_id": "e1", "eligible_facet_ids": ["f1"], "visible": visible}]
    assert bool(payload.exposure["evidence"]) is visible
    outcome = validate_planner_final_answer(payload, draft)
    if visible:
        assert outcome.delivery_report.status == "complete"
    else:
        assert "material_not_visible" in {error.code for error in outcome.errors}


def _two_facet_claim_case():
    """无参数；返回同单元内分别回应两个要求的合格主张及混合正文，供关联回归复用。"""
    original, draft = case(text="实验结果与作者解释。")
    raw = original.plan.model_dump(mode="json")
    facet = {**raw["coverage_review"]["inferred_facets"][0], "deliverable_kind": "claims"}
    raw["coverage_review"]["inferred_facets"] = [facet, {**facet, "facet_id": "f2", "label": "作者解释"}]
    raw["units"][0].update(covered_facet_ids=["f1", "f2"],
        claim_policy={"allowed_claim_types": ["paper_supported"]},
        expected_output={"output_goal": "测量资料及解释", "deliverable_kind": "claims"})
    plan = ResearchPlan.model_validate(raw)
    records = original.evidence_records
    claims = [ResearchClaim(claim_id=f"u1.c{index}", text="条件A下测量值为91%。", claim_type="paper_supported",
        validation_status="validated", supporting_evidence_ids=["e1"], supporting_spans=supporting_spans(records),
        addressed_facet_ids=[facet_id]) for index, facet_id in enumerate(["f1", "f2"], start=1)]
    payload = prepare_final_input(PlannerFinalAnswerInput(question=plan.question, plan=plan,
        unit_results=[result_with_exposure(plan.units[0], records, claims)], evidence_records=records))
    draft["blocks"][0].update(content_kind="fact", claim_ids=["u1.c1", "u1.c2"])
    return payload, draft


def test_final_claim_catalog_matches_each_claim_not_owner_or_section_facets():
    """无参数；混合块逐条关联主张要求，不能借单元范围或章节归组通过校验，无返回值。"""
    payload, draft = _two_facet_claim_case()
    # 1. 【研究交付】【逐条关联】同单元及同章节不能替第二条主张授予第一项要求
    rejected = validate_planner_final_answer(payload, draft)
    assert [(error.code, error.message) for error in rejected.errors if error.block_id == "b1"] == [
        ("unassociated_product", "主张未回应该正文关联的要求：u1.c2")]
    assert draft["blocks"][0]["text"] in rejected.answer
    assert payload.plan.units[0].covered_facet_ids == ["f1", "f2"]

    # 2. 【研究交付】【逐条关联】模型明确选择两个真实要求后才交付，程序没有自动补全关联
    draft["blocks"][0]["facet_ids"] = ["f1", "f2"]
    assert validate_planner_final_answer(payload, draft).delivery_report.status == "complete"
    assert payload.unit_results[0].claims[1].addressed_facet_ids == ["f2"]

    # 3. 【研究交付】【选择目录】提示目录暴露校验实际使用的每条主张范围和逐条相交规则
    prompt = _final_answer_prompt(payload)
    relationships = json.loads(prompt.split("Verified claims (text, identity and support in one catalog):\n\n", 1)[1].split("\n\n", 1)[0])
    assert {item["claim_id"]: item["addressed_facet_ids"] for item in relationships} == {
        "u1.c1": ["f1"], "u1.c2": ["f2"]}
    assert "For every selected claim" in prompt
    assert "at least one of that claim's addressed_facet_ids" in prompt
    assert "section_ids do not substitute for facet_ids" in prompt


def test_final_prompt_preserves_claim_attribution_and_limits_reasoning_to_support():
    """无参数；保留原始主张及摘要，同时明确作者归因不等于观测、摘要不是额外证据，无返回值。"""
    from dataclasses import replace

    original, _ = _two_facet_claim_case()
    claims = original.unit_results[0].claims
    attributed = "性能低于基线；作者将其归因于短预测窗口向长窗口迁移不佳。"
    unsupported = "原摘要声称论文没有对完整组合进行实验。"
    changed = claims[0].model_copy(update={"text": attributed, "reasoning_summary": unsupported})
    result = result_with_exposure(original.plan.units[0], original.evidence_records, [changed, claims[1]])
    payload = prepare_final_input(replace(original, unit_results=[result], exposure={}, context_text=""))
    prompt = _final_answer_prompt(payload)
    assert attributed in prompt and unsupported in prompt
    assert result.claims[0].reasoning_summary == unsupported
    assert "reasoning_summary explains a claim; it is not an additional source of paper facts" in prompt
    assert "Keep author explanations or causal attributions distinct from measured outcomes" in prompt
    assert "including in table columns" in prompt
    assert "An unobserved experiment is not proof that the paper performed no such experiment" in prompt


@pytest.mark.parametrize("version", ["research-planner-v1", "research-planner-v2", "research-planner-v3"])
def test_final_prompt_keeps_historical_claims_outside_current_facet_catalog(version):
    """参数为历史计划版本；历史主张保留可读性，不要求虚构当前目录关联或授予新版支持，无返回值。"""
    from dataclasses import replace

    original, draft = case(None)
    claim = ResearchClaim(claim_id="u1.c1", text="保存的主张", claim_type="paper_supported",
                          validation_status="validated", supporting_evidence_ids=["e1"])
    result = ResearchUnitResult(unit_id="u1", status="evidence_found", evidence_ids=["e1"], claims=[claim])
    payload = prepare_final_input(replace(original, plan=original.plan.model_copy(update={"contract_version": version}),
        unit_results=[result], exposure={}, context_text="", entrypoint="historical_retry",
        historical_input_context={"legacy_version": version, "readable_claim_ids": [claim.claim_id],
                                  "readable_evidence_ids": ["e1"]}))
    draft["blocks"][0].update(content_kind="history", claim_ids=[claim.claim_id], evidence_ids=[])
    # 1. 【研究交付】【历史边界】已有历史出口保留正文，但不会认定为当前完整交付
    outcome = validate_planner_final_answer(payload, draft)
    assert outcome.delivery_report.status == "partial"
    assert not {"unassociated_product", "unvalidated_claim_id"} & {error.code for error in outcome.errors}
    assert claim.addressed_facet_ids == []
    draft["blocks"][0]["content_kind"] = "fact"
    rejected = validate_planner_final_answer(payload, draft)
    assert "unvalidated_claim_id" in {error.code for error in rejected.errors}
    # 2. 【研究交付】【提示范围】目录没有历史主张，逐条关联指令只能适用于目录中的当前主张
    prompt = _final_answer_prompt(payload)
    relationships = json.loads(prompt.split("Verified claims (text, identity and support in one catalog):\n\n", 1)[1].split("\n\n", 1)[0])
    assert relationships == []
    assert "For every selected claim listed in Verified claims" in prompt
    assert "Authorized historical claims (support not assessed):" in prompt
    assert "Explicitly allowed historical claims outside that catalog remain history" in prompt


@pytest.mark.parametrize("text", ["#", "## ", "## Evidence ##", "Evidence\n===", "```\n```", "> ## Evidence"])
def test_empty_heading_or_structure_cannot_become_business_delivery(text):
    """参数为标准Markdown标题、空代码或引文标题；单有结构不能冒充实质材料正文。"""
    outcome = validate_planner_final_answer(*case(text=text))
    assert outcome.delivery_report.status == "failed"
    assert outcome.delivery_report.valid_business_block_ids == []


@pytest.mark.parametrize("mutation", ["missing_facet", "invalid_source", "only_heading", "model_unresolved"])
def test_internal_organization_never_excuses_real_missing_content(mutation):
    """参数为缺内容、坏引用或模型未完成；内部格式放宽不改变真实交付失败。"""
    payload, draft = case()
    if mutation == "missing_facet":
        draft["blocks"][0]["facet_ids"] = []
    elif mutation == "invalid_source":
        draft["blocks"][0]["evidence_ids"] = ["unknown"]
    elif mutation == "only_heading":
        draft["blocks"][0]["text"] = "## Evidence"
    else:
        draft["unresolved_items"] = [{"facet_id": "f1", "reason": "条件尚未完整解释"}]
    assert validate_planner_final_answer(payload, draft).delivery_report.status != "complete"


def test_old_section_semantics_and_report_policy_are_unknown():
    """无参数；历史未保存展示语义和规则版本时，不默认认作新规则已验收。"""
    outcome = validate_planner_final_answer(*case(None, text="## Evidence\n\n条件A下测量值为91%。"))
    assert outcome.delivery_report.section_results[0].status == "not_assessed"
    assert outcome.delivery_report.status == "partial"
    report = outcome.delivery_report.model_dump(mode="json")
    report.pop("delivery_policy_version", None)
    original = deepcopy(report)
    assert FinalDeliveryReport.model_validate(report).delivery_policy_version is None
    assert report == original


def test_new_report_has_server_owned_delivery_policy_and_unchanged_wire_version():
    """无参数；新生成报告标记新展示规则，原模型载体继续使用final-v3。"""
    outcome = validate_planner_final_answer(*case())
    assert outcome.delivery_report.delivery_policy_version == "research-delivery-v4"
    assert outcome.delivery_report.contract_version == "research-planner-final-v3"


@pytest.mark.parametrize("stage", ["exposure", "prompt", "client"])
@pytest.mark.asyncio
async def test_provider_preserves_program_validation_errors_before_response(stage, monkeypatch):
    """参数为调用前的失败阶段和替换器；程序校验异常原样抛出，不冒充模型信封错误。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from pydantic import ValidationError

    from app.services.internal_research import research_planner_final_provider as module
    from app.services.internal_research.research_planner_final_contracts import PlannerFinalAnswerDraft

    with pytest.raises(ValidationError) as original:
        PlannerFinalAnswerDraft.model_validate({})
    client = SimpleNamespace(ainvoke=AsyncMock(side_effect=original.value))
    if stage != "client":
        name = "_record_exposure" if stage == "exposure" else "_final_answer_prompt"
        monkeypatch.setattr(module, name, Mock(side_effect=original.value))
    payload, _ = case()
    with pytest.raises(ValidationError) as observed:
        await module.LLMPlannerFinalAnswerProvider(llm=client).generate(payload)
    assert observed.value is original.value
    assert client.ainvoke.await_count == (1 if stage == "client" else 0)
