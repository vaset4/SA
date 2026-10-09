"""【研究交付】【回归】从正式综合入口验证正文、引用、任务兑现与失败事实。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import socket
from typing import Any

import pytest

from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx
from app.models.research_planner_contracts import CURRENT_RESEARCH_CONTRACT_VERSION, ResearchClaim, ResearchPlan
from app.services.internal_research.research_planner_final_contracts import FINAL_CONTRACT_VERSION, PlannerFinalAnswerInput
from app.services.internal_research.research_planner_final_provider import LLMPlannerFinalAnswerProvider, prepare_final_input
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer, synthesize_planner_answer_async
from research_planner_test_utils import make_draft, make_facet, make_unit, result_with_exposure, supporting_spans


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """参数为替换器；在HTTP和数据库边界禁止真实外联，不影响Windows内部唤醒管道。"""
    import httpx
    import requests
    from app.services import database

    monkeypatch.setattr(httpx.AsyncClient, "send", _deny_connection)
    monkeypatch.setattr(httpx.Client, "send", _deny_connection)
    monkeypatch.setattr(requests.Session, "request", _deny_connection)
    monkeypatch.setattr(database, "get_pool", _deny_connection)
    monkeypatch.setattr(socket, "create_connection", _deny_connection)


def _deny_connection(*args, **kwargs):
    """参数为任何连接请求；真实外联立即失败，不模拟网络成功。"""
    raise AssertionError("O4 tests must not open network connections")


@dataclass
class FinalProvider:
    """【研究交付】【模型替身】只返回明确草稿并保存实际输入，不调用外部服务。"""

    draft: Any
    calls: int = 0
    seen: Any = None

    async def generate(self, payload):
        """参数为真实准备输入；记录一次调用并返回测试明确指定的模型输出。"""
        self.calls += 1
        self.seen = payload
        return deepcopy(self.draft)


def _case(kind="fact", *, sections=False):
    """参数为业务产物和章节要求；返回已登记来源、实际投影和明确v4正式计划的测试案例。"""
    claim_type = {"fact": "paper_supported", "inference": "evidence_grounded_inference",
                  "hypothesis": "model_prior_hypothesis", "history": "memory_continuation"}.get(kind)
    unit = make_unit("u1", allowed_claims=[claim_type] if claim_type else ["paper_supported"], output_role="evidence")
    facet = make_facet("f1")
    if kind == "materials":
        unit["expected_output"].update(deliverable_kind="materials", required_claim_types=[])
        unit["claim_policy"].update(allowed_claim_types=[], citation_required=False)
        facet.update(deliverable_kind="materials", min_material_count=1)
    else:
        facet["required_claim_types"] = [claim_type]
    if kind in {"hypothesis", "history", "inference"}:
        unit["claim_policy"]["citation_required"] = False
    raw = make_draft([unit], explicit_facets=[facet])
    if sections:
        raw["coverage_review"]["answer_sections"] = [{"section_id": "s1", "title": "结论", "facet_ids": ["f1"],
                    "required": True, "source": "explicit", "user_text_span": "结论", "rationale": "requested",
                    "presentation_kind": "exact_heading"}]
        raw["units"][0]["expected_output"]["answer_section_ids"] = ["s1"]
    plan = ResearchPlan.model_validate({**raw, "contract_version": CURRENT_RESEARCH_CONTRACT_VERSION})
    record = {"evidence_id": "ev-a", "paper_id": "paper-a", "layer": "direct_text", "usable_as_citation": True,
              "section": "Method", "excerpt": "The measured accuracy is 91% under condition A."}
    records = [] if kind in {"hypothesis", "history"} else [record]
    claims = [] if kind == "materials" else [ResearchClaim(claim_id="u1.c1", text="条件A下准确率为91%",
              claim_type=claim_type, validation_status="needs_validation" if not records else "validated",
              supporting_evidence_ids=["ev-a"] if records else [], reasoning_summary="依据明确来源",
              supporting_spans=supporting_spans(records),
              addressed_facet_ids=["f1"], validation_tasks=["实验验证"] if not records else [])]
    result = result_with_exposure(plan.units[0], records, claims)
    draft = {"contract_version": FINAL_CONTRACT_VERSION, "status": "answered", "blocks": [{
             "block_id": "b1", "content_kind": kind, "text": "## 结论\n\n条件A下，准确率为91%。" if sections else "条件A下，准确率为91%。",
             "facet_ids": ["f1"], "section_ids": ["s1"] if sections else [],
             "claim_ids": ["u1.c1"] if claims else [], "evidence_ids": ["ev-a"] if records else []}]}
    return plan, [result], records, draft


def _synthesize(case, *, draft=None, error="", repair=None):
    """参数为完整案例及可选输出变化；调用实际同步综合入口。"""
    plan, results, records, default_draft = case
    return synthesize_planner_answer(plan, results, project_paper_count=1, evidence_records=records,
               final_answer_draft=default_draft if draft is None else draft,
               final_provider_error=error, repair_context=repair)


@pytest.mark.parametrize("kind", ["fact", "inference", "materials", "hypothesis", "history"])
@pytest.mark.asyncio
async def test_each_product_can_complete_without_a_recommendation_unit(kind):
    """参数为实际产物；各类合法交付都经过最终provider，不被推荐或论文引用总门阻断。"""
    plan, results, records, draft = _case(kind)
    provider = FinalProvider(draft)
    payload = await synthesize_planner_answer_async(plan, results, project_paper_count=0,
               evidence_records=records, final_provider=provider)
    assert provider.calls == 1
    assert payload["final_delivery_report"]["status"] == "complete", payload["final_delivery_report"]
    assert payload["source_status"]["planner_final"] == "ok"
    assert "还没有" not in payload["answer"]


def test_same_body_drives_citations_with_unicode_and_preserved_whitespace():
    """无参数；表情、组合字符和前导空白不改变同次原始Markdown引用坐标。"""
    case = _case()
    case[3]["blocks"][0]["text"] = "  😀 e\u0301\n\n| 条件 | 数值 |\n|---|---|\n| A | 91% |"
    payload = _synthesize(case)
    binding = payload["final_citation_bindings"][0]
    assert payload["answer"].startswith("  😀")
    assert payload["answer"][binding["start"]:binding["end"]] == "[E1]"
    assert binding["evidence_id"] == "ev-a"
    assert binding["attempt_id"] == payload["final_generation_attempt"]["attempt_id"]
    assert payload["citations"][0]["evidence_id"] == "ev-a"


@pytest.mark.parametrize("mutation", ["unknown_claim", "unknown_evidence", "empty_citations", "wrong_kind", "missing_facets", "written_marker"])
def test_invalid_only_block_keeps_text_but_fails_delivery(mutation):
    """参数为坏关系；正文保留不等于有效交付，失败必须带真实原因。"""
    case = _case()
    block = case[3]["blocks"][0]
    if mutation == "unknown_claim":
        block["claim_ids"] = ["outside"]
    elif mutation == "unknown_evidence":
        block["evidence_ids"] = ["outside"]
    elif mutation == "empty_citations":
        block["evidence_ids"] = []
    elif mutation == "wrong_kind":
        block["content_kind"] = "hypothesis"
    elif mutation == "missing_facets":
        block["facet_ids"] = []
    else:
        block["text"] += " [E1]"
    payload = _synthesize(case)
    assert block["text"] in payload["answer"]
    assert payload["source_status"]["planner_final"] == "failed"
    assert payload["planner_final_validation_errors"]
    assert payload["final_citation_bindings"] == []
    assert payload["citations"] == []


@pytest.mark.parametrize("bad", [{"block_id": "bad", "text": "保留的坏段", "claim_ids": 1},
                                {"block_id": "empty", "text": " "}, 7])
def test_local_malformed_block_does_not_discard_valid_or_readable_text(bad):
    """参数为局部畸形输入；保留所有可读正文，合法段仍有独立有效引用。"""
    case = _case()
    case[3]["blocks"].append(bad)
    payload = _synthesize(case)
    assert payload["final_delivery_report"]["status"] == "partial"
    assert "条件A下" in payload["answer"]
    if isinstance(bad, dict) and bad.get("text", "").strip():
        assert bad["text"] in payload["answer"]
    assert len(payload["final_citation_bindings"]) == 1


def test_duplicate_block_ids_keep_both_texts_without_credit():
    """无参数；重复身份不自动重编号，所有同名块都失去关系信用。"""
    case = _case()
    case[3]["blocks"].append({**case[3]["blocks"][0], "text": "另一个同名正文"})
    payload = _synthesize(case)
    assert "另一个同名正文" in payload["answer"]
    assert payload["final_citation_bindings"] == []
    assert payload["source_status"]["planner_final"] == "failed"


@pytest.mark.parametrize("raw", [[], {}, {"contract_version": "old", "blocks": []},
                                {"contract_version": FINAL_CONTRACT_VERSION, "blocks": "wrong"}])
def test_top_level_contract_failure_is_not_rescued_by_saved_claims(raw):
    """参数为不可用顶层载体；已有成果说明可见，但不会把失败变为部分成功。"""
    payload = _synthesize(_case(), draft=raw)
    assert payload["answer"].strip()
    assert payload["final_delivery_report"]["status"] == "failed"
    assert payload["final_delivery_report"]["reasons"]


@pytest.mark.parametrize("blocks", [[], [{"block_id": "empty", "text": ""}],
    [{"block_id": "title", "text": "# 只有标题", "content_kind": "structure", "facet_ids": [],
      "section_ids": [], "claim_ids": [], "evidence_ids": []}]])
def test_called_provider_without_business_delivery_is_failed(blocks):
    """参数为空输出或空说明；有输入而未交付属于生成失败，不属于取证不足。"""
    payload = _synthesize(_case(), draft={"contract_version": FINAL_CONTRACT_VERSION, "blocks": blocks})
    assert payload["final_delivery_report"]["status"] == "failed"
    assert payload["final_delivery_report"]["generation_status"] == "completed"


def test_no_input_skips_provider_with_an_insufficient_report():
    """无参数；无业务输入不伪造模型调用，也不把说明当完整答案。"""
    plan, _, _, _ = _case()
    payload = synthesize_planner_answer(plan, [], project_paper_count=1)
    assert payload["final_delivery_report"]["status"] == "insufficient"
    assert payload["final_delivery_report"]["generation_status"] == "not_called"
    assert payload["source_status"]["planner_final"] == "skipped"


@pytest.mark.parametrize("change", [{"status": "incomplete"}, {"status": []}, {"unresolved_items": "bad"},
    {"unresolved_items": [{"facet_id": "f1", "reason": "还没完成"}]}, {"answer": "独立另一份正文"}])
def test_unresolved_or_malformed_auxiliary_fields_cannot_become_complete(change):
    """参数为自报未完成或畸形辅助声明；形式引用合法仍不能掩盖其缺口。"""
    case = _case()
    case[3].update(change)
    payload = _synthesize(case)
    assert "条件A下" in payload["answer"]
    assert payload["final_delivery_report"]["status"] == "partial"
    assert "独立另一份正文" not in payload["answer"]


def test_required_section_needs_its_heading_and_actual_business_body():
    """无参数；章节标题、关联和业务产物一起完成结构义务。"""
    case = _case(sections=True)
    assert _synthesize(case)["final_delivery_report"]["status"] == "complete"
    case[3]["blocks"][0]["text"] = "有结论但没有要求中的章节标题"
    payload = _synthesize(case)
    assert payload["final_delivery_report"]["status"] == "partial"
    assert payload["final_delivery_report"]["section_results"][0]["status"] == "missing"


def test_code_literal_is_not_a_model_written_citation():
    """无参数；代码中的引用字面量保持正文原义，服务端追加引用仍单独绑定。"""
    case = _case()
    case[3]["blocks"][0]["text"] = "示例 `print('[E1]')`\n\n```python\nprint('[E2]')\n```"
    payload = _synthesize(case)
    assert payload["final_delivery_report"]["status"] == "complete"
    assert len(payload["final_citation_bindings"]) == 1


def test_failure_and_eligibility_keep_actual_body_and_acquired_materials():
    """无参数；失败说明与资格风险都不能清空成果或改称成功交付。"""
    payload = _synthesize(_case(), error="TimeoutError: actual provider timeout",
                         repair={"final_answer_eligibility_report": {"can_call_provider": False}})
    assert payload["source_status"]["planner"] == "ok"
    assert payload["source_status"]["planner_final"] == "failed"
    assert payload["evidence_records"][0]["evidence_id"] == "ev-a"
    assert "actual provider timeout" in payload["source_status_reasons"]["planner_final"]
    assert "资格" in payload["answer"] or "缺口" in payload["answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_auxiliary", [False, True])
async def test_final_provider_preserves_local_raw_blocks_and_observes_one_call(bad_auxiliary):
    """无参数；真实provider边界只有一次调用，局部坏字段不会在严格嵌套模型处丢正文。"""
    case = _case()
    case[3]["blocks"].append({"text": "原样保留", "claim_ids": 1})
    if bad_auxiliary:
        case[3]["status"] = []
    llm = RawLLM(json.dumps(case[3], ensure_ascii=False))
    trace = TraceContext(request_id="o4-final")
    token = set_trace_ctx(trace)
    try:
        payload = await synthesize_planner_answer_async(case[0], case[1], project_paper_count=1,
                    evidence_records=case[2], final_provider=LLMPlannerFinalAnswerProvider(llm=llm))
        observed = export_evidence_observations(trace)
    finally:
        reset_trace_ctx(token)
    assert llm.calls == 1 and "原样保留" in payload["answer"]
    assert payload["final_delivery_report"]["status"] == "partial"
    events = [item["payload"] for item in observed["evidence_observations"]
              if item["stage"] == "structured_business_validation"]
    assert events[-1]["valid"] is False and events[-1]["accepted"] is False
    assert observed["evidence_capture_errors"] == []


@dataclass
class RawLLM:
    """【研究交付】【协议替身】仅实现现有单次ainvoke协议，禁止结构化自动回修。"""

    text: str
    calls: int = 0

    async def ainvoke(self, messages):
        """参数为真实生成提示；返回明确原始文本并记录调用次数。"""
        self.calls += 1
        assert FINAL_CONTRACT_VERSION in messages[0].content
        return self.text


@pytest.mark.asyncio
async def test_final_provider_does_not_extract_json_from_markdown_wrapper():
    """无参数；非法根载体显式失败，不猜测正文中的JSON片段。"""
    case = _case()
    llm = RawLLM("```json\n" + json.dumps(case[3]) + "\n```")
    payload = await synthesize_planner_answer_async(case[0], case[1], project_paper_count=1,
               evidence_records=case[2], final_provider=LLMPlannerFinalAnswerProvider(llm=llm))
    assert llm.calls == 1
    assert payload["final_delivery_report"]["status"] == "failed"


def test_prepared_final_exposure_records_only_actual_material_text():
    """无参数；预算不足时最终模型不会获得全池可见性，但继承主张仍明确进入输入。"""
    case = _case()
    payload = prepare_final_input(PlannerFinalAnswerInput(question=case[0].question, plan=case[0],
              unit_results=case[1], evidence_records=case[2]), evidence_max_chars=0)
    assert payload.exposure["evidence"] == []
    assert payload.exposure["claims"][0]["claim_id"] == "u1.c1"
    assert payload.exposure["omitted_evidence_ids"] == ["ev-a"]


@pytest.mark.parametrize("section_id", ["s1", ""])
def test_model_unresolved_preserves_formal_coverage_conflict(section_id):
    """无参数；模型未完成声明必须反映在要求行，形式覆盖仍保留为冲突事实。"""
    case = _case(sections=True)
    case[3]["unresolved_items"] = [{"facet_id": "f1", "section_id": section_id, "reason": "实验条件尚未解释完"}]
    report = _synthesize(case)["final_delivery_report"]
    assert report["status"] == "partial"
    for row in [*report["requirement_results"], *report["section_results"]]:
        assert row["status"] == "missing"
    assert any(reason["code"] == "model_delivery_conflict" for reason in report["requirement_results"][0]["reasons"])


def test_optional_unresolved_does_not_reduce_required_delivery():
    """无参数；明确未做可选扩展仍保留，但不降低已完成的必需交付。"""
    case = _case()
    optional = make_facet("extra", "inferred")
    review = case[0].coverage_review.model_copy(update={"inferred_facets": [
        type(case[0].coverage_review.explicit_facets[0]).model_validate(optional)]})
    case = (case[0].model_copy(update={"coverage_review": review}), *case[1:])
    case[3].update(status="incomplete", unresolved_items=[{"facet_id": "extra", "reason": "未开展可选扩展"}])
    report = _synthesize(case)["final_delivery_report"]
    assert report["status"] == "complete"
    assert report["unresolved_items"][0]["required"] is False


@pytest.mark.asyncio
async def test_current_final_budget_does_not_revalidate_old_claim_with_new_visibility(monkeypatch):
    """参数为设置替身；本次未重读原文仍可继承旧已核验主张，不更改原凭据。"""
    from app.services.internal_research import research_planner_final_provider

    monkeypatch.setattr(research_planner_final_provider, "configured_consumer_budget", lambda: 0)
    case = _case()
    original = deepcopy(case[1][0].metadata)
    provider = FinalProvider(case[3])
    payload = await synthesize_planner_answer_async(case[0], case[1], project_paper_count=1,
                evidence_records=case[2], final_provider=provider)
    assert payload["final_delivery_report"]["status"] == "complete"
    assert provider.seen.exposure["evidence"] == []
    assert provider.seen.exposure["omitted_evidence_ids"] == ["ev-a"]
    assert payload["final_citation_bindings"][0]["evidence_id"] == "ev-a"
    assert case[1][0].metadata == original


def _mixed_case():
    """无参数；返回事实、双对象比较和推荐混合要求及逐条明确的支持选择。"""
    units = [make_unit("u1", goal="核对A论文", facets=["f1"], output_role="evidence"),
             make_unit("u2", goal="核对B论文", facets=["f2"], paper_ids=["paper-b"], output_role="evidence")]
    for uid, facets, deps, role in [("u3", ["f3"], ["u1", "u2"], "synthesis"),
                                     ("u4", ["f3", "f4"], ["u3"], "final_recommendation")]:
        unit = make_unit(uid, goal=f"形成{role}产物", source_type="inference_from_units", allowed_claims=["evidence_grounded_inference"],
                facets=facets, deps=[{"unit_id": dep, "required": True} for dep in deps], output_role=role)
        unit["claim_policy"]["citation_required"] = False
        unit["evidence_policy"]["target_paper_ids"] = []
        units.append(unit)
    facets = [make_facet(f"f{index}") for index in range(1, 5)]
    facets[2].update(required_paper_ids=["paper-a", "paper-b"], required_claim_types=["evidence_grounded_inference"])
    facets[3].update(required_output_role="final_recommendation")
    plan = ResearchPlan.model_validate({**make_draft(units, explicit_facets=facets),
                                       "contract_version": CURRENT_RESEARCH_CONTRACT_VERSION})
    records = [{"evidence_id": f"ev-{key}", "paper_id": f"paper-{key}", "layer": "direct_text",
                "usable_as_citation": True, "section": "Method", "excerpt": f"Method {key} has measured support."}
               for key in ("a", "b")]
    results = []
    for index, record in enumerate(records):
        claim = ResearchClaim(claim_id=f"u{index + 1}.c1", text=record["excerpt"], claim_type="paper_supported",
                validation_status="validated", supporting_evidence_ids=[record["evidence_id"]],
                supporting_spans=supporting_spans([record]), addressed_facet_ids=[f"f{index + 1}"])
        results.append(result_with_exposure(plan.units[index], [record], [claim]))
    comparison = ResearchClaim(claim_id="u3.c1", text="比较两个对象", claim_type="evidence_grounded_inference",
        validation_status="validated", source_claim_ids=["u1.c1", "u2.c1"], source_unit_ids=["u1", "u2"],
        addressed_facet_ids=["f3"], reasoning_summary="依据双方事实比较")
    results.append(result_with_exposure(plan.units[2], [], [comparison], dependencies=results[:2]))
    recommendations = [ResearchClaim(claim_id=f"u4.c{index + 1}", text=text, claim_type="evidence_grounded_inference",
        validation_status="validated", source_claim_ids=["u3.c1"], source_unit_ids=["u3"],
        addressed_facet_ids=[facet], reasoning_summary="依据比较结果")
        for index, (text, facet) in enumerate([("比较补充", "f3"), ("建议先测方法A", "f4")])]
    results.append(result_with_exposure(plan.units[3], [], recommendations, dependencies=[results[2]]))
    blocks = [{"block_id": f"b{index + 1}", "text": claim.text,
        "content_kind": "fact" if index < 2 else "inference", "section_ids": [],
        "facet_ids": claim.addressed_facet_ids, "claim_ids": [claim.claim_id],
        "evidence_ids": [f"ev-{chr(ord('a') + index)}"] if index < 2 else ["ev-a", "ev-b"]}
        for index, claim in enumerate([results[0].claims[0], results[1].claims[0], comparison, recommendations[1]])]
    return plan, results, records, {"contract_version": FINAL_CONTRACT_VERSION, "blocks": blocks}


@pytest.mark.parametrize("reversed_claims", [False, True])
def test_mixed_recommendation_uses_the_associated_claim_independent_of_order(reversed_claims):
    """参数为推荐主张顺序；辅助比较不能代替推荐，真正关联的第二条主张可完成推荐。"""
    case = _mixed_case()
    if reversed_claims:
        case[1][-1] = case[1][-1].model_copy(update={"claims": list(reversed(case[1][-1].claims))})
    assert _synthesize(case)["final_delivery_report"]["status"] == "complete"
    case[3]["blocks"] = case[3]["blocks"][:-1]
    report = _synthesize(case)["final_delivery_report"]
    assert report["status"] == "partial"
    assert report["requirement_results"][-1]["status"] == "missing"


@pytest.mark.asyncio
async def test_missing_comparison_object_and_recommendation_are_visible_to_final_provider():
    """无参数；缺少B对象及推荐仍调用最终模型，提示保留统一缺口，不能伪造全部完成。"""
    case = _mixed_case()
    case[1][1] = case[1][1].model_copy(update={"status": "insufficient_evidence", "claims": [], "evidence_ids": []})
    case[3]["blocks"] = case[3]["blocks"][:1]
    provider = FinalProvider(case[3])
    payload = await synthesize_planner_answer_async(case[0], case[1], project_paper_count=2,
        evidence_records=case[2], final_provider=provider,
        repair_context={"final_answer_eligibility_report": {"can_call_provider": False, "reason": "对象B与推荐缺口"}})
    assert provider.calls == 1 and payload["final_delivery_report"]["status"] == "partial"
    report_text = provider.seen.context_text.split("Task fulfillment (acquired products, not final delivery):\n\n", 1)[1]
    report = json.loads(report_text.split("\n\nEligibility", 1)[0])
    comparison = next(row for row in report["facet_reports"] if row["facet_id"] == "f3")
    assert "paper-b" in comparison["missing_paper_ids"]
    assert "对象B与推荐缺口" in provider.seen.context_text
    assert "Verified support relationships" in provider.seen.context_text


@pytest.mark.parametrize("title", ["# 只有标题", "只有标题\n===", "只有标题\n---"])
def test_title_only_business_block_does_not_receive_delivery_credit(title):
    """参数为Markdown标题形式；选择了合法主张也不能让空标题变成业务交付。"""
    case = _case()
    case[3]["blocks"][0]["text"] = title
    assert _synthesize(case)["final_delivery_report"]["status"] == "failed"


@pytest.mark.asyncio
async def test_provider_returning_none_is_a_completed_but_malformed_attempt():
    """无参数；已调用模型却未返回载体必须记录生成失败，不能冒称没有调用或没有材料。"""
    case = _case()
    provider = FinalProvider(None)
    result = await synthesize_planner_answer_async(case[0], case[1], project_paper_count=1,
                evidence_records=case[2], final_provider=provider)
    assert provider.calls == 1
    assert result["final_delivery_report"]["status"] == "failed"
    assert result["final_delivery_report"]["generation_status"] == "completed"
    assert result["final_delivery_report"]["reasons"][0]["code"] == "malformed_provider_output"


@pytest.mark.parametrize("cite_related", [False, True])
def test_related_object_does_not_complete_comparison_or_gain_citation_credit(cite_related):
    """参数为是否误引相关材料；A支持/B相关不能交付完整A/B比较，也不能借全局B引用授信。"""
    case = _mixed_case()
    claim = case[1][2].claims[0].model_copy(update={"source_claim_ids": ["u1.c1"],
             "source_unit_ids": ["u1"], "related_evidence_ids": ["ev-b"]})
    case[1][2] = case[1][2].model_copy(update={"claims": [claim]})
    case[3]["blocks"] = case[3]["blocks"][:3]
    case[3]["blocks"][2]["evidence_ids"] = ["ev-a", "ev-b"] if cite_related else ["ev-a"]
    payload = _synthesize(case)
    assert payload["final_delivery_report"]["status"] == "partial"
    comparison = payload["final_delivery_report"]["requirement_results"][2]
    assert comparison["status"] == "missing"
    assert "比较两个对象" in payload["answer"]
    assert not any(binding["block_id"] == "b3" and binding["evidence_id"] == "ev-b"
                   for binding in payload["final_citation_bindings"])


def test_auxiliary_claim_of_recommendation_unit_cannot_deliver_recommendation_facet():
    """无参数；推荐单元内的比较辅助主张不能仅凭所属角色替代真正推荐主张。"""
    case = _mixed_case()
    case[3]["blocks"][-1]["claim_ids"] = ["u4.c1"]
    payload = _synthesize(case)
    assert payload["final_delivery_report"]["status"] == "partial"
    assert payload["final_delivery_report"]["requirement_results"][-1]["status"] == "missing"
    assert any(error["code"] == "unassociated_product" for error in payload["planner_final_validation_errors"])


def test_multiblock_unicode_citations_roundtrip_without_linking_a_bad_matching_marker():
    """无参数；多块表情及组合字符经保存读取后保持全文坐标，坏段同名标记不获得绑定。"""
    case = _case()
    first = case[3]["blocks"][0]
    first["text"] = "  😀 e\u0301\r\n\r\n第一段事实。"
    case[3]["blocks"].extend([{**first, "block_id": "b2", "text": "错误段里的 [E1] 仍可读。"},
                              {**first, "block_id": "b3", "text": "后续事实。"}])
    payload = json.loads(json.dumps(_synthesize(case), ensure_ascii=False))
    assert payload["final_delivery_report"]["status"] == "partial"
    assert payload["answer"].startswith(first["text"])
    bindings = payload["final_citation_bindings"]
    assert [binding["block_id"] for binding in bindings] == ["b1", "b3"]
    for binding in bindings:
        assert payload["answer"][binding["start"]:binding["end"]] == "[E1]"
        assert binding["attempt_id"] == payload["final_generation_attempt"]["attempt_id"]
    assert bindings[1]["start"] > payload["answer"].index("错误段里的 [E1]")
