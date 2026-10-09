"""【研究交付】【模型边界】单次生成原始正文载体，不提前丢弃局部错误段落。作者：xxx。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import TypeAdapter, ValidationError

from app.middleware.trace_context import evidence_scope, record_model_evidence, trace_label
from app.services.internal_research.research_planner_consumer_budget import (
    prepare_consumer_input, render_exposed_text, render_read_reports,
    configured_consumer_budget,
    content_hash,
)
from app.services.internal_research.research_planner_final_contracts import (
    FINAL_DRAFT_VERSION, FINAL_PROMPT_VERSION, FinalProductBlockDraft,
    PlannerFinalProductDraft, PlannerFinalAnswerInput, PlannerFinalProviderOutput,
)
from app.services.internal_research.research_planner_final_evidence import final_product_catalog
from app.services.internal_research.research_planner_fulfillment import build_evidence_sufficiency_report
from app.services.internal_research.research_planner_final_delivery import project_delivery_findings
from app.services.internal_research.research_planner_final_revision import (
    PATCH_VERSION, PlannerFinalPatch, candidate_revision_request,
)
from app.services.llm_observation import record_structured_validation, structured_operation


# 1. 【研究交付】【正文职责】生成与补丁共用正文边界，来源说明回应论文内容而不复述组装规则
_ANSWER_TEXT_GUIDANCE = (
    "Reader-facing answer content (blocks[].text):\n"
    "Write the findings and limitations in ordinary Chinese, preserving useful tables, lists and code fences. "
    "For source-location explanations, name the paper's section, table or figure and the finding it supports. "
    "Describe missing information in the user's terms so they can ask a follow-up question. "
    "Formatting instructions and structured field selections are not answer content."
)
_STRUCTURED_RESPONSE_SCOPE = "Structured response controls (not reader-facing prose):"


@dataclass(frozen=True, kw_only=True)
class LLMPlannerFinalAnswerProvider:
    """【研究交付】【模型调用】通过已注入的客户端执行一次最终生成。"""

    llm: Any

    async def generate(self, payload: PlannerFinalAnswerInput) -> PlannerFinalProviderOutput:
        """参数为已准备的实际可见输入；返回原始JSON或解析错误，不二次请求模型修复。"""
        if not payload.exposure:
            raise ValueError("final provider requires the prepared consumer exposure")
        with evidence_scope("final_context"), trace_label("planner_synthesize", "final_answer"):
            with structured_operation("PlannerFinalAnswerDraft"):
                _record_exposure(payload)
                response = await self.llm.ainvoke([HumanMessage(content=_final_answer_prompt(payload))])
                raw = response.content if hasattr(response, "content") else response
                if not isinstance(raw, str):
                    raise TypeError("最终模型返回的正文载体不是字符串")
                # 1. 【研究交付】【异常归因】仅已收到的模型信封进入容错，调用前的程序异常原样保留
                try:
                    parsed = _parse_observed_output(raw)
                except json.JSONDecodeError as exc:
                    return PlannerFinalProviderOutput(raw_text=raw, parsing_error=str(exc))
                except ValidationError:
                    # 2. 【研究交付】【局部保留】信封校验失败已记录；原JSON仍交给逐块正文校验
                    return PlannerFinalProviderOutput(payload=json.loads(raw), raw_text=raw,
                                                      expected_contract_version=FINAL_DRAFT_VERSION)
        return PlannerFinalProviderOutput(payload=parsed, raw_text=raw, expected_contract_version=FINAL_DRAFT_VERSION)

    async def revise(self, payload: PlannerFinalAnswerInput, candidate: dict[str, Any]) -> PlannerFinalProviderOutput:
        """参数为当前成果和唯一基稿；返回一次局部补丁，合同失败不自动整篇重写。"""
        if not payload.exposure:
            raise ValueError("final revision requires the prepared consumer exposure")
        with evidence_scope("final_context"), trace_label("planner_synthesize", "final_answer_revision"):
            with structured_operation("PlannerFinalPatch"):
                _record_exposure(payload)
                response = await self.llm.ainvoke([HumanMessage(content=_revision_prompt(payload, candidate))])
                raw = response.content if hasattr(response, "content") else response
                if not isinstance(raw, str):
                    raise TypeError("正文修复模型返回的载体不是字符串")
                try:
                    parsed = json.loads(raw)
                    PlannerFinalPatch.model_validate(parsed)
                except (json.JSONDecodeError, ValidationError) as exc:
                    record_structured_validation(raw, schema="PlannerFinalPatch", error=exc,
                                                 parsing={"parser": "strict_json", "repaired": False})
                    return PlannerFinalProviderOutput(raw_text=raw, parsing_error=str(exc))
                record_structured_validation(raw, schema="PlannerFinalPatch", result=parsed,
                                             parsing={"parser": "strict_json", "repaired": False})
                return PlannerFinalProviderOutput(payload=parsed, raw_text=raw, expected_contract_version=PATCH_VERSION)


def _revision_prompt(payload: PlannerFinalAnswerInput, candidate: dict[str, Any]) -> str:
    """参数为当前输入与基稿；只要求输出有问题依据的局部修改，原稿正文只展示一次。"""
    return "\n".join([
        f"Prompt version: {FINAL_PROMPT_VERSION}",
        "Return exactly one JSON patch object, no Markdown wrapper. Repair the specified Chinese answer issues "
        "using only current qualified products. Do not output unchanged blocks. Preserve the original question, "
        "conditions, author attribution, table columns and uncertainty. Review feedback is not a source or claim.",
        _ANSWER_TEXT_GUIDANCE,
        _STRUCTURED_RESPONSE_SCOPE,
        "All targets and anchors refer to the SAME immutable base. replace_blocks targets must be consecutive "
        "and ordered; [] deletes, multiple new blocks split/merge. insert_after(null) inserts at the beginning. "
        "Do not overlap targets, repeat anchors, or insert at a replaced anchor. An insertion needs a missing-content "
        "or missing-presentation issue. Expand targets only with an issue_index and a concrete relationship reason "
        "within that issue's related_block_ids. Do not drop stable obligations when deleting prose.",
        "New blocks have server-owned IDs and citations. Select product_kind=claims with claim_ids only; "
        "the server derives provenance and citations. Never mix claim types in one block; split supported tables "
        "while retaining meaningful headings/columns. materials only selects allowed visible evidence_ids. "
        "structure/gap cannot declare support or fulfill content. Do not embed citation markers or citation links in text values. "
        "Citation selections belong in structured "
        "support fields; blocks[].text contains the answer itself.",
        "Every selected claim must share at least one addressed_facet_id with its block.facet_ids; a claim with no "
        "addressed facets cannot be selected for a content block. section_ids cannot replace this relationship.",
        "A real heading with same-section structure/gap prose can fulfill presentation only, including source "
        "navigation or a limitation. It gives no requested-content credit and still receives full semantic review.",
        "Return the resulting unresolved_items and status. Explain missing requested results honestly; "
        "different experiment conditions or protocols cannot substitute for the requested results.",
        "Patch schema:", json.dumps(PlannerFinalPatch.model_json_schema(), ensure_ascii=False),
        "New block schema:", json.dumps(TypeAdapter(FinalProductBlockDraft).json_schema(), ensure_ascii=False),
        "Revision request:", json.dumps(candidate_revision_request(candidate), ensure_ascii=False),
        "Base blocks:", json.dumps(candidate["blocks"], ensure_ascii=False),
        f"Original question: {payload.question}",
        "Stable requirements:", payload.plan.coverage_review.model_dump_json(), payload.context_text,
        "Historical limits:", json.dumps(payload.historical_input_context, ensure_ascii=False),
    ])


def _parse_observed_output(raw: str) -> Any:
    """参数为一次模型原始文本；记录真实JSON/信封校验，不触发模型回修。"""
    parsing = {"parser": "strict_json", "repaired": False}
    try:
        parsed = json.loads(raw)
        PlannerFinalProductDraft.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        record_structured_validation(raw, schema="PlannerFinalAnswerDraft", error=exc, parsing=parsing)
        raise
    record_structured_validation(raw, schema="PlannerFinalAnswerDraft", result=parsed, parsing=parsing)
    return parsed


def prepare_final_input(payload: PlannerFinalAnswerInput, *, evidence_max_chars: int | None = None
                        ) -> PlannerFinalAnswerInput:
    """参数为原始成果和可选测试预算；返回一次准备的可见文本与投影，供模型和校验共同使用。"""
    catalog = final_product_catalog(payload)
    claims = [item.claim for item in catalog.supports.values() if item.qualified]
    claims.extend(catalog.historical_claims.values())
    allowed_ids = set(catalog.materials)
    for support in catalog.supports.values():
        if support.qualified:
            allowed_ids.update(support.evidence_ids)
    for claim in catalog.historical_claims.values():
        allowed_ids.update(claim.supporting_evidence_ids)
    records = [record for record in payload.evidence_records if record.get("evidence_id") in allowed_ids]
    budget = configured_consumer_budget() if evidence_max_chars is None else evidence_max_chars
    prepared = prepare_consumer_input(records, claims, evidence_max_chars=budget,
                                      prompt_version=FINAL_PROMPT_VERSION)
    fulfillment = build_evidence_sufficiency_report(plan=payload.plan, results=payload.unit_results,
                    evidence_records=payload.evidence_records, pass_index=1)
    units = {unit.unit_id: unit for unit in payload.plan.units}
    claim_products = [{**item.claim.model_dump(mode="json", exclude={"source_unit_ids"}), "owner_unit_id": item.owner_unit_id,
                "output_role": units[item.owner_unit_id].output_role,
                "selected_support_claim_ids": item.claim_ids,
                "allowed_evidence_ids": item.evidence_ids, "required_citation_ids": item.citable_evidence_ids}
               for item in catalog.supports.values() if item.qualified]
    claims_text = json.dumps(claim_products, ensure_ascii=False)
    # 1. 【研究交付】【产物目录】只列任务认可的材料及本次可见性，引用原文不会自动成为材料产物
    visible_ids = {item.evidence_id for item in prepared.exposure.evidence}
    material_products = [{"evidence_id": evidence_id,
                          "eligible_facet_ids": sorted(catalog.material_facets[evidence_id]),
                          "visible": evidence_id in visible_ids}
                         for evidence_id in catalog.materials]
    context = "\n\n".join(("Verified claims (text, identity and support in one catalog):", claims_text,
                "Allowed material products:", json.dumps(material_products, ensure_ascii=False),
                "Source excerpts (for claim support or listed material products):", prepared.evidence_text,
                "Task fulfillment (acquired products, not final delivery):",
                fulfillment.model_dump_json(), "Eligibility and remaining gaps (not a generation prohibition):",
                json.dumps(payload.eligibility_report, ensure_ascii=False)))
    # 2. 【研究输入】【实际摘要】目录合并改变了实际提示，不能沿用旧双份主张的渲染指纹
    historical_text = json.dumps([claim.model_dump(mode="json") for claim in catalog.historical_claims.values()], ensure_ascii=False)
    if catalog.historical_claims:
        context += "\n\nAuthorized historical claims (support not assessed):\n" + historical_text
    rendered_hash = content_hash({"evidence": prepared.evidence_text, "claims": claims_text, "history": historical_text})
    exposure = prepared.exposure.model_copy(update={"rendering_version": "research-final-consumer-v1",
        "rendered_hash": rendered_hash, "identity_normalized_rendered_hash": rendered_hash})
    return replace(payload, exposure=exposure.model_dump(mode="json"), context_text=context)


def _final_answer_prompt(payload: PlannerFinalAnswerInput) -> str:
    """参数为同次准备输入；返回任务中性的正文合同与稳定要求，不另报答案或引用账本。"""
    coverage = payload.plan.coverage_review
    requirements = {"facets": [item.model_dump(mode="json") for item in
                              [*coverage.explicit_facets, *coverage.inferred_facets]],
                    "answer_sections": [item.model_dump(mode="json") for item in coverage.answer_sections]}
    return "\n".join([
        "Return exactly one JSON object; no Markdown wrapper and no independent answer field.",
        _ANSWER_TEXT_GUIDANCE,
        "Do not force a fixed outline or recommendation. Address every required facet, including inferred facets.",
        _STRUCTURED_RESPONSE_SCOPE,
        "Select product_kind=claims for supported claims, materials for listed material products, or structure/gap. "
        "For claims select exact claim_ids; the server derives content_kind and the required citation closure. "
        "Never supply block_id, content_kind or evidence_ids on claims. Split claims of different types into separate "
        "blocks, including separate self-contained tables where necessary; retain column meanings and context.",
        "For every selected claim listed in Verified claims, block.facet_ids must include "
        "at least one of that claim's addressed_facet_ids. This applies to each listed claim, not just one claim in the block. "
        "For a cross-topic block, select the actual facets addressed by its claims or split the block; "
        "section_ids do not substitute for facet_ids. Do not relabel a claim to match a heading.",
        "Materials must not include claim_ids; select evidence_ids only from Allowed material products with visible=true, "
        "and facet_ids from those entries' eligible_facet_ids. An empty catalog permits no materials blocks.",
        "Source excerpts are not material deliverables unless listed in that catalog. "
        "Hypotheses and history remain explicitly unverified.",
        "Do not embed citation markers or citation links in text values. "
        "Citation selections belong in structured support fields; blocks[].text contains the answer itself.",
        "Structure/gap blocks do not fulfill content requirements. A title without a substantive body is not delivery.",
        "A real heading with same-section structure/gap prose can fulfill presentation only (source navigation or "
        "limitations); it gives no content credit and remains subject to full semantic and source review.",
        "Internal sections are organizational suggestions: their titles and grouping do not add delivery obligations.",
        "For user_section, use a real Markdown heading and substantive body; the suggested title need not match.",
        "For exact_heading, use the user's exact title as a real Markdown heading followed by substantive body.",
        "Associate the heading and its actual body with the same section_ids. Code or quoted headings do not count.",
        "A section with unknown historical presentation_kind cannot be treated as a newly verified obligation.",
        "Only report answered when all required work is delivered. Preserve unresolved requirements with reasons.",
        "Preserve the requested experiment or comparison protocol and its conditions. If the source does not "
        "provide the requested results, First state which requested items cannot be supplied and cite the "
        "actual limitation: distinguish missing datasets, procedures and measurements only as this question asks. "
        "Keep those facets in unresolved_items and report incomplete. Results from a different protocol cannot "
        "fill the requested answer's headings, tables or conclusions. Include such results only when useful, "
        "as explicitly separate background with their own protocol and no claim that they fulfill the request.",
        "Unknown historical support cannot be claimed as a new validated fact; select only authorized historical claims.",
        "Explicitly allowed historical claims outside that catalog remain history; do not invent facet associations "
        "or promote them to validated facts.",
        "A correct source link does not prove semantic entailment. Preserve conditions and uncertainty.",
        "Keep author explanations or causal attributions distinct from measured outcomes, including in table columns. "
        "A paper_supported claim can report what the authors propose; retain that attribution rather than "
        "presenting the proposed mechanism as experimentally established.",
        "reasoning_summary explains a claim; it is not an additional source of paper facts. "
        "Do not turn unsupported premises in it into answer statements. "
        "An unobserved experiment is not proof that the paper performed no such experiment. "
        "Report explicit source statements about unperformed work as such; otherwise describe only the visible gap.",
        f"Prompt version: {FINAL_PROMPT_VERSION}",
        "Response envelope schema:", json.dumps(PlannerFinalProductDraft.model_json_schema(), ensure_ascii=False),
        "Each blocks[] entry follows this schema:", json.dumps(TypeAdapter(FinalProductBlockDraft).json_schema(), ensure_ascii=False),
        f"Question: {payload.question}",
        "Stable delivery requirements:", json.dumps(requirements, ensure_ascii=False), payload.context_text,
        "Previous candidate content review (correct the specific issues using the actual current products; "
        "review text is not a new claim or source):",
        json.dumps(payload.revision_context.get("content_review") or {}, ensure_ascii=False),
        "Previous candidate delivery findings (authoritative citation, requirement and presentation gaps):",
        json.dumps(project_delivery_findings(payload.revision_context), ensure_ascii=False),
        "Historical limits:", json.dumps(payload.historical_input_context, ensure_ascii=False),
        render_read_reports([report for result in payload.unit_results
                             for report in result.metadata.get("read_reports", [])]),
    ])


def _record_exposure(payload: PlannerFinalAnswerInput) -> None:
    """参数为实际送模投影；分别记录本次原文范围及含截断说明的展示文本。"""
    records = {str(record["evidence_id"]): record for record in payload.evidence_records}
    for item in payload.exposure["evidence"]:
        record = records[item["evidence_id"]]
        body = record[item["field"]]
        text = body[item["start"]:item["end"]]
        record_model_evidence(record, text,
                              rendered_text=render_exposed_text(body, item["start"], item["end"]))
