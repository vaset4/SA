"""【研究主张】【模型草拟】只渲染已准备输入，保留明确支持片段。作者：xxx。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from app.middleware.trace_context import evidence_scope, record_model_evidence, trace_label
from app.models.research_planner_contracts import (
    BoundResearchClaimDraftBatch, PlannerCoverageReview, ResearchClaimDraft, ResearchClaimDraftRejection,
    ResearchUnit, ResearchUnitResult,
)
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.services.llm_observation import record_structured_conversion
from app.services.internal_research.research_planner_consumer_budget import (
    CLAIM_PROMPT_VERSION, PreparedConsumerInput, configured_consumer_budget, prepare_consumer_input,
    render_exposed_text, render_read_reports,
)


@dataclass(frozen=True, kw_only=True)
class LLMClaimDraftProvider:
    """【研究主张】【草拟端口】接收一次准备的输入，返回模型真实结构化草稿。"""

    llm: Any
    coverage_review: PlannerCoverageReview | None = None
    question: str = ""

    async def __call__(
        self, unit: ResearchUnit, evidence_records: list[dict[str, Any]],
        dependency_results: list[ResearchUnitResult], *, read_reports: list[dict[str, Any]] | None = None,
        prepared_input: PreparedConsumerInput | None = None,
    ) -> BoundResearchClaimDraftBatch:
        """参数为单元、材料、依赖与准备投影；返回草稿，不在 provider 另做裁剪。"""
        prepared = prepared_input or prepare_consumer_input(evidence_records,
            [claim for result in dependency_results if result.unit_id in {dep.unit_id for dep in unit.dependencies}
             for claim in result.claims], evidence_max_chars=configured_consumer_budget(),
            prompt_version=CLAIM_PROMPT_VERSION)
        drafter = self.llm.with_structured_output(ResearchClaimStrategyDraftBatch)
        with evidence_scope("claim_context", unit_id=unit.unit_id), trace_label("planner_execute", "claim_draft"):
            batch = await drafter.ainvoke([HumanMessage(content=build_claim_prompt(
                unit, evidence_records, dependency_results, read_reports=read_reports,
                coverage_review=self.coverage_review, prepared_input=prepared,
                question=self.question,
            ))])
            converted = bind_claim_batch(batch, prepared)
            # 1. 【研究主张】【转换身份】显式连接原模型响应与绑定版本后的对象，不反推或改造原响应
            record_structured_conversion(batch, {"claims": [draft.model_dump(mode="json") for draft in converted.claims]},
                                         schema=ResearchClaimStrategyDraftBatch.__name__)
        return converted


def bind_claim_batch(batch: ResearchClaimStrategyDraftBatch, prepared: PreparedConsumerInput) -> BoundResearchClaimDraftBatch:
    """参数为模型原始批次和实际曝光；返回可绑定主张及逐条拒绝，不捕获程序异常或重试模型。"""
    visible = {(item.evidence_id, item.field) for item in prepared.exposure.evidence}
    evidence_ids = {item.evidence_id for item in prepared.exposure.evidence}
    accepted: list[ResearchClaimDraft] = []
    positions: list[int] = []
    rejected: list[ResearchClaimDraftRejection] = []
    for position, claim in enumerate(batch.claims, start=1):
        outside = [f"{span.evidence_id}/{span.field}" for span in claim.supporting_spans
                   if (span.evidence_id, span.field) not in visible]
        outside.extend(evidence_id for evidence_id in claim.supporting_evidence_ids if evidence_id not in evidence_ids)
        if outside:
            # 1. 【研究主张】【选择拒绝】只有模型选择了未曝光来源才隔离整条主张，原编号和文字保持可追查
            message = f"claims[{position}] (1-based) support selection is outside this model input: {', '.join(dict.fromkeys(outside))}"
            rejected.append(ResearchClaimDraftRejection(claim_position=position, code="support_selection_not_exposed",
                message=message, evidence_ids=list(dict.fromkeys([*claim.supporting_evidence_ids,
                    *(span.evidence_id for span in claim.supporting_spans)])), raw_claim=claim.model_dump(mode="json")))
            continue
        # 2. 【研究主张】【严格绑定】保留原有版本绑定与模型构造校验，真实程序错误继续直接抛出
        accepted.extend(bind_claim_versions(batch.model_copy(update={"claims": [claim]}), prepared))
        positions.append(position)
    return BoundResearchClaimDraftBatch(claims=accepted, claim_positions=positions, rejected_claims=rejected)


def bind_claim_versions(batch: ResearchClaimStrategyDraftBatch, prepared: PreparedConsumerInput) -> list[ResearchClaimDraft]:
    """参数为模型支持选择与同次投影；返回绑定真实版本的草稿，未知材料明确失败。"""
    versions = {(item.evidence_id, item.field): item.evidence_version for item in prepared.exposure.evidence}
    claims = []
    for claim in batch.claims:
        spans = []
        for span in claim.supporting_spans:
            key = (span.evidence_id, span.field)
            if key not in versions:
                raise ValueError(f"support selection is outside this model input: {span.evidence_id}/{span.field}")
            spans.append({**span.model_dump(), "evidence_version": versions[key]})
        claims.append(ResearchClaimDraft.model_validate({**claim.model_dump(), "supporting_spans": spans}, strict=True))
    return claims


def build_claim_prompt(
    unit: ResearchUnit, evidence_records: list[dict[str, Any]], dependency_results: list[ResearchUnitResult], *,
    read_reports: list[dict[str, Any]] | None = None, coverage_review: PlannerCoverageReview | None = None,
    prepared_input: PreparedConsumerInput | None = None,
    question: str = "",
) -> str:
    """参数为真实输入和要求；返回原文、对象、数值上下文及来源选取规则的完整提示。"""
    dependency_results = [result for result in dependency_results
                          if result.unit_id in {dependency.unit_id for dependency in unit.dependencies}]
    prepared = prepared_input or prepare_consumer_input(evidence_records,
        [claim for result in dependency_results for claim in result.claims],
        evidence_max_chars=configured_consumer_budget(), prompt_version=CLAIM_PROMPT_VERSION)
    reports = list(read_reports or [])
    for result in dependency_results:
        reports.extend(result.metadata.get("read_reports", []))
    facets = [] if coverage_review is None else [facet.model_dump(mode="json")
        for facet in [*coverage_review.explicit_facets, *coverage_review.inferred_facets]
        if facet.facet_id in unit.covered_facet_ids]
    _observe_exposure(evidence_records, prepared)
    return "\n\n".join([
        "Return one JSON object matching the supplied claim strategy schema. No prose outside JSON.",
        "Draft only claim types explicitly allowed by the unit. An empty claims array is a valid honest result.",
        "Direct support requires supporting_evidence_ids AND supporting_spans selecting evidence_id, "
        "field, start, end and quote. Copy evidence_id, field, start and end exactly from one matching evidence entry "
        "in Visible ranges. These bounds identify the authoritative visible source region, not the quote's character "
        "offset. Do not count or recalculate character positions or subtract Markdown fences. "
        "quote must be a verbatim substring inside that selected region. The server binds the version from this input.",
        "Keep the original objects, numbers, units and experimental conditions together. An exact quote is "
        "traceability, not proof that it entails your statement. Do not change a number or its conditions.",
        "Preserve the source's quantifiers and scope. Do not strengthen a reported aggregate or unspecified "
        "deviation into all datasets, every condition or always; only make universal claims explicitly supported "
        "by the selected source. Preserve what is an author statement rather than an independently verified fact.",
        "Both text and reasoning_summary must stay within the selected support. "
        "reasoning_summary must not introduce additional factual premises, including claims that experiments "
        "were absent. Absence from the visible excerpts does not establish that an experiment was never performed; "
        "only explicit source statements can establish that it was not performed or remains future work. "
        "Otherwise state the limit of the visible evidence.",
        "Preserve author attribution for proposed causal explanations and keep them distinct from directly "
        "observed experimental outcomes. A paper_supported claim may report the authors' explanation, "
        "but must not turn it into an experimentally established mechanism.",
        "Use paper_supported only for direct_text citable material. Metadata/directory support can justify an "
        "allowed evidence_grounded_inference, never paper facts. Every inference requires reasoning_summary.",
        "source_claim_ids must select the specific eligible claims from direct dependencies below. "
        "Ownership is derived by the server. Never inherit a whole unit. "
        "Selecting an upstream claim inherits only its already verified support and does not assert rereading omitted text.",
        "related_evidence_ids are background only and provide no factual support. Hypotheses and memory_continuation "
        "remain needs_validation; they cannot support facts. Do not invent a specific memory identity.",
        "addressed_facet_ids must name only assigned facets this exact claim addresses. A recommendation unit's "
        "auxiliary facts must not claim the recommendation facet merely because of ownership. List order has no priority.",
        "Compare each proposed facet association with the original question: it must concern the same requested "
        "object, experiment protocol, conditions and metrics. General background or results from a different "
        "or proxy protocol may remain useful context but must not be associated as the requested result merely "
        "because the topic matches. Leave addressed_facet_ids empty for such background claims. "
        "A source-backed limitation may address the affected facet only by explicitly stating which requested "
        "items under that exact protocol cannot be supplied; it does not establish that those results exist. "
        "Do not reduce a missing experiment to a missing metric while implying its other requested items exist.",
        "Explicit material dependencies use the actual visible records, without invented upstream claims.",
        "Original question: " + question,
        "Unit and assigned requirements: " + json.dumps({"goal": unit.goal, "research_action": unit.research_action,
            "allowed_claim_types": unit.claim_policy.allowed_claim_types,
            "output_goal": unit.expected_output.output_goal, "output_role": unit.output_role,
            "validation_tasks": unit.expected_output.validation_tasks, "facets": facets}, ensure_ascii=False),
        "Evidence records:\n" + prepared.evidence_text,
        "Dependency claims:\n" + prepared.claims_text,
        "Visible ranges: " + prepared.exposure.model_dump_json(),
        render_read_reports(reports),
    ])


def _observe_exposure(records: list[dict[str, Any]], prepared: PreparedConsumerInput) -> None:
    """参数为权威材料及准备投影；分别观察实际原文范围与含截断说明的送模文本。"""
    registry = {str(record["evidence_id"]): record for record in records}
    for item in prepared.exposure.evidence:
        record = registry[item.evidence_id]
        body = str(record[item.field])
        visible = body[item.start:item.end]
        record_model_evidence(record, visible, rendered_text=render_exposed_text(body, item.start, item.end))
