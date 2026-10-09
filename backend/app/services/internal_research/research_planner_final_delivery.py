"""【研究交付】【报告投影】要求交付验收与外层状态的唯一映射。作者：xxx。"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence, TypedDict
from uuid import uuid4

from pydantic import ValidationError

from app.services.internal_research.research_planner_final_contracts import (
    DELIVERY_POLICY_VERSION, REVISION_DELIVERY_POLICY_VERSION,
    FinalAnswerBlock, FinalDeliveryReason, FinalDeliveryReport, FinalRequirementDelivery,
    FinalDeliveryStatus, FinalSectionPresentation, PlannerFinalAnswerInput, PlannerFinalStatus,
)
from app.models.research_planner_contracts import AnswerSectionRequirement, RequestedFacet
from app.services.internal_research.research_planner_fulfillment import evaluate_delivered_facet

_DELIVERY_PROJECTIONS: dict[FinalDeliveryStatus, tuple[str, str, PlannerFinalStatus]] = {
    "complete": ("completed", "answered", "ok"),
    "partial": ("incomplete", "partial_answer", "partial"),
    "insufficient": ("incomplete", "insufficient_answer", "skipped"),
    "failed": ("failed", "planner_failed", "failed"),
}


class FinalDeliveryProjection(TypedDict):
    """【研究交付】【投影合同】保留最终状态的封闭词表，调用方无需再次猜测或转换。"""

    turn_status: str
    chat_result_status: str
    planner_final_status: PlannerFinalStatus
    termination_reason: str


def project_delivery_findings(report: FinalDeliveryReport | Mapping[str, Any]) -> dict[str, Any]:
    """参数为同次权威交付报告；返回模型修正所需的真实缺口，不重新计算要求或主张类型满足。"""
    data = report.model_dump(mode="json") if isinstance(report, FinalDeliveryReport) else report
    findings = {key: data[key] for key in ("status", "requirement_results", "section_results", "validation_errors",
                                         "unresolved_items", "reasons") if key in data}
    # 1. 【研究输入】【原因去重】逐要求与未解决目录已给出的同一原因不在总表重复，独有错误仍保留
    represented = [reason for row in [*data.get("requirement_results", []), *data.get("section_results", [])]
                   for reason in row.get("reasons", [])]
    represented.extend(data.get("unresolved_items", []))
    if "reasons" in findings:
        findings["reasons"] = [reason for reason in findings["reasons"] if reason not in represented]
    return findings


def make_terminal_delivery_report(
    *, status: Literal["failed", "insufficient"], reason: str, reason_code: str, stage: str,
    entrypoint: str, attempt_id: str = "", evidence_not_found: bool = False,
    exposure: dict[str, Any] | None = None,
) -> FinalDeliveryReport:
    """参数为真实终止状态、原因和入口身份；返回未伪造业务交付的终止报告。"""
    if not reason.strip() or not reason_code.strip():
        raise ValueError("terminal delivery requires an explicit reason and reason code")
    return FinalDeliveryReport(
        delivery_policy_version=DELIVERY_POLICY_VERSION,
        attempt_id=attempt_id or uuid4().hex, entrypoint=entrypoint, status=status,
        generation_status="failed" if stage == "provider" else "not_called",
        reasons=[FinalDeliveryReason(source=stage, code=reason_code, message=reason)],
        evidence_not_found=evidence_not_found, exposure=dict(exposure or {}),
    )


def project_final_delivery(report: FinalDeliveryReport | Mapping[str, Any]) -> FinalDeliveryProjection:
    """参数为同次交付报告；返回回合、聊天、最终生成状态和明确终止原因。"""
    current = report if isinstance(report, FinalDeliveryReport) else FinalDeliveryReport.model_validate(dict(report))
    turn_status, chat_status, final_status = _DELIVERY_PROJECTIONS[current.status]
    if current.status == "insufficient" and current.evidence_not_found:
        chat_status = "evidence_not_found"
    return {
        "turn_status": turn_status, "chat_result_status": chat_status, "planner_final_status": final_status,
        "termination_reason": current.reasons[0].code if current.reasons else f"delivery_{current.status}",
    }


def delivery_report_from_state(state: Mapping[str, Any]) -> FinalDeliveryReport | None:
    """参数为实际入口状态；返回已校验报告，新研究缺报或报告损坏时返回明确合同失败。"""
    raw = state.get("final_delivery_report")
    if raw is not None:
        try:
            return raw if isinstance(raw, FinalDeliveryReport) else FinalDeliveryReport.model_validate(raw)
        except ValidationError as exc:
            return make_terminal_delivery_report(
                status="failed", reason=str(exc), reason_code="invalid_delivery_report",
                stage="contract", entrypoint="research",
            )
    if state.get("final_delivery_applicability") == "research":
        return make_terminal_delivery_report(
            status="failed", reason="研究交付出口缺少本次交付报告", reason_code="missing_delivery_report",
            stage="contract", entrypoint="research",
        )
    return None


def delivery_state_fields(report: FinalDeliveryReport) -> dict[str, Any]:
    """参数为本次已生成报告；返回同次消息/重试字段，不从历史版本补拼缺项。"""
    return {
        "final_delivery_report": report.model_dump(mode="json"),
        "final_citation_bindings": [item.model_dump(mode="json") for item in report.citation_bindings],
        "final_generation_attempt": {"attempt_id": report.attempt_id, "entrypoint": report.entrypoint,
                                     "generation_status": report.generation_status, "exposure": report.exposure},
        "final_delivery_applicability": "research",
    }


def evaluate_delivery_requirements(payload: PlannerFinalAnswerInput, blocks: list[FinalAnswerBlock], *,
                                   unresolved: Sequence[FinalDeliveryReason] = (),
                                   presentation: FinalSectionPresentation | None = None,
                                   ) -> tuple[list[FinalRequirementDelivery], list[FinalRequirementDelivery]]:
    """参数为同次输入和全部正文；只从有效业务块验收要求与章节，不使用全材料池冒充交付。"""
    facets = [*payload.plan.coverage_review.explicit_facets, *payload.plan.coverage_review.inferred_facets]
    requirements = _apply_model_unresolved([_facet_delivery(payload, facet, blocks) for facet in facets],
                                          unresolved, section=False)
    sections = [_section_delivery(payload, section, blocks, requirements=requirements,
                presentation=presentation or {}, unresolved=unresolved)
                for section in payload.plan.coverage_review.answer_sections]
    if payload.delivery_policy_version == REVISION_DELIVERY_POLICY_VERSION:
        return requirements, sections
    return requirements, _apply_model_unresolved(sections, unresolved, section=True)


def _section_delivery(payload: PlannerFinalAnswerInput, section: AnswerSectionRequirement,
                      blocks: list[FinalAnswerBlock], *, requirements: list[FinalRequirementDelivery],
                      presentation: FinalSectionPresentation, unresolved: Sequence[FinalDeliveryReason]
                      ) -> FinalRequirementDelivery:
    """参数为章节、真实正文与内容验收；返回内部追踪、明确展示义务或历史未知，不新增内容要求。"""
    if section.presentation_kind is None:
        return FinalRequirementDelivery(requirement_id=section.section_id, required=section.required,
            status="not_assessed", reasons=[FinalDeliveryReason(source="history",
                code="historical_presentation_not_assessed", message="历史计划未保存章节的展示义务来源",
                section_id=section.section_id, required=section.required)])
    # 1. 【研究交付】【内部组织】自拟章节只追踪对应内容，标题和归组不增加完成义务
    if section.presentation_kind == "internal":
        rows = [row for row in requirements if row.requirement_id in section.facet_ids]
        return FinalRequirementDelivery(requirement_id=section.section_id, required=False,
            status="delivered" if all(row.status == "delivered" for row in rows) else "missing",
            block_ids=sorted({bid for row in rows for bid in row.block_ids}))
    # 2. 【研究交付】【用户展示】只认可同次最终Markdown内真实标题下、与该章节关联的有效正文
    headings = presentation.get(section.section_id, {})
    indices = {index for title, body in headings.items() for index in body
               if section.presentation_kind == "user_section" or title == section.title.strip()}
    selected = [block for block in blocks if block.block_index in indices]
    if payload.delivery_policy_version == REVISION_DELIVERY_POLICY_VERSION:
        # 3. 【研究交付】【展示与结果】出处导航和不足说明可以完成展示，内容和来源语义仍独立验收
        return FinalRequirementDelivery(requirement_id=section.section_id, required=section.required,
            status="delivered" if selected else "missing", block_ids=[block.block_id for block in selected],
            reasons=[] if selected else [FinalDeliveryReason(source="final_delivery",
                code="user_presentation_not_delivered", message=f"用户分节尚缺真实标题或其下同节有效正文：{section.title}",
                section_id=section.section_id, required=section.required)])
    # 4. 【研究交付】【历史政策】旧版仍只接受业务正文，保留原有缺失状态与错误原因
    selected = [block for block in selected if block.business_delivery]
    facets = [*payload.plan.coverage_review.explicit_facets, *payload.plan.coverage_review.inferred_facets]
    rows = [_facet_delivery(payload, facet, selected) for facet in facets if facet.facet_id in section.facet_ids]
    rows = _apply_model_unresolved(rows, unresolved, section=False)
    complete = bool(selected) and all(not row.required or row.status == "delivered" for row in rows)
    code = "section_content_not_delivered" if selected else "user_presentation_not_delivered"
    message = f"用户要求的章节尚缺实质内容：{section.title}" if selected else f"用户要求的分节或标题尚未交付：{section.title}"
    reasons = [] if complete else [FinalDeliveryReason(source="final_delivery", code=code,
        message=message, section_id=section.section_id, required=section.required)]
    return FinalRequirementDelivery(requirement_id=section.section_id, required=section.required,
        status="delivered" if complete else "missing", block_ids=[block.block_id for block in selected], reasons=reasons)


def _apply_model_unresolved(rows: list[FinalRequirementDelivery], unresolved: Sequence[FinalDeliveryReason], *,
                            section: bool) -> list[FinalRequirementDelivery]:
    """参数为形式验收与模型声明；返回保留冲突和未解决事实的新验收记录，不覆盖声明。"""
    evaluated = []
    for row in rows:
        declarations = [reason for reason in unresolved
                        if (reason.section_id if section else reason.requirement_id) == row.requirement_id]
        if not declarations:
            evaluated.append(row)
            continue
        conflicts = []
        if row.status == "delivered":
            conflicts = [FinalDeliveryReason(source="model", code="model_delivery_conflict",
                         message="形式产物已覆盖，但模型明确声明该项未完成", required=row.required,
                         requirement_id="" if section else row.requirement_id,
                         section_id=row.requirement_id if section else "")]
        evaluated.append(row.model_copy(update={"status": "missing",
                         "reasons": [*row.reasons, *declarations, *conflicts]}))
    return evaluated


def _facet_delivery(payload: PlannerFinalAnswerInput, facet: RequestedFacet,
                     blocks: list[FinalAnswerBlock]) -> FinalRequirementDelivery:
    """参数为输入、单项要求和候选正文；以共享满足规则区分未取得与最终未交付。"""
    selected = [block for block in blocks if block.business_delivery and facet.facet_id in block.facet_ids]
    claim_ids = sorted({cid for block in selected for cid in block.claim_ids})
    evidence_ids = sorted({eid for block in selected for eid in block.evidence_ids})
    if payload.historical_input_context:
        return FinalRequirementDelivery(requirement_id=facet.facet_id, required=facet.required is not False,
                    status="not_assessed", block_ids=[block.block_id for block in selected],
                    claim_ids=claim_ids, evidence_ids=evidence_ids,
                    reasons=[FinalDeliveryReason(source="history", code="historical_requirement_not_assessed",
                             message="历史成果未保存新版要求及支持核验事实", requirement_id=facet.facet_id,
                             required=facet.required is not False)])
    observed = evaluate_delivered_facet(payload.plan, facet, payload.unit_results, payload.evidence_records,
                                        claim_ids=claim_ids, evidence_ids=evidence_ids)
    reasons: list[FinalDeliveryReason] = []
    if not observed.satisfied or not selected:
        acquired = evaluate_delivered_facet(
            payload.plan, facet, payload.unit_results, payload.evidence_records,
            claim_ids=[claim.claim_id for result in payload.unit_results for claim in result.claims],
            evidence_ids=[str(record["evidence_id"]) for record in payload.evidence_records],
        )
        reasons.append(FinalDeliveryReason(
            source="final_delivery" if acquired.satisfied else "task_execution",
            code="requirement_not_delivered" if acquired.satisfied else "required_product_missing",
            message=f"尚未完整回答：{facet.label}", requirement_id=facet.facet_id, required=facet.required is True,
        ))
    return FinalRequirementDelivery(requirement_id=facet.facet_id, required=facet.required is True,
               status="delivered" if observed.satisfied and selected else "missing",
               block_ids=[block.block_id for block in selected], claim_ids=claim_ids,
               evidence_ids=evidence_ids, reasons=reasons)
