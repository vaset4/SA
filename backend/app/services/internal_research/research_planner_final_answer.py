"""【研究交付】【最终校验】正文可读性、来源关系与实际交付分别判定。作者：xxx。"""

from __future__ import annotations

from typing import Any, Literal
from dataclasses import replace
from time import perf_counter

from pydantic import ValidationError

from app.services.internal_research.research_planner_final_blocks import (
    final_error, normalize_product_blocks, render_final_blocks, validate_final_blocks,
)
from app.services.internal_research.research_planner_final_contracts import (
    FINAL_CONTRACT_VERSION, FINAL_DRAFT_VERSION, REVISION_DELIVERY_POLICY_VERSION,
    FinalAnswerBlockDraft, FinalDeliveryReason, FinalDeliveryReport,
    FinalUnresolvedDraft, PlannerFinalAnswerDraft, PlannerFinalAnswerInput, PlannerFinalAnswerProvider,
    PlannerFinalErrorCode, PlannerFinalProviderOutput, PlannerFinalValidationError, PlannerFinalValidationOutcome,
)
from app.services.internal_research.research_planner_final_delivery import (
    evaluate_delivery_requirements, make_terminal_delivery_report, project_final_delivery,
)
from app.services.internal_research.research_planner_final_evidence import final_product_catalog
from app.services.internal_research.research_planner_final_provider import LLMPlannerFinalAnswerProvider
from app.services.llm_observation import record_business_validation

__all__ = ["validate_planner_final_answer", "PlannerFinalAnswerDraft", "PlannerFinalAnswerInput",
           "PlannerFinalAnswerProvider", "PlannerFinalValidationOutcome", "FinalAnswerBlockDraft",
           "PlannerFinalValidationError", "LLMPlannerFinalAnswerProvider"]
_AUX_FIELDS = frozenset(PlannerFinalAnswerDraft.model_fields)


def validate_planner_final_answer(payload: PlannerFinalAnswerInput, raw_draft: Any, *,
                                  provider_error: str = "") -> PlannerFinalValidationOutcome:
    """参数为实际可见输入、原始草稿与真实调用错误；返回同次正文、引用和交付事实。"""
    started = perf_counter()
    if provider_error:
        outcome = _terminal_outcome(payload, "failed", "provider_failed", provider_error, stage="provider")
    elif raw_draft is None:
        catalog = final_product_catalog(payload)
        if catalog.has_business_input:
            outcome = _terminal_outcome(payload, "failed", "final_provider_not_called",
                                        "已有可交付成果，但未取得本次最终模型输出", stage="generation")
        else:
            outcome = _terminal_outcome(payload, "insufficient", "insufficient_delivery",
                                        "尚未取得能回应本次要求的合法业务产物", stage="task_execution")
    else:
        outcome = _validate_output(payload, raw_draft)
    observed = raw_draft.payload if isinstance(raw_draft, PlannerFinalProviderOutput) else raw_draft
    schema = "PlannerFinalAnswerDraft"
    if isinstance(raw_draft, PlannerFinalProviderOutput) and raw_draft.source_schema:
        observed, schema = raw_draft.source_payload, raw_draft.source_schema
    record_business_validation(observed, {
        "validator": "planner_final", "schema": schema, "phase": "initial",
        "attempt_id": payload.attempt_id, "structure_valid": not outcome.errors,
        "presentation_valid": all(not row.required or row.status != "missing" for row in outcome.delivery_report.section_results),
        "semantic_validation": "not_assessed", "precheck_ms": (perf_counter() - started) * 1000,
        "validation_stage": "final_claim_evidence_references", "valid": outcome.status == "ok",
        "accepted": outcome.status == "ok", "status": outcome.status,
        "errors": [error.model_dump(mode="json") for error in outcome.errors],
        "accepted_blocks": [block.model_dump(mode="json") for block in outcome.blocks if block.valid],
        "delivery_status": outcome.delivery_report.status,
    })
    return outcome


def _validate_output(payload: PlannerFinalAnswerInput, raw: Any) -> PlannerFinalValidationOutcome:
    """参数为实际输入和模型原始结果；仅根载体不可用时终止解析，局部错误保留正文。"""
    if isinstance(raw, PlannerFinalProviderOutput):
        if raw.parsing_error:
            return _terminal_outcome(payload, "failed", "malformed_provider_output", raw.parsing_error,
                                     stage="contract", generation_completed=True)
        if raw.expected_contract_version and (not isinstance(raw.payload, dict)
                or raw.payload.get("contract_version") != raw.expected_contract_version):
            return _terminal_outcome(payload, "failed", "unsupported_contract_version",
                                     "模型返回的合同版本与本次请求不符", stage="contract", generation_completed=True)
        raw = raw.payload
    elif isinstance(raw, PlannerFinalAnswerDraft):
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, dict) or not isinstance(raw.get("blocks"), list):
        return _terminal_outcome(payload, "failed", "malformed_provider_output",
                                 "最终载体必须是含正文块列表的JSON对象", stage="contract", generation_completed=True)
    if raw.get("contract_version") not in {FINAL_CONTRACT_VERSION, FINAL_DRAFT_VERSION}:
        return _terminal_outcome(payload, "failed", "unsupported_contract_version",
                                 "最终载体缺少或使用不支持的合同版本", stage="contract", generation_completed=True)
    if not payload.exposure:
        return _terminal_outcome(payload, "failed", "final_exposure_missing",
                                 "最终输出缺少同次实际可见输入凭据", stage="contract", generation_completed=True)
    return _validate_readable_output(payload, raw)


def _validate_readable_output(payload: PlannerFinalAnswerInput, raw: dict[str, Any]) -> PlannerFinalValidationOutcome:
    """参数为根合同已通过的载体；派生产物、保留可读错误并组装同次交付事实。"""
    catalog = final_product_catalog(payload)
    raw_blocks = raw["blocks"]
    draft_errors: list[PlannerFinalValidationError] = []
    if raw["contract_version"] == FINAL_DRAFT_VERSION:
        payload = replace(payload, delivery_policy_version=REVISION_DELIVERY_POLICY_VERSION)
        raw_blocks, draft_errors = normalize_product_blocks(raw_blocks, catalog, attempt_id=payload.attempt_id)
    blocks, errors = validate_final_blocks(payload, raw_blocks, catalog)
    if draft_errors:
        errors.extend(draft_errors)
        blocks = [block.model_copy(update={"valid": False, "business_delivery": False,
                  "validation_errors": [*block.validation_errors, *[error for error in draft_errors
                                            if error.block_index == block.block_index]]})
                  if any(error.block_index == block.block_index for error in draft_errors) else block for block in blocks]
    confidence, unresolved, auxiliary_errors = _read_auxiliary(payload, raw)
    errors.extend(auxiliary_errors)
    answer, bindings, presentation = render_final_blocks(blocks, payload.attempt_id,
                                                        citable_evidence_ids=catalog.citable_evidence_ids)
    requirements, sections = evaluate_delivery_requirements(payload, blocks, unresolved=unresolved,
                                                           presentation=presentation)
    business_ids = [block.block_id for block in blocks if block.business_delivery]
    reasons = [reason for row in [*requirements, *sections] for reason in row.reasons]
    reasons.extend(unresolved)
    if not business_ids:
        errors.append(final_error("invalid_block", "模型已返回，但没有交付有效业务正文"))
        code = "empty_delivery" if not raw["blocks"] else "no_business_delivery"
        reasons.insert(0, FinalDeliveryReason(source="final_delivery", code=code,
                                              message="已有生成尝试却没有有效业务交付"))
    reasons.extend(FinalDeliveryReason(source="contract", code=error.code, message=error.message) for error in errors)
    required_missing = any(row.required and row.status != "delivered" for row in [*requirements, *sections])
    status = "complete"
    if not business_ids:
        status = "failed"
    elif errors or required_missing or any(reason.required for reason in unresolved):
        status = "partial"
    report = FinalDeliveryReport(delivery_policy_version=payload.delivery_policy_version,
             attempt_id=payload.attempt_id, entrypoint=payload.entrypoint, status=status,
             generation_status="completed", requirement_results=requirements, section_results=sections,
             validation_errors=errors, unresolved_items=unresolved, reasons=reasons,
             valid_business_block_ids=business_ids, citation_bindings=bindings, exposure=payload.exposure)
    projection = project_final_delivery(report)
    return PlannerFinalValidationOutcome(status=projection["planner_final_status"], answer=answer,
             confidence=confidence, failure_reason=reasons[0].message if reasons and status != "complete" else "",
             blocks=blocks, errors=errors, delivery_report=report)


def _read_auxiliary(payload: PlannerFinalAnswerInput, raw: dict[str, Any]
                    ) -> tuple[str, list[FinalDeliveryReason], list[PlannerFinalValidationError]]:
    """参数为载体和当前要求；保存未完成声明，辅助字段损坏不清空可读正文。"""
    errors: list[PlannerFinalValidationError] = []
    unresolved: list[FinalDeliveryReason] = []
    unknown = set(raw) - _AUX_FIELDS
    if unknown:
        errors.append(final_error("invalid_auxiliary_field", f"最终载体包含合同外字段：{sorted(unknown)}"))
    status = raw.get("status", "answered")
    if not isinstance(status, str) or status not in {"answered", "incomplete", "unable"}:
        errors.append(final_error("invalid_auxiliary_field", "最终完成声明非法", field="status"))
    confidence = raw.get("confidence", "")
    if not isinstance(confidence, str):
        confidence = ""
        errors.append(final_error("invalid_auxiliary_field", "置信说明必须是字符串", field="confidence"))
    reason = raw.get("failure_reason", "")
    if not isinstance(reason, str):
        reason = ""
        errors.append(final_error("invalid_auxiliary_field", "未完成说明必须是字符串", field="failure_reason"))
    items = raw.get("unresolved_items", [])
    if not isinstance(items, list):
        items = []
        errors.append(final_error("invalid_auxiliary_field", "未解决事项必须是列表", field="unresolved_items"))
    for item in items:
        try:
            declaration = FinalUnresolvedDraft.model_validate(item)
        except ValidationError as exc:
            errors.append(final_error("invalid_auxiliary_field", str(exc), field="unresolved_items"))
            continue
        declaration_reason, declaration_errors = _unresolved_reason(payload, declaration)
        unresolved.append(declaration_reason)
        errors.extend(declaration_errors)
    incomplete = isinstance(status, str) and status in {"incomplete", "unable"}
    if reason.strip() or (incomplete and not unresolved):
        unresolved.append(FinalDeliveryReason(source="model", code="model_reports_incomplete",
                           message=reason or "模型明确声明本次任务尚未完成",
                           required=not unresolved or any(item.required for item in unresolved)))
    if payload.historical_input_context:
        unresolved.append(FinalDeliveryReason(source="history", code="historical_support_not_assessed",
                           message="历史成果的精确支持或交付事实未按新版合同核验"))
    return confidence, unresolved, errors


def _unresolved_reason(payload: PlannerFinalAnswerInput, item: FinalUnresolvedDraft
                       ) -> tuple[FinalDeliveryReason, list[PlannerFinalValidationError]]:
    """参数为输入和模型未解决声明；必需性取自稳定要求，未知关联保留为全局未解决。"""
    facets = [*payload.plan.coverage_review.explicit_facets, *payload.plan.coverage_review.inferred_facets]
    required = {facet.facet_id: facet.required is not False for facet in facets}
    sections = {section.section_id: section.required for section in payload.plan.coverage_review.answer_sections}
    errors = []
    if item.facet_id and item.facet_id not in required:
        errors.append(final_error("unknown_facet_id", f"未解决声明关联了未知要求：{item.facet_id}"))
    if item.section_id and item.section_id not in sections:
        errors.append(final_error("unknown_section_id", f"未解决声明关联了未知章节：{item.section_id}"))
    if not item.reason.strip():
        errors.append(final_error("invalid_auxiliary_field", "未解决声明缺少具体原因", field="unresolved_items"))
    flags = [required[item.facet_id]] if item.facet_id in required else []
    if item.section_id in sections:
        flags.append(sections[item.section_id])
    return FinalDeliveryReason(source="model", code="model_requirement_unresolved", message=item.reason,
            requirement_id=item.facet_id, section_id=item.section_id, required=any(flags) if flags else True), errors


def _terminal_outcome(payload: PlannerFinalAnswerInput, status: Literal["failed", "insufficient"], code: str, reason: str, *,
                       stage: str, generation_completed: bool = False) -> PlannerFinalValidationOutcome:
    """参数为输入和真实失败/不足事实；返回有原因的终止报告，已有材料不会变成成功正文。"""
    report = make_terminal_delivery_report(status=status, reason=reason, reason_code=code, stage=stage,
              entrypoint=payload.entrypoint, attempt_id=payload.attempt_id, exposure=payload.exposure,
              evidence_not_found=status == "insufficient" and not payload.evidence_records)
    requirements, sections = evaluate_delivery_requirements(payload, [])
    report = report.model_copy(update={"requirement_results": requirements, "section_results": sections,
             "reasons": [*report.reasons, *(reason for row in [*requirements, *sections] for reason in row.reasons)]})
    errors = []
    if status == "failed":
        error_code: PlannerFinalErrorCode = "malformed_provider_output"
        if code == "provider_failed":
            error_code = "provider_failed"
        elif code == "unsupported_contract_version":
            error_code = "unsupported_contract_version"
        errors = [final_error(error_code, reason)]
        report = report.model_copy(update={"validation_errors": errors})
    if generation_completed:
        report = report.model_copy(update={"generation_status": "completed"})
    return PlannerFinalValidationOutcome(status=project_final_delivery(report)["planner_final_status"], answer="",
              confidence="low", failure_reason=reason, blocks=[], errors=errors, delivery_report=report)
