"""【研究规划】【可信转换】集中验证模型策略并生成唯一执行计划。作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.models.research_planner_contracts import (
    CURRENT_RESEARCH_CONTRACT_VERSION, PlannerCoverageReview, PlannerFailure, PlannerMetadata, PlannerRepairRecord,
    PlannerValidationIssue, PlannerValidationResult, ResearchPlan,
)
from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft, ResearchReplanStrategyDraft
from app.services.internal_research.research_planner_conversion import project_strategy_units
from app.services.internal_research.research_planner_failure_trace import failure_from_result
from app.services.internal_research.research_planner_validation_rules import (
    actual_plan_type, dependency_issues, fulfillment_contract_issues, issue, normalize_coverage,
    plan_shape_issues, scope_issues, strategy_permission_issues, user_presentation_issues,
)
from app.services.llm_observation import record_plan_validation

PlanStrategy = ResearchPlanStrategyDraft | ResearchReplanStrategyDraft


@dataclass(frozen=True, kw_only=True)
class PlannerValidationOptions:
    """【研究规划】【可信输入】问题、范围、用户原话与记忆资格均来自本轮服务端状态。"""

    question: str = ""
    user_texts: tuple[str, ...] = ()
    memory_grounded_allowed: bool = False
    allowed_paper_ids: frozenset[str] | None = None
    max_units: int = 8
    requirements_baseline: PlannerCoverageReview | None = None


def validate_or_repair_plan(
    draft: PlanStrategy | dict[str, Any] | str, *, options: PlannerValidationOptions | None = None,
    repair_draft: PlanStrategy | dict[str, Any] | str | None = None,
) -> PlannerValidationResult:
    """参数为原策略、可信输入及可选明确回修；返回计划或原始失败，不自动改写策略。"""
    opts = options or PlannerValidationOptions()
    schema = ResearchReplanStrategyDraft if opts.requirements_baseline is not None else ResearchPlanStrategyDraft
    first = _validate_once(draft, opts, repair_attempted=False)
    record_plan_validation(draft, first, phase="initial", schema=schema.__name__)
    if first.is_valid:
        return first
    if repair_draft is None:
        return failure_from_result(first)
    repaired = _validate_once(repair_draft, opts, repair_attempted=True)
    record_plan_validation(repair_draft, repaired, phase="explicit_repair", schema=schema.__name__)
    return repaired if repaired.is_valid else failure_from_result(repaired, original=first)


def _parse_draft(raw: Any, *, replan: bool) -> tuple[PlanStrategy | None, list[PlannerValidationIssue]]:
    """参数为原始策略与是否重规划；严格解析完整数据，返回带字段路径的格式错误。"""
    schema = ResearchReplanStrategyDraft if replan else ResearchPlanStrategyDraft
    try:
        if isinstance(raw, str):
            return schema.model_validate_json(raw, strict=True), []
        payload = raw.model_dump(mode="python") if isinstance(raw, (ResearchPlanStrategyDraft, ResearchReplanStrategyDraft)) else raw
        return schema.model_validate(payload, strict=True), []
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False)
        return None, [issue("invalid_plan_format", str(exc), details={"field_errors": errors,
                                                                   "failure_stage": "model_schema"})]


def _validate_once(raw: Any, opts: PlannerValidationOptions, *, repair_attempted: bool) -> PlannerValidationResult:
    """参数为策略与尝试身份；保留程序转换异常的独立归因，异常仍原样抛给调用方。"""
    try:
        return _validate_strategy(raw, opts, repair_attempted=repair_attempted)
    except Exception as exc:
        failure = PlannerFailure(code="planner_conversion_error", message=str(exc), trace={
            "failure_stage": "program_conversion", "error_type": type(exc).__name__,
            "repair_attempted": repair_attempted,
        })
        record_plan_validation(raw, PlannerValidationResult(failure=failure),
            phase="explicit_repair" if repair_attempted else "initial",
            schema=(ResearchReplanStrategyDraft if opts.requirements_baseline is not None else ResearchPlanStrategyDraft).__name__)
        raise


def _validate_strategy(raw: Any, opts: PlannerValidationOptions, *, repair_attempted: bool) -> PlannerValidationResult:
    """参数为策略与可信输入；验证来源、依赖和要求后投影，最终正式模型再次校验。"""
    draft, issues = _parse_draft(raw, replan=opts.requirements_baseline is not None)
    if draft is None:
        return _invalid_result(issues, repair_attempted)
    if not opts.question.strip():
        return _invalid_result([issue("missing_trusted_question", "planning requires the current user question")], repair_attempted)
    if not draft.units:
        return _invalid_result([issue("empty_plan", "clarification is not an executable plan")], repair_attempted)
    original_ids = [unit.unit_id for unit in draft.units]
    if len(set(original_ids)) != len(original_ids):
        return _invalid_result([issue("duplicate_unit_id", "unit ids must be unique before projection")], repair_attempted)
    review = _requirements(draft, opts.requirements_baseline)
    # 1. 【研究规划】【策略许可】先验证模型选择，来源矛盾不能被转换成别的意图
    issues.extend(strategy_permission_issues(draft.units, memory_allowed=opts.memory_grounded_allowed))
    issues.extend(user_presentation_issues(review, (opts.question, *opts.user_texts)))
    if issues:
        return _invalid_result(issues, repair_attempted)
    units, id_map = project_strategy_units(draft.units, review)
    issues.extend(plan_shape_issues(units, max_units=opts.max_units))
    issues.extend(scope_issues(units, opts.allowed_paper_ids))
    issues.extend(dependency_issues(units))
    coverage, coverage_issues = normalize_coverage(review, units)
    issues.extend(coverage_issues)
    issues.extend(fulfillment_contract_issues(review, units, baseline=opts.requirements_baseline,
                                             allowed_paper_ids=opts.allowed_paper_ids))
    if issues:
        return _invalid_result(issues, repair_attempted)
    metadata = PlannerMetadata(normalized_unit_id_map=id_map, repair_attempted=repair_attempted,
        repaired=repair_attempted, repairs=[PlannerRepairRecord(code="repair_draft_used",
            message="explicit repair draft supplied")] if repair_attempted else [])
    # 2. 【研究规划】【执行投影】身份和覆盖派生后重新经过正式合同，不用 model_copy 越过校验
    plan = ResearchPlan.model_validate({
        "question": opts.question, "plan_type": actual_plan_type(units),
        "units": [unit.model_dump(mode="python") for unit in units],
        "coverage_review": coverage.model_dump(mode="python"), "metadata": metadata.model_dump(mode="python"),
        "contract_version": CURRENT_RESEARCH_CONTRACT_VERSION,
    }, strict=True)
    return PlannerValidationResult(plan=plan)


def _requirements(draft: PlanStrategy, baseline: PlannerCoverageReview | None) -> PlannerCoverageReview:
    """参数为策略与已接受要求；返回首次要求或原封不动的重规划基准。"""
    if baseline is not None:
        return baseline
    if not isinstance(draft, ResearchPlanStrategyDraft):
        raise TypeError("initial planning requires requested facets")
    return PlannerCoverageReview.model_validate({
        "explicit_facets": [facet.model_dump() for facet in draft.requested_facets if facet.source == "explicit"],
        "inferred_facets": [facet.model_dump() for facet in draft.requested_facets if facet.source == "inferred"],
        "answer_sections": [section.model_dump() for section in draft.answer_sections],
    }, strict=True)


def _invalid_result(issues: list[PlannerValidationIssue], repair_attempted: bool) -> PlannerValidationResult:
    """参数为真实错误及回修身份；返回可追溯的显式失败，不丢弃原始字段路径。"""
    trace = PlannerMetadata(validation_errors=issues, repair_attempted=repair_attempted,
        repairs=[PlannerRepairRecord(code="repair_draft_used", message="explicit repair draft supplied")]
        if repair_attempted else []).model_dump(mode="json")
    trace["failure_stage"] = "model_schema" if issues[0].code == "invalid_plan_format" else "plan_semantics"
    return PlannerValidationResult(failure=PlannerFailure(code=issues[0].code, message=issues[0].message, trace=trace))
