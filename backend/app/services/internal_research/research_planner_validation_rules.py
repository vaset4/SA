"""Deterministic rule helpers for research planner validation."""

from __future__ import annotations

from typing import Any

from app.models.research_planner_contracts import (
    PlannerCoverageReview,
    PlannerRepairRecord,
    PlannerValidationIssue,
    ResearchUnit,
)
from app.models.research_planner_draft_contracts import ResearchUnitStrategyDraft
from app.services.internal_research.research_planner_fulfillment import declared_paper_scope, requirements_version

def plan_shape_issues(
    units: list[ResearchUnit],
    *,
    max_units: int,
) -> list[PlannerValidationIssue]:
    """参数为计划草案、规范化单元和数量限制；返回单元形状及计划类型不一致的问题。"""
    issues: list[PlannerValidationIssue] = []
    seen: set[tuple[str, str, str]] = set()
    # 1. 【研究规划】【数量准入】与增量动作共用0表示不限数量的配置语义
    if max_units != 0 and len(units) > max_units:
        issues.append(issue("too_many_units", f"plan has {len(units)} units"))
    for unit in units:
        issues.extend(unit_shape_issues(unit, seen))
    return issues


def scope_issues(
    units: list[ResearchUnit],
    allowed_paper_ids: frozenset[str] | None,
) -> list[PlannerValidationIssue]:
    if allowed_paper_ids is None:
        return []
    issues: list[PlannerValidationIssue] = []
    for unit in units:
        out_of_scope = [pid for pid in unit.evidence_policy.target_paper_ids if pid not in allowed_paper_ids]
        if out_of_scope:
            issues.append(issue("out_of_scope_paper_id", "unit targets paper outside resolved scope", unit.unit_id))
    return issues


def dependency_issues(units: list[ResearchUnit]) -> list[PlannerValidationIssue]:
    """参数为执行单元；返回未知上游、空推断依赖及环路，绝不删除错误依赖。"""
    unit_ids = {unit.unit_id for unit in units}
    issues: list[PlannerValidationIssue] = []
    for unit in units:
        if unit.evidence_policy.source_type == "inference_from_units" and not unit.dependencies:
            issues.append(issue("missing_inference_dependencies", "inference requires actual dependency declarations", unit.unit_id))
        for dep in unit.dependencies:
            if dep.unit_id not in unit_ids:
                issues.append(issue("bad_dependency_id", f"unknown dependency {dep.unit_id}", unit.unit_id))
    if has_dependency_cycle(units):
        issues.append(issue("cyclic_dependency", "unit dependencies contain a cycle"))
    return issues


def strategy_permission_issues(
    units: list[ResearchUnitStrategyDraft], *, memory_allowed: bool,
) -> list[PlannerValidationIssue]:
    """参数为模型选择和真实记忆资格；返回来源能力或上游引用矛盾，不收紧或替换模型策略。"""
    issues: list[PlannerValidationIssue] = []
    ids = {unit.unit_id for unit in units}
    for unit in units:
        policy = unit.evidence_policy
        kinds = {intent.claim_type for intent in unit.claim_intents}
        if "paper_supported" in kinds and not policy.can_produce_citations:
            issues.append(issue("claim_source_not_citable", "paper_supported requires a citable source", unit.unit_id))
        if policy.source_type == "external_context_needed" and kinds - {"external_needed"}:
            issues.append(issue("external_claim_policy_conflict", "external source only supports external_needed", unit.unit_id))
        if policy.source_type != "external_context_needed" and "external_needed" in kinds:
            issues.append(issue("external_intent_source_conflict", "external_needed requires explicit external source", unit.unit_id))
        if "memory_continuation" in kinds and not memory_allowed:
            issues.append(issue("memory_intent_without_memory", "memory_continuation requires available project memory", unit.unit_id))
        if policy.evidence_granularity == "parent_chunk" and policy.source_type != "local_direct_text":
            issues.append(issue("evidence_granularity_not_supported_by_source", "parent context requires direct text retrieval", unit.unit_id))
        if policy.verify_with_direct_text and policy.source_type != "local_structure_hint":
            issues.append(issue("verify_with_direct_text_not_supported_by_source", "text verification requires structure hint source", unit.unit_id))
        for dep in unit.dependencies:
            if dep.unit_id not in ids:
                issues.append(issue("bad_dependency_id", f"unknown dependency {dep.unit_id}", unit.unit_id))
    return issues


def user_presentation_issues(
    review: PlannerCoverageReview, user_texts: tuple[str, ...],
) -> list[PlannerValidationIssue]:
    """参数为要求基准与可信用户输入；只核验原文存在，不冒称验证模型理解正确。"""
    issues: list[PlannerValidationIssue] = []
    for facet in review.explicit_facets:
        # 1. 【研究规划】【原话回修】分别说明空缺与不匹配，保留同一校验边界和错误码
        span = facet.user_text_span
        if not span.strip() or not any(span in text for text in user_texts):
            reason = "is empty" if not span.strip() else "does not match user text"
            issues.append(issue("unverified_facet_basis", (
                f"facet {facet.facet_id}.user_text_span {reason}; copy a nonempty contiguous span from "
                "visible Question or user history, preserving whitespace; do not use assistant text or excerpt markers"
            )))
    for section in review.answer_sections:
        if section.presentation_kind is None:
            issues.append(issue("missing_presentation_kind", f"section {section.section_id} lacks live presentation semantics"))
            continue
        if section.presentation_kind == "internal":
            continue
        span = section.user_text_span
        if not span.strip() or not any(span in text for text in user_texts):
            issues.append(issue("unverified_presentation_basis", f"section {section.section_id} has no matching user text"))
        if section.presentation_kind == "exact_heading" and section.title not in span:
            issues.append(issue("unverified_exact_heading", f"section {section.section_id} heading is absent from its user span"))
        if not section.required:
            issues.append(issue("user_presentation_not_required", f"section {section.section_id} user presentation must remain required"))
    return issues


def normalize_coverage(
    review: PlannerCoverageReview,
    units: list[ResearchUnit],
    *, require_complete: bool = True,
) -> tuple[PlannerCoverageReview, list[PlannerValidationIssue]]:
    facets = {facet.facet_id: facet for facet in [*review.explicit_facets, *review.inferred_facets]}
    unit_map = {unit.unit_id: list(unit.covered_facet_ids) for unit in units}
    covered = {facet_id for ids in unit_map.values() for facet_id in ids}
    invalid = sorted(facet_id for facet_id in covered if facet_id not in facets)
    issues = [issue("invalid_facet_id", f"unknown covered facet {facet_id}") for facet_id in invalid]
    missing_explicit = sorted({facet.facet_id for facet in review.explicit_facets if facet.required} - covered)
    if missing_explicit and require_complete:
        issues.append(issue(
            "explicit_facet_uncovered",
            f"explicit requested facets are uncovered: {', '.join(missing_explicit)}",
            details={"missing_explicit_facet_ids": missing_explicit},
        ))
    normalized = PlannerCoverageReview.model_validate({**review.model_dump(),
        "covered_facet_ids": sorted(covered & set(facets)),
        "missing_explicit_facet_ids": missing_explicit,
        "missing_inferred_facet_ids": sorted({facet.facet_id for facet in review.inferred_facets} - covered),
        "unit_facet_map": unit_map,
        "repair_status": coverage_repair_status(review.repair_status, missing_explicit),
    }, strict=True)
    return normalized, issues


def fulfillment_contract_issues(
    review: PlannerCoverageReview,
    units: list[ResearchUnit],
    *,
    baseline: PlannerCoverageReview | None = None,
    allowed_paper_ids: frozenset[str] | None = None,
    require_complete: bool = True,
) -> list[PlannerValidationIssue]:
    """参数为归一后的要求和单元及稳定基准；返回声明上不可能完成或降低要求的合同错误。"""
    issues: list[PlannerValidationIssue] = []
    by_id = {unit.unit_id: unit for unit in units}
    for unit in units:
        kind = unit.expected_output.deliverable_kind
        required = set(unit.expected_output.required_claim_types)
        if kind is None:
            issues.append(issue("missing_deliverable_kind", "live unit must declare deliverable_kind", unit.unit_id))
        if kind == "materials" and required:
            issues.append(issue("material_output_requires_claims", "material output cannot require claims", unit.unit_id))
        if not required <= set(unit.claim_policy.allowed_claim_types):
            issues.append(issue("unsatisfiable_claim_output", "required claim types exceed final source policy", unit.unit_id))
        if require_complete:
            issues.extend(_dependency_product_issues(unit, by_id))
    facets = [*review.explicit_facets, *review.inferred_facets]
    if len({facet.facet_id for facet in facets}) != len(facets):
        issues.append(issue("duplicate_facet_id", "facet ids must be unique"))
    for facet in facets:
        issues.extend(_facet_contract_issues(facet, units, by_id, allowed_paper_ids, require_complete=require_complete))
    issues.extend(_section_contract_issues(review, units))
    if baseline is not None and requirements_version(review) != requirements_version(baseline):
        issues.append(issue("requirements_baseline_changed", "replanning cannot replace or lower required facets"))
    return issues


def _dependency_product_issues(
    unit: ResearchUnit, by_id: dict[str, ResearchUnit],
) -> list[PlannerValidationIssue]:
    """参数为消费单元和上游声明；返回所需产物/类型未被上游承诺的错误。"""
    issues: list[PlannerValidationIssue] = []
    for dep in unit.dependencies:
        if dep.required_product is None:
            issues.append(issue("missing_dependency_product", "live dependency must declare required_product", unit.unit_id))
            continue
        upstream = by_id.get(dep.unit_id)
        if upstream is None:
            continue
        required = set(dep.required_claim_types)
        if dep.required_product == "materials":
            valid = not required and upstream.evidence_policy.source_type not in {"inference_from_units", "external_context_needed"}
        else:
            allowed = {"model_prior_hypothesis", "memory_continuation"} if dep.required_product == "hypotheses" else {
                "paper_supported", "evidence_grounded_inference",
            }
            promised = set(upstream.expected_output.required_claim_types)
            valid = upstream.expected_output.deliverable_kind == "claims" and bool(promised & allowed)
            valid = valid and required <= promised and required <= allowed
        if not valid:
            issues.append(issue("unsatisfiable_dependency_product", f"dependency {dep.unit_id} cannot promise requested product",
                                unit.unit_id))
    return issues


def _facet_contract_issues(
    facet: Any, units: list[ResearchUnit], by_id: dict[str, ResearchUnit],
    allowed_paper_ids: frozenset[str] | None,
    *, require_complete: bool = True,
) -> list[PlannerValidationIssue]:
    """参数为要求与计划范围；返回必需性、对象和多单元联合能力不能兑现的错误。"""
    issues = _facet_requirement_issues(facet, allowed_paper_ids)
    # 1. 【研究循环】【要求准入】当前读取不承诺未来产物；实际覆盖在交付时统一检查
    if not require_complete:
        return issues
    targets = set(facet.required_paper_ids)
    owners = [unit for unit in units if facet.facet_id in unit.covered_facet_ids]
    if facet.required and not owners:
        issues.append(issue("required_facet_uncovered", f"required facet {facet.facet_id} is uncovered"))
    scopes = [declared_paper_scope(unit, by_id) for unit in owners]
    covered = set().union(*scopes) if scopes else set()
    if targets and scopes and all(scopes) and not targets <= covered:
        issues.append(issue("facet_objects_uncovered", f"facet {facet.facet_id} objects lack declared source coverage"))
    if facet.deliverable_kind == "claims":
        claim_owners = [unit for unit in owners if unit.expected_output.deliverable_kind == "claims"
                        and (facet.required_output_role is None or unit.output_role == facet.required_output_role)]
        promised = {kind for unit in claim_owners for kind in unit.expected_output.required_claim_types}
        if facet.required and (not claim_owners or not set(facet.required_claim_types) <= promised):
            issues.append(issue("facet_output_uncovered", f"facet {facet.facet_id} claim requirements lack promised outputs"))
        issues.extend(_facet_object_output_issues(facet, claim_owners, by_id))
    return issues



def _facet_requirement_issues(facet: Any, allowed_paper_ids: frozenset[str] | None) -> list[PlannerValidationIssue]:
    """参数为用户要求与权限范围；校验要求自身，不要求未来行动提前承诺产物。"""
    issues: list[PlannerValidationIssue] = []
    if facet.required is None or facet.deliverable_kind is None:
        issues.append(issue("missing_facet_contract", f"facet {facet.facet_id} must declare required and deliverable_kind"))
    if facet.source == "explicit" and facet.required is not True:
        issues.append(issue("explicit_facet_not_required", f"explicit facet {facet.facet_id} must be required"))
    if facet.required and facet.source == "inferred" and not facet.rationale:
        issues.append(issue("required_inference_without_basis", f"required facet {facet.facet_id} needs rationale"))
    if facet.deliverable_kind == "materials" and facet.required_claim_types:
        issues.append(issue("material_facet_requires_claims", f"material facet {facet.facet_id} cannot require claims"))
    if facet.required_output_role is not None and (facet.required_output_role != "final_recommendation"
                                                  or facet.deliverable_kind != "claims"):
        issues.append(issue("invalid_facet_role", f"facet {facet.facet_id} role obligation requires recommendation claims"))
    targets = set(facet.required_paper_ids)
    if not set(facet.required_paper_counts) <= targets:
        issues.append(issue("facet_count_object_undeclared", f"facet {facet.facet_id} quantity objects must be required objects"))
    if allowed_paper_ids is not None and not targets <= allowed_paper_ids:
        issues.append(issue("out_of_scope_facet_object", f"facet {facet.facet_id} objects exceed resolved scope"))
    return issues


def _section_contract_issues(review: PlannerCoverageReview, units: list[ResearchUnit]) -> list[PlannerValidationIssue]:
    """参数为唯一章节要求及单元引用；拒绝双重真相、无依据必需章节和未知关联。"""
    issues: list[PlannerValidationIssue] = []
    sections = {section.section_id: section for section in review.answer_sections}
    facets = {facet.facet_id for facet in [*review.explicit_facets, *review.inferred_facets]}
    if len(sections) != len(review.answer_sections):
        issues.append(issue("duplicate_answer_section_id", "answer section identities must be unique"))
    for section in review.answer_sections:
        if not set(section.facet_ids) <= facets or (section.required and not section.facet_ids):
            issues.append(issue("invalid_answer_section_facets", f"section {section.section_id} must link declared facets"))
        if section.presentation_kind != "internal" and section.source == "explicit" and not section.required:
            issues.append(issue("explicit_section_not_required", f"explicit section {section.section_id} must be required"))
        if section.presentation_kind != "internal" and section.required and section.source == "inferred" and not section.rationale:
            issues.append(issue("required_section_without_basis", f"section {section.section_id} needs task rationale"))
    for unit in units:
        if unit.expected_output.required_answer_sections:
            issues.append(issue("legacy_answer_sections_in_live_plan", "live sections belong to coverage_review", unit.unit_id))
        if not set(unit.expected_output.answer_section_ids) <= set(sections):
            issues.append(issue("unknown_answer_section_id", "unit references undeclared answer section", unit.unit_id))
    return issues


def _facet_object_output_issues(
    facet: Any, claim_owners: list[ResearchUnit], by_id: dict[str, ResearchUnit],
) -> list[PlannerValidationIssue]:
    """参数为主张要求及认领单元；验证每个必需对象都有承诺相应产物类型的来源路径。"""
    issues: list[PlannerValidationIssue] = []
    for paper_id in facet.required_paper_ids:
        owners = [unit for unit in claim_owners if not declared_paper_scope(unit, by_id)
                  or paper_id in declared_paper_scope(unit, by_id)]
        promised = {kind for unit in owners for kind in unit.expected_output.required_claim_types}
        if not owners or not set(facet.required_claim_types) <= promised:
            issues.append(issue("facet_object_output_uncovered", f"facet {facet.facet_id} lacks promised output for {paper_id}"))
    return issues


def actual_plan_type(units: list[ResearchUnit]) -> str:
    return "single_unit" if len(units) == 1 else "multi_unit"


def issue(
    code: str,
    message: str,
    unit_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> PlannerValidationIssue:
    return PlannerValidationIssue(code=code, message=message, unit_id=unit_id, details=details or {})


def repair_record(
    code: str,
    message: str,
    unit_id: str | None = None,
    original: Any = None,
    repaired: Any = None,
) -> PlannerRepairRecord:
    return PlannerRepairRecord(
        code=code,
        message=message,
        unit_id=unit_id,
        original=original,
        repaired=repaired,
    )






def unit_shape_issues(unit: ResearchUnit, seen: set[tuple[str, str, str]]) -> list[PlannerValidationIssue]:
    key = (unit.goal.lower(), unit.research_action.lower(), unit.evidence_policy.source_type)
    issues: list[PlannerValidationIssue] = []
    if key in seen:
        issues.append(issue("over_decomposition_duplicate", "duplicate research unit", unit.unit_id))
    seen.add(key)
    if unit.claim_policy.citation_required and not unit.evidence_policy.can_produce_citations:
        issues.append(issue("citation_policy_not_citable", "citation required without direct text", unit.unit_id))
    return issues


def has_dependency_cycle(units: list[ResearchUnit]) -> bool:
    graph = {unit.unit_id: [dep.unit_id for dep in unit.dependencies] for unit in units}
    pending = set(graph)
    resolved: set[str] = set()
    changed = True
    while changed:
        changed = False
        for unit_id in list(pending):
            if all(dep_id in resolved for dep_id in graph[unit_id]):
                pending.remove(unit_id)
                resolved.add(unit_id)
                changed = True
    return bool(pending)


def coverage_repair_status(current: str, missing_explicit: list[str]) -> str:
    if missing_explicit:
        return "failed"
    if current == "needed":
        return "repaired"
    return current
