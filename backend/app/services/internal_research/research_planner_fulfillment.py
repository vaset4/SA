"""【研究规划】【任务满足】纯判定要求、实际产物与依赖资格。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.models.research_planner_contracts import (
    EvidenceGap, EvidenceSufficiencyReport,
    FacetSufficiencyReport, PlannerCoverageReview, RequestedFacet, ResearchClaim,
    ResearchPlan, ResearchUnit, ResearchUnitResult, UnitDependency, UnitSufficiencyReport,
)
from app.services.internal_research.evidence_identity import register_evidence_records
from app.services.internal_research.research_planner_support import (
    record_citable as record_citable, resolve_result_support, resolve_support,
)

_SUCCESS_STATUSES = frozenset({"evidence_found", "inferred"})
_FACT_TYPES = frozenset({"paper_supported", "evidence_grounded_inference"})
_HYPOTHESIS_TYPES = frozenset({"model_prior_hypothesis", "memory_continuation"})
_BODY_SOURCES = frozenset({"local_direct_text", "local_section_text"})


def count_task_materials(records: Iterable[dict[str, Any]]) -> int:
    """参数为任务关联材料；返回独立来源组数，同源重叠范围合并，不改写 O2 分组观察。"""
    groups: dict[str, dict[tuple[str, ...], list[tuple[int, int]]]] = {}
    for record in records:
        group = str(record.get("material_group_id") or record.get("evidence_id") or "")
        if not group:
            continue
        ranges = groups.setdefault(group, {})
        for span in record.get("source_spans") or []:
            fragments = span.get("source_fragments") or []
            for fragment in fragments:
                source_key = (str(span["paper_id"]), str(span["batch_id"]), str(fragment["source_item"]))
                ranges.setdefault(source_key, []).append((int(fragment["source_start"]), int(fragment["source_end"])))
    merged: list[dict[tuple[str, ...], list[tuple[int, int]]]] = []
    for ranges in groups.values():
        overlap = [index for index, known in enumerate(merged) if _source_ranges_overlap(known, ranges)]
        combined = {key: list(values) for key, values in ranges.items()}
        for index in reversed(overlap):
            for key, values in merged.pop(index).items():
                combined.setdefault(key, []).extend(values)
        merged.append(combined)
    return len(merged)


def _source_ranges_overlap(
    left: dict[tuple[str, ...], list[tuple[int, int]]], right: dict[tuple[str, ...], list[tuple[int, int]]],
) -> bool:
    """参数为两个已知规范来源范围；返回是否存在同论文同版本来源交叠，不猜测 legacy 映射。"""
    return any(start < other_end and other_start < end for key in left.keys() & right.keys()
               for start, end in left[key] for other_start, other_end in right[key])


@dataclass(frozen=True, kw_only=True)
class DependencyValidationOutcome:
    """【研究规划】【依赖资格】分开可验证依据、待验证上下文与已读材料。"""

    satisfied: bool
    blocking_reasons: list[str]
    validated_results: tuple[ResearchUnitResult, ...] = ()
    hypothesis_results: tuple[ResearchUnitResult, ...] = ()
    material_records: tuple[dict[str, Any], ...] = ()


def requirements_version(review: PlannerCoverageReview) -> str:
    """参数为当前要求声明；返回必需要求规范内容的摘要，不使用单元或诊断编号。"""
    facets = [facet.model_dump(mode="json") for facet in [*review.explicit_facets, *review.inferred_facets]
              if facet.required]
    for facet in facets:
        for key in ("required_paper_ids", "required_claim_types"):
            facet[key] = sorted(facet[key])
    sections = [section.model_dump(mode="json") for section in review.answer_sections if section.required]
    for section in sections:
        # 1. 【研究规划】【历史要求】缺失的新展示事实不能改变已保存旧基准的摘要
        if section.get("presentation_kind") is None:
            section.pop("presentation_kind", None)
        section["facet_ids"] = sorted(section["facet_ids"])
    payload = json.dumps({"facets": sorted(facets, key=lambda item: item["facet_id"]),
                          "sections": sorted(sections, key=lambda item: item["section_id"])},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_dependencies(
    unit: ResearchUnit,
    results_by_id: Mapping[str, ResearchUnitResult],
    *,
    evidence_records: Sequence[dict[str, Any]] = (),
    units_by_id: Mapping[str, ResearchUnit] | None = None,
) -> DependencyValidationOutcome:
    """参数为消费单元、实际结果及材料/计划；返回依赖是否成立与按资格分开的只读投影。"""
    registry = register_evidence_records({}, evidence_records)
    reasons: list[str] = []
    validated: list[ResearchUnitResult] = []
    hypotheses: list[ResearchUnitResult] = []
    materials: list[dict[str, Any]] = []
    for dep in unit.dependencies:
        result = results_by_id.get(dep.unit_id)
        claims, records, reason = _dependency_products(dep, result, registry, units_by_id, results_by_id)
        if reason:
            if dep.required:
                reasons.append(reason)
            continue
        if result is None:
            continue
        projected = result.model_copy(update={"claims": claims})
        if dep.required_product == "hypotheses":
            hypotheses.append(projected)
        elif dep.required_product == "claims":
            validated.append(projected)
        else:
            materials.extend(dict(record) for record in records)
            validated.append(projected)
    return DependencyValidationOutcome(
        satisfied=not reasons, blocking_reasons=reasons, validated_results=tuple(validated),
        hypothesis_results=tuple(hypotheses), material_records=tuple(materials),
    )


def _dependency_products(
    dep: UnitDependency,
    result: ResearchUnitResult | None,
    registry: Mapping[str, dict[str, Any]],
    units_by_id: Mapping[str, ResearchUnit] | None,
    results_by_id: Mapping[str, ResearchUnitResult],
) -> tuple[list[ResearchClaim], list[dict[str, Any]], str]:
    """参数为单项依赖与上游成果；返回可用产物及不能满足该用途的明确原因。"""
    if dep.required_product is None:
        return [], [], f"dependency {dep.unit_id} has no declared product"
    if result is None:
        return [], [], f"dependency {dep.unit_id} has no current valid result"
    if result.status not in _SUCCESS_STATUSES:
        return [], [], f"dependency {dep.unit_id} status is {result.status}"
    upstream = units_by_id.get(dep.unit_id) if units_by_id else None
    records = [registry[eid] for eid in result.evidence_ids if eid in registry]
    if dep.required_product == "materials":
        valid = records_for_unit(upstream, records, read_reports=result.metadata.get("read_reports", [])) if upstream else records
        minimum = upstream.evidence_policy.min_citations_per_target if upstream else 0
        if not valid or count_task_materials(valid) < minimum:
            return [], [], f"dependency {dep.unit_id} lacks required materials"
        if dep.selections is not None:
            selected_ids = [item.evidence_id for item in dep.selections]
            by_id = {record["evidence_id"]: record for record in valid}
            if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids).issubset(by_id):
                return [], [], f"dependency {dep.unit_id} selects unavailable or duplicate materials"
            valid = [by_id[evidence_id] for evidence_id in selected_ids]
        return [], valid, ""
    wanted_status = "needs_validation" if dep.required_product == "hypotheses" else "validated"
    allowed = _HYPOTHESIS_TYPES if dep.required_product == "hypotheses" else _FACT_TYPES
    support = resolve_result_support(list(results_by_id.values()), list(registry.values()), units_by_id=units_by_id)
    claims = [claim for claim in result.claims if claim.validation_status == wanted_status
              and claim.claim_type in allowed and support[claim.claim_id].qualified]
    missing = set(dep.required_claim_types) - {claim.claim_type for claim in claims}
    if not claims or missing:
        return [], [], f"dependency {dep.unit_id} sources lack {wanted_status} claims: {sorted(missing)}"
    return claims, records, ""


def records_for_unit(
    unit: ResearchUnit | None, records: Sequence[dict[str, Any]], *, read_reports: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """参数为声明单元和实际材料；返回符合来源、范围与明确定位条件的材料。"""
    if unit is None:
        return list(records)
    policy = unit.evidence_policy
    valid = [record for record in records if not policy.target_paper_ids
             or record.get("paper_id") in policy.target_paper_ids]
    if policy.source_type in _BODY_SOURCES:
        valid = [record for record in valid if record_citable(record)]
    if policy.source_type in {"local_metadata", "local_chunk_metadata"}:
        valid = [record for record in valid if record.get("layer") == "metadata"]
    if policy.source_type == "local_section_structure":
        valid = [record for record in valid if record.get("layer") == "structure_hint"]
    if policy.source_type == "local_structure_hint":
        valid = [record for record in valid if record.get("layer") == "structure_hint"
                 or (policy.verify_with_direct_text and record_citable(record))]
    if policy.source_type == "local_section_text" and policy.required_sections:
        valid = _section_records(unit, valid, read_reports)
    if policy.required_pages:
        valid = [record for record in valid if record.get("page_num") in policy.required_pages
                 or bool(set(record.get("pages") or []) & set(policy.required_pages))]
    if policy.required_block_types:
        valid = [record for record in valid if record.get("block_type") in policy.required_block_types]
    return valid


def _section_records(
    unit: ResearchUnit, records: Sequence[dict[str, Any]], read_reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """参数为章节声明、材料和权威读取投影；按 O2 已解析定位及实际读取范围确认材料归属。"""
    expected = {section.casefold() for section in unit.evidence_policy.required_sections}
    approved: set[tuple[Any, ...]] = set()
    fields = ("paper_id", "batch_id", "chunk_id", "field", "start", "end")
    for report in read_reports:
        requested = {str(value).casefold() for value in report.get("requested", [])}
        if report.get("operation") != "sections" or not requested & expected or report.get("status") not in {"complete", "partial"}:
            continue
        # 1. 【研究规划】【章节定位】短标题/编号的解析由知识访问完成，不能再用显示标题的字符串相等推翻真实读取
        approved.update(tuple(span.get(key) for key in fields) for span in report.get("read_spans", []))
    valid: list[dict[str, Any]] = []
    for record in records:
        spans = {tuple(span.get(key) for key in fields) for span in record.get("source_spans", [])}
        if str(record.get("section") or "").casefold() in expected or (spans and spans <= approved):
            valid.append(record)
    return valid


def claim_links_valid(claim: ResearchClaim, registry: Mapping[str, dict[str, Any]]) -> bool:
    """参数为主张和权威材料池；返回已声明材料链接的存在与引用资格。"""
    if any(eid not in registry for eid in [*claim.supporting_evidence_ids, *claim.related_evidence_ids]):
        return False
    if claim.claim_type == "paper_supported":
        return bool(claim.supporting_evidence_ids) and all(record_citable(registry[eid])
                                                         for eid in claim.supporting_evidence_ids)
    return True


def evaluate_unit_fulfillment(
    unit: ResearchUnit,
    result: ResearchUnitResult | None,
    *,
    registry: Mapping[str, dict[str, Any]],
    results_by_id: Mapping[str, ResearchUnitResult],
    plan: ResearchPlan | None = None,
    pass_index: int = 1,
) -> UnitSufficiencyReport:
    """参数为单元声明与真实成果；返回局部产物验收，不要求单元独立完成多单元共同要求。"""
    results_by_id = {**results_by_id, **({result.unit_id: result} if result is not None else {})}
    records = records_for_unit(unit, [registry[eid] for eid in result.evidence_ids if eid in registry] if result else [],
                               read_reports=result.metadata.get("read_reports", []) if result else [])
    dependencies = evaluate_dependencies(unit, results_by_id, evidence_records=list(registry.values()),
                                         units_by_id={item.unit_id: item for item in plan.units} if plan else None)
    claims = _usable_claims(unit, result, registry, results_by_id=results_by_id, plan=plan) if dependencies.satisfied else []
    gaps = _unit_product_gaps(unit, result, records, claims, registry)
    gaps.extend(make_gap(unit, "dependency_unresolved", reason) for reason in dependencies.blocking_reasons)
    if plan:
        gaps.extend(_local_object_gaps(unit, plan, records, claims, registry, results_by_id=results_by_id))
    targets = {pid: count_task_materials(record for record in records if record.get("paper_id") == pid)
               for pid in unit.evidence_policy.target_paper_ids}
    return UnitSufficiencyReport(
        unit_id=unit.unit_id, pass_index=pass_index, hard_gaps=gaps, target_coverage=targets,
        claim_coverage={claim.claim_id: list(claim.supporting_evidence_ids) for claim in result.claims} if result else {},
        blocking_for_final=bool(gaps), fulfilled=not gaps,
        repair_recommended=dependencies.satisfied and any(gap_repairable(unit, gap) for gap in gaps),
        usable_evidence_ids=[str(record["evidence_id"]) for record in records],
        usable_claim_ids=[claim.claim_id for claim in claims],
    )


def _usable_claims(
    unit: ResearchUnit, result: ResearchUnitResult | None, registry: Mapping[str, dict[str, Any]],
    *, results_by_id: Mapping[str, ResearchUnitResult], plan: ResearchPlan | None = None,
) -> list[ResearchClaim]:
    """参数为单元、结果及材料池；返回状态、类型和引用均合法的主张，不补造要求关联。"""
    if result is None or result.status in {"blocked", "planned", "external_needed"}:
        return []
    support = (resolve_support(plan, list(results_by_id.values()), list(registry.values())) if plan is not None
               else resolve_result_support(list(results_by_id.values()), list(registry.values()),
                                            units_by_id={unit.unit_id: unit}))
    return [claim for claim in result.claims if support[claim.claim_id].qualified]


def _unit_product_gaps(
    unit: ResearchUnit, result: ResearchUnitResult | None, records: list[dict[str, Any]],
    claims: list[ResearchClaim], registry: Mapping[str, dict[str, Any]],
) -> list[EvidenceGap]:
    """参数为单元和选定产物；返回空产物、缺类型、失效引用与材料总量缺口。"""
    if result is None:
        return [make_gap(unit, "unit_result_missing", "unit did not return a result")]
    gaps = [make_gap(unit, "evidence_id_missing", f"evidence id {eid} is not registered")
            for eid in result.evidence_ids if eid not in registry]
    if result.status in {"planned", "blocked", "external_needed"}:
        gaps.append(make_gap(unit, "unit_state_unresolved", f"unit is {result.status}"))
    kind = unit.expected_output.deliverable_kind
    minimum = unit.evidence_policy.min_citations_per_target
    material_needed = kind == "materials" or unit.evidence_policy.source_type in _BODY_SOURCES
    count = count_task_materials(records)
    if material_needed and (not records or count < minimum):
        gap_type = "target_coverage_missing" if len(unit.evidence_policy.target_paper_ids) == 1 else "direct_text_missing"
        target = unit.evidence_policy.target_paper_ids[0] if len(unit.evidence_policy.target_paper_ids) == 1 else ""
        gaps.append(make_gap(unit, gap_type, "declared materials missing", target=target, required=minimum, found=count))
    if kind != "materials" and not claims:
        gaps.append(make_gap(unit, "claim_output_missing", "unit has no eligible claim output"))
    missing_types = set(unit.expected_output.required_claim_types) - {claim.claim_type for claim in claims}
    gaps.extend(make_gap(unit, "required_claim_type_missing", f"required claim type missing: {kind}", requirement=kind)
                for kind in sorted(missing_types))
    gaps.extend(make_gap(unit, "claim_support_missing", f"claim references invalid evidence: {claim.claim_id}")
                for claim in result.claims if not claim_links_valid(claim, registry))
    if unit.evidence_policy.verify_with_direct_text and not any(record_citable(record) for record in records):
        targets = unit.evidence_policy.target_paper_ids
        gaps.append(make_gap(unit, "structure_verification_missing", "structure_hint_direct_text_verification_missing",
                             target=targets[0] if len(targets) == 1 else ""))
    return gaps


def _local_object_gaps(
    unit: ResearchUnit, plan: ResearchPlan, records: list[dict[str, Any]],
    claims: list[ResearchClaim], registry: Mapping[str, dict[str, Any]],
    *, results_by_id: Mapping[str, ResearchUnitResult],
) -> list[EvidenceGap]:
    """参数为当前单元及计划/产物；只验收该单元读取能力实际承担的必需对象。"""
    gaps: list[EvidenceGap] = []
    scope = declared_paper_scope(unit, {item.unit_id: item for item in plan.units})
    for facet in [*plan.coverage_review.explicit_facets, *plan.coverage_review.inferred_facets]:
        if not facet.required or facet.facet_id not in unit.covered_facet_ids:
            continue
        # 1. 【研究规划】【分工验收】材料前置只承担本单元的材料交付，最终主张仍由要求聚合报告验收
        associated = records if unit.expected_output.deliverable_kind == "materials" else _facet_records(
            facet, records, claims, registry, results_by_id=results_by_id,
        )
        for paper_id in facet.required_paper_ids:
            if scope and paper_id not in scope:
                continue
            count = count_task_materials(record for record in associated if record.get("paper_id") == paper_id)
            minimum = facet.required_paper_counts.get(paper_id, 1)
            if count < minimum:
                gaps.append(make_gap(unit, "target_coverage_missing", "required facet object lacks eligible output",
                                     facet=facet, target=paper_id, required=minimum, found=count))
    return gaps


def declared_paper_scope(unit: ResearchUnit, units_by_id: Mapping[str, ResearchUnit]) -> set[str]:
    """参数为单元和计划映射；返回自身或依赖所声明论文范围，空集合表示未收窄。"""
    targets = set(unit.evidence_policy.target_paper_ids)
    if targets or unit.evidence_policy.source_type != "inference_from_units":
        return targets
    pending = [dep.unit_id for dep in unit.dependencies]
    seen = {unit.unit_id}
    while pending:
        unit_id = pending.pop()
        if unit_id in seen or unit_id not in units_by_id:
            continue
        seen.add(unit_id)
        upstream = units_by_id[unit_id]
        targets.update(upstream.evidence_policy.target_paper_ids)
        pending.extend(dep.unit_id for dep in upstream.dependencies)
    return targets


def _facet_records(
    facet: RequestedFacet, records: Sequence[dict[str, Any]], claims: Sequence[ResearchClaim],
    registry: Mapping[str, dict[str, Any]],
    *, results_by_id: Mapping[str, ResearchUnitResult],
) -> list[dict[str, Any]]:
    """参数为要求与有效产物；返回真实响应该要求的材料关联，主张自报不足以证明语义。"""
    if facet.deliverable_kind == "materials":
        return list(records)
    relevant = [claim for claim in claims if facet.facet_id in claim.addressed_facet_ids]
    ids = {eid for claim in relevant for eid in _claim_material_ids(claim, results_by_id, registry)}
    return [registry[eid] for eid in sorted(ids) if eid in registry]


def _claim_material_ids(
    claim: ResearchClaim, results: Mapping[str, ResearchUnitResult], registry: Mapping[str, dict[str, Any]],
) -> set[str]:
    """参数为主张、有效成果和原材料；返回统一支持解释中的明确来源，背景不授信。"""
    support = resolve_result_support(list(results.values()), list(registry.values()))
    item = support.get(claim.claim_id)
    return set(item.evidence_ids) if item is not None and item.qualified else set()


def build_evidence_sufficiency_report(
    *, plan: ResearchPlan, results: list[ResearchUnitResult], evidence_records: list[dict[str, Any]], pass_index: int,
) -> EvidenceSufficiencyReport:
    """参数为计划及累计真实成果；返回局部与跨单元要求的统一满足报告。"""
    registry = register_evidence_records({}, evidence_records)
    by_id = {result.unit_id: result for result in results}
    unit_reports = [evaluate_unit_fulfillment(unit, by_id.get(unit.unit_id), registry=registry,
                    results_by_id=by_id, plan=plan, pass_index=pass_index) for unit in plan.units]
    facets = [_facet_report(facet, plan, by_id, registry, unit_reports)
              for facet in [*plan.coverage_review.explicit_facets, *plan.coverage_review.inferred_facets]]
    blocking = required_unit_ids(plan)
    if plan.metadata.execution_mode == "incremental":
        # 1. 【研究循环】【累计交付】已由后续有效成果满足的要求，不被早先失败的探索重复阻断
        missing = {item.facet_id for item in facets if item.required and not item.satisfied}
        blocking = {unit.unit_id for unit in plan.units if set(unit.covered_facet_ids) & missing}
    return EvidenceSufficiencyReport(
        pass_index=pass_index, unit_reports=unit_reports,
        blocking_unit_ids=[item.unit_id for item in unit_reports if item.blocking_for_final
                           and item.unit_id in blocking],
        repair_recommended=any(item.repair_recommended for item in unit_reports if item.unit_id in blocking),
        contract_version=plan.contract_version, requirements_version=requirements_version(plan.coverage_review),
        facet_reports=facets,
        fulfilled_required_facet_ids=[item.facet_id for item in facets if item.required and item.satisfied],
        missing_required_facet_ids=[item.facet_id for item in facets if item.required and not item.satisfied],
        has_usable_results=any(item.usable_evidence_ids or item.usable_claim_ids for item in unit_reports),
        fulfilled_requirement_keys=sorted({key for item in facets if item.required
                                           for key in item.fulfilled_requirement_keys}
                                          | _fulfilled_dependency_keys(plan, by_id, registry)),
    )


def evaluate_delivered_facet(
    plan: ResearchPlan, facet: RequestedFacet, results: Sequence[ResearchUnitResult],
    evidence_records: Sequence[dict[str, Any]], *, claim_ids: Sequence[str], evidence_ids: Sequence[str],
) -> FacetSufficiencyReport:
    """参数为完整成果及正文实际选中产物；复用要求验收，继承支持仍读取完整来源图。"""
    registry = register_evidence_records({}, evidence_records)
    by_id = {result.unit_id: result for result in results}
    selected_claims, selected_evidence = set(claim_ids), set(evidence_ids)
    reports = []
    for unit in plan.units:
        report = evaluate_unit_fulfillment(unit, by_id.get(unit.unit_id), registry=registry,
                                           results_by_id=by_id, plan=plan)
        reports.append(report.model_copy(update={
            "usable_claim_ids": [key for key in report.usable_claim_ids if key in selected_claims],
            "usable_evidence_ids": [key for key in report.usable_evidence_ids if key in selected_evidence],
        }))
    return _facet_report(facet, plan, by_id, registry, reports)


def _facet_report(
    facet: RequestedFacet, plan: ResearchPlan, results: Mapping[str, ResearchUnitResult],
    registry: Mapping[str, dict[str, Any]], unit_reports: list[UnitSufficiencyReport],
) -> FacetSufficiencyReport:
    """参数为单项要求和实际结果；聚合多个单元的真实关联，保留逐对象缺失。"""
    owners = [unit for unit in plan.units if facet.facet_id in unit.covered_facet_ids]
    reports = {item.unit_id: item for item in unit_reports}
    claims: list[ResearchClaim] = []
    records: list[dict[str, Any]] = []
    for unit in owners:
        result = results.get(unit.unit_id)
        if result is None:
            continue
        report = reports[unit.unit_id]
        if any(gap.gap_type == "dependency_unresolved" for gap in report.hard_gaps):
            continue
        claims.extend(claim for claim in result.claims if claim.claim_id in report.usable_claim_ids
                      and facet.facet_id in claim.addressed_facet_ids
                      and (facet.required_output_role is None or unit.output_role == facet.required_output_role))
        if result.status not in {"blocked", "planned", "external_needed"}:
            records.extend(registry[eid] for eid in report.usable_evidence_ids)
    associated = _facet_records(facet, records, claims, registry, results_by_id=results)
    coverage = {pid: count_task_materials(record for record in associated if record.get("paper_id") == pid)
                for pid in facet.required_paper_ids}
    missing = [pid for pid, count in coverage.items() if count < facet.required_paper_counts.get(pid, 1)]
    owner = owners[0] if owners else plan.units[0]
    gaps = [make_gap(owner, "target_coverage_missing", "required facet object lacks eligible output", facet=facet,
                     target=pid, required=facet.required_paper_counts.get(pid, 1), found=coverage[pid]) for pid in missing]
    count = count_task_materials(associated)
    if count < facet.min_material_count or (facet.deliverable_kind == "materials" and not associated):
        gaps.append(make_gap(owner, "facet_material_missing", "required facet materials missing", facet=facet,
                             required=facet.min_material_count, found=count))
    if facet.deliverable_kind == "claims" and not claims:
        gaps.append(make_gap(owner, "facet_claim_missing", "required facet has no associated eligible claim", facet=facet))
    missing_types = set(facet.required_claim_types) - {claim.claim_type for claim in claims}
    gaps.extend(make_gap(owner, "required_claim_type_missing", f"required facet claim type missing: {kind}",
                         facet=facet, requirement=kind) for kind in sorted(missing_types))
    object_gaps, fulfilled = _object_requirement_facts(facet, claims, coverage, registry, results, owner=owner)
    gaps.extend(object_gaps)
    if not gaps:
        fulfilled.append(f"{facet.facet_id}:fulfilled")
    return FacetSufficiencyReport(
        facet_id=facet.facet_id, required=bool(facet.required), satisfied=not gaps,
        evidence_ids=sorted({str(record["evidence_id"]) for record in associated}),
        claim_ids=sorted({claim.claim_id for claim in claims}), target_coverage=coverage,
        missing_paper_ids=sorted(set(missing) | {gap.target_paper_id for gap in object_gaps}), hard_gaps=gaps,
        fulfilled_requirement_keys=fulfilled,
    )


def _object_requirement_facts(
    facet: RequestedFacet, claims: list[ResearchClaim], coverage: dict[str, int],
    registry: Mapping[str, dict[str, Any]], results: Mapping[str, ResearchUnitResult], *, owner: ResearchUnit,
) -> tuple[list[EvidenceGap], list[str]]:
    """参数为要求、对象材料和主张；返回逐对象产物缺口与真正满足的稳定要求键。"""
    gaps: list[EvidenceGap] = []
    fulfilled: list[str] = []
    for paper_id, count in coverage.items():
        types = {claim.claim_type for claim in claims
                 if any(registry[eid].get("paper_id") == paper_id for eid in _claim_material_ids(claim, results, registry)
                        if eid in registry)}
        missing = set(facet.required_claim_types) - types if facet.deliverable_kind == "claims" else set()
        for kind in sorted(missing):
            gaps.append(make_gap(owner, "required_claim_type_missing", f"required object claim type missing: {kind}",
                                 facet=facet, target=paper_id, requirement=kind))
        if not missing and count >= facet.required_paper_counts.get(paper_id, 1):
            fulfilled.append(f"{facet.facet_id}:{paper_id}:output")
    return gaps, fulfilled


def required_unit_ids(plan: ResearchPlan) -> set[str]:
    """参数为计划；返回必需要求的认领单元及其必需前置，普通探索不进入完成分母。"""
    facets = {facet.facet_id for facet in [*plan.coverage_review.explicit_facets, *plan.coverage_review.inferred_facets]
              if facet.required}
    by_id = {unit.unit_id: unit for unit in plan.units}
    required = {unit.unit_id for unit in plan.units if set(unit.covered_facet_ids) & facets}
    pending = list(required)
    while pending:
        unit = by_id[pending.pop()]
        for dep in unit.dependencies:
            if dep.required and dep.unit_id not in required and dep.unit_id in by_id:
                required.add(dep.unit_id)
                pending.append(dep.unit_id)
    return required


def _fulfilled_dependency_keys(
    plan: ResearchPlan, results: Mapping[str, ResearchUnitResult], registry: Mapping[str, dict[str, Any]],
) -> set[str]:
    """参数为计划与当前成果；返回已成立必需前置的稳定产物键，不使用单元编号或查询措辞。"""
    by_id = {unit.unit_id: unit for unit in plan.units}
    required = required_unit_ids(plan)
    required_facets = {facet.facet_id for facet in [*plan.coverage_review.explicit_facets, *plan.coverage_review.inferred_facets]
                       if facet.required}
    keys: set[str] = set()
    for unit in plan.units:
        if unit.unit_id not in required:
            continue
        for dependency in unit.dependencies:
            if not dependency.required:
                continue
            _claims, _records, reason = _dependency_products(dependency, results.get(dependency.unit_id), registry,
                                                            by_id, results)
            if reason:
                continue
            upstream = by_id[dependency.unit_id]
            fact = {"facets": sorted(set(unit.covered_facet_ids) & required_facets),
                    "product": dependency.required_product, "types": sorted(dependency.required_claim_types),
                    "objects": sorted(declared_paper_scope(upstream, by_id))}
            keys.add("dependency:" + json.dumps(fact, sort_keys=True, ensure_ascii=False))
    return keys


def make_gap(
    unit: ResearchUnit, gap_type: str, reason: str, *, facet: RequestedFacet | None = None,
    target: str = "", required: int = 1, found: int = 0, requirement: str = "",
) -> EvidenceGap:
    """参数为单元及缺失条件；返回稳定要求键与当前执行定位分别保存的缺口。"""
    dimension = facet.label if facet else next(iter(unit.evidence_policy.evidence_dimensions), unit.goal)
    key = ":".join((facet.facet_id if facet else "unit", target, gap_type, requirement))
    return EvidenceGap(
        gap_id=f"{unit.unit_id}:{key}", unit_id=unit.unit_id, gap_type=gap_type, reason=reason,
        facet_id=facet.facet_id if facet else "", requirement_key=key, evidence_dimension=dimension,
        target_paper_id=target, required_count=required, found_count=found,
    )


def gap_repairable(unit: ResearchUnit, gap: EvidenceGap) -> bool:
    """参数为单元及缺口；返回现有定向正文补读能否承担该缺口。"""
    if gap.gap_type == "structure_verification_missing":
        return unit.evidence_policy.source_type == "local_structure_hint"
    return unit.evidence_policy.source_type in _BODY_SOURCES and gap.gap_type in {
        "direct_text_missing", "target_coverage_missing", "facet_material_missing",
    }
