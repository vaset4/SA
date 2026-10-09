"""【研究主张】【支持闭包】统一解释明确材料和上游主张的支持资格。作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.models.research_planner_contracts import (
    CURRENT_RESEARCH_CONTRACT_VERSION, ConsumerExposure, ResearchClaim, ResearchPlan,
    ResearchUnit, ResearchUnitResult,
)
from app.services.internal_research.evidence_identity import register_evidence_records
from app.services.internal_research.research_planner_consumer_budget import (
    claim_content_hash, content_hash, evidence_version,
)

FACT_TYPES = frozenset({"paper_supported", "evidence_grounded_inference"})
HYPOTHESIS_TYPES = frozenset({"model_prior_hypothesis", "memory_continuation"})


@dataclass(frozen=True, kw_only=True)
class ClaimSupport:
    """【研究主张】【支持事实】保存单条主张的明确闭包和不可用原因，相关背景不授信。"""

    claim: ResearchClaim
    owner_unit_id: str
    qualified: bool
    hypothesis: bool
    evidence_ids: tuple[str, ...] = ()
    citable_evidence_ids: tuple[str, ...] = ()
    paper_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class _SupportContext:
    registry: Mapping[str, dict[str, Any]]
    owners: Mapping[str, ResearchUnitResult]
    claims: Mapping[str, ResearchClaim]
    units: Mapping[str, ResearchUnit]
    duplicate_ids: frozenset[str]


def record_citable(record: Mapping[str, Any] | None) -> bool:
    """参数为原始材料；返回其现有正文引用资格，不将目录和元数据升级。"""
    return bool(record and record.get("evidence_id") and record.get("usable_as_citation")
                and record.get("layer") == "direct_text")


def resolve_support(
    plan: ResearchPlan, results: Sequence[ResearchUnitResult], evidence_records: Sequence[dict[str, Any]], *,
    exposure: ConsumerExposure | None = None,
) -> dict[str, ClaimSupport]:
    """参数为当前计划、成果、材料及可选本次投影；返回逐主张支持闭包与错误。"""
    resolved = resolve_result_support(results, evidence_records,
                                      units_by_id={unit.unit_id: unit for unit in plan.units}, allow_saved_units=False)
    if plan.contract_version != CURRENT_RESEARCH_CONTRACT_VERSION:
        return {key: replace(item, qualified=False, reasons=(*item.reasons, "historical_support_unknown"))
                for key, item in resolved.items()}
    if exposure is None:
        return resolved
    return {key: _filter_current_exposure(item, exposure) for key, item in resolved.items()}


def resolve_result_support(
    results: Sequence[ResearchUnitResult], evidence_records: Sequence[dict[str, Any]], *,
    units_by_id: Mapping[str, ResearchUnit] | None = None,
    allow_saved_units: bool = True,
) -> dict[str, ClaimSupport]:
    """参数为成果、材料、单元及是否允许保存声明；返回支持解析，完整计划禁止恢复已移除单元。"""
    registry = register_evidence_records({}, evidence_records)
    owners: dict[str, ResearchUnitResult] = {}
    claims: dict[str, ResearchClaim] = {}
    duplicates: set[str] = set()
    units = dict(units_by_id or {})
    for result in results:
        if allow_saved_units and result.unit_id not in units:
            raw = result.metadata.get("provenance", {}).get("unit")
            if raw is not None:
                units[result.unit_id] = ResearchUnit.model_validate(raw)
        for claim in result.claims:
            if claim.claim_id in claims:
                duplicates.add(claim.claim_id)
            owners[claim.claim_id], claims[claim.claim_id] = result, claim
    context = _SupportContext(registry=registry, owners=owners, claims=claims, units=units,
                              duplicate_ids=frozenset(duplicates))
    return {key: _resolve_claim(key, context, frozenset()) for key in claims}


def _resolve_claim(claim_id: str, context: _SupportContext, seen: frozenset[str]) -> ClaimSupport:
    """参数为主张身份和本次来源图；返回仅沿显式选择边展开的闭包，循环明确失败。"""
    claim, owner = context.claims[claim_id], context.owners[claim_id]
    hypothesis = claim.claim_type in HYPOTHESIS_TYPES
    reasons = _claim_state_errors(claim, owner, context)
    if claim_id in seen:
        reasons.append(f"support_cycle:{claim_id}")
        return ClaimSupport(claim=claim, owner_unit_id=owner.unit_id, qualified=False,
                            hypothesis=hypothesis, reasons=tuple(reasons))
    exposure = _saved_exposure(owner)
    reasons.extend(direct_support_errors(claim, context.registry, exposure))
    evidence_ids = set(claim.supporting_evidence_ids)
    inherited_ids: set[str] = set()
    for source_id in claim.source_claim_ids:
        error = _source_error(source_id, claim, owner, context, exposure)
        if error:
            reasons.append(error)
            continue
        upstream = _resolve_claim(source_id, context, seen | {claim_id})
        if not upstream.qualified or (upstream.hypothesis and not hypothesis):
            reasons.extend(upstream.reasons or (f"source_claim_not_factual:{source_id}",))
            continue
        if not hypothesis:
            evidence_ids.update(upstream.evidence_ids)
        inherited_ids.update((source_id, *upstream.claim_ids))
    if hypothesis:
        evidence_ids.clear()
    valid_ids = tuple(sorted(eid for eid in evidence_ids if eid in context.registry))
    return ClaimSupport(claim=claim, owner_unit_id=owner.unit_id, qualified=not reasons, hypothesis=hypothesis,
                        evidence_ids=valid_ids, claim_ids=tuple(sorted(inherited_ids)), reasons=tuple(reasons),
                        citable_evidence_ids=tuple(eid for eid in valid_ids if record_citable(context.registry[eid])),
                        paper_ids=tuple(sorted({str(context.registry[eid]["paper_id"]) for eid in valid_ids
                                                if context.registry[eid].get("paper_id")})))


def _claim_state_errors(claim: ResearchClaim, owner: ResearchUnitResult, context: _SupportContext) -> list[str]:
    """参数为主张、实际归属及计划；返回状态、角色许可和来源身份错误。"""
    errors: list[str] = []
    if owner.status in {"planned", "blocked", "external_needed"}:
        errors.append(f"claim_result_unavailable:{owner.unit_id}")
    expected_status = "needs_validation" if claim.claim_type in HYPOTHESIS_TYPES else "validated"
    if claim.validation_status != expected_status or claim.claim_type not in FACT_TYPES | HYPOTHESIS_TYPES:
        errors.append(f"claim_status_invalid:{claim.claim_id}")
    unit = context.units.get(owner.unit_id)
    if unit is None:
        errors.append(f"claim_owner_unknown:{owner.unit_id}")
    elif claim.claim_type not in unit.claim_policy.allowed_claim_types:
        errors.append(f"claim_type_not_allowed:{claim.claim_id}")
    if claim.claim_id in context.duplicate_ids:
        errors.append(f"claim_id_ambiguous:{claim.claim_id}")
    derived = {context.owners[key].unit_id for key in claim.source_claim_ids if key in context.owners}
    if set(claim.source_unit_ids) != derived:
        errors.append(f"source_units_not_derived:{claim.claim_id}")
    if claim.claim_type == "evidence_grounded_inference" and not (claim.supporting_evidence_ids or claim.source_claim_ids):
        errors.append(f"inference_support_missing:{claim.claim_id}")
    return errors


def _saved_exposure(result: ResearchUnitResult) -> ConsumerExposure | None:
    """参数为主张产生时的成果；读取保存投影，不按当前配置重算历史可见性。"""
    raw = result.metadata.get("provenance", {}).get("exposure")
    return ConsumerExposure.model_validate(raw) if raw is not None else None


def direct_support_errors(
    claim: Any, registry: Mapping[str, dict[str, Any]], exposure: ConsumerExposure | None,
) -> list[str]:
    """参数为主张、原材料和产生时投影；返回片段/版本/摘录不一致的明确原因。"""
    ids = set(claim.supporting_evidence_ids)
    errors: list[str] = []
    if ids != {span.evidence_id for span in claim.supporting_spans}:
        errors.append("supporting_spans_must_match_evidence_ids")
    if ids and exposure is None:
        errors.append("claim_exposure_missing")
    for span in claim.supporting_spans:
        record = registry.get(span.evidence_id)
        if record is None:
            errors.append(f"support_evidence_unknown:{span.evidence_id}")
            continue
        body = record.get(span.field)
        visible = [item for item in exposure.evidence if item.evidence_id == span.evidence_id] if exposure else []
        matches = [item for item in visible if item.field == span.field and item.start <= span.start < span.end <= item.end]
        version_valid = span.evidence_version == evidence_version(record)
        if not version_valid or not matches or any(item.evidence_version != span.evidence_version for item in matches):
            errors.append(f"support_span_not_exposed:{span.evidence_id}")
        elif not isinstance(body, str) or span.quote not in body[span.start:span.end]:
            errors.append(f"support_quote_mismatch:{span.evidence_id}")
        elif any(content_hash(body[item.start:item.end]) != item.content_hash for item in matches):
            errors.append(f"support_exposure_hash_mismatch:{span.evidence_id}")
    if claim.claim_type == "paper_supported" and (not ids or any(not record_citable(registry.get(eid)) for eid in ids)):
        errors.append("paper_supported_evidence_not_citable")
    if claim.claim_type in HYPOTHESIS_TYPES and ids:
        errors.append("hypothesis_cannot_carry_factual_support")
    return errors


def _source_error(
    source_id: str, claim: ResearchClaim, owner: ResearchUnitResult, context: _SupportContext,
    exposure: ConsumerExposure | None,
) -> str:
    """参数为明确来源边、主张及可见性；返回非直接、未送模、跨运行或不合格依赖错误。"""
    source = context.claims.get(source_id)
    upstream = context.owners.get(source_id)
    unit = context.units.get(owner.unit_id)
    if source is None or upstream is None or unit is None:
        return f"source_claim_unknown:{source_id}"
    dependencies = {dep.unit_id: dep for dep in unit.dependencies}
    dependency = dependencies.get(upstream.unit_id)
    if dependency is None or dependency.required_product == "materials":
        return f"source_claim_not_direct_dependency:{source_id}"
    if upstream.status not in {"inferred", "evidence_found"}:
        return f"source_claim_result_unavailable:{source_id}"
    if dependency.required_product == "hypotheses" and claim.claim_type not in HYPOTHESIS_TYPES:
        return f"hypothesis_not_factual_support:{source_id}"
    identities = [result.metadata.get("provenance", {}).get("execution_identity") for result in (owner, upstream)]
    if identities[0] != identities[1]:
        return f"source_claim_cross_run:{source_id}"
    bindings = owner.metadata.get("claim_id_rebindings", {})
    original_id = bindings.get(source_id, source_id)
    visible = {item.claim_id: item.content_hash for item in exposure.claims} if exposure else {}
    original_source = source.model_copy(update={"source_claim_ids": [bindings.get(key, key) for key in source.source_claim_ids]})
    if visible.get(original_id) != claim_content_hash(original_source):
        return f"source_claim_not_exposed:{source_id}"
    return ""


def _filter_current_exposure(item: ClaimSupport, exposure: ConsumerExposure) -> ClaimSupport:
    """参数为原支持解析和本次输入；限制本次可以选择的主张，不要求重读继承原文。"""
    visible = {entry.claim_id: entry.content_hash for entry in exposure.claims}
    if visible.get(item.claim.claim_id) == claim_content_hash(item.claim):
        return item
    return replace(item, qualified=False, reasons=(*item.reasons, "claim_not_in_current_exposure"))
