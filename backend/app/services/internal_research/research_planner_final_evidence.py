"""【研究交付】【成果目录】消费统一支持解释，不重新扩展整单元来源。作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.research_planner_contracts import ConsumerExposure, ResearchClaim
from app.services.internal_research.evidence_identity import register_evidence_records
from app.services.internal_research.research_planner_final_contracts import PlannerFinalAnswerInput
from app.services.internal_research.research_planner_fulfillment import records_for_unit
from app.services.internal_research.research_planner_support import ClaimSupport, record_citable, resolve_support


@dataclass(frozen=True, kw_only=True)
class FinalProductCatalog:
    """【研究交付】【可用成果】区分当前支持、实际材料与明确授权的历史投影。"""

    supports: dict[str, ClaimSupport]
    materials: dict[str, dict[str, Any]]
    material_facets: dict[str, frozenset[str]]
    historical_claims: dict[str, ResearchClaim]
    citable_evidence_ids: frozenset[str]

    @property
    def has_business_input(self) -> bool:
        """无参数；返回本次是否有能回应任务的合法业务产物。"""
        return bool(self.materials or self.historical_claims or any(
            item.qualified and item.claim.addressed_facet_ids for item in self.supports.values()
        ))


def final_product_catalog(payload: PlannerFinalAnswerInput) -> FinalProductCatalog:
    """参数为当前计划和成果；返回支持模块认可的主张及真实任务材料目录。"""
    registry = register_evidence_records({}, payload.evidence_records)
    exposure = ConsumerExposure.model_validate(payload.exposure) if payload.exposure else None
    supports = resolve_support(payload.plan, payload.unit_results, payload.evidence_records, exposure=exposure)
    material_facets: dict[str, set[str]] = {}
    by_id = {result.unit_id: result for result in payload.unit_results}
    facets = [*payload.plan.coverage_review.explicit_facets, *payload.plan.coverage_review.inferred_facets]
    material_requirements = {facet.facet_id for facet in facets if facet.deliverable_kind == "materials"}
    for unit in payload.plan.units:
        result = by_id.get(unit.unit_id)
        addressed = set(unit.covered_facet_ids) & material_requirements
        if result is None or result.status in {"blocked", "planned", "external_needed"} or not addressed:
            continue
        records = records_for_unit(unit, [registry[eid] for eid in result.evidence_ids if eid in registry],
                                   read_reports=result.metadata.get("read_reports", []))
        for record in records:
            material_facets.setdefault(str(record["evidence_id"]), set()).update(addressed)
    historical = _historical_claims(payload, registry)
    return FinalProductCatalog(
        supports=supports, materials={eid: registry[eid] for eid in material_facets},
        material_facets={eid: frozenset(ids) for eid, ids in material_facets.items()}, historical_claims=historical,
        citable_evidence_ids=frozenset(eid for eid, record in registry.items() if record_citable(record)),
    )


def _historical_claims(
    payload: PlannerFinalAnswerInput, registry: dict[str, dict[str, Any]],
) -> dict[str, ResearchClaim]:
    """参数为显式历史适配和保存材料；返回已核对可读身份的历史主张，不授予新版支持信用。"""
    context = payload.historical_input_context
    if payload.entrypoint != "historical_retry" or not context:
        return {}
    readable = set(context.get("readable_claim_ids", []))
    allowed_evidence = set(context.get("readable_evidence_ids", [])) & registry.keys()
    return {claim.claim_id: claim for result in payload.unit_results for claim in result.claims
            if claim.claim_id in readable and set(claim.supporting_evidence_ids) <= allowed_evidence}


def claim_allowed_evidence(payload: PlannerFinalAnswerInput, claim_ids: list[str]) -> set[str]:
    """参数为正式输入与明确选择的主张；返回统一支持闭包中的论文引用，不含 related。"""
    supports = resolve_support(payload.plan, payload.unit_results, payload.evidence_records)
    return {eid for cid in claim_ids if cid in supports and supports[cid].qualified
            for eid in supports[cid].citable_evidence_ids}
