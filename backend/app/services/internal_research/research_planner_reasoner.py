"""【研究主张】【草拟校验】从一次准备的可见输入建立明确支持关系。作者：xxx。"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from app.models.research_planner_contracts import (
    BoundResearchClaimDraftBatch, ResearchClaim, ResearchClaimDraft, ResearchClaimDraftRejection,
    ResearchPlan, ResearchUnit, ResearchUnitResult, UnprovidedEvidenceRange,
)
from app.services.llm_observation import record_business_validation
from app.services.internal_research.evidence_identity import register_evidence_records
from app.services.internal_research.research_planner_consumer_budget import (
    CLAIM_PROMPT_VERSION, PreparedConsumerInput, configured_consumer_budget, prepare_consumer_input, unprovided_range,
)
from app.services.internal_research.research_planner_fulfillment import (
    DependencyValidationOutcome as DependencyValidationOutcome,
    evaluate_dependencies as validate_unit_dependencies,
)
from app.services.internal_research.research_planner_support import HYPOTHESIS_TYPES, resolve_result_support


ClaimDraftProviderResult = BoundResearchClaimDraftBatch | Sequence[ResearchClaimDraft | dict[str, Any]]


class ClaimDraftProvider(Protocol):
    """接收真实准备输入的草拟端口；不能在模型边界重新裁剪。"""

    def __call__(
        self, unit: ResearchUnit, evidence_records: list[dict[str, Any]],
        dependency_results: list[ResearchUnitResult], *, read_reports: list[dict[str, Any]] | None = None,
        prepared_input: PreparedConsumerInput | None = None,
    ) -> Awaitable[ClaimDraftProviderResult] | ClaimDraftProviderResult:
        """参数为单元、实际材料、依赖、读取说明和输入投影；返回真实草稿。"""
        ...


@dataclass(frozen=True, kw_only=True)
class PreparedReasoningInput:
    """【研究主张】【一次准备】保存合格依赖、材料及模型实际输入。"""

    records: tuple[dict[str, Any], ...]
    results: tuple[ResearchUnitResult, ...]
    consumer: PreparedConsumerInput
    blocking_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ClaimValidationOutcome:
    """【研究主张】【校验成果】正式主张及产生时投影共同进入单元结果。"""

    claims: list[ResearchClaim]
    rejected_reasons: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[str]:
        """无参数；返回即时校验的明确拒绝原因。"""
        return self.rejected_reasons


@dataclass(frozen=True, kw_only=True)
class ResearchUnitReasoner:
    """【研究主张】【推理入口】共享依赖资格及来源解析，不调用额外语义裁判。"""

    draft_provider: ClaimDraftProvider | None = None
    memory_grounded_allowed: bool = False

    def prepare(
        self, unit: ResearchUnit, evidence_records: list[dict[str, Any]], dependency_results: list[ResearchUnitResult], *,
        dependency_evidence_records: list[dict[str, Any]] | None = None, plan: ResearchPlan | None = None,
    ) -> PreparedReasoningInput:
        """参数为单元及真实材料/依赖；返回缓存判断和草拟共用的同一可见投影。"""
        registry = list(register_evidence_records({}, [*(dependency_evidence_records or []), *evidence_records]).values())
        dependencies = validate_unit_dependencies(unit, {result.unit_id: result for result in dependency_results},
                        evidence_records=registry, units_by_id={item.unit_id: item for item in plan.units} if plan else None)
        results = (*dependencies.validated_results, *dependencies.hypothesis_results)
        records = tuple(register_evidence_records({}, [*evidence_records, *dependencies.material_records]).values())
        selected_ranges = _selected_material_ranges(unit, dependency_results, records)
        consumer = prepare_consumer_input(records, [claim for result in results for claim in result.claims],
                    evidence_max_chars=configured_consumer_budget(),
                    prompt_version=CLAIM_PROMPT_VERSION, selected_ranges=selected_ranges)
        return PreparedReasoningInput(records=records, results=results, consumer=consumer,
                                      blocking_reasons=tuple(dependencies.blocking_reasons))

    async def reason(
        self, unit: ResearchUnit, evidence_records: list[dict[str, Any]], dependency_results: list[ResearchUnitResult], *,
        read_reports: list[dict[str, Any]] | None = None,
        dependency_evidence_records: list[dict[str, Any]] | None = None,
        prepared_input: PreparedReasoningInput | None = None, plan: ResearchPlan | None = None,
        execution_identity: Mapping[str, Any] | None = None,
    ) -> ClaimValidationOutcome:
        """参数为真实草拟输入及运行身份；返回校验主张与产生时投影，错误不伪造成功。"""
        prepared = prepared_input or self.prepare(unit, evidence_records, dependency_results,
                    dependency_evidence_records=dependency_evidence_records, plan=plan)
        if prepared.blocking_reasons:
            return ClaimValidationOutcome(claims=[], rejected_reasons=list(prepared.blocking_reasons))
        if self.draft_provider is None:
            raise ValueError("claim drafting requires an explicit draft provider")
        raw = self.draft_provider(unit, list(prepared.records), list(prepared.results),
                                 read_reports=read_reports, prepared_input=prepared.consumer)
        resolved = await raw if isinstance(raw, Awaitable) else raw
        batch: BoundResearchClaimDraftBatch | None = None
        drafts: list[ResearchClaimDraft | dict[str, Any]]
        if isinstance(resolved, BoundResearchClaimDraftBatch):
            batch, drafts = resolved, list(resolved.claims)
        else:
            drafts = list(resolved)
        return validate_claim_drafts(unit, list(prepared.records), dependency_results, drafts,
                    memory_grounded_allowed=self.memory_grounded_allowed,
                    dependency_evidence_records=dependency_evidence_records, prepared_input=prepared.consumer,
                    plan=plan, execution_identity=execution_identity,
                    draft_positions=batch.claim_positions if batch is not None else None,
                    draft_rejections=batch.rejected_claims if batch is not None else ())


def _selected_material_ranges(
    unit: ResearchUnit, results: list[ResearchUnitResult], records: Sequence[dict[str, Any]],
) -> dict[str, UnprovidedEvidenceRange]:
    """参数为当前选择、已有结果和合格依赖材料；仅解引用系统曾提供且版本未变的范围。"""
    catalog = {entry["range_id"]: entry for result in results
               for entry in result.metadata.get("provenance", {}).get("exposure", {}).get("unprovided_ranges", [])}
    registry = {record["evidence_id"]: record for record in records}
    owners = {result.unit_id: set(result.evidence_ids) for result in results}
    selected: dict[str, UnprovidedEvidenceRange] = {}
    for dependency in unit.dependencies:
        for selection in dependency.selections or []:
            if selection.evidence_id not in owners.get(dependency.unit_id, set()) or selection.evidence_id not in registry:
                raise ValueError("所选材料不属于该依赖的合格已返回材料")
            if selection.range_id is None:
                continue
            raw = catalog.get(selection.range_id)
            if raw is None or raw["evidence_id"] != selection.evidence_id:
                raise ValueError("所选材料范围不属于已返回的依赖材料或未提供目录")
            item = UnprovidedEvidenceRange.model_validate(raw)
            if item != unprovided_range(registry[item.evidence_id], start=item.start, end=item.end):
                raise ValueError("所选材料范围的证据版本已变化")
            if item.evidence_id in selected and selected[item.evidence_id] != item:
                raise ValueError("同一材料不能同时选择不一致的范围")
            selected[item.evidence_id] = item
    return selected


def validate_claim_drafts(
    unit: ResearchUnit, evidence_records: list[dict[str, Any]], dependency_results: list[ResearchUnitResult],
    drafts: list[ResearchClaimDraft | dict[str, Any]], *, memory_grounded_allowed: bool = False,
    dependency_evidence_records: list[dict[str, Any]] | None = None,
    prepared_input: PreparedConsumerInput | None = None, plan: ResearchPlan | None = None,
    execution_identity: Mapping[str, Any] | None = None,
    draft_positions: Sequence[int] | None = None,
    draft_rejections: Sequence[ResearchClaimDraftRejection] = (),
) -> ClaimValidationOutcome:
    """参数为单元、原材料、依赖与草稿；返回明确支持核验结果，不证明自然语言蕴含。"""
    registry = list(register_evidence_records({}, [*(dependency_evidence_records or []), *evidence_records]).values())
    units = {item.unit_id: item for item in plan.units} if plan else {}
    units[unit.unit_id] = unit
    dependency = validate_unit_dependencies(unit, {result.unit_id: result for result in dependency_results},
                                            evidence_records=registry, units_by_id=units)
    if not dependency.satisfied:
        return ClaimValidationOutcome(claims=[], rejected_reasons=dependency.blocking_reasons)
    consumer = prepared_input or ResearchUnitReasoner().prepare(unit, evidence_records, dependency_results,
                                dependency_evidence_records=registry, plan=plan).consumer
    provenance = {"unit": unit.model_dump(mode="json"), "exposure": consumer.exposure.model_dump(mode="json"),
                  "execution_identity": dict(execution_identity) if execution_identity is not None else None}
    if draft_rejections:
        provenance["draft_rejections"] = [item.model_dump(mode="json") for item in draft_rejections]
    source_owners = {claim.claim_id: result.unit_id for result in dependency_results for claim in result.claims}
    coerced = [item if isinstance(item, ResearchClaimDraft) else ResearchClaimDraft.model_validate(item) for item in drafts]
    candidates: list[ResearchClaim] = []
    errors = [item.message for item in draft_rejections]
    positions = list(draft_positions) if draft_positions is not None else list(range(1, len(coerced) + 1))
    original_positions = {}
    for position, draft in zip(positions, coerced, strict=True):
        reason = _draft_error(unit, draft, evidence_records, memory_grounded_allowed)
        if reason:
            errors.append(f"claims[{position}] (1-based) {reason}" if draft_positions is not None else reason)
            continue
        claim = _to_claim(unit, draft, position if draft_positions is not None else len(candidates) + 1, source_owners)
        candidates.append(claim)
        original_positions[claim.claim_id] = position
    result = ResearchUnitResult(unit_id=unit.unit_id, status="inferred", claims=candidates,
                                evidence_ids=[str(record["evidence_id"]) for record in evidence_records],
                                metadata={"provenance": provenance})
    supported = resolve_result_support([*dependency_results, result], registry, units_by_id=units,
                                       allow_saved_units=plan is None)
    claims = [claim for claim in candidates if supported[claim.claim_id].qualified]
    errors.extend(f"claims[{original_positions[claim.claim_id]}] (1-based) {reason}" if draft_positions is not None else reason
                  for claim in candidates for reason in supported[claim.claim_id].reasons)
    record_business_validation({"claims": [draft.model_dump(mode="json") for draft in coerced]}, {
        "validator": "research_claims", "schema": "ResearchClaimStrategyDraftBatch", "phase": "initial",
        "validation_stage": "claim_evidence_references", "unit_id": unit.unit_id,
        "valid": not errors, "accepted": not errors, "errors": errors,
        "bound_claim_positions": positions,
        "accepted_claim_positions": [original_positions[claim.claim_id] for claim in claims],
        "acceptance_status": "partial" if claims and errors else "rejected" if errors else "accepted",
        "rejected_drafts": [item.model_dump(mode="json") for item in draft_rejections],
        "available_evidence_ids": list(consumer.exposure.acquired_evidence_ids),
        "exposure": consumer.exposure.model_dump(mode="json"),
        "accepted_claims": [claim.model_dump(mode="json") for claim in claims],
    })
    return ClaimValidationOutcome(claims=claims, rejected_reasons=errors, provenance=provenance)


def _draft_error(
    unit: ResearchUnit, draft: ResearchClaimDraft, records: list[dict[str, Any]], memory_allowed: bool,
) -> str:
    """参数为草稿与任务许可；返回关系解析前的明确类型和要求归属错误。"""
    if set(draft.addressed_facet_ids) - set(unit.covered_facet_ids):
        return "claim addresses unassigned facets"
    if set(draft.related_evidence_ids) - {str(record["evidence_id"]) for record in records}:
        return "claim related evidence ids are unknown"
    if draft.claim_type not in unit.claim_policy.allowed_claim_types:
        return f"{draft.claim_type} is not allowed for {unit.unit_id}"
    if draft.claim_type == "paper_supported" and not draft.supporting_evidence_ids:
        return "paper_supported claims require supporting_evidence_ids"
    if draft.claim_type == "evidence_grounded_inference":
        if not draft.reasoning_summary:
            return "evidence_grounded_inference requires reasoning_summary"
        if not (draft.supporting_evidence_ids or draft.source_claim_ids):
            return "evidence_grounded_inference requires evidence or source_claim_ids"
    if draft.claim_type in HYPOTHESIS_TYPES and draft.supporting_evidence_ids:
        return f"{draft.claim_type} cannot carry supporting_evidence_ids"
    if draft.claim_type == "memory_continuation" and not memory_allowed:
        return "memory_continuation requires project memory read in this turn; specific memory binding is unknown"
    return ""


def _to_claim(
    unit: ResearchUnit, draft: ResearchClaimDraft, index: int, source_owners: Mapping[str, str],
) -> ResearchClaim:
    """参数为校验草稿及真实上游归属；派生单元来源，保持假设待验证身份。"""
    return ResearchClaim(**draft.model_dump(), claim_id=f"{unit.unit_id}.c{index}",
                         validation_status="needs_validation" if draft.claim_type in HYPOTHESIS_TYPES else "validated",
                         source_unit_ids=sorted({source_owners[key] for key in draft.source_claim_ids if key in source_owners}))
