"""Test builders for research planner core contracts."""

from __future__ import annotations

from typing import Any, Sequence

from app.models.research_planner_contracts import ClaimSupportSpan, ResearchClaim, ResearchClaimDraft, ResearchUnit, ResearchUnitResult
from app.services.internal_research.research_planner_consumer_budget import evidence_text_field, evidence_version, prepare_consumer_input


def supporting_spans(records: list[dict[str, Any]], ids: list[str] | None = None) -> list[ClaimSupportSpan]:
    """参数为测试明确选中的真实原文；返回可核对片段，禁止为未知 ID 补造来源。"""
    spans = []
    for record in records:
        if ids is not None and record["evidence_id"] not in ids:
            continue
        field, text = evidence_text_field(record)
        if text:
            spans.append(ClaimSupportSpan(evidence_id=record["evidence_id"], evidence_version=evidence_version(record),
                         field=field, start=0, end=len(text), quote=text))
    return spans


def result_with_exposure(
    unit: ResearchUnit, records: list[dict[str, Any]], claims: list[ResearchClaim], *,
    dependencies: Sequence[ResearchUnitResult] = (),
) -> ResearchUnitResult:
    """参数为测试明确原文、主张与依赖；保存草拟时的实际投影供真实支持解析复核。"""
    prepared = prepare_consumer_input(records, [claim for result in dependencies for claim in result.claims],
                                     evidence_max_chars=100000, prompt_version="test-explicit-v3")
    return ResearchUnitResult(unit_id=unit.unit_id, status="evidence_found", claims=claims,
                              evidence_ids=[record["evidence_id"] for record in records], metadata={"provenance": {
                                  "unit": unit.model_dump(mode="json"), "exposure": prepared.exposure.model_dump(mode="json"),
                                  "execution_identity": None,
                              }})


def explicit_claim_provider(unit: Any, records: list[dict[str, Any]], dependencies: list[Any], **kwargs: Any) -> list[ResearchClaimDraft]:
    """参数为正式入口送入的测试材料与依赖；外部模型替身明确选取可见来源。"""
    prepared = kwargs["prepared_input"]
    visible_ids = {item.evidence_id for item in prepared.exposure.evidence}
    selected = [record for record in records if record["evidence_id"] in visible_ids]
    allowed = unit.claim_policy.allowed_claim_types
    if "paper_supported" in allowed and selected:
        kind = "paper_supported"
        selected = [record for record in selected if record.get("usable_as_citation") and record.get("layer") == "direct_text"]
    elif "evidence_grounded_inference" in allowed and (selected or dependencies):
        kind = "evidence_grounded_inference"
    elif "model_prior_hypothesis" in allowed:
        return [ResearchClaimDraft(text="待验证方案", claim_type="model_prior_hypothesis",
                                  validation_tasks=["实验验证"], addressed_facet_ids=unit.covered_facet_ids)]
    else:
        return []
    spans = []
    for item in prepared.exposure.evidence:
        if item.evidence_id not in {record["evidence_id"] for record in selected}:
            continue
        record = next(record for record in selected if record["evidence_id"] == item.evidence_id)
        spans.append(ClaimSupportSpan(evidence_id=item.evidence_id, evidence_version=item.evidence_version,
                     field=item.field, start=item.start, end=item.end, quote=record[item.field][item.start:item.end]))
    return [ResearchClaimDraft(text="依据已选来源的结论", claim_type=kind,
            supporting_evidence_ids=[record["evidence_id"] for record in selected], supporting_spans=spans,
            source_claim_ids=[claim.claim_id for result in dependencies for claim in result.claims],
            reasoning_summary="依据已选的原文与上游结论", addressed_facet_ids=unit.covered_facet_ids)]


def make_policy(
    source_type: str = "local_direct_text",
    *,
    paper_ids: list[str] | None = None,
    locator_queries: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "target_paper_ids": paper_ids or ["paper-a"],
        "required_sections": ["Method"],
        "evidence_dimensions": ["mechanism"],
        "locator_queries": locator_queries or ["attention mechanism"],
        "min_citations_per_target": 1,
    }


def make_claim_policy(
    allowed: list[str] | None = None,
    *,
    citation_required: bool = True,
    bounded: bool = True,
) -> dict[str, Any]:
    return {
        "allowed_claim_types": allowed or ["paper_supported", "evidence_grounded_inference"],
        "citation_required": citation_required,
        "bounded_inference_allowed": bounded,
    }


def make_unit(
    unit_id: str = "draft-a",
    *,
    goal: str = "Verify the paper evidence",
    action: str = "locate direct paper evidence",
    source_type: str = "local_direct_text",
    allowed_claims: list[str] | None = None,
    deps: list[dict[str, Any]] | None = None,
    facets: list[str] | None = None,
    paper_ids: list[str] | None = None,
    operation_family: str | None = "evidence_lookup",
    output_role: str | None = None,
) -> dict[str, Any]:
    unit = {
        "unit_id": unit_id,
        "goal": goal,
        "research_action": action,
        "operation_family": operation_family,
        "evidence_policy": make_policy(source_type, paper_ids=paper_ids),
        "claim_policy": make_claim_policy(allowed_claims),
        "dependencies": [{"required_product": "claims", **dep} for dep in deps or []],
        "expected_output": {
            "deliverable_kind": "claims",
            "output_goal": "one validated finding",
            "required_claim_types": (allowed_claims or ["paper_supported"])[:1],
            "answer_section_ids": [],
            "validation_tasks": [],
        },
        "covered_facet_ids": ["f1"] if facets is None else facets,
    }
    if output_role is not None:
        unit["output_role"] = output_role
    return unit


def make_facet(facet_id: str, source: str = "explicit") -> dict[str, Any]:
    return {
        "facet_id": facet_id,
        "label": f"Facet {facet_id}",
        "source": source,
        "user_text_span": f"text {facet_id}",
        "rationale": "requested by user",
        "required": source == "explicit",
        "deliverable_kind": "claims",
        "required_paper_ids": [],
        "required_claim_types": [],
    }


def make_draft(
    units: list[dict[str, Any]] | None = None,
    *,
    plan_type: str | None = None,
    explicit_facets: list[dict[str, Any]] | None = None,
    inferred_facets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    unit_list = units or [make_unit(facets=["f1"])]
    actual_type = "single_unit" if len(unit_list) == 1 else "multi_unit"
    return {
        "contract_version": "research-planner-v3",
        "question": "How do the papers use attention?",
        "plan_type": plan_type or actual_type,
        "units": unit_list,
        "coverage_review": {
            "explicit_facets": explicit_facets or [make_facet("f1")],
            "inferred_facets": inferred_facets or [],
            "covered_facet_ids": [],
            "missing_explicit_facet_ids": [],
            "missing_inferred_facet_ids": [],
            "unit_facet_map": {},
            "repair_status": "not_needed",
        },
        "metadata": {
            "analyzer_plan_type": plan_type or actual_type,
            "analyzer_reasoning": "fixture",
            "normalized_unit_id_map": {},
            "validation_errors": [],
            "repairs": [],
            "repair_attempted": False,
            "repaired": False,
        },
    }
