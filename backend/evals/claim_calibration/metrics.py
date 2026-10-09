"""Claim/Citation 确定性指标。作者：xxx。"""

from __future__ import annotations

from collections.abc import Iterable

from evals.contracts import MetricResult

from .contracts import ClaimExtraction, ClaimMetricReport, ClaimReviewItem

METRIC_NAMES = ("citation_validity", "citation_completeness", "evidence_link_coverage")


def evaluate_claim_metrics(extraction: ClaimExtraction) -> ClaimMetricReport:
    """参数为冻结主张提取；返回有效性、完整性和 Evidence 链接覆盖指标。"""
    if extraction.status == "invalid_input":
        metrics = tuple(_unmeasured(name, "invalid_claim_extraction") for name in METRIC_NAMES)
        return ClaimMetricReport(status="invalid_input", reason=extraction.reason, metrics=metrics)
    if extraction.status == "missing":
        metrics = tuple(_unmeasured(name, extraction.reason) for name in METRIC_NAMES)
        return ClaimMetricReport(status="missing", reason=extraction.reason, metrics=metrics)
    if extraction.status == "not_measured":
        metrics = tuple(_not_measured(name, extraction.reason) for name in METRIC_NAMES)
        return ClaimMetricReport(status="not_measured", reason=extraction.reason, metrics=metrics)
    claims = extraction.items
    metrics = (
        citation_validity(claims), citation_completeness(claims), evidence_link_coverage(claims),
    )
    return ClaimMetricReport(status="measured", reason="deterministic_claim_observation", metrics=metrics,
                             claim_ids=tuple(item.claim_id for item in claims))


def citation_validity(claims: Iterable[ClaimReviewItem]) -> MetricResult:
    """参数为主张评审项；按实际引用逐条判断是否落到可追溯 Evidence。"""
    items = tuple(claims)
    cited = [(claim, evidence_id) for claim in items for evidence_id in claim.cited_evidence_ids]
    if not items:
        return _unmeasured("citation_validity", "no_claims")
    if not cited:
        applicable = [claim for claim in items if claim.requires_citation]
        if not applicable:
            return _unmeasured("citation_validity", "no_citations_and_no_paper_fact_claims")
        return MetricResult(name="citation_validity", status="measured", reason="claims_have_no_citations",
                            value=0.0, numerator=0.0, denominator=float(len(applicable)))
    if any("evidence_records" in claim.missing_fields for claim, _ in cited):
        return MetricResult(name="citation_validity", status="missing", reason="cited_evidence_records_missing")
    if any(evidence_id in claim.citable_evidence_ids and not claim.evidence_span_links.get(evidence_id)
           for claim, evidence_id in cited):
        return MetricResult(name="citation_validity", status="missing", reason="cited_source_mapping_missing")
    valid = sum(bool(evidence_id in claim.citable_evidence_ids and claim.evidence_span_links.get(evidence_id))
                for claim, evidence_id in cited)
    return _ratio("citation_validity", valid, len(cited), "cited_evidence_is_citable_and_maps_to_source")


def citation_completeness(claims: Iterable[ClaimReviewItem]) -> MetricResult:
    """参数为主张评审项；仅以需要论文引用的实际主张作为完整性分母。"""
    items = tuple(claims)
    applicable = tuple(item for item in items if item.requires_citation)
    if not items:
        return _unmeasured("citation_completeness", "claims_missing")
    if not applicable:
        return _unmeasured("citation_completeness", "no_paper_fact_claims")
    if any("answer_position" in item.missing_fields for item in applicable):
        return MetricResult(name="citation_completeness", status="missing", reason="paper_fact_answer_position_missing")
    cited = sum(bool(item.cited_evidence_ids) for item in applicable)
    return _ratio("citation_completeness", cited, len(applicable), "paper_fact_claim_citation_presence")


def evidence_link_coverage(claims: Iterable[ClaimReviewItem]) -> MetricResult:
    """参数为主张评审项；判断主张声明的支持 Evidence 是否有原文 span 链接。"""
    items = tuple(claims)
    links = [(claim, evidence_id) for claim in items for evidence_id in claim.supporting_evidence_ids]
    if not links:
        return _unmeasured("evidence_link_coverage", "no_supporting_evidence_links")
    if any("evidence_spans" in claim.missing_fields for claim, _ in links):
        return MetricResult(name="evidence_link_coverage", status="missing", reason="supporting_source_mapping_missing")
    linked = sum(bool(claim.evidence_span_links.get(evidence_id)) for claim, evidence_id in links)
    return _ratio("evidence_link_coverage", linked, len(links), "supporting_evidence_span_mapping")


def _ratio(name: str, numerator: int, denominator: int, reason: str) -> MetricResult:
    """参数为指标计数；返回保留分子分母的比例。"""
    return MetricResult(name=name, status="measured", reason=reason, value=numerator / denominator,
                        numerator=float(numerator), denominator=float(denominator))


def _unmeasured(name: str, reason: str) -> MetricResult:
    """参数为指标名和原因；返回不携带分数的四态结果。"""
    not_applicable = {"no_claims", "no_paper_fact_claims", "no_citations_and_no_paper_fact_claims", "no_supporting_evidence_links"}
    if reason in not_applicable:
        return MetricResult(name=name, status="not_measured", reason=reason)
    return MetricResult(name=name, status="missing", reason=reason)


def _not_measured(name: str, reason: str) -> MetricResult:
    """参数为指标名和不适用原因；返回退出分母的四态结果。"""
    return MetricResult(name=name, status="not_measured", reason=reason)
