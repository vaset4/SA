"""双人标注一致性、分歧率和仲裁率。作者：xxx。"""

from __future__ import annotations

from collections import Counter

from .contracts import AgreementReport, AnnotationRecord, DIMENSIONS


def compare_annotations(first: AnnotationRecord, second: AnnotationRecord) -> AgreementReport:
    """参数为同一运行的两份人工记录；返回逐维可重建一致性。"""
    if first.case_id != second.case_id or first.run_id != second.run_id:
        raise ValueError("annotations must target the same run")
    if first.rubric_version != second.rubric_version:
        raise ValueError("annotation rubric versions must match")
    if first.annotator_id == second.annotator_id:
        raise ValueError("agreement requires two annotators")
    agreeing = sum(first.dimensions[dimension] == second.dimensions[dimension] for dimension in DIMENSIONS)
    comparable = len(DIMENSIONS)
    disagreement = comparable - agreeing
    return AgreementReport(
        case_id=first.case_id, run_id=first.run_id, rubric_version=first.rubric_version,
        annotation_ids=(first.annotation_id, second.annotation_id),
        annotator_ids=(first.annotator_id, second.annotator_id), comparable_dimensions=comparable,
        agreeing_dimensions=agreeing, disagreement_dimensions=disagreement,
        agreement_rate=agreeing / comparable, disagreement_rate=disagreement / comparable,
        arbitration_required=disagreement > 0,
    )


def summarize_agreement(
    pairs: list[tuple[AnnotationRecord, AnnotationRecord]], arbitration_count: int = 0,
) -> dict[str, object]:
    """参数为双人标注对和仲裁数；按案例及维度汇总，不制造缺失分数。"""
    reports = [compare_annotations(first, second) for first, second in pairs]
    dimensions: Counter[str] = Counter()
    for first, second in pairs:
        for dimension in DIMENSIONS:
            dimensions[dimension] += int(first.dimensions[dimension] == second.dimensions[dimension])
    comparable = sum(item.comparable_dimensions for item in reports)
    agreeing = sum(item.agreeing_dimensions for item in reports)
    return {
        "pair_count": len(reports), "agreement_rate": agreeing / comparable if comparable else None,
        "disagreement_rate": (comparable - agreeing) / comparable if comparable else None,
        "arbitration_count": arbitration_count,
        "arbitration_rate": arbitration_count / len(reports) if reports else None,
        "dimension_agreement_counts": dict(dimensions),
        "pairs": [item.model_dump(mode="json") for item in reports],
    }
