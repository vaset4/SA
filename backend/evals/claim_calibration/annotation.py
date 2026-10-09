"""人工六维 rubric、标注和仲裁记录。作者：xxx。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import (
    AnnotationRecord, ArbitrationRecord, ClaimAnnotation, ClaimExtraction, DIMENSIONS,
    HumanDimension, HumanScore, RUBRIC_VERSION, RubricAnchor, RubricDefinition,
    RubricDimensionDefinition, annotation_is_eligible,
)

_RUBRIC_TEXT: tuple[tuple[HumanDimension, str, tuple[str, str, str]], ...] = (
    ("task_completion", "任务完成", ("未完成用户研究目标", "完成部分目标或存在关键遗漏", "完整完成目标且交付形式可用")),
    ("factual_correctness", "事实正确", ("关键事实错误或关系颠倒", "次要错误或条件表达不完整", "关键事实、单位、条件和关系正确")),
    ("evidence_attribution", "证据与归因", ("证据不支持主张或来源错误", "部分主张支持不足或归因不清", "主张均由对应可复核证据支持")),
    ("coverage_synthesis", "覆盖与综合", ("遗漏关键证据或错误调和冲突", "覆盖主要证据但综合不完整", "覆盖关键证据并公平呈现冲突条件")),
    ("epistemic_honesty", "认识诚实", ("将不确定或无证据内容写成确定事实", "部分限定语不足", "明确区分事实、推断、假设和证据不足")),
    ("research_usefulness", "研究可用性", ("结论无法支持后续研究行动", "提供有限参考但需大量返工", "结论清晰且能支持下一步判断或行动")),
)


def default_rubric() -> RubricDefinition:
    """无参数；返回版本化六维 0-2 分人工评分锚点。"""
    dimensions = tuple(RubricDimensionDefinition(
        dimension=dimension, title_zh=title, anchors=_anchors(criteria),
    ) for dimension, title, criteria in _RUBRIC_TEXT)
    return RubricDefinition(dimensions=dimensions)


def _anchors(criteria: tuple[str, str, str]) -> tuple[RubricAnchor, RubricAnchor, RubricAnchor]:
    """参数为三个判定文本；返回按 0、1、2 固定顺序的评分锚点。"""
    return (
        RubricAnchor(score=0, criteria=criteria[0]),
        RubricAnchor(score=1, criteria=criteria[1]),
        RubricAnchor(score=2, criteria=criteria[2]),
    )


def create_annotation(
    extraction: ClaimExtraction, *, annotation_id: str, annotator_id: str,
    dimensions: Mapping[HumanDimension, HumanScore], reasons: Mapping[HumanDimension, str],
    claim_annotations: tuple[ClaimAnnotation, ...] = (), rubric_version: str = RUBRIC_VERSION,
) -> AnnotationRecord:
    """参数为冻结提取、标注身份和六维评分；返回独立人工记录。"""
    if not annotation_is_eligible(extraction):
        raise ValueError("human annotation requires reviewed agent_live measured run")
    if not annotator_id.strip():
        raise ValueError("annotator_id is required")
    return AnnotationRecord(
        annotation_id=annotation_id, case_id=extraction.case_id, run_id=extraction.run_id,
        annotator_id=annotator_id, rubric_version=rubric_version, dimensions=dict(dimensions),
        claim_annotations=claim_annotations, reasons=dict(reasons),
    )


def create_arbitration(
    annotations: tuple[AnnotationRecord, AnnotationRecord], *, arbitration_id: str,
    arbiter_id: str, resolved_dimensions: Mapping[HumanDimension, HumanScore],
    resolved_claims: tuple[ClaimAnnotation, ...] = (), reason: str = "",
) -> ArbitrationRecord:
    """参数为两份独立标注和仲裁结论；返回保留分歧来源的仲裁记录。"""
    first, second = annotations
    if first.case_id != second.case_id or first.run_id != second.run_id:
        raise ValueError("annotations must target the same run")
    if first.annotator_id == second.annotator_id:
        raise ValueError("arbitration requires two annotators")
    if first.rubric_version != second.rubric_version:
        raise ValueError("annotation rubric versions must match")
    disagreements = tuple(dimension for dimension in DIMENSIONS if first.dimensions[dimension] != second.dimensions[dimension])
    return ArbitrationRecord(
        arbitration_id=arbitration_id, case_id=first.case_id, run_id=first.run_id,
        annotation_ids=(first.annotation_id, second.annotation_id), arbiter_id=arbiter_id,
        rubric_version=first.rubric_version, disagreements=disagreements,
        resolved_dimensions=dict(resolved_dimensions), resolved_claims=resolved_claims, reason=reason,
    )


def annotation_summary(annotation: AnnotationRecord) -> dict[str, Any]:
    """参数为单份人工记录；返回不丢失原分的可读摘要。"""
    return {
        "annotation_id": annotation.annotation_id, "annotator_id": annotation.annotator_id,
        "rubric_version": annotation.rubric_version, "dimensions": dict(annotation.dimensions),
        "claim_count": len(annotation.claim_annotations), "reasons": dict(annotation.reasons),
    }
