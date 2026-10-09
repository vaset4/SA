"""人工 rubric、分歧、仲裁和 judge 校准回归。作者：xxx。"""

from __future__ import annotations

import pytest

from evals.claim_calibration.agreement import compare_annotations, summarize_agreement
from evals.claim_calibration.annotation import create_annotation, create_arbitration, default_rubric
from evals.claim_calibration.contracts import ClaimExtraction, DIMENSIONS
from evals.claim_calibration.judge import compare_judge_to_human, record_judge_result


def _extraction(*, eligible: bool = True) -> ClaimExtraction:
    """参数为人工资格；返回可用于标注合同测试的提取结果。"""
    return ClaimExtraction(
        case_id="case-1", run_id="run-1", run_mode="agent_live" if eligible else "agent_fixture",
        review_status="reviewed", human_quality_eligible=eligible, status="measured",
        reason="claims_and_answer_positions_extracted", items=(), projection_hash="0" * 64,
    )


def _scores(value: int = 2) -> dict:
    """参数为统一分值；返回六维 rubric 分数。"""
    return {dimension: value for dimension in DIMENSIONS}


def _reasons() -> dict:
    """无参数；返回六维人工判定理由。"""
    return {dimension: f"{dimension} 已核对" for dimension in DIMENSIONS}


def test_default_rubric_has_six_ordered_dimensions_and_anchors() -> None:
    """无参数；验证六维 rubric 的每个维度都有 0、1、2 判定锚点。"""
    rubric = default_rubric()
    assert tuple(item.dimension for item in rubric.dimensions) == DIMENSIONS
    assert all(tuple(anchor.score for anchor in item.anchors) == (0, 1, 2) for item in rubric.dimensions)


def test_annotation_requires_reviewed_live_answer_and_all_dimensions() -> None:
    """无参数；验证 fixture 不能冒充人工答案质量且六维不可缺项。"""
    with pytest.raises(ValueError, match="reviewed agent_live"):
        create_annotation(_extraction(eligible=False), annotation_id="a1", annotator_id="u1",
                          dimensions=_scores(), reasons=_reasons())
    incomplete = _scores()
    incomplete.pop("research_usefulness")
    with pytest.raises(ValueError, match="six rubric"):
        create_annotation(_extraction(), annotation_id="a1", annotator_id="u1",
                          dimensions=incomplete, reasons=_reasons())


def test_two_annotators_preserve_disagreement_and_arbitration() -> None:
    """无参数；验证双人原分、分歧率和仲裁留痕可重建。"""
    first = create_annotation(_extraction(), annotation_id="a1", annotator_id="u1",
                              dimensions=_scores(), reasons=_reasons())
    second_scores = _scores()
    second_scores["factual_correctness"] = 0
    second = create_annotation(_extraction(), annotation_id="a2", annotator_id="u2",
                               dimensions=second_scores, reasons=_reasons())
    agreement = compare_annotations(first, second)
    arbitration = create_arbitration(
        (first, second), arbitration_id="arb1", arbiter_id="u3",
        resolved_dimensions={"factual_correctness": 1}, reason="核对单位和否定条件",
    )
    summary = summarize_agreement([(first, second)], arbitration_count=1)
    assert agreement.disagreement_dimensions == 1
    assert arbitration.disagreements == ("factual_correctness",)
    assert summary["arbitration_rate"] == 1.0


def test_same_person_cannot_count_as_two_annotators() -> None:
    """无参数；验证同一标注者重复评分不能形成双人一致性。"""
    first = create_annotation(_extraction(), annotation_id="a1", annotator_id="u1",
                              dimensions=_scores(), reasons=_reasons())
    second = create_annotation(_extraction(), annotation_id="a2", annotator_id="u1",
                               dimensions=_scores(), reasons=_reasons())
    with pytest.raises(ValueError, match="two annotators"):
        compare_annotations(first, second)


def test_judge_remains_diagnostic_and_keeps_versioned_snapshots() -> None:
    """无参数；验证自动 judge 只做诊断且可与人工逐维对照。"""
    annotation = create_annotation(_extraction(), annotation_id="a1", annotator_id="u1",
                                   dimensions=_scores(), reasons=_reasons())
    judge = record_judge_result(
        judge_id="j1", case_id="case-1", run_id="run-1", request={"claim_id": "c1"},
        response={"score": 2}, dimensions=_scores(), error_labels=("false_resolution",),
    )
    comparison = compare_judge_to_human(judge, annotation)
    assert judge.status == "diagnostic"
    assert judge.hard_gate_eligible is False
    assert comparison["agreement_rate"] == 1.0
    assert comparison["error_labels"] == ["false_resolution"]


def test_rubric_versions_cannot_be_compared_or_arbitrated_silently() -> None:
    """无参数；验证 rubric 升版必须产生新记录，不能静默混算历史分数。"""
    first = create_annotation(_extraction(), annotation_id="a1", annotator_id="u1",
                              dimensions=_scores(), reasons=_reasons(), rubric_version="v1")
    second = create_annotation(_extraction(), annotation_id="a2", annotator_id="u2",
                               dimensions=_scores(), reasons=_reasons(), rubric_version="v2")
    with pytest.raises(ValueError, match="versions must match"):
        compare_annotations(first, second)
