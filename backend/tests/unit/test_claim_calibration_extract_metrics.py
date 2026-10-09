"""Claim/Citation 提取和确定性指标回归。作者：xxx。"""

from __future__ import annotations

from evals.claim_calibration.extract import extract_claims
from evals.claim_calibration.metrics import evaluate_claim_metrics
from evals.contracts import EvalCase, RunRecord, snapshot


def _record(*, run_mode: str = "agent_live", review_status: str = "reviewed", claims: list[dict] | None = None) -> dict:
    """参数为运行模式、审核状态和主张；返回冻结评测记录。"""
    actual_claims = claims if claims is not None else [{
        "claim_id": "c1", "text": "论文报告提升 10%。", "claim_type": "paper_supported",
        "validation_status": "validated", "supporting_evidence_ids": ["e1"],
        "related_evidence_ids": [], "source_unit_ids": [],
    }]
    observation = {
        "answer": "论文报告提升 10%。", "review_status": review_status,
        "research_unit_results": [{"unit_id": "u1", "status": "evidence_found", "claims": actual_claims}],
        "planner_final_blocks": [{"role": "answer", "text": "论文报告提升 10%。", "claim_ids": ["c1"], "evidence_ids": ["e1"]}],
        "evidence_records": [{"evidence_id": "e1", "layer": "direct_text", "usable_as_citation": True, "span_ids": ["s1"]}],
    }
    case = EvalCase(case_id="case-1", case_version="v1", source_type="agent_scenario",
                    input_snapshot=snapshot({"question": "结果如何"}), truth_snapshot=snapshot({}))
    run = RunRecord(
        run_id="run-1", case_id="case-1", case_version="v1", run_mode=run_mode,
        input_snapshot=case.input_snapshot, observation_snapshot=snapshot(observation), provenance="captured",
    )
    return {"eval_case": case.model_dump(mode="json"), "run_record": run.model_dump(mode="json")}


def _benchmark() -> dict:
    """无参数；返回最小 span 身份基准，Evidence 记录直接指向该 span。"""
    from evals.contracts import snapshot_hash

    quote = "论文报告提升 10%。"
    return {
        "dataset_version": "v1", "review_status": "reviewed", "sources": [],
        "spans": [{"span_id": "s1", "source_snapshot_id": "a" * 64, "block_id": "b1",
                   "start": 0, "end": len(quote), "quote": quote, "quote_sha256": snapshot_hash(quote)}],
        "groups": [], "index_snapshots": [],
    }


def test_extract_and_measure_citation_dimensions_separately() -> None:
    """无参数；验证有效性、完整性和原文链接覆盖各自保留分母。"""
    extraction = extract_claims(_record(), _benchmark())
    metrics = {item.name: item for item in evaluate_claim_metrics(extraction).metrics}
    assert extraction.human_quality_eligible is True
    assert extraction.items[0].answer_positions[0].block_index == 0
    assert metrics["citation_validity"].value == 1.0
    assert metrics["citation_completeness"].denominator == 1.0
    assert metrics["evidence_link_coverage"].value == 1.0


def test_missing_citation_can_be_incomplete_without_changing_validity_semantics() -> None:
    """无参数；验证无引用论文事实进入 completeness 分母且 validity 明确为零。"""
    record = _record()
    observation = {
        "answer": "论文报告提升 10%。", "review_status": "reviewed",
        "research_unit_results": [{"unit_id": "u1", "status": "evidence_found", "claims": [{
            "claim_id": "c1", "text": "论文报告提升 10%。", "claim_type": "paper_supported",
            "validation_status": "validated", "supporting_evidence_ids": ["e1"],
        }]}],
        "planner_final_blocks": [{"role": "answer", "text": "论文报告提升 10%。", "claim_ids": ["c1"], "evidence_ids": []}],
        "evidence_records": [{"evidence_id": "e1", "layer": "direct_text", "usable_as_citation": True, "span_ids": ["s1"]}],
    }
    record["run_record"]["observation_snapshot"] = snapshot(observation)
    extraction = extract_claims(record, _benchmark())
    metrics = {item.name: item for item in evaluate_claim_metrics(extraction).metrics}
    assert metrics["citation_validity"].value == 0.0
    assert metrics["citation_completeness"].value == 0.0


def test_valid_citation_does_not_claim_semantic_support() -> None:
    """无参数；验证可追溯引用即使不是声明支持证据也只影响 validity。"""
    record = _record()
    observation = {
        "answer": "论文报告提升 10%。", "review_status": "reviewed",
        "research_unit_results": [{"unit_id": "u1", "status": "evidence_found", "claims": [{
            "claim_id": "c1", "text": "论文报告提升 10%。", "claim_type": "paper_supported",
            "validation_status": "validated", "supporting_evidence_ids": ["e1"],
        }]}],
        "planner_final_blocks": [{"role": "answer", "text": "论文报告提升 10%。", "claim_ids": ["c1"], "evidence_ids": ["e2"]}],
        "evidence_records": [
            {"evidence_id": "e1", "layer": "direct_text", "usable_as_citation": True, "span_ids": ["s1"]},
            {"evidence_id": "e2", "layer": "direct_text", "usable_as_citation": True, "span_ids": ["s1"]},
        ],
    }
    record["run_record"]["observation_snapshot"] = snapshot(observation)
    extraction = extract_claims(record, _benchmark())
    metrics = {item.name: item for item in evaluate_claim_metrics(extraction).metrics}
    assert extraction.items[0].cited_evidence_ids != extraction.items[0].supporting_evidence_ids
    assert metrics["citation_validity"].value == 1.0
    assert metrics["citation_completeness"].value == 1.0


def test_no_paper_fact_claims_make_completeness_not_applicable() -> None:
    """无参数；验证假设主张不会被错误放入论文引用分母。"""
    extraction = extract_claims(_record(claims=[{
        "claim_id": "c1", "text": "可以进一步实验。", "claim_type": "model_prior_hypothesis",
        "validation_status": "needs_validation", "supporting_evidence_ids": [],
    }]), _benchmark())
    result = evaluate_claim_metrics(extraction)
    completeness = next(item for item in result.metrics if item.name == "citation_completeness")
    assert completeness.status == "not_measured"
    assert completeness.reason == "no_paper_fact_claims"


def test_missing_claim_extraction_does_not_guess_from_answer() -> None:
    """无参数；验证只有回答文本而无结构化主张时保持 missing。"""
    record = _record()
    record["run_record"]["observation_snapshot"] = snapshot({"answer": "看起来论文提升了 10%。"})
    extraction = extract_claims(record)
    assert extraction.status == "missing"
    assert extraction.items == ()


def test_fixture_and_candidate_are_diagnostic_only() -> None:
    """无参数；验证 fixture 或未审核回答不进入人工质量分母。"""
    assert extract_claims(_record(run_mode="agent_fixture"), _benchmark()).human_quality_eligible is False
    assert extract_claims(_record(review_status="candidate"), _benchmark()).human_quality_eligible is False


def test_not_measured_extraction_does_not_become_measured_metrics() -> None:
    """无参数；验证不适用提取结果保持 not_measured 四态。"""
    extraction = extract_claims(_record(claims=[{
        "claim_id": "c1", "text": "可以进一步实验。", "claim_type": "model_prior_hypothesis",
        "validation_status": "needs_validation", "supporting_evidence_ids": [],
    }]), _benchmark())
    extraction = extraction.model_copy(update={"status": "not_measured", "reason": "case_not_applicable"})
    result = evaluate_claim_metrics(extraction)
    assert result.status == "not_measured"
    assert all(item.status == "not_measured" for item in result.metrics)
