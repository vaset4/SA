"""评测真值与运行观察隔离回归。作者：xxx。"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from evals.ragas import evals, rag_adapter
from evals.ragas.eval_contracts import compute_golden_evidence_presence
from evals.contracts import freeze_record, rebuild_grading_input, record_contracts


def _sample() -> dict:
    """返回带独立金标且实际未命中的样本；无参数。"""
    return {
        "question": "Q",
        "retrieved_contexts": ["unrelated context"],
        "response": "actual answer",
        "reference_answer": "reference",
        "golden_evidence": ["target-evidence"],
    }


@pytest.mark.parametrize("metadata", [
    {},
    {"paper_ids": ["target-evidence"]},
    {"notes": {"id": "target-evidence"}},
    {"critical_facts": [{"must_include": ["target-evidence"]}]},
])
def test_expected_or_scope_metadata_never_counts_as_actual_evidence(metadata: dict) -> None:
    """参数为非观察元信息；验证其不能使无关检索命中金标。"""
    record = rag_adapter.normalize_eval_record({**_sample(), "metadata": metadata})
    assert compute_golden_evidence_presence(record, record.get("metadata", {})) == 0.0


def test_live_input_does_not_mutate_nested_scope() -> None:
    """无参数；构造 live 输入后原始范围仍与冻结样本相同。"""
    sample = {"question": "Q", "project_id": "p", "structured_query": {"resolved_scope": {}}}
    before = deepcopy(sample)
    rag_adapter.LiveRAGAdapter()._build_live_input(sample)
    assert sample == before


def test_live_input_rejects_truth_in_nested_execution_options() -> None:
    """无参数；嵌套执行选项中的标准答案必须显式报错。"""
    sample = {"question": "Q", "structured_query": {"rubric": {"answer": "secret"}}}
    with pytest.raises(ValueError, match="rubric"):
        rag_adapter.LiveRAGAdapter()._build_live_input(sample)


def test_live_record_preserves_empty_observations() -> None:
    """无参数；空回答和空检索不能被失败文本或占位文本替代。"""
    record = rag_adapter.LiveRAGAdapter()._build_live_record(
        _sample(), {"answer": "", "failure_reason": "retrieval failed"},
    )
    assert record["response"] == ""
    assert record["retrieved_contexts"] == []


def test_grader_output_cannot_replace_observed_answer_or_evidence() -> None:
    """无参数；评分器只返回指标，不能改写实际答案与证据。"""
    result = evals._normalize_evaluation_result([_sample()], {
        "metrics": {"faithfulness": 0.9},
        "records": [{"faithfulness": 0.9, "response": "forged answer",
                     "metadata": {"citations": [{"evidence_id": "target-evidence"}]}}],
    })
    assert result["records"][0]["response"] == "actual answer"
    assert result["records"][0]["golden_evidence_presence_rate"] == 0.0


def _graded(record: dict[str, Any]) -> dict[str, Any]:
    """参数为案例；返回一次无需外部服务的实际评分记录。"""
    return evals._normalize_evaluation_result([record], {
        "metrics": {"faithfulness": 0.9}, "records": [{"faithfulness": 0.9}],
    })["records"][0]


def test_snapshots_rebuild_and_cannot_be_changed_through_nested_aliases() -> None:
    """无参数；保存后重建评分输入，修改源对象或投影均不影响冻结观察。"""
    sample = {**_sample(), "metadata": {"citations": [{"evidence_id": "actual"}]}}
    frozen = freeze_record(sample)
    sample["metadata"]["citations"][0]["evidence_id"] = "mutated"
    graded = _graded(frozen)
    restored = json.loads(json.dumps(graded))
    projection = rebuild_grading_input(restored)
    assert projection["observation"]["metadata"]["citations"][0]["evidence_id"] == "actual"
    projection["observation"]["metadata"]["citations"].clear()
    assert rebuild_grading_input(restored)["observation"]["metadata"]["citations"]
    assert _graded(restored)["grading_result"] == graded["grading_result"]


def test_modified_snapshot_or_identity_is_rejected_on_rebuild() -> None:
    """无参数；检查持久化运行身份和输入快照的篡改可被定位。"""
    graded = _graded(_sample())
    altered = deepcopy(graded)
    altered["run_record"]["observation_snapshot"] = json.dumps({"response": "changed"})
    with pytest.raises(ValueError, match="checksum"):
        rebuild_grading_input(altered)
    altered["run_record"]["case_id"] = "different-case"
    with pytest.raises(ValueError, match="identity"):
        record_contracts(altered)


@pytest.mark.parametrize(("golden", "status"), [
    (None, "missing"), ([], "not_measured"), (["target"], "measured"), ({"bad": True}, "invalid_input"),
])
def test_golden_measurements_distinguish_absence_and_actual_zero(golden: Any, status: str) -> None:
    """参数为真值与期望状态；实际空检索为零分，缺真值不能冒充零分。"""
    sample = {**_sample(), "retrieved_contexts": [], "golden_evidence": golden}
    graded = _graded(sample)
    metric = next(item for item in graded["grading_result"]["measurements"] if item["name"] == "golden_evidence_presence_rate")
    assert metric["status"] == status
    assert metric["value"] == (0.0 if status == "measured" else None)
    if status == "measured":
        assert (metric["numerator"], metric["denominator"]) == (0, 1)


def test_bad_fact_pattern_is_invalid_input_instead_of_a_passing_score() -> None:
    """无参数；非法事实正则产生可定位的数据错误。"""
    graded = _graded({**_sample(), "critical_facts": [{"must_include": [{"regex": "["}]}]})
    metrics = {item["name"]: item for item in graded["grading_result"]["measurements"]}
    assert metrics["critical_fact_coverage"]["status"] == "invalid_input"
    assert "critical_fact_coverage" not in graded


def test_grader_mutation_cannot_change_saved_actual_run(tmp_path: Path) -> None:
    """参数为临时目录；真实 runner 冻结后，评分器修改投影不污染记录。"""
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")

    def grader(records: list[dict[str, Any]]) -> dict[str, Any]:
        records[0]["response"] = "forged answer"
        records[0]["retrieved_contexts"] = ["target-evidence"]
        return {"metrics": {"faithfulness": 0.9}, "records": [{"faithfulness": 0.9}]}

    report = evals.run_evaluation(dataset_path=dataset, output_dir=tmp_path / "reports", evaluator=grader)
    row = report["records"][0]
    assert row["response"] == "actual answer"
    assert row["golden_evidence_presence_rate"] == 0.0
    assert rebuild_grading_input(json.loads(Path(report["json_report_path"]).read_text(encoding="utf-8"))["records"][0])
    assert "Measurement Coverage" in Path(report["markdown_report_path"]).read_text(encoding="utf-8")


def test_suite_uses_measured_counts_for_optional_metric() -> None:
    """无参数；不同数据集的有效样本数决定指标均值，不使用数据集数量平均。"""
    rows = [
        {"project_metrics": {"golden_evidence_presence_rate": 1.0},
         "measurement_coverage": {"golden_evidence_presence_rate": {"measured_count": 1}}},
        {"project_metrics": {"golden_evidence_presence_rate": 0.0},
         "measurement_coverage": {"golden_evidence_presence_rate": {"measured_count": 3}}},
    ]
    metrics = evals._mean_metric_maps(rows, field="project_metrics", metric_names=("golden_evidence_presence_rate",))
    assert metrics["golden_evidence_presence_rate"] == 0.25


def test_empty_actual_run_is_reported_without_calling_quality_provider(tmp_path: Path) -> None:
    """参数为临时目录；实际空结果保留报告和缺口，不生成假的 RAGAS 分数。"""
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text(json.dumps({**_sample(), "response": "", "retrieved_contexts": []}) + "\n", encoding="utf-8")
    report = evals.run_evaluation(dataset_path=dataset, output_dir=tmp_path / "reports")
    assert report["records"][0]["response"] == ""
    assert report["records"][0]["golden_evidence_presence_rate"] == 0.0
    assert report["measurement_coverage"]["faithfulness"]["missing_count"] == 1
    assert not report["gate"]["passed"]


def test_loader_preserves_explicit_empty_truth_requirements(tmp_path: Path) -> None:
    """参数为临时目录；真正加载旧格式时保留明确空要求，不能转成缺标注。"""
    sample = {**_sample(), "golden_evidence": [], "critical_facts": []}
    dataset = tmp_path / "empty-truth.jsonl"
    dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    replay = rag_adapter.load_jsonl_dataset(dataset)[0]
    live = rag_adapter.load_live_jsonl_dataset(dataset)[0]
    for loaded in (replay, live):
        assert loaded["golden_evidence"] == []
        assert loaded["critical_facts"] == []
    result = _graded(replay)
    metric = next(item for item in result["grading_result"]["measurements"] if item["name"] == "golden_evidence_presence_rate")
    assert metric["status"] == "not_measured"


def test_structured_context_cannot_stringify_truth_into_retrieved_text() -> None:
    """无参数；结构对象不能把其标注元信息串成实际检索文本。"""
    sample = {**_sample(), "retrieved_contexts": [{"text": "irrelevant", "metadata": {"golden_evidence": ["target-evidence"]}}]}
    with pytest.raises(ValueError, match="text strings"):
        rag_adapter.normalize_eval_record(sample)


def test_unrecognized_fact_rule_is_invalid_instead_of_full_score() -> None:
    """无参数；只有名称没有规则的事实项不能得到满分。"""
    result = _graded({**_sample(), "critical_facts": [{"name": "empty-rule"}]})
    metrics = {item["name"]: item for item in result["grading_result"]["measurements"]}
    assert metrics["fact_contract_score"]["status"] == "invalid_input"
    assert "fact_contract_score" not in result


def test_metric_regrading_removes_old_scalar_scores() -> None:
    """无参数；本次评分缺数据时不保留上次兼容标量作为实际成绩。"""
    old = _graded(_sample())
    new = evals._normalize_evaluation_result([old], {"records": [{}]})["records"][0]
    assert "faithfulness" not in new
    metric = next(item for item in new["grading_result"]["measurements"] if item["name"] == "faithfulness")
    assert metric["status"] == "missing"


def test_observation_keys_do_not_count_as_evidence_text() -> None:
    """无参数；空观察的字段名不属于可检索的证据内容。"""
    sample = {**_sample(), "golden_evidence": ["retrieved_contexts"], "retrieved_contexts": []}
    assert compute_golden_evidence_presence(sample, {}) == 0.0


def test_live_trace_is_captured_and_dataset_runtime_metadata_is_ignored() -> None:
    """无参数；保留图实际 trace，样本中的伪 Evidence 不进入运行记录。"""
    sample = {**_sample(), "metadata": {"evidence_records": [{"evidence_id": "target-evidence"}]}}
    observed = {"answer": "real answer", "trace": {"events": []}, "planner_trace": {"status": "ok"},
                "turn_outcome": {"status": "done"}, "source_status_reasons": {"milvus": "empty"}}
    record = rag_adapter.normalize_eval_record(rag_adapter.LiveRAGAdapter()._build_live_record(sample, observed))
    frozen = freeze_record(record, mode="live", execution_input={"user_query": "Q"})
    case, run = record_contracts(frozen)
    actual = json.loads(run.observation_snapshot)
    assert actual["trace"] == observed["trace"]
    assert actual["turn_outcome"] == observed["turn_outcome"]
    assert actual["source_status_reasons"] == observed["source_status_reasons"]
    assert "evidence_records" in run.missing_fields
    assert "evidence_records" not in actual["metadata"]
    assert case.case_id == run.case_id


def test_whitespace_observations_remain_empty_after_loader_normalization(tmp_path: Path) -> None:
    """参数为临时目录；空白输入规范化后不能被原始别名覆回并触发评分模型。"""
    dataset = tmp_path / "whitespace.jsonl"
    dataset.write_text(json.dumps({**_sample(), "response": "   ", "retrieved_contexts": [" \t "]}) + "\n", encoding="utf-8")
    record = rag_adapter.load_jsonl_dataset(dataset)[0]
    assert record["response"] == ""
    assert record["retrieved_contexts"] == []
    assert not evals.ragas_eligible(record)
