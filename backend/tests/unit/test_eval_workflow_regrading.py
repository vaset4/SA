"""【流程评测】【复评回归】新旧政策、独立指标与不可覆盖离线命令。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.contracts import MetricResult, snapshot
from evals.suites.adapters import load_suite_sources
from evals.suites.aggregate import _false_success_check
from evals.suites.execute import _write_case_artifacts
from evals.suites.execution_contracts import PreparedCase, PreparedJob
from evals.suites.policy import MeasurementPolicy, workflow_policy
from evals.suites.run_eval import main
from test_agent_workflow import _record
from test_agent_workflow_delivery import current_turn, no_network as no_network
from test_eval_suites_contracts_adapters import make_slots
from test_eval_workflow_suite import _workflow_manifest


@pytest.mark.parametrize("version", ["workflow-retrieval-v1", "workflow-retrieval-v2"])
def test_worker_uses_exact_frozen_policy_version(tmp_path: Path, version: str) -> None:
    """参数为旧新策略；正式worker输出使用冻结版本，加载旧JSON不能静默改写评分版本。"""
    policy = workflow_policy(version=version)
    original = policy.model_dump(mode="json")
    assert MeasurementPolicy.model_validate(original).model_dump(mode="json") == original
    record = _record([current_turn()])
    case = record["eval_case"]
    job = PreparedJob(job_id="agent", kind="agent", cases=(PreparedCase(case_id=case["case_id"],
                      case_version=case["case_version"], record_snapshot=snapshot(record["agent_case"])),),
                      slots=tuple(slot.model_copy(update={"run_id": None}) for slot in make_slots((record,))))
    path, _ = _write_case_artifacts(record, job, output=tmp_path, policy=policy)
    row = json.loads(path.read_text(encoding="utf-8"))["rows"][0]
    assert row["grader_version"] == policy.workflow_grader_version
    metrics = {metric["name"]: metric for metric in row["measurements"]}
    assert metrics["error_handling_correctness"]["value"] == (0 if version.endswith("v1") else 1)


def test_policy_rejects_mixed_semantic_versions() -> None:
    """新工作流评分必须配新政策，禁止用V1标签掩盖已改变语义。"""
    raw = workflow_policy(version="workflow-retrieval-v1").model_dump(mode="json")
    raw["workflow_grader_version"] = "agent-workflow-v2"
    with pytest.raises(ValueError, match="workflow_policy_grader_version_mismatch"):
        MeasurementPolicy.model_validate(raw)


@pytest.mark.parametrize("false_value,error_value,expected", [(0, 0, "passed"), (1, 1, "failed"), (None, 1, "missing")])
def test_suite_false_success_is_independent_of_error_handling(tmp_path: Path, false_value: int | None,
                                                              error_value: int, expected: str) -> None:
    """参数为独立指标组合；聚合只消费假成功原指标，错误处理通过不能补缺失。"""
    inputs = load_suite_sources(_workflow_manifest(tmp_path))
    score = next(item for item in inputs.scores if item.grading.grader == "agent_workflow")
    metrics = (MetricResult(name="false_success", status="missing" if false_value is None else "measured",
                           value=false_value, reason="frozen_observation"),
               MetricResult(name="error_handling_correctness", status="measured", value=error_value, reason="frozen_error"))
    score = score.model_copy(update={"grading": score.grading.model_copy(update={"measurements": metrics})})
    assert _false_success_check(inputs.runs[0], (score,))[0] == expected


def test_v2_missing_metric_does_not_fall_back_to_old_error_metric(tmp_path: Path) -> None:
    """V2缺整个指标时必须missing，不能悄悄读取旧错误处理指标得到通过。"""
    inputs = load_suite_sources(_workflow_manifest(tmp_path))
    score = next(item for item in inputs.scores if item.grading.grader == "agent_workflow")
    metric = MetricResult(name="error_handling_correctness", status="measured", value=1, reason="recovered")
    score = score.model_copy(update={"grading": score.grading.model_copy(update={"measurements": (metric,)})})
    assert _false_success_check(inputs.runs[0], (score,))[0] == "missing"


def test_regrade_cli_keeps_original_bytes_and_records_source_code_identity(tmp_path: Path) -> None:
    """正式CLI只写全新版本目录，保留原JSONL字节和来源SHA并拒绝覆盖。"""
    source = tmp_path / "original.jsonl"
    content = (snapshot(_record([current_turn()])) + "\n").encode("utf-8")
    source.write_bytes(content)
    output = tmp_path / "regrade-v2"
    argv = ["regrade-workflow", "--frozen", str(source), "--output-dir", str(output), "--grader-version", "agent-workflow-v2"]
    assert main(argv) == 0
    report_bytes = (output / "workflow.json").read_bytes()
    report = json.loads(report_bytes)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_frozen_sha256"] == hashlib.sha256(content).hexdigest()
    assert provenance["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert "agent_outcome/workflow_delivery.py" in provenance["code_identity"]["files"]
    assert report["rows"][0]["grader_version"] == "agent-workflow-v2"
    assert report["regrading"]["provider_called"] is False
    assert main(argv) == 1
    assert (output / "workflow.json").read_bytes() == report_bytes
    assert source.read_bytes() == content


def test_regrade_cli_requires_explicit_version(tmp_path: Path) -> None:
    """未明确指定评分版本时命令拒绝运行，不按当前机器默认值解释历史快照。"""
    with pytest.raises(SystemExit):
        main(["regrade-workflow", "--frozen", "old.jsonl", "--output-dir", str(tmp_path / "new")])
