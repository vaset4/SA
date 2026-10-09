"""工作流策略冻结、真实产物导入与秘密导出边界回归。作者：xxx。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from evals.agent_outcome.workflow import grade_workflow_record
from evals.contracts import read_snapshot, record_contracts, snapshot
from evals.ragas.frozen_bridge import score_frozen_record
from evals.suites.adapters import load_suite_sources
from evals.suites.aggregate import aggregate_suite
from evals.suites.compare import assess_comparability
from evals.suites.contracts import ComparisonRequest, SuiteManifest
from evals.suites.execute import ExportLogFormatter, _write_case_artifacts, load_prepared
from evals.suites.execution_contracts import PreparedCase, PreparedJob, PreparedSuite
from evals.suites.policy import workflow_policy
from evals.suites.report import build_suite_report, render_suite_report
from evals.suites.runtime import CapturedGraph, sanitize_frozen_record
from test_eval_suites_contracts_adapters import make_agent_record, make_slots, write_json, write_source
from test_eval_suites_execution import prepared_fixture
from test_ragas_frozen_bridge import _Scorer


def _workflow_manifest(root: Path) -> Path:
    """参数为临时目录；通过真实评分器冻结一条失败和一个未执行槽位，返回清单。"""
    record = make_agent_record(success=False)
    policy = workflow_policy()
    report = {"schema_version": "agent-workflow-report-v1", "measurement_policy": policy.model_dump(mode="json"),
              "rows": [grade_workflow_record(record)]}
    source = write_source(root, "workflow", (record,), report=report, with_identity=True)
    ragas = asyncio.run(score_frozen_record(record, kind="agent", scorer=_Scorer(), policy=policy))
    digest = write_json(root / "ragas.json", ragas)
    source = source.model_copy(update={"ragas_path": "ragas.json", "ragas_sha256": digest})
    slot = make_slots((record,), required=("scope", "citation", "truth_leakage", "required_trace", "false_success"))[0]
    missing = slot.model_copy(update={"repeat_id": "repeat-2", "run_id": None})
    execution = [{**slot.model_dump(mode="json"), "status": "failed", "error": "actual_product_failure"},
                 {**missing.model_dump(mode="json"), "status": "not_run", "error": "cancelled_before_slot"}]
    manifest = SuiteManifest(suite_id="workflow-fixture", purpose="regression", dataset_version="v1",
                             sources=(source,), slots=(slot, missing), measurement_policy_snapshot=snapshot(policy.model_dump(mode="json")),
                             execution_snapshot=snapshot({"runs": execution}))
    write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    return root / "manifest.json"


def test_new_workflow_source_preserves_failures_missing_slots_and_retrieval(tmp_path: Path) -> None:
    """失败和未执行槽位不会从报告消失；工作流和检索切片分开且不含回答质量或task_success。"""
    inputs = load_suite_sources(_workflow_manifest(tmp_path))
    summary = aggregate_suite(inputs)
    markdown = render_suite_report(build_suite_report(inputs, summary))
    assert {score.grading.grader for score in inputs.scores} == {"agent_workflow", "ragas_frozen"}
    names = {metric.name for score in inputs.scores for metric in score.grading.measurements}
    assert "workflow_completion" in names and "context_recall" in names
    assert not names.intersection({"task_success", "faithfulness", "factual_correctness"})
    assert "预期槽位：2" in markdown and "not_run" in markdown and "actual_product_failure" in markdown
    assert "Judge API" in markdown and "workflow_rule_compliance" in markdown
    checks = {check.name: check for check in summary.checks if check.run_id}
    assert checks["scope"].status == "passed"
    assert checks["citation"].reason != "required_assertion_not_recorded"


@pytest.mark.parametrize("tamper", ["run", "projection", "query", "policy"])
def test_workflow_and_retrieval_artifact_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    """即使更新文件校验和，错误运行、投影、查询或策略也不能混入同一次测评。"""
    path = _workflow_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    target = tmp_path / (source["ragas_path"] if tamper == "query" else source["report_path"])
    payload = json.loads(target.read_text(encoding="utf-8"))
    row = payload if tamper == "query" else payload["rows"][0]
    if tamper == "run":
        row["grading_result"]["run_id"] = "another-run"
    elif tamper == "projection":
        row["grading_result"]["projection_hash"] = "0" * 64
    elif tamper == "query":
        row["samples"][0]["retrieval_query"] = "forged query"
    else:
        payload["measurement_policy"]["ragas_metrics"] = ["faithfulness"]
    digest = write_json(target, payload)
    source["ragas_sha256" if tamper == "query" else "report_sha256"] = digest
    source["identity_path"] = source["identity_sha256"] = None
    write_json(path, manifest)
    with pytest.raises(ValueError):
        load_suite_sources(path)


def test_policy_is_frozen_and_old_prepared_identity_is_unchanged(tmp_path: Path) -> None:
    """新增可选策略不能改变旧prepared哈希；在旧包里插入新策略必须重新prepare。"""
    old = prepared_fixture(tmp_path)
    assert "measurement_policy" not in old.model_dump(mode="json")["profile"]
    assert load_prepared(tmp_path / "prepared" / "prepared.json") == old
    modified = old.model_dump(mode="json")
    modified["profile"]["measurement_policy"] = workflow_policy().model_dump(mode="json")
    with pytest.raises(ValueError, match="checksum"):
        PreparedSuite.model_validate(modified)


def test_policy_change_blocks_comparison_even_when_code_change_allowed(tmp_path: Path) -> None:
    """允许代码变化也不能把新测评策略与旧口径当作同分母提升。"""
    summary = aggregate_suite(load_suite_sources(_workflow_manifest(tmp_path)))
    changed_manifest = summary.inputs.manifest.model_copy(update={"measurement_policy_snapshot": None})
    baseline = summary.model_copy(update={"inputs": summary.inputs.model_copy(update={"manifest": changed_manifest})})
    decision = assess_comparability(summary, baseline, ComparisonRequest(changed_factors=("code",)))
    assert "measurement_policy_changed" in decision.reasons


def test_worker_writes_workflow_sidecar_without_old_evidence_mapping_gate(tmp_path: Path) -> None:
    """实际输出函数生成workflow文件；精确映射是否可评分不决定工作流产物是否存在。"""
    record = make_agent_record()
    case, _ = record_contracts(record)
    job = PreparedJob(job_id="agent", kind="agent", cases=(PreparedCase(case_id=case.case_id,
                      case_version=case.case_version, record_snapshot=snapshot(record["agent_case"])),),
                      slots=tuple(slot.model_copy(update={"run_id": None}) for slot in make_slots((record,))))
    report_path, frozen_path = _write_case_artifacts(record, job, output=tmp_path, policy=workflow_policy())
    assert report_path.name == "workflow.json" and frozen_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["rows"][0]["run_id"] == record["run_record"]["run_id"]


def test_snapshot_and_log_redaction_preserves_original_objects() -> None:
    """凭证含JSON转义字符时快照仍可解析，原始对象和异常日志记录不被改写。"""
    secret = 'fixture-key-"\\tail'
    original = {"run_record": {"observation_snapshot": snapshot({"error": {"type": "ConnectionError", "message": secret}})}}
    before = deepcopy(original)
    clean = sanitize_frozen_record(original, secrets=(secret,))
    assert read_snapshot(clean["run_record"]["observation_snapshot"])["error"] == {
        "type": "ConnectionError", "message": "[redacted]"}
    assert original == before
    error = ConnectionError(secret)
    log = logging.LogRecord("fixture", logging.ERROR, __file__, 1, "%s", (secret,), (type(error), error, None))
    formatted = ExportLogFormatter(None, secrets=(secret,)).format(log)
    assert secret not in formatted and "ConnectionError" in formatted
    assert log.exc_text is None and str(error) == secret


@pytest.mark.asyncio
async def test_partial_events_redact_before_disk_but_yield_original(tmp_path: Path) -> None:
    """中断前逐条写盘的事件也脱敏，业务消费者仍收到原始对象。"""
    item = {"error": "fixture-partial-key", "kind": "transport"}

    class Graph:
        """返回固定原始事件的真实流接口替身。"""

        async def astream(self, inputs: dict[str, Any], *, stream_mode: list[str]) -> Any:
            """参数为业务输入和流模式；返回一次实际测试事件。"""
            yield item

    target = tmp_path / "partial.jsonl"
    graph = CapturedGraph(graph=Graph(), path=target, secrets=("fixture-partial-key",))
    values = [value async for value in graph.astream({}, stream_mode=["values"])]
    assert values[0] is item and item["error"] == "fixture-partial-key"
    assert "fixture-partial-key" not in target.read_text(encoding="utf-8")
