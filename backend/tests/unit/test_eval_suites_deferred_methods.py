"""两方法、延后评分与不可覆盖评分清单的合同回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from evals.contracts import snapshot, snapshot_hash
from evals.suites.adapters import load_suite_sources
from evals.suites.aggregate import aggregate_suite
from evals.suites.compare import _run_reasons
from evals.suites.contracts import ComparisonRequest, IdentityCondition, SuiteManifest
from evals.suites.execute import _score_completed_run
from evals.suites.execution_contracts import ExecutionJob, ExecutionProfile, ProducedRun
from evals.suites.policy import workflow_policy
from evals.suites.prepare import prepare_suite, sha256_file
from evals.suites.report import build_suite_report, render_suite_report, write_suite_report
from evals.suites.runtime import search_declared_retrieval
from evals.suites.scoring import score_suite
from test_eval_suites_execution import FixturePreparation, native_profile, prepared_fixture
from test_eval_workflow_suite import _workflow_manifest
from test_ragas_frozen_bridge import _Scorer


def deferred_manifest(root: Path) -> Path:
    """参数为临时目录；建立真实冻结工作流文件，移除评分引用以表示已采集尚未评分。"""
    path = _workflow_manifest(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["grading_timing"] = "deferred"
    for source in payload["sources"]:
        source["ragas_path"] = source["ragas_sha256"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_old_prepared_hash_and_explicit_new_contracts(tmp_path: Path) -> None:
    """参数为目录；旧内容无新增字段，新方法/调度必须显式冻结且Agent不能声明方法。"""
    prepared = prepared_fixture(tmp_path)
    payload = prepared.model_dump(mode="json")
    assert "grading_timing" not in payload["profile"]
    assert "repetition_order" not in payload["profile"]
    assert "retrieval_method" not in payload["jobs"][0]
    assert snapshot_hash(snapshot({key: value for key, value in payload.items() if key != "prepared_id"})) == prepared.prepared_id
    job = prepared.profile.jobs[0].model_dump(mode="json")
    with pytest.raises(ValueError, match="requires_retrieval_job"):
        ExecutionJob.model_validate({**job, "kind": "agent", "retrieval_method": "dense"})
    with pytest.raises(ValueError, match="requires_measurement_policy"):
        ExecutionProfile.model_validate({**payload["profile"], "grading_timing": "deferred"})
    with pytest.raises(ValueError, match="single_experimental_factor"):
        ComparisonRequest(changed_factors=("retrieval_method", "code"))


def test_dense_and_hybrid_use_identical_authorized_publication_scope() -> None:
    """无参数；两个真实入口都使用完整可信范围，纯向量额外显式带论文过滤且不读答案范围。"""
    store = Mock()
    scope = {"user_id": "owner", "paper_ids": ["a", "b"], "project_id": "project",
             "local_retrieval_allowed": True, "filter_expr": "owner-filter", "publication": {"batch": "frozen"}}
    search_declared_retrieval(store, "same query", 5, scope=scope, method="dense")
    store.local_rag_search.assert_called_once_with("same query", top_k=5, filter_paper_ids=["a", "b"], scope=scope)
    store.search_scope.assert_not_called()
    search_declared_retrieval(store, "same query", 5, scope=scope, method="hybrid")
    store.search_scope.assert_called_once_with("same query", top_k=5, scope=scope)
    with pytest.raises(ValueError, match="unknown_retrieval_method"):
        search_declared_retrieval(store, "query", 5, scope=scope, method="other")


def test_deferred_zero_judge_and_report_not_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录及替身；采集评分接点零judge，报告保留未开始及完整槽位分母。"""
    prepared = prepared_fixture(tmp_path / "prepared-case")
    prepared = prepared.model_copy(update={"profile": prepared.profile.model_copy(update={
        "measurement_policy": workflow_policy(), "grading_timing": "deferred"})})
    produced = ProducedRun(job_id="retrieval", case_id="rag-test", case_version="v1", repeat_id="1", status="completed",
                           run_id="run", report_path="report", frozen_path="frozen", identity_path="identity")
    judge = Mock(side_effect=AssertionError("deferred must not judge"))
    monkeypatch.setattr("evals.ragas.frozen_bridge.score_frozen_record", judge)
    assert _score_completed_run(prepared, prepared.jobs[0], produced) == produced
    judge.assert_not_called()
    monkeypatch.undo()
    root = tmp_path / "report"
    root.mkdir()
    inputs = load_suite_sources(deferred_manifest(root))
    summary = aggregate_suite(inputs)
    report = render_suite_report(build_suite_report(inputs, summary))
    assert "not_started" in report and "0 / 2" in report
    assert not any(score.grading.grader == "ragas_frozen" for score in inputs.scores)
    write_suite_report(root / "summary", build_suite_report(inputs, summary))
    assert "grading_not_started" in (root / "summary" / "results.csv").read_text(encoding="utf-8-sig")


def test_suite_score_preserves_original_and_continues_explicit_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录及替身；只读冻结输入追加评分，二次清单不重复评分、不可覆盖、篡改会拒绝。"""
    from evals.suites import scoring
    from evals.ragas.frozen_bridge import score_frozen_record

    source = deferred_manifest(tmp_path)
    original = {path: sha256_file(path) for path in tmp_path.iterdir() if path.is_file()}
    calls = []

    async def score(record, *, kind, policy):
        """参数为冻结记录和策略；记录调用并用确定性评分器验证正式评分合同。"""
        calls.append(record)
        return await score_frozen_record(record, kind=kind, policy=policy, scorer=_Scorer())

    monkeypatch.setattr(scoring, "score_frozen_record", score)
    monkeypatch.setattr(scoring, "grading_identity", lambda policy: {"judge": "offline-test"})
    output = score_suite(source, output=tmp_path / "scored-001.json")
    assert len(calls) == 1
    assert all(sha256_file(path) == digest for path, digest in original.items())
    manifest = SuiteManifest.model_validate_json(output.read_text(encoding="utf-8"))
    artifact = tmp_path / str(manifest.sources[0].ragas_path)
    assert sha256_file(artifact) == manifest.sources[0].ragas_sha256
    assert "grading_seconds" in json.loads(artifact.read_text(encoding="utf-8"))
    score_suite(output, output=tmp_path / "scored-002.json")
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        score_suite(source, output=output)
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_suite_sources(output)


def test_method_comparison_only_ignores_declared_method(tmp_path: Path) -> None:
    """参数为目录；只有方法差异获豁免，K或其他检索配置变化仍拒绝比较。"""
    run = load_suite_sources(_workflow_manifest(tmp_path)).runs[0]
    dense = IdentityCondition(name="retrieval", status="recorded", snapshot=snapshot({"method": "dense", "top_k": 5}), reason="test")
    hybrid = dense.model_copy(update={"snapshot": snapshot({"method": "hybrid", "top_k": 5})})
    before = run.model_copy(update={"conditions": (dense,)})
    current = run.model_copy(update={"conditions": (hybrid,)})
    request = ComparisonRequest(changed_factors=("retrieval_method",))
    assert not _run_reasons(current, before, request)
    changed = hybrid.model_copy(update={"snapshot": snapshot({"method": "hybrid", "top_k": 6})})
    assert "undeclared_change:retrieval_configuration" in _run_reasons(current.model_copy(update={"conditions": (changed,)}), before, request)


def test_new_method_is_prepared_and_round_order_is_explicit(tmp_path: Path) -> None:
    """参数为目录；准备包固定方法和调度顺序，旧模式仍保留原有逐题重复。"""
    profile = native_profile(tmp_path)
    payload = profile.model_dump(mode="json")
    dataset = Path(payload["jobs"][0]["dataset_path"])
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    dataset.write_text(snapshot(raw) + "\n" + snapshot({**raw, "id": "rag-second"}) + "\n", encoding="utf-8")
    dataset_manifest = tmp_path / "manifest.json"
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    manifest["files"][dataset.name] = sha256_file(dataset)
    dataset_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    payload["jobs"][0]["case_ids"].append("rag-second")
    payload.update(measurement_policy=workflow_policy().model_dump(mode="json"), grading_timing="deferred", repetition_order="round-first")
    payload["jobs"][0]["retrieval_method"] = "dense"
    prepared = asyncio.run(prepare_suite(ExecutionProfile.model_validate(payload), tmp_path / "new-prepared", runtime=FixturePreparation()))
    assert prepared.jobs[0].retrieval_method == "dense"
    assert prepared.profile.grading_timing == "deferred"
    assert [(slot.case_id, slot.repeat_id) for slot in prepared.jobs[0].slots] == [
        ("rag-test", "1"), ("rag-second", "1"), ("rag-test", "2"), ("rag-second", "2")]


@pytest.mark.parametrize("method", ["dense", "hybrid"])
def test_worker_case_reaches_declared_real_search_port(tmp_path: Path, method: str) -> None:
    """参数为目录和方法；通过正式worker执行函数验证方法不会只停留在profile名称。"""
    from evals.suites.execute import _execute_case
    from evals.suites.identity import RuntimeBindings

    prepared = prepared_fixture(tmp_path)
    job = prepared.jobs[0].model_copy(update={"retrieval_method": method})
    live = Mock()
    live.execution_scope.side_effect = lambda scope: scope
    live.store.local_rag_search.return_value = []
    live.store.search_scope.return_value = []
    bindings = RuntimeBindings(prepared=prepared, kind="retrieval", live=live, retrieval_method=method)
    record = asyncio.run(_execute_case(job, job.cases[0], bindings=bindings, output=tmp_path))
    assert record["run_record"]["run_mode"] == "retrieval_live"
    if method == "dense":
        live.store.local_rag_search.assert_called_once()
        live.store.search_scope.assert_not_called()
    else:
        live.store.search_scope.assert_called_once()
        live.store.local_rag_search.assert_not_called()


def test_failed_score_requires_explicit_new_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录和替身；真实评分失败留存，默认不重评失败，显式重评才产生新attempt。"""
    from evals.suites import scoring

    source = deferred_manifest(tmp_path)
    failed = Mock(side_effect=RuntimeError("judge unavailable"))
    monkeypatch.setattr(scoring, "score_frozen_record", failed)
    monkeypatch.setattr(scoring, "grading_identity", lambda policy: {"judge": "offline-test"})
    output = score_suite(source, output=tmp_path / "attempt-001.json")
    first = SuiteManifest.model_validate_json(output.read_text(encoding="utf-8")).sources[0]
    old_digest = sha256_file(tmp_path / str(first.ragas_path))
    assert failed.call_count == 1
    second = score_suite(output, output=tmp_path / "attempt-002.json")
    assert failed.call_count == 1
    third = score_suite(second, output=tmp_path / "attempt-003.json", retry_failed=True)
    assert failed.call_count == 2
    new = SuiteManifest.model_validate_json(third.read_text(encoding="utf-8")).sources[0]
    assert new.ragas_path != first.ragas_path
    assert sha256_file(tmp_path / str(first.ragas_path)) == old_digest


def test_scoring_interruption_keeps_explicit_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录和替身；中断保留新清单与产品原件，下一次显式清单续评不会启动产品。"""
    from evals.suites import scoring

    source = deferred_manifest(tmp_path)
    original_hash = sha256_file(source)
    monkeypatch.setattr(scoring, "grading_identity", lambda policy: {"judge": "offline-test"})
    monkeypatch.setattr(scoring, "score_frozen_record", Mock(side_effect=KeyboardInterrupt))
    output = tmp_path / "interrupted.json"
    with pytest.raises(KeyboardInterrupt):
        score_suite(source, output=output)
    assert sha256_file(source) == original_hash
    assert json.loads(output.read_text(encoding="utf-8"))["sources"][0]["ragas_path"] is None
    monkeypatch.setattr(scoring, "score_frozen_record", Mock(side_effect=RuntimeError("judge down")))
    resumed = score_suite(output, output=tmp_path / "resumed.json")
    assert load_suite_sources(resumed).manifest.sources[0].ragas_path is not None


def test_method_delta_uses_paired_measured_cases_and_keeps_missing_denominator() -> None:
    """无参数；合法零分进入配对均值，未测案例保留在预定分母和排除明细中。"""
    from evals.contracts import MetricResult, read_snapshot
    from evals.suites.compare import _compare_method_group
    from evals.suites.contracts import CaseMeasurement, MetricGroup, RunMeasurement

    measured = MetricResult(name="context_precision", status="measured", value=0.0, reason="real_zero")
    run = RunMeasurement(repeat_id="1", run_id="run", measurements=(measured,), value=0.0, status="measured", reason="actual")
    case = CaseMeasurement(case_id="a", case_version="v1", value=0.0, complete=True, runs=(run,))
    missing = case.model_copy(update={"case_id": "b", "value": None, "complete": False,
        "runs": (run.model_copy(update={"status": "missing", "value": None, "measurements": ()}),)})
    group = MetricGroup(key_snapshot="{}", metric="context_precision", mean=0.0, cases=(case, missing),
                        states=(), complete_case_count=1)
    result = _compare_method_group(group, group)
    assert result.delta == 0 and result.status == "comparable"
    pairs = read_snapshot(result.pairs_snapshot)
    assert pairs["paired_cases"] == 1 and pairs["expected_cases"] == 2
    assert pairs["excluded"][0]["case_id"] == "b"


def test_failed_attempt_keeps_judge_identity_separate_from_failure(tmp_path: Path) -> None:
    """参数为目录；同judge的一次失败不阻断其他题配对，但judge配置变化仍不可比。"""
    from evals.contracts import read_snapshot
    from evals.suites.compare import _grading_identities

    summary = aggregate_suite(load_suite_sources(_workflow_manifest(tmp_path)))
    score = next(score for score in summary.inputs.scores if score.grading.grader == "ragas_frozen")
    detail = {**read_snapshot(score.detail_snapshot), "attempt": {"identity_before": {"model": "fixed"}}}
    first = summary.model_copy(update={"inputs": summary.inputs.model_copy(update={
        "scores": (score.model_copy(update={"detail_snapshot": snapshot(detail)}),)})})
    failed = {**detail, "judge": {"status": "failed", "error": "unavailable"}}
    second = summary.model_copy(update={"inputs": summary.inputs.model_copy(update={
        "scores": (score.model_copy(update={"detail_snapshot": snapshot(failed)}),)})})
    assert _grading_identities(first, score.run_id) == _grading_identities(second, score.run_id)
    different = {**failed, "attempt": {"identity_before": {"model": "changed"}}}
    third = summary.model_copy(update={"inputs": summary.inputs.model_copy(update={
        "scores": (score.model_copy(update={"detail_snapshot": snapshot(different)}),)})})
    assert _grading_identities(first, score.run_id) != _grading_identities(third, score.run_id)
