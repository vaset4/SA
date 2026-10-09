"""P1 独立范围、稳定发布身份与资源边界回归。作者：xxx。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.suites.prepare import _case_scopes
from evals.contracts import read_snapshot, snapshot


def test_retrieval_scope_uses_authorized_library_not_truth() -> None:
    """无参数；改变目标证据仍保持事前声明的两篇授权范围，返回无。"""
    class Runtime:
        """仅回显授权范围的测试端口。"""

        async def resolve_scope(self, profile, *, paper_ids):
            """参数为配置及论文；返回实际收到的范围。"""
            return {"paper_ids": list(paper_ids)}

    profile = SimpleNamespace(paper_bindings=[SimpleNamespace(paper_key="a", paper_id="one"),
                                             SimpleNamespace(paper_key="b", paper_id="two")],
                              jobs=[SimpleNamespace(job_id="rag", kind="retrieval")])
    row = {"id": "case", "authorized_paper_keys": ["a", "b"],
           "metadata": {"review_material": {"evidence": [{"paper_key": "a"}]}}}
    actual = asyncio.run(_case_scopes(profile, {"rag": (row,)}, runtime=Runtime()))
    assert actual["case"]["paper_ids"] == ["one", "two"]


def test_missing_authorization_does_not_infer_from_evidence() -> None:
    """无参数；没有授权声明时明确失败，不从答案材料猜检索范围，返回无。"""
    profile = SimpleNamespace(paper_bindings=[SimpleNamespace(paper_key="a", paper_id="one")],
                              jobs=[SimpleNamespace(job_id="rag", kind="retrieval")])
    row = {"id": "case", "metadata": {"review_material": {"evidence": [{"paper_key": "a"}]}}}
    with pytest.raises(ValueError, match="authorized_paper_keys"):
        asyncio.run(_case_scopes(profile, {"rag": (row,)}, runtime=object()))


def test_temporary_lease_does_not_change_comparison_identity() -> None:
    """无参数；只忽略publication租约差异，批次及普通业务lease字段变化仍保留。"""
    from evals.suites.adapters import stable_execution_input

    first = {"project_scope": {"publication": {"lease_id": "one", "batch_ids": ["batch"]}}, "lease_id": "business"}
    second = {"project_scope": {"publication": {"lease_id": "two", "batch_ids": ["batch"]}}, "lease_id": "business"}
    assert stable_execution_input(snapshot(first)) == stable_execution_input(snapshot(second))
    second["project_scope"]["publication"]["batch_ids"] = ["changed"]
    assert stable_execution_input(snapshot(first)) != stable_execution_input(snapshot(second))
    assert "lease_id" in read_snapshot(stable_execution_input(snapshot(first)))
    assert first["project_scope"]["publication"]["lease_id"] == "one"


class PublicationConnection:
    """模拟发布事务边界，记录真实SQL调用以检验是否创建了占用。"""

    def __init__(self):
        """无参数；初始化SQL调用列表。"""
        self.writes = []

    @asynccontextmanager
    async def acquire(self):
        """无参数；提供池连接上下文，返回当前测试连接。"""
        yield self

    @asynccontextmanager
    async def transaction(self):
        """无参数；提供事务上下文，返回当前测试连接。"""
        yield self

    async def fetch(self, query, *args):
        """参数为SQL和绑定值；分别返回论文与已发布批次行。"""
        if "FROM papers" in query:
            return [{"paper_id": "paper", "index_batch_id": "batch"}]
        return [{"paper_id": "paper", "batch_id": "batch", "embedding_fingerprint": "embedding", "collection_name": "collection"}]

    async def execute(self, query, *args):
        """参数为SQL和绑定值；保存写入调用，无返回值。"""
        self.writes.append((query, args))


def test_stable_publication_is_read_only_and_exact_acquisition_rejects_drift() -> None:
    """无参数；稳定快照不写租约，准确领取才写，批次不符时在写入前拒绝。"""
    from app.repositories.upload_publication_repository import UploadPublicationRepository

    connection = PublicationConnection()

    async def pool():
        """无参数；返回测试事务连接池。"""
        return connection

    async def exercise():
        """无参数；依次核验只读、准确领取和批次漂移，返回无。"""
        repository = UploadPublicationRepository(pool_provider=pool)
        stable = await repository.snapshot(["paper"], user_id="user", acquire_lease=False)
        assert "lease_id" not in stable and not connection.writes
        active = await repository.snapshot(["paper"], user_id="user", expected_snapshot=stable)
        assert active["lease_id"] and len(connection.writes) == 1
        with pytest.raises(ValueError):
            await repository.snapshot(["paper"], user_id="user", expected_snapshot={**stable, "batch_ids": ["old"]})
        assert len(connection.writes) == 1

    asyncio.run(exercise())


def test_timeout_preserves_slot_and_actual_cleanup(tmp_path: Path, monkeypatch) -> None:
    """参数为目录和替身；运行时超时保留原槽位及清理结果，不伪造完成产物。"""
    import subprocess
    from evals.suites.execute import execute_job
    from evals.suites.contracts import RunSlot

    class Process:
        """明确在首次等待时超时的进程替身。"""
        pid = 123
        returncode = 1

        def wait(self, *, timeout):
            """参数为截止时间；抛出真实进程超时异常。"""
            raise subprocess.TimeoutExpired("worker", timeout)

    (tmp_path / "logs").mkdir()
    monkeypatch.setattr("evals.suites.execute.subprocess.Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr("evals.suites.execute._stop_worker", lambda process, **kwargs: {"forced": True, "pid": process.pid})
    slot = RunSlot(case_id="case", case_version="v1", repeat_id="1", run_id=None, required_checks=())
    result = execute_job(SimpleNamespace(job_id="agent"), slot, prepared_path=tmp_path / "prepared.json",
                         output_dir=tmp_path, timeout_seconds=0.01)
    assert result.status == "timed_out" and result.case_id == "case" and result.run_id is None
    assert read_snapshot(result.cleanup_snapshot)["forced"] is True
    assert result.elapsed_seconds >= 0


def test_model_observer_preserves_response_without_premature_completion() -> None:
    """无参数；请求观察器原样返回响应，仅完整响应解析边界可记录完成及用量。"""
    from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx
    from evals.suites.runtime import ModelRequestObserver

    response = SimpleNamespace(json=lambda: {"usage": {"prompt_tokens": 13, "completion_tokens": 5}})
    trace = TraceContext(request_id="usage", user_id="user")
    token = set_trace_ctx(trace)
    try:
        observer = ModelRequestObserver(send=lambda *args: response)
        assert observer.invoke(SimpleNamespace(base_url="https://example.test/v1"), {}, {"model": "model"}) is response
        events = export_evidence_observations(trace)["evidence_observations"]
    finally:
        reset_trace_ctx(token)
    assert [event["stage"] for event in events] == ["model_request_parameters"]


def test_ragas_sidecar_is_bound_to_original_run_and_reported(tmp_path: Path) -> None:
    """参数为目录；同run的独立RAGAS评分可汇总，跨run替换即使更新文件哈希仍拒绝。"""
    from evals.contracts import MetricResult, build_grading, record_contracts
    from evals.suites.adapters import load_suite_sources
    from evals.suites.aggregate import aggregate_suite
    from test_eval_suites_contracts_adapters import make_evidence_record, make_slots, write_json, write_manifest, write_source

    record = make_evidence_record()
    source = write_source(tmp_path, "evidence", (record,))
    metric = MetricResult(name="context_recall", status="measured", value=0.5, reason="test_judge")
    grading = build_grading(record, [metric], grader="ragas_frozen", grader_version="test-v1")
    case, run = record_contracts(record)
    sidecar = {"case_id": case.case_id, "run_id": run.run_id, "grading_result": grading, "measurements": grading["measurements"]}
    digest = write_json(tmp_path / "ragas.json", sidecar)
    source = source.model_copy(update={"ragas_path": "ragas.json", "ragas_sha256": digest})
    manifest = write_manifest(tmp_path, (source,), make_slots((record,)))
    summary = aggregate_suite(load_suite_sources(manifest))
    assert next(group for group in summary.groups if group.metric == "context_recall").mean == 0.5
    wrong = build_grading(make_evidence_record("another"), [metric], grader="ragas_frozen", grader_version="test-v1")
    digest = write_json(tmp_path / "ragas.json", {**sidecar, "grading_result": wrong})
    source = source.model_copy(update={"ragas_sha256": digest})
    manifest = write_manifest(tmp_path, (source,), make_slots((record,)))
    with pytest.raises(ValueError, match="ragas_grading_run"):
        load_suite_sources(manifest)


def test_negative_case_needs_no_fabricated_proof_or_task_success(tmp_path: Path) -> None:
    """参数为目录；无答案场景可准备且参考答案留在truth，任务完成明确不适用。"""
    from evals.agent_outcome.contracts import AgentRun
    from evals.agent_outcome.scoring import measure_task_success
    from evals.suites.prepare import CaseBindings, prepare_agent_case
    from test_eval_suites_execution import prepared_fixture

    prepared = prepared_fixture(tmp_path)
    benchmark = {"dataset_version": "fixture-v1", "review_status": "candidate", "sources": [], "spans": [], "groups": []}
    raw = {"id": "negative", "title_zh": "资料不足", "task_family": "insufficient_evidence", "review_status": "candidate",
           "input": {"question_zh": "资料没有提供的测量是什么？", "paper_keys": ["native"]},
           "initial_conditions": {"authorized_paper_keys": ["native"]}, "turns": [{"turn": 1, "user_message": "问题"}],
           "evidence_benchmark": benchmark, "answerability": "insufficient_evidence", "reference_answer": "无法确定"}
    bindings = CaseBindings(profile=prepared.profile, sources=prepared.sources, index=prepared.index,
                            rag_cases={}, scopes={"negative": read_snapshot(prepared.scope_snapshot)})
    case = prepare_agent_case(raw, bindings)
    assert "无法确定" not in case.input_snapshot
    assert read_snapshot(case.truth_snapshot)["reference_answer"] == "无法确定"
    run = AgentRun(run_id="run", case_id=case.case_id, case_version=case.case_version, run_mode="agent_live",
                   status="completed", input_snapshot=case.input_snapshot, observation_snapshot="{}", environment_snapshot="{}")
    result = measure_task_success(case, run, facts=None, predicates=())
    assert result.status == "not_measured" and result.value is None


def test_strict_mapping_gap_is_frozen_without_fabricating_mapping(tmp_path: Path) -> None:
    """参数为目录；真实文本无法精确对齐时保留缺口和空映射，仍允许独立评分准备。"""
    from evals.evidence.corpus import build_index_from_chunks
    from evals.suites.prepare import prepare_suite
    from test_eval_suites_execution import FixturePreparation, native_profile

    class MissingMapping(FixturePreparation):
        """返回实际不同文本对应的映射缺口。"""

        async def export_index(self, sources):
            """参数为原文；返回与PDF不同的索引文本及严格映射诊断。"""
            return build_index_from_chunks(sources, [{"paper_id": sources[0].paper_id, "chunk_id": "chunk",
                                                    "text": "An independently parsed representation."}], retrieval_backend="fixture")

    profile = native_profile(tmp_path)
    prepared = asyncio.run(prepare_suite(profile, tmp_path / "prepared", runtime=MissingMapping()))
    assert not prepared.index.mappings
    assert read_snapshot(prepared.mapping_issues_snapshot)["issues"][0]["reason"] == "index_text_not_located"


def test_index_export_filters_frozen_published_batch(monkeypatch) -> None:
    """参数为替身；索引导出必须同时限定论文与已冻结批次，不能导出旧版或构建中行。"""
    import sys
    from unittest.mock import Mock
    from evals.suites.runtime import LivePreparationRuntime

    iterator = Mock()
    iterator.next.side_effect = [[{"paper_id": "paper", "chunk_id": "child", "batch_id": "published"}], []]
    client = Mock()
    client.query_iterator.return_value = iterator
    live = LivePreparationRuntime()
    live._store = SimpleNamespace(_get_client=lambda: client)
    live.publication = {"batch_ids": ["published"]}
    monkeypatch.setattr(live, "require_existing_index", lambda: None)
    monkeypatch.setitem(sys.modules, "app.services.milvus_store", SimpleNamespace(_COUNT_BATCH_SIZE=10, _FIELD_OUTPUTS=["chunk_id"]))
    rows = live._index_rows((SimpleNamespace(paper_id="paper"),))
    assert client.query_iterator.call_args.kwargs["filter"] == 'paper_id in ["paper"] and batch_id in ["published"]'
    assert rows[0]["batch_id"] == "published"
    iterator.close.assert_called_once()


def test_forced_worker_cleanup_terminates_actual_process(tmp_path: Path, monkeypatch) -> None:
    """参数为目录与短清理宽限；终止本测试创建的实际长等待进程并核对退出状态。"""
    import subprocess
    import sys
    from evals.suites.execute import _stop_worker

    monkeypatch.setattr("evals.suites.execute.CANCELLATION_GRACE_SECONDS", 1.0)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        result = _stop_worker(process, target=tmp_path / "not-created")
        assert result["forced"] is True and process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
