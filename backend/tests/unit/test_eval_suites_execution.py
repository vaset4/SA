"""固定案例准备、实际运行产物、进程失败与条件漂移回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pdf_fixtures import write_pdf
import pytest
from pydantic import ValidationError

from evals.contracts import create_case, read_snapshot, snapshot, snapshot_hash
from evals.evidence.contracts import EvidenceBenchmark, EvidenceSpan
from evals.evidence.corpus import build_index_from_chunks, build_native_source_snapshot
from evals.suites.contracts import IdentityCondition, RunSlot
from evals.suites.execute import execute_job, load_prepared, run_worker
from evals.suites.execution_contracts import ExecutionProfile, PreparedCase, PreparedJob, PreparedSuite
from evals.suites.identity import RuntimeBindings, capture_conditions, verify_conditions
from evals.suites.prepare import PreparationFailure, prepare_suite, sha256_file, write_new_json


def native_profile(root: Path) -> ExecutionProfile:
    """参数为临时目录；创建真实小 PDF 和对应候选，用于离线验证准备文件流程。"""
    paper_root = root / "paper"
    paper_root.mkdir(parents=True)
    pdf = paper_root / "source.pdf"
    write_pdf(pdf, ("The measured evidence is in this original paper.",))
    source = build_native_source_snapshot("document-alias", pdf, paper_dir=paper_root)
    text = source.blocks[0].text
    span = EvidenceSpan(span_id="target", source_snapshot_id=source.snapshot_id, block_id=source.blocks[0].block_id,
                        start=0, end=len(text), quote=text, quote_sha256=snapshot_hash(text))
    benchmark = EvidenceBenchmark.model_validate({"dataset_version": "fixture-v1", "review_status": "candidate",
        "sources": [source], "spans": [span], "target_paper_ids": [source.paper_id],
        "groups": [{"group_id": "target", "importance": "critical", "proofs": [{"span_ids": ["target"]}]}]})
    dataset = root / "rag_candidates.jsonl"
    raw = {"id": "rag-test", "question": "Find the measured evidence", "paper_ids": [source.paper_id], "authorized_paper_keys": ["native"],
           "source_type": "fixture", "evidence_benchmark": benchmark.model_dump(mode="json"),
           "metadata": {"review_material": {"evidence": [{"paper_key": "native", "evidence_key": "target"}]}}}
    dataset.write_text(snapshot(raw) + "\n", encoding="utf-8")
    write_new_json(root / "manifest.json", {"files": {dataset.name: sha256_file(dataset)}, "source_snapshots": {"native": source.model_dump(mode="json")}})
    return ExecutionProfile.model_validate({"suite_id": "pipeline-fixture", "dataset_version": "fixture-v1", "purpose": "regression",
        "user_id": "fixture-user", "project_id": "fixture-project", "uploads_root": str(paper_root), "paper_root": str(paper_root),
        "paper_bindings": [{"paper_key": "native", "paper_id": "actual-paper", "pdf_sha256": source.pdf_sha256}], "retrieval_top_k": 1,
        "jobs": [{"job_id": "retrieval", "kind": "retrieval", "dataset_path": str(dataset), "case_ids": ["rag-test"], "repetitions": 2,
                  "required_checks": ["scope", "truth_leakage", "required_trace"]}]})


class FixturePreparation:
    """提供具名测试归属与索引端口；只验证软件流程，不冒充真实服务验收。"""

    def __init__(self, *, mapping_issue: bool = False) -> None:
        """参数为是否注入已知映射缺口；初始化受控依赖。"""
        self.mapping_issue = mapping_issue

    async def resolve_scope(self, profile: ExecutionProfile, *, paper_ids: tuple[str, ...]) -> dict[str, Any]:
        """参数为配置和请求范围；返回显式测试归属合同。"""
        return {"user_id": profile.user_id, "project_id": profile.project_id, "paper_ids": list(paper_ids),
                "filter_expr": "paper_id in " + json.dumps(paper_ids), "local_retrieval_allowed": True}

    async def verify_bindings(self, profile: ExecutionProfile) -> tuple[dict[str, Any], ...]:
        """参数为固定配置；返回仅属于测试用户的实际测试元数据。"""
        return tuple({"paper_id": item.paper_id, "user_id": profile.user_id, "title": "Fixture paper"} for item in profile.paper_bindings)

    async def export_index(self, sources):
        """参数为经 PDF 校验的原文；构造来自真实测试正文的完整映射。"""
        chunks = [{"paper_id": source.paper_id, "chunk_id": "actual-child", "text": source.blocks[0].text} for source in sources]
        index, issues = build_index_from_chunks(sources, chunks, retrieval_backend="fixture")
        return index, [{"reason": "test_mapping_gap"}] if self.mapping_issue else issues

    async def snapshot_initial(self, profile: ExecutionProfile, *, papers: tuple[dict[str, Any], ...]) -> str:
        """参数为固定范围与元数据；返回显式空的测试初始仓储，供重复隔离断言。"""
        return snapshot({"schema_version": "eval-storage-v1", "user_id": profile.user_id, "project_id": profile.project_id,
                         "papers": list(papers), "questions": [], "items": [], "evidence_refs": [], "links": [], "question_links": [], "versions": []})


def portable_planner_record() -> dict[str, Any]:
    """无参数；返回自造单元测试题和显式模型替身环境，不读取私人历史数据集。"""
    text = "Fixture direct text for portable-evidence."
    return {
        "id": "portable-planner", "source_type": "fixture", "case_version": "fixture-planner-v1",
        "category": "single_paper_lookup", "question": "Find the stated evidence in the fixture paper.",
        "reference_answer": text, "golden_evidence": [text], "critical_facts": [text], "rubric": {},
        "expected": {"depth": "lookup", "min_units": 1, "source_types": ["local_direct_text"],
                     "failure_stage": "none"},
        "fixture": {"version": "planner-fixture-v1", "source": "portable-unit-test",
            "environment": {"depth": "lookup", "failure_stage": "none"},
            "seed": {"answer": text, "research_plan": {"units": [{
                "unit_id": "u1", "goal": "Find the fixture evidence", "research_action": "Read the fixture text",
                "evidence_policy": {"source_type": "local_direct_text"}}]},
                "research_unit_results": [{"unit_id": "u1", "evidence_ids": ["portable-evidence"],
                    "claims": [{"text": text, "claim_type": "paper_supported",
                                "supporting_evidence_ids": ["portable-evidence"]}]}]}},
    }


def prepared_fixture(root: Path, *, planner_only: bool = False) -> PreparedSuite:
    """参数为新临时目录和执行种类；返回由真实测试原文准备的固定输入。"""
    profile = native_profile(root)
    prepared = asyncio.run(prepare_suite(profile, root / "prepared", runtime=FixturePreparation()))
    if not planner_only:
        return prepared
    raw = portable_planner_record()
    case = create_case(raw)
    job = PreparedJob(job_id="planner", kind="planner_fixture",
                      cases=(PreparedCase(case_id=case.case_id, case_version=case.case_version, record_snapshot=snapshot(raw)),),
                      slots=(RunSlot(case_id=case.case_id, case_version=case.case_version, repeat_id="1", run_id=None, required_checks=("truth_leakage", "required_trace")),))
    body = prepared.model_dump(mode="json", exclude={"prepared_id"})
    body["jobs"] = [job.model_dump(mode="json")]
    return PreparedSuite.model_validate({**body, "prepared_id": snapshot_hash(snapshot(body))})


def test_prepare_keeps_candidate_scope_and_fixed_slots(tmp_path: Path) -> None:
    """参数为目录；校验原生 PDF、绑定新身份并固定重复槽位，候选不自动升级金标。"""
    prepared = prepared_fixture(tmp_path)
    assert load_prepared(tmp_path / "prepared" / "prepared.json") == prepared
    raw = read_snapshot(prepared.jobs[0].cases[0].record_snapshot)
    assert raw["evidence_benchmark"]["review_status"] == "candidate"
    assert raw["paper_ids"] == ["actual-paper"]
    assert read_snapshot(create_case(raw).input_snapshot)["project_scope"]["user_id"] == "fixture-user"
    assert [slot.repeat_id for slot in prepared.jobs[0].slots] == ["1", "2"]
    assert "evidence_benchmark" not in create_case(raw).input_snapshot


def test_prepare_mapping_gap_has_failure_report_and_no_baseline(tmp_path: Path) -> None:
    """参数为目录；映射缺口必须保留，不能用目标摘录补出成功索引。"""
    profile = native_profile(tmp_path)
    output = tmp_path / "failed"
    with pytest.raises(PreparationFailure, match="mapping_incomplete"):
        asyncio.run(prepare_suite(profile, output, runtime=FixturePreparation(mapping_issue=True)))
    assert not (output / "prepared.json").exists()
    assert json.loads((output / "preparation-failure.json").read_text(encoding="utf-8"))["details"] == [{"reason": "test_mapping_gap"}]


def test_prepared_checksum_and_duplicate_slots_are_rejected(tmp_path: Path) -> None:
    """参数为目录；准备后改题或把同一槽位复制两次均须明确失败。"""
    prepared = prepared_fixture(tmp_path)
    raw = prepared.model_dump(mode="json")
    raw["profile"]["retrieval_top_k"] += 1
    with pytest.raises(ValidationError, match="checksum"):
        PreparedSuite.model_validate(raw)
    job = prepared.jobs[0].model_dump(mode="json")
    job["slots"].append(job["slots"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        PreparedJob.model_validate(job)


def test_worker_start_failure_is_not_a_successful_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录与替身；进程启动失败保留原因，不读取其他题或历史结果补槽。"""
    prepared = prepared_fixture(tmp_path)
    output = tmp_path / "run"
    (output / "logs").mkdir(parents=True)
    monkeypatch.setattr("evals.suites.execute.sys.executable", str(tmp_path / "missing-python.exe"))
    result = execute_job(prepared.jobs[0], prepared.jobs[0].slots[0], prepared_path=tmp_path / "prepared" / "prepared.json", output_dir=output)
    assert result.status == "failed" and result.run_id is None
    assert result.error and "FileNotFoundError" in result.error


def test_condition_drift_prevents_comparable_identity() -> None:
    """无参数；同一次执行前后条件变化不能被标成稳定环境。"""
    before = (IdentityCondition(name="code", status="recorded", snapshot="before", reason="test"),)
    after = (IdentityCondition(name="code", status="recorded", snapshot="after", reason="test"),)
    assert verify_conditions(before, after) == ("runtime_condition_drift:code",)


def test_worker_retains_frozen_run_when_post_verification_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录与条件探针；运行后验证失败仍保留真实评分、冻结记录及不可比身份。"""
    prepared = prepared_fixture(tmp_path, planner_only=True)
    captured = 0

    def conditions(bindings: RuntimeBindings):
        """参数为实际依赖；第二次采集注入文件变化以复现运行中漂移。"""
        nonlocal captured
        captured += 1
        result = capture_conditions(bindings)
        if captured == 2:
            return tuple(item.model_copy(update={"snapshot": "changed-during-run"}) if item.name == "code" else item for item in result)
        return result

    monkeypatch.setattr("evals.suites.execute.capture_conditions", conditions)
    job = prepared.jobs[0]
    result = asyncio.run(run_worker(prepared, job, slot=job.slots[0], output=tmp_path / "worker"))
    assert result.status == "failed" and result.frozen_path and Path(result.frozen_path).is_file()
    assert result.error and "runtime_condition_drift:code" in result.error
    identity = json.loads(Path(result.identity_path).read_text(encoding="utf-8"))
    assert next(item for item in identity["runs"][0]["conditions"] if item["name"] == "environment")["status"] == "missing"
