"""离线套件读取真实报告形状、冻结投影与历史身份边界。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evals.agent_outcome.contracts import freeze_agent_run, load_agent_case
from evals.agent_outcome.report import build_agent_report
from evals.contracts import MetricResult, build_grading, freeze_record, read_snapshot, record_contracts, snapshot, snapshot_hash
from evals.evidence.contracts import EvidenceBenchmark, EvidenceSpan, SourceBlock, make_source_snapshot
from evals.evidence.corpus import OFFLINE_BACKEND, build_index_from_chunks
from evals.evidence.report import build_evidence_report
from evals.evidence.run_eval import _offline_observation
from evals.research_planner.metrics import score_record
from evals.suites.adapters import load_suite_sources
from evals.suites.contracts import (
    CheckName, FrozenReportRef, IdentityCondition, RecordedIdentity, ReportKind,
    RunConditions, RunSlot, SuiteManifest, SuitePurpose,
)

_FIXTURE_TOP_K = 1


def measured(name: str, value: float, *, denominator: float = 1) -> MetricResult:
    """参数为指标名、测试值及分母；返回明确的流程 fixture 测量。"""
    return MetricResult(name=name, status="measured", reason="explicit_test_fixture", value=value,
                        numerator=value * denominator, denominator=denominator)


def make_planner_record(case_id: str = "case-a", *, metrics: tuple[MetricResult, ...] | None = None) -> dict[str, Any]:
    """参数为案例及可选受控测量；通过真实 Planner 评分入口生成冻结报告行。"""
    raw = {"id": case_id, "case_version": "case-v1", "source_type": "fixture", "category": "lookup",
           "question": "查找论文的方法原文", "expected": {"min_units": 0},
           "replay": {"research_plan": {"units": []}, "research_unit_results": [], "planner_trace": {}}}
    record = freeze_record(raw, mode="planner_replay", execution_input={"question": raw["question"]}, observation=raw["replay"])
    result = score_record(record)
    if metrics is not None:
        result = {**result, "grading_result": build_grading(record, list(metrics), grader="research_planner")}
    return result


def make_agent_record(case_id: str = "agent-a", *, success: bool = True) -> dict[str, Any]:
    """参数为案例与真实 fixture 终态；复用 Agent 冻结合同供实际评分器读取。"""
    evidence = make_evidence_record()
    benchmark = read_snapshot(record_contracts(evidence)[0].truth_snapshot)["evidence_benchmark"]
    index_id = benchmark["index_snapshots"][0]["index_snapshot_id"]
    scope = {"user_id": "fixture-owner", "paper_ids": ["paper-a"], "local_retrieval_allowed": True}
    raw = {"id": case_id, "case_version": "case-v1", "title_zh": "取证流程 fixture", "task_family": "lookup",
           "input": {"question_zh": "查找论文方法", "paper_keys": ["paper-a"], "project_scope": scope},
           "initial_conditions": {"history": []}, "turns": [{"turn": 1, "user_message": "查找论文方法"}],
           "required_outcomes": [{"predicate": "authorized_scope_preserved"}],
           "evidence_benchmark": benchmark,
           "evidence_targets": {"required_span_ids": ["method"], "required_group_ids": ["method"], "required_paper_ids": ["paper-a"]},
           "allowed_terminal_states": [{"state": "completed", "task_completed": True}]}
    case = load_agent_case(raw)
    status = "completed" if success else "failed"
    observed = {"turn": 1, "status": status, "terminal_state": status, "events": [],
                "input_snapshot": snapshot({"project_scope": scope, "user_query": "查找论文方法"}),
                "evidence_observation_snapshot": record_contracts(evidence)[1].observation_snapshot,
                "values_snapshot": snapshot({"turn_outcome": {"status": status}, "evidence_records": [{
                    "evidence_id": "e1", "paper_id": "paper-a", "chunk_id": "child-a", "parent_id": "parent-a",
                    "index_snapshot_id": index_id, "excerpt": "The method predicts traffic flow.",
                }], "citations": [{"evidence_id": "e1"}]})}
    return freeze_agent_run(case, [observed], run_mode="agent_fixture", environment={"provider": "fixture"}, status=status)


def make_evidence_record(case_id: str = "evidence-a") -> dict[str, Any]:
    """参数为案例标识；构造可核验原文与索引，再调用实际离线观察合同。"""
    text = "The method predicts traffic flow."
    source = make_source_snapshot(paper_id="paper-a", pdf_sha256=snapshot_hash("fixture-pdf"),
                                  artifact_manifest_snapshot="{}",
                                  blocks=(SourceBlock(block_id="body", page_num=1, section_path="Methods", text=text),))
    chunk = {"paper_id": "paper-a", "chunk_id": "child-a", "parent_id": "parent-a", "text": text, "full_text": text}
    index, issues = build_index_from_chunks((source,), [chunk], retrieval_backend=OFFLINE_BACKEND)
    assert not issues
    span = EvidenceSpan(span_id="method", source_snapshot_id=source.snapshot_id, block_id="body", start=0,
                        end=len(text), quote=text, quote_sha256=snapshot_hash(text))
    group = {"group_id": "method", "importance": "critical", "proofs": [{"span_ids": ["method"]}]}
    benchmark = EvidenceBenchmark.model_validate({
        "dataset_version": "case-v1", "task_family": "lookup", "review_status": "reviewed", "sources": [source],
        "spans": [span], "groups": [group], "target_paper_ids": ["paper-a"],
        "paper_requirements": [{"paper_id": "paper-a", "groups": [group]}], "index_snapshots": [index],
        "ranking_policy": {"stage": "query_raw", "top_k": _FIXTURE_TOP_K},
    })
    inputs = {"question": "查找方法原文", "paper_ids": ["paper-a"]}
    raw = {"id": case_id, "case_version": "case-v1", "source_type": "fixture", **inputs,
           "evidence_benchmark": benchmark.model_dump(mode="json")}
    observation = _offline_observation([chunk], inputs, index=index, stage="query_raw")
    return freeze_record(raw, mode="replay", execution_input=inputs, observation=observation)


def write_json(path: Path, value: Any) -> str:
    """参数为临时文件与对象；写入确定字节并返回文件校验和。"""
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_conditions(record: dict[str, Any], overrides: dict[str, str]) -> RunConditions:
    """参数为 fixture 原记录及显式变量；记录测试已知条件，不代表真实模型运行。"""
    case, run = record_contracts(record)
    truth = read_snapshot(case.truth_snapshot)
    observed = read_snapshot(run.observation_snapshot)
    events = list(observed.get("evidence_observations", []))
    for turn in observed.get("turns", []):
        events.extend(read_snapshot(turn.get("evidence_observation_snapshot", "{}")).get("evidence_observations", []))
    names = ["code", "model", "embedding", "prompt", "retrieval", "evaluator", "environment"]
    if "evidence_benchmark" not in truth:
        names.extend(("source", "index"))
    elif not events:
        names.append("index")
    conditions = tuple(IdentityCondition.model_validate({
        "name": name, "status": "recorded", "snapshot": snapshot({"fixture_version": overrides.get(name, name + "-v1")}),
        "reason": "explicit_fixture_execution_conditions",
    }) for name in names)
    return RunConditions(run_id=run.run_id, conditions=conditions)


def write_source(root: Path, kind: ReportKind, records: tuple[dict[str, Any], ...], *,
                 report: dict[str, Any] | None = None, with_identity: bool = False,
                 identity_overrides: dict[str, str] | None = None) -> FrozenReportRef:
    """参数为目录、来源、原记录及选项；调用已有报告器写出实际格式及绑定校验和。"""
    root.mkdir(parents=True, exist_ok=True)
    if report is None:
        builders = {"planner": lambda values: {"records": values}, "evidence": build_evidence_report, "agent": build_agent_report}
        report = builders[kind](list(records))
    report_path = root / f"{kind}-report.json"
    report_digest = write_json(report_path, report)
    frozen_path, frozen_digest = None, None
    if kind != "planner":
        frozen = root / f"{kind}-frozen.jsonl"
        frozen.write_text("".join(snapshot(record) + "\n" for record in records), encoding="utf-8", newline="\n")
        frozen_path, frozen_digest = frozen.name, hashlib.sha256(frozen.read_bytes()).hexdigest()
    identity_path, identity_digest = None, None
    if with_identity:
        identity = RecordedIdentity(report_sha256=report_digest, frozen_sha256=frozen_digest,
                                    runs=tuple(_fixture_conditions(record, identity_overrides or {}) for record in records))
        identity_file = root / f"{kind}-identity.json"
        identity_path = identity_file.name
        identity_digest = write_json(identity_file, identity.model_dump(mode="json"))
    return FrozenReportRef(source_id=kind, kind=kind, report_path=report_path.name, report_sha256=report_digest,
                           frozen_path=frozen_path, frozen_sha256=frozen_digest,
                           identity_path=identity_path, identity_sha256=identity_digest)


def make_slots(records: tuple[dict[str, Any], ...], *, required: tuple[CheckName, ...] = ("truth_leakage",)) -> tuple[RunSlot, ...]:
    """参数为实际记录及适用检查；按题声明重复编号，保留原 run_id。"""
    counts: Counter[tuple[str, str]] = Counter()
    slots = []
    for record in records:
        case, run = record_contracts(record)
        counts[(case.case_id, case.case_version)] += 1
        slots.append(RunSlot(case_id=case.case_id, case_version=case.case_version,
                             repeat_id=f"repeat-{counts[(case.case_id, case.case_version)]}", run_id=run.run_id, required_checks=required))
    return tuple(slots)


def write_manifest(root: Path, sources: tuple[FrozenReportRef, ...], slots: tuple[RunSlot, ...], *,
                   purpose: SuitePurpose = "regression", dataset_version: str = "suite-fixture-v1") -> Path:
    """参数为来源、预期槽位及用途；返回明确标注 fixture 的套件清单。"""
    manifest = SuiteManifest(suite_id="suite-fixture", purpose=purpose, dataset_version=dataset_version, sources=sources, slots=slots)
    path = root / "manifest.json"
    write_json(path, manifest.model_dump(mode="json"))
    return path


@pytest.mark.parametrize("kind", ("planner", "evidence", "agent"))
def test_real_report_shapes_preserve_original_grading(tmp_path: Path, kind: ReportKind) -> None:
    """参数为目录和来源；验证现有评分器产物逐项不变且原文件未被改写。"""
    builders = {"planner": make_planner_record, "evidence": make_evidence_record, "agent": make_agent_record}
    records = (builders[kind](),)
    source = write_source(tmp_path, kind, records)
    original = (tmp_path / source.report_path).read_bytes()
    manifest = write_manifest(tmp_path, (source,), make_slots(records))
    result = load_suite_sources(manifest)
    report = json.loads(original)
    rows = report["rows"] if kind == "agent" else report["records"]
    assert [score.grading.model_dump(mode="json") for score in result.scores] == [row["grading_result"] for row in rows]
    assert (tmp_path / source.report_path).read_bytes() == original
    assert result.runs[0].case.source_type in {"fixture", "agent_scenario"}


def test_checksum_change_is_rejected(tmp_path: Path) -> None:
    """参数为目录；旧引用不能静默读取后来被改写的报告。"""
    records = (make_planner_record(),)
    source = write_source(tmp_path, "planner", records)
    path = write_manifest(tmp_path, (source,), make_slots(records))
    report = tmp_path / source.report_path
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_suite_sources(path)


def test_changed_frozen_projection_is_rejected(tmp_path: Path) -> None:
    """参数为目录；即使文件重新生成校验和，原评分也不能用于不同的运行观察。"""
    original = make_planner_record()
    changed = {**original, "run_record": {**original["run_record"], "observation_snapshot": snapshot({"planner_trace": {"changed": True}})}}
    records = (changed,)
    source = write_source(tmp_path, "planner", records)
    path = write_manifest(tmp_path, (source,), make_slots(records))
    with pytest.raises(ValueError, match="projection checksum"):
        load_suite_sources(path)


def test_crosswired_grading_does_not_join_by_row_order(tmp_path: Path) -> None:
    """参数为目录；两题评分互换时通过身份校验暴露错误，不能按行号拼接。"""
    first, second = make_planner_record("first"), make_planner_record("second")
    records = ({**first, "grading_result": second["grading_result"]}, {**second, "grading_result": first["grading_result"]})
    source = write_source(tmp_path, "planner", records)
    path = write_manifest(tmp_path, (source,), make_slots(records))
    with pytest.raises(ValueError, match="contracts differ"):
        load_suite_sources(path)


def test_historical_identity_is_not_filled_from_current_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录和环境替身；当前模型配置不能成为历史模型身份。"""
    monkeypatch.setenv("LLM_MODEL", "different-current-model")
    records = (make_planner_record(),)
    source = write_source(tmp_path, "planner", records)
    inputs = load_suite_sources(write_manifest(tmp_path, (source,), make_slots(records)))
    model = next(condition for condition in inputs.runs[0].conditions if condition.name == "model")
    assert model.status == "missing"
    assert model.snapshot is None
    assert "different-current-model" not in inputs.model_dump_json()


def test_duplicate_run_cannot_fill_two_repeat_slots(tmp_path: Path) -> None:
    """参数为目录；同一次执行不能假装成两次重复实验。"""
    records = (make_planner_record(),)
    source = write_source(tmp_path, "planner", records)
    first = make_slots(records)[0]
    second = first.model_copy(update={"repeat_id": "repeat-2"})
    with pytest.raises(ValidationError, match="duplicate"):
        write_manifest(tmp_path, (source,), (first, second))


def test_same_score_cannot_be_imported_twice(tmp_path: Path) -> None:
    """参数为目录；给同一原报告换来源名称不能重复增加成绩。"""
    records = (make_planner_record(),)
    source = write_source(tmp_path, "planner", records)
    copy = source.model_copy(update={"source_id": "duplicate"})
    path = write_manifest(tmp_path, (source, copy), make_slots(records))
    with pytest.raises(ValueError, match="duplicate original score"):
        load_suite_sources(path)


def test_evidence_stage_cannot_be_relabelled(tmp_path: Path) -> None:
    """参数为目录；检索命中的原评分不能通过换标签冒充最终上下文覆盖。"""
    records = (make_evidence_record(),)
    report = build_evidence_report(list(records))
    report["records"][0]["stage"] = "final_input"
    source = write_source(tmp_path, "evidence", records, report=report)
    path = write_manifest(tmp_path, (source,), make_slots(records))
    with pytest.raises(ValueError, match="stage observation"):
        load_suite_sources(path)


def test_candidate_review_cannot_be_promoted_by_report_label(tmp_path: Path) -> None:
    """参数为目录；报告标签不能把未审核 Agent 案例升级为 reviewed。"""
    records = (make_agent_record(),)
    report = build_agent_report(records)
    report["rows"][0]["review_status"] = "reviewed"
    source = write_source(tmp_path, "agent", records, report=report)
    path = write_manifest(tmp_path, (source,), make_slots(records))
    with pytest.raises(ValueError, match="review status"):
        load_suite_sources(path)


@pytest.mark.parametrize("field,value", (("retrieval_backend", "other-backend"), ("query", "another-query"), ("top_k", 9)))
def test_evidence_metadata_must_match_frozen_stage(tmp_path: Path, field: str, value: Any) -> None:
    """参数为字段和篡改值；保留原哈希也不能替换查询、检索器或 k 的标签。"""
    records = (make_evidence_record(),)
    report = build_evidence_report(list(records))
    report["records"][0][field] = value
    source = write_source(tmp_path, "evidence", records, report=report)
    path = write_manifest(tmp_path, (source,), make_slots(records))
    with pytest.raises(ValueError, match="metadata"):
        load_suite_sources(path)


def test_agent_status_must_match_its_frozen_observation(tmp_path: Path) -> None:
    """参数为目录；Agent 专属状态与通用观察矛盾时不能由报告标签掩盖。"""
    record = make_agent_record()
    changed = {**record, "agent_run": {**record["agent_run"], "status": "failed"}}
    report = build_agent_report((changed,))
    report["rows"][0]["run_status"] = "completed"
    source = write_source(tmp_path, "agent", (changed,), report=report)
    path = write_manifest(tmp_path, (source,), make_slots((changed,)))
    with pytest.raises(ValueError, match="status"):
        load_suite_sources(path)
