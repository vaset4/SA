"""Evidence 捕获、冻结、回放和真实语料边界的确定性集成。作者：xxx。"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Literal

import pytest

from app.middleware.trace_context import (
    evidence_scope, get_trace_ctx, record_evidence_observation,
    record_model_evidence, record_retrieval_hits,
)
from app.models.knowledge_access_contracts import KnowledgeMaterial
from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials
from evals.contracts import read_snapshot, snapshot, snapshot_hash, validate_execution_input
from evals.evidence import run_eval
from evals.evidence.alignment import EvidenceDataError
from evals.evidence.contracts import EvidenceBenchmark, EvidenceSpan, SourceBlock, make_source_snapshot
from evals.evidence.corpus import (
    build_index_from_chunks, load_source_snapshot, resolve_source_paths, validate_case_assets,
)
from evals.evidence.live_capture import CapturedRunFailure, capture_live_run
from evals.evidence.report import build_evidence_report, group_report_rows
from evals.ragas import evals as ragas_evals
from evals.ragas import rag_adapter

_PAPER = "fixture_paper"
_TEXT = "The method predicts traffic flow."
_BACKGROUND = "An unrelated paragraph about soil moisture."


def _read_material(chunk: dict[str, Any], text: str, *, field: Literal["text", "full_text"] = "text") -> KnowledgeMaterial:
    """参数为隔离语料块、实际读取正文及字段；返回知识读取边界的材料替身。"""
    return KnowledgeMaterial(
        paper_id=chunk["paper_id"], batch_id="fixture-batch", group_kind="parent",
        group_locator=chunk["parent_id"], material_group_id=f"fixture-group:{chunk['parent_id']}",
        text=text, source_spans=({"paper_id": chunk["paper_id"], "batch_id": "fixture-batch",
                                  "chunk_id": chunk["chunk_id"], "field": field, "start": 0, "end": len(text),
                                  "text_start": 0, "text_end": len(text)},),
        chunk_id=chunk["chunk_id"], parent_id=chunk["parent_id"], section_path="Method", section="Method",
        page_num=1, source_location_status="legacy_chunk", block_type="text",
    )


def _write_json(path: Path, payload: Any) -> None:
    """参数为临时路径与测试对象；写入明确声明的流程替身数据。"""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_fixture_corpus(root: Path) -> None:
    """参数为临时 uploads；创建互异 PDF 字节及可核验的两个正文块供边界测试。"""
    directory = root / f"{_PAPER}.mineru"
    directory.mkdir(parents=True)
    (root / f"{_PAPER}.pdf").write_bytes(b"explicit test fixture uploaded PDF bytes")
    (directory / "fixture_origin.pdf").write_bytes(b"explicit test fixture retained PDF bytes")
    (directory / "source.md").write_text(f"# 1 Method\n\n{_TEXT}\n\n# 2 Background\n\n{_BACKGROUND}", encoding="utf-8")
    _write_json(directory / "manifest.json", {"paper_id": _PAPER, "parser_mode": "mineru_fixture", "status": "completed"})
    _write_json(directory / "content_list.json", [
        {"type": "text", "text": "1 Method", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": _TEXT, "page_idx": 0},
        {"type": "text", "text": "2 Background", "text_level": 1, "page_idx": 1},
        {"type": "text", "text": _BACKGROUND, "page_idx": 1},
    ])


@pytest.fixture
def fixture_case(tmp_path: Path) -> dict[str, Any]:
    """参数为临时目录；返回已声明流程替身身份的真实可冻结案例及完整索引。"""
    uploads = tmp_path / "uploads"
    _write_fixture_corpus(uploads)
    source = load_source_snapshot(_PAPER, uploads_dir=uploads)
    chunks = [{"paper_id": _PAPER, "chunk_id": f"child-{index}", "parent_id": f"parent-{index}",
               "text": text, "full_text": text} for index, text in enumerate((_TEXT, _BACKGROUND))]
    index, issues = build_index_from_chunks((source,), chunks, retrieval_backend="pipeline_fixture")
    assert issues == []
    quote = "traffic flow"
    span = EvidenceSpan(span_id="flow", source_snapshot_id=source.snapshot_id, block_id="content-item-000001",
                        start=_TEXT.index(quote), end=_TEXT.index(quote) + len(quote), quote=quote, quote_sha256=snapshot_hash(quote))
    group = {"group_id": "task", "importance": "critical", "proofs": [{"span_ids": ["flow"]}]}
    benchmark = EvidenceBenchmark.model_validate({
        "dataset_version": "explicit-fixture-v1", "task_family": "single_paper", "review_status": "reviewed",
        "sources": [source], "spans": [span], "groups": [group], "target_paper_ids": [_PAPER],
        "paper_requirements": [{"paper_id": _PAPER, "groups": [group]}],
        "target_sections": [{"paper_id": _PAPER, "section_path": "1 Method"}],
        "index_snapshots": [index], "ranking_policy": {"stage": "query_raw", "top_k": 2},
    })
    sample = {"id": "fixture-case", "dataset_version": "explicit-fixture-v1", "source_type": "pipeline_fixture",
              "question": "What traffic flow task does the method predict?", "reference_answer": "",
              "paper_ids": [_PAPER], "evidence_benchmark": benchmark.model_dump(mode="json")}
    return {"sample": sample, "source": source, "index": index, "chunks": chunks, "uploads": uploads}


class ObservedFixtureGraph:
    """明确的流程替身，记录实际收到的请求与模拟客户端消息，不代表真实模型质量。"""

    def __init__(self, chunks: list[dict[str, Any]], *, fail: bool = False) -> None:
        """参数为固定返回和失败开关；保存测试依赖并初始化调用记录。"""
        self.chunks = chunks
        self.fail = fail
        self.inputs: list[dict[str, Any]] = []
        self.messages: list[dict[str, str]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """参数为真实适配器传入请求；返回固定业务结果并执行真实观察 API。"""
        validate_execution_input(payload)
        self.inputs.append(deepcopy(payload))
        with evidence_scope("retrieval_query", unit_id="unit-1"):
            record_retrieval_hits("query_raw", self.chunks, query=payload["user_query"],
                                  scope={"status": "constrained", "allowed_paper_ids": [_PAPER]},
                                  retrieval_backend="pipeline_fixture")
            if self.fail:
                raise OSError("fixture retrieval unavailable")
        evidence = [{**chunk, "evidence_id": "e-" + chunk["chunk_id"], "expanded_text": chunk["text"],
                     "layer": "direct_text"} for chunk in self.chunks]
        with evidence_scope("planner_execute"):
            record_evidence_observation("execution_evidence_pool", {"records": evidence, "retrieval_backend": "pipeline_fixture"})
        with evidence_scope("final_context"):
            visible = _TEXT[:10]
            record_model_evidence(evidence[0], visible, rendered_text=visible + "...")
            self.messages = [{"role": "user", "content": "Actual selected excerpt: " + visible + "..."}]
            context = get_trace_ctx()
            assert context is not None
            call = context.begin_call()
            call.input_messages = deepcopy(self.messages)
            call.input_total_chars = len(self.messages[0]["content"])
            call.http_attempt_count = 1
        return {"answer": "Actual fixture answer", "search_results": self.chunks, "evidence_records": evidence}


def _captured_record(fixture: dict[str, Any]) -> tuple[dict[str, Any], ObservedFixtureGraph]:
    """参数为显式 fixture；通过真实 live 适配器捕获并冻结本次返回。"""
    graph = ObservedFixtureGraph(fixture["chunks"])
    adapter = rag_adapter.LiveRAGAdapter(rag_node_factory=lambda: graph,
                                        evidence_index_snapshot_id=fixture["index"].index_snapshot_id)
    records = asyncio.run(adapter._load_async([fixture["sample"]]))
    return records[0], graph


def _metric(report: dict[str, Any], stage: str, name: str) -> dict[str, Any]:
    """参数为报告、阶段和指标名；返回该阶段的确定性测试测量。"""
    row = next(row for row in report["records"] if row["stage"] == stage)
    return next(metric for metric in row["measurements"] if metric["name"] == name)


def test_capture_freeze_grade_replay_preserves_actual_inputs(fixture_case: dict[str, Any], tmp_path: Path) -> None:
    """参数为流程替身和目录；验证真实调用观察冻结、截断、重评分和回放完全一致。"""
    frozen, graph = _captured_record(fixture_case)
    original = deepcopy(frozen)
    report = build_evidence_report([frozen])
    assert _metric(report, "query_raw", "evidence_group_recall_at_k")["value"] == 1.0
    assert _metric(report, "evidence_pool", "evidence_group_pool_coverage")["value"] == 1.0
    assert _metric(report, "final_input", "evidence_group_context_coverage")["value"] == 0.0
    assert frozen == original
    assert len(graph.inputs) == 1
    assert "evidence_benchmark" not in snapshot(graph.inputs[0])
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    assert observed["model_call_observations"][0]["input_messages"] == graph.messages
    target = tmp_path / "first-run"
    run_eval._write_outputs({"adapter": "live", "dataset_path": "fixture", "evidence_benchmark": report}, [frozen], output_dir=target)
    replay = tmp_path / "replay"
    assert run_eval.main(["--adapter", "replay", "--dataset", str(target / "frozen.jsonl"), "--output-dir", str(replay)]) == 0
    assert json.loads((replay / "report.json").read_text(encoding="utf-8"))["evidence_benchmark"] == report


@pytest.mark.parametrize("decorated", [False, True])
def test_real_pool_conversion_uses_published_body_independent_of_preview(fixture_case: dict[str, Any], decorated: bool) -> None:
    """参数为候选预览样式；真实转换只采用知识材料正文，索引预览不改变内容承诺。"""
    frozen, _ = _captured_record(fixture_case)
    preview = f"[Page 1] {_TEXT[:10]}..." if decorated else _TEXT
    chunk = fixture_case["chunks"][0]
    records = convert_knowledge_materials((_read_material(chunk, _TEXT),), hits=[{**chunk, "preview": preview}])
    assert records[0]["expanded_text"] == ""
    assert records[0]["excerpt"] == _TEXT
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    event = next(event for event in observed["evidence_observations"] if event["stage"] == "execution_evidence_pool")
    event["payload"]["records"] = records
    changed = {**frozen, "run_record": {**frozen["run_record"], "observation_snapshot": snapshot(observed)}}
    metric = _metric(build_evidence_report([changed]), "evidence_pool", "evidence_group_pool_coverage")
    assert metric["status"] == "measured"
    assert metric["value"] == 1.0


def test_real_expanded_pool_covers_only_actual_clipped_parent(fixture_case: dict[str, Any]) -> None:
    """参数为流程案例；知识读取只返回父块一部分时，转换不扩展到尚未读取的正文。"""
    visible, hidden = "Early expanded result.", "Hidden tail result."
    parent_text = f"{_TEXT} {visible} " + "paragraph context " * 300 + hidden
    source = make_source_snapshot(
        paper_id=_PAPER, pdf_sha256="a" * 64, artifact_manifest_snapshot=snapshot({"source_type": "pipeline_fixture"}),
        blocks=(SourceBlock(block_id="paragraph", page_num=1, section_path="Method", text=parent_text),),
    )
    chunks = [{"paper_id": _PAPER, "chunk_id": "child-0", "parent_id": "parent-0", "text": _TEXT, "full_text": parent_text}]
    index, issues = build_index_from_chunks((source,), chunks, retrieval_backend="pipeline_fixture")
    assert issues == []
    spans = [EvidenceSpan(span_id=identity, source_snapshot_id=source.snapshot_id, block_id="paragraph",
                         start=parent_text.index(quote), end=parent_text.index(quote) + len(quote),
                         quote=quote, quote_sha256=snapshot_hash(quote)) for identity, quote in (("visible", visible), ("hidden", hidden))]
    groups = [{"group_id": span.span_id, "importance": "critical", "proofs": [{"span_ids": [span.span_id]}]} for span in spans]
    benchmark = EvidenceBenchmark.model_validate({
        "dataset_version": "explicit-fixture-v1", "review_status": "reviewed", "sources": [source], "spans": spans,
        "groups": groups, "index_snapshots": [index], "ranking_policy": {"stage": "query_raw", "top_k": 2},
    })
    sample = {**fixture_case["sample"], "evidence_benchmark": benchmark.model_dump(mode="json")}
    frozen, _ = _captured_record({**fixture_case, "sample": sample, "index": index, "chunks": chunks})
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    event = next(event for event in observed["evidence_observations"] if event["stage"] == "execution_evidence_pool")
    event["payload"]["records"] = convert_knowledge_materials((
        _read_material(chunks[0], f"{_TEXT} {visible}", field="full_text"),
    ))
    actual = event["payload"]["records"][0]["excerpt"]
    assert visible in actual and hidden not in actual and len(actual) < len(parent_text)
    changed = {**frozen, "run_record": {**frozen["run_record"], "observation_snapshot": snapshot(observed)}}
    metric = _metric(build_evidence_report([changed]), "evidence_pool", "evidence_group_pool_coverage")
    assert metric["status"] == "measured"
    assert (metric["value"], metric["numerator"], metric["denominator"]) == (0.5, 1, 2)


@pytest.mark.parametrize("mutation", ["missing_rendered", "forged_tail", "missing_model_call", "capture_error", "not_sent", "empty_messages"])
def test_model_input_gaps_never_receive_full_text_credit(fixture_case: dict[str, Any], mutation: str) -> None:
    """参数为观察破坏方式；部分记录、缺调用或伪造尾部不能变为模型已读全文。"""
    frozen, _ = _captured_record(fixture_case)
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    event = next(event for event in observed["evidence_observations"] if event["stage"] == "model_evidence_text")
    if mutation == "missing_rendered":
        event["payload"].pop("rendered_text")
    elif mutation == "forged_tail":
        event["payload"]["source_text"] = _TEXT
    elif mutation == "missing_model_call":
        observed["model_call_observations"] = []
    elif mutation == "not_sent":
        observed["model_call_observations"][0]["http_attempt_count"] = 0
    elif mutation == "empty_messages":
        observed["model_call_observations"][0]["input_messages"] = []
    else:
        observed["evidence_capture_errors"] = [{"stage": "model_evidence_text", "scope_id": event["scope"]["scope_id"], "reason": "unserializable"}]
    changed = {**frozen, "run_record": {**frozen["run_record"], "observation_snapshot": snapshot(observed)}}
    metric = _metric(build_evidence_report([changed]), "final_input", "evidence_group_context_coverage")
    assert metric["status"] == "missing"
    assert metric["value"] is None


@pytest.mark.parametrize("mutation", ["stage", "scope", "model_call_id", "parent_scope_id", "unit_id",
                                      "model_scope_id", "capture_scope", "capture_stage",
                                      "effective_scope_missing_ids", "effective_scope_string_ids"])
def test_malformed_event_is_invalid_input_not_key_error(fixture_case: dict[str, Any], mutation: str) -> None:
    """参数为损坏字段；外部快照结构错误报告 invalid_input，而不是程序 KeyError。"""
    frozen, _ = _captured_record(fixture_case)
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    if mutation == "stage":
        observed["evidence_observations"][1].pop("stage")
    elif mutation == "scope":
        observed["evidence_observations"][1]["scope"].pop("scope_id")
    elif mutation == "model_call_id":
        observed["model_call_observations"][0].pop("call_id")
    elif mutation in {"parent_scope_id", "unit_id"}:
        observed["evidence_observations"][1]["scope"][mutation] = []
    elif mutation == "model_scope_id":
        observed["model_call_observations"][0]["evidence_scope_id"] = []
    elif mutation.startswith("effective_scope_"):
        scope = {"user_id": "fixture-user", "local_retrieval_allowed": True, "filter_expr": "fixture-filter"}
        if mutation == "effective_scope_string_ids":
            scope["paper_ids"] = "abc"
        query = next(event for event in observed["evidence_observations"] if event["stage"] == "query_raw")
        query["payload"]["scope"] = scope
    else:
        observed["evidence_capture_errors"] = [{"scope_id": [] if mutation == "capture_scope" else None,
                                                "stage": [] if mutation == "capture_stage" else "query_raw"}]
    changed = {**frozen, "run_record": {**frozen["run_record"], "observation_snapshot": snapshot(observed)}}
    report = build_evidence_report([changed])
    assert report["coverage"]["evidence_observation_integrity"]["invalid_input_count"] == 1


def test_replay_rejects_changed_frozen_observation(fixture_case: dict[str, Any]) -> None:
    """参数为流程替身；冻结后修改观察必须被完整性校验拒绝。"""
    frozen, _ = _captured_record(fixture_case)
    frozen["evidence_integrity"] = run_eval._record_integrity(frozen)
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    observed["response"] = "forged answer"
    frozen["run_record"]["observation_snapshot"] = snapshot(observed)
    with pytest.raises(EvidenceDataError, match="integrity_mismatch"):
        run_eval._replay_records([frozen], argparse.Namespace(index_snapshot=None))


@pytest.mark.parametrize("invalid_text", [False, True])
def test_cli_exit_distinguishes_invalid_input_from_measured_zero(
    fixture_case: dict[str, Any], tmp_path: Path, *, invalid_text: bool,
) -> None:
    """参数为案例、目录和文本有效性；无效观察退出 1，真实零召回及缺排序标注仍退出 0。"""
    frozen, _ = _captured_record(fixture_case)
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    event = next(event for event in observed["evidence_observations"] if event["stage"] == "query_raw")
    hit = {**event["payload"]["hits"][1], "rank": 1}
    if invalid_text:
        hit["text"] = "Changed text does not belong to the declared index."
    event["payload"]["hits"] = [hit]
    changed = {**frozen, "run_record": {**frozen["run_record"], "observation_snapshot": snapshot(observed)}}
    report = build_evidence_report([changed])
    metric = _metric(report, "query_raw", "evidence_group_recall_at_k")
    assert metric["status"] == ("invalid_input" if invalid_text else "measured")
    assert metric["value"] == (None if invalid_text else 0.0)
    output = tmp_path / "input"
    run_eval._write_outputs({"adapter": "live", "dataset_path": "fixture", "evidence_benchmark": report}, [changed], output_dir=output)
    assert run_eval.main(["--adapter", "replay", "--dataset", str(output / "frozen.jsonl"),
                          "--output-dir", str(tmp_path / "replay")]) == int(invalid_text)


@pytest.mark.parametrize("mutation", ["undeclared", "changed_text", "late_declaration"])
def test_index_mapping_needs_actual_declaration_and_matching_text(fixture_case: dict[str, Any], mutation: str) -> None:
    """参数为版本或正文破坏方式；ID 相同或当前索引存在不能替代原运行身份。"""
    frozen, _ = _captured_record(fixture_case)
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    events = observed["evidence_observations"]
    if mutation == "undeclared":
        events.pop(0)
    elif mutation == "late_declaration":
        events.append(events.pop(0))
    else:
        next(event for event in events if event["stage"] == "query_raw")["payload"]["hits"][0]["text"] = "changed original text"
    changed = {**frozen, "run_record": {**frozen["run_record"], "observation_snapshot": snapshot(observed)}}
    report = build_evidence_report([changed])
    if mutation == "late_declaration":
        assert report["coverage"]["evidence_observation_integrity"]["invalid_input_count"] == 1
    else:
        metric = _metric(report, "query_raw", "evidence_group_recall_at_k")
        assert metric["status"] == ("missing" if mutation == "undeclared" else "invalid_input")


def test_candidate_scope_violation_is_kept_outside_quality_mean(fixture_case: dict[str, Any]) -> None:
    """参数为流程替身；候选无黄金分数，但真实越界仍明确失败且不会被平均消失。"""
    frozen, _ = _captured_record(fixture_case)
    truth = read_snapshot(frozen["eval_case"]["truth_snapshot"])
    truth["evidence_benchmark"]["review_status"] = "candidate"
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    event = next(event for event in observed["evidence_observations"] if event["stage"] == "query_raw")
    event["payload"]["hits"].append({"paper_id": "outside", "chunk_id": "x", "text": "outside", "rank": 3})
    changed = {**frozen, "eval_case": {**frozen["eval_case"], "truth_snapshot": snapshot(truth)},
               "run_record": {**frozen["run_record"], "observation_snapshot": snapshot(observed)}}
    report = build_evidence_report([changed])
    assert _metric(report, "query_raw", "evidence_group_recall_at_k")["status"] == "not_measured"
    assert _metric(report, "query_raw", "scope_adherence")["value"] == 0.0
    assert report["scope_violation_count"] == 1


@pytest.mark.parametrize("page", [True, 0.5, "0", -1, None])
def test_corpus_rejects_non_integer_raw_pages(fixture_case: dict[str, Any], page: Any) -> None:
    """参数为伪页码；保留原始整数坐标边界，解析器转换不能掩盖非法位置。"""
    path = fixture_case["uploads"] / f"{_PAPER}.mineru" / "content_list.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    items[1]["page_idx"] = page
    _write_json(path, items)
    with pytest.raises(EvidenceDataError, match="source_page_missing"):
        load_source_snapshot(_PAPER, uploads_dir=fixture_case["uploads"])


def test_corpus_ignores_noise_without_physical_page(fixture_case: dict[str, Any]) -> None:
    """参数为临时语料；未进入正文的噪音项目可不带页码，不能因提前校验改变正文集合。"""
    path = fixture_case["uploads"] / f"{_PAPER}.mineru" / "content_list.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    _write_json(path, [*items, {"type": "footer", "text": "ignored footer"}])
    source = load_source_snapshot(_PAPER, uploads_dir=fixture_case["uploads"])
    assert source.blocks == fixture_case["source"].blocks


def test_corpus_identity_checks_each_artifact_and_blocks_path_escape(fixture_case: dict[str, Any]) -> None:
    """参数为临时真实文件；分别验证两种 PDF 身份、解析修改及目录逃逸拒绝。"""
    source = fixture_case["source"]
    artifacts = read_snapshot(source.artifact_manifest_snapshot)["files"]
    assert artifacts["uploaded_pdf"]["sha256"] != artifacts["retained_pdf"]["sha256"]
    assert source.parser_engine_version is None
    benchmark = EvidenceBenchmark.model_validate(fixture_case["sample"]["evidence_benchmark"])
    path = fixture_case["uploads"] / f"{_PAPER}.mineru" / "source.md"
    path.write_text("changed markdown", encoding="utf-8")
    assert validate_case_assets(benchmark, uploads_dir=fixture_case["uploads"])[0]["status"] == "invalid_input"
    with pytest.raises(EvidenceDataError, match="invalid_paper_path_identity"):
        resolve_source_paths("../fixture_paper", uploads_dir=fixture_case["uploads"])


def test_validate_offline_cli_and_overwrite_guard(fixture_case: dict[str, Any], tmp_path: Path) -> None:
    """参数为可解析 corpus；运行真实切块和 BM25，并核实命令输出及历史目录保护。"""
    dataset = tmp_path / "candidate.jsonl"
    sample = deepcopy(fixture_case["sample"])
    sample["evidence_benchmark"]["review_status"] = "candidate"
    _write_json(dataset, sample)
    shared = ["--dataset", str(dataset), "--uploads-dir", str(fixture_case["uploads"])]
    assert run_eval.main([*shared, "--adapter", "validate", "--output-dir", str(tmp_path / "validation")]) == 0
    target = tmp_path / "offline"
    assert run_eval.main([*shared, "--adapter", "offline", "--output-dir", str(target)]) == 0
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert report["adapter"] == "offline"
    frozen = json.loads((target / "frozen.jsonl").read_text(encoding="utf-8"))
    assert "response" not in read_snapshot(frozen["run_record"]["observation_snapshot"])
    # 1. 已冻结产物只能回放；再次执行不能复用旧运行而丢失新观察
    for adapter in ("offline", "live"):
        with pytest.raises(EvidenceDataError, match="execution_requires_unfrozen_cases"):
            run_eval.main(["--adapter", adapter, "--dataset", str(target / "frozen.jsonl"),
                           "--output-dir", str(tmp_path / f"{adapter}-rerun")])
    with pytest.raises(EvidenceDataError, match="execution_requires_unfrozen_cases"):
        run_eval._failed_record(frozen, ValueError("new failure"), mode="live")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_eval.main([*shared, "--adapter", "offline", "--output-dir", str(target)])


def test_live_failure_keeps_actual_capture_without_answer(fixture_case: dict[str, Any]) -> None:
    """参数为失败流程替身；真实异常保留原始 cause、已有观察和输入，回答不存在。"""
    graph = ObservedFixtureGraph(fixture_case["chunks"], fail=True)
    inputs = {"user_query": "actual question"}
    with pytest.raises(CapturedRunFailure) as failed:
        asyncio.run(capture_live_run(graph, inputs, index_snapshot_id=fixture_case["index"].index_snapshot_id))
    assert isinstance(failed.value.__cause__, OSError)
    captured = failed.value.capture
    frozen = run_eval._failed_record(fixture_case["sample"], failed.value, mode="live",
                                     inputs=read_snapshot(captured.input_snapshot), observation=read_snapshot(captured.observation_snapshot))
    observed = read_snapshot(frozen["run_record"]["observation_snapshot"])
    assert "response" not in observed
    assert observed["evidence_observations"]
    assert frozen["run_record"]["provenance"] == "captured"
    assert "captured_execution_input" not in frozen["run_record"]["missing_fields"]
    assert get_trace_ctx() is None


@pytest.mark.parametrize("mode", ["live", "replay"])
def test_preflight_failure_does_not_claim_captured_execution_input(fixture_case: dict[str, Any], mode: str) -> None:
    """参数为执行模式；调用前校验失败仅保留声明输入，不能伪称已有真实执行捕获。"""
    record = run_eval._failed_record(fixture_case["sample"], EvidenceDataError("missing", "source_unavailable"), mode=mode)
    run = record["run_record"]
    assert run["provenance"] == "legacy"
    assert "captured_execution_input" in run["missing_fields"]
    observed = read_snapshot(run["observation_snapshot"])
    assert "source_unavailable" in observed["failure_reason"]
    assert "evidence_observations" not in observed
    assert "response" not in observed


@pytest.mark.parametrize("invalid_truth", [False, True])
def test_failure_output_replays_without_revalidating_unexecuted_input(
    fixture_case: dict[str, Any], tmp_path: Path, *, invalid_truth: bool,
) -> None:
    """参数为流程案例、输出目录及真值状态；失败运行可重评分且保持失败退出状态。"""
    sample = deepcopy(fixture_case["sample"])
    if invalid_truth:
        sample["evidence_benchmark"]["spans"][0]["end"] = -1
    record = run_eval._failed_record(sample, ValueError("explicit preflight failure"), mode="live")
    report = build_evidence_report([record])
    output = tmp_path / "failed"
    run_eval._write_outputs({"adapter": "live", "dataset_path": "fixture", "evidence_benchmark": report}, [record], output_dir=output)
    replay = tmp_path / "replay"
    assert run_eval.main(["--adapter", "replay", "--dataset", str(output / "frozen.jsonl"), "--output-dir", str(replay)]) == 1
    assert json.loads((replay / "report.json").read_text(encoding="utf-8"))["evidence_benchmark"] == report


def _fixture_grader(records: list[dict[str, Any]]) -> dict[str, Any]:
    """参数为可变评分副本；故意修改输入来验证原冻结运行不会污染。"""
    records[0]["response"] = "forged answer"
    return {"metrics": {"faithfulness": 0.9}, "records": [{"faithfulness": 0.9}]}


def test_ragas_loaders_and_report_preserve_independent_evidence(fixture_case: dict[str, Any], tmp_path: Path) -> None:
    """参数为流程替身；两种 loader 保留真值且 RAGAS 报告单独展示 Evidence 评分。"""
    frozen, _ = _captured_record(fixture_case)
    dataset = tmp_path / "ragas.jsonl"
    _write_json(dataset, frozen)
    assert rag_adapter.load_jsonl_dataset(dataset)[0]["evidence_benchmark"] == frozen["evidence_benchmark"]
    live_dataset = tmp_path / "live.jsonl"
    _write_json(live_dataset, fixture_case["sample"])
    assert rag_adapter.load_live_jsonl_dataset(live_dataset)[0]["evidence_benchmark"] == frozen["evidence_benchmark"]
    report = ragas_evals.run_evaluation(dataset_path=dataset, output_dir=tmp_path / "ragas", evaluator=_fixture_grader)
    assert report["records"][0]["response"] == "Actual fixture answer"
    assert report["evidence_benchmark"]["grader_version"] == "evidence-benchmark-v1"
    assert "Evidence Benchmark" in Path(report["markdown_report_path"]).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_benchmark"):
        validate_execution_input({"structured_query": {"nested": [{"evidence_benchmark": {}}]}})


async def _resolved_fixture_scope() -> dict[str, Any]:
    """无参数；模拟现有项目范围服务的异步返回，不读取真实业务数据库。"""
    return {"paper_ids": [_PAPER]}


class ProjectScopeFixtureAdapter(rag_adapter.LiveRAGAdapter):
    """用于验证同步范围解析内部运行异步服务的现有调用约定。"""

    def _resolve_sample_scope(self, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为案例；沿真实同步 resolver 的 asyncio.run 约定获取项目范围。"""
        scope = asyncio.run(_resolved_fixture_scope())
        return {**sample, **scope}

    async def _load_async(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """参数为已解析案例；记录范围并显式模拟无结果的执行失败。"""
        assert samples[0]["paper_ids"] == [_PAPER]
        raise OSError("explicit fixture graph failure after project scope resolution")


def test_live_cli_project_scope_resolves_outside_running_loop(fixture_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为案例和替身注入器；同步项目解析可调用异步服务并保留后续真实失败。"""
    monkeypatch.setattr(rag_adapter, "LiveRAGAdapter", ProjectScopeFixtureAdapter)
    sample = {**fixture_case["sample"], "project_id": "fixture-project"}
    options = argparse.Namespace(index_snapshot=None, uploads_dir=fixture_case["uploads"], paper_dir=None)
    records = asyncio.run(run_eval._live_records([sample], options))
    observed = read_snapshot(records[0]["run_record"]["observation_snapshot"])
    assert "after project scope resolution" in observed["failure_reason"]
    assert "asyncio.run() cannot be called" not in observed["failure_reason"]
    assert "response" not in observed


def test_group_means_weight_cases_equally_and_keep_actual_counts() -> None:
    """无参数；三次成功调用的一题与一次失败调用的一题在题级均值中各占一半。"""
    rows = []
    for case_id, scores in (("many-calls", [1, 1, 1]), ("one-call", [0])):
        for number, score in enumerate(scores):
            rows.append({"case_id": case_id, "run_id": case_id, "call_id": str(number), "stage": "query_raw",
                         "retrieval_backend": "pipeline_fixture", "source_type": "pipeline_fixture", "task_family": "lookup",
                         "breakdowns": [], "measurements": [{"name": "evidence_group_recall_at_k", "status": "measured",
                         "value": score, "numerator": score, "denominator": 1, "reason": "verified_observation"}]})
    group = next(group for group in group_report_rows(rows) if group["dimension"] == "task_family")
    assert group["mean"] == 0.5
    assert group["case_count"] == 2
    counts = [item["measurement"]["denominator"] for case in group["case_results"] for item in case["observations"]]
    assert counts == [1, 1, 1, 1]


def test_all_report_dimensions_keep_real_and_fixture_sources_separate() -> None:
    """无参数；同阶段和检索后端的真实题与流程替身不得混入论文、等级或任务均值。"""
    rows = []
    for source_type, score in (("real_paper", 1.0), ("synthetic_fixture", 0.0)):
        measurements = [{"name": "evidence_group_recall_at_k", "status": "measured", "value": score,
                         "numerator": int(score), "denominator": 1, "reason": "verified_observation"}]
        rows.append({"case_id": source_type, "run_id": source_type, "call_id": source_type, "stage": "query_raw",
                     "retrieval_backend": "same_declared_backend", "source_type": source_type, "task_family": "lookup",
                     "measurements": measurements, "breakdowns": [
                         {"dimension": "paper", "label": "same-paper", "measurements": measurements},
                         {"dimension": "importance", "label": "critical", "measurements": measurements},
                     ]})
    groups = group_report_rows(rows)
    for dimension in ("task_family", "paper", "importance"):
        selected = [group for group in groups if group["dimension"] == dimension]
        assert len(selected) == 2
        assert {group["source_type"]: group["mean"] for group in selected} == {"real_paper": 1.0, "synthetic_fixture": 0.0}


class QueryFailureFixtureGraph:
    """显式模拟同批多查询中一次失败，区分返回前失败与业务过滤失败。"""

    def __init__(self, chunks: list[dict[str, Any]], *, after_raw: bool) -> None:
        """参数为真实测试命中和失败位置；保存注入的查询结果。"""
        self.chunks = chunks
        self.after_raw = after_raw

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """参数为实际问题；执行一个成功查询和一个指定阶段失败的查询。"""
        with evidence_scope("direct_text_adapter", details={"retrieval_backend": "pipeline_fixture"}):
            with evidence_scope("retrieval_query"):
                record_retrieval_hits("query_raw", self.chunks, query=payload["user_query"], retrieval_backend="pipeline_fixture")
                record_retrieval_hits("query_filtered", self.chunks, query=payload["user_query"], retrieval_backend="pipeline_fixture")
            with evidence_scope("retrieval_query"):
                if self.after_raw:
                    record_retrieval_hits("query_raw", self.chunks, query=payload["user_query"], retrieval_backend="pipeline_fixture")
                raise OSError("explicit failed query stage")


@pytest.mark.parametrize("after_raw", [False, True])
def test_failed_query_stage_cannot_disappear_or_erase_completed_raw(fixture_case: dict[str, Any], after_raw: bool) -> None:
    """参数为失败位置；失败调用保留 missing，已真实返回的 raw 仍可评分。"""
    graph = QueryFailureFixtureGraph(fixture_case["chunks"], after_raw=after_raw)
    with pytest.raises(CapturedRunFailure) as failed:
        asyncio.run(capture_live_run(graph, {"user_query": fixture_case["sample"]["question"]},
                                     index_snapshot_id=fixture_case["index"].index_snapshot_id))
    capture = failed.value.capture
    frozen = run_eval._failed_record(fixture_case["sample"], failed.value, mode="live",
                                     inputs=read_snapshot(capture.input_snapshot), observation=read_snapshot(capture.observation_snapshot))
    report = build_evidence_report([frozen])
    raws = [row for row in report["records"] if row["stage"] == "query_raw"]
    measured = [next(metric for metric in row["measurements"] if metric["name"] == "evidence_group_recall_at_k") for row in raws]
    assert sum(metric["status"] == "measured" for metric in measured) == (2 if after_raw else 1)
    failed_stage = "query_filtered" if after_raw else "query_raw"
    grouped = next(group for group in report["groups"] if group["dimension"] == "task_family"
                   and group["stage"] == failed_stage and group["metric"] == "evidence_group_recall_at_k")
    assert grouped["retrieval_backend"] == "pipeline_fixture"
    assert grouped["mean"] is None
