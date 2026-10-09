"""【评测导出】【快照合同】真实检索正文不能按字段后缀解析成JSON。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from evals.contracts import (
    MetricResult, build_grading, freeze_record, read_snapshot, rebuild_grading_input,
    record_contracts, snapshot, snapshot_hash,
)
from evals.evidence.contracts import EvidenceBenchmark, ObservedItem, SourceBlock, make_source_snapshot
from evals.evidence.corpus import build_index_from_chunks
from evals.suites.runtime import sanitize_frozen_record

# 1. 【评测导出】【真实复现】run001首个case的IndexMapping正文；仅保留公开正文，不复制用户身份或运行目录
_RETRIEVED_TEXT = "(i) Mean Absolute Error"


def _retrieval_case() -> dict[str, Any]:
    """无参数；用现役来源/索引合同构建与真实prepared case相同的完整非隐私结构。"""
    source = make_source_snapshot(
        paper_id="fixture-paper", pdf_sha256=snapshot_hash("fixture-pdf"), artifact_manifest_snapshot="{}",
        blocks=(SourceBlock(block_id="body", page_num=1, section_path="Metrics", text=_RETRIEVED_TEXT),),
    )
    index, issues = build_index_from_chunks(
        (source,), [{"paper_id": source.paper_id, "chunk_id": "chunk-1", "text": _RETRIEVED_TEXT}],
        retrieval_backend="fixture",
    )
    assert not issues and index.mappings and index.units
    benchmark = EvidenceBenchmark.model_validate({
        "dataset_version": "snapshot-export-v1", "review_status": "candidate", "sources": [source],
        "spans": [], "groups": [], "index_snapshots": [index],
    })
    return {"id": "snapshot-export", "case_version": "v1", "source_type": "real_paper", "reference_answer": _RETRIEVED_TEXT,
            "question": "Which error metric is listed?", "evidence_benchmark": benchmark.model_dump(mode="json")}


def _retrieval_record(*, secret: str = "") -> dict[str, Any]:
    """参数为待脱敏测试凭证；冻结真实形状的IndexMapping/IndexUnit/ObservedItem与嵌套轮次快照。"""
    case = _retrieval_case()
    actual_text = _RETRIEVED_TEXT + (f" {secret}" if secret else "")
    observed = ObservedItem(paper_id="fixture-paper", chunk_id="chunk-1", payload_kind="child", text_snapshot=actual_text)
    observation = {
        "evidence_observations": [{"stage": "query_raw", "payload": {"hits": [{"paper_id": "fixture-paper", "text": actual_text}]}}],
        "model_call_observations": [], "evidence_records": [observed.model_dump(mode="json")],
        "turns": [{"values_snapshot": snapshot({"error": secret, "evidence_records": [observed.model_dump(mode="json")]})}],
    }
    return freeze_record(case, mode="retrieval_live", execution_input={"question": case["question"]}, observation=observation)


@pytest.mark.parametrize("text", [_RETRIEVED_TEXT, '{ "b": 2, "a": 1 }', "[]", "not-json {", ""])
def test_text_snapshot_stays_literal_even_when_it_looks_like_json(text: str) -> None:
    """参数为真实正文及JSON形状正文；文本原样保留，不试解析或重排。"""
    original = {"text_snapshot": text}
    assert sanitize_frozen_record(original, secrets=()) == original


def test_complete_retrieval_case_and_observed_item_export_without_json_parsing() -> None:
    """无参数；整条案例、索引映射/单位以及实际观察均按正文合同导出，所有身份与哈希保持。"""
    record = _retrieval_record()
    before = deepcopy(record)
    clean = sanitize_frozen_record(record, secrets=())
    assert record == before and snapshot(clean) == snapshot(before) and clean is not record
    case, run = record_contracts(clean)
    benchmark = EvidenceBenchmark.model_validate(read_snapshot(case.truth_snapshot)["evidence_benchmark"])
    assert benchmark.index_snapshots[0].mappings[0].text_snapshot == _RETRIEVED_TEXT
    assert benchmark.index_snapshots[0].units[0].text_sha256 == snapshot_hash(_RETRIEVED_TEXT)
    assert read_snapshot(run.observation_snapshot)["evidence_records"][0]["text_snapshot"] == _RETRIEVED_TEXT
    assert snapshot(freeze_record(clean)) == snapshot(clean)
    prepared_case = {"record_snapshot": snapshot(_retrieval_case())}
    assert sanitize_frozen_record(prepared_case, secrets=()) == prepared_case


def test_nested_json_and_plain_strings_redact_without_mutation_and_regrade_consistently() -> None:
    """无参数；JSON转义凭证在普通正文及嵌套真快照内都清除，重建评分投影哈希一致。"""
    secret = 'fixture-key-"\\tail'
    record = _retrieval_record(secret=secret)
    record["diagnostic"] = {"text_snapshot": f"prefix {secret} suffix", "message": secret}
    before = deepcopy(record)
    clean = sanitize_frozen_record(record, secrets=(secret,))
    assert record == before and clean["diagnostic"] == {"text_snapshot": "prefix [redacted] suffix", "message": "[redacted]"}
    _, run = record_contracts(clean)
    observation = read_snapshot(run.observation_snapshot)
    assert observation["evidence_records"][0]["text_snapshot"] == _RETRIEVED_TEXT + " [redacted]"
    assert observation["evidence_observations"][0]["payload"]["hits"][0]["text"] == _RETRIEVED_TEXT + " [redacted]"
    turn = observation["turns"][0]
    assert read_snapshot(turn["values_snapshot"])["error"] == "[redacted]"
    assert read_snapshot(turn["values_snapshot"])["evidence_records"][0]["text_snapshot"] == _RETRIEVED_TEXT + " [redacted]"
    assert secret not in snapshot(clean)
    assert snapshot(freeze_record(clean)) == snapshot(clean)
    measurement = MetricResult(name="export", status="measured", value=1, reason="fixture")
    grading = build_grading(clean, [measurement], grader="export_fixture", grader_version="v1")
    projection = rebuild_grading_input({**clean, "grading_result": grading})
    assert grading["projection_hash"] == snapshot_hash(snapshot(projection))
    old_grading = build_grading(before, [measurement], grader="export_fixture", grader_version="v1")
    assert grading["projection_hash"] != old_grading["projection_hash"]
    with pytest.raises(ValueError, match="checksum mismatch"):
        rebuild_grading_input({**clean, "grading_result": old_grading})


@pytest.mark.parametrize("field", ["input_snapshot", "truth_snapshot", "observation_snapshot", "environment_snapshot",
                                   "values_snapshot", "evidence_observation_snapshot", "artifact_manifest_snapshot", "record_snapshot"])
def test_declared_json_snapshot_corruption_remains_an_explicit_error(field: str) -> None:
    """参数为正式JSON快照字段；损坏合同仍抛错，不能把解析失败降级成普通字符串。"""
    with pytest.raises(ValueError):
        sanitize_frozen_record({field: "not-json"}, secrets=())
    with pytest.raises(ValueError, match="must contain an object"):
        sanitize_frozen_record({field: "[]"}, secrets=())


class _PaperStore:
    """仅替换外部向量库，保留生产检索适配层及真实正文结构。"""

    def hybrid_search(self, query: str, *, top_k: int, filter_expr: str | None) -> list[dict[str, Any]]:
        """参数为实际问题、条数与范围；返回本测试来源中确实存在的正文。"""
        assert query and top_k == 1
        return [{"paper_id": "fixture-paper", "chunk_id": "chunk-1", "text": _RETRIEVED_TEXT}]


@pytest.mark.asyncio
async def test_retrieval_export_artifact_and_ragas_pipeline_retains_actual_text(tmp_path: Path) -> None:
    """参数为临时目录；生产检索冻结、脱敏、证据产物和CP/CR接线完整处理非JSON正文。"""
    from app.services.retrieval_service import search_chunks
    from evals.contracts import create_case
    from evals.ragas.frozen_bridge import score_frozen_record
    from evals.suites.contracts import RunSlot
    from evals.suites.execute import _write_case_artifacts
    from evals.suites.execution_contracts import PreparedCase, PreparedJob
    from evals.suites.policy import workflow_policy
    from evals.suites.runtime import RetrievalRuntime, run_retrieval_job

    raw = _retrieval_case()
    case = create_case(raw)
    index_id = raw["evidence_benchmark"]["index_snapshots"][0]["index_snapshot_id"]
    scope = {"user_id": "fixture-owner", "paper_ids": ["fixture-paper"], "local_retrieval_allowed": True,
             "filter_expr": 'paper_id in ["fixture-paper"]'}
    runtime = RetrievalRuntime(store=_PaperStore(), scope_snapshot=snapshot(scope), top_k=1,
                               index_id=index_id, backend="fixture", search=search_chunks)
    record = await run_retrieval_job(case, runtime)
    assert "failure_reason" not in read_snapshot(record["run_record"]["observation_snapshot"])
    before = deepcopy(record)
    clean = sanitize_frozen_record(record, secrets=())
    assert record == before and snapshot(clean) == snapshot(before)
    job = PreparedJob(job_id="retrieval", kind="retrieval", cases=(PreparedCase(
        case_id=case.case_id, case_version=case.case_version, record_snapshot=snapshot(raw)),),
        slots=(RunSlot(case_id=case.case_id, case_version=case.case_version, repeat_id="1", run_id=None, required_checks=()),))
    report, frozen = _write_case_artifacts(clean, job, output=tmp_path, policy=workflow_policy())
    assert report.is_file() and frozen.is_file()
    calls: list[str] = []

    async def scorer(metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为真实桥接样本；只记录CP/CR测试调用，禁止网络或答案质量调度。"""
        calls.append(metric)
        assert sample["retrieved_contexts"] == [_RETRIEVED_TEXT]
        assert "response" not in sample
        return {"value": 0.75}

    scores = await score_frozen_record(clean, kind="retrieval", scorer=scorer, policy=workflow_policy())
    assert calls == ["context_precision", "context_recall"]
    assert all(row["status"] == "measured" for row in scores["measurements"])
    rebuild_grading_input({**clean, "grading_result": scores["grading_result"]})
