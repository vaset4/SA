"""Evidence 原文校验、离线检索、冻结回放与真实在线评测命令。作者：xxx。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pydantic import ValidationError  # noqa: E402

from evals.contracts import (  # noqa: E402
    RunMode, create_case, freeze_record, read_snapshot, rebuild_grading_input, record_contracts,
    snapshot, snapshot_hash, validate_execution_input,
)
from evals.evidence.alignment import EvidenceDataError, validate_benchmark  # noqa: E402
from evals.evidence.contracts import EvidenceBenchmark, IndexSnapshot  # noqa: E402
from evals.evidence.corpus import (  # noqa: E402
    OFFLINE_BACKEND, build_index_from_chunks, load_index_snapshot, resolve_source_paths,
    snapshot_current_chunks, validate_case_assets,
)
from evals.evidence.report import build_evidence_report, render_evidence_report  # noqa: E402
from evals.ragas.section_recall import _rank_chunks, load_jsonl  # noqa: E402


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    """参数为命令行列表；返回明确选择模式、输入和新输出目录的选项。"""
    parser = argparse.ArgumentParser(description="Versioned Evidence benchmark over real frozen observations")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter", choices=("validate", "offline", "replay", "live"), required=True)
    parser.add_argument("--uploads-dir", type=Path, default=BACKEND_ROOT / "data" / "uploads")
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--grading-policy", choices=("reviewed", "candidate_diagnostic"), default="reviewed")
    parser.add_argument("--index-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _benchmark(sample: dict[str, Any]) -> EvidenceBenchmark:
    """参数为案例或冻结记录；从真值快照加载 Evidence 合同并验证全部引用。"""
    case = record_contracts(sample)[0] if "eval_case" in sample else create_case(sample)
    raw = read_snapshot(case.truth_snapshot).get("evidence_benchmark")
    if raw is None:
        raise EvidenceDataError("missing", "evidence_benchmark_missing")
    result = EvidenceBenchmark.model_validate(raw)
    validate_benchmark(result)
    return result


def _require_unfrozen_cases(samples: list[dict[str, Any]]) -> None:
    """参数为待执行案例；拒绝将历史冻结运行作为新执行输入，避免复用旧观察。"""
    if any("eval_case" in sample or "run_record" in sample for sample in samples):
        raise EvidenceDataError("invalid_input", "execution_requires_unfrozen_cases:use_replay_for_frozen_records")


def _case_validation(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """参数为一题与校验选项；返回实际产物、索引和人工审核的独立状态。"""
    case = record_contracts(sample)[0] if "eval_case" in sample else create_case(sample)
    try:
        benchmark = _benchmark(sample)
        assets = validate_case_assets(benchmark, uploads_dir=args.uploads_dir, paper_dir=args.paper_dir)
        index = load_index_snapshot(args.index_snapshot, sources=benchmark.sources) if args.index_snapshot else None
        status = "invalid_input" if any(row["status"] == "invalid_input" for row in assets) else "measured"
        if status == "measured" and any(row["status"] == "missing" for row in assets):
            status = "missing"
        return {"case_id": case.case_id, "status": status, "assets": assets, "review_status": benchmark.review_status,
                "gold_eligible": benchmark.review_status == "reviewed" and status == "measured",
                "index_snapshot_id": index.index_snapshot_id if index else None,
                "index_status": "measured" if index or benchmark.index_snapshots else "missing",
                "ranking_judgments_status": "measured" if benchmark.ranking_judgments else "missing"}
    except (EvidenceDataError, ValidationError) as exc:
        return {"case_id": case.case_id, "status": exc.status if isinstance(exc, EvidenceDataError) else "invalid_input",
                "reason": str(exc), "gold_eligible": False}


def _source_chunks(benchmark: EvidenceBenchmark, args: argparse.Namespace, cache: dict[str, Any]) -> list[dict[str, Any]]:
    """参数为原文身份、缓存目录与本次缓存；返回经过原文一致性校验的当前 child 全集。"""
    chunks: list[dict[str, Any]] = []
    for source in benchmark.sources:
        if source.snapshot_id not in cache:
            paths = resolve_source_paths(source.paper_id, uploads_dir=args.uploads_dir)
            cache[source.snapshot_id] = snapshot_current_chunks(source, source_paths=paths)
        chunks.extend(cache[source.snapshot_id]["chunks"])
    return chunks


def _offline_observation(hits: list[dict[str, Any]], inputs: dict[str, Any], *, index: IndexSnapshot, stage: str) -> dict[str, Any]:
    """参数为真实 BM25 返回、执行输入和索引；冻结真实顺序，不补造模型回答。"""
    request_id, scope_id = str(uuid4()), str(uuid4())
    payload = {
        "query": inputs["question"], "retrieval_backend": OFFLINE_BACKEND, "index_snapshot_id": index.index_snapshot_id,
        "scope": {"status": "constrained", "allowed_paper_ids": inputs["paper_ids"]},
        "hits": [{**hit, "rank": rank, "index_snapshot_id": index.index_snapshot_id} for rank, hit in enumerate(hits, start=1)],
    }
    event = {"schema_version": "evidence-observation-v1", "event_id": str(uuid4()), "request_id": request_id,
             "scope": {"scope_id": scope_id, "request_id": request_id, "parent_scope_id": None, "stage": "offline_bm25", "unit_id": None},
             "stage": stage, "observed_at_ns": time.monotonic_ns(), "payload": payload}
    return {"evidence_observations": [event], "model_call_observations": [], "evidence_capture_errors": [],
            "retrieved_contexts": [hit["text"] for hit in hits], "context_origin": OFFLINE_BACKEND,
            "metadata": {"adapter": "offline", "retrieval_backend": OFFLINE_BACKEND}}


def _offline_sample(sample: dict[str, Any], args: argparse.Namespace, cache: dict[str, Any]) -> dict[str, Any]:
    """参数为案例、选项和本次解析缓存；执行原有 BM25 后冻结独立观察与新索引映射。"""
    _require_unfrozen_cases([sample])
    benchmark = _benchmark(sample)
    inputs = read_snapshot(create_case(sample).input_snapshot)
    if not isinstance(inputs.get("question"), str) or not inputs["question"].strip():
        raise EvidenceDataError("invalid_input", "offline_question_missing")
    paper_ids = inputs.get("paper_ids")
    if not isinstance(paper_ids, list) or not paper_ids or any(not isinstance(item, str) for item in paper_ids):
        raise EvidenceDataError("missing", "offline_execution_paper_scope_missing")
    if benchmark.ranking_policy is None:
        raise EvidenceDataError("missing", "offline_ranking_policy_missing")
    all_chunks = _source_chunks(benchmark, args, cache)
    corpus_papers = {source.paper_id for source in benchmark.sources}
    if not set(paper_ids).issubset(corpus_papers):
        raise EvidenceDataError("missing", "offline_scope_source_missing")
    index, issues = build_index_from_chunks(benchmark.sources, all_chunks, retrieval_backend=OFFLINE_BACKEND)
    if args.index_snapshot:
        declared = load_index_snapshot(args.index_snapshot, sources=benchmark.sources)
        if declared.index_snapshot_id != index.index_snapshot_id:
            raise EvidenceDataError("invalid_input", "offline_current_index_differs_from_declaration")
    indexes = tuple(item for item in benchmark.index_snapshots if item.index_snapshot_id != index.index_snapshot_id) + (index,)
    current_truth = {**benchmark.model_dump(mode="json"), "index_snapshots": [item.model_dump(mode="json") for item in indexes]}
    prepared = {**sample, "evidence_benchmark": current_truth}
    candidates = [chunk for chunk in all_chunks if chunk["paper_id"] in paper_ids]
    # 1. 检索只读取问题和授权论文内真实切块；证据标注不参与排序
    hits = _rank_chunks(inputs["question"], candidates, benchmark.ranking_policy.top_k)
    observation = _offline_observation(hits, inputs, index=index, stage=benchmark.ranking_policy.stage)
    observation["offline_mapping_issues"] = issues
    frozen: dict[str, Any] = freeze_record(prepared, mode="replay", execution_input=inputs, observation=observation)
    return frozen


def _failed_record(
    sample: dict[str, Any], exc: Exception, *, mode: RunMode,
    inputs: dict[str, Any] | None = None, observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """参数为失败案例、原异常和真实已发生输入/观察；冻结失败原因，回答保持缺失。"""
    _require_unfrozen_cases([sample])
    actual = dict(observation) if observation is not None else {}
    actual["failure_reason"] = f"{type(exc).__name__}:{exc}"
    frozen: dict[str, Any] = freeze_record(sample, mode=mode, execution_input=inputs, observation=actual)
    return frozen


def _offline_records(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """参数为案例集和目录；按题执行并保留实际文件或数据错误，不掩盖程序异常。"""
    cache: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for sample in samples:
        try:
            records.append(_offline_sample(sample, args, cache))
        except (EvidenceDataError, ValidationError, OSError) as exc:
            records.append(_failed_record(sample, exc, mode="replay"))
    return records


def _replay_records(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """参数为已冻结 JSONL 和可选索引；校验既有评分输入及原版本，绝不补写历史观察。"""
    for sample in samples:
        case, run = record_contracts(sample)
        validate_execution_input(read_snapshot(run.input_snapshot))
        if "grading_result" in sample:
            rebuild_grading_input(sample)
        integrity = sample.get("evidence_integrity")
        if not isinstance(integrity, dict) and "grading_result" not in sample:
            raise EvidenceDataError("missing", "frozen_evidence_integrity_missing")
        expected = _record_integrity(sample)
        if integrity is not None and integrity != expected:
            raise EvidenceDataError("invalid_input", "frozen_evidence_integrity_mismatch")
        if args.index_snapshot:
            benchmark = _benchmark(sample)
            index = load_index_snapshot(args.index_snapshot, sources=benchmark.sources)
            if index.index_snapshot_id not in {item.index_snapshot_id for item in benchmark.index_snapshots}:
                raise EvidenceDataError("invalid_input", "replay_index_not_in_original_case")
    return samples


async def _live_records(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """参数为真实案例与运行前索引；复用产品范围解析及 live 适配器，逐题冻结真实失败。"""
    from evals.evidence.live_capture import CapturedRunFailure
    from evals.ragas.rag_adapter import LiveRAGAdapter

    _require_unfrozen_cases(samples)
    records: list[dict[str, Any]] = []
    for sample in samples:
        prepared = sample
        try:
            benchmark = _benchmark(sample)
            issues = validate_case_assets(benchmark, uploads_dir=args.uploads_dir, paper_dir=args.paper_dir)
            if any(issue["status"] != "measured" for issue in issues):
                raise EvidenceDataError("invalid_input", f"live_source_artifacts_unverified:{snapshot({'issues': issues})}")
            index = load_index_snapshot(args.index_snapshot, sources=benchmark.sources) if args.index_snapshot else None
            if index:
                indices = tuple(item for item in benchmark.index_snapshots if item.index_snapshot_id != index.index_snapshot_id) + (index,)
                prepared = {**sample, "evidence_benchmark": {**benchmark.model_dump(mode="json"),
                            "index_snapshots": [item.model_dump(mode="json") for item in indices]}}
            adapter = LiveRAGAdapter(evidence_index_snapshot_id=index.index_snapshot_id if index else None)
            resolved = await asyncio.to_thread(adapter._resolve_sample_scope, prepared)
            records.extend(await adapter._load_async([resolved]))
        except CapturedRunFailure as exc:
            records.append(_failed_record(prepared, exc, mode="live", inputs=read_snapshot(exc.capture.input_snapshot),
                                          observation=read_snapshot(exc.capture.observation_snapshot)))
        except (EvidenceDataError, ValidationError, RuntimeError, ValueError, OSError) as exc:
            records.append(_failed_record(prepared, exc, mode="live"))
    return records


def _record_integrity(record: dict[str, Any]) -> dict[str, str]:
    """参数为冻结记录；返回可重放输入、真值和实际观察的独立完整性校验值。"""
    case, run = record_contracts(record)
    return {"schema_version": "evidence-integrity-v1", "case_sha256": snapshot_hash(snapshot(case.model_dump(mode="json"))),
            "run_sha256": snapshot_hash(snapshot(run.model_dump(mode="json")))}


def _write_outputs(report: dict[str, Any], records: list[dict[str, Any]], *, output_dir: Path) -> None:
    """参数为结果、冻结运行及新目录；拒绝覆盖历史运行并写入可重建 JSONL。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Evidence Evaluation", "", f"Adapter: `{report['adapter']}`", f"Dataset: `{report['dataset_path']}`"]
    if "evidence_benchmark" in report:
        lines.extend(render_evidence_report(report["evidence_benchmark"]))
    if "validation" in report:
        lines.extend(["", "## Validation", "", "```json", json.dumps(report["validation"], ensure_ascii=False, indent=2), "```"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frozen = [{**record, "evidence_integrity": _record_integrity(record)} for record in records]
    (output_dir / "frozen.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in frozen), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """参数为可选命令行；执行失败或输入无效返回 1，其余完成运行返回 0，不按质量分数设门禁。"""
    args = _arguments(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output_dir}")
    samples = load_jsonl(args.dataset)
    if args.adapter in {"offline", "live"}:
        _require_unfrozen_cases(samples)
    if args.adapter == "live":
        from evals.ragas.rag_adapter import load_live_jsonl_dataset

        samples = load_live_jsonl_dataset(args.dataset)
    if not samples:
        raise EvidenceDataError("invalid_input", "dataset_empty")
    report: dict[str, Any] = {"adapter": args.adapter, "dataset_path": str(args.dataset.resolve()),
                              "created_at": datetime.now(timezone.utc).isoformat()}
    records: list[dict[str, Any]] = []
    if args.adapter == "validate":
        report["validation"] = [_case_validation(sample, args) for sample in samples]
        failed = any(row["status"] != "measured" for row in report["validation"])
    else:
        if args.adapter == "offline":
            records = _offline_records(samples, args)
        elif args.adapter == "replay":
            records = _replay_records(samples, args)
        else:
            records = asyncio.run(_live_records(samples, args))
        report["evidence_benchmark"] = build_evidence_report(records, grading_policy=args.grading_policy)
        failed = any(read_snapshot(record["run_record"]["observation_snapshot"]).get("failure_reason") for record in records)
        failed = failed or any(counts["invalid_input_count"] for counts in report["evidence_benchmark"]["coverage"].values())
    _write_outputs(report, records, output_dir=args.output_dir)
    print(f"Evidence report: {args.output_dir.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
