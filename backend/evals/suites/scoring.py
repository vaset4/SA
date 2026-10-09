"""从明确套件清单独立评分，保留每次尝试与原产品。作者：xxx。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Literal

from evals.contracts import read_snapshot, snapshot, snapshot_hash
from evals.ragas.frozen_bridge import build_failed_score, score_frozen_record

from .adapters import load_suite_sources, read_frozen_source
from .contracts import FrozenReportRef, SuiteManifest
from .identity import grading_identity
from .policy import MeasurementPolicy
from .prepare import sha256_file, write_new_json, write_replaced_json
from .runtime import export_secrets, sanitize_frozen_record


def score_attempt(record: dict[str, Any], *, kind: Literal["retrieval", "agent"],
                  policy: MeasurementPolicy, output: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """参数为同次冻结记录、通道、策略、新路径及来源；返回独立评分，失败保留缺失指标和原因。"""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite grading artifact: {output}")
    started = time.perf_counter()
    before = grading_identity(policy)
    try:
        result = asyncio.run(score_frozen_record(record, kind=kind, policy=policy))
    except Exception as exc:
        result = build_failed_score(record, kind=kind, policy=policy, error=f"{type(exc).__name__}:{exc}")
    after = grading_identity(policy)
    if before != after:
        result = {**result, "grading_status": "failed", "grading_error": "grading_identity_changed_during_attempt"}
    result = sanitize_frozen_record({**result, "attempt": {**provenance, "identity_before": before,
                                    "identity_after": after, "identity_drift": before != after},
                                    "grading_seconds": time.perf_counter() - started}, secrets=export_secrets(kind))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(output, result)
    return result


def score_suite(manifest_path: Path, *, output: Path, retry_failed: bool = False) -> Path:
    """参数为已冻结清单、新评分清单及是否显式重评失败；逐槽追加attempt并返回可独立报告的新清单。"""
    root = manifest_path.resolve().parent
    output = output.resolve()
    if output.parent != root:
        raise ValueError("grading_manifest_must_share_product_suite_root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite grading manifest: {output}")
    # 1. 【评测套件】【独立评分】先校验全部原产品、评分及身份哈希，缺失输入不能由其他attempt补拼
    load_suite_sources(manifest_path)
    manifest = SuiteManifest.model_validate_json(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.measurement_policy_snapshot is None:
        raise ValueError("suite_scoring_requires_explicit_measurement_policy")
    policy = MeasurementPolicy.model_validate(read_snapshot(manifest.measurement_policy_snapshot))
    sources = list(manifest.sources)
    for source in sources:
        for value in (source.report_path, source.frozen_path, source.identity_path, source.ragas_path):
            if value is not None and (Path(value).is_absolute() or not (root / value).resolve().is_relative_to(root)):
                raise ValueError("grading_source_must_be_relative_within_suite_root")
    provenance = {"source_manifest": manifest_path.name, "source_manifest_sha256": sha256_file(manifest_path),
                  "attempt_id": output.stem}
    attempt_root = root / "grading" / output.stem
    attempt_root.mkdir(parents=True, exist_ok=False)
    manifest = manifest.model_copy(update={"grading_attempt_snapshot": snapshot({**provenance, "status": "in_progress"})})
    write_new_json(output, manifest.model_dump(mode="json"))
    for index, source in enumerate(sources):
        if source.kind == "planner":
            continue
        loaded = read_frozen_source(source, base_dir=root)
        previous = read_snapshot(loaded.ragas_snapshot) if loaded.ragas_snapshot else None
        if previous is not None and not (retry_failed and previous.get("grading_status") == "failed"):
            continue
        records = read_snapshot(loaded.records_snapshot)["records"]
        if len(records) != 1:
            raise ValueError("suite_score_requires_single_record_per_source")
        target = attempt_root / f"{snapshot_hash(source.source_id)}.json"
        result = score_attempt(records[0], kind="retrieval" if source.kind == "evidence" else "agent",
                               policy=policy, output=target, provenance={**provenance, "source_id": source.source_id,
                               "frozen_path": source.frozen_path, "frozen_sha256": source.frozen_sha256,
                               "previous_ragas_path": source.ragas_path, "previous_ragas_sha256": source.ragas_sha256})
        sources[index] = source.model_copy(update={"ragas_path": target.relative_to(root).as_posix(),
                                                    "ragas_sha256": sha256_file(target)})
        # 2. 【评测套件】【评分进度】只更新本次新清单；中断后显式以该清单继续，不猜测最新目录
        manifest = _scored_manifest(manifest, sources=tuple(sources), result=result)
        write_replaced_json(output, manifest.model_dump(mode="json"))
    manifest = manifest.model_copy(update={"grading_attempt_snapshot": snapshot({**provenance, "status": "finished"})})
    write_replaced_json(output, manifest.model_dump(mode="json"))
    load_suite_sources(output)
    return output


def _scored_manifest(manifest: SuiteManifest, *, sources: tuple[FrozenReportRef, ...], result: dict[str, Any]) -> SuiteManifest:
    """参数为原清单、本次明确引用及评分；返回仅更新评分轴的新清单，产品终态保持原事实。"""
    execution = read_snapshot(manifest.execution_snapshot) if manifest.execution_snapshot else {"runs": []}
    rows = [{**row, "grading_status": result["grading_status"], "grading_error": result.get("grading_error")}
            if row.get("run_id") == result["run_id"] else row for row in execution["runs"]]
    return manifest.model_copy(update={"sources": sources, "execution_snapshot": snapshot({**execution, "runs": rows})})
