"""从已审核 Evidence 真值构造 Oracle 原文载荷。作者：xxx。"""

from __future__ import annotations

from typing import Any

from evals.evidence.alignment import EvidenceDataError, covered_span_ids, validate_benchmark
from evals.evidence.contracts import EvidenceBenchmark


def build_oracle_evidence(
    benchmark: EvidenceBenchmark,
    *,
    index_snapshot_id: str,
) -> tuple[dict[str, Any], ...]:
    """参数为已审核 Evidence 真值和索引版本；返回不含答案提示的原文证据。"""
    if benchmark.review_status != "reviewed":
        raise EvidenceDataError("invalid_input", "oracle_requires_reviewed_benchmark")
    validate_benchmark(benchmark)
    index = next((item for item in benchmark.index_snapshots if item.index_snapshot_id == index_snapshot_id), None)
    if index is None:
        raise EvidenceDataError("missing", "oracle_index_snapshot_missing")
    mappings = tuple(item for item in index.mappings if item.index_snapshot_id == index_snapshot_id)
    ranges = tuple(mapping_range for mapping in mappings for mapping_range in mapping.ranges)
    covered = covered_span_ids(benchmark.spans, ranges)
    required_span_ids = {
        span_id
        for group in benchmark.groups
        for proof in group.proofs
        for span_id in proof.span_ids
    }
    if not required_span_ids.issubset(covered):
        raise EvidenceDataError("missing", "oracle_evidence_mapping_incomplete")
    blocks = {
        (source.snapshot_id, block.block_id): block
        for source in benchmark.sources
        for block in source.blocks
    }
    spans = {span.span_id: span for span in benchmark.spans}
    payloads: list[dict[str, Any]] = []
    for group in benchmark.groups:
        for proof in group.proofs:
            for span_id in proof.span_ids:
                span = spans[span_id]
                block = blocks[(span.source_snapshot_id, span.block_id)]
                payloads.append({
                    "evidence_id": span.span_id,
                    "paper_id": _paper_id(benchmark, span.source_snapshot_id),
                    "chunk_id": span.block_id,
                    "section": block.section_path,
                    "page_num": block.page_num,
                    "excerpt": span.quote,
                    "source_text": span.quote,
                    "index_snapshot_id": index_snapshot_id,
                    "oracle_group_id": group.group_id,
                })
    return tuple(_dedupe_payloads(payloads))


def _paper_id(benchmark: EvidenceBenchmark, source_snapshot_id: str) -> str:
    """参数为 Evidence 版本身份；返回对应论文身份。"""
    source = next((item for item in benchmark.sources if item.snapshot_id == source_snapshot_id), None)
    if source is None:
        raise EvidenceDataError("invalid_input", "oracle_span_source_missing")
    return source.paper_id


def _dedupe_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """参数为证据载荷；按版本、块和摘录去重，保持真值顺序。"""
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for payload in payloads:
        key = (str(payload["index_snapshot_id"]), str(payload["chunk_id"]), str(payload["excerpt"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(payload)
    return result
