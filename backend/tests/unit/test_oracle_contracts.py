"""Oracle 配对合同与原文来源边界回归。作者：xxx。"""

from __future__ import annotations

import pytest

from evals.evidence.alignment import build_index_mapping
from evals.evidence.contracts import (
    EvidenceBenchmark,
    EvidenceGroup,
    EvidenceProof,
    EvidenceSpan,
    IndexUnit,
    make_index_snapshot,
    make_source_snapshot,
)
from evals.contracts import snapshot_hash
from evals.oracle.contracts import OraclePairSpec
from evals.oracle.source import build_oracle_evidence


def _benchmark(review_status: str = "reviewed") -> EvidenceBenchmark:
    """参数为审核状态；返回包含唯一原文片段和索引映射的测试真值。"""
    text = "The masking ratio is 30%."
    source = make_source_snapshot(
        paper_id="paper-lsttn", pdf_sha256="a" * 64,
        artifact_manifest_snapshot="{}",
        blocks=({"block_id": "block-1", "page_num": 1, "section_path": "method", "text": text},),
    )
    span = EvidenceSpan(
        span_id="span-1", source_snapshot_id=source.snapshot_id, block_id="block-1",
        start=0, end=len(text), quote=text, quote_sha256=snapshot_hash(text),
    )
    mapping = build_index_mapping(
        source, text, index_snapshot_id="placeholder", paper_id=source.paper_id,
        chunk_id="chunk-1", parent_id="parent-1",
    )
    unit = IndexUnit(
        paper_id=source.paper_id, chunk_id="chunk-1", parent_id="parent-1",
        text_snapshot=text, text_sha256=snapshot_hash(text),
    )
    index = make_index_snapshot(
        source_snapshot_ids=(source.snapshot_id,), retrieval_backend="milvus",
        units=(unit,), mappings=(mapping,),
    )
    group = EvidenceGroup(group_id="group-1", importance="critical", proofs=(EvidenceProof(span_ids=(span.span_id,)),))
    return EvidenceBenchmark(
        dataset_version="oracle-dataset-v1", review_status=review_status,
        sources=(source,), spans=(span,), groups=(group,), target_paper_ids=(source.paper_id,),
        index_snapshots=(index,),
    )


def test_oracle_source_requires_reviewed_truth_and_returns_only_original_text() -> None:
    """无参数；候选真值不能进入 Oracle，已审核真值不携带答案字段。"""
    with pytest.raises(Exception, match="reviewed"):
        build_oracle_evidence(_benchmark("candidate"), index_snapshot_id=_benchmark().index_snapshots[0].index_snapshot_id)
    benchmark = _benchmark()
    payloads = build_oracle_evidence(benchmark, index_snapshot_id=benchmark.index_snapshots[0].index_snapshot_id)
    assert payloads[0]["excerpt"] == "The masking ratio is 30%."
    assert "reference_answer" not in payloads[0]
    assert "gold_claim" not in payloads[0]


def test_pair_spec_rejects_shared_cache_namespace() -> None:
    """无参数；Actual 和 Oracle 不能共享缓存命名空间。"""
    with pytest.raises(ValueError, match="cache namespaces"):
        OraclePairSpec(
            pair_id="pair-1", case_id="case-1", case_version="v1",
            input_snapshot='{"question":"Q"}', truth_snapshot="{}",
            control_snapshot='{"target_evidence_ids":["span-1"]}', index_snapshot_id="idx-1",
            actual_cache_namespace="pair-1:actual", oracle_cache_namespace="pair-1:actual",
        )
