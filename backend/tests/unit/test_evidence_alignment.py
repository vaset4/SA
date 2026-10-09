"""验证原文身份、唯一定位、截断和跨片段覆盖。作者：xxx。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evals.contracts import snapshot, snapshot_hash
from evals.evidence.alignment import (
    EvidenceDataError, align_observed_item, build_index_mapping, covered_span_ids,
    interval_covered, locate_span, normalize_with_mapping, validate_benchmark,
    validate_mapping, validate_span,
)
from evals.evidence.contracts import (
    EvidenceBenchmark, EvidenceGroup, EvidenceProof, EvidenceSpan, IndexMapping,
    IndexSnapshot, IndexUnit, MappedRange, ObservedItem, SourceBlock,
    SourceSnapshot, make_index_snapshot, make_source_snapshot,
)


def _source(text: str, *, paper_id: str = "paper-a") -> SourceSnapshot:
    """参数为测试正文及论文；返回明确标识为流程测试的完整原文快照。"""
    return make_source_snapshot(
        paper_id=paper_id, pdf_sha256=snapshot_hash("fixture-pdf"),
        artifact_manifest_snapshot=snapshot({"fixture": True, "text_sha256": snapshot_hash(text)}),
        blocks=(SourceBlock(block_id="block-a", page_num=1, section_path="Methods", text=text),),
    )


def _span(source: SourceSnapshot, quote: str, *, span_id: str = "target") -> EvidenceSpan:
    """参数为测试原文、摘录及标识；返回经唯一定位的片段。"""
    block_id, start, end = locate_span(source, quote)
    return EvidenceSpan(
        span_id=span_id, source_snapshot_id=source.snapshot_id, block_id=block_id,
        start=start, end=end, quote=quote, quote_sha256=snapshot_hash(quote),
    )


def _benchmark(source: SourceSnapshot, *, text: str, chunk_id: str = "chunk-a") -> EvidenceBenchmark:
    """参数为原文及实际 child 正文；返回含真实映射的测试真值。"""
    mapping = build_index_mapping(source, text, index_snapshot_id="pending", paper_id=source.paper_id, chunk_id=chunk_id)
    index = make_index_snapshot(
        source_snapshot_ids=(source.snapshot_id,), retrieval_backend="fixture",
        units=(IndexUnit(paper_id=source.paper_id, chunk_id=chunk_id, text_snapshot=text, text_sha256=snapshot_hash(text)),),
        mappings=(mapping,),
    )
    return EvidenceBenchmark(
        dataset_version="fixture-v1", review_status="reviewed", sources=(source,),
        spans=(), groups=(), index_snapshots=(index,),
    )


def test_exact_locator_rejects_repetition_even_in_same_section() -> None:
    """无参数；验证同章重复摘录必须消歧，返回 None。"""
    source = _source("same text; same text")
    with pytest.raises(EvidenceDataError, match="span_match_count=2"):
        locate_span(source, "same text")
    explicit = EvidenceSpan(
        span_id="reviewed-position", source_snapshot_id=source.snapshot_id,
        block_id="block-a", start=11, end=20, quote="same text", quote_sha256=snapshot_hash("same text"),
    )
    validate_span(explicit, source)


def test_locator_constraints_and_overlapping_matches_are_real() -> None:
    """无参数；验证块/章节/页约束和相互重叠的重复位置，返回 None。"""
    source = make_source_snapshot(
        paper_id="paper-a", pdf_sha256=snapshot_hash("pdf"), artifact_manifest_snapshot="{}",
        blocks=(
            SourceBlock(block_id="one", page_num=1, section_path="A", text="repeat"),
            SourceBlock(block_id="two", page_num=2, section_path="B", text="repeat"),
        ),
    )
    assert locate_span(source, "repeat", section_paths=("B",), page_nums=(2,)) == ("two", 0, 6)
    assert locate_span(source, "repeat", block_ids=("one",)) == ("one", 0, 6)
    with pytest.raises(EvidenceDataError, match="span_match_count=2"):
        locate_span(_source("aaa"), "aa")


@pytest.mark.parametrize("value", [True, 0.5, "1", -1])
def test_span_rejects_non_integer_or_negative_positions(value: object) -> None:
    """参数为错误位置；验证布尔值和可转换字符串不会被当作真实坐标。"""
    span = _span(_source("alpha"), "alpha")
    with pytest.raises(ValidationError):
        EvidenceSpan.model_validate({**span.model_dump(), "start": value})


def test_source_and_quote_checksums_reject_changed_content() -> None:
    """无参数；验证同 ID 内容变化和摘录 checksum 伪造均失败，返回 None。"""
    source = _source("alpha")
    payload = source.model_dump(mode="json")
    payload["blocks"][0]["text"] = "changed"
    with pytest.raises(ValidationError, match="source_snapshot_checksum_mismatch"):
        SourceSnapshot.model_validate(payload)
    span = _span(source, "alpha")
    with pytest.raises(ValidationError, match="span_checksum_mismatch"):
        EvidenceSpan.model_validate({**span.model_dump(), "quote_sha256": "0" * 64})
    with pytest.raises(EvidenceDataError, match="source_version_mismatch"):
        validate_span(span, _source("alpha", paper_id="paper-b"))


def test_source_nested_fields_are_frozen_and_duplicate_blocks_fail() -> None:
    """无参数；验证深层集合不可变及重复块身份被拒绝，返回 None。"""
    source = _source("alpha")
    with pytest.raises(ValidationError, match="frozen_instance"):
        source.blocks[0].text = "changed"
    with pytest.raises(ValidationError, match="duplicate source block"):
        make_source_snapshot(
            paper_id="paper-a", pdf_sha256=source.pdf_sha256,
            artifact_manifest_snapshot="{}", blocks=(source.blocks[0], source.blocks[0]),
        )


def test_partial_ranges_do_not_cover_a_gap() -> None:
    """无参数；验证缺一个字符仍未召回完整片段，重叠不会重复计数。"""
    assert not interval_covered(0, 10, [(0, 4), (5, 10)])
    assert interval_covered(0, 10, [(0, 6), (4, 10)])
    assert not interval_covered(0, 10, [(0, 4), (0, 4)])
    with pytest.raises(EvidenceDataError, match="coverage_observed_range_invalid"):
        interval_covered(0, 10, [(False, 10)])


def test_multi_hit_coverage_deduplicates_by_source_and_block() -> None:
    """无参数；验证合并真实区间可覆盖片段，其他版本和块不可补齐。"""
    source = _source("abcdefghij")
    span = _span(source, "abcdefghij")
    first = build_index_mapping(source, "abcde", index_snapshot_id="v1", paper_id="paper-a", chunk_id="one")
    second = build_index_mapping(source, "efghij", index_snapshot_id="v1", paper_id="paper-a", chunk_id="two")
    assert covered_span_ids((span,), first.ranges) == frozenset()
    assert covered_span_ids((span,), (*first.ranges, *second.ranges, *first.ranges)) == {"target"}
    other = second.ranges[0].model_copy(update={"block_id": "other"})
    assert covered_span_ids((span,), (*first.ranges, other)) == frozenset()


def test_mapping_checks_actual_text_paper_and_payload_ranges() -> None:
    """无参数；验证 ID 相同不能绕过实际文本与来源检查，返回 None。"""
    source = _source("alpha beta")
    mapping = build_index_mapping(source, "alpha", index_snapshot_id="v1", paper_id="paper-a", chunk_id="one")
    changed = IndexMapping.model_validate({**mapping.model_dump(), "text_snapshot": "omega", "text_sha256": snapshot_hash("omega")})
    with pytest.raises(EvidenceDataError, match="mapping_text_mismatch"):
        validate_mapping(changed, {source.snapshot_id: source})
    with pytest.raises(EvidenceDataError, match="mapping_paper_mismatch"):
        validate_mapping(mapping.model_copy(update={"paper_id": "paper-b"}), {source.snapshot_id: source})
    with pytest.raises(EvidenceDataError, match="mapping_source_snapshot_missing"):
        validate_mapping(mapping, {})


def test_unmapped_payload_is_missing_and_not_silently_deleted() -> None:
    """无参数；验证无法解释的正文保留 missing，不用局部映射推导完整观察。"""
    source = _source("alpha")
    mapping = build_index_mapping(source, "alpha", index_snapshot_id="v1", paper_id="paper-a", chunk_id="one")
    changed = IndexMapping.model_validate({**mapping.model_dump(), "text_snapshot": "alpha unknown", "text_sha256": snapshot_hash("alpha unknown")})
    with pytest.raises(EvidenceDataError, match="mapping_payload_not_fully_located"):
        validate_mapping(changed, {source.snapshot_id: source})
    with pytest.raises(EvidenceDataError, match="index_text_not_located"):
        build_index_mapping(source, "unknown", index_snapshot_id="v1", paper_id="paper-a", chunk_id="one")


def test_mapping_can_join_exact_lines_from_separate_blocks() -> None:
    """无参数；验证跨块正文按真实行边界定位，不将相同章节当作同一块。"""
    source = make_source_snapshot(
        paper_id="paper-a", pdf_sha256=snapshot_hash("pdf"), artifact_manifest_snapshot="{}",
        blocks=(
            SourceBlock(block_id="one", page_num=1, section_path="A", text="alpha"),
            SourceBlock(block_id="two", page_num=2, section_path="B", text="beta"),
        ),
    )
    mapping = build_index_mapping(source, "alpha\nbeta", index_snapshot_id="v1", paper_id="paper-a", chunk_id="one")
    assert tuple(item.block_id for item in mapping.ranges) == ("one", "two")
    assert tuple((item.payload_start, item.payload_end) for item in mapping.ranges) == ((0, 5), (6, 10))


def test_whitespace_mapping_keeps_original_offsets_and_detects_tampering() -> None:
    """无参数；验证归一后的字符仍指向原文位置，错误映射不会通过校验。"""
    source = _source("  alpha \n\t beta  ")
    normalized, characters = normalize_with_mapping(source.blocks[0].text)
    assert normalized == "alpha beta"
    assert characters[5] == (7, 11)
    mapping = build_index_mapping(
        source, normalized, index_snapshot_id="v1", paper_id="paper-a", chunk_id="one", normalization="whitespace-v1",
    )
    assert covered_span_ids((_span(source, "alpha \n\t beta"),), mapping.ranges) == {"target"}
    interval = mapping.ranges[0]
    invalid = interval.model_copy(update={"character_ranges": ((0, 1), *interval.character_ranges[1:])})
    with pytest.raises(EvidenceDataError, match="mapping_character_mapping_mismatch"):
        validate_mapping(mapping.model_copy(update={"ranges": (invalid,)}), {source.snapshot_id: source})


def test_rechunking_changes_index_version_but_preserves_span_truth() -> None:
    """无参数；验证新 chunk 身份建立新索引映射，原文真值和旧索引保持可解释。"""
    source = _source("alpha beta")
    first = _benchmark(source, text="alpha beta", chunk_id="old")
    second = _benchmark(source, text="alpha beta", chunk_id="new")
    old_index, new_index = first.index_snapshots[0], second.index_snapshots[0]
    assert old_index.index_snapshot_id != new_index.index_snapshot_id
    span = _span(source, "beta")
    assert covered_span_ids((span,), old_index.mappings[0].ranges) == {"target"}
    assert covered_span_ids((span,), new_index.mappings[0].ranges) == {"target"}
    tampered = old_index.model_dump(mode="json")
    tampered["units"][0]["text_snapshot"] = "changed"
    tampered["units"][0]["text_sha256"] = snapshot_hash("changed")
    with pytest.raises(ValidationError, match="index_snapshot_checksum_mismatch"):
        IndexSnapshot.model_validate(tampered)


def test_same_child_id_with_different_text_is_invalid() -> None:
    """无参数；验证正文变化不能通过同 chunk ID 获得原文命中。"""
    source = _source("alpha beta")
    benchmark = _benchmark(source, text="alpha beta")
    item = ObservedItem(paper_id="paper-a", chunk_id="chunk-a", payload_kind="child", text_snapshot="beta")
    with pytest.raises(EvidenceDataError, match="observed_payload_identity_or_text_mismatch"):
        align_observed_item(item, benchmark, index_snapshot_id=benchmark.index_snapshots[0].index_snapshot_id)


def test_model_excerpt_excludes_truncated_tail_and_normalizes_whitespace() -> None:
    """无参数；验证模型实际摘录仅覆盖被发送部分，归一坐标不冒充原文坐标。"""
    source = _source("alpha \n\t beta omitted")
    benchmark = _benchmark(source, text=source.blocks[0].text)
    item = ObservedItem(paper_id="paper-a", chunk_id="chunk-a", payload_kind="model_excerpt", text_snapshot="alpha beta")
    ranges = align_observed_item(item, benchmark, index_snapshot_id=benchmark.index_snapshots[0].index_snapshot_id)
    included = _span(source, "alpha \n\t beta", span_id="included")
    omitted = _span(source, "omitted", span_id="omitted")
    assert covered_span_ids((included, omitted), ranges) == {"included"}


def test_child_observation_cannot_borrow_expanded_parent_text() -> None:
    """无参数；验证父块实际存在仍不能给只返回 child 的阶段补证据。"""
    source = _source("child content and parent-only evidence")
    child_text = "child content"
    mappings = tuple(build_index_mapping(
        source, text, index_snapshot_id="pending", paper_id="paper-a", chunk_id="one", parent_id="parent", payload_kind=kind,
    ) for kind, text in (("child", child_text), ("parent", source.blocks[0].text)))
    index = make_index_snapshot(
        source_snapshot_ids=(source.snapshot_id,), retrieval_backend="fixture", mappings=mappings,
        units=(IndexUnit(paper_id="paper-a", chunk_id="one", parent_id="parent", text_snapshot=child_text, text_sha256=snapshot_hash(child_text)),),
    )
    benchmark = EvidenceBenchmark(dataset_version="fixture", review_status="reviewed", sources=(source,), spans=(), groups=(), index_snapshots=(index,))
    item = ObservedItem(paper_id="paper-a", chunk_id="one", parent_id="parent", payload_kind="child", text_snapshot=child_text)
    ranges = align_observed_item(item, benchmark, index_snapshot_id=index.index_snapshot_id)
    assert not covered_span_ids((_span(source, "parent-only evidence"),), ranges)


def test_unknown_proof_reference_invalidates_truth_without_shrinking_groups() -> None:
    """无参数；验证不存在的片段引用使真值无效，不删除该组缩小分母。"""
    source = _source("alpha")
    benchmark = _benchmark(source, text="alpha")
    group = EvidenceGroup(group_id="broken", importance="critical", proofs=(EvidenceProof(span_ids=("unknown",)),))
    with pytest.raises(EvidenceDataError, match="proof_span_reference_missing"):
        validate_benchmark(benchmark.model_copy(update={"groups": (group,)}))


def test_empty_and_duplicate_proofs_are_rejected() -> None:
    """无参数；验证空证明和重复 span 无法充当已经满足的证据，返回 None。"""
    with pytest.raises(ValidationError):
        EvidenceProof(span_ids=())
    with pytest.raises(ValidationError, match="duplicate proof span"):
        EvidenceProof(span_ids=("a", "a"))
    with pytest.raises(ValidationError):
        MappedRange(source_snapshot_id="0" * 64, block_id="a", source_start=0, source_end=1, payload_start=True, payload_end=1)
