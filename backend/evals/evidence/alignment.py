"""原文唯一定位、实际文本映射与完整区间覆盖。作者：xxx。"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Literal

from evals.contracts import snapshot_hash
from evals.evidence.contracts import (
    EvidenceBenchmark, EvidenceGroup, EvidenceSpan, IndexMapping, IndexSnapshot, IndexUnit,
    MappedRange, Normalization, ObservedItem, PayloadKind, RankingJudgments,
    RankingPolicy, SourceBlock, SourceSnapshot, ranking_unit_key, require_unique,
)


class EvidenceDataError(ValueError):
    """保存实际资料边界的缺失或无效原因，供评分器分项展示。"""

    def __init__(self, status: Literal["missing", "invalid_input"], reason: str):
        """参数为测量状态和原因；初始化可被评分边界识别的异常。"""
        super().__init__(reason)
        self.status = status
        self.reason = reason


def normalize_with_mapping(text: str, *, offset: int = 0) -> tuple[str, tuple[tuple[int, int], ...]]:
    """参数为原文和原文起点；返回折叠空白的文本及每个显示字符的原文范围。"""
    tokens = list(re.finditer(r"\S+", text))
    characters: list[str] = []
    ranges: list[tuple[int, int]] = []
    previous_end = 0
    for token in tokens:
        if characters:
            characters.append(" ")
            ranges.append((offset + previous_end, offset + token.start()))
        characters.extend(token.group())
        ranges.extend((offset + position, offset + position + 1) for position in range(token.start(), token.end()))
        previous_end = token.end()
    return "".join(characters), tuple(ranges)


def _match_offsets(text: str, quote: str) -> tuple[int, ...]:
    """参数为正文和非空摘录；返回全部匹配起点，保留重叠和重复位置。"""
    matches: list[int] = []
    position = text.find(quote)
    while position >= 0:
        matches.append(position)
        position = text.find(quote, position + 1)
    return tuple(matches)


def locate_span(
    source: SourceSnapshot, quote: str, *, block_ids: Iterable[str] | None = None,
    section_paths: Iterable[str] | None = None, page_nums: Iterable[int] | None = None,
) -> tuple[str, int, int]:
    """参数为原文、摘录及显式块/章/页限制；仅在位置唯一时返回块和字符区间。"""
    if not isinstance(quote, str) or not quote.strip():
        raise EvidenceDataError("invalid_input", "empty_evidence_quote")
    allowed_blocks = frozenset(block_ids) if block_ids is not None else None
    allowed_sections = frozenset(section_paths) if section_paths is not None else None
    allowed_pages = frozenset(page_nums) if page_nums is not None else None
    blocks = (
        block for block in source.blocks
        if (allowed_blocks is None or block.block_id in allowed_blocks)
        and (allowed_sections is None or block.section_path in allowed_sections)
        and (allowed_pages is None or block.page_num in allowed_pages)
    )
    matches = [(block.block_id, start, start + len(quote)) for block in blocks for start in _match_offsets(block.text, quote)]
    if len(matches) != 1:
        raise EvidenceDataError("invalid_input", f"span_match_count={len(matches)}")
    return matches[0]


def validate_span(span: EvidenceSpan, source: SourceSnapshot) -> None:
    """参数为显式片段和原文快照；验证版本、块、位置、正文及校验和。"""
    if span.source_snapshot_id != source.snapshot_id:
        raise EvidenceDataError("invalid_input", "source_version_mismatch")
    block = next((block for block in source.blocks if block.block_id == span.block_id), None)
    if block is None:
        raise EvidenceDataError("invalid_input", "source_block_missing")
    if not 0 <= span.start < span.end <= len(block.text):
        raise EvidenceDataError("invalid_input", "span_range_invalid")
    if block.text[span.start:span.end] != span.quote:
        raise EvidenceDataError("invalid_input", "span_text_mismatch")
    if snapshot_hash(span.quote) != span.quote_sha256:
        raise EvidenceDataError("invalid_input", "span_checksum_mismatch")


def _mapping_block(interval: MappedRange, sources: Mapping[str, SourceSnapshot], *, paper_id: str) -> SourceBlock:
    """参数为映射区间、原文索引和论文；返回对应块，缺产物和错误身份分别抛错。"""
    source = sources.get(interval.source_snapshot_id)
    if source is None:
        raise EvidenceDataError("missing", "mapping_source_snapshot_missing")
    if source.paper_id != paper_id:
        raise EvidenceDataError("invalid_input", "mapping_paper_mismatch")
    block = next((block for block in source.blocks if block.block_id == interval.block_id), None)
    if block is None:
        raise EvidenceDataError("invalid_input", "mapping_block_missing")
    return block


def _validate_interval(interval: MappedRange, block: SourceBlock, *, payload: str) -> None:
    """参数为映射区间、原文块及实际正文；逐字符验证精确或空白归一的对应关系。"""
    if not 0 <= interval.source_start < interval.source_end <= len(block.text):
        raise EvidenceDataError("invalid_input", "mapping_source_range_invalid")
    if not 0 <= interval.payload_start < interval.payload_end <= len(payload):
        raise EvidenceDataError("invalid_input", "mapping_payload_range_invalid")
    source_text = block.text[interval.source_start:interval.source_end]
    payload_text = payload[interval.payload_start:interval.payload_end]
    if interval.normalization == "exact-text-v1":
        if source_text != payload_text:
            raise EvidenceDataError("invalid_input", "mapping_text_mismatch")
        return
    parts = tuple(re.finditer(r"\s+|\S", source_text))
    normalized = "".join(" " if part.group().isspace() else part.group() for part in parts)
    characters = tuple((interval.source_start + part.start(), interval.source_start + part.end()) for part in parts)
    if normalized != payload_text or characters != interval.character_ranges:
        raise EvidenceDataError("invalid_input", "mapping_character_mapping_mismatch")


def _validate_payload_coverage(mapping: IndexMapping) -> None:
    """参数为已校验映射；非空白正文不得藏在未定位区间，也不允许冲突重叠。"""
    cursor = 0
    for interval in sorted(mapping.ranges, key=lambda item: item.payload_start):
        if interval.payload_start < cursor:
            raise EvidenceDataError("invalid_input", "mapping_payload_ranges_overlap")
        if mapping.text_snapshot[cursor:interval.payload_start].strip():
            raise EvidenceDataError("missing", "mapping_payload_not_fully_located")
        cursor = interval.payload_end
    if mapping.text_snapshot[cursor:].strip():
        raise EvidenceDataError("missing", "mapping_payload_not_fully_located")


def validate_mapping(mapping: IndexMapping, sources: Mapping[str, SourceSnapshot]) -> None:
    """参数为索引正文及可复核原文；验证每段内容、来源和完整映射，返回 None。"""
    if snapshot_hash(mapping.text_snapshot) != mapping.text_sha256:
        raise EvidenceDataError("invalid_input", "index_payload_checksum_mismatch")
    for interval in mapping.ranges:
        block = _mapping_block(interval, sources, paper_id=mapping.paper_id)
        _validate_interval(interval, block, payload=mapping.text_snapshot)
    _validate_payload_coverage(mapping)


def _locate_mapping_range(
    source: SourceSnapshot, text: str, *, normalization: Normalization,
    block_ids: frozenset[str] | None, payload_start: int,
) -> MappedRange:
    """参数为实际正文及定位模式；返回唯一映射区间，不以近似匹配消除歧义。"""
    matches: list[MappedRange] = []
    for block in source.blocks:
        if block_ids is not None and block.block_id not in block_ids:
            continue
        characters: tuple[tuple[int, int], ...] = ()
        haystack = block.text
        if normalization == "whitespace-v1":
            haystack, characters = normalize_with_mapping(block.text)
        for start in _match_offsets(haystack, text):
            end = start + len(text)
            selected = characters[start:end] if normalization == "whitespace-v1" else ()
            matches.append(MappedRange(
                source_snapshot_id=source.snapshot_id, block_id=block.block_id,
                source_start=selected[0][0] if selected else start,
                source_end=selected[-1][1] if selected else end,
                payload_start=payload_start, payload_end=payload_start + len(text),
                normalization=normalization, character_ranges=selected,
            ))
    if not matches:
        raise EvidenceDataError("missing", "index_text_not_located")
    if len(matches) != 1:
        raise EvidenceDataError("invalid_input", f"index_text_match_count={len(matches)}")
    return matches[0]


def build_index_mapping(
    source: SourceSnapshot, text: str, *, index_snapshot_id: str, paper_id: str,
    chunk_id: str, parent_id: str | None = None, payload_kind: PayloadKind = "child",
    normalization: Normalization = "exact-text-v1", block_ids: Iterable[str] | None = None,
) -> IndexMapping:
    """参数为实际索引/摘录正文及版本身份；返回可校验映射，无法定位的部分明确抛错。"""
    if source.paper_id != paper_id:
        raise EvidenceDataError("invalid_input", "mapping_paper_mismatch")
    allowed = frozenset(block_ids) if block_ids is not None else None
    if normalization == "whitespace-v1" and normalize_with_mapping(text)[0] != text:
        raise EvidenceDataError("invalid_input", "payload_not_whitespace_normalized")
    ranges: tuple[MappedRange, ...] = ()
    if text.strip():
        try:
            ranges = (_locate_mapping_range(source, text, normalization=normalization, block_ids=allowed, payload_start=0),)
        except EvidenceDataError as exc:
            if exc.status != "missing" or "\n" not in text:
                raise
            # 1. 跨原文块的正文仅按真实行边界拆开，每一行仍必须唯一定位
            lines = tuple(re.finditer(r"[^\r\n]+", text))
            ranges = tuple(_locate_mapping_range(
                source, line.group(), normalization=normalization,
                block_ids=allowed, payload_start=line.start(),
            ) for line in lines if line.group().strip())
    mapping = IndexMapping(
        index_snapshot_id=index_snapshot_id, paper_id=paper_id, chunk_id=chunk_id,
        parent_id=parent_id, payload_kind=payload_kind, text_snapshot=text,
        text_sha256=snapshot_hash(text), ranges=ranges,
    )
    validate_mapping(mapping, {source.snapshot_id: source})
    return mapping


def interval_covered(start: int, end: int, intervals: Iterable[tuple[int, int]]) -> bool:
    """参数为目标和已核实区间；仅在无缺口覆盖完整目标时返回 True。"""
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise EvidenceDataError("invalid_input", "coverage_target_range_invalid")
    ranges = tuple(intervals)
    if any(type(left) is not int or type(right) is not int or not 0 <= left < right for left, right in ranges):
        raise EvidenceDataError("invalid_input", "coverage_observed_range_invalid")
    cursor = start
    for left, right in sorted(ranges):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def covered_span_ids(spans: tuple[EvidenceSpan, ...], ranges: tuple[MappedRange, ...]) -> frozenset[str]:
    """参数为目标片段及已验证实际区间；返回完整覆盖的独立片段身份。"""
    by_block: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for item in ranges:
        by_block[(item.source_snapshot_id, item.block_id)].append((item.source_start, item.source_end))
    return frozenset(span.span_id for span in spans if interval_covered(
        span.start, span.end, by_block[(span.source_snapshot_id, span.block_id)],
    ))


def validate_groups(groups: tuple[EvidenceGroup, ...], spans: Mapping[str, EvidenceSpan]) -> None:
    """参数为目标证据组和片段索引；验证组唯一及每份证明的真实引用。"""
    require_unique(tuple(group.group_id for group in groups), label="evidence group")
    for group in groups:
        for proof in group.proofs:
            if any(span_id not in spans for span_id in proof.span_ids):
                raise EvidenceDataError("invalid_input", "proof_span_reference_missing")


def validate_index_snapshot(index: IndexSnapshot, sources: Mapping[str, SourceSnapshot]) -> None:
    """参数为索引与来源快照；核对实际 child 全集、原文关联及映射内容。"""
    if any(identity not in sources for identity in index.source_snapshot_ids):
        raise EvidenceDataError("missing", "index_source_snapshot_missing")
    allowed_papers = {sources[identity].paper_id for identity in index.source_snapshot_ids}
    units = {ranking_unit_key(unit.paper_id, unit.chunk_id): unit for unit in index.units}
    if any(unit.paper_id not in allowed_papers for unit in index.units):
        raise EvidenceDataError("invalid_input", "index_unit_paper_mismatch")
    for mapping in index.mappings:
        unit = units.get(ranking_unit_key(mapping.paper_id, mapping.chunk_id))
        if unit is None or unit.parent_id != mapping.parent_id:
            raise EvidenceDataError("invalid_input", "mapping_index_unit_mismatch")
        if any(item.source_snapshot_id not in index.source_snapshot_ids for item in mapping.ranges):
            raise EvidenceDataError("invalid_input", "mapping_source_version_mismatch")
        if mapping.payload_kind == "child" and mapping.text_snapshot != unit.text_snapshot:
            raise EvidenceDataError("invalid_input", "mapping_index_text_mismatch")
        validate_mapping(mapping, sources)


def validate_observed_child(item: ObservedItem, index: IndexSnapshot) -> IndexUnit:
    """参数为真实 child 和声明索引；校验身份及实际文本，返回对应排序单位。"""
    if item.paper_id is None or item.chunk_id is None or item.text_snapshot is None:
        raise EvidenceDataError("missing", "observed_item_identity_or_text_missing")
    if item.index_snapshot_id is not None and item.index_snapshot_id != index.index_snapshot_id:
        raise EvidenceDataError("invalid_input", "observed_index_version_mismatch")
    unit = next((unit for unit in index.units if (unit.paper_id, unit.chunk_id) == (item.paper_id, item.chunk_id)), None)
    if unit is None:
        raise EvidenceDataError("missing", "observed_child_index_unit_missing")
    if item.payload_kind != "child":
        raise EvidenceDataError("invalid_input", "ranking_unit_payload_mismatch")
    if unit.parent_id is not None and item.parent_id is None:
        raise EvidenceDataError("missing", "observed_parent_identity_missing")
    if unit.parent_id != item.parent_id or unit.text_snapshot != item.text_snapshot:
        raise EvidenceDataError("invalid_input", "observed_payload_identity_or_text_mismatch")
    return unit


def validate_ranking_judgments(
    judgments: RankingJudgments, index: IndexSnapshot, *, policy: RankingPolicy,
) -> None:
    """参数为排序标注、实际索引和固定策略；校验适用身份、全部单位和等级。"""
    if judgments.index_snapshot_id != index.index_snapshot_id:
        raise EvidenceDataError("invalid_input", "ranking_index_version_mismatch")
    if (judgments.stage, judgments.ranking_unit) != (policy.stage, policy.ranking_unit):
        raise EvidenceDataError("invalid_input", "ranking_policy_mismatch")
    universe = {ranking_unit_key(item.paper_id, item.chunk_id) for item in index.units}
    labeled = {item.unit_key for item in judgments.labels}
    if labeled - universe:
        raise EvidenceDataError("invalid_input", "ranking_judgment_unit_not_in_index")
    if universe - labeled:
        raise EvidenceDataError("missing", "ranking_judgments_incomplete")
    if any(item.relevance >= len(policy.gains) for item in judgments.labels):
        raise EvidenceDataError("invalid_input", "ranking_relevance_out_of_range")


def validate_paper_requirements(benchmark: EvidenceBenchmark) -> None:
    """参数为真值；验证每篇要求仅引用自己的原文，缺失要求保持 missing。"""
    requirements = benchmark.paper_requirements
    declared = {item.paper_id for item in requirements or ()}
    targets = set(benchmark.target_paper_ids)
    if declared - targets:
        raise EvidenceDataError("invalid_input", "cross_paper_requirement_not_targeted")
    if len(targets) > 1 and targets - declared:
        raise EvidenceDataError("missing", "cross_paper_requirements_missing")
    spans = {item.span_id: item for item in benchmark.spans}
    sources = {item.snapshot_id: item for item in benchmark.sources}
    for requirement in requirements or ():
        validate_groups(requirement.groups, spans)
        referenced = {span_id for group in requirement.groups for proof in group.proofs for span_id in proof.span_ids}
        if any(spans[span_id].source_snapshot_id not in sources for span_id in referenced):
            raise EvidenceDataError("missing", "cross_paper_source_snapshot_missing")
        if any(sources[spans[span_id].source_snapshot_id].paper_id != requirement.paper_id for span_id in referenced):
            raise EvidenceDataError("invalid_input", "cross_paper_proof_source_mismatch")


def validate_benchmark(benchmark: EvidenceBenchmark, *, include_ranking: bool = True) -> None:
    """参数为证据真值及是否校验独立排序标注；验证原文、证明及索引，不删除无效目标。"""
    sources = {item.snapshot_id: item for item in benchmark.sources}
    spans = {item.span_id: item for item in benchmark.spans}
    for span in benchmark.spans:
        if span.source_snapshot_id not in sources:
            raise EvidenceDataError("missing", "span_source_snapshot_missing")
        validate_span(span, sources[span.source_snapshot_id])
    validate_groups(benchmark.groups, spans)
    for index in benchmark.index_snapshots:
        validate_index_snapshot(index, sources)
    if not include_ranking:
        return
    validate_paper_requirements(benchmark)
    indexes = {item.index_snapshot_id: item for item in benchmark.index_snapshots}
    for judgments in benchmark.ranking_judgments:
        if benchmark.ranking_policy is None:
            raise EvidenceDataError("missing", "ranking_policy_missing")
        if judgments.index_snapshot_id not in indexes:
            raise EvidenceDataError("missing", "ranking_index_snapshot_missing")
        validate_ranking_judgments(judgments, indexes[judgments.index_snapshot_id], policy=benchmark.ranking_policy)


def _clip_mapping_range(interval: MappedRange, start: int, *, end: int) -> MappedRange | None:
    """参数为原映射和实际摘录边界；返回相交区间的新坐标，不包含被截断尾部。"""
    left, right = max(interval.payload_start, start), min(interval.payload_end, end)
    if left >= right:
        return None
    delta = left - interval.payload_start
    characters = interval.character_ranges[delta:delta + right - left]
    source_start = characters[0][0] if characters else interval.source_start + delta
    source_end = characters[-1][1] if characters else source_start + right - left
    return MappedRange(
        source_snapshot_id=interval.source_snapshot_id, block_id=interval.block_id,
        source_start=source_start, source_end=source_end,
        payload_start=left - start, payload_end=right - start,
        normalization=interval.normalization, character_ranges=characters,
    )


def _normalized_mapping(mapping: IndexMapping) -> IndexMapping:
    """参数为已核实完整映射；返回空白归一正文及复合后的逐字符原文坐标。"""
    text, payload_ranges = normalize_with_mapping(mapping.text_snapshot)
    normalized_ranges: list[MappedRange] = []
    for interval in mapping.ranges:
        positions = tuple((position, left, right) for position, (left, right) in enumerate(payload_ranges)
                          if interval.payload_start <= left and right <= interval.payload_end)
        if not positions:
            continue
        characters = tuple(_original_character_range(interval, left, end=right) for _, left, right in positions)
        normalized_ranges.append(MappedRange(
            source_snapshot_id=interval.source_snapshot_id, block_id=interval.block_id,
            source_start=characters[0][0], source_end=characters[-1][1],
            payload_start=positions[0][0], payload_end=positions[-1][0] + 1,
            normalization="whitespace-v1", character_ranges=characters,
        ))
    return IndexMapping(
        **mapping.model_dump(exclude={"text_snapshot", "text_sha256", "ranges"}),
        text_snapshot=text, text_sha256=snapshot_hash(text), ranges=tuple(normalized_ranges),
    )


def _original_character_range(interval: MappedRange, start: int, *, end: int) -> tuple[int, int]:
    """参数为一个已验证区间和显示字符范围；返回该字符对应的原文完整范围。"""
    left, right = start - interval.payload_start, end - interval.payload_start
    if interval.normalization == "exact-text-v1":
        return interval.source_start + left, interval.source_start + right
    return interval.character_ranges[left][0], interval.character_ranges[right - 1][1]


def _project_excerpt(mapping: IndexMapping, text: str, *, sources: Mapping[str, SourceSnapshot]) -> tuple[MappedRange, ...]:
    """参数为同版本已验证正文和模型实际摘录；唯一截取并校验原文坐标。"""
    exact_matches = _match_offsets(mapping.text_snapshot, text)
    base = mapping if exact_matches else _normalized_mapping(mapping)
    matches = exact_matches or _match_offsets(base.text_snapshot, text)
    if len(matches) != 1:
        status: Literal["missing", "invalid_input"] = "missing" if not matches else "invalid_input"
        raise EvidenceDataError(status, f"model_excerpt_match_count={len(matches)}")
    start = matches[0]
    ranges = tuple(clipped for interval in base.ranges
                   if (clipped := _clip_mapping_range(interval, start, end=start + len(text))) is not None)
    result = IndexMapping(
        **base.model_dump(exclude={"payload_kind", "text_snapshot", "text_sha256", "ranges"}),
        payload_kind="model_excerpt", text_snapshot=text, text_sha256=snapshot_hash(text), ranges=ranges,
    )
    validate_mapping(result, sources)
    return ranges


def _observed_mapping_candidates(
    item: ObservedItem, benchmark: EvidenceBenchmark, *, index_snapshot_id: str | None,
) -> tuple[IndexMapping, ...]:
    """参数为真实片段、真值和版本；返回身份一致的映射，不用旧版本或其他父块补齐。"""
    if item.paper_id is None or item.chunk_id is None:
        raise EvidenceDataError("missing", "observed_item_identity_or_text_missing")
    identity = item.index_snapshot_id or index_snapshot_id
    if identity is None:
        raise EvidenceDataError("missing", "observed_index_snapshot_missing")
    if item.index_snapshot_id is not None and index_snapshot_id is not None and item.index_snapshot_id != index_snapshot_id:
        raise EvidenceDataError("invalid_input", "observed_index_version_mismatch")
    index = next((index for index in benchmark.index_snapshots if index.index_snapshot_id == identity), None)
    if index is None:
        raise EvidenceDataError("missing", "observed_index_snapshot_unavailable")
    candidates = tuple(mapping for mapping in index.mappings if (mapping.paper_id, mapping.chunk_id, mapping.parent_id)
                       == (item.paper_id, item.chunk_id, item.parent_id))
    if not candidates:
        raise EvidenceDataError("missing", "observed_item_mapping_missing")
    return candidates


def align_observed_item(
    item: ObservedItem, benchmark: EvidenceBenchmark, *, index_snapshot_id: str | None,
) -> tuple[MappedRange, ...]:
    """参数为该阶段真实片段、真值及版本；只映射实际返回正文，身份不符时拒绝加分。"""
    if item.text_snapshot is None:
        raise EvidenceDataError("missing", "observed_item_identity_or_text_missing")
    candidates = _observed_mapping_candidates(item, benchmark, index_snapshot_id=index_snapshot_id)
    sources = {source.snapshot_id: source for source in benchmark.sources}
    for mapping in candidates:
        if mapping.payload_kind == item.payload_kind and mapping.text_snapshot == item.text_snapshot:
            validate_mapping(mapping, sources)
            return mapping.ranges
    if item.payload_kind != "model_excerpt":
        raise EvidenceDataError("invalid_input", "observed_payload_identity_or_text_mismatch")
    if not item.text_snapshot.strip():
        return ()
    projected: set[tuple[MappedRange, ...]] = set()
    for mapping in candidates:
        try:
            projected.add(_project_excerpt(mapping, item.text_snapshot, sources=sources))
        except EvidenceDataError as exc:
            if exc.status != "missing":
                raise
    if len(projected) != 1:
        status: Literal["missing", "invalid_input"] = "missing" if not projected else "invalid_input"
        raise EvidenceDataError(status, f"model_excerpt_mapping_count={len(projected)}")
    return projected.pop()
