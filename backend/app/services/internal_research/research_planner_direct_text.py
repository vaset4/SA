"""Planner-owned direct-text retrieval preferences."""

from __future__ import annotations

import re
from typing import Any


_METHOD_SECTION_HINTS = ("method", "approach", "architecture", "algorithm", "implementation")
_RESULT_SECTION_HINTS = ("result", "evaluation", "experiment", "benchmark")
_ABLATION_SECTION_HINTS = ("ablation", "comparison", "analysis", "result")
_LIMITATION_SECTION_HINTS = ("limitation", "discussion", "failure", "threat")
_DEFINITION_SECTION_HINTS = ("definition", "preliminar", "theory")
_METHOD_CONTENT_MARKERS = (
    "method",
    "mechanism",
    "architecture",
    "algorithm",
    "implementation",
    "framework",
    "model",
    "module",
    "方法",
    "机制",
    "架构",
    "算法",
    "框架",
    "模型",
    "模块",
)
_ARCHITECTURE_POSITIVE_TERMS = (
    "architecture",
    "framework",
    "module",
    "component",
    "layer",
    "encoder",
    "decoder",
    "embedding",
    "attention",
    "transformer",
    "graph",
    "spatial",
    "temporal",
    "spatiotemporal",
    "data flow",
    "pipeline",
    "架构",
    "框架",
    "模块",
    "组件",
    "编码器",
    "解码器",
    "嵌入",
    "注意力",
    "图",
    "空间",
    "时间",
    "时空",
    "数据流",
)
_ARCHITECTURE_NEGATIVE_TERMS = (
    "dataset",
    "data set",
    "train validation test",
    "6 2 2",
    "sliding window",
    "window size",
    "loss",
    "pretrain loss",
    "metric",
    "baseline",
    "experiment",
    "evaluation",
    "table",
    "数据集",
    "训练 验证 测试",
    "滑动窗口",
    "损失",
    "预训练损失",
    "指标",
    "基线",
    "实验",
    "评估",
)

_DIMENSION_SECTION_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("method", "mechanism", "architecture", "algorithm", "implementation", "方法", "机制", "架构", "算法"),
        _METHOD_SECTION_HINTS,
    ),
    (("result", "metric", "evaluation", "experiment", "benchmark"), _RESULT_SECTION_HINTS),
    (("ablation", "comparison"), _ABLATION_SECTION_HINTS),
    (("limitation", "failure", "threat"), _LIMITATION_SECTION_HINTS),
    (("definition", "theory", "equation"), _DEFINITION_SECTION_HINTS),
)
_DIMENSION_BLOCK_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("result", "metric", "evaluation", "experiment", "benchmark"), ("table",)),
    (("ablation", "comparison"), ("table", "paragraph")),
    (("definition", "theory", "equation"), ("formula", "paragraph")),
)


def rank_direct_text_hits(hits: list[dict[str, Any]], policy: Any) -> list[dict[str, Any]]:
    """Rank scoped Milvus hits using planner structural preferences."""
    target_ids = {str(item).strip() for item in _policy_values(policy, "target_paper_ids") if str(item).strip()}
    required_sections = _normalized_terms(_policy_values(policy, "required_sections"))
    preferred_blocks = {_normalize_block_type(item) for item in _policy_values(policy, "preferred_block_types")}
    dimensions = _policy_values(policy, "evidence_dimensions")
    dimension_sections, dimension_blocks = _dimension_hints(dimensions)
    content_profile = _content_profile(required_sections, dimensions)
    ranked = [
        (
            _hit_rank(
                hit,
                index=index,
                target_ids=target_ids,
                required_sections=required_sections,
                preferred_blocks=preferred_blocks,
                dimension_sections=dimension_sections,
                dimension_blocks=dimension_blocks,
                content_profile=content_profile,
            ),
            hit,
        )
        for index, hit in enumerate(hits)
    ]
    return [hit for _, hit in sorted(ranked, key=lambda item: item[0], reverse=True)]


def _hit_rank(
    hit: dict[str, Any],
    *,
    index: int,
    target_ids: set[str],
    required_sections: tuple[str, ...],
    preferred_blocks: set[str],
    dimension_sections: tuple[str, ...],
    dimension_blocks: set[str],
    content_profile: str,
) -> tuple[int, int, int, int, int, int]:
    block_type = _normalize_block_type(str(hit.get("block_type") or ""))
    target_match = int(not target_ids or str(hit.get("paper_id") or "").strip() in target_ids)
    required_match = int(_matches_section(hit, required_sections))
    preferred_block_match = int(bool(block_type) and block_type in preferred_blocks)
    dimension_match = int(_matches_section(hit, dimension_sections) or block_type in dimension_blocks)
    content_score = _content_score(hit, content_profile)
    return (target_match, required_match, content_score, preferred_block_match, dimension_match, -index)


def _matches_section(hit: dict[str, Any], terms: tuple[str, ...]) -> bool:
    if not terms:
        return False
    candidates = _section_candidates(str(hit.get("section_path") or ""))
    if not candidates:
        candidates = _section_candidates(str(hit.get("section") or ""))
    return any(_section_term_matches(candidate, term) for candidate in candidates for term in terms)


def _section_candidates(value: str) -> tuple[str, ...]:
    parts = re.split(r"\s*/\s*|\s+\+\s+", value)
    return tuple(_normalize_text(part) for part in parts if _normalize_text(part))


def _section_term_matches(candidate: str, term: str) -> bool:
    return candidate == term or term in candidate or candidate in term


def _dimension_hints(dimensions: list[str]) -> tuple[tuple[str, ...], set[str]]:
    normalized = _normalized_terms(dimensions)
    section_hints = set()
    block_hints: set[str] = set()
    for dimension in normalized:
        section_hints.update(_mapped_hints(dimension, _DIMENSION_SECTION_HINTS))
        block_hints.update(_mapped_hints(dimension, _DIMENSION_BLOCK_HINTS))
    return tuple(sorted(section_hints)), block_hints


def _content_profile(required_sections: tuple[str, ...], dimensions: list[str]) -> str:
    terms = [*required_sections, *_normalized_terms(dimensions)]
    if any(any(marker in term for marker in _METHOD_CONTENT_MARKERS) for term in terms):
        return "architecture_method"
    return ""


def _content_score(hit: dict[str, Any], profile: str) -> int:
    if profile != "architecture_method":
        return 0
    text = _hit_text(hit)
    positive = sum(1 for term in _ARCHITECTURE_POSITIVE_TERMS if term in text)
    negative = sum(1 for term in _ARCHITECTURE_NEGATIVE_TERMS if term in text)
    return positive * 2 - negative * 3


def _hit_text(hit: dict[str, Any]) -> str:
    fields = (
        "section_path",
        "section",
        "title",
        "heading",
        "preview",
        "text",
        "excerpt",
        "expanded_text",
        "summary",
    )
    return _normalize_text(" ".join(str(hit.get(field) or "") for field in fields))


def _mapped_hints(dimension: str, table: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]) -> tuple[str, ...]:
    for markers, hints in table:
        if any(marker in dimension for marker in markers):
            return hints
    return ()


def _normalized_terms(values: list[str]) -> tuple[str, ...]:
    return tuple(term for term in (_normalize_text(str(value)) for value in values) if term)


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[_\-\W]+", " ", value.casefold())
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_block_type(value: str) -> str:
    return _normalize_text(value).replace(" ", "_")


def _policy_values(policy: Any, field: str) -> list[str]:
    raw = policy.get(field) if isinstance(policy, dict) else getattr(policy, field, [])
    return list(raw or [])
