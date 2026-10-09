"""
从论文与分析结果中抽取第一版研究对象。

这是图谱构建链里最偏“信息抽取”的一步，但当前实现仍然相对保守：
- 优先利用 paper.analysis 里的结构化字段
- 再用少量正则从 full_text 中补充 datasets / metrics / tasks

它不是完整 NER/IE 系统，而是一个可控、可维护的第一版规则抽取器。
"""

from __future__ import annotations

import re

from app.models.graph_models import ResearchObjectExtraction
from app.models.models import Paper
from app.services.graph.research_object_normalizer import dedupe_preserve_order

_DATASET_PATTERNS = (
    re.compile(
        r"\b(ImageNet(?:-1K)?|CIFAR-10|CIFAR-100|MNIST|SQuAD|GLUE|MS COCO|METR-LA|PEMS-BAY|WMT14|LibriSpeech)\b",
        re.IGNORECASE,
    ),
)
_METRIC_PATTERNS = (
    re.compile(
        r"\b(accuracy|f1(?: score)?|precision|recall|bleu|rouge(?:-l)?|map|mrr|mae|rmse|mape|auc|auroc|r2|ndcg)\b",
        re.IGNORECASE,
    ),
)
_TASK_PATTERNS = (
    re.compile(r"\b(machine translation|text classification|question answering|summarization|information retrieval)\b", re.IGNORECASE),
)
_NOISE_TEXT_PATTERN = re.compile(r"^(?:n/?a|none|unknown|not explicitly stated)$", re.IGNORECASE)
_AFFILIATION_HINTS = (
    "university",
    "institute",
    "department",
    "faculty",
    "school",
    "college",
    "laboratory",
    "lab",
    "centre",
    "center",
)


def _match_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    """对给定文本跑一组正则，并保持去重后的原始出现顺序。"""
    values: list[str] = []
    for pattern in patterns:
        for match in pattern.findall(text or ""):
            if isinstance(match, tuple):
                values.extend([item for item in match if item])
            else:
                values.append(match)
    return dedupe_preserve_order(values)


def _clean_text(value: str | None, *, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:.-")
    if len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0].strip(" \t\r\n,;:.-")
    return text


def _looks_like_affiliation(text: str) -> bool:
    lowered = text.lower()
    return len(text.split()) >= 4 and any(hint in lowered for hint in _AFFILIATION_HINTS)


def _clean_entities(
    values: list[str],
    *,
    limit: int = 180,
    max_items: int = 8,
    drop_affiliation: bool = False,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _clean_text(raw, limit=limit)
        if not text or _NOISE_TEXT_PATTERN.fullmatch(text):
            continue
        if drop_affiliation and _looks_like_affiliation(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned[:max_items]


def _normalize_method(value: str) -> str:
    text = _clean_text(value, limit=140)
    if len(text) <= 80:
        return text
    for separator in (",", " which ", " that "):
        if separator in text:
            candidate = _clean_text(text.split(separator, 1)[0], limit=80)
            if candidate:
                return candidate
    return text


def _extract_tasks(paper: Paper, full_text: str) -> list[str]:
    """从 analysis 与全文中提取任务名称。"""
    analysis = paper.get("analysis") or {}
    candidates: list[str] = []
    candidates.extend(_match_patterns(analysis.get("problem", ""), _TASK_PATTERNS))
    candidates.extend(_match_patterns(" ".join(analysis.get("keywords", [])), _TASK_PATTERNS))
    candidates.extend(_match_patterns(full_text, _TASK_PATTERNS))
    return _clean_entities(dedupe_preserve_order(candidates), limit=80, max_items=6)


def _extract_claims(paper: Paper) -> list[str]:
    """从 analysis 中提取论文 claim / contribution。"""
    analysis = paper.get("analysis") or {}
    claims: list[str] = []
    claims.extend(analysis.get("contribution", []))
    tldr = analysis.get("tldr", "")
    if tldr:
        claims.append(tldr)
    return _clean_entities(dedupe_preserve_order(claims), limit=220, max_items=4)


def extract_research_objects(paper: Paper, full_text: str = "") -> ResearchObjectExtraction:
    """
    从 `paper` 与 `paper.analysis` 提取研究对象。

    返回值会直接喂给图谱写入服务，因此字段结构需要稳定、缺省可控。
    """
    analysis = paper.get("analysis") or {}
    methodology_detail = analysis.get("methodology_detail", "")
    extraction: ResearchObjectExtraction = {
        "authors": _clean_entities(dedupe_preserve_order(paper.get("authors", [])), limit=80, max_items=10, drop_affiliation=True),
        "venue": "",
        "methods": _clean_entities([_normalize_method(analysis.get("method", ""))], limit=80, max_items=4),
        "datasets": _clean_entities(dedupe_preserve_order(
            _match_patterns(methodology_detail, _DATASET_PATTERNS) + _match_patterns(full_text, _DATASET_PATTERNS)
        ), limit=80, max_items=8),
        "metrics": _clean_entities(dedupe_preserve_order(
            _match_patterns(methodology_detail, _METRIC_PATTERNS) + _match_patterns(full_text, _METRIC_PATTERNS)
        ), limit=60, max_items=8),
        "claims": _extract_claims(paper),
        "limitations": _clean_entities([analysis.get("limitation", "")], limit=180, max_items=2),
        "tasks": _extract_tasks(paper, full_text),
    }
    return extraction
