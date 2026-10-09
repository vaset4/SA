"""研究对象归一化。"""

from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")
_TRIM_PUNCT = " \t\r\n,;:()[]{}<>\"'`“”‘’。，；："


def normalize_text(value: str) -> str:
    """统一文本清洗规则。"""
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = _SPACE_RE.sub(" ", normalized.strip().lower())
    return normalized.strip(_TRIM_PUNCT)


def normalize_entity_name(value: str) -> str:
    """实体名归一化入口。"""
    return normalize_text(value)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """保持顺序去重，并过滤空值。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_entity_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value.strip())
    return result


def build_entity_key(library_id: str, label: str, name: str) -> str:
    """生成稳定的实体键。"""
    normalized = normalize_entity_name(name)
    if not normalized:
        raise ValueError(f"{label} name is empty after normalization")
    return f"{library_id}:{label}:{normalized}"
