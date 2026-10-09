"""Stable identity helpers for research evidence records."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence, cast

from app.models.knowledge_access_contracts import READ_CONTRACT_VERSION, SourceSpan

_DIGEST_CHARS = 16
_MAX_TEXT_KEY_CHARS = 160
_TOKEN_PATTERN = re.compile(r"\s+")
_READ_DIGEST_CHARS = 32
_LEGACY_COMMITMENT_FIELDS = ("paper_id", "chunk_id", "parent_id", "layer", "source", "excerpt", "expanded_text")
_READ_COMMITMENT_FIELDS = (*_LEGACY_COMMITMENT_FIELDS, "read_contract_version", "material_group_id",
                           "source_spans", "source_location_status")


def make_read_evidence_id(
    *, paper_id: str, material_group_id: str, source_spans: Sequence[SourceSpan], text: str,
) -> str:
    """参数为论文、材料组、真实跨度及正文；返回包含来源和内容承诺的稳定新版证据 ID。"""
    if not paper_id or not material_group_id or not source_spans or not text:
        raise ValueError("新版正文证据缺少论文、材料组、来源跨度或正文")
    payload = {"version": READ_CONTRACT_VERSION, "paper_id": paper_id, "material_group_id": material_group_id,
               "source_spans": source_spans, "text": text}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:_READ_DIGEST_CHARS]
    return f"ev-r{READ_CONTRACT_VERSION}-{digest}"


def validate_read_evidence(record: Mapping[str, Any]) -> None:
    """参数为新版或保存的历史证据；新版核对组键、跨度和内容承诺，历史 ID 保持原样。"""
    if "read_contract_version" not in record:
        return
    if record["read_contract_version"] != READ_CONTRACT_VERSION:
        raise ValueError("不支持的正文证据读取合同版本")
    paper_id = str(record.get("paper_id") or "")
    spans = record.get("source_spans")
    text = str(record.get("expanded_text") or record.get("excerpt") or "")
    if not isinstance(spans, list) or not spans:
        raise ValueError("新版正文证据缺少来源跨度")
    for span in spans:
        _validate_source_span(span, paper_id=paper_id, text=text)
    expected = make_read_evidence_id(
        paper_id=paper_id, material_group_id=str(record.get("material_group_id") or ""),
        source_spans=spans, text=text,
    )
    if record.get("evidence_id") != expected:
        raise ValueError("正文证据身份与保存的来源或内容承诺不一致")
    if record.get("source_location_status") not in {"exact", "legacy_chunk", "no_physical_location"}:
        raise ValueError("新版正文证据缺少明确来源精度")


def _validate_source_span(span: Mapping[str, Any], *, paper_id: str, text: str) -> None:
    """参数为跨度、论文和返回正文；校验可回读定位以及 Unicode 半开区间。"""
    if span.get("paper_id") != paper_id or not span.get("batch_id") or not span.get("chunk_id"):
        raise ValueError("正文证据来源跨度缺少一致的论文、批次或块定位")
    if span.get("field") not in {"text", "full_text"}:
        raise ValueError("正文证据来源跨度包含无效读取字段")
    positions = [span.get(key) for key in ("start", "end", "text_start", "text_end")]
    if any(type(value) is not int for value in positions):
        raise ValueError("正文证据来源跨度必须使用整数字符区间")
    start, end, text_start, text_end = cast(tuple[int, int, int, int], tuple(positions))
    if not (0 <= start < end and 0 <= text_start < text_end <= len(text)):
        raise ValueError("正文证据来源跨度超出实际读取正文")
    if end - start != text_end - text_start:
        raise ValueError("正文证据来源区间与输出区间长度不一致")


def count_material_groups(records: Iterable[Mapping[str, Any]]) -> int:
    """参数为可引用材料集合；新版按材料组计数，历史保存记录按原证据身份计数。"""
    groups: set[tuple[str, str]] = set()
    for record in records:
        validate_read_evidence(record)
        if "read_contract_version" in record:
            groups.add(("read", str(record["material_group_id"])))
        else:
            groups.add(("legacy", str(record.get("evidence_id") or "")))
    return len(groups)


def evidence_commitment_changes(existing: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
    """参数为相同 ID 的两条记录；返回发生变化的来源或内容承诺字段，排名与查询不参与。"""
    keys = _READ_COMMITMENT_FIELDS if "read_contract_version" in existing or "read_contract_version" in record else _LEGACY_COMMITMENT_FIELDS
    return [key for key in keys if existing.get(key) != record.get(key)]


def register_evidence_records(
    registry: Mapping[str, dict[str, Any]], records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """参数为已有登记与新记录；返回合并副本，同 ID 来源或正文冲突明确失败而非首次吞掉。"""
    merged = dict(registry)
    for record in records:
        validate_read_evidence(record)
        evidence_id = str(record.get("evidence_id") or "").strip()
        if not evidence_id:
            raise ValueError("证据登记缺少 evidence_id")
        if evidence_id in merged:
            changed = evidence_commitment_changes(merged[evidence_id], record)
            if changed:
                raise ValueError(f"duplicate_evidence_conflict:{evidence_id}:{','.join(changed)}")
        else:
            merged[evidence_id] = dict(record)
    return merged


def make_evidence_id(
    *,
    paper_id: str,
    parent_id: str = "",
    chunk_id: str = "",
    section: str = "",
    excerpt: str = "",
    namespace: str = "direct_text",
) -> str:
    """Return a deterministic evidence id for the same source passage."""
    anchor = _clean(parent_id) or _clean(chunk_id) or _text_anchor(section=section, excerpt=excerpt)
    payload = "|".join([_clean(namespace), _clean(paper_id), anchor])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return f"ev-{digest}"


def _text_anchor(*, section: str, excerpt: str) -> str:
    text = _clean(f"{section} {excerpt}")[:_MAX_TEXT_KEY_CHARS]
    return text or "unknown"


def _clean(value: str) -> str:
    return _TOKEN_PATTERN.sub(" ", str(value or "").strip().lower())
