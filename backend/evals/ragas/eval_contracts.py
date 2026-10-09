"""Deterministic paper-research evaluation contract helpers."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from evals.contracts import MetricResult, runtime_metadata

CONTRACT_FIELDS = ("golden_evidence", "critical_facts", "rubric", "evidence_benchmark")


def extract_contract_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """参数为旧/新样本；返回独立真值字段，读取历史 metadata 只用于迁移。"""
    raw_metadata = payload.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    contract: dict[str, Any] = {}
    for field in CONTRACT_FIELDS:
        value = payload.get(field, metadata.get(field))
        if field in payload or field in metadata:
            contract[field] = deepcopy(value)
    return contract


def with_contract_fields(record: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """参数为记录和独立真值；返回新记录，真值不再复制进运行 metadata。"""
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    result = {**deepcopy(record), **deepcopy(contract)}
    if "metadata" in record:
        result["metadata"] = runtime_metadata(metadata)
    return result


def compute_critical_fact_metrics(record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, float | None]:
    """参数为记录及旧元信息；从统一测量规则投影旧标量接口。"""
    measured = contract_measurements({**record, "metadata": metadata})
    names = ("critical_fact_coverage", "critical_fact_error_rate", "fact_contract_score")
    return {name: measured[name].value for name in names}


def compute_golden_evidence_presence(record: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    """参数为评分投影与实际元信息；返回仅由实际证据匹配得到的诊断命中率。"""
    return _golden_measurement(record, metadata).value


def _count_fact_matches(response: str, facts: list[dict[str, Any]], *, required: bool) -> tuple[int, int]:
    total = 0
    hits = 0
    keys = ("must_include", "required", "must_cover", "value", "values") if required else ("must_not_include", "must_not", "forbidden")
    for fact in facts:
        terms = _fact_terms(fact, keys)
        total += len(terms)
        hits += sum(1 for term in terms if _matches_term(response, term))
    return total, hits


def _fact_terms(fact: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    terms: list[Any] = []
    for key in keys:
        value = fact.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            terms.extend(value)
        else:
            terms.append(value)
    return terms


def _matches_term(text: str, term: Any) -> bool:
    if isinstance(term, list):
        return any(_matches_term(text, item) for item in term)
    if isinstance(term, dict):
        return _matches_dict_term(text, term)
    expected = str(term or "").strip()
    return bool(expected) and expected.casefold() in text.casefold()


def _matches_dict_term(text: str, term: dict[str, Any]) -> bool:
    if isinstance(term.get("any_of"), list):
        return any(_matches_term(text, item) for item in term["any_of"])
    if isinstance(term.get("aliases"), list):
        return any(_matches_term(text, item) for item in term["aliases"])
    if isinstance(term.get("all_of"), list):
        return all(_matches_term(text, item) for item in term["all_of"])
    pattern = str(term.get("regex") or term.get("pattern") or "").strip()
    return bool(pattern) and re.search(pattern, text, flags=re.IGNORECASE) is not None


def _fact_contract_score(coverage: float | None, error_rate: float | None) -> float | None:
    if coverage is None and error_rate is None:
        return None
    if coverage is None:
        return 1.0 - float(error_rate or 0.0)
    if error_rate is None:
        return coverage
    return (coverage + (1.0 - error_rate)) / 2


def _golden_evidence_ids(record: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    raw = record.get("golden_evidence", metadata.get("golden_evidence"))
    items = raw if isinstance(raw, list) else []
    ids: list[str] = []
    for item in items:
        ids.extend(_evidence_ids_from_item(item))
    return _dedupe(ids)


def _evidence_ids_from_item(item: Any) -> list[str]:
    if isinstance(item, dict):
        values = [item.get(key) for key in ("evidence_id", "chunk_id", "paper_id", "parent_id")]
        return [str(value).strip() for value in values if str(value or "").strip()]
    text = str(item or "").strip()
    return [text] if text else []


def _evidence_haystack(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    """参数为实际观察载体；只搜索检索文本和引用/Evidence 的受控字段。"""
    fields = ("evidence_id", "chunk_id", "paper_id", "parent_id", "excerpt", "text", "full_text", "expanded_text")
    carriers = [metadata.get("citations", []), record.get("evidence_records", []), metadata.get("evidence_records", [])]
    evidence: list[dict[str, Any]] = []
    for items in carriers:
        if not isinstance(items, list):
            continue
        evidence.extend({key: item[key] for key in fields if isinstance(item.get(key), (str, int))}
                        for item in items if isinstance(item, dict))
    contexts = record.get("retrieved_contexts", [])
    if not isinstance(contexts, list) or any(not isinstance(item, str) for item in contexts):
        raise ValueError("retrieved_contexts must contain only text strings")
    values = [str(value) for item in evidence for value in item.values()]
    return "\n".join([*contexts, *values]).casefold()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def contract_measurements(record: dict[str, Any]) -> dict[str, MetricResult]:
    """参数为独立评分投影；返回金标与事实合同的真实计数和测量状态。"""
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    result = {"golden_evidence_presence_rate": _golden_measurement(record, metadata)}
    names = ("critical_fact_coverage", "critical_fact_error_rate", "fact_contract_score")
    raw = record.get("critical_facts", metadata.get("critical_facts"))
    if raw is None:
        return {**result, **{name: _unmeasured(name, "missing", "critical_facts_missing") for name in names}}
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        return {**result, **{name: _unmeasured(name, "invalid_input", "critical_facts_invalid") for name in names}}
    if not raw:
        return {**result, **{name: _unmeasured(name, "not_measured", "no_fact_requirements") for name in names}}
    if "response" not in record:
        return {**result, **{name: _unmeasured(name, "missing", "response_missing") for name in names}}
    try:
        required, covered = _count_fact_matches(str(record["response"]), raw, required=True)
        forbidden, errors = _count_fact_matches(str(record["response"]), raw, required=False)
    except re.error as exc:
        return {**result, **{name: _unmeasured(name, "invalid_input", f"invalid_fact_pattern: {exc}") for name in names}}
    if not required and not forbidden:
        return {**result, **{name: _unmeasured(name, "invalid_input", "fact_has_no_recognized_terms") for name in names}}
    result[names[0]] = ratio_measurement(names[0], covered, required)
    result[names[1]] = ratio_measurement(names[1], errors, forbidden)
    score = _fact_contract_score(covered / required if required else None, errors / forbidden if forbidden else 0.0)
    result[names[2]] = MetricResult(name=names[2], status="measured", value=score, reason="legacy_composite_formula")
    return result


def _golden_measurement(record: dict[str, Any], metadata: dict[str, Any]) -> MetricResult:
    """参数为评分投影与元信息；区分真值缺失、无要求和真实未命中。"""
    name = "golden_evidence_presence_rate"
    raw = record.get("golden_evidence", metadata.get("golden_evidence"))
    if raw is None:
        return _unmeasured(name, "missing", "golden_evidence_missing")
    if not isinstance(raw, list):
        return _unmeasured(name, "invalid_input", "golden_evidence_invalid")
    if not raw:
        return _unmeasured(name, "not_measured", "no_evidence_requirements")
    if any(not isinstance(item, (str, dict)) or not _evidence_ids_from_item(item) for item in raw):
        return _unmeasured(name, "invalid_input", "golden_evidence_item_invalid")
    expected = _golden_evidence_ids(record, metadata)
    if not expected:
        return _unmeasured(name, "invalid_input", "golden_evidence_has_no_ids")
    if "retrieved_contexts" not in record and "evidence_records" not in record and "citations" not in metadata:
        return _unmeasured(name, "missing", "evidence_observation_missing")
    haystack = _evidence_haystack(record, metadata)
    hits = sum(item.casefold() in haystack for item in expected)
    return ratio_measurement(name, hits, len(expected))


def ratio_measurement(name: str, hits: int, total: int) -> MetricResult:
    """参数为指标名和实际分子/分母；无适用项时返回明确未测量状态。"""
    if not total:
        return _unmeasured(name, "not_measured", "no_applicable_items")
    return MetricResult(name=name, status="measured", value=hits / total, numerator=hits, denominator=total, reason="counted")


def _unmeasured(name: str, status: Any, reason: str) -> MetricResult:
    """参数为指标、状态和原因；返回无伪造数值的测量结果。"""
    return MetricResult(name=name, status=status, reason=reason)
