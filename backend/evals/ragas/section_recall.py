"""Section-level recall helpers for re-chunk-stable paper evidence labels."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_TOP_K = 5
BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ID_LABEL_RE = re.compile(
    r"^(?:chunk|parent|paper|evidence)[_-]|^upload_[0-9a-f]+$|^[A-Za-z0-9_-]+_\d+_\d+$",
    re.IGNORECASE,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        records.append(payload)
    return records


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def paper_ids_for_record(record: dict[str, Any]) -> list[str]:
    raw_ids = record.get("paper_ids")
    if isinstance(raw_ids, list):
        return [str(item).strip() for item in raw_ids if str(item).strip()]
    paper_id = str(record.get("paper_id") or "").strip()
    return [paper_id] if paper_id else []


def validate_plain_golden_evidence(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records, start=1):
        labels = record.get("golden_evidence")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"record {index} must define non-empty plain golden_evidence")
        for label in labels:
            _validate_label(label, index)


def evaluate_section_recall(
    *,
    records: list[dict[str, Any]],
    sections_by_paper: dict[str, list[dict[str, Any]]],
    chunks_by_paper: dict[str, list[dict[str, Any]]],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    validate_plain_golden_evidence(records)
    rows = [
        _evaluate_record(
            record=record,
            sections_by_paper=sections_by_paper,
            chunks=_candidate_chunks(record, chunks_by_paper),
            top_k=top_k,
        )
        for record in records
    ]
    recalls = [row["section_recall"] for row in rows]
    return {
        "sample_count": len(rows),
        "top_k": top_k,
        "section_recall": round(sum(recalls) / len(recalls), 6) if recalls else None,
        "retrieved_tokens": sum(int(row["retrieved_tokens"]) for row in rows),
        "records": rows,
    }


def _validate_label(label: Any, record_index: int) -> None:
    if not isinstance(label, str):
        raise ValueError(f"record {record_index} golden_evidence labels must be plain strings")
    text = label.strip()
    if not text:
        raise ValueError(f"record {record_index} golden_evidence labels must not be empty")
    if _ID_LABEL_RE.match(text):
        raise ValueError(f"record {record_index} golden_evidence must be text spans, got id-like {text!r}")


def _evaluate_record(
    *,
    record: dict[str, Any],
    sections_by_paper: dict[str, list[dict[str, Any]]],
    chunks: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    expected = _resolve_expected_sections(record, sections_by_paper)
    ranked = _rank_chunks(str(record.get("question") or ""), chunks, top_k)
    retrieved = {
        _section_key(str(chunk.get("paper_id") or ""), _chunk_section_path(chunk))
        for chunk in ranked
        if _chunk_section_path(chunk)
    }
    hits = expected.intersection(retrieved)
    return {
        "question": str(record.get("question") or ""),
        "paper_ids": paper_ids_for_record(record),
        "expected_sections": sorted(expected),
        "retrieved_sections": sorted(retrieved),
        "section_recall": round(len(hits) / len(expected), 6) if expected else 0.0,
        "retrieved_tokens": _retrieved_parent_tokens(ranked),
        "retrieved_chunks": [_chunk_summary(chunk) for chunk in ranked],
    }


def _resolve_expected_sections(
    record: dict[str, Any],
    sections_by_paper: dict[str, list[dict[str, Any]]],
) -> set[str]:
    paper_ids = paper_ids_for_record(record)
    labels = [str(item) for item in record.get("golden_evidence", [])]
    resolved: set[str] = set()
    for label in labels:
        matches = _find_label_sections(label, paper_ids, sections_by_paper)
        if len(matches) != 1:
            raise ValueError(f"golden_evidence span must resolve to one section: {label!r} -> {matches}")
        resolved.add(matches[0])
    return resolved


def _find_label_sections(
    label: str,
    paper_ids: list[str],
    sections_by_paper: dict[str, list[dict[str, Any]]],
) -> list[str]:
    needle = normalize_text(label)
    matches: list[str] = []
    for paper_id in paper_ids:
        for section in sections_by_paper.get(paper_id, []):
            if needle and needle in normalize_text(str(section.get("text") or "")):
                matches.append(_section_key(paper_id, _section_path(section)))
    return _dedupe(matches)


def _rank_chunks(query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    terms = _terms(query)
    if not terms:
        return chunks[:top_k]
    doc_terms = [_terms(_chunk_search_text(chunk)) for chunk in chunks]
    doc_freq = _document_frequency(doc_terms)
    avg_len = sum(len(items) for items in doc_terms) / max(len(doc_terms), 1)
    scored = [
        (_bm25_score(terms, terms_in_doc, doc_freq, len(doc_terms), avg_len), index, chunk)
        for index, (chunk, terms_in_doc) in enumerate(zip(chunks, doc_terms))
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for score, _index, chunk in scored if score > 0][:top_k]


def _bm25_score(
    query_terms: list[str],
    doc_terms: list[str],
    doc_freq: Counter[str],
    doc_count: int,
    avg_len: float,
) -> float:
    counts = Counter(doc_terms)
    score = 0.0
    for term in set(query_terms):
        freq = counts.get(term, 0)
        if not freq:
            continue
        idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * len(doc_terms) / max(avg_len, 1.0))
        score += idf * freq * (BM25_K1 + 1) / denom
    return score


def _candidate_chunks(record: dict[str, Any], chunks_by_paper: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {**chunk, "paper_id": paper_id}
        for paper_id in paper_ids_for_record(record)
        for chunk in chunks_by_paper.get(paper_id, [])
    ]


def _retrieved_parent_tokens(chunks: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    total = 0
    for chunk in chunks:
        parent_id = str(chunk.get("parent_id") or chunk.get("chunk_id") or "")
        if parent_id in seen:
            continue
        seen.add(parent_id)
        total += len(_terms(str(chunk.get("full_text") or chunk.get("text") or "")))
    return total


def _document_frequency(doc_terms: list[list[str]]) -> Counter[str]:
    return Counter(term for terms in doc_terms for term in set(terms))


def _terms(text: str) -> list[str]:
    return [item.casefold() for item in _TOKEN_RE.findall(str(text or ""))]


def _chunk_search_text(chunk: dict[str, Any]) -> str:
    return " ".join(str(chunk.get(key) or "") for key in ("embedding_text", "text", "section", "section_path"))


def _chunk_section_path(chunk: dict[str, Any]) -> str:
    return str(chunk.get("section_path") or chunk.get("section") or "").strip()


def _section_path(section: dict[str, Any]) -> str:
    return str(section.get("section_path") or section.get("title") or "").strip()


def _section_key(paper_id: str, section_path: str) -> str:
    return f"{paper_id}::{normalize_text(section_path)}"


def _chunk_summary(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": str(chunk.get("paper_id") or ""),
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "parent_id": str(chunk.get("parent_id") or ""),
        "section_path": _chunk_section_path(chunk),
        "block_type": str(chunk.get("block_type") or ""),
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
