"""LLM-based research object extraction for Neo4j graph writes.

This module sits next to the legacy regex extractor. Runtime settings choose
which path is used; failures here are explicit and never fall back silently.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.models.graph_models import ResearchObjectExtraction
from app.models.models import Paper
from app.services.graph.academic_schema import (
    ACADEMIC_DOC_EXTRACTION_PROMPT,
    ACADEMIC_ENTITY_TYPES,
    AcademicEntity,
)
from app.services.graph.graph_extraction_text import (
    GRAPH_EXTRACTION_CHAR_BUDGET,
    build_graph_extraction_text,
)
from app.services.graph.research_object_extractor import _clean_entities, _extract_claims
from app.services.graph.research_object_normalizer import dedupe_preserve_order, normalize_entity_name
from app.services.llm_service import get_llm

logger = logging.getLogger(__name__)
LlmExtractionAliasMap = dict[str, list[str]]

MAX_EVIDENCE_EXCERPT_CHARS = 200
MAX_ENTITY_NAME_CHARS = 120
MAX_ALIAS_ITEMS = 8
NOISE_ENTITY_NAMES = {
    "beijing",
    "tang",
    "tang et al",
    "neural network",
    "deep learning",
    "machine learning",
    "ai",
    "artificial intelligence",
}
GENERIC_ALIAS_NAMES = {
    "model",
    "method",
    "algorithm",
    "dataset",
    "benchmark",
    "metric",
    "task",
    "deep learning",
    "machine learning",
    "neural network",
}


class ExtractionFailure(RuntimeError):
    """LLM extraction failed and must be visible to the ingest graph step."""

    def __init__(self, paper_id: str, attempts: int, last_error: str) -> None:
        self.paper_id = paper_id
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"ExtractionFailure paper_id={paper_id} attempts={attempts}: {last_error}")


def _paper_id(paper: Paper) -> str:
    return str(paper.get("paper_id") or "-")


def _library_id(paper: Paper) -> str:
    return str(paper.get("library_id") or "default")


def _text(value: Any, *, limit: int = MAX_ENTITY_NAME_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:.-")
    return text[:limit].strip(" \t\r\n,;:.-")


async def extract_with_llm(
    paper: Paper,
    full_text: str,
    *,
    llm: Any = None,
    max_retries: int | None = None,
) -> tuple[ResearchObjectExtraction, LlmExtractionAliasMap]:
    settings = get_settings()
    retries = settings.graph_llm_max_retries if max_retries is None else max_retries
    source_text, audit = _prepare_source_text(paper, full_text)
    started_at = time.monotonic()
    response, entities, attempts = await _call_llm_with_retry(
        paper,
        source_text,
        llm=llm or get_llm(),
        max_retries=retries,
    )
    extraction, alias_map = _to_research_object_extraction(paper, entities)
    _log_success(paper, extraction, response, attempts, started_at, audit)
    return extraction, alias_map


def _prepare_source_text(paper: Paper, full_text: str) -> tuple[str, dict[str, Any]]:
    source_text, audit = build_graph_extraction_text(paper, full_text)
    if not source_text:
        raise ExtractionFailure(_paper_id(paper), 0, "graph extraction text is empty")
    if len(source_text) > GRAPH_EXTRACTION_CHAR_BUDGET:
        raise ExtractionFailure(
            _paper_id(paper),
            0,
            f"graph extraction text exceeds budget: {len(source_text)}>{GRAPH_EXTRACTION_CHAR_BUDGET}",
        )
    return source_text, audit


async def _call_llm_with_retry(
    paper: Paper,
    source_text: str,
    *,
    llm: Any,
    max_retries: int,
) -> tuple[Any, list[AcademicEntity], int]:
    last_error = ""
    total_attempts = max(0, int(max_retries)) + 1
    prompt = ACADEMIC_DOC_EXTRACTION_PROMPT.replace("{full_text}", source_text)
    for attempt in range(1, total_attempts + 1):
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = _load_json_payload(str(response.content))
            return response, _validate_entity_payload(raw), attempt
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "extract_with_llm retry: paper_id=%s attempt=%d/%d error=%s",
                _paper_id(paper),
                attempt,
                total_attempts,
                last_error,
            )
    raise ExtractionFailure(_paper_id(paper), total_attempts, last_error)


def _load_json_payload(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM extraction output must be a JSON object")
    return payload


def _validate_entity_payload(raw: dict[str, Any]) -> list[AcademicEntity]:
    raw_entities = raw.get("entities", [])
    if not isinstance(raw_entities, list):
        raise ValueError("entities must be a list")
    entities: list[AcademicEntity] = []
    seen_names: set[str] = set()
    for item in raw_entities:
        entity = _validate_entity(item)
        if entity is None:
            continue
        normalized = normalize_entity_name(entity["canonical_name"])
        if normalized in seen_names:
            continue
        seen_names.add(normalized)
        entities.append(entity)
    return entities


def _validate_entity(item: Any) -> AcademicEntity | None:
    if not isinstance(item, dict):
        raise ValueError("entity item must be an object")
    entity_type = item.get("type")
    if entity_type not in ACADEMIC_ENTITY_TYPES:
        return None
    canonical_name = _text(item.get("canonical_name"))
    if _is_noise_name(canonical_name):
        return None
    aliases = _clean_aliases(item.get("aliases"), canonical_name)
    evidence_excerpt = _text(item.get("evidence_excerpt"), limit=MAX_EVIDENCE_EXCERPT_CHARS)
    return {
        "canonical_name": canonical_name,
        "type": entity_type,
        "aliases": aliases,
        "evidence_excerpt": evidence_excerpt,
    }


def _is_noise_name(value: str) -> bool:
    normalized = normalize_entity_name(value)
    return not normalized or normalized in NOISE_ENTITY_NAMES


def _clean_aliases(raw_aliases: Any, canonical_name: str) -> list[str]:
    if not isinstance(raw_aliases, list):
        return []
    aliases = [
        _text(alias)
        for alias in raw_aliases
        if isinstance(alias, str) and _text(alias)
    ]
    normalized_canonical = normalize_entity_name(canonical_name)
    filtered = [
        alias for alias in aliases
        if normalize_entity_name(alias) not in {normalized_canonical, *GENERIC_ALIAS_NAMES}
    ]
    return dedupe_preserve_order(filtered)[:MAX_ALIAS_ITEMS]


def _to_research_object_extraction(
    paper: Paper,
    entities: list[AcademicEntity],
) -> tuple[ResearchObjectExtraction, LlmExtractionAliasMap]:
    grouped: dict[str, list[str]] = {entity_type: [] for entity_type in ACADEMIC_ENTITY_TYPES}
    alias_map: LlmExtractionAliasMap = {}
    for entity in entities:
        grouped[entity["type"]].append(entity["canonical_name"])
        alias_map[entity["canonical_name"]] = entity["aliases"]
    analysis = paper.get("analysis") or {}
    extraction: ResearchObjectExtraction = {
        "authors": _clean_entities(dedupe_preserve_order(paper.get("authors", [])), limit=80, max_items=10, drop_affiliation=True),
        "venue": _text(analysis.get("venue")),
        "methods": dedupe_preserve_order(grouped["Method"]),
        "datasets": dedupe_preserve_order(grouped["Dataset"]),
        "metrics": dedupe_preserve_order(grouped["Metric"]),
        "claims": _extract_claims(paper),
        "limitations": _clean_entities([analysis.get("limitation", "")], limit=180, max_items=2),
        "tasks": dedupe_preserve_order(grouped["Task"]),
    }
    return extraction, alias_map


def _log_success(
    paper: Paper,
    extraction: ResearchObjectExtraction,
    response: Any,
    attempts: int,
    started_at: float,
    audit: dict[str, Any],
) -> None:
    breakdown = {
        "method": len(extraction["methods"]),
        "dataset": len(extraction["datasets"]),
        "metric": len(extraction["metrics"]),
        "task": len(extraction["tasks"]),
    }
    usage = _extract_token_usage(response)
    logger.info(
        "extract_with_llm done: paper_id=%s library=%s entity_count=%d entity_breakdown=%s "
        "extraction_duration_ms=%d extraction_attempts=%d llm_token_in=%s llm_token_out=%s "
        "input_chars=%s selected_sections=%s dropped_sections=%s",
        _paper_id(paper),
        _library_id(paper),
        sum(breakdown.values()),
        breakdown,
        int((time.monotonic() - started_at) * 1000),
        attempts,
        usage["input"],
        usage["output"],
        audit.get("input_chars"),
        audit.get("selected_sections", []),
        audit.get("dropped_sections", []),
    )


def _extract_token_usage(response: Any) -> dict[str, int | str]:
    metadata = getattr(response, "response_metadata", None)
    raw = getattr(response, "raw", None)
    usage = {}
    if isinstance(metadata, dict):
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if not usage and isinstance(raw, dict):
        usage = raw.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    return {
        "input": usage.get("prompt_tokens") or usage.get("input_tokens") or "unknown",
        "output": usage.get("completion_tokens") or usage.get("output_tokens") or "unknown",
    }
