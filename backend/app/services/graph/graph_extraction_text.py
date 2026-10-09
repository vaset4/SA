"""Build bounded high-signal text for doc-level graph extraction."""

from __future__ import annotations

import re
from typing import Any

from app.models.models import Paper

GRAPH_EXTRACTION_CHAR_BUDGET = 80000
SECTION_PATTERN = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?([A-Za-z][A-Za-z0-9 .:/_-]{2,80})\s*$", re.MULTILINE)
DROP_SECTION_PATTERN = re.compile(r"\b(?:references|bibliography|acknowledg(?:e)?ments?|affiliations?)\b", re.IGNORECASE)
KEEP_SECTION_PATTERN = re.compile(
    r"\b(?:abstract|introduction|method|approach|model|algorithm|experiment|evaluation|result|discussion|conclusion)\b",
    re.IGNORECASE,
)


def build_graph_extraction_text(paper: Paper, full_text: str) -> tuple[str, dict[str, Any]]:
    """Construct bounded document text for LLM graph extraction."""
    analysis = paper.get("analysis") or {}
    parts = [
        f"# {paper.get('title', '')}".strip(),
        str(analysis.get("abstract") or paper.get("abstract") or "").strip(),
        select_high_signal_sections(full_text),
    ]
    text = "\n\n".join(part for part in parts if part).strip()
    return text, {
        "input_chars": len(text),
        "selected_sections": ["title", "abstract", "high_signal_text"],
        "dropped_sections": ["references", "acknowledgements", "affiliations"],
        "budget": GRAPH_EXTRACTION_CHAR_BUDGET,
    }


def select_high_signal_sections(full_text: str) -> str:
    """Prefer method/evaluation/conclusion sections and drop references."""
    text = str(full_text or "").strip()
    if not text:
        return ""
    sections = split_sections(text)
    if not sections:
        return drop_late_references(text)
    selected = [
        body for heading, body in sections
        if KEEP_SECTION_PATTERN.search(heading) and not DROP_SECTION_PATTERN.search(heading)
    ]
    if selected:
        return "\n\n".join(selected).strip()
    return drop_late_references(text)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into heading/body pairs."""
    matches = list(SECTION_PATTERN.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def drop_late_references(text: str) -> str:
    """Remove trailing references/bibliography from raw text."""
    match = re.search(r"\n\s*(?:references|bibliography)\s*\n", text, re.IGNORECASE)
    return text[: match.start()].strip() if match else text
