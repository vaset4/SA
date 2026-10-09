"""论文分析材料整理服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)
INLINE_FORMULA_RE = re.compile(r"(\$[^$\n]+\$|\\\[[\s\S]*?\\\])")


@dataclass
class AnalysisSection:
    heading: str
    original_markdown: str
    formula_blocks: list[str] = field(default_factory=list)


@dataclass
class PaperAnalysisBundle:
    paper_id: str
    sections: dict[str, AnalysisSection]
    figures: list[dict[str, str]]
    references: list[str]
    source_markdown: str
    source_markdown_path: str


def _canonical_section_key(heading: str) -> str:
    lowered = heading.strip().lower()
    if "abstract" in lowered:
        return "abstract"
    if any(token in lowered for token in ("method", "approach", "model")):
        return "method"
    if any(token in lowered for token in ("experiment", "evaluation", "result")):
        return "experiment"
    if any(token in lowered for token in ("conclusion", "discussion")):
        return "conclusion"
    if any(token in lowered for token in ("reference", "bibliograph")):
        return "references"
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or "section"


def _extract_sections(source_markdown: str) -> dict[str, AnalysisSection]:
    matches = list(HEADING_RE.finditer(source_markdown))
    if not matches:
        formulas = INLINE_FORMULA_RE.findall(source_markdown)
        return {
            "full_text": AnalysisSection(
                heading="Full Text",
                original_markdown=source_markdown.strip(),
                formula_blocks=formulas,
            )
        }
    sections: dict[str, AnalysisSection] = {}
    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source_markdown)
        body = source_markdown[start:end].strip()
        key = _canonical_section_key(heading)
        formulas = INLINE_FORMULA_RE.findall(body)
        sections[key] = AnalysisSection(heading=heading, original_markdown=body, formula_blocks=formulas)
    return sections


def _extract_references(source_markdown: str, sections: dict[str, AnalysisSection]) -> list[str]:
    reference_section = sections.get("references")
    body = reference_section.original_markdown if reference_section else source_markdown
    references: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") or re.match(r"^\d+\.", stripped):
            references.append(stripped)
    return references


def _extract_figures(content_list: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    figures: list[dict[str, str]] = []
    for item in content_list or []:
        item_type = str(item.get("type") or item.get("category_type") or "").strip().lower()
        if item_type not in {"image", "figure"}:
            continue
        caption = str(item.get("caption") or item.get("text") or item.get("img_caption") or "").strip()
        path = str(item.get("img_path") or item.get("image_path") or item.get("path") or "").strip()
        page = str(item.get("page_idx") or item.get("page_num") or "").strip()
        figures.append({"caption": caption, "path": path, "page": page})
    return figures


def build_analysis_bundle(
    *,
    paper_id: str,
    source_markdown: str,
    content_list: list[dict[str, Any]] | None = None,
    source_markdown_path: str = "",
) -> PaperAnalysisBundle:
    cleaned_markdown = str(source_markdown or "").strip()
    sections = _extract_sections(cleaned_markdown)
    references = _extract_references(cleaned_markdown, sections)
    figures = _extract_figures(content_list)
    return PaperAnalysisBundle(
        paper_id=paper_id,
        sections=sections,
        figures=figures,
        references=references,
        source_markdown=cleaned_markdown,
        source_markdown_path=source_markdown_path,
    )
