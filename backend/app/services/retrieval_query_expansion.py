"""Query expansion rules for scoped paper retrieval."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

DEFAULT_LIGHTRAG_EXPANSION_MODE = "hybrid"
DEFAULT_LIGHTRAG_EXPANSION_TOP_K = 5
_MAX_LIGHTRAG_LOCATOR_QUERIES = 5
_MAX_LIGHTRAG_LOCATOR_CHARS = 160

SUMMARY_QUERY_HINTS = (
    "summarize",
    "summary",
    "overview",
    "one sentence",
    "what is this paper about",
    "what does this paper",
    "main contribution",
    "主要贡献",
    "一句话",
    "总结",
    "概述",
    "讲了什么",
    "这篇论文",
)
SUMMARY_EVIDENCE_TERMS = ("abstract", "introduction", "conclusion", "contribution")


@dataclass(frozen=True, kw_only=True)
class QueryExpansionResult:
    """Additional locator queries from non-citable expansion providers."""

    locator_queries: list[str] = field(default_factory=list)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def expand_single_paper_summary_queries(
    *,
    query: str,
    queries: list[str],
    structured_query: dict[str, Any] | None,
) -> list[str]:
    """Add narrow section-oriented queries for single-paper summary questions."""
    if not _is_single_paper_summary_query(query, structured_query):
        return _unique_queries(queries)
    expanded = list(queries)
    expanded.extend(f"{query} {term}" for term in SUMMARY_EVIDENCE_TERMS)
    expanded.extend(_paper_title_queries(structured_query))
    return _unique_queries(expanded)


async def expand_lightrag_locator_queries(
    *,
    query: str,
    queries: list[str],
    provider: Any,
    library_id: str | None = None,
    mode: str = DEFAULT_LIGHTRAG_EXPANSION_MODE,
    top_k: int = DEFAULT_LIGHTRAG_EXPANSION_TOP_K,
) -> QueryExpansionResult:
    """Use LightRAG text only to add locator queries; never evidence records."""
    if provider is None:
        raise ValueError("LightRAG query expansion requires a provider")
    context = await _call_lightrag(provider, query=query, library_id=library_id, mode=mode, top_k=top_k)
    added = _locator_queries_from_lightrag_context(context)
    return QueryExpansionResult(
        locator_queries=_unique_queries([*queries, *added]),
        evidence_records=[],
        metadata={"source": "lightrag", "context_chars": len(context)},
    )


async def _call_lightrag(provider: Any, *, query: str, library_id: str | None, mode: str, top_k: int) -> str:
    target = getattr(provider, "collect_library_graph_hint", None)
    if not callable(target):
        target = getattr(provider, "query", None)
    if not callable(target):
        raise ValueError("LightRAG query expansion provider has no query method")
    kwargs: dict[str, Any] = {"query": query, "mode": mode, "top_k": top_k}
    if library_id and _supports_library_id(target):
        kwargs["library_id"] = library_id
    result = target(**kwargs)
    return str(await result if inspect.isawaitable(result) else result)


def _supports_library_id(target: Any) -> bool:
    try:
        return "library_id" in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


def _locator_queries_from_lightrag_context(context: str) -> list[str]:
    lines = [_clean_lightrag_line(line) for line in str(context or "").splitlines()]
    candidates = [line for line in lines if line]
    return [line[:_MAX_LIGHTRAG_LOCATOR_CHARS] for line in candidates[:_MAX_LIGHTRAG_LOCATOR_QUERIES]]


def _clean_lightrag_line(line: str) -> str:
    return str(line or "").strip().lstrip("-*•0123456789.、) ").strip()


def _is_single_paper_summary_query(query: str, structured_query: dict[str, Any] | None) -> bool:
    normalized_query = str(query or "").strip().casefold()
    if not normalized_query or not any(hint in normalized_query for hint in SUMMARY_QUERY_HINTS):
        return False
    structured = structured_query if isinstance(structured_query, dict) else {}
    return _paper_focus_mode(structured) == "single_paper" or len(_scoped_paper_ids(structured)) == 1


def _paper_focus_mode(structured: dict[str, Any]) -> str:
    paper_focus = structured.get("paper_focus")
    if isinstance(paper_focus, dict):
        return str(paper_focus.get("mode") or "").strip().lower()
    return ""


def _scoped_paper_ids(structured: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for raw in (structured.get("paper_ids"), (structured.get("resolved_scope") or {}).get("paper_ids")):
        if isinstance(raw, list):
            candidates.extend(str(item).strip() for item in raw if str(item).strip())
    return _unique_queries(candidates)


def _paper_title_queries(structured_query: dict[str, Any] | None) -> list[str]:
    structured = structured_query if isinstance(structured_query, dict) else {}
    titles = structured.get("paper_titles")
    if not isinstance(titles, list):
        return []
    return [f"{title} abstract" for title in titles if str(title or "").strip()]


def _unique_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_query in queries:
        query = str(raw_query or "").strip()
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        result.append(query)
    return result
