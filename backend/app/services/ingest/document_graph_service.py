"""
上传文档图谱增强写入服务。

这是上传流水线的第四阶段，负责把论文送入 LightRAG 和 Neo4j 等图谱增强后端。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from app.config import get_settings
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.models import Paper
from app.services.graph import get_research_object_graph_service

settings = get_settings()
logger = logging.getLogger(__name__)
# LightRAG 在当前 CPU + 本地离线模型环境下处理 50k 级长文本容易触发 60s worker timeout。
# 这里仍保守控制输入体积，但不再按原文顺序生硬截断，而是优先拼高价值章节。
LIGHTRAG_INSERT_CHAR_LIMIT = 6000
_SECTION_PRIORITY_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\babstract\b", re.IGNORECASE), 120),
    (re.compile(r"\bkeywords?\b", re.IGNORECASE), 115),
    (re.compile(r"\bintroduction\b", re.IGNORECASE), 105),
    (re.compile(r"\b(?:method|approach|model|algorithm)\b", re.IGNORECASE), 110),
    (re.compile(r"\b(?:experiment|evaluation|result|results)\b", re.IGNORECASE), 100),
    (re.compile(r"\bdiscussion\b", re.IGNORECASE), 95),
    (re.compile(r"\bconclusion\b", re.IGNORECASE), 100),
]
_EXCLUDED_SECTION_PATTERN = re.compile(
    r"\b(?:references|bibliography|acknowledg(?:e)?ments?|copyright)\b",
    re.IGNORECASE,
)


def _metadata_preamble(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    analysis = metadata.get("analysis") if isinstance(metadata.get("analysis"), dict) else {}
    keywords = analysis.get("keywords") or metadata.get("keywords") or metadata.get("categories") or []
    keyword_text = ", ".join(str(item).strip() for item in keywords if str(item).strip())
    parts: list[str] = []
    title = str(metadata.get("title") or "").strip()
    abstract = str(metadata.get("abstract") or "").strip()
    if title:
        parts.append(f"# {title}")
    if abstract:
        parts.append(f"## Abstract\n{abstract}")
    if keyword_text:
        parts.append(f"## Keywords\n{keyword_text}")
    return "\n\n".join(parts).strip()


def _score_section(section: str) -> int:
    normalized = str(section or "").strip()
    if not normalized or normalized.lower() in {"untitled", "unknown"}:
        return 10
    for pattern, score in _SECTION_PRIORITY_RULES:
        if pattern.search(normalized):
            return score
    return 60


def _parent_records_from_chunks(chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not chunks:
        return records
    seen_parent_ids: set[str] = set()
    for chunk in chunks:
        parent_id = str(chunk.get("parent_id") or "").strip()
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        records.append(
            {
                "parent_id": parent_id,
                "section": str(chunk.get("section") or "untitled").strip() or "untitled",
                "page_num": int(chunk.get("page_num") or 0),
                "text": str(chunk.get("full_text") or "").strip(),
            }
        )
    return records


def build_graph_source_text(
    full_text: str,
    *,
    chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    limit: int = LIGHTRAG_INSERT_CHAR_LIMIT,
) -> tuple[str, dict[str, Any]]:
    """按章节优先级组装 LightRAG graph source text。"""
    original_chars = len(full_text or "")
    preamble = _metadata_preamble(metadata)
    records = _parent_records_from_chunks(chunks)
    if not records:
        fallback = (preamble + "\n\n" + (full_text or "").strip()).strip() if preamble else (full_text or "").strip()
        text = fallback[:limit].strip()
        return text, {
            "selected_sections": ["fallback_full_text"] if text else [],
            "excluded_sections": [],
            "original_chars": original_chars,
            "final_chars": len(text),
            "truncated": len(fallback) > len(text),
        }

    selected_sections: list[str] = []
    excluded_sections: list[str] = []
    ranked_records = sorted(
        enumerate(records),
        key=lambda item: (-_score_section(item[1]["section"]), item[1]["page_num"], item[0]),
    )
    parts: list[str] = [preamble] if preamble else []
    current_length = len(preamble)
    truncated = False
    for _, record in ranked_records:
        section = record["section"]
        if _EXCLUDED_SECTION_PATTERN.search(section):
            excluded_sections.append(section)
            continue
        body = record["text"].strip()
        if not body:
            continue
        block = body if section.lower() in {"untitled", "unknown"} else f"## {section}\n{body}"
        separator = "\n\n" if parts else ""
        candidate_length = current_length + len(separator) + len(block)
        if candidate_length <= limit:
            parts.append(block)
            current_length = candidate_length
            selected_sections.append(section)
            continue
        remaining = limit - current_length - len(separator)
        if remaining <= 0:
            truncated = True
            break
        clipped = block[:remaining].strip()
        if clipped:
            parts.append(clipped)
            selected_sections.append(section)
        truncated = True
        break
    text = "\n\n".join(part for part in parts if part.strip()).strip()
    return text, {
        "selected_sections": selected_sections,
        "excluded_sections": excluded_sections,
        "original_chars": original_chars,
        "final_chars": len(text),
        "truncated": truncated,
    }


def trim_graph_text(
    full_text: str,
    limit: int = LIGHTRAG_INSERT_CHAR_LIMIT,
    *,
    chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """兼容旧入口，返回新的 graph source text。"""
    text, _ = build_graph_source_text(full_text, chunks=chunks, metadata=metadata, limit=limit)
    return text


async def insert_to_lightrag(
    paper_id: str,
    full_text: str,
    *,
    library_id: str = DEFAULT_LIBRARY_ID,
    chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    把全文写入 LightRAG 图谱增强层。

    这里会优先拼接 title/abstract/high-value sections，再控制字符预算。
    """
    from app.services.lightrag_service import get_lightrag_service

    original_chars = len(full_text or "")
    text, audit = build_graph_source_text(full_text, chunks=chunks, metadata=metadata)
    if not text:
        raise ValueError("LightRAG 入库文本为空")
    logger.info(
        "LightRAG graph source 已组装: paper_id=%s library=%s original_chars=%d final_chars=%d truncated=%s selected_sections=%s excluded_sections=%s",
        paper_id or "-",
        library_id,
        original_chars,
        len(text),
        audit.get("truncated", False),
        audit.get("selected_sections", []),
        audit.get("excluded_sections", []),
    )
    await get_lightrag_service().insert(text, paper_id=paper_id, library_id=library_id)


def schedule_lightrag_insert(
    paper_id: str,
    full_text: str,
    *,
    library_id: str = DEFAULT_LIBRARY_ID,
    chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """在启用 GraphRAG 时创建 LightRAG 写入任务。"""
    if not settings.graph_rag_enabled:
        return None
    started_at = time.monotonic()
    logger.info(
        "LightRAG 后台建图已调度: paper_id=%s library=%s original_chars=%d char_limit=%d",
        paper_id or "-",
        library_id,
        len(full_text or ""),
        LIGHTRAG_INSERT_CHAR_LIMIT,
    )
    task = asyncio.create_task(
        insert_to_lightrag(
            paper_id,
            full_text,
            library_id=library_id,
            chunks=chunks,
            metadata=metadata,
        )
    )

    def _log_background_failure(future: asyncio.Task) -> None:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        try:
            future.result()
            logger.info(
                "LightRAG 后台建图完成: paper_id=%s library=%s duration_ms=%d",
                paper_id or "-",
                library_id,
                duration_ms,
            )
        except Exception as exc:  # pragma: no cover - 运行时日志分支
            logger.warning(
                "LightRAG 后台建图失败: paper_id=%s library=%s duration_ms=%d error=%s",
                paper_id or "-",
                library_id,
                duration_ms,
                exc,
            )

    task.add_done_callback(_log_background_failure)
    return task


async def build_research_object_graph_payload(paper: Paper, full_text: str):
    """构建 Neo4j 研究对象图谱 payload。"""
    return await get_research_object_graph_service().build_payload(paper, full_text)


async def insert_to_neo4j_graph(paper: Paper, full_text: str) -> None:
    """把研究对象子图写入 Neo4j。"""
    await get_research_object_graph_service().upsert_paper_subgraph(paper, full_text)


def schedule_research_object_graph_insert(paper: Paper, full_text: str):
    """在启用 Neo4j 时创建研究对象图谱写入任务。"""
    if not settings.neo4j_enabled:
        return None
    started_at = time.monotonic()
    logger.info(
        "Neo4j 研究对象建图已调度: paper_id=%s library=%s",
        paper.get("paper_id", "-"),
        paper.get("library_id", DEFAULT_LIBRARY_ID),
    )
    task = asyncio.create_task(insert_to_neo4j_graph(paper, full_text))

    def _log_background_failure(future: asyncio.Task) -> None:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        try:
            future.result()
            logger.info(
                "Neo4j 研究对象建图完成: paper_id=%s duration_ms=%d",
                paper.get("paper_id", "-"),
                duration_ms,
            )
        except Exception as exc:  # pragma: no cover - 运行时日志分支
            logger.warning(
                "Neo4j 研究对象建图失败: paper_id=%s duration_ms=%d error=%s",
                paper.get("paper_id", "-"),
                duration_ms,
                exc,
            )

    task.add_done_callback(_log_background_failure)
    return task
