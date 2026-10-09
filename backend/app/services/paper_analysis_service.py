"""手动论文分析服务。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import get_settings
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.services import paper_db, project_service
from app.services.ingest import mineru_markdown_service
from app.services.llm_service import OpenAICompatibleLLM
from app.services.paper_analysis_material_service import build_analysis_bundle
from app.nodes.paper_analyzer import analyze_paper as analyze_single_paper

logger = logging.getLogger(__name__)


def _read_mineru_artifacts(paper_id: str) -> tuple[str, list[dict[str, Any]], str]:
    artifact_dir = mineru_markdown_service.artifact_dir_for(paper_id)
    source_path = artifact_dir / "source.md"
    content_list_path = artifact_dir / "content_list.json"
    markdown_text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    content_list: list[dict[str, Any]] = []
    if content_list_path.exists():
        content_list = mineru_markdown_service._load_content_list(content_list_path)  # type: ignore[attr-defined]
    return markdown_text, content_list, str(source_path) if source_path.exists() else ""


def _build_paper_analysis_llm() -> OpenAICompatibleLLM:
    settings = get_settings()
    timeout = min(int(settings.llm_timeout), int(settings.paper_analysis_llm_timeout_seconds))
    max_retries = int(settings.paper_analysis_llm_max_retries)
    return OpenAICompatibleLLM(
        model=settings.llm_model_name,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=timeout,
        min_request_interval_seconds=settings.llm_min_request_interval_seconds,
        request_exception_max_retries=max_retries,
        empty_response_max_retries=max_retries,
        requests_per_minute_limit=settings.llm_requests_per_minute_limit,
    )


def _mark_analysis_failed(paper_id: str, error: Exception, *, user_id: str) -> None:
    message = str(error).strip() or error.__class__.__name__
    paper_db.update_analysis_status(
        paper_id,
        "failed",
        error=message,
        clear_analysis=True,
        user_id=user_id,
    )
    logger.error(
        "paper analysis failed: paper_id=%s user_id=%s error=%s",
        paper_id,
        user_id,
        message,
        exc_info=True,
    )


async def analyze_paper_by_id(paper_id: str, *, user_id: str = "anonymous") -> dict[str, Any]:
    paper = await asyncio.to_thread(paper_db.get_paper, paper_id, user_id=user_id)
    if not paper:
        raise LookupError(f"论文不存在: {paper_id}")

    await asyncio.to_thread(
        paper_db.update_analysis_status,
        paper_id,
        "running",
        clear_analysis=True,
        user_id=user_id,
    )
    markdown_text = ""
    content_list: list[dict[str, Any]] = []
    source_markdown_path = ""
    if paper.get("source") == "upload":
        markdown_text, content_list, source_markdown_path = _read_mineru_artifacts(paper_id)
    if not markdown_text:
        markdown_text = str(paper.get("abstract") or "").strip()

    bundle = build_analysis_bundle(
        paper_id=paper_id,
        source_markdown=markdown_text,
        content_list=content_list,
        source_markdown_path=source_markdown_path,
    )
    llm_client = _build_paper_analysis_llm()
    try:
        analyzed_paper = await analyze_single_paper(
            {**paper, "analysis": None},
            full_text=bundle.source_markdown or paper.get("abstract") or "",
            force_analysis=True,
            llm_client=llm_client,
        )
    except Exception as exc:
        await asyncio.to_thread(_mark_analysis_failed, paper_id, exc, user_id=user_id)
        raise
    analysis = dict(analyzed_paper.get("analysis") or {})
    analysis["references"] = bundle.references
    analysis["figures"] = bundle.figures
    analysis["source_markdown_path"] = bundle.source_markdown_path
    analysis["section_summaries"] = {
        key: {
            "heading": section.heading,
            "formula_blocks": section.formula_blocks,
        }
        for key, section in bundle.sections.items()
    }
    await asyncio.to_thread(paper_db.update_analysis, paper_id, analysis, user_id=user_id)
    await asyncio.to_thread(
        paper_db.update_analysis_status,
        paper_id,
        "analysis_ready",
        user_id=user_id,
    )
    updated = await asyncio.to_thread(paper_db.get_paper, paper_id, user_id=user_id) or {
        "paper_id": paper_id,
        "library_id": DEFAULT_LIBRARY_ID,
        "analysis": analysis,
        "analysis_status": "analysis_ready",
    }
    await project_service.record_paper_analysis_completed(updated, user_id=user_id)
    return updated
