"""
论文与笔记相关 facade。

这层把 paper 相关的多后端协作逻辑收敛起来：
paper_db、Milvus、LightRAG、Neo4j、project/library 都会在这里汇合。
"""

from __future__ import annotations

from typing import Any
import re

import app.services.paper_db as paper_db
from app.services import library_service, paper_metadata_service, task_service
from app.services.library_ingest import add_uploaded_paper
from app.services.milvus_store import get_milvus_store

YEAR_PREFIX_PATTERN = re.compile(r"^(19|20)\d{2}")
DEFAULT_NOTE_TYPE = "reading_note"
DEFAULT_NOTE_SOURCE = "manual"
NOTE_TYPE_ALIASES = {
    "reading_note": "reading_note",
    "reading note": "reading_note",
    "reading": "reading_note",
    "evidence_note": "evidence_note",
    "evidence note": "evidence_note",
    "evidence": "evidence_note",
    "comparison_note": "comparison_note",
    "comparison note": "comparison_note",
    "comparison": "comparison_note",
    "question_note": "question_note",
    "question note": "question_note",
    "question": "question_note",
    "idea_note": "idea_note",
    "idea note": "idea_note",
    "idea": "idea_note",
    "experiment_note": "experiment_note",
    "experiment note": "experiment_note",
    "experiment": "experiment_note",
    "research_note": "research_note",
    "research note": "research_note",
    "internal_research_note": "internal_research_note",
    "internal research note": "internal_research_note",
    "external_reference_note": "external_reference_note",
    "external reference note": "external_reference_note",
}
SOURCE_TYPE_ALIASES = {
    "manual": "manual",
    "local_rag": "local_rag",
    "local rag": "local_rag",
    "internal_research": "internal_research",
    "internal research": "internal_research",
    "deep_research": "deep_research",
    "deep research": "deep_research",
}

ALLOWED_UPLOAD_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/markdown",
    "text/html",
    "text/plain",
    "application/x-tex",
    "text/x-tex",
}


def validate_upload(filename: str, content_type: str | None) -> None:
    """校验上传文件的协议约束。"""
    if not filename.strip():
        raise ValueError("上传文件缺少文件名")
    if content_type and content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise ValueError(f"不支持的 MIME 类型: {content_type}")


def _normalize_library_id(library_id: str | None) -> str:
    """统一 library_id 入口。"""
    return library_service.normalize_library_id(library_id)


def _paper_year(paper: dict[str, Any]) -> str:
    published_date = str(paper.get("published_date") or "").strip()
    match = YEAR_PREFIX_PATTERN.match(published_date)
    return match.group(0) if match else ""


def _paper_category(paper: dict[str, Any]) -> str:
    categories = paper_metadata_service.normalize_categories(paper.get("categories"))
    return ", ".join(categories)


def _paper_keywords(paper: dict[str, Any]) -> list[str]:
    analysis = paper_metadata_service.normalize_analysis_payload(paper.get("analysis"))
    return paper_metadata_service.normalize_categories(analysis.get("keywords"))


def serialize_paper(paper: dict[str, Any] | None) -> dict[str, Any] | None:
    """补充前端展示所需的兼容视图字段。"""
    if not paper:
        return None
    normalized_analysis = paper_metadata_service.normalize_analysis_payload(paper.get("analysis"))
    return {
        **paper,
        "authors": paper_metadata_service.normalize_authors(paper.get("authors")),
        "categories": paper_metadata_service.normalize_categories(paper.get("categories")),
        "analysis": normalized_analysis,
        "analysis_status": str(paper.get("analysis_status") or ("analysis_ready" if normalized_analysis else "not_started")).strip(),
        "analysis_error": str(paper.get("analysis_error") or "").strip(),
        "analysis_updated_at": paper.get("analysis_updated_at"),
        "year": _paper_year(paper),
        "category": _paper_category(paper),
        "keywords": paper_metadata_service.normalize_categories(normalized_analysis.get("keywords")),
    }


def _paper_id_filter_expr(paper_ids: list[str]) -> str:
    escaped = [paper_id.replace("\\", "\\\\").replace('"', '\\"') for paper_id in paper_ids if paper_id]
    if not escaped:
        return ""
    return " or ".join(f'paper_id == "{paper_id}"' for paper_id in escaped)


def _call_with_optional_user_kw(func, *args, user_id: str = "anonymous", **kwargs):
    """兼容未声明 user_id 关键字参数的旧测试桩。"""
    try:
        return func(*args, user_id=user_id, **kwargs)
    except TypeError as exc:
        if "user_id" not in str(exc):
            raise
        return func(*args, **kwargs)


def _normalize_note_enum(value: str | None, mapping: dict[str, str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if not normalized:
        return default
    mapped = mapping.get(normalized) or mapping.get(normalized.replace("_", " "))
    return mapped or default


def _normalize_note_payload(payload: dict[str, Any]) -> dict[str, Any]:
    content = str(payload.get("content") or "").strip()
    if not content:
        raise ValueError("笔记内容不能为空")
    normalized = {
        "paper_id": str(payload.get("paper_id") or "").strip() or None,
        "project_id": str(payload.get("project_id") or "").strip() or None,
        "conversation_id": str(payload.get("conversation_id") or "").strip() or None,
        "session_id": str(payload.get("session_id") or "").strip() or None,
        "content": content,
        "note_type": _normalize_note_enum(payload.get("note_type"), NOTE_TYPE_ALIASES, DEFAULT_NOTE_TYPE),
        "source_type": _normalize_note_enum(payload.get("source_type"), SOURCE_TYPE_ALIASES, DEFAULT_NOTE_SOURCE),
        "citation": str(payload.get("citation") or "").strip(),
        "excerpt": str(payload.get("excerpt") or "").strip(),
        "evidence_reference": payload.get("evidence_reference")
        if isinstance(payload.get("evidence_reference"), dict)
        else {},
    }
    if not any(
        normalized[key]
        for key in ("paper_id", "project_id", "conversation_id", "session_id")
    ):
        raise ValueError("至少提供一个关联对象：paper_id / project_id / conversation_id / session_id")
    evidence_reference = normalized["evidence_reference"]
    evidence_paper_ids = evidence_reference.get("paper_ids") if isinstance(evidence_reference, dict) else None
    if normalized["note_type"] in {"research_note", "internal_research_note"} or normalized["source_type"] == "internal_research":
        if not normalized["project_id"] or not normalized["session_id"] or not isinstance(evidence_paper_ids, list) or not [item for item in evidence_paper_ids if item]:
            raise ValueError("正式研究笔记需要绑定 project、session 和 cited paper_ids")
    return normalized


async def ingest_uploaded_paper(
    filename: str,
    raw: bytes,
    library_id: str | None,
    *,
    project_id: str | None = None,
    user_id: str,
    request_id: str,
    graph_requested: bool = False,
) -> dict[str, Any]:
    """【上传】【兼容入口】将文件、账号和稳定意图交给同一持久任务，返回受理回执。"""
    return await add_uploaded_paper(filename, raw, library_id, project_id=project_id,
        user_id=user_id, request_id=request_id, graph_requested=graph_requested)


def list_papers_page(limit: int, offset: int, library_id: str | None = None, *, user_id: str = "anonymous") -> dict[str, Any]:
    """读取业务论文库分页结果。"""
    resolved_library_id = _normalize_library_id(library_id)
    papers = [
        serialize_paper(paper) or {}
        for paper in paper_db.list_papers(limit=limit, offset=offset, library_id=resolved_library_id, user_id=user_id)
    ]
    total = paper_db.count_papers(library_id=resolved_library_id, user_id=user_id)
    return {
        "papers": papers,
        "total": total,
        "limit": limit,
        "offset": offset,
        "library_id": resolved_library_id,
    }


def search_papers(
    query: str,
    top_k: int,
    library_id: str | None,
    year_min: int | None,
    year_max: int | None,
    category: str | None,
    authors: str | None,
    *,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """
    直接走 Milvus 检索论文。

    library 约束最终会被翻译成 paper_id 过滤表达式，而不是直接用 library_id 查 Milvus。
    """
    filters: list[str] = []
    resolved_library_id = _normalize_library_id(library_id)
    scoped_paper_ids = paper_db.list_paper_ids(resolved_library_id, user_id=user_id)
    if not scoped_paper_ids:
        return {"results": [], "filter_expr": "", "library_id": resolved_library_id}
    library_filter = _paper_id_filter_expr(scoped_paper_ids)
    if library_filter:
        filters.append(f"({library_filter})")
    if year_min is not None:
        filters.append(f"year >= {year_min}")
    if year_max is not None:
        filters.append(f"year <= {year_max}")
    if category:
        filters.append(f'category == "{category.replace(chr(34), "")}"')
    if authors:
        escaped = authors.replace("\\", "\\\\").replace('"', '\\"')
        filters.append(f'authors like "%{escaped}%"')
    filter_expr = " and ".join(filters)
    from app.services.upload_read_service import attach_publication_snapshot, release_publication_snapshot

    scope = paper_db._run(attach_publication_snapshot({"paper_ids": scoped_paper_ids, "user_id": user_id,
        "filter_expr": filter_expr, "local_retrieval_allowed": True}))
    try:
        if not scope["local_retrieval_allowed"]:
            raise ValueError("论文尚未完成核验，暂不可检索")
        results = get_milvus_store().search_scope(query=query, top_k=top_k, scope=scope)
        return {"results": results, "filter_expr": filter_expr, "library_id": resolved_library_id}
    finally:
        paper_db._run(release_publication_snapshot(scope))


def get_paper_or_none(paper_id: str, *, user_id: str = "anonymous"):
    """按主键读取论文详情。"""
    return serialize_paper(_call_with_optional_user_kw(paper_db.get_paper, paper_id, user_id=user_id))


def update_paper_metadata_or_raise(
    paper_id: str,
    payload: dict[str, Any],
    *,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """更新论文元数据。"""
    existing = _call_with_optional_user_kw(paper_db.get_paper, paper_id, user_id=user_id)
    if not existing:
        raise LookupError("论文不存在")
    patch = paper_metadata_service.build_metadata_update_patch(payload)
    if "analysis" in patch:
        patch["analysis"] = {
            **paper_metadata_service.normalize_analysis_payload(existing.get("analysis")),
            **patch["analysis"],
        }
    updated = _call_with_optional_user_kw(paper_db.update_paper_metadata, paper_id, patch, user_id=user_id)
    if not updated:
        raise LookupError("论文不存在")
    return serialize_paper(updated) or {}


async def autofill_paper_metadata_or_raise(
    paper_id: str,
    *,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """【上传】【元信息补全】旧接口也转交可恢复任务，完成后同步发布业务信息和索引。"""
    from app.services import embedded_worker_service, upload_service

    task = await upload_service.request_upload_operation(paper_id, "metadata", user_id=user_id)
    embedded_worker_service.ensure_embedded_worker_running(task_types=["upload_ingest"])
    return task


async def request_paper_deletion_or_raise(paper_id: str, *, user_id: str = "anonymous") -> dict[str, Any]:
    """请求论文删除并返回可轮询的后台任务。"""
    return await task_service.create_paper_delete_task(paper_id, user_id=user_id)


def create_note(payload: dict[str, Any], *, user_id: str = "anonymous") -> dict[str, Any]:
    """创建研究笔记。"""
    normalized = _normalize_note_payload(payload)
    return {
        "note_id": paper_db.add_note(
            normalized["paper_id"],
            normalized["content"],
            user_id=user_id,
            project_id=normalized["project_id"],
            conversation_id=normalized["conversation_id"],
            session_id=normalized["session_id"],
            note_type=normalized["note_type"],
            source_type=normalized["source_type"],
            citation=normalized["citation"],
            excerpt=normalized["excerpt"],
            evidence_reference=normalized["evidence_reference"],
        )
    }


def list_notes(
    *,
    user_id: str = "anonymous",
    paper_id: str | None = None,
    project_id: str | None = None,
    conversation_id: str | None = None,
    session_id: str | None = None,
    note_type: str | None = None,
    source_type: str | None = None,
    include_project_papers: bool = False,
    include_project_sessions: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    """查询研究笔记。"""
    return {
        "notes": paper_db.list_notes(
            user_id=user_id,
            paper_id=str(paper_id or "").strip() or None,
            project_id=str(project_id or "").strip() or None,
            conversation_id=str(conversation_id or "").strip() or None,
            session_id=str(session_id or "").strip() or None,
            note_type=_normalize_note_enum(note_type, NOTE_TYPE_ALIASES, DEFAULT_NOTE_TYPE) if note_type else None,
            source_type=_normalize_note_enum(source_type, SOURCE_TYPE_ALIASES, DEFAULT_NOTE_SOURCE) if source_type else None,
            include_project_papers=include_project_papers,
            include_project_sessions=include_project_sessions,
            limit=limit,
        )
    }


def add_note(paper_id: str, content: str, *, user_id: str = "anonymous") -> dict[str, Any]:
    """为指定论文写入笔记。"""
    return create_note({"paper_id": paper_id, "content": content}, user_id=user_id)


def get_notes(paper_id: str, *, user_id: str = "anonymous") -> dict[str, Any]:
    """读取论文关联笔记。"""
    return {"notes": paper_db.get_notes(paper_id, user_id=user_id)}
