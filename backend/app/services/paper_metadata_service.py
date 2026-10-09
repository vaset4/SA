"""论文元数据编辑与 AI 补全服务。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models.ingest_models import IngestedDocumentMetadata
from app.services.ingest import document_parse_service

AUTHOR_SPLIT_PATTERN = re.compile(r"\s*(?:;|\n|\band\b)\s*", re.IGNORECASE)
CATEGORY_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|，|、|\|)\s*")
CONTRIBUTION_SPLIT_PATTERN = re.compile(r"\s*(?:\n+|;|；|•|\u2022)\s*")
YEAR_ONLY_PATTERN = re.compile(r"^(19|20)\d{2}$")
DATE_PATTERN = re.compile(r"^(19|20)\d{2}(?:[-/](0?[1-9]|1[0-2]))?(?:[-/](0?[1-9]|[12]\d|3[01]))?$")
METHOD_TYPES = {"experimental", "theoretical", "survey", "hybrid"}
AFFILIATION_LIKE_PATTERN = re.compile(
    r"\b(?:faculty|department|school|college|center|centre|institute|university|laboratory|lab)\b",
    re.IGNORECASE,
)

# 这个服务不直接改数据库，它只负责：
#   - 清洗前端编辑输入
#   - 从全文/analysis 中推断可补齐的元数据
# 最终 patch 是否落库，由上层 service/repository 决定。


class PaperCategoryAutofillResult(BaseModel):
    """AI 推断的论文分类结果。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: str = Field(default="")


def _clean_text(value: Any, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit is not None and len(text) > limit:
        return text[:limit].strip()
    return text


def normalize_authors(raw_value: list[str] | str | None) -> list[str]:
    """把输入作者标准化为作者列表。"""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        items = [_clean_text(item, 200) for item in raw_value]
    else:
        text = str(raw_value or "").strip()
        if not text:
            return []
        items = [_clean_text(item, 200) for item in AUTHOR_SPLIT_PATTERN.split(text)]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:20]


def normalize_categories(raw_value: list[str] | str | None) -> list[str]:
    """把输入分类标准化为分类列表。"""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        items = [_clean_text(item, 120) for item in raw_value]
    else:
        text = str(raw_value or "").strip()
        if not text:
            return []
        items = [_clean_text(item, 120) for item in CATEGORY_SPLIT_PATTERN.split(text)]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:10]


def normalize_text_list(
    raw_value: list[str] | str | None,
    *,
    split_pattern: re.Pattern[str] = CONTRIBUTION_SPLIT_PATTERN,
    item_limit: int = 400,
    max_items: int = 8,
) -> list[str]:
    """把自由文本或字符串列表清洗成稳定的短文本数组。"""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        items = [_clean_text(item, item_limit) for item in raw_value]
    else:
        text = str(raw_value or "").strip()
        if not text:
            return []
        items = [_clean_text(item, item_limit) for item in split_pattern.split(text)]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:max_items]


def normalize_methodology_type(raw_value: str | None) -> str:
    """把方法论类型收敛到约定枚举，避免任意字符串破坏下游展示。"""
    value = _clean_text(raw_value, 40).lower()
    if not value:
        return ""
    if value not in METHOD_TYPES:
        raise ValueError("方法类型需填写 experimental / theoretical / survey / hybrid")
    return value


def normalize_analysis_payload(raw_value: Any) -> dict[str, Any]:
    """规范化 analysis 结构，保证前端和图谱服务读取到稳定字段。"""
    analysis = raw_value if isinstance(raw_value, dict) else {}
    try:
        methodology_type = normalize_methodology_type(analysis.get("methodology_type"))
    except ValueError:
        methodology_type = _clean_text(analysis.get("methodology_type"), 40).lower()
    normalized = {
        "problem": _clean_text(analysis.get("problem"), 800),
        "method": _clean_text(analysis.get("method"), 800),
        "contribution": normalize_text_list(analysis.get("contribution")),
        "limitation": _clean_text(analysis.get("limitation"), 800),
        "conclusion": _clean_text(analysis.get("conclusion"), 1200),
        "conclusion_zh": _clean_text(analysis.get("conclusion_zh"), 1200),
        "keywords": normalize_categories(analysis.get("keywords")),
        "related_work": normalize_text_list(
            analysis.get("related_work"),
            split_pattern=CATEGORY_SPLIT_PATTERN,
            item_limit=240,
            max_items=12,
        ),
        "references": normalize_text_list(
            analysis.get("references"),
            split_pattern=CATEGORY_SPLIT_PATTERN,
            item_limit=400,
            max_items=30,
        ),
        "tldr": _clean_text(analysis.get("tldr"), 1200),
        "methodology_type": methodology_type,
        "methodology_detail": _clean_text(analysis.get("methodology_detail"), 2000),
        "source_markdown_path": _clean_text(analysis.get("source_markdown_path"), 600),
    }
    section_summaries = analysis.get("section_summaries")
    if isinstance(section_summaries, dict) and section_summaries:
        normalized["section_summaries"] = section_summaries
    figures = analysis.get("figures")
    if isinstance(figures, list) and figures:
        normalized["figures"] = figures
    return {
        key: value
        for key, value in normalized.items()
        if value not in ("", [])
    }


def normalize_published_date(raw_value: str | int | None) -> str:
    """把年份或日期标准化为 YYYY-MM-DD。"""
    value = str(raw_value or "").strip()
    if not value:
        return ""
    if YEAR_ONLY_PATTERN.fullmatch(value):
        return f"{value}-01-01"
    if not DATE_PATTERN.fullmatch(value):
        raise ValueError("年份需填写 YYYY 或 YYYY-MM-DD")
    normalized = value.replace("/", "-")
    parts = normalized.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1].zfill(2)}-01"
    return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"


def build_metadata_update_patch(payload: dict[str, Any]) -> dict[str, Any]:
    """把前端元数据编辑请求映射为 papers 表补丁。"""
    patch: dict[str, Any] = {}
    if "title" in payload:
        title = _clean_text(payload.get("title"), 400)
        if not title:
            raise ValueError("标题不能为空")
        patch["title"] = title
    if "authors" in payload:
        patch["authors"] = normalize_authors(payload.get("authors"))
    if "abstract" in payload:
        patch["abstract"] = _clean_text(payload.get("abstract"), 3000)
    if "published_date" in payload:
        patch["published_date"] = normalize_published_date(payload.get("published_date"))
    elif "year" in payload:
        patch["published_date"] = normalize_published_date(payload.get("year"))
    if "category" in payload:
        patch["categories"] = normalize_categories(payload.get("category"))
    if "doi" in payload:
        patch["doi"] = _clean_text(payload.get("doi"), 200)
    if "arxiv_id" in payload:
        patch["arxiv_id"] = _clean_text(payload.get("arxiv_id"), 120)
    analysis_patch: dict[str, Any] = {}
    if "keywords" in payload:
        analysis_patch["keywords"] = normalize_categories(payload.get("keywords"))
    if "problem" in payload:
        analysis_patch["problem"] = _clean_text(payload.get("problem"), 800)
    if "method" in payload:
        analysis_patch["method"] = _clean_text(payload.get("method"), 800)
    if "contribution" in payload:
        analysis_patch["contribution"] = normalize_text_list(payload.get("contribution"))
    if "limitation" in payload:
        analysis_patch["limitation"] = _clean_text(payload.get("limitation"), 800)
    if "tldr" in payload:
        analysis_patch["tldr"] = _clean_text(payload.get("tldr"), 1200)
    if "methodology_type" in payload:
        analysis_patch["methodology_type"] = normalize_methodology_type(payload.get("methodology_type"))
    if "methodology_detail" in payload:
        analysis_patch["methodology_detail"] = _clean_text(payload.get("methodology_detail"), 2000)
    if "related_work" in payload:
        analysis_patch["related_work"] = normalize_text_list(
            payload.get("related_work"),
            split_pattern=CATEGORY_SPLIT_PATTERN,
            item_limit=240,
            max_items=12,
        )
    if analysis_patch:
        patch["analysis"] = analysis_patch
    return patch


def _current_metadata(paper: dict[str, Any]) -> IngestedDocumentMetadata:
    return {
        "title": _clean_text(paper.get("title"), 400),
        "authors": normalize_authors(paper.get("authors")),
        "published_date": _clean_text(paper.get("published_date"), 32),
        "abstract": _clean_text(paper.get("abstract"), 3000),
    }


def _load_full_text_from_file(paper: dict[str, Any]) -> str:
    """从论文本地文件重建全文文本，供元数据自动补全使用。"""
    file_path = Path(str(paper.get("pdf_path") or "").strip())
    if not file_path.exists() or not file_path.is_file():
        return ""
    try:
        ingested = document_parse_service.parse_document_file(
            file_path,
            str(paper.get("paper_id") or "paper"),
            filename=file_path.name,
            include_metadata=False,
        )
    except Exception:
        return ""
    return ingested.get("full_text", "")


def _analysis_context(analysis: Any) -> list[str]:
    if not isinstance(analysis, dict):
        return []
    parts: list[str] = []
    tldr = _clean_text(analysis.get("tldr"), 500)
    if tldr:
        parts.append(f"TLDR: {tldr}")
    keywords = normalize_categories(analysis.get("keywords"))
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    method = _clean_text(analysis.get("method"), 500)
    if method:
        parts.append(f"Method: {method}")
    problem = _clean_text(analysis.get("problem"), 500)
    if problem:
        parts.append(f"Problem: {problem}")
    return parts


def _compose_fallback_text(paper: dict[str, Any]) -> str:
    """文件不可读时，把 title/abstract/analysis 拼成退化版推断上下文。"""
    parts: list[str] = []
    title = _clean_text(paper.get("title"), 400)
    authors = normalize_authors(paper.get("authors"))
    published_date = _clean_text(paper.get("published_date"), 32)
    abstract = _clean_text(paper.get("abstract"), 3000)
    if title:
        parts.append(f"Title: {title}")
    if authors:
        parts.append(f"Authors: {'; '.join(authors)}")
    if published_date:
        parts.append(f"Published date: {published_date}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    parts.extend(_analysis_context(paper.get("analysis")))
    return "\n".join(item for item in parts if item.strip()).strip()


def _resolve_autofill_text(paper: dict[str, Any]) -> str:
    return _load_full_text_from_file(paper) or _compose_fallback_text(paper)


def _looks_like_bad_abstract(value: str) -> bool:
    text = _clean_text(value, 3000)
    if not text:
        return False
    if AFFILIATION_LIKE_PATTERN.search(text) and len(text.split()) < 60:
        return True
    return bool(re.match(r"^\d+\s", text) and AFFILIATION_LIKE_PATTERN.search(text))


def _infer_category(full_text: str, paper: dict[str, Any]) -> str:
    """
    用 LLM 推断单个简洁分类。

    这里只补一个 category，而不是整套 taxonomy，目的是作为“缺失字段兜底”，而不是
    重新定义论文学科标签体系。
    """
    if not full_text.strip():
        return ""
    try:
        from app.services.llm_service import StructuredOutputError, get_llm
    except Exception:
        return ""

    prompt = (
        "You infer one concise academic paper category from the paper content.\n"
        "Rules:\n"
        "- Return only one short category.\n"
        "- Prefer domain labels such as NLP, Computer Vision, LLM, Machine Learning, Robotics, Biology.\n"
        "- Use empty string when unsure.\n\n"
        f"Title: {_clean_text(paper.get('title'), 400)}\n"
        f"Abstract: {_clean_text(paper.get('abstract'), 2000)}\n"
        f"Existing categories: {normalize_categories(paper.get('categories'))}\n\n"
        f"Document excerpt:\n{full_text[:4000]}"
    )
    try:
        result = get_llm().with_structured_output(PaperCategoryAutofillResult).invoke(prompt)
    except (ValidationError, StructuredOutputError):
        # 1. 【论文元信息】【合同失败】非法分类响应不能变成成功的空分类
        raise
    except Exception:
        return ""
    categories = normalize_categories(result.category)
    return categories[0] if categories else ""


def build_autofill_metadata_patch(paper: dict[str, Any]) -> dict[str, Any]:
    """
    根据现有论文内容生成仅补齐缺失字段的补丁。

    关键原则是“只补缺，不覆盖用户已确认值”，因此 title/authors/abstract/category
    只有在当前缺失时才会自动填充。
    """
    full_text = _resolve_autofill_text(paper)
    if not full_text:
        return {}
    file_path = Path(str(paper.get("pdf_path") or paper.get("title") or "paper.txt"))
    filename = file_path.name or "paper.txt"
    extracted = document_parse_service.extract_document_metadata(file_path, full_text, filename)
    current = _current_metadata(paper)
    patch: dict[str, Any] = {}

    if not current["title"] and extracted.get("title"):
        patch["title"] = _clean_text(extracted["title"], 400)
    if not current["authors"] and extracted.get("authors"):
        patch["authors"] = normalize_authors(extracted.get("authors"))
    if not current["published_date"] and extracted.get("published_date"):
        patch["published_date"] = normalize_published_date(extracted.get("published_date"))
    if (not current["abstract"] or _looks_like_bad_abstract(current["abstract"])) and extracted.get("abstract"):
        patch["abstract"] = _clean_text(extracted["abstract"], 3000)

    existing_categories = normalize_categories(paper.get("categories"))
    if not existing_categories:
        category = _infer_category(full_text, paper)
        if category:
            patch["categories"] = [category]

    return patch
