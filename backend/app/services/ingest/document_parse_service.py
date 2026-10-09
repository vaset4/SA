"""
上传文档解析服务。

这是上传流水线的第一阶段：校验文件、落盘、解析成 chunks、恢复全文并补基础元数据。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.models.ingest_models import IngestedDocument, IngestedDocumentMetadata
from app.models.models import Chunk
from app.services import document_parser
from app.services.ingest import mineru_markdown_service

settings = get_settings()

DEFAULT_MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".txt",
    ".text",
    ".tex",
    ".latex",
}
ZIP_MAGIC_EXTENSIONS = {".docx", ".pptx"}
PDF_DATE_PATTERN = re.compile(r"D:(\d{4})(\d{2})?(\d{2})?")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
DATE_PATTERN = re.compile(r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])([-/](0?[1-9]|[12]\d|3[01]))?\b")
ABSTRACT_PATTERN = re.compile(
    r"(?is)\babstract\b[\s:\-]*\n?(.*?)(?:\n\s*(?:keywords?|index terms)\b|\n\s*(?:1|i)\.?\s+introduction\b|\n\s*introduction\b)",
)
AUTHOR_SPLIT_PATTERN = re.compile(r"\s*(?:;|\band\b|\n)\s*", re.IGNORECASE)
NOISY_TITLE_PREFIXES = ("arxiv:", "http://", "https://", "www.")
AFFILIATION_START_PATTERN = re.compile(
    r"\b(?:\d+\s+)?(?:school|department|academy|center|centre|institute|college|university|laboratory|lab|faculty)\b",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"(?:\{[^{}]+\}@\S+|\b[\w.+-]+@[\w.-]+\.\w+\b)")
LATEX_INLINE_PATTERN = re.compile(r"\$([^$]+)\$")
NAME_PART_PATTERN = re.compile(r"^[A-Za-z][A-Za-z'.-]*$")
FRONT_MATTER_DATE_HINT_PATTERN = re.compile(
    r"\b(?:arxiv|published|accepted|conference|journal|proceedings|copyright|©)\b",
    re.IGNORECASE,
)
# MinerU 把正文摊在多个字段里：正文在 text，表名在 table_caption，图注在 image_caption 等。
# 只清洗 text 会让表名带着 <sub> 标签进索引（"Table 1" 变成 T<suba</subbl<sube</sub 1）。
# 不含 table_body 与 code_body：表格 HTML 交给 HTML 解析器处理标签，代码正文按原文保留。
_MINERU_TEXT_FIELDS = (
    "text", "title", "caption", "content",
    "table_caption", "table_footnote",
    "image_caption", "image_footnote", "img_caption", "img_footnote",
    "chart_caption", "chart_footnote", "figure_caption",
    "code_caption", "code_footnote",
    "list_items", "list_body",
    "footnote", "near_text", "latex", "formula",
)


class UploadMetadataLLMResult(BaseModel):
    """LLM 兜底提取上传文档元数据。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    published_date: str = ""
    abstract: str = ""


def upload_dir() -> Path:
    """返回上传目录。"""
    path = Path(settings.data_dir) / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_suffix(filename: str) -> str:
    """返回受支持的规范化扩展名。"""
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in ALLOWED_EXTENSIONS else ""


def resolve_max_upload_size_mb() -> int:
    """解析当前上传大小限制。"""
    return int(getattr(settings, "max_upload_size_mb", DEFAULT_MAX_UPLOAD_SIZE_MB))


def validate_extension(filename: str) -> str:
    """校验扩展名是否合法。"""
    ext = normalize_suffix(filename)
    if not ext:
        raise ValueError(f"不支持的文件格式: {Path(filename or '').suffix or '(none)'}")
    return ext


def validate_size(content: bytes) -> None:
    """校验上传体积是否超限。"""
    max_size_mb = resolve_max_upload_size_mb()
    max_size = max_size_mb * 1024 * 1024
    if len(content) > max_size:
        size_mb = len(content) / 1024 / 1024
        raise ValueError(f"文件大小 {size_mb:.1f}MB 超过限制 {max_size_mb}MB")


def validate_magic_bytes(ext: str, content: bytes) -> None:
    """按 magic bytes 做基础格式校验。"""
    if ext == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("文件扩展名是 .pdf，但内容不是合法 PDF")
    if ext in ZIP_MAGIC_EXTENSIONS and not content.startswith(b"PK\x03\x04"):
        raise ValueError(f"文件扩展名是 {ext}，但内容不是合法 Office 容器")


def save_uploaded_file(filename: str, content: bytes, paper_id: str) -> Path:
    """把上传文件安全落盘。"""
    ext = validate_extension(filename)
    validate_size(content)
    validate_magic_bytes(ext, content)
    file_path = upload_dir() / f"{paper_id}{ext}"
    file_path.write_bytes(content)
    return file_path


def merge_parent_texts(chunks: Iterable[Chunk]) -> str:
    """
    按 parent_id 去重后恢复近似全文。

    parser 拆 chunk 后可能重复引用同一父段落，这里通过 parent_id 进行去重拼回。
    """
    parent_texts: list[str] = []
    seen_parent_ids: set[str] = set()
    for chunk in chunks:
        parent_id = chunk["parent_id"]
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        parent_texts.append(chunk["full_text"])
    return "\n\n".join(parent_texts).strip()


def _build_ingested_document(
    file_path: Path,
    *,
    file_type: str,
    chunks: list[Chunk],
    parser_mode: str,
    artifact_payload: dict | None = None,
    parsed_content: dict | None = None,
) -> IngestedDocument:
    """参数为文件、分块和解析来源；返回上传文档，规范正文项目随内容版本一起持久传递。"""
    if not chunks:
        raise ValueError("文件解析成功，但未生成可索引分块")
    full_text = merge_parent_texts(chunks)
    if not full_text:
        raise ValueError("文件分块为空，无法生成全文内容")
    ingested: IngestedDocument = {
        "file_path": str(file_path),
        "chunks": chunks,
        "full_text": full_text,
        "file_type": file_type,
        "parser_mode": parser_mode,
    }
    if parsed_content is not None:
        ingested["content_schema_version"] = parsed_content["content_schema_version"]
        ingested["source_items"] = parsed_content["source_items"]
    if artifact_payload:
        if artifact_payload.get("artifact_dir"):
            ingested["artifact_dir"] = str(artifact_payload["artifact_dir"])
        if artifact_payload.get("source_markdown_path"):
            ingested["source_markdown_path"] = str(artifact_payload["source_markdown_path"])
        if artifact_payload.get("sidecar_path"):
            ingested["sidecar_path"] = str(artifact_payload["sidecar_path"])
        if artifact_payload.get("manifest_path"):
            ingested["manifest_path"] = str(artifact_payload["manifest_path"])
        if artifact_payload.get("warnings"):
            ingested["warnings"] = list(artifact_payload["warnings"])
        if artifact_payload.get("normalized_page_count") is not None:
            ingested["normalized_page_count"] = int(artifact_payload["normalized_page_count"])
    return ingested


def _parse_pdf_document(file_path: Path, paper_id: str) -> IngestedDocument:
    artifact_manifest_exists = mineru_markdown_service.manifest_path_for(paper_id).exists()
    should_use_mineru = bool(settings.mineru_enabled or artifact_manifest_exists)
    if should_use_mineru:
        artifact_payload = mineru_markdown_service.parse_pdf_document(file_path, paper_id)
        return parse_pdf_artifacts(file_path, paper_id, artifact_payload)
    return parse_local_document(file_path, paper_id)


def parse_local_document(file_path: Path, paper_id: str) -> IngestedDocument:
    """参数为本地路径和论文身份；返回正文和规范来源项目，只执行已选择的本地解析器。"""
    file_type = file_path.suffix.lower().lstrip(".")
    parsed = document_parser.get_parser_registry().parse_document(str(file_path), paper_id)
    return _build_ingested_document(file_path, file_type=file_type, chunks=parsed["chunks"],
        parser_mode=file_type + "_parser_local", parsed_content=parsed)


def parse_pdf_artifacts(file_path: Path, paper_id: str, artifacts: dict) -> IngestedDocument:
    """【上传】【正文转换】把已核验的 MinerU 产物转换为既有分块结构，不再次调用解析 API。"""
    normalized = _normalize_mineru_markup(artifacts)
    parsed = document_parser.parse_markdown_document(
        artifacts["markdown_text"], paper_id, content_list=normalized.get("content_list"),
        parser_mode=str(normalized.get("parser_mode") or "mineru_online"),
        text_normalizer=mineru_markdown_service.strip_sup_sub_markup,
    )
    document = _build_ingested_document(file_path, file_type="pdf", chunks=parsed["chunks"],
        parser_mode=str(normalized.get("parser_mode") or "mineru_online"), artifact_payload=artifacts,
        parsed_content=parsed)
    return {**document, "full_text": normalized["markdown_text"].strip()}


def _normalize_mineru_markup(artifacts: dict) -> dict:
    """【上传】【正文清洗】剥离 <sub>/<sup> 排版标签；句读被裹进标签会让分块切不开句子。"""
    content_list = artifacts.get("content_list") or []
    cleaned_items = [_normalize_mineru_item(item) for item in content_list]
    return {**artifacts, "markdown_text": mineru_markdown_service.strip_sup_sub_markup(artifacts["markdown_text"]),
        "content_list": cleaned_items}


def _normalize_mineru_item(item: object) -> object:
    """
    逐个内容块剥离排版标签。

    参数 item 为 MinerU content_list 的一项；返回同结构的新块，只替换带正文的字段，
    块类型、坐标、图片路径等原样保留。
    """
    if not isinstance(item, dict):
        return item
    replacements = {field: _strip_markup_value(item[field])
        for field in _MINERU_TEXT_FIELDS if field in item}
    return {**item, **replacements} if replacements else item


def _strip_markup_value(value: object) -> object:
    """参数为字段原值；按原有形状剥离标签：字符串直接剥，列表逐项剥，其他类型不动。"""
    if isinstance(value, str):
        return mineru_markdown_service.strip_sup_sub_markup(value)
    if isinstance(value, list):
        return [_strip_markup_value(entry) for entry in value]
    return value


def _clean_text(value: str, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit is not None and len(text) > limit:
        return text[:limit].strip()
    return text


def _split_authors(raw_value: str) -> list[str]:
    value = _clean_text(raw_value)
    if not value:
        return []
    front_matter_authors = _extract_authors_from_front_matter_line(value)
    if len(front_matter_authors) >= 2:
        return front_matter_authors[:12]
    parts = [item.strip(" ,;") for item in AUTHOR_SPLIT_PATTERN.split(value) if item.strip(" ,;")]
    if len(parts) <= 1 and "," in value:
        parts = [item.strip(" ,;") for item in value.split(",") if item.strip(" ,;")]
    authors: list[str] = []
    for item in parts:
        lowered = item.lower()
        if any(marker in lowered for marker in ("university", "department", "abstract", "@", "http", "www.")):
            continue
        if YEAR_PATTERN.search(item):
            continue
        if len(item.split()) > 6:
            continue
        authors.append(item)
    seen: set[str] = set()
    deduped: list[str] = []
    for author in authors:
        key = author.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(author)
    return deduped[:12]


def _normalize_latex_name_fragment(fragment: str) -> str:
    cleaned = re.sub(r"\\[A-Za-z]+", " ", fragment)
    cleaned = re.sub(r"\^\s*\{[^}]*\}", " ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = re.sub(r"[^A-Za-z\s-]", " ", cleaned)
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return ""
    collapsed: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if len(token) == 1:
            letters: list[str] = []
            while index < len(tokens) and len(tokens[index]) == 1:
                letters.append(tokens[index])
                index += 1
            collapsed.append("".join(letters))
            continue
        collapsed.append(token)
        index += 1
    return " ".join(collapsed).strip()


def _normalize_author_line(raw_value: str) -> str:
    value = EMAIL_PATTERN.sub(" ", str(raw_value or ""))
    value = LATEX_INLINE_PATTERN.sub(lambda match: f" {_normalize_latex_name_fragment(match.group(1))} ", value)
    affiliation_match = AFFILIATION_START_PATTERN.search(value)
    if affiliation_match:
        value = value[: affiliation_match.start()]
    value = re.sub(r"[†‡*]+", " ", value)
    value = re.sub(r"\b\d+\b", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,;")


def _extract_authors_from_front_matter_line(raw_value: str) -> list[str]:
    if not raw_value or not any(marker in raw_value for marker in (",", "@", "$", "†", "*")):
        return []
    value = _normalize_author_line(raw_value)
    if not value:
        return []
    candidates = [item.strip(" ,;") for item in re.split(r"\s*(?:,|;|\band\b)\s*", value) if item.strip(" ,;")]
    authors: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        parts = item.split()
        if len(parts) < 2 or len(parts) > 4:
            continue
        if not all(NAME_PART_PATTERN.match(part) for part in parts):
            continue
        normalized = " ".join(parts)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        authors.append(normalized)
    return authors[:12]


def _parse_pdf_date(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    match = PDF_DATE_PATTERN.search(value)
    if match:
        year = match.group(1)
        month = match.group(2) or "01"
        day = match.group(3) or "01"
        return f"{year}-{month}-{day}"
    if DATE_PATTERN.search(value):
        normalized = value.replace("/", "-")
        parts = normalized.split("-")
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1].zfill(2)}-01"
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    year_match = YEAR_PATTERN.search(value)
    if year_match:
        return f"{year_match.group(0)}-01-01"
    return ""


def _extract_pdf_metadata(file_path: Path) -> IngestedDocumentMetadata:
    """参数为原文件路径；返回 PDF 标题、作者及日期，无元数据可为空但读取失败明确报错。"""
    from app.services.pdf_document import open_pdf

    metadata: IngestedDocumentMetadata = {
        "title": "",
        "authors": [],
        "published_date": "",
        "abstract": "",
    }
    if file_path.suffix.lower() != ".pdf":
        return metadata
    with open_pdf(file_path) as document:
        raw = document.metadata()
        metadata["title"] = _clean_text(raw.get("title", ""), 400)
        metadata["authors"] = _split_authors(raw.get("author", ""))
        if metadata["title"] or metadata["authors"]:
            metadata["published_date"] = (
                _parse_pdf_date(raw.get("creationDate", ""))
                or _parse_pdf_date(raw.get("modDate", ""))
            )
        else:
            metadata["published_date"] = ""
        metadata["abstract"] = ""
    return metadata


def _extract_abstract_from_text(full_text: str) -> str:
    snippet = str(full_text or "")[:8000]
    match = ABSTRACT_PATTERN.search(snippet)
    if match:
        return _clean_text(match.group(1), 3000)

    paragraphs = [_clean_text(item) for item in str(full_text or "").split("\n\n")]
    paragraphs = [item for item in paragraphs if item]
    if not paragraphs:
        return ""
    for paragraph in paragraphs[:8]:
        lowered = paragraph.lower()
        if lowered.startswith(("abstract", "keywords", "index terms", "introduction")):
            continue
        if AFFILIATION_START_PATTERN.search(paragraph) or EMAIL_PATTERN.search(paragraph):
            continue
        if "@" in paragraph or "http" in lowered or "www." in lowered:
            continue
        if len(paragraph) < 80:
            continue
        if paragraph.count(".") + paragraph.count("。") < 1:
            continue
        if YEAR_PATTERN.search(paragraph) and len(paragraph.split()) < 12:
            continue
        return _clean_text(paragraph, 1500)
    if len(paragraphs) >= 2:
        return _clean_text(paragraphs[1], 1500)
    return _clean_text(paragraphs[0], 1500)


def _looks_like_author_line(line: str) -> bool:
    front_matter_authors = _extract_authors_from_front_matter_line(line)
    if len(front_matter_authors) >= 2:
        return True
    lowered = line.lower()
    authors = _split_authors(line)
    return (
        len(authors) >= 2
        and not lowered.startswith("abstract")
        and "university" not in lowered
        and len(line) <= 200
    )


def _extract_title_from_chunks(chunks: Iterable[Chunk] | None) -> str:
    if not chunks:
        return ""
    seen_sections: set[str] = set()
    for chunk in chunks:
        section = _clean_text(chunk.get("section", ""), 400)
        if not section:
            continue
        lowered = section.lower()
        if lowered in {"untitled", "unknown", "abstract", "introduction"}:
            continue
        if lowered in seen_sections:
            continue
        seen_sections.add(lowered)
        if _section_is_plausible_title(section):
            return section
    return ""


def _section_is_plausible_title(section: str) -> bool:
    lowered = section.lower()
    if lowered.startswith(NOISY_TITLE_PREFIXES):
        return False
    if re.match(r"^(?:abstract|references|bibliography)\b", lowered):
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*)\s+", section):
        return False
    return 8 <= len(section) <= 220


def _extract_title_from_text(full_text: str, filename: str, *, chunks: Iterable[Chunk] | None = None) -> str:
    title_from_chunks = _extract_title_from_chunks(chunks)
    if title_from_chunks:
        return title_from_chunks
    lines = [_clean_text(line) for line in str(full_text or "").splitlines()]
    lines = [line for line in lines if line]
    title_lines: list[str] = []
    for line in lines[:12]:
        lowered = line.lower()
        if lowered.startswith("abstract"):
            break
        if lowered.startswith(NOISY_TITLE_PREFIXES):
            continue
        if len(line) < 8 or len(line) > 220:
            continue
        if _looks_like_author_line(line):
            if title_lines:
                break
            continue
        title_lines.append(line)
        if len(" ".join(title_lines)) >= 40 or len(title_lines) >= 2:
            break
    title = _clean_text(" ".join(title_lines), 400)
    if title and title.lower() != Path(filename or "").stem.lower():
        return title
    return ""


def _extract_authors_from_text(full_text: str) -> list[str]:
    lines = [_clean_text(line) for line in str(full_text or "").splitlines()]
    lines = [line for line in lines if line]
    for line in lines[:8]:
        lowered = line.lower()
        if lowered.startswith("abstract"):
            break
        authors = _extract_authors_from_front_matter_line(line)
        if len(authors) >= 2:
            return authors
        if len(line) > 220:
            continue
        authors = _split_authors(line)
        if len(authors) >= 2:
            return authors
    return []


def _extract_published_date_from_text(full_text: str) -> str:
    lines = [_clean_text(line) for line in str(full_text or "").splitlines()]
    lines = [line for line in lines if line][:12]
    for line in lines:
        lowered = line.lower()
        if len(line) > 160 and not FRONT_MATTER_DATE_HINT_PATTERN.search(lowered):
            continue
        direct_match = DATE_PATTERN.search(line)
        if direct_match:
            normalized = direct_match.group(0).replace("/", "-")
            parts = normalized.split("-")
            if len(parts) == 2:
                return f"{parts[0]}-{parts[1].zfill(2)}-01"
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    for line in lines:
        lowered = line.lower()
        if len(line) > 120 and not FRONT_MATTER_DATE_HINT_PATTERN.search(lowered):
            continue
        year_match = YEAR_PATTERN.search(line)
        if year_match:
            return f"{year_match.group(0)}-01-01"
    return ""


def _extract_text_metadata(
    full_text: str,
    filename: str,
    *,
    chunks: Iterable[Chunk] | None = None,
) -> IngestedDocumentMetadata:
    return {
        "title": _extract_title_from_text(full_text, filename, chunks=chunks),
        "authors": _extract_authors_from_text(full_text),
        "published_date": _extract_published_date_from_text(full_text),
        "abstract": _extract_abstract_from_text(full_text),
    }


def _merge_metadata(*sources: IngestedDocumentMetadata) -> IngestedDocumentMetadata:
    merged: IngestedDocumentMetadata = {
        "title": "",
        "authors": [],
        "published_date": "",
        "abstract": "",
    }
    for source in sources:
        if not source:
            continue
        if source.get("title") and not merged["title"]:
            merged["title"] = source["title"]
        if source.get("authors") and not merged["authors"]:
            merged["authors"] = source["authors"]
        if source.get("published_date") and not merged["published_date"]:
            merged["published_date"] = source["published_date"]
        if source.get("abstract") and not merged["abstract"]:
            merged["abstract"] = source["abstract"]
    return merged


def _needs_llm_metadata(metadata: IngestedDocumentMetadata) -> bool:
    return not all([
        metadata.get("title"),
        metadata.get("authors"),
        metadata.get("published_date"),
        metadata.get("abstract"),
    ])


def _fill_metadata_with_llm(full_text: str, metadata: IngestedDocumentMetadata) -> IngestedDocumentMetadata:
    """补全原文中缺失的元信息，调用失败明确抛出，由独立任务记录。"""
    from typing import cast
    from app.services.ingest.upload_metadata_service import complete_metadata, metadata_client

    if not _needs_llm_metadata(metadata):
        return metadata
    return cast(IngestedDocumentMetadata, complete_metadata(full_text, metadata, client=metadata_client()))


def extract_document_metadata(
    file_path: Path,
    full_text: str,
    filename: str,
    *,
    chunks: Iterable[Chunk] | None = None,
    use_llm: bool = True,
) -> IngestedDocumentMetadata:
    """提取原文中可识别的信息；恢复流水线可关闭 LLM 并单独补全。"""
    pdf_metadata = _extract_pdf_metadata(file_path)
    heuristic_metadata = _extract_text_metadata(full_text, filename, chunks=chunks)
    merged = _merge_metadata(pdf_metadata, heuristic_metadata)
    return _fill_metadata_with_llm(full_text, merged) if use_llm else merged


def parse_document_file(
    file_path: Path,
    paper_id: str,
    *,
    filename: str | None = None,
    include_metadata: bool = True,
) -> IngestedDocument:
    """对已经落盘的文件执行统一解析。"""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        ingested = _parse_pdf_document(file_path, paper_id)
    else:
        parsed = document_parser.get_parser_registry().parse_document(str(file_path), paper_id)
        ingested = _build_ingested_document(
            file_path,
            file_type=ext.lstrip("."),
            chunks=parsed["chunks"],
            parser_mode=f"{ext.lstrip('.')}_parser",
            parsed_content=parsed,
        )
    if include_metadata:
        ingested["metadata"] = extract_document_metadata(
            file_path,
            ingested["full_text"],
            filename or file_path.name,
            chunks=ingested["chunks"],
        )
    return ingested


def ingest_uploaded_document(filename: str, content: bytes, paper_id: str) -> IngestedDocument:
    """
    执行上传文件的纯解析阶段。

    返回值是后续分析、索引和图谱阶段共享的共同输入。
    """
    ext = validate_extension(filename)
    file_path = save_uploaded_file(filename, content, paper_id)
    ingested = parse_document_file(file_path, paper_id, filename=filename, include_metadata=True)
    ingested["file_type"] = ext.lstrip(".")
    return ingested
