"""
引用格式生成服务。

这个模块负责把论文元数据格式化为 APA / MLA / GBT7714 / BibTeX / IEEE
 等引用文本，属于“元数据整理与输出”链路，不参与全文检索或会话状态管理。

当前元数据来源按优先级分为三层：
  - 本地 `paper_db`：优先读取业务库中已保存的论文主数据
  - CrossRef：当提供 DOI 且本地没有足够数据时，补充外部元数据
  - arXiv：当提供 arXiv ID 且本地未命中时，通过 `arxiv_tool` 补充

需要注意：
  - 它不会直接操作底层数据库连接池，只是调用 `paper_db` 的同步包装函数
  - 它输出的是“格式化后的引用字符串”，不是引用网络或论文正文内容
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal

import aiohttp

import app.services.paper_db as paper_db
from app.tools.arxiv_tool import get_arxiv_tool

logger = logging.getLogger(__name__)
CitationStyle = Literal["apa", "mla", "gbt7714", "bibtex", "ieee"]
SUPPORTED_STYLES: set[str] = {"apa", "mla", "gbt7714", "bibtex", "ieee"}
ARXIV_ID_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")

# 这里的整体策略是“先找最可信、最便宜的数据源”：
#   1. 本地业务库
#   2. DOI -> CrossRef
#   3. arXiv ID -> arXiv
# 这样既能复用已有清洗结果，也减少不必要的外部请求。


def _paper_year(metadata: dict) -> str:
    """提取可用于引用格式的年份；缺失或非法时回退为 `n.d.`。"""
    year = str(metadata.get("year", "") or "").strip()
    return year if year.isdigit() else "n.d."


def _title_case(title: str) -> str:
    """压缩标题中的多余空白，保持输出稳定。"""
    return " ".join(str(title or "").split())


def _split_name(name: str) -> tuple[str, list[str]]:
    """把作者名拆成姓和名前缀，供多种引用风格复用。"""
    cleaned = " ".join(str(name or "").replace(".", " ").split())
    if not cleaned:
        return "Unknown", []
    parts = cleaned.split(" ")
    return parts[-1], parts[:-1]


def _author_last_first(name: str) -> str:
    """输出 `Last, First` 形式作者名。"""
    last, first_parts = _split_name(name)
    if not first_parts:
        return last
    return f"{last}, {' '.join(first_parts)}"


def _author_initials(name: str) -> str:
    """输出 IEEE 风格常见的姓名首字母缩写。"""
    last, first_parts = _split_name(name)
    initials = " ".join(f"{part[0]}." for part in first_parts if part)
    return f"{initials} {last}".strip()


def _crossref_year(item: dict) -> str:
    """从 CrossRef 返回的多个候选日期字段中提取年份。"""
    for key in ("published-print", "published-online", "issued", "created"):
        date_parts = item.get(key, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
            if year.isdigit():
                return year
    return ""


def _normalize_local(paper: dict) -> dict:
    """把本地论文记录统一归一化为格式化器内部使用的元数据结构。"""
    return {
        "title": _title_case(paper.get("title", "")),
        "authors": [str(author) for author in paper.get("authors", []) if str(author).strip()],
        "year": str(paper.get("published_date", "") or "")[:4],
        "journal": "arXiv preprint" if paper.get("arxiv_id") else "",
        "volume": "",
        "issue": "",
        "pages": "",
        "doi": str(paper.get("doi", "") or ""),
        "arxiv_id": str(paper.get("arxiv_id", "") or ""),
        "url": str(paper.get("pdf_url", "") or ""),
    }


async def _fetch_crossref(doi: str) -> dict | None:
    """通过 DOI 调 CrossRef，补充引用格式所需的标准元数据。"""
    url = f"https://api.crossref.org/works/{doi}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    return None
                payload = await response.json()
    except Exception as exc:
        logger.warning("CrossRef 查询失败 (%s): %s", doi, exc)
        return None

    item = payload.get("message", {})
    authors = []
    for author in item.get("author", []):
        given = str(author.get("given", "") or "").strip()
        family = str(author.get("family", "") or "").strip()
        full_name = " ".join(part for part in (given, family) if part)
        if full_name:
            authors.append(full_name)
    return {
        "title": _title_case(" ".join(item.get("title", []))),
        "authors": authors,
        "year": _crossref_year(item),
        "journal": _title_case(" ".join(item.get("container-title", []))),
        "volume": str(item.get("volume", "") or ""),
        "issue": str(item.get("issue", "") or ""),
        "pages": str(item.get("page", "") or ""),
        "doi": doi,
        "arxiv_id": "",
        "url": str(item.get("URL", "") or ""),
    }


async def _fetch_arxiv(arxiv_id: str) -> dict | None:
    """通过 arXiv 工具读取论文元数据，并转换为统一格式。"""
    paper = await asyncio.to_thread(get_arxiv_tool().get_by_id, arxiv_id)
    if not paper:
        return None
    return _normalize_local(paper)


def _apa_authors(authors: list[str]) -> str:
    """格式化 APA 作者串。"""
    formatted = [_author_last_first(name) for name in authors if name.strip()]
    if not formatted:
        return "Unknown"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def _mla_authors(authors: list[str]) -> str:
    """格式化 MLA 作者串。"""
    formatted = [_author_last_first(name) for name in authors if name.strip()]
    if not formatted:
        return "Unknown"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {authors[1]}"
    return f"{formatted[0]} et al."


def _ieee_authors(authors: list[str]) -> str:
    """格式化 IEEE 作者串。"""
    formatted = [_author_initials(name) for name in authors if name.strip()]
    if not formatted:
        return "Unknown"
    if len(formatted) > 6:
        return ", ".join(formatted[:6]) + ", et al."
    return ", ".join(formatted)


def _format_apa(metadata: dict) -> str:
    """按 APA 风格生成引用文本。"""
    authors = _apa_authors(metadata["authors"])
    year = _paper_year(metadata)
    title = metadata["title"] or "Untitled"
    journal = metadata.get("journal", "")
    doi = metadata.get("doi", "")
    url = metadata.get("url", "")
    if journal:
        suffix = f" *{journal}*."
    else:
        suffix = "."
    locator = f" https://doi.org/{doi}" if doi else f" {url}" if url else ""
    return f"{authors} ({year}). {title}.{suffix}{locator}".strip()


def _format_mla(metadata: dict) -> str:
    """按 MLA 风格生成引用文本。"""
    authors = _mla_authors(metadata["authors"])
    title = metadata["title"] or "Untitled"
    journal = metadata.get("journal", "")
    year = _paper_year(metadata)
    url = metadata.get("url", "")
    if journal:
        body = f'*{journal}*, {year}'
    else:
        body = year
    tail = f", {url}" if url else ""
    return f'{authors}. "{title}." {body}{tail}.'


def _format_gbt7714(metadata: dict) -> str:
    """按 GBT 7714 风格生成引用文本。"""
    authors = [_author_last_first(name) for name in metadata["authors"] if name.strip()]
    author_text = ", ".join(authors[:3]) if authors else "Unknown"
    if len(authors) > 3:
        author_text += ", 等"
    title = metadata["title"] or "Untitled"
    journal = metadata.get("journal", "")
    year = _paper_year(metadata)
    if journal:
        return f"{author_text}. {title}[J]. {journal}, {year}."
    return f"{author_text}. {title}[Z]. {year}."


def _format_bibtex(metadata: dict) -> str:
    """按 BibTeX 条目格式生成引用文本。"""
    first_author = _split_name(metadata["authors"][0])[0].lower() if metadata["authors"] else "unknown"
    key = re.sub(r"[^a-z0-9]", "", f"{first_author}{_paper_year(metadata)}")
    authors = " and ".join(_author_last_first(name) for name in metadata["authors"] if name.strip())
    lines = [
        f"@article{{{key},",
        f"  author = {{{authors or 'Unknown'}}},",
        f"  title = {{{metadata['title'] or 'Untitled'}}},",
        f"  year = {{{_paper_year(metadata)}}},",
    ]
    if metadata.get("journal"):
        lines.append(f"  journal = {{{metadata['journal']}}},")
    if metadata.get("doi"):
        lines.append(f"  doi = {{{metadata['doi']}}},")
    if metadata.get("url"):
        lines.append(f"  url = {{{metadata['url']}}},")
    lines.append("}")
    return "\n".join(lines)


def _format_ieee(metadata: dict) -> str:
    """按 IEEE 风格生成引用文本。"""
    authors = _ieee_authors(metadata["authors"])
    title = metadata["title"] or "Untitled"
    journal = metadata.get("journal", "")
    year = _paper_year(metadata)
    url = metadata.get("url", "")
    if journal:
        tail = f'*{journal}*, {year}'
    else:
        tail = year
    if url:
        tail = f"{tail}, {url}"
    return f'{authors}, "{title}," {tail}.'


class CitationFormatter:
    """
    引用格式生成器。

    它的输入是论文标识或外部元数据标识（`paper_id / doi / arxiv_id`），
    输出是最终引用字符串；不会返回结构化 citation graph，也不会直接返回
    原始元数据。
    """

    async def format(
        self,
        paper_id: str = "",
        doi: str = "",
        arxiv_id: str = "",
        style: CitationStyle = "apa",
    ) -> str:
        """格式化单篇论文引用；至少需要一个有效标识能解析到元数据。"""
        selected_style = style.lower()
        if selected_style not in SUPPORTED_STYLES:
            raise ValueError(f"不支持的引用格式: {style}")

        metadata = await self._get_metadata(paper_id=paper_id, doi=doi, arxiv_id=arxiv_id)
        if not metadata:
            raise ValueError("无法获取论文元数据")

        formatters = {
            "apa": _format_apa,
            "mla": _format_mla,
            "gbt7714": _format_gbt7714,
            "bibtex": _format_bibtex,
            "ieee": _format_ieee,
        }
        return formatters[selected_style](metadata)

    async def format_batch(self, paper_ids: list[str], style: CitationStyle = "apa") -> list[dict]:
        """
        批量格式化多篇本地论文，返回 `paper_id -> citation` 列表。

        当前实现是串行处理，优先保证单条报错语义清晰；如果后续批量规模变大，可以再并发化。
        """
        results = []
        for paper_id in paper_ids:
            citation = await self.format(paper_id=paper_id, style=style)
            results.append({"paper_id": paper_id, "citation": citation})
        return results

    async def _get_metadata(self, paper_id: str = "", doi: str = "", arxiv_id: str = "") -> dict | None:
        """
        按“本地业务库优先，外部源补充”的顺序收集元数据。

        当前不会直接使用 `database.py` 连接池，而是通过 `paper_db` 的业务读
        接口获取本地论文信息。
        """
        if paper_id:
            local_paper = await asyncio.to_thread(paper_db.get_paper, paper_id)
            if local_paper:
                return _normalize_local(local_paper)
            if ARXIV_ID_PATTERN.fullmatch(paper_id):
                local_by_arxiv = await asyncio.to_thread(paper_db.get_by_arxiv_id, paper_id)
                if local_by_arxiv:
                    return _normalize_local(local_by_arxiv)

        if doi:
            metadata = await _fetch_crossref(doi)
            if metadata:
                return metadata

        if arxiv_id:
            local_by_arxiv = await asyncio.to_thread(paper_db.get_by_arxiv_id, arxiv_id)
            if local_by_arxiv:
                return _normalize_local(local_by_arxiv)
            metadata = await _fetch_arxiv(arxiv_id)
            if metadata:
                return metadata

        return None


_FORMATTER: CitationFormatter | None = None


def get_citation_formatter() -> CitationFormatter:
    """返回引用格式服务单例。"""
    global _FORMATTER
    if _FORMATTER is None:
        _FORMATTER = CitationFormatter()
    return _FORMATTER
