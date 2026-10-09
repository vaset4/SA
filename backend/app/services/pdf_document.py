"""统一读取 PDF 页、文字和元数据，作者：xxx，2026-10-08 22:00:00。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class PdfDocument:
    """【文档】【PDF 读取】在一个打开周期内复用读取器，不缓存或改写原文件。"""

    _reader: PdfReader

    @property
    def page_count(self) -> int:
        """无参数；返回真实物理页数，含空白页和扫描页。"""
        return len(self._reader.pages)

    def page_text(self, index: int) -> str:
        """参数为从零开始的物理页索引；返回原生文字，不对扫描内容执行 OCR。"""
        return self._reader.pages[index].extract_text()

    def metadata(self) -> dict[str, str]:
        """无参数；返回标题、作者和 PDF 日期原值，缺少的可选字段为空字符串。"""
        raw = self._reader.metadata
        names = {"title": "/Title", "author": "/Author", "creationDate": "/CreationDate", "modDate": "/ModDate"}
        return {name: str(raw.get(key) or "") if raw else "" for name, key in names.items()}


@contextmanager
def open_pdf(source: Path | str | bytes) -> Iterator[PdfDocument]:
    """参数为路径或 PDF 字节；返回受管读取上下文，拒绝需密码、零页和不可读文件。"""
    stream = BytesIO(source) if isinstance(source, bytes) else Path(source).open("rb")
    with stream:
        try:
            # 1. 【文档】【PDF 读取】不恢复损坏结构；加密标记不等于必须输入密码
            reader = PdfReader(stream, strict=True)
            try:
                if reader.is_encrypted and not reader.decrypt(""):
                    raise ValueError("原 PDF 需要密码，无法解析")
                document = PdfDocument(reader)
                if not document.page_count:
                    raise ValueError("原 PDF 为空，无法解析")
                yield document
            finally:
                reader.close()
        except PdfReadError as error:
            raise ValueError(f"原 PDF 无法读取: {error}") from error
