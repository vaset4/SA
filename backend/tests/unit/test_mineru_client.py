from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import document_parser
from app.services.ingest import document_parse_service, mineru_markdown_service
from app.services.ingest.mineru_client import MinerUClient


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, body: dict | None = None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._body = body
        self.content = content
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(body or {}, ensure_ascii=False)

    def json(self) -> dict:
        if self._body is None:
            raise ValueError("no json body")
        return self._body


class _FakeSession:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict]] = []
        self.put_calls: list[tuple[str, dict]] = []
        self.get_calls: list[str] = []

    def request(self, method: str, url: str, **kwargs):  # noqa: ANN003
        self.requests.append((method, url, kwargs))
        if url.endswith("/file-urls/batch"):
            return _FakeResponse(
                body={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.example/paper.pdf"],
                    },
                }
            )
        if url.endswith("/extract-results/batch/batch-1"):
            return _FakeResponse(
                body={
                    "code": 0,
                    "data": [
                        {
                            "file_name": "paper.pdf",
                            "state": "done",
                            "extract_result": [
                                {
                                    "full_zip_url": "https://result.example/paper.zip",
                                }
                            ],
                        }
                    ],
                }
            )
        raise AssertionError(f"unexpected request: {method} {url}")

    def put(self, url: str, **kwargs):  # noqa: ANN003
        self.put_calls.append((url, kwargs))
        return _FakeResponse(status_code=200)

    def get(self, url: str, **kwargs):  # noqa: ANN003
        self.get_calls.append(url)
        return _FakeResponse(status_code=200, content=b"zip-bytes")


def test_mineru_client_parse_local_file_uses_upload_poll_and_download(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    session = _FakeSession()
    client = MinerUClient(
        base_url="https://mineru.net/api/v4",
        api_token="token",
        session=session,
        sleeper=lambda *_args, **_kwargs: None,
    )

    batch_file, result, archive = client.parse_local_file(pdf_path)

    assert batch_file.batch_id == "batch-1"
    assert result.full_zip_url == "https://result.example/paper.zip"
    assert archive == b"zip-bytes"
    assert session.put_calls == [("https://upload.example/paper.pdf", {"data": b"%PDF-1.4", "timeout": client.timeout_seconds})]
    assert session.get_calls == ["https://result.example/paper.zip"]
    assert session.requests[0][0] == "POST"
    assert session.requests[0][1].endswith("/file-urls/batch")
    assert session.requests[0][2]["json"] == {
        "enable_formula": True,
        "enable_table": True,
        "files": [{"name": "paper.pdf"}],
    }


def test_parse_document_file_prefers_mineru_for_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(document_parse_service.settings, "mineru_enabled", True, raising=False)
    monkeypatch.setattr(
        mineru_markdown_service,
        "parse_pdf_document",
        lambda file_path, paper_id: {
            "parser_mode": "mineru_online",
            "artifact_dir": str(tmp_path / "paper.mineru"),
            "manifest_path": str(tmp_path / "paper.mineru" / "manifest.json"),
            "source_markdown_path": str(tmp_path / "paper.mineru" / "source.md"),
            "sidecar_path": str(tmp_path / "paper.mineru" / "content_list.json"),
            "markdown_text": "# Abstract\n\nMarkdown body",
            "content_list": [
                {"type": "text", "text": "Abstract", "text_level": 1, "page_idx": 0},
                {"type": "text", "text": "Markdown body", "page_idx": 0},
            ],
            "warnings": [],
            "normalized_page_count": 1,
        },
    )
    class _LocalRegistry:
        def parse_document(self, file_path: str, paper_id: str):  # noqa: ANN001
            raise AssertionError("local PdfParser should not be used")

    monkeypatch.setattr(document_parser, "get_parser_registry", lambda: _LocalRegistry())

    ingested = document_parse_service.parse_document_file(pdf_path, "paper_x", include_metadata=False)

    assert ingested["parser_mode"] == "mineru_online"
    assert ingested["file_type"] == "pdf"
    assert ingested["full_text"] == "# Abstract\n\nMarkdown body"
    assert ingested["normalized_page_count"] == 1


def test_mineru_structured_chunking_error_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(document_parse_service.settings, "mineru_enabled", True, raising=False)
    monkeypatch.setattr(
        mineru_markdown_service,
        "parse_pdf_document",
        lambda file_path, paper_id: {
            "parser_mode": "mineru_online",
            "markdown_text": "alpha\nbeta",
            "content_list": [
                {"type": "text", "text": "alpha", "page_idx": 0},
                {"type": "text", "text": "beta", "page_idx": 0},
            ],
            "warnings": [],
        },
    )

    class _LocalRegistry:
        def parse_document(self, file_path: str, paper_id: str):  # noqa: ANN001
            raise AssertionError("local PdfParser fallback must not be used")

    monkeypatch.setattr(document_parser, "get_parser_registry", lambda: _LocalRegistry())

    with pytest.raises(document_parser.StructuredChunkingError) as exc_info:
        document_parse_service.parse_document_file(pdf_path, "paper_bad", include_metadata=False)

    assert exc_info.value.code == "missing_section_path"


def test_parse_pdf_artifacts_strips_sup_sub_markup_before_chunking(tmp_path: Path) -> None:
    """句读被 <sub> 标签包裹时必须先剥标签再分块，否则整段无法切句并报 sentence_too_large。"""
    sentence = "k<sub>s.</sub> H<sub>owever,</sub> the model adapts to varying spatial granularity. "
    paragraph = sentence * 30
    ingested = document_parse_service.parse_pdf_artifacts(tmp_path / "paper.pdf", "paper_x", {
        "parser_mode": "mineru_online",
        "markdown_text": "# Ab<sub>s</sub>t<sub>rac</sub>t\n\n" + paragraph,
        "content_list": [],
    })

    assert "<sub>" not in ingested["full_text"]
    assert len(ingested["chunks"]) > 1


def test_parse_pdf_artifacts_strips_markup_from_table_caption(tmp_path: Path) -> None:
    """表名存在 table_caption 而不是 text 字段，同样要剥标签，否则 "Table 1" 碎成标签片段进索引。"""
    ingested = document_parse_service.parse_pdf_artifacts(tmp_path / "paper.pdf", "paper_caption", {
        "parser_mode": "mineru_online",
        "markdown_text": "# Paper\n\n## 4.1 Datasets\n\nplaceholder\n",
        "content_list": [
            {"type": "text", "text_level": 2, "text": "4.1 Datasets", "page_idx": 0},
            {
                "type": "table",
                "page_idx": 0,
                "table_caption": ["T<sub>a</sub>bl<sub>e</sub> 1: Th<sub>e</sub> datasets."],
                "table_footnote": ["R<sub>e</sub>d denotes the best."],
                "table_body": "<table><tr><td>METR-LA</td><td>207</td></tr></table>",
            },
        ],
    })

    table = next(chunk for chunk in ingested["chunks"] if chunk["block_type"] == "table")
    assert "Table 1: The datasets." in table["embedding_text"]
    assert "Red denotes the best." in table["embedding_text"]
    assert "<sub>" not in table["embedding_text"] and "</sub>" not in table["embedding_text"]
    # 行边界仍按原始 HTML 还原，剥标签不影响列对应关系
    assert "| METR-LA | 207 |" in table["embedding_text"]


def test_extract_text_metadata_uses_chunk_title_and_front_matter_authors() -> None:
    full_text = (
        "Bing $\\mathbf { V } \\mathbf { u } ^ { * 1 }$ , Haoteng $\\mathbf { Y i n ^ { * 2 , 3 } }$ , "
        "Zhanxing Zhu †3,4 1 School of Mathematical Sciences, Peking University, Beijing, China "
        "2 Academy for Advanced Interdisciplinary Studies, Peking University, Beijing, China "
        "3 Center for Data Science, Peking University, Beijing, China "
        "4 Beijing Institute of Big Data Research (BIBDR), Beijing, China {byu, htyin, zhanxing.zhu}@pku.edu.cn\n\n"
        "Timely accurate traffic forecast is crucial for urban traffic control and guidance."
    )
    chunks = [
        {
            "chunk_id": "paper_0_0",
            "parent_id": "paper_0",
            "text": "author block",
            "full_text": "author block",
            "section": "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting",
            "page_num": 1,
        }
    ]

    metadata = document_parse_service._extract_text_metadata(full_text, "paper.pdf", chunks=chunks)

    assert metadata["title"] == "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting"
    assert metadata["authors"] == ["Bing Vu", "Haoteng Yin", "Zhanxing Zhu"]
    assert metadata["published_date"] == ""


def test_extract_pdf_metadata_ignores_creation_date_without_primary_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/CreationDate": "D:20241012120007"})
    writer.write(pdf_path)
    writer.close()

    metadata = document_parse_service._extract_pdf_metadata(pdf_path)

    assert metadata["title"] == ""
    assert metadata["authors"] == []
    assert metadata["published_date"] == ""
