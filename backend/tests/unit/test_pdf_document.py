"""公开版 PDF 读取及真实上传产物回归，作者：xxx。"""
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image
from pypdf import PdfWriter
import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from app.api.facades.paper_facade import validate_upload
from app.models.upload_models import ArtifactIntegrityError
from app.services import document_parser, upload_service
from app.services.ingest.document_parse_service import _extract_pdf_metadata, parse_local_document
from app.services.ingest.upload_artifacts import write_artifact
from app.services.ingest.upload_parse_service import UploadDocumentParser
from app.services.pdf_document import PdfDocument, open_pdf
from pdf_fixtures import PAGE_SIZE, pdf_bytes, write_pdf


@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-256"])
@pytest.mark.parametrize("password", ["", "private-test-password"])
def test_password_rules_use_actual_readability(algorithm, password):
    """参数为加密算法和用户密码；空密码可读，需密码文件明确拒绝。"""
    writer = PdfWriter(clone_from=BytesIO(pdf_bytes()))
    writer.encrypt(password, owner_password="test-owner-password", algorithm=algorithm)
    output = BytesIO()
    writer.write(output)
    writer.close()
    if password:
        with pytest.raises(ValueError, match="需要密码"):
            with open_pdf(output.getvalue()):
                pytest.fail("受密码保护的 PDF 不得通过受理")
        return
    with open_pdf(output.getvalue()) as document:
        assert document.page_count == 1
        assert document.page_text(0).strip() == "Original evidence."


@pytest.mark.parametrize("content", [b"", b"not a PDF", b"%PDF-1.4", pdf_bytes()[:200]])
def test_broken_pdf_is_not_reported_as_empty_success(content):
    """参数为损坏字节；必须保留读取失败与原始异常，不能返回空文档。"""
    with pytest.raises(ValueError, match="无法读取") as error:
        with open_pdf(content):
            pytest.fail("坏文件不能通过")
    assert error.value.__cause__ is not None


def test_zero_page_rejected_but_blank_and_scanned_pages_remain_valid():
    """无参数；零页拒绝，合法空白和扫描页保留页数且不凭空产生 OCR 文字。"""
    empty = BytesIO()
    writer = PdfWriter()
    writer.write(empty)
    writer.close()
    with pytest.raises(ValueError, match="为空"):
        with open_pdf(empty.getvalue()):
            pytest.fail("零页文件不能通过")
    scan = BytesIO()
    Image.new("RGB", (30, 30), "white").save(scan, format="PDF")
    with open_pdf(scan.getvalue()) as document:
        assert document.page_count == 1 and document.page_text(0) == ""
    with open_pdf(pdf_bytes(("", "Evidence on page two."))) as document:
        assert document.page_count == 2
        assert document.page_text(0) == ""
        assert document.page_text(1).strip() == "Evidence on page two."


def test_chinese_and_columns_keep_known_text_and_physical_pages(tmp_path):
    """参数为目录；中文和多栏文字均从真实 PDF 读取，页码包括中间空白页。"""
    path = tmp_path / "columns.pdf"
    canvas = Canvas(str(path), pagesize=PAGE_SIZE, invariant=1)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    canvas.setFont("STSong-Light", 12)
    canvas.drawString(50, 740, "研究方法与实验结果")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(50, 710, "Left column evidence.")
    canvas.drawString(320, 710, "Right column evidence.")
    canvas.showPage()
    canvas.showPage()
    canvas.drawString(50, 740, "Third physical page.")
    canvas.save()
    sections = document_parser.PdfParser().collect_sections(str(path))
    assert [section["page_num"] for section in sections] == [1, 3]
    with open_pdf(path) as document:
        text = document.page_text(0)
        assert "研究方法与实验结果" in text
        assert "Left column evidence." in text and "Right column evidence." in text


def test_metadata_reads_fields_without_extracting_body(tmp_path, monkeypatch):
    """参数为目录和替身；独立读取元信息，不为标题作者重复提取全文。"""
    path = tmp_path / "metadata.pdf"
    writer = PdfWriter(clone_from=BytesIO(pdf_bytes()))
    writer.add_metadata({"/Title": "PDF research", "/Author": "Alice; Bob", "/CreationDate": "D:20241012120007"})
    writer.write(path)
    writer.close()
    def unexpected_text(self, index):
        """参数为读取器和页索引；元数据调用不应进入全文提取。"""
        pytest.fail("元数据读取不应提取全文")
    monkeypatch.setattr(PdfDocument, "page_text", unexpected_text)
    metadata = _extract_pdf_metadata(path)
    assert metadata["title"] == "PDF research"
    assert metadata["authors"] == ["Alice", "Bob"]
    assert metadata["published_date"] == "2024-10-12"


def test_upload_acceptance_reads_real_pages_and_preserves_bytes(tmp_path, monkeypatch):
    """参数为目录和配置替身；真实受理保存原字节，坏文件在写入前失败。"""
    monkeypatch.setattr(upload_service, "get_settings", lambda: SimpleNamespace(data_dir=str(tmp_path)))
    content = pdf_bytes(("Original evidence.", ""))
    reference = upload_service._save_source("paper.pdf", content, paper_id="test-paper")
    assert Path(reference["path"]).read_bytes() == content
    with pytest.raises(ValueError, match="无法读取"):
        upload_service._save_source("paper.pdf", b"%PDF-1.4", paper_id="broken-paper")
    assert not (tmp_path / "uploads" / "broken-paper").exists()


def test_epub_mime_is_rejected():
    """无参数；协议入口拒绝 EPUB MIME，避免受理已经移除的电子书格式。"""
    with pytest.raises(ValueError, match="MIME"):
        validate_upload("paper.epub", "application/epub+zip")


def image_bundle(root: Path, content: bytes, extension: str) -> tuple[UploadDocumentParser, bytes, bytes]:
    """参数为目录、图片字节和后缀；返回真实解析器、ZIP 字节和对应 PDF 原文。"""
    source = pdf_bytes()
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("full.md", "# Evidence\n\nOriginal evidence.")
        archive.writestr("content_list.json", json.dumps([{"type": "text", "text": "Original evidence.", "page_idx": 0}]))
        archive.writestr("model.json", json.dumps([{"page_info": {"page_no": 0}}]))
        archive.writestr("figure." + extension, content)
    source_ref = write_artifact(root, source, suffix=".pdf")
    runner = SimpleNamespace(execution=SimpleNamespace(paper_id="paper", parse_id="parse"))
    parser = UploadDocumentParser(runner=runner, client=SimpleNamespace(), source=source_ref,
                                 filename="paper.pdf", root=root)
    return parser, output.getvalue(), source


@pytest.mark.parametrize("format,extension", [("PNG", "png"), ("JPEG", "jpg")])
@pytest.mark.parametrize("damaged", [False, True])
def test_mineru_images_must_decode_completely(tmp_path, format, extension, damaged):
    """参数为目录、格式和损坏标志；真实 ZIP 物化时必须解码完整图片，坏图保留根因。"""
    output = BytesIO()
    Image.new("RGB", (100, 100), "red").save(output, format=format)
    data = output.getvalue()
    if damaged:
        data = data[:len(data) // 2]
    parser, archive_ref, source = image_bundle(tmp_path, data, extension)
    if damaged:
        with pytest.raises(ArtifactIntegrityError, match="图片资源损坏") as error:
            parser._materialize_bundle(archive_ref, source)
        assert error.value.__cause__ is not None
        return
    result = parser._materialize_bundle(archive_ref, source)
    assert result["normalized_page_count"] == 1
    assert "figure." + extension in result["resources"]


def test_local_pdf_sources_reach_parent_child_chunks(tmp_path):
    """参数为目录；本地 PDF 解析保留父子块正文和真实页来源。"""
    path = write_pdf(tmp_path / "local.pdf", ("1 Method\nFirst page evidence.", "", "2 Results\nThird page evidence."))
    result = parse_local_document(path, "paper")
    assert {item["page_num"] for item in result["source_items"].values()} == {1, 3}
    assert "First page evidence." in result["full_text"]
    assert "Third page evidence." in result["full_text"]
    assert all(chunk["text"] and chunk["parent_id"] and chunk["full_text"] for chunk in result["chunks"])
