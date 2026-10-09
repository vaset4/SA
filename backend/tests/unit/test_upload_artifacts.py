"""原文件和解析产物完整性回归，作者：xxx。"""
from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from pdf_fixtures import write_pdf
import pytest

from app.models.upload_models import ArtifactIntegrityError
from app.services.ingest.upload_artifacts import (
    extract_mineru_archive, fingerprint, read_json_artifact, verify_artifact,
    verify_mineru_pages, write_artifact, write_json_artifact,
)


def test_immutable_artifact_reuses_same_bytes_and_rejects_damage(tmp_path) -> None:
    """同一有效产物复用，磁盘损坏后不能被当成命中缓存。"""
    first = write_artifact(tmp_path, b"source", suffix=".pdf")
    assert write_artifact(tmp_path, b"source", suffix=".pdf") == first
    Path(first["path"]).write_bytes(b"broken")
    with pytest.raises(ArtifactIntegrityError, match="指纹"):
        verify_artifact(first)
    with pytest.raises(ArtifactIntegrityError, match="指纹"):
        write_artifact(tmp_path, b"source", suffix=".pdf")


def test_json_round_trip_checks_bytes_and_directory(tmp_path) -> None:
    """JSON 读取先核对来源和完整字节，且禁止越过受管根目录。"""
    value = {"text": "完整正文", "chunks": [{"id": "one", "text": "段落"}]}
    reference = write_json_artifact(tmp_path / "artifacts", value)
    assert read_json_artifact(reference, root=tmp_path) == value
    with pytest.raises(ArtifactIntegrityError, match="受管目录"):
        verify_artifact(reference, root=tmp_path / "another")
    Path(reference["path"]).unlink()
    with pytest.raises(ArtifactIntegrityError, match="丢失"):
        read_json_artifact(reference)


def test_fingerprints_bind_values_and_reject_nan(tmp_path) -> None:
    """字段顺序不改变指纹，内容或模型改变必须改变指纹。"""
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"model": "a"}) != fingerprint({"model": "b"})
    with pytest.raises(ValueError):
        write_json_artifact(tmp_path, {"embedding": [float("nan")]})


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "C:/outside.txt", "folder\\..\\outside.txt"])
def test_archive_rejects_path_escape(tmp_path, name) -> None:
    """供应商 ZIP 中的越界路径不能写入受管目录之外。"""
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ArtifactIntegrityError, match="越界"):
        extract_mineru_archive(content.getvalue(), tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()


def _pdf_fixture(directory: Path, *, blank_second: bool = False) -> tuple[Path, Path]:
    """创建两页真实 PDF 及页结构产物，返回路径。"""
    source = directory / "source.pdf"
    write_pdf(source, ("First page text", "" if blank_second else "Second page text"))
    model = directory / "model.json"
    model.write_text(json.dumps([{"page_info": {"page_no": 0}}, {"page_info": {"page_no": 1}}]), encoding="utf-8")
    return source, model


def test_detectable_missing_text_page_fails(tmp_path) -> None:
    """正文漏掉可确认有文字的一整页时不能通过完成核验。"""
    source, model = _pdf_fixture(tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="遗漏"):
        verify_mineru_pages(source, [{"page_idx": 0}], model_path=model)
    assert verify_mineru_pages(source, [{"page_idx": 0}, {"page_idx": 1}], model_path=model) == 2


def test_blank_page_is_not_treated_as_missing_text(tmp_path) -> None:
    """空白页保留在总页数里，不机械要求它产生正文块。"""
    source, model = _pdf_fixture(tmp_path, blank_second=True)
    assert verify_mineru_pages(source, [{"page_idx": 0}], model_path=model) == 2


def test_same_count_wrong_page_set_fails(tmp_path) -> None:
    """页数相同但页集合错误同样不能通过核验。"""
    source, model = _pdf_fixture(tmp_path)
    model.write_text(json.dumps([{"page_info": {"page_no": 0}}, {"page_info": {"page_no": 2}}]), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="页集合"):
        verify_mineru_pages(source, [{"page_idx": 0}], model_path=model)
