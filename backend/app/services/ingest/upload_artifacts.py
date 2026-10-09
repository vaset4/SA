"""上传不可变产物保存及结构完整性核验，作者：xxx。"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any
import zipfile

from app.models.upload_models import ArtifactIntegrityError, ArtifactRef
from app.services.pdf_document import open_pdf

HASH_BLOCK_BYTES = 1024 * 1024
MAX_ARCHIVE_ENTRIES = 20_000
MAX_EXPANDED_ARCHIVE_BYTES = 1024 * 1024 * 1024


def fingerprint(value: Any) -> str:
    """接收可序列化输入，返回稳定配置或载荷指纹，拒绝 NaN 等无效数字。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_file(path: Path) -> ArtifactRef:
    """流式核对实际文件，返回绝对路径、字节数与 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size": size}


def verify_artifact(reference: ArtifactRef, *, root: Path | None = None) -> Path:
    """核对受管路径和完整字节指纹；文件缺失或损坏直接失败，不重建假产物。"""
    path = Path(reference["path"]).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise ArtifactIntegrityError("上传产物路径超出受管目录")
    if not path.is_file():
        raise ArtifactIntegrityError(f"上传产物已丢失: {path.name}")
    actual = inspect_file(path)
    if actual["sha256"] != reference["sha256"] or actual["size"] != reference["size"]:
        raise ArtifactIntegrityError(f"上传产物指纹不匹配: {path.name}")
    return path


def write_artifact(directory: Path, content: bytes, *, suffix: str) -> ArtifactRef:
    """按内容指纹原子保存产物；相同产物复用，损坏的同名产物明确拒绝。"""
    if not re.fullmatch(r"\.[a-z0-9]+", suffix):
        raise ValueError("上传产物扩展名无效")
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    target = directory / (digest + suffix)
    reference: ArtifactRef = {"path": str(target.resolve()), "sha256": digest, "size": len(content)}
    if target.exists():
        verify_artifact(reference)
        return reference
    descriptor, temporary = tempfile.mkstemp(prefix="upload-", suffix=".part", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    verify_artifact(reference)
    return reference


def write_json_artifact(directory: Path, value: Any) -> ArtifactRef:
    """接收解析或索引载荷，保存可独立核验的完整 JSON 产物。"""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return write_artifact(directory, raw, suffix=".json")


def read_json_artifact(reference: ArtifactRef, *, root: Path | None = None) -> Any:
    """先验证原字节，再返回 JSON；损坏缓存不能作为已完成检查点。"""
    try:
        return json.loads(read_artifact_bytes(reference, root=root).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("上传 JSON 产物损坏") from exc


def read_artifact_bytes(reference: ArtifactRef, *, root: Path | None = None) -> bytes:
    """读取并核对同一份字节，避免先核验再读取时拿到被替换的文件内容。"""
    path = Path(reference["path"]).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise ArtifactIntegrityError("上传产物路径超出受管目录")
    if not path.is_file():
        raise ArtifactIntegrityError(f"上传产物已丢失: {path.name}")
    content = path.read_bytes()
    if len(content) != reference["size"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ArtifactIntegrityError(f"上传产物指纹不匹配: {path.name}")
    return content


def extract_mineru_archive(content: bytes, directory: Path) -> None:
    """【解析】【结果解包】核对 ZIP 路径、类型与展开体积后提取到指定受管目录。"""
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        paths: set[str] = set()
        if len(entries) > MAX_ARCHIVE_ENTRIES or sum(item.file_size for item in entries) > MAX_EXPANDED_ARCHIVE_BYTES:
            raise ArtifactIntegrityError("MinerU 结果包展开体积或文件数量超出允许范围")
        for item in entries:
            relative = PurePosixPath(item.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
                raise ArtifactIntegrityError("MinerU 结果包含越界路径")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ArtifactIntegrityError("MinerU 结果不允许符号链接")
            target = root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(root):
                raise ArtifactIntegrityError("MinerU 结果路径超出产物目录")
            normalized = os.path.normcase(str(target))
            if normalized in paths:
                raise ArtifactIntegrityError("MinerU 结果包含重复或大小写冲突的路径")
            paths.add(normalized)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, HASH_BLOCK_BYTES)


def verify_mineru_pages(source: Path | bytes, content_list: list[dict[str, Any]], *, model_path: Path) -> int:
    """【解析】【页级核验】核对真实 PDF 页数、model 页集合和可确认正文页的覆盖。"""
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(model, list) or not model:
        raise ArtifactIntegrityError("MinerU 结果缺少可核验的 model 页信息")
    pages_info = [item.get("page_info") for item in model if isinstance(item, dict)]
    if not all(isinstance(info, dict) for info in pages_info):
        raise ArtifactIntegrityError("MinerU 结果缺少可核验的 model 页信息")
    with open_pdf(source) as document:
        page_count = document.page_count
        expected = set(range(page_count))
        pages = [info.get("page_no") for info in pages_info]
        if len(pages) != page_count or set(pages) != expected:
            raise ArtifactIntegrityError("MinerU 结果页集合与原 PDF 不一致")
        content_pages = {item["page_idx"] for item in content_list if isinstance(item, dict) and "page_idx" in item}
        if not content_pages.issubset(expected):
            raise ArtifactIntegrityError("MinerU 正文包含超出原 PDF 的页号")
        missing = [index for index in expected - content_pages if document.page_text(index).strip()]
        if missing:
            raise ArtifactIntegrityError(f"MinerU 遗漏了原 PDF 中有文字的页面: {missing}")
    return page_count
