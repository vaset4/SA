"""MinerU Markdown 产物持久化与复用服务。"""

from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.ingest.mineru_client import MinerUClient

settings = get_settings()

# MinerU 新版输出会把上下标排版写成 <sub>/<sup> 标签，句读可能被裹进标签里（如 k<sub>s.</sub>），
# 下游按 ". " 切句时会因此失配，必须先把标签剥离、只保留内部文字。
_SUP_SUB_TAG_PATTERN = re.compile(r"</?(?:sub|sup)>")


def strip_sup_sub_markup(text: str) -> str:
    """去掉文本中的 <sub>/<sup> 排版标签，保留内部文字。"""
    return _SUP_SUB_TAG_PATTERN.sub("", text)


def artifact_dir_for(paper_id: str) -> Path:
    """返回论文对应的 MinerU 产物目录。"""
    path = Path(settings.data_dir) / "uploads" / f"{paper_id}.mineru"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path_for(paper_id: str) -> Path:
    return artifact_dir_for(paper_id) / "manifest.json"


def parse_pdf_document(file_path: Path, paper_id: str) -> dict[str, Any]:
    """解析 PDF 并返回标准化 Markdown 产物。"""
    artifact_dir = artifact_dir_for(paper_id)
    existing = _load_existing_artifacts(artifact_dir, paper_id, file_path)
    if existing is not None:
        return existing

    client = MinerUClient()
    warnings: list[str] = []
    manifest_path = artifact_dir / "manifest.json"
    try:
        _, extract_result, archive_bytes = client.parse_local_file(file_path)
        _extract_archive(archive_bytes, artifact_dir)
        source_markdown_path = _materialize_source_markdown(artifact_dir)
        sidecar_path = _materialize_content_list(artifact_dir)
        markdown_text = source_markdown_path.read_text(encoding="utf-8")
        content_list = _load_content_list(sidecar_path)
        page_count = _count_pages(content_list)
        manifest = {
            "paper_id": paper_id,
            "source_file_path": str(file_path),
            "parser_mode": "mineru_online",
            "mineru_batch_id": extract_result.batch_id,
            "artifact_paths": {
                "artifact_dir": str(artifact_dir),
                "source_markdown": str(source_markdown_path),
                "content_list": str(sidecar_path),
            },
            "created_at": _utc_now(),
            "status": "completed",
            "warnings": warnings,
            "normalized_page_count": page_count,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "parser_mode": "mineru_online",
            "artifact_dir": str(artifact_dir),
            "manifest_path": str(manifest_path),
            "source_markdown_path": str(source_markdown_path),
            "sidecar_path": str(sidecar_path),
            "markdown_text": markdown_text,
            "content_list": content_list,
            "warnings": warnings,
            "normalized_page_count": page_count,
        }
    except Exception as exc:
        failed_manifest = {
            "paper_id": paper_id,
            "source_file_path": str(file_path),
            "parser_mode": "mineru_online",
            "artifact_paths": {
                "artifact_dir": str(artifact_dir),
            },
            "created_at": _utc_now(),
            "status": "failed",
            "warnings": warnings,
            "error": str(exc),
        }
        manifest_path.write_text(json.dumps(failed_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def _load_existing_artifacts(artifact_dir: Path, paper_id: str, file_path: Path) -> dict[str, Any] | None:
    manifest_path = artifact_dir / "manifest.json"
    source_markdown_path = artifact_dir / "source.md"
    sidecar_path = artifact_dir / "content_list.json"
    if not (manifest_path.exists() and source_markdown_path.exists() and sidecar_path.exists()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            return None
        markdown_text = source_markdown_path.read_text(encoding="utf-8")
        content_list = _load_content_list(sidecar_path)
    except Exception:
        return None
    return {
        "parser_mode": str(manifest.get("parser_mode") or "mineru_online"),
        "artifact_dir": str(artifact_dir),
        "manifest_path": str(manifest_path),
        "source_markdown_path": str(source_markdown_path),
        "sidecar_path": str(sidecar_path),
        "markdown_text": markdown_text,
        "content_list": content_list,
        "warnings": list(manifest.get("warnings") or []),
        "normalized_page_count": int(manifest.get("normalized_page_count") or _count_pages(content_list)),
        "source_file_path": str(file_path),
        "reused": True,
    }


def _extract_archive(archive_bytes: bytes, artifact_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive.extractall(artifact_dir)


def _materialize_source_markdown(artifact_dir: Path) -> Path:
    canonical_path = artifact_dir / "source.md"
    source_candidates = [
        canonical_path,
        *artifact_dir.rglob("full.md"),
        *artifact_dir.rglob("*_full.md"),
    ]
    source_path = _first_existing_path(source_candidates)
    if source_path is None:
        raise ValueError("MinerU 结果缺少 full.md")
    if source_path != canonical_path:
        shutil.copyfile(source_path, canonical_path)
    return canonical_path


def _materialize_content_list(artifact_dir: Path) -> Path:
    canonical_path = artifact_dir / "content_list.json"
    sidecar_candidates = [
        canonical_path,
        *artifact_dir.rglob("*_content_list.json"),
        *artifact_dir.rglob("content_list_v2.json"),
        *artifact_dir.rglob("*_content_list_v2.json"),
    ]
    sidecar_path = _first_existing_path(sidecar_candidates)
    if sidecar_path is None:
        raise ValueError("MinerU 结果缺少 content_list.json")
    if sidecar_path != canonical_path:
        shutil.copyfile(sidecar_path, canonical_path)
    return canonical_path


def _first_existing_path(candidates: list[Path]) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_content_list(sidecar_path: Path) -> list[dict[str, Any]]:
    """读取 sidecar 后规范化正文块列表。"""
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return normalize_content_list(payload)


def normalize_content_list(payload: Any) -> list[dict[str, Any]]:
    """接收已解码且核验的 JSON，返回明确的正文块列表。"""
    if isinstance(payload, list):
        if any(not isinstance(item, dict) for item in payload):
            raise ValueError("MinerU 正文块必须是对象")
        return payload
    if isinstance(payload, dict):
        maybe_list = payload.get("content_list") or payload.get("data") or payload.get("items")
        if isinstance(maybe_list, list):
            return normalize_content_list(maybe_list)
    raise ValueError("MinerU content_list.json 结构异常")


def _count_pages(content_list: list[dict[str, Any]]) -> int:
    pages: set[int] = set()
    for item in content_list:
        raw_page = item.get("page_idx")
        if isinstance(raw_page, int):
            pages.add(raw_page)
        elif isinstance(raw_page, str) and raw_page.isdigit():
            pages.add(int(raw_page))
    return len(pages)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
