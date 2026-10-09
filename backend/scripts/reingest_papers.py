"""旧 MinerU 缓存的离线分块预览；实际重建使用论文详情的任务入口，作者：xxx。

Usage:
  python backend/scripts/reingest_papers.py --dry-run
  python backend/scripts/reingest_papers.py --dry-run --paper-id upload_2292fcc2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOADS_DIR = BACKEND_ROOT / "data" / "uploads"
COMPLETED_STATUS = "completed"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@dataclass(frozen=True)
class PaperCandidate:
    paper_id: str
    pdf_path: Path
    artifact_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class ReingestOptions:
    uploads_dir: Path
    paper_ids: tuple[str, ...]
    dry_run: bool


@dataclass(frozen=True)
class ReingestDeps:
    parse_document: Callable[[Path, str], dict[str, Any]]


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(argv)
    configure_data_dir(options.uploads_dir)
    results = run_reingest(options)
    print(json.dumps({"papers": results}, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> ReingestOptions:
    parser = argparse.ArgumentParser(description="Preview cached MinerU chunking; use paper rebuild tasks to publish changes")
    parser.add_argument("--uploads-dir", type=Path, default=DEFAULT_UPLOADS_DIR)
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paper_ids = tuple(str(item).strip() for item in args.paper_id if str(item).strip())
    return ReingestOptions(
        uploads_dir=args.uploads_dir,
        paper_ids=paper_ids,
        dry_run=bool(args.dry_run),
    )


def configure_data_dir(uploads_dir: Path) -> None:
    """Point app settings at the data root that owns the selected uploads dir."""
    os.environ["DATA_DIR"] = str(uploads_dir.resolve().parent)


def run_reingest(options: ReingestOptions, deps: ReingestDeps | None = None) -> list[dict[str, Any]]:
    """读取已缓存原文并返回分块预览；旧直写模式在访问任何存储前拒绝。"""
    if not options.dry_run:
        raise ValueError("旧脚本只支持 --dry-run；请在论文详情使用“重新处理”，通过独立批次核验发布")
    candidates = discover_candidates(options.uploads_dir, options.paper_ids)
    active_deps = deps or default_deps()
    plans = [build_reingest_plan(candidate, active_deps) for candidate in candidates]
    return [{**without_chunks(plan), "action": "dry_run"} for plan in plans]


def discover_candidates(uploads_dir: Path, paper_ids: tuple[str, ...]) -> list[PaperCandidate]:
    requested = set(paper_ids)
    candidates: list[PaperCandidate] = []
    for artifact_dir in sorted(uploads_dir.glob("*.mineru")):
        paper_id = artifact_dir.name.removesuffix(".mineru")
        if requested and paper_id not in requested:
            continue
        candidate = candidate_from_artifact(uploads_dir, paper_id, artifact_dir)
        if candidate is not None:
            candidates.append(candidate)
    found = {candidate.paper_id for candidate in candidates}
    missing = sorted(requested - found)
    if missing:
        raise FileNotFoundError(f"no completed cached MinerU artifacts for: {', '.join(missing)}")
    if not candidates:
        raise FileNotFoundError(f"no completed cached MinerU papers under {uploads_dir}")
    return candidates


def candidate_from_artifact(
    uploads_dir: Path,
    paper_id: str,
    artifact_dir: Path,
) -> PaperCandidate | None:
    manifest_path = artifact_dir / "manifest.json"
    source_path = artifact_dir / "source.md"
    content_list_path = artifact_dir / "content_list.json"
    pdf_path = uploads_dir / f"{paper_id}.pdf"
    if not (manifest_path.exists() and source_path.exists() and content_list_path.exists() and pdf_path.exists()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != COMPLETED_STATUS:
        return None
    return PaperCandidate(
        paper_id=paper_id,
        pdf_path=pdf_path,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
    )


def default_deps() -> ReingestDeps:
    """返回只读解析端口，不再提供数据库或索引写入端口。"""
    return ReingestDeps(parse_document=_parse_cached_document)


def _parse_cached_document(path: Path, paper_id: str) -> dict[str, Any]:
    """仅读取候选缓存并运行本地分块，返回原文和分块；缓存损坏时不重新调用解析 API。"""
    from app.services.document_parser import parse_markdown_text
    from app.services.ingest.mineru_markdown_service import normalize_content_list

    artifact_dir = path.with_suffix(".mineru")
    text = (artifact_dir / "source.md").read_text(encoding="utf-8")
    content_list = normalize_content_list(json.loads((artifact_dir / "content_list.json").read_text(encoding="utf-8")))
    chunks = parse_markdown_text(text, paper_id, content_list=content_list, parser_mode="mineru_online")
    return {"file_path": str(path), "file_type": "pdf", "parser_mode": "mineru_online", "full_text": text, "chunks": chunks}


def build_reingest_plan(candidate: PaperCandidate, deps: ReingestDeps) -> dict[str, Any]:
    ingested = deps.parse_document(candidate.pdf_path, candidate.paper_id)
    chunks = list(ingested.get("chunks") or [])
    if not chunks:
        raise ValueError(f"{candidate.paper_id} parsed successfully but produced no chunks")
    file_type = str(ingested.get("file_type") or candidate.pdf_path.suffix.lstrip(".") or "pdf")
    return {
        "paper_id": candidate.paper_id,
        "pdf_path": str(candidate.pdf_path),
        "artifact_dir": str(candidate.artifact_dir),
        "parser_mode": str(ingested.get("parser_mode") or ""),
        "file_type": file_type,
        "chunk_count": len(chunks),
        "parent_count": count_parents(chunks),
        "sect_key_distribution": sect_key_distribution(chunks),
        "chunks": chunks,
    }


def count_parents(chunks: list[dict[str, Any]]) -> int:
    return len({str(chunk.get("parent_id") or "") for chunk in chunks if chunk.get("parent_id")})


def sect_key_distribution(chunks: list[dict[str, Any]]) -> dict[str, int]:
    from app.services import document_parser

    keys: Counter[str] = Counter()
    seen_parent_ids: set[str] = set()
    for chunk in chunks:
        parent_id = str(chunk.get("parent_id") or chunk.get("chunk_id") or "")
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        section_path = str(chunk.get("section_path") or chunk.get("section") or "")
        keys[document_parser._sect_key(section_path)] += 1
    return dict(sorted(keys.items()))


def without_chunks(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "chunks"}


if __name__ == "__main__":
    raise SystemExit(main())
