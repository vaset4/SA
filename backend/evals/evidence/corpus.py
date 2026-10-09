"""只读原文产物、当前切块及版本化索引的校验入口。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from evals.contracts import read_snapshot, snapshot, snapshot_hash
from evals.evidence.alignment import EvidenceDataError, build_index_mapping, validate_benchmark
from evals.evidence.contracts import (
    EvidenceBenchmark, IndexMapping, IndexSnapshot, IndexUnit, SourceBlock,
    PayloadKind, SourceSnapshot, make_index_snapshot, make_source_snapshot,
)

OFFLINE_BACKEND = "offline_current_chunker_bm25_over_embedding_text"
_PAPER_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_HASH_READ_SIZE = 1024 * 1024
NATIVE_PDF_EXTRACTION_MODE = "pypdf_pdf_page_text_v1"


@dataclass(frozen=True, kw_only=True)
class SourcePaths:
    """受控 uploads 根目录内的原文及解析产物路径。"""

    uploads_dir: Path
    uploaded_pdf: Path
    retained_pdf: Path
    manifest: Path
    markdown: Path
    content_list: Path
    parser_mode: str


def _checked_file(path: Path, root: Path) -> Path:
    """参数为文件和允许根目录；返回存在且未越界的真实路径。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise EvidenceDataError("invalid_input", "source_path_outside_uploads")
    if not resolved.is_file():
        raise EvidenceDataError("missing", f"source_artifact_missing:{path.name}")
    return resolved


def load_json_object(path: Path) -> dict[str, Any]:
    """参数为 JSON 文件；返回对象，文件缺失或格式错误保留明确原因。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceDataError("missing", f"json_file_missing:{path.name}") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise EvidenceDataError("invalid_input", f"json_file_invalid:{path.name}") from exc
    if not isinstance(payload, dict):
        raise EvidenceDataError("invalid_input", f"json_object_required:{path.name}")
    return payload


def resolve_source_paths(paper_id: str, *, uploads_dir: Path) -> SourcePaths:
    """参数为论文 ID 与 uploads 目录；返回完整且唯一的原文产物路径。"""
    if not _PAPER_ID.fullmatch(paper_id):
        raise EvidenceDataError("invalid_input", "invalid_paper_path_identity")
    root = uploads_dir.resolve()
    artifact_dir = root / f"{paper_id}.mineru"
    manifest_path = _checked_file(artifact_dir / "manifest.json", root)
    manifest = load_json_object(manifest_path)
    if manifest.get("paper_id") != paper_id:
        raise EvidenceDataError("invalid_input", "source_manifest_paper_mismatch")
    parser_mode = manifest.get("parser_mode")
    if not isinstance(parser_mode, str) or not parser_mode.strip():
        raise EvidenceDataError("missing", "source_parser_mode_missing")
    retained = list(artifact_dir.glob("*_origin.pdf"))
    if len(retained) != 1:
        status: Literal["missing", "invalid_input"] = "missing" if not retained else "invalid_input"
        raise EvidenceDataError(status, f"retained_pdf_count:{len(retained)}")
    return SourcePaths(
        uploads_dir=root, uploaded_pdf=_checked_file(root / f"{paper_id}.pdf", root),
        retained_pdf=_checked_file(retained[0], root), manifest=manifest_path,
        markdown=_checked_file(artifact_dir / "source.md", root),
        content_list=_checked_file(artifact_dir / "content_list.json", root),
        parser_mode=parser_mode,
    )


def load_content_list(path: Path) -> list[dict[str, Any]]:
    """参数为 MinerU sidecar；沿用现有容器读取规则并验证每项为对象。"""
    from app.services.ingest.mineru_markdown_service import _load_content_list

    try:
        items = _load_content_list(path)
    except FileNotFoundError as exc:
        raise EvidenceDataError("missing", "source_content_list_missing") from exc
    except (ValueError, UnicodeError) as exc:
        raise EvidenceDataError("invalid_input", "source_content_list_invalid") from exc
    if not isinstance(items, list) or not items or any(not isinstance(item, dict) for item in items):
        raise EvidenceDataError("invalid_input", "source_content_items_invalid")
    return items


def _file_fingerprint(path: Path, root: Path) -> dict[str, Any]:
    """参数为已校验文件和根目录；返回相对路径、字节数与流式 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while content := stream.read(_HASH_READ_SIZE):
            digest.update(content)
    return {"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def fingerprint_source_artifacts(paths: SourcePaths) -> dict[str, Any]:
    """参数为真实产物路径；分别保留上传、留存和解析文件身份。"""
    files = {name: _file_fingerprint(getattr(paths, name), paths.uploads_dir) for name in (
        "uploaded_pdf", "retained_pdf", "manifest", "markdown", "content_list",
    )}
    return {
        "schema_version": "evidence-artifacts-v1", "files": files,
        "parser_manifest": load_json_object(paths.manifest),
        "page_numbering": "content_list.page_idx + 1",
        "block_identity": "content-item-{zero_based_index:06d}",
    }


def load_source_blocks(paths: SourcePaths) -> tuple[SourceBlock, ...]:
    """参数为解析路径；按原 content item 身份保留正文、章节和真实页码。"""
    from app.services.document_parser import _content_item_to_markdown_block

    blocks: list[SourceBlock] = []
    headings: list[str] = []
    for index, item in enumerate(load_content_list(paths.content_list)):
        # 1. 【评测来源】【页码核验】非法原始页号先记录为来源输入错误，避免正文转换改变错误分类
        raw_page = item.get("page_idx")
        if raw_page is not None and (type(raw_page) is not int or raw_page < 0):
            raise EvidenceDataError("invalid_input", f"source_page_missing:content-item-{index:06d}")
        block = _content_item_to_markdown_block(item)
        if block is None:
            continue
        if block["type"] == "heading":
            headings = headings[:int(block["level"]) - 1] + [block["text"].strip()]
        page_num = block["page_num"]
        if raw_page is None or type(page_num) is not int or page_num <= 0:
            raise EvidenceDataError("invalid_input", f"source_page_missing:content-item-{index:06d}")
        blocks.append(SourceBlock(
            block_id=f"content-item-{index:06d}", page_num=page_num,
            section_path=" / ".join(headings), text=block["text"].strip(),
        ))
    if not blocks:
        raise EvidenceDataError("invalid_input", "source_has_no_text_blocks")
    return tuple(blocks)


def load_source_snapshot(paper_id: str, *, uploads_dir: Path) -> SourceSnapshot:
    """参数为论文 ID 和受控原文目录；返回可重建且指纹已校验的原文快照。"""
    paths = resolve_source_paths(paper_id, uploads_dir=uploads_dir)
    manifest = load_json_object(paths.manifest)
    if manifest.get("status") != "completed":
        raise EvidenceDataError("invalid_input", "source_parse_not_completed")
    artifacts = fingerprint_source_artifacts(paths)
    # 1. 表格和公式保留原始结构产物，文本定位不能替代版面复核
    artifacts["structured_content_items"] = [
        {"block_id": f"content-item-{index:06d}", "item": item}
        for index, item in enumerate(load_content_list(paths.content_list))
        if item.get("type") in {"table", "equation", "formula", "image", "figure"}
    ]
    return make_source_snapshot(
        paper_id=paper_id, pdf_sha256=artifacts["files"]["uploaded_pdf"]["sha256"],
        artifact_manifest_snapshot=snapshot(artifacts), blocks=load_source_blocks(paths),
        parser_engine_version=manifest.get("parser_engine_version"),
    )


def _chunk_mappings(source: SourceSnapshot, chunk: dict[str, Any]) -> tuple[list[IndexMapping], list[dict[str, Any]]]:
    """参数为原文和真实 chunk；分别定位 child/parent，缺口保留为可审计条目。"""
    mappings: list[IndexMapping] = []
    issues: list[dict[str, Any]] = []
    payloads: list[tuple[PayloadKind, str]] = [("child", chunk["text"])]
    if chunk.get("parent_id") and chunk.get("full_text"):
        payloads.append(("parent", chunk["full_text"]))
    for kind, text in payloads:
        try:
            mappings.append(build_index_mapping(
                source, text, index_snapshot_id="pending", paper_id=source.paper_id,
                chunk_id=chunk["chunk_id"], parent_id=chunk.get("parent_id") or None,
                payload_kind=kind,
            ))
        except EvidenceDataError as exc:
            issues.append({"paper_id": source.paper_id, "chunk_id": chunk["chunk_id"],
                           "payload_kind": kind, "status": exc.status, "reason": exc.reason})
    return mappings, issues


def build_index_from_chunks(
    sources: tuple[SourceSnapshot, ...], chunks: list[dict[str, Any]], *, retrieval_backend: str,
) -> tuple[IndexSnapshot, list[dict[str, Any]]]:
    """参数为真实原文、child 全集及来源；返回内容寻址索引和未定位的映射缺口。"""
    sources_by_paper = {source.paper_id: source for source in sources}
    if len(sources_by_paper) != len(sources):
        raise EvidenceDataError("invalid_input", "multiple_current_sources_for_paper")
    units: list[IndexUnit] = []
    mappings: list[IndexMapping] = []
    issues: list[dict[str, Any]] = []
    for chunk in chunks:
        paper_id = chunk.get("paper_id")
        source = sources_by_paper.get(paper_id) if isinstance(paper_id, str) else None
        if source is None:
            raise EvidenceDataError("invalid_input", "index_paper_source_missing")
        text = chunk.get("text")
        unit = IndexUnit.model_validate({
            "paper_id": source.paper_id, "chunk_id": chunk.get("chunk_id"), "parent_id": chunk.get("parent_id") or None,
            "text_snapshot": text, "text_sha256": snapshot_hash(text) if isinstance(text, str) else None,
        })
        units.append(unit)
        mapped, missing = _chunk_mappings(source, chunk)
        mappings.extend(mapped)
        issues.extend(missing)
    index = make_index_snapshot(
        source_snapshot_ids=tuple(source.snapshot_id for source in sources),
        retrieval_backend=retrieval_backend, units=tuple(units), mappings=tuple(mappings),
    )
    return index, issues


def snapshot_current_chunks(source: SourceSnapshot, *, source_paths: SourcePaths) -> dict[str, Any]:
    """参数为原文与真实缓存路径；执行当前切块器并返回离线索引、chunk 和缺口。"""
    from app.services.document_parser import parse_markdown_text

    current = load_source_snapshot(source.paper_id, uploads_dir=source_paths.uploads_dir)
    if current.snapshot_id != source.snapshot_id:
        raise EvidenceDataError("invalid_input", "source_artifacts_changed")
    chunks = parse_markdown_text(
        source_paths.markdown.read_text(encoding="utf-8"), source.paper_id,
        content_list=load_content_list(source_paths.content_list), parser_mode=source_paths.parser_mode,
    )
    if not chunks:
        raise EvidenceDataError("invalid_input", "current_chunker_returned_empty")
    actual = [{**chunk, "paper_id": source.paper_id} for chunk in chunks]
    index, issues = build_index_from_chunks((source,), actual, retrieval_backend=OFFLINE_BACKEND)
    return {"index_snapshot": index.model_dump(mode="json"), "chunks": actual, "mapping_issues": issues}


def load_index_snapshot(path: Path, *, sources: tuple[SourceSnapshot, ...]) -> IndexSnapshot:
    """参数为声明的索引文件和原文；只读校验内部指纹、全部引用与文本映射。"""
    index = IndexSnapshot.model_validate(load_json_object(path))
    validate_benchmark(EvidenceBenchmark(
        dataset_version="index-validation-v1", review_status="candidate", sources=sources,
        spans=(), groups=(), index_snapshots=(index,),
    ))
    return index


def build_native_source_snapshot(paper_id: str, pdf_path: Path, *, paper_dir: Path) -> SourceSnapshot:
    """参数为论文身份、PDF 路径和允许目录；返回绑定原字节、真实页和解析器版本的快照。"""
    from importlib.metadata import version
    from app.services.pdf_document import open_pdf

    path = _checked_file(pdf_path, paper_dir)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    blocks: list[SourceBlock] = []
    try:
        with open_pdf(content) as document:
            for index in range(document.page_count):
                text = document.page_text(index)
                if text.strip():
                    blocks.append(SourceBlock(block_id=f"pypdf-page-{index + 1:06d}",
                        page_num=index + 1, section_path="", text=text))
    except ValueError as error:
        raise EvidenceDataError("invalid_input", f"native_pdf_unreadable:{error}") from error
    if not blocks:
        raise EvidenceDataError("invalid_input", "native_pdf_has_no_text; use_MinerU_for_OCR")
    artifacts = {
        "schema_version": "pdf-annotation-artifacts-v1", "extraction_mode": NATIVE_PDF_EXTRACTION_MODE,
        "files": {"source_pdf": {"path": path.relative_to(paper_dir.resolve()).as_posix(),
                                "sha256": digest, "bytes": len(content)}},
    }
    return make_source_snapshot(paper_id=paper_id, pdf_sha256=digest,
        artifact_manifest_snapshot=snapshot(artifacts), blocks=tuple(blocks),
        parser_engine_version=f"pypdf:{version('pypdf')}")


def validate_native_source(source: SourceSnapshot, *, paper_dir: Path) -> Path:
    """参数为冻结原生 PDF 身份和允许目录；重建校验真实页、正文、字节和引擎后返回路径。"""

    artifacts = read_snapshot(source.artifact_manifest_snapshot)
    mode = artifacts.get("extraction_mode")
    if mode == "pymupdf_pdf_text_blocks":
        raise EvidenceDataError("invalid_input",
            "legacy_native_pdf_extraction_unsupported: 旧 PyMuPDF 标注需重新准备并复核，不能直接沿用")
    if mode != NATIVE_PDF_EXTRACTION_MODE:
        raise EvidenceDataError("invalid_input", "native_pdf_extraction_contract_unknown")
    file_info = artifacts.get("files", {}).get("source_pdf", {})
    relative = Path(file_info.get("path", ""))
    path = _checked_file(paper_dir / relative, paper_dir)
    actual = _file_fingerprint(path, paper_dir.resolve())
    if actual["sha256"] != source.pdf_sha256 or actual["sha256"] != file_info.get("sha256"):
        raise EvidenceDataError("invalid_input", "native_pdf_checksum_mismatch")
    if actual["bytes"] != file_info.get("bytes"):
        raise EvidenceDataError("invalid_input", "native_pdf_size_mismatch")
    current = build_native_source_snapshot(source.paper_id, path, paper_dir=paper_dir)
    if current.pdf_sha256 != source.pdf_sha256 or current.pdf_sha256 != file_info.get("sha256"):
        raise EvidenceDataError("invalid_input", "native_pdf_checksum_mismatch")
    if source.parser_engine_version != current.parser_engine_version:
        raise EvidenceDataError("invalid_input", "native_pdf_parser_version_mismatch; rebuild_and_review_annotations")
    # 1. 【Evidence评测】【原文校验】整组页原文参与比对，遗漏和伪造块均不能冒充完整快照
    if source.blocks != current.blocks:
        raise EvidenceDataError("invalid_input", "native_pdf_block_mismatch")
    return path


def _validate_source_assets(
    source: SourceSnapshot, *, uploads_dir: Path, paper_dir: Path | None,
) -> None:
    """参数为冻结原文及允许目录；按实际产物类型核对身份，不匹配时抛出原文错误。"""
    artifacts = read_snapshot(source.artifact_manifest_snapshot)
    if artifacts.get("schema_version") == "pdf-annotation-artifacts-v1":
        if paper_dir is None:
            raise EvidenceDataError("missing", "native_pdf_root_not_declared")
        validate_native_source(source, paper_dir=paper_dir)
        return
    current = load_source_snapshot(source.paper_id, uploads_dir=uploads_dir)
    if current.snapshot_id != source.snapshot_id:
        raise EvidenceDataError("invalid_input", "source_artifacts_changed")


def validate_case_assets(
    benchmark: EvidenceBenchmark, *, uploads_dir: Path, paper_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """参数为案例及两类允许原文目录；分别校验原生 PDF 和 MinerU，返回真实状态。"""
    validate_benchmark(benchmark)
    rows: list[dict[str, Any]] = []
    for source in benchmark.sources:
        try:
            _validate_source_assets(source, uploads_dir=uploads_dir, paper_dir=paper_dir)
            rows.append({"paper_id": source.paper_id, "status": "measured", "reason": "source_artifacts_verified"})
        except EvidenceDataError as exc:
            rows.append({"paper_id": source.paper_id, "status": exc.status, "reason": exc.reason})
    return rows
