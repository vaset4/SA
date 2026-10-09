"""独立索引批次载荷、向量缓存与完整集合核验，作者：xxx。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
import struct
from typing import Any

from app.config import get_embedding_identity, get_settings
from app.models.upload_models import ArtifactIntegrityError
from app.services.ingest.document_index_service import build_upload_metadata
from app.services.ingest.upload_artifacts import fingerprint, read_json_artifact, write_json_artifact
from app.services.ingest.upload_content_identity import content_identity
from app.services.ingest.upload_recovery import UploadStageRunner
from app.services.source_io_service import run_upload_file_io

INDEX_FORMAT_VERSION = 3
TEXT_BYTE_LIMIT = 8192
FULL_TEXT_BYTE_LIMIT = 65535
INDEX_STRING_LIMITS = {"section": 512, "block_type": 64, "section_path": 1024, "asset_ref": 512,
    "parser_mode": 32, "authors": 2048, "venue": 512, "category": 1024, "file_type": 64}


def embedding_identity() -> str:
    """返回共享配置的向量身份；无参数，上传缓存与发布核对使用同一指纹。"""
    return get_embedding_identity(get_settings())


def _clip_utf8(value: str, limit: int) -> str:
    """为索引预览按实际 UTF-8 字节限长，完整内容仍保存在独立正文产物。"""
    encoded = value.encode("utf-8")
    return value if len(encoded) <= limit else encoded[:limit].decode("utf-8", errors="ignore")


def index_texts(chunks: list[dict[str, Any]]) -> list[str]:
    """返回实际用于 BM25 和向量编码的相同文本，不让编码与存储正文错位。"""
    texts = [_clip_utf8(str(chunk.get("embedding_text") or chunk["text"]), TEXT_BYTE_LIMIT) for chunk in chunks]
    if not texts or any(not text.strip() for text in texts):
        raise ArtifactIntegrityError("索引载荷包含空分块")
    return texts


def normalize_vectors(vectors: Any, *, count: int, dimension: int) -> list[list[float]]:
    """核对完整向量数量、维度和有限数值，统一到 Milvus 的 float32 精度。"""
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ArtifactIntegrityError("向量数量与全部分块不一致")
    normalized = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimension:
            raise ArtifactIntegrityError("向量维度与现役模型不一致")
        if any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in vector):
            raise ArtifactIntegrityError("向量包含非有限数字")
        normalized.append(list(struct.unpack(f"<{dimension}f", struct.pack(f"<{dimension}f", *vector))))
    return normalized


def version_chunks(chunks: list[dict[str, Any]], *, batch_id: str, paper_id: str) -> list[dict[str, Any]]:
    """生成批次独立的 chunk/parent 身份，保留原始身份供来源追溯。"""
    original_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(original_ids)) != len(chunks):
        raise ArtifactIntegrityError("正文出现重复 chunk ID，不能完整发布")
    return [{**chunk, "paper_id": paper_id, "batch_id": batch_id, "source_chunk_id": chunk["chunk_id"],
        "source_parent_id": chunk["parent_id"], "chunk_id": f"v3:{batch_id}:{index}",
        "parent_id": f"v3:{batch_id}:p:{fingerprint(chunk['parent_id'])[:24]}"} for index, chunk in enumerate(chunks)]


def build_index_rows(chunks: list[dict[str, Any]], vectors: list[list[float]], *, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """构造有限长索引载荷；预览字段不充当永久全文来源。"""
    rows = []
    for chunk, text, vector in zip(chunks, index_texts(chunks), vectors, strict=True):
        row = {"chunk_id": chunk["chunk_id"], "paper_id": chunk["paper_id"], "batch_id": chunk["batch_id"],
            "parent_id": chunk["parent_id"], "embedding_text": text,
            "display_text": _clip_utf8(str(chunk.get("display_text") or chunk["text"]), TEXT_BYTE_LIMIT),
            "full_text": _clip_utf8(chunk["full_text"], FULL_TEXT_BYTE_LIMIT), "page_num": int(chunk["page_num"]),
            "year": int(metadata["year"]), "dense_vector": vector}
        for name, limit in INDEX_STRING_LIMITS.items():
            row[name] = _clip_utf8(str(metadata.get(name, chunk.get(name, ""))), limit)
        rows.append(row)
    return rows


def verify_index_rows(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> str:
    """核对每个 ID、每个标量载荷及向量；同数量的错误内容也必须失败。"""
    expected_by_id = {row["chunk_id"]: row for row in expected}
    actual_by_id = {row["chunk_id"]: row for row in actual}
    if len(expected_by_id) != len(expected) or len(actual_by_id) != len(actual) or expected_by_id.keys() != actual_by_id.keys():
        raise ArtifactIntegrityError("索引完整 chunk ID 集合与预期不一致")
    for chunk_id, row in expected_by_id.items():
        stored = {key: actual_by_id[chunk_id].get(key) for key in row}
        vector = stored["dense_vector"]
        to_list = getattr(vector, "tolist", None)
        if callable(to_list):
            vector = to_list()
        stored["dense_vector"] = normalize_vectors([vector], count=1, dimension=len(row["dense_vector"]))[0]
        if fingerprint(row) != fingerprint(stored):
            raise ArtifactIntegrityError(f"索引实际载荷与核验清单不同: {chunk_id}")
    return fingerprint([expected_by_id[key] for key in sorted(expected_by_id)])


@dataclass(frozen=True, slots=True, kw_only=True)
class UploadBatchIndexer:
    """用注入的编码和索引端口建立私有批次，完成后返回发布证据。"""

    runner: UploadStageRunner
    store: Any
    encoder: Any
    publication: Any
    root: Path

    async def _encode(self, texts: list[str]) -> dict[str, Any]:
        """成功编码后立即保存全部向量，索引失败或换 worker 时可以复用。"""
        vectors = await asyncio.to_thread(self.encoder.embed_texts, texts)
        vectors = normalize_vectors(vectors, count=len(texts), dimension=get_settings().embedding_dimension)
        directory = self.root / "uploads" / self.runner.execution.paper_id / "vectors"
        reference = await run_upload_file_io(self.runner.execution, write_json_artifact, directory, {"vectors": vectors, "texts_hash": fingerprint(texts)})
        return {"vectors_ref": reference}

    async def prepare(self, parsed: dict[str, Any], *, paper: dict[str, Any], metadata_missing: list[str]) -> dict[str, Any]:
        """【上传】【新索引批次】复用核验向量，私有写入并逐项回读，通过才返回可发布清单。"""
        execution = self.runner.execution
        document = parsed["document"]
        from app.services.upload_read_service import validate_content_schema

        # 1. 【上传】【正文来源】发布前核对完整来源；旧文档缺少内容版本时仍保持旧格式
        if "content_schema_version" in document:
            validate_content_schema(document)
        texts = index_texts(document["chunks"])
        identity = embedding_identity()
        encoded = await self.runner.run("embeddings", partial(self._encode, texts), inputs={"model": identity, "texts": texts})
        saved = await run_upload_file_io(self.runner.execution, read_json_artifact, encoded["vectors_ref"], root=self.root)
        if saved["texts_hash"] != fingerprint(texts):
            raise ArtifactIntegrityError("已保存向量对应的正文不一致")
        vectors = normalize_vectors(saved["vectors"], count=len(texts), dimension=get_settings().embedding_dimension)
        chunks = version_chunks(document["chunks"], batch_id=execution.batch_id, paper_id=execution.paper_id)
        metadata = build_upload_metadata(paper, document["file_type"])
        rows = build_index_rows(chunks, vectors, metadata=metadata)
        directory = self.root / "uploads" / execution.paper_id / "batches" / execution.batch_id
        index_ref = await run_upload_file_io(self.runner.execution, write_json_artifact, directory, rows)
        manifest = {"format": INDEX_FORMAT_VERSION, "batch_id": execution.batch_id, "paper_id": execution.paper_id,
            "user_id": execution.user_id, "paper": paper, "metadata_missing": metadata_missing, "chunks": chunks,
            "source_ref": parsed["source_ref"], "document_ref": parsed["document_ref"], "resources": parsed["resources"],
            "index_ref": index_ref, "embedding_fingerprint": identity, "collection_name": self.store.collection_name}
        manifest["content_identity"] = content_identity(document, source_ref=parsed["source_ref"], document_ref=parsed["document_ref"])
        if "content_schema_version" in document:
            manifest.update(content_schema_version=document["content_schema_version"], source_items=document["source_items"])
        manifest["payload_sha256"] = fingerprint(sorted(rows, key=lambda row: row["chunk_id"]))
        manifest_ref = await run_upload_file_io(self.runner.execution, write_json_artifact, directory, manifest)
        evidence = {"manifest_ref": manifest_ref, "embedding_fingerprint": identity, "collection_name": self.store.collection_name,
            "chunk_count": len(rows), "payload_sha256": manifest["payload_sha256"]}
        await self.publication.stage_batch(execution, evidence)
        written = await self.runner.run("index_" + execution.batch_id, partial(self._write, rows),
            inputs=manifest_ref["sha256"], budget_stage="index")
        return {**evidence, "payload_sha256": written["payload_sha256"]}

    async def _write(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """只写入当前独立批次；失败不删除此前已发布批次。"""
        await self.runner.repository.check_execution(self.runner.execution)
        actual = await asyncio.to_thread(self.store.write_upload_batch, self.runner.execution.batch_id, rows)
        return {"payload_sha256": verify_index_rows(rows, actual)}
