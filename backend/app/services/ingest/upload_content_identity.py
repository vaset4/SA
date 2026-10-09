"""来源产物的确定性正文身份与版本比较，作者：xxx。"""
from __future__ import annotations

from typing import Any, Literal

from app.models.source_access_models import SourceVersionChange
from app.models.upload_models import ArtifactIntegrityError, UploadExecution
from app.services.ingest.upload_artifacts import fingerprint

CONTENT_IDENTITY_VERSION = 1
SOURCE_CONTENT_FIELDS = ("source_kind", "source_index", "text", "block_type", "page_num", "region", "source_start", "source_end")


def content_identity(document: dict[str, Any], *, source_ref: dict[str, Any],
                     document_ref: dict[str, Any]) -> dict[str, Any]:
    """参数为已核验正文及产物引用；返回排除元信息、索引配置与文件路径的正文身份。"""
    if not isinstance(document.get("full_text"), str) or not document["full_text"].strip():
        raise ArtifactIntegrityError("正文身份缺少实际解析文本")
    source_items = document.get("source_items")
    normalized = {"full_text": document["full_text"], "content_schema_version": document.get("content_schema_version")}
    if source_items is not None:
        normalized["source_items"] = {key: {name: item[name] for name in SOURCE_CONTENT_FIELDS if name in item}
                                      for key, item in source_items.items()}
    return {"version": CONTENT_IDENTITY_VERSION, "source_sha256": source_ref["sha256"],
        "document_sha256": document_ref["sha256"], "normalized_sha256": fingerprint(normalized)}


def compare_source_versions(execution: UploadExecution, *, manifest: dict[str, Any], manifest_sha256: str,
                            previous: dict[str, Any] | None, previous_manifest_sha256: str | None) -> SourceVersionChange:
    """参数为执行身份与已核验的前后清单；返回绑定旧指针的版本事实，旧资料不足明确 unknown。"""
    current_identity = manifest.get("content_identity") or {}
    source_sha = manifest["source_ref"]["sha256"]
    document_sha = manifest["document_ref"]["sha256"]
    normalized_sha = current_identity.get("normalized_sha256")
    classification: Literal["initial", "unchanged", "changed", "unknown"] = "initial"
    if previous is not None:
        previous_identity = previous.get("content_identity") or {}
        old_source = (previous.get("source_ref") or {}).get("sha256")
        old_document = (previous.get("document_ref") or {}).get("sha256")
        same_normalized = (current_identity.get("version") == CONTENT_IDENTITY_VERSION
            and previous_identity.get("version") == CONTENT_IDENTITY_VERSION
            and normalized_sha and previous_identity.get("normalized_sha256"))
        classification = "unknown"
        if old_source and old_source != source_sha:
            classification = "changed"
        elif old_source == source_sha and old_document == document_sha:
            classification = "unchanged"
        elif old_source == source_sha and same_normalized:
            classification = "unchanged" if normalized_sha == previous_identity["normalized_sha256"] else "changed"
    return SourceVersionChange(paper_id=execution.paper_id, user_id=execution.user_id,
        previous_batch_id=execution.previous_batch_id, batch_id=execution.batch_id,
        previous_manifest_sha256=previous_manifest_sha256, manifest_sha256=manifest_sha256,
        classification=classification, source_sha256=source_sha, document_sha256=document_sha,
        normalized_sha256=normalized_sha)
