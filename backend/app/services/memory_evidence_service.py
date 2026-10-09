"""【研究记忆】【原文核验】记忆引用的精确定位与当前来源校验，作者：xxx。"""
from __future__ import annotations

from typing import Any
import hashlib

from app.repositories.memory_source_repository import read_source_state
from app.repositories.source_access_repository import get_source_access_repository
from app.repositories.upload_publication_repository import get_upload_publication_repository
from app.services.source_io_service import run_source_io
from app.services.upload_read_service import _load_manifest, _validate_publication, batch_id_from_chunk


class MemoryEvidenceError(ValueError):
    """原文支持不成立或定位有歧义；候选坐标由实际原文生成。"""

    def __init__(self, message: str, *, candidates: list[dict[str, Any]] | None = None) -> None:
        """参数为原因和可选定位候选；保存供 API 展示的明确错误信息。"""
        super().__init__(message)
        self.candidates = candidates or []


async def _manifest(batch_id: str, *, paper_id: str, user_id: str) -> dict[str, Any]:
    """参数为固定批次和论文归属；文件占用内重验权限，返回原版本完整清单。"""
    async with get_source_access_repository().file_access((paper_id,)) as connection:
        record = await get_upload_publication_repository().content_reference(batch_id, user_id=user_id, connection=connection)
        if record["paper_id"] != paper_id:
            raise MemoryEvidenceError("引用分块不属于所选论文")
        manifest = await run_source_io(_load_manifest, record)
    _validate_publication(manifest, batch_id=batch_id, user_id=user_id, snapshot={"paper_batches": {paper_id: batch_id}})
    return manifest


def _same_content(original: dict[str, Any], current: dict[str, Any]) -> bool:
    """参数为已核验的历史和当前清单；只按可靠原文件/正文身份判等，不用向量行存在性。"""
    if original.get("batch_id") == current.get("batch_id"):
        return True
    old_source = (original.get("source_ref") or {}).get("sha256")
    new_source = (current.get("source_ref") or {}).get("sha256")
    if not old_source or old_source != new_source:
        return False
    old_doc = (original.get("document_ref") or {}).get("sha256")
    new_doc = (current.get("document_ref") or {}).get("sha256")
    if old_doc and old_doc == new_doc:
        return True
    old_identity, new_identity = original.get("content_identity") or {}, current.get("content_identity") or {}
    return bool(old_identity.get("version") == new_identity.get("version") == 1
                and old_identity.get("normalized_sha256")
                and old_identity["normalized_sha256"] == new_identity.get("normalized_sha256"))


def _locate_quote(manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """参数为真实清单与待核对引用；返回唯一原文区间，有歧义则返回可选择的真实候选。"""
    quote = evidence.get("quote_text") or evidence.get("quote") or ""
    if not isinstance(quote, str) or not quote.strip():
        raise MemoryEvidenceError("请选择需要支持判断的原文片段")
    field = evidence.get("source_field") or (evidence.get("verification") or {}).get("source_field") or "full_text"
    if field not in {"text", "full_text"}:
        raise MemoryEvidenceError("原文读取字段无效")
    matches, seen = [], set()
    for chunk in manifest["chunks"]:
        if evidence.get("chunk_id") and chunk["chunk_id"] != evidence["chunk_id"]:
            continue
        text = chunk.get(field) or ""
        start = text.find(quote)
        while start >= 0:
            identity = (chunk.get("parent_id") if field == "full_text" else chunk["chunk_id"], start)
            if identity not in seen:
                matches.append({"paper_id": chunk["paper_id"], "chunk_id": chunk["chunk_id"],
                    "text": quote, "page_num": chunk.get("page_num"), "section": chunk.get("section_path") or chunk.get("section"),
                    "source_field": field, "source_start": start, "source_end": start + len(quote)})
                seen.add(identity)
            start = text.find(quote, start + 1)
    selected_start = evidence.get("source_start", (evidence.get("verification") or {}).get("source_start"))
    if selected_start is not None:
        matches = [match for match in matches if match["source_start"] == selected_start]
    selected_end = evidence.get("source_end", (evidence.get("verification") or {}).get("source_end"))
    if selected_end is not None:
        matches = [match for match in matches if match["source_end"] == selected_end]
    if len(matches) != 1:
        raise MemoryEvidenceError("原句存在多个位置，请选择具体片段" if matches else "所选论文原文中找不到该片段", candidates=matches)
    return matches[0]


async def validate_memory_evidence(evidence: dict[str, Any], *, user_id: str, project_id: str) -> dict[str, Any]:
    """参数为引用与可信项目归属；返回精确原句、不可变版本及本次核验事实，不隐式降级。"""
    paper_id = str(evidence.get("paper_id") or "")
    if not paper_id or not user_id or not project_id:
        raise MemoryEvidenceError("原文核验需要明确用户、项目和论文")
    current = await read_source_state(user_id=user_id, project_id=project_id, paper_id=paper_id)
    batch_id = batch_id_from_chunk(evidence["chunk_id"]) if evidence.get("chunk_id") else current["current_batch_id"]
    original = await _manifest(batch_id, paper_id=paper_id, user_id=user_id)
    if batch_id != current["current_batch_id"]:
        active = await _manifest(current["current_batch_id"], paper_id=paper_id, user_id=user_id)
        if not _same_content(original, active):
            raise MemoryEvidenceError("来源版本已变化或缺少可靠比较依据，请重新绑定当前原文")
    location = _locate_quote(original, evidence)
    quote = location["text"]
    verification = {"version": 1, "batch_id": batch_id, "current_batch_id": current["current_batch_id"],
        "source_field": location["source_field"], "source_start": location["source_start"], "source_end": location["source_end"],
        "checked_now": True}
    return {**evidence, "paper_id": paper_id, "chunk_id": location["chunk_id"], "quote_text": quote,
        "quote_hash": hashlib.sha256(quote.encode("utf-8")).hexdigest(), "page_num": location["page_num"],
        "section": location["section"], "ref_role": evidence.get("ref_role") or "primary_support",
        "source_ref": original.get("source_ref") or {}, "document_ref": original.get("document_ref") or {},
        "verification": verification, "support": evidence.get("support") or {}}
