"""Operational mixins for LightRAG service.

The main service owns runtime creation. These mixins keep insert/query and
local maintenance behavior out of the runtime bootstrap module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time
from typing import Any

from app.config import get_settings
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.services.lightrag_domain_prompt import lint_lightrag_entities_file
from app.services.lightrag_insert_status import validate_lightrag_doc_processed

settings = get_settings()
logger = logging.getLogger(__name__)
PAPER_ID_PATTERN = re.compile(r"\[Paper ID:\s*([^\]\n]+)\]")


def extract_paper_id_from_payload(payload: Any) -> str:
    """Extract original paper_id from a LightRAG document payload."""
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, dict):
        text = str(payload.get("content") or payload.get("content_summary") or "")
    else:
        text = str(payload or "")
    match = PAPER_ID_PATTERN.search(text)
    return match.group(1).strip() if match else ""


class LightRAGInsertMixin:
    async def insert(self, text: str, paper_id: str = "", *, library_id: str | None = None) -> None:
        """Insert long-form paper text into LightRAG."""
        self._ensure_enabled()
        runtime = self._get_runtime(library_id)
        await self.initialize(runtime.library_id)
        if not text.strip():
            raise ValueError("LightRAG insert 失败：文本内容为空")
        tagged_text = _tag_text(runtime.library_id, text, paper_id)
        if runtime.insert_semaphore.locked():
            logger.info(
                "LightRAG insert waiting for slot: library=%s paper_id=%s chars=%d",
                runtime.library_id,
                paper_id or "-",
                len(text),
            )
        started_at = time.monotonic()
        async with runtime.insert_semaphore:
            await _insert_tagged_text(runtime, tagged_text, paper_id, text, started_at)


def _tag_text(library_id: str, text: str, paper_id: str) -> str:
    if paper_id:
        return f"[Library ID: {library_id}]\n[Paper ID: {paper_id}]\n\n{text}"
    return f"[Library ID: {library_id}]\n\n{text}"


async def _insert_tagged_text(runtime: Any, tagged_text: str, paper_id: str, text: str, started_at: float) -> None:
    logger.info(
        "LightRAG insert start: library=%s paper_id=%s chars=%d",
        runtime.library_id,
        paper_id or "-",
        len(text),
    )
    try:
        if paper_id:
            track_id = await runtime.rag.ainsert(tagged_text, ids=[paper_id])
            await validate_lightrag_doc_processed(runtime, paper_id, str(track_id or ""))
        else:
            await runtime.rag.ainsert(tagged_text)
    except Exception:
        logger.exception("LightRAG insert failed: library=%s paper_id=%s chars=%d", runtime.library_id, paper_id or "-", len(text))
        raise
    logger.info(
        "LightRAG insert done: library=%s paper_id=%s chars=%d duration_ms=%d",
        runtime.library_id,
        paper_id or "-",
        len(text),
        int((time.monotonic() - started_at) * 1000),
    )


class LightRAGQueryMixin:
    async def query(self, query: str, mode: str = "hybrid", top_k: int = 5, *, library_id: str | None = None) -> str:
        """Run a LightRAG query and return aggregated text."""
        self._ensure_enabled()
        runtime = self._get_runtime(library_id)
        await self.initialize(runtime.library_id)
        if not query.strip():
            raise ValueError("LightRAG query 失败：查询内容为空")
        params, graph_rerank_controlled = _build_query_params(mode, top_k)
        started_at = time.monotonic()
        logger.info(
            "LightRAG query start: library=%s mode=%s top_k=%d graph_rerank_disabled=%s graph_rerank_controlled=%s query=%s",
            runtime.library_id,
            mode,
            top_k,
            not bool(getattr(settings, "graph_rerank_enabled", False)),
            graph_rerank_controlled,
            query[:80],
        )
        result = await runtime.rag.aquery(query, param=params)
        _log_query_done(runtime.library_id, mode, top_k, graph_rerank_controlled, started_at)
        return str(result)

    async def collect_library_graph_hint(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        *,
        library_id: str | None = None,
    ) -> str:
        """Call LightRAG as the library-scoped graph hint layer."""
        return await self.query(query=query, mode=mode, top_k=top_k, library_id=library_id)


def _build_query_params(mode: str, top_k: int) -> tuple[Any, bool]:
    from app.services.lightrag_service import QueryParam

    graph_rerank_enabled = bool(getattr(settings, "graph_rerank_enabled", False))
    try:
        return QueryParam(mode=mode, top_k=top_k, enable_rerank=graph_rerank_enabled), True
    except TypeError:
        return QueryParam(mode=mode, top_k=top_k), False


def _log_query_done(library_id: str, mode: str, top_k: int, graph_rerank_controlled: bool, started_at: float) -> None:
    logger.info(
        "LightRAG query done: library=%s mode=%s top_k=%d graph_rerank_disabled=%s graph_rerank_controlled=%s duration_ms=%d",
        library_id,
        mode,
        top_k,
        not bool(getattr(settings, "graph_rerank_enabled", False)),
        graph_rerank_controlled,
        int((time.monotonic() - started_at) * 1000),
    )


class LightRAGMaintenanceMixin:
    @staticmethod
    def _serialize_delete_result(result: Any, paper_id: str) -> dict[str, Any]:
        """Convert third-party delete result to a plain dict."""
        if isinstance(result, dict):
            payload = dict(result)
        else:
            payload = _object_to_payload(result)
        payload.setdefault("status", "success")
        payload.setdefault("doc_id", paper_id)
        return payload

    async def delete_by_doc_id(self, paper_id: str, *, library_id: str | None = None, delete_llm_cache: bool = True) -> dict[str, Any]:
        """Delete one LightRAG document by stable paper_id/doc_id."""
        self._ensure_enabled()
        runtime = self._get_runtime(library_id)
        await self.initialize(runtime.library_id)
        if not paper_id.strip():
            raise ValueError("LightRAG delete 失败：paper_id 为空")
        result = await runtime.rag.adelete_by_doc_id(paper_id, delete_llm_cache=delete_llm_cache)
        return self._serialize_delete_result(result, paper_id)

    def list_document_records(self, *, library_id: str | None = None) -> list[dict[str, str]]:
        """Read LightRAG document keys and their embedded paper_id."""
        docs_path = self._get_runtime(library_id).working_dir / "kv_store_full_docs.json"
        payload = _read_json_dict(docs_path, "读取 LightRAG 文档索引失败")
        return [{"doc_id": str(doc_id or "").strip(), "paper_id": extract_paper_id_from_payload(item)} for doc_id, item in payload.items()]

    def prune_llm_cache_by_paper_ids(self, paper_ids: set[str], *, library_id: str | None = None) -> dict[str, Any]:
        """Delete cache entries that still reference specific paper IDs."""
        cache_path = self._get_runtime(library_id).working_dir / "kv_store_llm_response_cache.json"
        if not cache_path.exists() or not paper_ids:
            return {"cache_entry_count": 0, "deleted_cache_keys": []}
        payload = _read_json_dict(cache_path, "读取 LightRAG LLM cache 失败")
        next_payload, deleted_keys = _drop_cache_entries(payload, paper_ids)
        _write_json_if_changed(cache_path, next_payload, deleted_keys)
        return {"cache_entry_count": len(payload), "deleted_cache_keys": deleted_keys}

    def reconcile_orphan_llm_cache(self, valid_paper_ids: set[str], *, library_id: str | None = None) -> dict[str, Any]:
        """Delete cache entries referencing paper IDs no longer in the business DB."""
        cache_path = self._get_runtime(library_id).working_dir / "kv_store_llm_response_cache.json"
        if not cache_path.exists():
            return {"cache_entry_count": 0, "deleted_cache_keys": [], "deleted_paper_ids": []}
        payload = _read_json_dict(cache_path, "读取 LightRAG LLM cache 失败")
        next_payload, deleted_keys, deleted_paper_ids = _drop_orphan_cache_entries(payload, valid_paper_ids)
        _write_json_if_changed(cache_path, next_payload, deleted_keys)
        return {"cache_entry_count": len(payload), "deleted_cache_keys": deleted_keys, "deleted_paper_ids": sorted(deleted_paper_ids)}

    async def reconcile_orphan_documents(self, valid_paper_ids: set[str], *, library_id: str | None = None) -> dict[str, Any]:
        """Delete LightRAG documents whose paper IDs no longer exist."""
        records = self.list_document_records(library_id=library_id)
        orphan_records = [item for item in records if item.get("doc_id") and item.get("paper_id") and item["paper_id"] not in valid_paper_ids]
        deleted = [await self.delete_by_doc_id(item["doc_id"], library_id=library_id) for item in orphan_records]
        cache_cleanup = self.reconcile_orphan_llm_cache(valid_paper_ids, library_id=library_id)
        return _orphan_document_result(library_id, records, orphan_records, deleted, cache_cleanup)

    async def get_stats(self, *, library_id: str | None = None) -> dict[str, Any]:
        """Return LightRAG local working directory status and file stats."""
        if not settings.graph_rag_enabled:
            return {"status": "disabled", "working_dir": str(self._working_root)}
        runtime = self._get_runtime(library_id)
        if not runtime.initialized:
            return {"status": "not_initialized", "library_id": runtime.library_id, "working_dir": str(runtime.working_dir)}
        files = sorted(path.name for path in runtime.working_dir.glob("*") if path.is_file())
        total_size_kb = sum(path.stat().st_size for path in runtime.working_dir.glob("*") if path.is_file()) // 1024
        return {"status": "initialized", "library_id": runtime.library_id, "working_dir": str(runtime.working_dir), "file_count": len(files), "total_size_kb": total_size_kb, "files": files}

    def lint_lightrag_entities(self, *, library_id: str | None = None) -> dict[str, Any]:
        """Remove non-academic entity_type entries from LightRAG full entity file."""
        runtime = self._get_runtime(library_id)
        result = lint_lightrag_entities_file(runtime.working_dir / "kv_store_full_entities.json")
        logger.info("LightRAG entity lint done: library=%s deleted=%d kept=%d by_type=%s", runtime.library_id, result["deleted"], result["kept"], result["by_type"])
        return result


def _object_to_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("status", "message", "doc_id", "status_code", "file_path"):
        value = getattr(result, key, None)
        if value is not None:
            payload[key] = str(value) if isinstance(value, Path) else value
    if not payload and hasattr(result, "__dict__"):
        payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(result).items() if not key.startswith("_")}
    return payload


def _read_json_dict(path: Path, warning: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("%s: %s", warning, path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _drop_cache_entries(payload: dict[str, Any], paper_ids: set[str]) -> tuple[dict[str, Any], list[str]]:
    deleted_keys: list[str] = []
    next_payload: dict[str, Any] = {}
    for key, value in payload.items():
        serialized = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        if any(paper_id in serialized for paper_id in paper_ids):
            deleted_keys.append(str(key))
            continue
        next_payload[str(key)] = value
    return next_payload, deleted_keys


def _drop_orphan_cache_entries(payload: dict[str, Any], valid_paper_ids: set[str]) -> tuple[dict[str, Any], list[str], set[str]]:
    deleted_keys: list[str] = []
    deleted_paper_ids: set[str] = set()
    next_payload: dict[str, Any] = {}
    for key, value in payload.items():
        serialized = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        matched = {item.strip() for item in PAPER_ID_PATTERN.findall(serialized) if item.strip()}
        stale = {paper_id for paper_id in matched if paper_id not in valid_paper_ids}
        if stale:
            deleted_keys.append(str(key))
            deleted_paper_ids.update(stale)
            continue
        next_payload[str(key)] = value
    return next_payload, deleted_keys, deleted_paper_ids


def _write_json_if_changed(path: Path, payload: dict[str, Any], deleted_keys: list[str]) -> None:
    if deleted_keys:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _orphan_document_result(
    library_id: str | None,
    records: list[dict[str, str]],
    orphan_records: list[dict[str, str]],
    deleted: list[dict[str, Any]],
    cache_cleanup: dict[str, Any],
) -> dict[str, Any]:
    return {
        "library_id": (library_id or DEFAULT_LIBRARY_ID),
        "document_count": len(records),
        "orphan_count": len(orphan_records),
        "deleted_doc_ids": [item.get("doc_id") for item in orphan_records],
        "deleted_paper_ids": [item.get("paper_id") for item in orphan_records],
        "results": deleted,
        "cache_cleanup": cache_cleanup,
    }
