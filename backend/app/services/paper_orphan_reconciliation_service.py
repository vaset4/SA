"""论文跨存储孤儿数据对账服务。

Author: xxx
"""

from __future__ import annotations

import asyncio
from collections import Counter
import logging
from typing import Any

from app.models.library_models import DEFAULT_LIBRARY_ID
from app.repositories.library_repository import get_library_repository
from app.repositories.paper_repository import get_paper_repository
from app.repositories.research_memory_repository import get_research_memory_repository
from app.services.graph import get_neo4j_maintenance_service
from app.services.lightrag_service import get_lightrag_service
from app.services.milvus_store import get_milvus_store
from app.services.paper_file_cleanup_service import (
    delete_managed_orphan_paths,
    list_orphan_managed_paths,
)

logger = logging.getLogger(__name__)


async def _reconcile_milvus(valid_paper_ids: set[str], apply: bool) -> dict[str, Any]:
    """
    对账 Milvus 中的论文主键，并按显式 apply 删除孤儿向量。

    Args:
        valid_paper_ids: 业务库全量论文主键
        apply: 是否执行清理
    Returns:
        Milvus 扫描与清理结果
    """
    store = get_milvus_store()
    indexed_ids = await asyncio.to_thread(store.list_paper_ids)
    orphan_ids = sorted(set(indexed_ids) - valid_paper_ids)
    if apply:
        for paper_id in orphan_ids:
            await asyncio.to_thread(store.delete_paper, paper_id)
        if orphan_ids:
            await asyncio.to_thread(store.flush_collection)
    return {
        "indexed_paper_count": len(set(indexed_ids)),
        "orphan_count": len(orphan_ids),
        "orphan_paper_ids": orphan_ids,
        "deleted_paper_ids": orphan_ids if apply else [],
    }


async def _reconcile_lightrag(
    valid_by_library: dict[str, set[str]],
    library_ids: set[str],
    apply: bool,
) -> dict[str, Any]:
    """
    按 library 对账 LightRAG 文档，并按显式 apply 删除孤儿文档。

    Args:
        valid_by_library: 每个 library 的业务论文主键
        library_ids: 需要扫描的 library 主键
        apply: 是否执行清理
    Returns:
        LightRAG 扫描与清理结果
    """
    service = get_lightrag_service()
    document_count = 0
    orphan_documents: list[dict[str, str]] = []
    for library_id in sorted(library_ids):
        records = await asyncio.to_thread(service.list_document_records, library_id=library_id)
        document_count += len(records)
        for record in records:
            paper_id = str(record.get("paper_id") or "").strip()
            doc_id = str(record.get("doc_id") or "").strip()
            if paper_id and doc_id and paper_id not in valid_by_library.get(library_id, set()):
                orphan_documents.append({"library_id": library_id, "doc_id": doc_id, "paper_id": paper_id})
    if apply:
        for document in orphan_documents:
            await service.delete_by_doc_id(document["doc_id"], library_id=document["library_id"])
    return {
        "library_count": len(library_ids),
        "document_count": document_count,
        "orphan_count": len(orphan_documents),
        "orphan_documents": orphan_documents,
        "deleted_documents": orphan_documents if apply else [],
    }


async def _reconcile_neo4j(valid_paper_ids: set[str], apply: bool) -> dict[str, Any]:
    """
    对账 Neo4j Paper 节点，并按显式 apply 删除孤儿子图。

    Args:
        valid_paper_ids: 业务库全量论文主键
        apply: 是否执行清理
    Returns:
        Neo4j 扫描与清理结果
    """
    service = get_neo4j_maintenance_service()
    indexed_ids = await service.list_paper_ids()
    orphan_ids = sorted(set(indexed_ids) - valid_paper_ids)
    if apply:
        for paper_id in orphan_ids:
            await service.delete_paper_subgraph(paper_id)
    return {
        "paper_node_count": len(set(indexed_ids)),
        "orphan_count": len(orphan_ids),
        "orphan_paper_ids": orphan_ids,
        "deleted_paper_ids": orphan_ids if apply else [],
    }


async def _reconcile_memory_refs(valid_paper_ids: set[str], apply: bool) -> dict[str, Any]:
    """
    对账研究记忆证据引用，并按显式 apply 在事务中清理孤儿引用。

    Args:
        valid_paper_ids: 业务库全量论文主键
        apply: 是否执行清理
    Returns:
        memory evidence 扫描与清理结果
    """
    repository = get_research_memory_repository()
    refs = await repository.list_orphan_evidence_refs(valid_paper_ids)
    apply_result = await repository.remove_orphan_evidence_refs(refs) if apply else {}
    return {
        "orphan_count": len(refs),
        "orphan_refs": [
            {key: ref.get(key) for key in ("ref_id", "memory_id", "paper_id")}
            for ref in refs
        ],
        "deleted_ref_count": int(apply_result.get("deleted_evidence_refs", 0)),
        "updated_memory_ids": apply_result.get("updated_memory_ids", []),
        "stale_memory_ids": apply_result.get("stale_memory_ids", []),
    }


async def _reconcile_files(papers: list[dict[str, Any]], apply: bool) -> dict[str, Any]:
    """
    对账 DATA_DIR/uploads 受控文件，并按显式 apply 删除孤儿路径。

    Args:
        papers: 业务库全量论文生命周期记录
        apply: 是否执行清理
    Returns:
        受控文件扫描与清理结果
    """
    orphan_paths = await asyncio.to_thread(list_orphan_managed_paths, papers)
    deleted_paths = await asyncio.to_thread(delete_managed_orphan_paths, orphan_paths) if apply else []
    return {
        "orphan_count": len(orphan_paths),
        "orphan_paths": orphan_paths,
        "deleted_paths": deleted_paths,
    }


async def reconcile_paper_orphans(*, apply: bool = False) -> dict[str, Any]:
    """
    以 PostgreSQL 论文生命周期记录为真相源，对账五类派生存储。

    Args:
        apply: False 仅报告，True 才执行精确清理
    Returns:
        可序列化的跨存储对账报告
    """
    papers = await get_paper_repository().list_reconciliation_records()
    valid_paper_ids = {str(paper.get("paper_id") or "").strip() for paper in papers}
    valid_paper_ids.discard("")
    valid_by_library: dict[str, set[str]] = {}
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "").strip()
        library_id = str(paper.get("library_id") or DEFAULT_LIBRARY_ID).strip() or DEFAULT_LIBRARY_ID
        if paper_id:
            valid_by_library.setdefault(library_id, set()).add(paper_id)
    libraries = await get_library_repository().list_libraries()
    library_ids = {DEFAULT_LIBRARY_ID, *valid_by_library}
    library_ids.update(str(item.get("library_id") or "").strip() for item in libraries)
    library_ids.discard("")
    logger.info("【论文对账】【执行开始】mode=%s paper_count=%d", "apply" if apply else "dry_run", len(papers))
    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "applied": apply,
        "truth_source": {
            "paper_count": len(valid_paper_ids),
            "deletion_status_counts": dict(Counter(str(item.get("deletion_status") or "active") for item in papers)),
            "deletion_task_status_counts": dict(
                Counter(str(item.get("deletion_task_status")) for item in papers if item.get("deletion_task_status"))
            ),
        },
        "milvus": await _reconcile_milvus(valid_paper_ids, apply),
        "lightrag": await _reconcile_lightrag(valid_by_library, library_ids, apply),
        "neo4j": await _reconcile_neo4j(valid_paper_ids, apply),
        "memory_evidence_refs": await _reconcile_memory_refs(valid_paper_ids, apply),
        "files": await _reconcile_files(papers, apply),
    }
    orphan_count = sum(int(report[key]["orphan_count"]) for key in ("milvus", "lightrag", "neo4j", "files", "memory_evidence_refs"))
    logger.info("【论文对账】【执行完成】mode=%s orphan_count=%d", report["mode"], orphan_count)
    return report
