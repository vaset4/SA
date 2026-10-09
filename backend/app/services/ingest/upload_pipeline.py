"""持久上传、独立补全和原子发布编排，作者：xxx。"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from functools import partial
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.upload_models import UploadCancelledError, UploadExecution, UploadFailure, UploadLeaseLostError, UploadPausedError
from app.repositories.upload_publication_repository import get_upload_publication_repository
from app.repositories.upload_repository import get_upload_repository
from app.services.ingest import document_analysis_service, document_graph_service, upload_metadata_service
from app.services.ingest.mineru_client import MinerUClient
from app.services.ingest.upload_artifacts import read_json_artifact, verify_artifact, write_json_artifact
from app.services.ingest.upload_index_service import UploadBatchIndexer
from app.services.ingest.upload_parse_service import UploadDocumentParser
from app.services.ingest.upload_content_identity import compare_source_versions, content_identity
from app.services.ingest.upload_recovery import UploadStageRunner, classify_upload_failure
from app.services.source_io_service import run_upload_file_io
from app.models.source_access_models import SourceVersionChange
from app.models.upload_models import ArtifactIntegrityError

logger = logging.getLogger(__name__)


def get_upload_embedding_client():
    """取得当前上传的编码端口，测试可注入编码器而不加载真实模型。"""
    from app.services.embedding_client import get_embedding_client

    return get_embedding_client()


async def _published_document(execution: UploadExecution, root: Path) -> dict[str, Any]:
    """补全和建图直接读取已发布原文，不再次提交 MinerU 解析。"""
    if not execution.previous_batch_id:
        raise ValueError("论文尚无核验通过的正文")
    record = await get_upload_publication_repository().content_reference(execution.previous_batch_id, user_id=execution.user_id)
    manifest = await run_upload_file_io(execution, read_json_artifact, record["manifest_ref"], root=root)
    document = await run_upload_file_io(execution, read_json_artifact, manifest["document_ref"], root=root)
    await run_upload_file_io(execution, verify_artifact, manifest["source_ref"], root=root)
    return {"document": document, "document_ref": manifest["document_ref"],
        "source_ref": manifest["source_ref"], "resources": manifest["resources"]}


async def _complete_metadata(parsed: dict[str, Any], runner: UploadStageRunner, root: Path) -> dict[str, Any]:
    """元信息调用独立计数，完成后保存新的文档产物以便原子同步索引和引用。"""
    document = parsed["document"]
    client = upload_metadata_service.metadata_client()
    metadata = await asyncio.to_thread(upload_metadata_service.complete_metadata,
        document["full_text"], document["metadata"], client=client)
    updated = {**document, "metadata": metadata}
    directory = root / "uploads" / runner.execution.paper_id / "metadata"
    reference = await run_upload_file_io(runner.execution, write_json_artifact, directory, updated)
    return {"document_ref": reference}


async def _build_graph(parsed: dict[str, Any], execution: UploadExecution) -> dict[str, Any]:
    """只在用户请求图谱时调用增强服务；失败不触碰已发布正文与向量。"""
    if not get_settings().graph_rag_enabled:
        raise UploadPausedError(UploadFailure("graph", "not_enabled", "图谱功能尚未启用，论文检索仍可正常使用"))
    document = parsed["document"]
    record = await get_upload_repository().get_document(execution.paper_id, user_id=execution.user_id)
    if record is None:
        raise LookupError("上传来源不存在")
    await document_graph_service.insert_to_lightrag(execution.paper_id, document["full_text"],
        library_id=record["library_id"], chunks=document["chunks"], metadata=document["metadata"])
    return {"paper_id": execution.paper_id, "graph_status": "completed"}


async def _prepare_result(runner: UploadStageRunner, source: dict[str, Any], root: Path) -> dict[str, Any]:
    """按操作类型复用原文或继续解析，再建立与旧版本隔离的索引批次。"""
    from app.services.milvus_store import get_milvus_store

    execution = runner.execution
    await runner.repository.progress(execution, stage="parse", percent=15, message="读取已保存原文与处理进度")
    if execution.operation in {"metadata", "graph"}:
        parsed = await _published_document(execution, root)
    else:
        parser = UploadDocumentParser(runner=runner, client=MinerUClient(), source=source["source_ref"], filename=source["filename"], root=root)
        parsed = {**(await parser.parse()), "source_ref": source["source_ref"]}
    if execution.operation == "graph":
        result = await runner.run("graph_" + execution.task_id, partial(_build_graph, parsed, execution), inputs=parsed["document_ref"])
        return {"_upload_graph": {"execution": asdict(execution), "result": result}}
    if execution.operation == "metadata":
        reference = await runner.run("metadata_" + execution.task_id, partial(_complete_metadata, parsed, runner, root), inputs=parsed["document_ref"])
        parsed = {**parsed, **reference, "document": await run_upload_file_io(execution, read_json_artifact, reference["document_ref"], root=root)}
    paper = {**document_analysis_service.build_upload_paper(source["filename"], parsed["document"], execution.paper_id),
        "user_id": execution.user_id, "library_id": source["library_id"]}
    await runner.repository.progress(execution, stage="index", percent=60, message="建立独立索引并核验完整内容")
    indexer = UploadBatchIndexer(runner=runner, store=get_milvus_store(), encoder=get_upload_embedding_client(),
        publication=get_upload_publication_repository(), root=root)
    evidence = await indexer.prepare(parsed, paper=paper, metadata_missing=upload_metadata_service.missing_metadata(parsed["document"]["metadata"]))
    await get_upload_publication_repository().verify_batch(execution, evidence)
    return {"_upload_publication": {"execution": asdict(execution), **evidence}}


async def prepare_upload_task(task: dict[str, Any]) -> dict[str, Any]:
    """【上传】【任务执行】持久领取独立执行身份，任何失败都保留来源和真实阶段。"""
    repository = get_upload_repository()
    execution = await repository.begin_attempt(task)
    root = Path(get_settings().data_dir).resolve()
    try:
        source = await repository.get_document(execution.paper_id, user_id=execution.user_id)
        if source is None:
            raise LookupError("上传原文件记录不存在")
        return await _prepare_result(UploadStageRunner(repository, execution), source, root)
    except (UploadLeaseLostError, UploadCancelledError):
        raise
    except UploadPausedError as error:
        raise UploadPausedError(error.failure, execution=execution) from error
    except Exception as error:
        failure = classify_upload_failure(error, execution.operation)
        raise UploadPausedError(failure, execution=execution) from error


async def commit_upload_result(prepared: dict[str, Any], *, task_id: str, worker_id: str) -> dict[str, Any]:
    """【上传】【完成提交】由 worker 完成边界提交发布，成功后再处理旧索引清理。"""
    graph = prepared.get("_upload_graph")
    payload = graph or prepared["_upload_publication"]
    execution = UploadExecution(**payload["execution"])
    if execution.task_id != task_id or execution.worker_id != worker_id:
        raise UploadLeaseLostError("准备发布的执行身份与当前 worker 不一致")
    publication = get_upload_publication_repository()
    if graph:
        task = await publication.finish_graph(execution, graph["result"])
    else:
        root = Path(get_settings().data_dir).resolve()
        manifest = await run_upload_file_io(execution, read_json_artifact, payload["manifest_ref"], root=root)
        change = await _prepare_source_change(execution, manifest, payload["manifest_ref"], root=root)
        task = await publication.publish(execution, manifest, source_change=change)
    await cleanup_retired_batches(execution.paper_id, user_id=execution.user_id)
    return task


async def _prepare_source_change(execution: UploadExecution, manifest: dict[str, Any],
                                  reference: dict[str, Any], *, root: Path) -> SourceVersionChange:
    """参数为上传执行和新清单；事务前核验前后原文身份，返回绑定原发布指针的比较事实。"""
    await run_upload_file_io(execution, verify_artifact, manifest["source_ref"], root=root)
    document = await run_upload_file_io(execution, read_json_artifact, manifest["document_ref"], root=root)
    expected = content_identity(document, source_ref=manifest["source_ref"], document_ref=manifest["document_ref"])
    if "content_identity" in manifest and manifest["content_identity"] != expected:
        raise ArtifactIntegrityError("发布正文身份与实际产物不一致")
    previous, previous_sha = None, None
    if execution.previous_batch_id:
        record = await get_upload_publication_repository().content_reference(execution.previous_batch_id, user_id=execution.user_id)
        previous = await run_upload_file_io(execution, read_json_artifact, record["manifest_ref"], root=root)
        previous_sha = record["manifest_ref"]["sha256"]
    return compare_source_versions(execution, manifest=manifest, manifest_sha256=reference["sha256"],
        previous=previous, previous_manifest_sha256=previous_sha)


async def cleanup_retired_batches(paper_id: str, *, user_id: str) -> None:
    """提交成功后清理无读占用的旧向量；错误独立记录，不撤销完成事实。"""
    from app.services.milvus_store import get_milvus_store

    publication = get_upload_publication_repository()
    try:
        candidates = await publication.retire_candidates(paper_id, user_id=user_id)
    except Exception as error:
        logger.error("【上传】【旧批次清理】枚举待清理批次失败 paper_id=%s error=%s", paper_id, error)
        return
    for batch in candidates:
        cleanup_error = ""
        try:
            if batch["collection_name"]:
                store = get_milvus_store()
                if batch["collection_name"] != store.collection_name:
                    raise ValueError("旧批次属于迁移保留集合，应按迁移清单清理")
                await asyncio.to_thread(store.delete_upload_batch, batch["batch_id"])
        except Exception as error:
            logger.error("【上传】【旧批次清理】已发布结果保留，旧向量清理失败 batch_id=%s error=%s", batch["batch_id"], error)
            cleanup_error = str(error)
        try:
            await publication.record_cleanup(batch["batch_id"], cleanup_error)
        except Exception:
            logger.error("【上传】【清理记录】清理结果暂未持久保存 batch_id=%s", batch["batch_id"], exc_info=True)
