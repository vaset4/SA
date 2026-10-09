"""上传受理和用户可见操作入口，作者：xxx。"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
import uuid

from app.config import get_settings
from app.models.upload_models import ArtifactRef, UploadRequest
from app.repositories.upload_repository import get_upload_repository
from app.services import library_service, paper_db, project_service
from app.services.ingest import document_parse_service
from app.services.ingest.upload_artifacts import fingerprint, write_artifact
from app.services.pdf_document import open_pdf

logger = logging.getLogger(__name__)


def _save_source(filename: str, content: bytes, *, paper_id: str) -> ArtifactRef:
    """【上传】【原文件受理】检查格式和 PDF 可读性，原子保存不可变来源。"""
    suffix = document_parse_service.validate_extension(filename)
    document_parse_service.validate_size(content)
    document_parse_service.validate_magic_bytes(suffix, content)
    if not content:
        raise ValueError("上传文件为空")
    if suffix == ".pdf":
        with open_pdf(content) as pdf:
            for index in range(pdf.page_count):
                pdf.page_text(index)
    root = Path(get_settings().data_dir).resolve()
    return write_artifact(root / "uploads" / paper_id / "source", content, suffix=suffix)


async def receive_upload(filename: str, content: bytes, *, user_id: str, request_id: str,
                         library_id: str | None = None, project_id: str | None = None,
                         graph_requested: bool = False) -> dict[str, Any]:
    """保存原文件后原子受理，同请求查回、同账号同库同文件复用原论文。"""
    from app.services.task_service import require_upload_owner

    owner = require_upload_owner(user_id)
    request_id = str(uuid.UUID(request_id))
    await asyncio.to_thread(paper_db.init_db)
    if project_id:
        project = await project_service.require_project(project_id, user_id=owner)
        library_id = library_id or project["default_library_id"]
    library = await library_service.require_library(library_id, owner=owner)
    paper_id = "upload_" + uuid.uuid4().hex
    source = await asyncio.to_thread(_save_source, filename, content, paper_id=paper_id)
    identity = fingerprint({"sha256": source["sha256"], "size": source["size"], "filename": filename,
        "library_id": library["library_id"], "project_id": project_id, "graph_requested": graph_requested})
    request = UploadRequest(owner, request_id, identity, paper_id, str(uuid.uuid4()), uuid.uuid4().hex,
        library["library_id"], project_id, filename, source, graph_requested)
    repository = get_upload_repository()
    task = await repository.accept(request)
    if task["payload"]["paper_id"] != paper_id:
        await asyncio.to_thread(_remove_unused_source, source, paper_id=paper_id)
    return await repository.describe_task(task)


def _remove_unused_source(source: ArtifactRef, *, paper_id: str) -> None:
    """只清理并发去重后本次未被采用的候选文件，受理结果不依赖清理成功。"""
    root = (Path(get_settings().data_dir) / "uploads" / paper_id).resolve()
    path = Path(source["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("待清理候选文件不属于本次上传")
    try:
        path.unlink()
        path.parent.rmdir()
        root.rmdir()
    except OSError as error:
        logger.error("【上传】【候选文件清理】已复用原论文，未采用的候选文件清理失败 paper_id=%s error=%s", paper_id, error)


async def lookup_upload_request(request_id: str, *, user_id: str) -> dict[str, Any]:
    """响应丢失或页面重载后查回原受理结果，未知请求返回不存在。"""
    repository = get_upload_repository()
    task = await repository.get_request(str(uuid.UUID(request_id)), user_id=user_id)
    if task is None:
        raise LookupError("上传尚未受理，可使用原标识重新发送文件")
    return await repository.describe_task(task)


async def continue_upload(task_id: str, *, user_id: str) -> dict[str, Any]:
    """由用户明确继续原任务，保持原论文身份及成功检查点。"""
    repository = get_upload_repository()
    task = await repository.resume(task_id, user_id=user_id)
    return await repository.describe_task(task)


async def request_upload_operation(paper_id: str, operation: str, *, user_id: str) -> dict[str, Any]:
    """为已保存论文请求重新处理、信息补全或可选图谱；返回独立任务状态。"""
    from typing import cast
    from app.models.upload_models import UploadOperation

    if operation not in {"rebuild", "metadata", "graph"}:
        raise ValueError("不支持的上传操作")
    repository = get_upload_repository()
    task = await repository.request_operation(paper_id, user_id=user_id, operation=cast(UploadOperation, operation))
    return await repository.describe_task(task)


async def get_paper_upload_status(paper_id: str, *, user_id: str) -> dict[str, Any]:
    """返回本账号论文的基础及增强状态，供详情页刷新后恢复操作入口。"""
    from app.repositories.task_repository import get_task_repository

    repository = get_upload_repository()
    document = await repository.get_document(paper_id, user_id=user_id)
    if document is None or document["state"] == "deleted" or document["deletion_status"] not in (None, "active"):
        raise LookupError("上传论文不存在")
    task = await get_task_repository().get_task_run(document["task_id"], user_id=user_id)
    if task is None:
        raise LookupError("原上传任务不存在")
    return await repository.describe_task(task)
