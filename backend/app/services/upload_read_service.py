"""已发布上传的检索快照与不可变引用正文读取，作者：xxx。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from app.config import get_settings
from app.models.research_contracts import RetrievalScopeError
from app.models.knowledge_access_contracts import CONTENT_SCHEMA_VERSION
from app.models.upload_models import ArtifactIntegrityError
from app.repositories.upload_publication_repository import READ_LEASE_SECONDS, get_upload_publication_repository
from app.repositories.source_access_repository import get_source_access_repository
from app.services import paper_db
from app.services.ingest.upload_artifacts import read_artifact_bytes, read_json_artifact, verify_artifact
from app.services.ingest.upload_index_service import embedding_identity
from app.services.source_io_service import run_source_io

logger = logging.getLogger(__name__)
READ_HEARTBEAT_SECONDS = READ_LEASE_SECONDS / 3


@dataclass
class _ReadLifetime:
    """当前执行与其线程共享续期失败事实，禁止失败后继续读取。"""

    error: Exception | None = None


_read_lifetime: ContextVar[_ReadLifetime | None] = ContextVar("source_read_lifetime", default=None)


def batch_filter(batch_ids: list[str]) -> str:
    """把服务端批次身份编码为明确过滤条件，空范围绝不退化为全库搜索。"""
    import json

    return "batch_id in " + json.dumps(batch_ids, ensure_ascii=True)


async def attach_publication_snapshot(scope: dict[str, Any]) -> dict[str, Any]:
    """在请求级范围中固定发布批次，所有查询及缓存共同使用该版本。"""
    if not scope.get("paper_ids"):
        return scope
    snapshot = await get_upload_publication_repository().snapshot(scope["paper_ids"], user_id=scope["user_id"], allow_empty=True)
    snapshot = {**snapshot, "access_scope": {"project_id": scope.get("project_id") or "",
        "requested_paper_ids": list(scope.get("accepted_paper_ids") or []), "paper_ids": list(scope["paper_ids"])}}
    return {**scope, "publication": snapshot,
        "filter_expr": f"({scope['filter_expr']}) and ({batch_filter(snapshot['batch_ids'])})",
        "local_retrieval_allowed": bool(snapshot["batch_ids"])}


async def read_stable_project_scope(project_id: str | None, *, user_id: str,
                                    paper_ids: list[str]) -> dict[str, Any]:
    """【检索】【评测准备】参数为项目、用户和论文；只读核对授权及当前批次，返回不带临时租约的范围。"""
    from app.services.project_scope_service import _build_project_context_scope

    scope = await _build_project_context_scope(project_id, requested_paper_ids=paper_ids, user_id=user_id)
    publication = await get_upload_publication_repository().snapshot(paper_ids, user_id=user_id, acquire_lease=False)
    return {**scope, "publication": publication,
            "filter_expr": f"({scope['filter_expr']}) and ({batch_filter(publication['batch_ids'])})",
            "local_retrieval_allowed": bool(publication["batch_ids"])}


def publication_from_scope(scope: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    从请求级范围里取出本轮固定的发布快照。

    研究取证要在「章节名单 — 读正文 — 按元数据筛块」几次调用间沿用同一版本，
    取值口径集中在这里，避免各处自己判字段。取不到时返回 None，调用方按
    「读当前版本」处理（知识检查器一类合法读当前版本的入口就走这条路）。

    Args:
        scope: 请求级项目范围合同，快照挂在 publication 字段上
    Returns:
        dict[str, Any] | None: 本轮发布快照；未固定时返回 None
    """
    publication = (scope or {}).get("publication")
    return publication if isinstance(publication, dict) and publication else None


async def release_publication_snapshot(scope: dict[str, Any] | None) -> None:
    """【检索】【结束读取】释放当前范围的占用并清理已退役向量；参数为原范围，无返回值。"""
    snapshot = (scope or {}).get("publication")
    if not snapshot:
        return
    await get_upload_publication_repository().release_snapshot(snapshot)
    from app.services.ingest.upload_pipeline import cleanup_retired_batches

    for paper_id in snapshot["paper_batches"]:
        await cleanup_retired_batches(paper_id, user_id=snapshot["user_id"])


@asynccontextmanager
async def publication_scope_lifetime(scope: dict[str, Any] | None) -> AsyncIterator[None]:
    """参数为可信范围；持有原版本并受管续期，实际工作退出后释放，无返回值。"""
    snapshot = publication_from_scope(scope)
    if snapshot is None:
        yield
        return
    lifetime = _ReadLifetime()
    token = _read_lifetime.set(lifetime)
    owner = asyncio.current_task()
    heartbeat = None
    try:
        await authorize_publication_read(snapshot, user_id=(scope or {})["user_id"])
        heartbeat = asyncio.create_task(_renew_read_lifetime(snapshot, lifetime, owner))
        yield
    finally:
        try:
            finish = asyncio.create_task(_finish_read_lifetime(scope, heartbeat))
            cancelled = False
            while not finish.done():
                try:
                    await asyncio.shield(finish)
                except asyncio.CancelledError:
                    cancelled = True
            finish.result()
        finally:
            _read_lifetime.reset(token)
        if lifetime.error is not None:
            raise lifetime.error
        if cancelled:
            raise asyncio.CancelledError


async def _finish_read_lifetime(scope: dict[str, Any] | None, heartbeat: asyncio.Task | None) -> None:
    """参数为原范围和受管心跳；等待续期退出并释放持久占用，失败由执行者显式处理。"""
    if heartbeat is not None:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
    await release_publication_snapshot(scope)


async def _renew_read_lifetime(snapshot: dict[str, Any], lifetime: _ReadLifetime,
                              owner: asyncio.Task | None) -> None:
    """参数为固定快照、执行失败事实及任务；续期失败停止持有者，保留真实错误。"""
    try:
        while True:
            await asyncio.sleep(READ_HEARTBEAT_SECONDS)
            await authorize_publication_read(snapshot, user_id=snapshot["user_id"])
    except Exception as error:
        lifetime.error = error
        logger.error("【来源】【占用续期】固定版本续期失败 lease_id=%s", snapshot.get("lease_id"), exc_info=True)
        if owner is not None:
            owner.cancel()


async def authorize_publication_read(publication: dict[str, Any], *, user_id: str,
                                     paper_ids: tuple[str, ...] | None = None) -> None:
    """参数为可信快照、账号及目标论文；重验当前范围和原租约，失败直接停止读取。"""
    lifetime = _read_lifetime.get()
    if lifetime is not None and lifetime.error is not None:
        raise lifetime.error
    if not user_id or publication.get("user_id") != user_id or not publication.get("lease_id"):
        raise RetrievalScopeError("读取缺少已授权的发布快照")
    mapping = publication.get("paper_batches")
    if not isinstance(mapping, dict) or set(mapping.values()) != set(publication.get("batch_ids", [])):
        raise RetrievalScopeError("发布快照论文与批次身份不一致")
    access_scope = publication.get("access_scope")
    allowed_papers = access_scope["paper_ids"] if access_scope is not None else mapping
    if paper_ids is not None and not set(paper_ids).issubset(allowed_papers):
        raise RetrievalScopeError("读取论文不在本轮固定发布范围内")
    if access_scope is not None:
        from app.services.project_scope_service import _build_project_context_scope

        # 1. 【来源】【当前授权】显式自有论文与项目默认范围沿用各自语义，不重新选择发布批次
        current = await _build_project_context_scope(access_scope["project_id"], user_id=user_id,
            requested_paper_ids=access_scope["requested_paper_ids"])
        if not set(allowed_papers).issubset(current["paper_ids"]):
            raise RetrievalScopeError("论文已移出当前可读取的项目范围")
    await get_upload_publication_repository().renew_snapshot(publication)


@asynccontextmanager
async def open_upload_source(paper_id: str, *, user_id: str) -> AsyncIterator[dict[str, Any]]:
    """参数为论文和账号；返回已核验下载字节，响应消费结束后才释放跨进程文件占用。"""
    repository = get_source_access_repository()
    async with repository.file_access((paper_id,)) as connection:
        record = await repository.source_reference(connection, paper_id, user_id=user_id)
        content = await run_source_io(read_artifact_bytes, record["source_ref"], root=Path(get_settings().data_dir))
        yield {"filename": record["filename"], "content": content}


def require_publication_scope(scope: dict[str, Any], *, collection_name: str) -> str:
    """核对现役模型、集合及持久读占用，返回当前请求唯一允许使用的过滤条件。"""
    snapshot = scope.get("publication")
    if not isinstance(snapshot, dict) or not snapshot.get("batch_ids") or snapshot.get("user_id") != scope.get("user_id"):
        raise RetrievalScopeError("请求缺少已授权且可用的索引快照")
    if snapshot["collection_names"] != [collection_name] or snapshot["embedding_fingerprints"] != [embedding_identity()]:
        raise RetrievalScopeError("论文索引与现役集合或向量模型不一致，请重新处理")
    paper_db._run(authorize_publication_read(snapshot, user_id=scope["user_id"]))
    return f"({scope['filter_expr']}) and ({batch_filter(snapshot['batch_ids'])})"


def batch_id_from_chunk(chunk_id: str) -> str:
    """从不可变引用身份中读取批次；不把旧式或非法 ID 猜成当前版本。"""
    parts = chunk_id.split(":")
    if len(parts) != 3 or parts[0] != "v3" or len(parts[1]) != 32 or not parts[2].isdigit():
        raise LookupError("引用没有可追溯的发布批次，请重新检索论文")
    return parts[1]


def read_published_chunk(chunk_id: str, *, user_id: str, paper_id: str | None = None,
                         publication: dict[str, Any] | None = None) -> dict[str, Any]:
    """参数为不可变引用、账号与可选范围；同步等待实际读取结束，返回原版本完整块。"""
    future = asyncio.run_coroutine_threadsafe(read_published_chunk_async(chunk_id, user_id=user_id,
        paper_id=paper_id, publication=publication), paper_db._ensure_loop())
    return future.result()


async def read_published_chunk_async(chunk_id: str, *, user_id: str, paper_id: str | None = None,
                                     publication: dict[str, Any] | None = None) -> dict[str, Any]:
    """参数为引用和可信身份；持跨进程占用重验授权并读原文，取消等待文件线程退出。"""
    batch_id = batch_id_from_chunk(chunk_id)
    repository = get_upload_publication_repository()
    record = await repository.content_reference(batch_id, user_id=user_id)
    actual_paper = record["paper_id"]
    if paper_id is not None and actual_paper != paper_id:
        raise LookupError("引用不属于这篇论文")
    async with get_source_access_repository().file_access((actual_paper,)) as connection:
        if publication is not None:
            await authorize_publication_read(publication, user_id=user_id, paper_ids=(actual_paper,))
            if publication["paper_batches"][actual_paper] != batch_id:
                raise RetrievalScopeError("引用不在本轮固定发布批次中")
        record = await repository.content_reference(batch_id, user_id=user_id, connection=connection)
        manifest = await run_source_io(_load_manifest, record)
    _validate_publication(manifest, batch_id=batch_id, user_id=user_id,
        snapshot={"paper_batches": {actual_paper: batch_id}})
    chunks = manifest["chunks"]
    index = int(chunk_id.split(":")[2])
    if index >= len(chunks) or chunks[index]["chunk_id"] != chunk_id:
        raise LookupError("引用分块不在已发布正文清单中")
    return dict(chunks[index])


def _load_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """参数为已授权批次记录；读取并校验完整清单及来源产物，返回原始清单。"""
    manifest = read_json_artifact(record["manifest_ref"], root=Path(get_settings().data_dir))
    if "content_schema_version" in manifest:
        validate_content_schema(manifest)
        _verify_source_artifacts(manifest, root=Path(get_settings().data_dir))
    return manifest


def read_published_batches(batch_ids: tuple[str, ...], *, user_id: str,
                           publication: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """参数为固定批次、账号与快照；每批授权并核验一次，返回请求独占的发布正文。"""
    future = asyncio.run_coroutine_threadsafe(_read_published_batches(batch_ids, user_id=user_id,
        publication=deepcopy(publication)), paper_db._ensure_loop())
    return future.result()


async def _read_published_batches(batch_ids: tuple[str, ...], *, user_id: str,
                                  publication: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """参数为固定批次、账号与快照；锁内重验每批当前授权，返回完整且不可变的正文。"""
    snapshot = deepcopy(publication)
    if not user_id or snapshot.get("user_id") != user_id or not snapshot.get("lease_id"):
        raise RetrievalScopeError("正文读取缺少已授权的发布快照")
    if not set(batch_ids).issubset(snapshot.get("batch_ids", [])):
        raise RetrievalScopeError("正文定位批次不在本轮发布快照中")
    repository = get_upload_publication_repository()
    # 1. 【知识访问】【固定版本】续用已有读取占用，不创建新快照或切换现役版本
    await authorize_publication_read(snapshot, user_id=user_id)
    manifests: dict[str, dict[str, Any]] = {}
    if not batch_ids:
        return manifests
    papers = tuple(paper for paper, batch in snapshot["paper_batches"].items() if batch in batch_ids)
    async with get_source_access_repository().file_access(papers) as connection:
        # 2. 【知识访问】【读取授权】等待文件占用期间可能撤回项目权限，进入临界区后重新核对
        await authorize_publication_read(snapshot, user_id=user_id, paper_ids=papers)
        for batch_id in dict.fromkeys(batch_ids):
            record = await repository.content_reference(batch_id, user_id=user_id, connection=connection)
            manifest = await run_source_io(_load_manifest, record)
            _validate_publication(manifest, batch_id=batch_id, user_id=user_id, snapshot=snapshot)
            manifests[batch_id] = manifest
    return manifests


def _validate_publication(manifest: dict[str, Any], *, batch_id: str, user_id: str,
                          snapshot: dict[str, Any]) -> None:
    """参数为发布清单及可信版本身份；核对正文身份、顺序地址与授权论文，无返回值。"""
    if not isinstance(manifest, dict) or manifest.get("format") != 3:
        raise ArtifactIntegrityError("正文清单缺少支持的发布格式")
    paper_id = manifest.get("paper_id")
    if manifest.get("batch_id") != batch_id or manifest.get("user_id") != user_id:
        raise ArtifactIntegrityError("正文清单身份与已授权批次不一致")
    if snapshot.get("paper_batches", {}).get(paper_id) != batch_id:
        raise RetrievalScopeError("正文论文不在本轮固定发布快照中")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise ArtifactIntegrityError("正文清单缺少完整分块列表")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or chunk.get("chunk_id") != f"v3:{batch_id}:{index}":
            raise ArtifactIntegrityError("正文分块地址与发布顺序不一致")
        if chunk.get("paper_id") != paper_id or chunk.get("batch_id") != batch_id:
            raise ArtifactIntegrityError("正文分块所属论文或批次不一致")


def validate_content_schema(document: dict[str, Any]) -> None:
    """参数为声明新版的文档或清单；严格验证来源文本与父子范围，畸形数据直接失败。"""
    if type(document.get("content_schema_version")) is not int or document["content_schema_version"] != CONTENT_SCHEMA_VERSION:
        raise ArtifactIntegrityError("不支持的正文内容格式版本")
    items = document.get("source_items")
    if not isinstance(items, dict) or not items:
        raise ArtifactIntegrityError("新版正文缺少规范来源项目")
    for key, item in items.items():
        _validate_source_item(key, item)
    chunks = document.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ArtifactIntegrityError("新版正文缺少完整分块")
    parents: dict[str, tuple[str, Any]] = {}
    for chunk in chunks:
        _validate_chunk_fragments(chunk, items)
        parent_id = chunk.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ArtifactIntegrityError("新版分块缺少父块身份")
        value = (chunk["full_text"], chunk["parent_source_fragments"])
        if parent_id in parents and parents[parent_id] != value:
            raise ArtifactIntegrityError("同一父块的正文或来源范围不一致")
        parents[parent_id] = value


def _validate_source_item(key: str, item: Any) -> None:
    """参数为来源项目键和值；核对稳定身份、文本、原始位置及实际页号，无返回值。"""
    if not isinstance(key, str) or not key or not isinstance(item, dict) or item.get("source_item") != key or not isinstance(item.get("text"), str):
        raise ArtifactIntegrityError("规范来源项目身份或文本无效")
    if not isinstance(item.get("source_kind"), str) or not item["source_kind"] or type(item.get("source_index")) is not int or item["source_index"] < 0:
        raise ArtifactIntegrityError("规范来源项目缺少原始产物定位")
    if not isinstance(item.get("block_type"), str) or not item["block_type"]:
        raise ArtifactIntegrityError("规范来源项目缺少结构类型")
    if "source_start" in item or "source_end" in item:
        if any(type(item.get(field)) is not int for field in ("source_start", "source_end")):
            raise ArtifactIntegrityError("原始产物区间必须同时提供整数起止坐标")
        if not 0 <= item["source_start"] < item["source_end"]:
            raise ArtifactIntegrityError("原始产物区间必须非负且起点早于终点")
    _validate_source_location(item)


def _validate_source_location(location: dict[str, Any]) -> None:
    """参数为项目或片段的物理位置；只接受实际整数页码及当前可核验的位置合同，无返回值。"""
    page_num = location.get("page_num")
    if page_num is not None and (type(page_num) is not int or page_num < 1):
        raise ArtifactIntegrityError("来源页码必须是实际存在的 1 起始页码")
    # 1. 【知识访问】【来源核验】当前内容版本尚未定义区域坐标口径，不能把任意区域声明为已核验
    if location.get("region") is not None:
        raise ArtifactIntegrityError("当前正文内容版本不支持可核验的区域坐标合同")


def _validate_chunk_fragments(chunk: dict[str, Any], items: dict[str, Any]) -> None:
    """参数为新版块与规范来源；验证两套正文映射，合成空白不计为原文范围。"""
    if not isinstance(chunk, dict):
        raise ArtifactIntegrityError("新版正文分块不是结构化对象")
    for field, fragment_key in (("text", "source_fragments"), ("full_text", "parent_source_fragments")):
        text = chunk.get(field)
        fragments = chunk.get(fragment_key)
        if not isinstance(text, str) or not text.strip() or not isinstance(fragments, list) or not fragments:
            raise ArtifactIntegrityError("新版正文缺少阅读文本或对应来源片段")
        cursor = 0
        for fragment in fragments:
            _validate_fragment(fragment, text=text, items=items)
            start, end = fragment["text_start"], fragment["text_end"]
            if start < cursor or text[cursor:start].strip():
                raise ArtifactIntegrityError("正文来源片段交叠或遗漏非空文本")
            cursor = end
        if text[cursor:].strip():
            raise ArtifactIntegrityError("正文尾部缺少来源映射")
    if chunk.get("display_text") != chunk["text"]:
        raise ArtifactIntegrityError("新版展示正文与阅读正文不一致")


def _validate_fragment(fragment: Any, *, text: str, items: dict[str, Any]) -> None:
    """参数为单片段、阅读正文与来源表；核对 Unicode 区间及逐字对应，无返回值。"""
    if not isinstance(fragment, dict) or not isinstance(fragment.get("source_item"), str) or fragment["source_item"] not in items:
        raise ArtifactIntegrityError("来源片段指向不存在的规范项目")
    keys = ("source_start", "source_end", "text_start", "text_end")
    if any(type(fragment.get(key)) is not int for key in keys):
        raise ArtifactIntegrityError("来源片段区间必须是 Unicode 字符整数坐标")
    item = items[fragment["source_item"]]
    start, end = fragment["source_start"], fragment["source_end"]
    text_start, text_end = fragment["text_start"], fragment["text_end"]
    if not 0 <= start < end <= len(item["text"]) or not 0 <= text_start < text_end <= len(text):
        raise ArtifactIntegrityError("来源片段区间超出实际文本")
    if item["text"][start:end] != text[text_start:text_end]:
        raise ArtifactIntegrityError("来源片段与阅读正文内容不一致")
    _validate_source_location(fragment)
    for key in ("page_num", "region"):
        if key in fragment and fragment[key] != item.get(key):
            raise ArtifactIntegrityError("来源片段物理位置与规范项目不一致")


def _verify_source_artifacts(manifest: dict[str, Any], *, root: Path) -> None:
    """参数为新版清单和受管根目录；每个原始产物只核验一次，不重新解析原文件。"""
    references = [item.get("artifact_ref") for item in manifest["source_items"].values()]
    verified: set[tuple[str, str, int]] = set()
    for reference in references:
        if not isinstance(reference, dict):
            raise ArtifactIntegrityError("发布来源项目缺少不可变原始产物引用")
        if not isinstance(reference.get("path"), str) or not isinstance(reference.get("sha256"), str) or type(reference.get("size")) is not int:
            raise ArtifactIntegrityError("发布来源项目的产物引用字段无效")
        identity = (reference["path"], reference["sha256"], reference["size"])
        if identity not in verified:
            verify_artifact(reference, root=root)
            verified.add(identity)
