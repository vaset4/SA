"""从持久原文和 MinerU 原任务恢复正文解析，作者：xxx。"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
import time
from typing import Any

from app.config import get_settings
from app.models.upload_models import ArtifactIntegrityError, ArtifactRef, UploadFailure, UploadPausedError
from app.services.ingest import document_parse_service, mineru_markdown_service
from app.services import document_parser
from app.services.ingest.mineru_client import MinerUClient, MinerUTimeoutError
from app.services.ingest.upload_artifacts import (
    extract_mineru_archive, inspect_file, read_artifact_bytes, read_json_artifact,
    verify_artifact, verify_mineru_pages, write_artifact, write_json_artifact,
)
from app.services.ingest.upload_recovery import UploadStageRunner
from app.services.source_io_service import run_upload_file_io

# 1. 【解析】【处理身份】PDF 读取器变更后，旧检查点必须以新解析身份重新处理
PARSER_FORMAT_VERSION = 4
MAX_PARSE_WAIT_SECONDS = 600
UPLOAD_SLOT_LIFETIME_SECONDS = 24 * 60 * 60
CHUNK_PARAMETER_KEYS = (
    "child_chunk_overlap_tokens", "child_chunk_target_size", "chunk_merge_floor",
    "embedding_model", "parent_chunk_max_size", "parent_chunk_overlap_tokens",
)


def _select_artifact(files: list[Path], patterns: tuple[str, ...]) -> Path:
    """按已支持的输出格式选择唯一产物，多份同类结果明确拒绝而非取首项。"""
    for pattern in patterns:
        matches = [path for path in files if path.match(pattern)]
        if len(matches) > 1:
            raise ArtifactIntegrityError(f"MinerU 结果包含多份 {pattern}，无法确认对应原文")
        if matches:
            return matches[0]
    raise ArtifactIntegrityError(f"MinerU 结果缺少必需产物: {patterns[0]}")


@dataclass(frozen=True, slots=True, kw_only=True)
class UploadDocumentParser:
    """一次上传执行的解析组件；供应商、存储目录和执行控制通过参数注入。"""

    runner: UploadStageRunner
    client: MinerUClient
    source: ArtifactRef
    filename: str
    root: Path

    @property
    def directory(self) -> Path:
        """返回本论文本解析代次独立的产物目录。"""
        execution = self.runner.execution
        return self.root / "uploads" / execution.paper_id / "parses" / execution.parse_id

    def _parameters(self) -> dict[str, Any]:
        """记录影响解析内容的配置，不把凭证或重试等待参数纳入内容身份。"""
        settings = get_settings()
        return {
            "version": PARSER_FORMAT_VERSION, "source_sha256": self.source["sha256"],
            "source_size": self.source["size"], "mineru_enabled": settings.mineru_enabled,
            "mineru_endpoint": self.client.base_url,
            "chunking": {**{name: getattr(settings, name) for name in CHUNK_PARAMETER_KEYS},
                         "tokenizer": document_parser._get_token_counter().identity()},
        }

    async def parse(self) -> dict[str, Any]:
        """【上传】【正文恢复】先核对原文件；已完成正文直接复用，否则继续原解析任务。"""
        await run_upload_file_io(self.runner.execution, verify_artifact, self.source, root=self.root)
        parameters = await run_upload_file_io(self.runner.execution, self._parameters)
        reference = await self.runner.run("document", partial(self._parse_document, parameters), inputs=parameters)
        document = await run_upload_file_io(self.runner.execution, read_json_artifact, reference["document_ref"], root=self.root)
        if not isinstance(document, dict) or not document.get("chunks") or not document.get("full_text"):
            raise ArtifactIntegrityError("已保存的正文缺少有效分块或全文")
        from app.services.upload_read_service import validate_content_schema

        validate_content_schema(document)
        for resource in reference.get("resources", {}).values():
            await run_upload_file_io(self.runner.execution, verify_artifact, resource, root=self.root)
        return {"document": document, **reference}

    async def _parse_document(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """分别处理供应商正文与已登记的本地格式，基础解析不调用元信息 LLM。"""
        source_path = Path(self.source["path"])
        if source_path.suffix.lower() == ".pdf" and parameters["mineru_enabled"]:
            bundle = await self._mineru_bundle(parameters)
            document = await run_upload_file_io(self.runner.execution, self._convert_bundle, bundle)
            resources = bundle["resources"]
        else:
            local_document = await run_upload_file_io(self.runner.execution, document_parse_service.parse_local_document,
                source_path, self.runner.execution.paper_id)
            document = {**local_document, "source_items": {key: {**item, "artifact_ref": self.source}
                for key, item in local_document["source_items"].items()}}
            resources = {}
        metadata = await run_upload_file_io(self.runner.execution, document_parse_service.extract_document_metadata,
            source_path, document["full_text"], self.filename, chunks=document["chunks"], use_llm=False)
        from app.services.upload_read_service import validate_content_schema

        prepared = {**document, "metadata": metadata}
        validate_content_schema(prepared)
        reference = await run_upload_file_io(self.runner.execution, write_json_artifact, self.directory, prepared)
        return {"document_ref": reference, "resources": resources}

    async def _mineru_bundle(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """申请、上传、等待、下载分别持久记录；任一后续失败都保留原 batch_id。"""
        slot = await self.runner.run("mineru_slot", self._request_slot, inputs=parameters, repeatable=False)
        upload = await self.runner.repository.checkpoint(self.runner.execution, "mineru_upload")
        if upload is not None and upload["state"] == "pending":
            # 1. 【解析】【上传核查】PUT 结果未知时只查原任务，不能依据 waiting-file 盲目重传
            await self._wait_for_original(slot)
            await self.runner.repository.complete_checkpoint(self.runner.execution, upload, {"uploaded": True, "reconciled": True})
        await self.runner.run("mineru_upload", partial(self._upload_source, slot), inputs=slot["batch_id"], repeatable=False)
        await self._wait_for_original(slot)
        bundle = await self.runner.run("mineru_artifacts", partial(self._download_bundle, slot), inputs=slot["batch_id"])
        return bundle

    async def _request_slot(self) -> dict[str, Any]:
        """保存槽位身份后，后续步骤才可以上传原文件。"""
        parse_id = self.runner.execution.parse_id
        slot = await asyncio.to_thread(self.client.request_upload_slot, parse_id + ".pdf", data_id=parse_id)
        return {**asdict(slot), "received_at": time.time()}

    async def _upload_source(self, slot: dict[str, Any]) -> dict[str, Any]:
        """向有效槽位上传已核验的同一份原始字节，拒绝过期地址。"""
        if time.time() - float(slot["received_at"]) >= UPLOAD_SLOT_LIFETIME_SECONDS:
            raise UploadPausedError(UploadFailure("mineru_upload", "slot_expired", "原上传地址已过期，原文件仍已保存，请重新处理", can_resume=False))
        content = await run_upload_file_io(self.runner.execution, read_artifact_bytes, self.source, root=self.root)
        await asyncio.to_thread(self.client.upload_bytes, slot["upload_url"], content)
        return {"uploaded": True, "batch_id": slot["batch_id"]}

    async def _new_poll_budget(self) -> dict[str, Any]:
        """为本次用户继续轮次记录等待期限，重启 worker 不延长期限。"""
        return {"deadline": time.time() + min(self.client.timeout_seconds, MAX_PARSE_WAIT_SECONDS)}

    async def _wait_for_original(self, slot: dict[str, Any]) -> dict[str, Any]:
        """读取同一原任务结果；普通排队等待与错误重试采用独立时限。"""
        document = await self.runner.repository.get_document(self.runner.execution.paper_id, user_id=self.runner.execution.user_id)
        if document is None:
            raise ArtifactIntegrityError("上传来源记录已丢失")
        budget = await self.runner.run("poll_budget_" + str(document["resume_sequence"]), self._new_poll_budget,
            inputs={"batch_id": slot["batch_id"], "resume_sequence": document["resume_sequence"]})
        return await self.runner.run("mineru_result", partial(self._poll_result, slot, budget), inputs=slot["batch_id"])

    async def _poll_result(self, slot: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
        """先查询再判断期限，人工继续时已完成的远端结果仍可直接取回。"""
        while True:
            await self.runner.repository.check_execution(self.runner.execution)
            result = await asyncio.to_thread(self.client.get_result, slot["batch_id"], file_name=slot["file_name"], data_id=slot["data_id"])
            if result is not None and result.state == "done":
                return asdict(result)
            if time.time() >= float(budget["deadline"]):
                state = result.state if result is not None else "waiting"
                raise MinerUTimeoutError(f"解析服务仍未完成（{state}），已保留原任务，请稍后继续")
            await asyncio.sleep(self.client.poll_interval_seconds)

    async def _download_bundle(self, slot: dict[str, Any]) -> dict[str, Any]:
        """下载前重新读取原任务的结果地址，避免永久复用已过期的签名 URL。"""
        result = await asyncio.to_thread(self.client.get_result, slot["batch_id"], file_name=slot["file_name"], data_id=slot["data_id"])
        if result is None or result.state != "done":
            raise ArtifactIntegrityError("原解析任务当前未提供已完成结果")
        archive = await asyncio.to_thread(self.client.download_result_zip, result.full_zip_url)
        source_bytes = await run_upload_file_io(self.runner.execution, read_artifact_bytes, self.source, root=self.root)
        return await run_upload_file_io(self.runner.execution, self._materialize_bundle, archive, source_bytes)

    def _materialize_bundle(self, archive: bytes, source_bytes: bytes) -> dict[str, Any]:
        """【解析】【产物核验】完整提取后核对必需文件、页集合和资源，成功才保存检查点。"""
        archive_ref = write_artifact(self.directory / "archives", archive, suffix=".zip")
        directory = self.directory / "results" / archive_ref["sha256"]
        extract_mineru_archive(archive, directory)
        files = [path for path in directory.rglob("*") if path.is_file()]
        markdown = _select_artifact(files, ("full.md", "*_full.md"))
        content_path = _select_artifact(files, ("*_content_list.json", "content_list.json", "*_content_list_v2.json", "content_list_v2.json"))
        model = _select_artifact(files, ("*_model.json", "model.json"))
        content_list = mineru_markdown_service._load_content_list(content_path)
        page_count = verify_mineru_pages(source_bytes, content_list, model_path=model)
        if not markdown.read_text(encoding="utf-8").strip():
            raise ArtifactIntegrityError("MinerU 返回了空正文")
        resources = {path.relative_to(markdown.parent).as_posix(): inspect_file(path)
            for path in files if path.is_relative_to(markdown.parent) and path not in {markdown, content_path, model}}
        for reference in resources.values():
            if Path(reference["path"]).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                from PIL import Image
                try:
                    with Image.open(reference["path"]) as image:
                        image.load()
                except (OSError, SyntaxError, ValueError) as error:
                    raise ArtifactIntegrityError("MinerU 返回的图片资源损坏") from error
        return {"archive_ref": archive_ref, "markdown_ref": inspect_file(markdown), "content_ref": inspect_file(content_path),
                "model_ref": inspect_file(model), "normalized_page_count": page_count,
                "resources": resources, "artifact_dir": str(markdown.parent)}

    def _convert_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """消费核验后的正文与资源，使用现有结构化分块器并保留原始定位。"""
        markdown = read_artifact_bytes(bundle["markdown_ref"], root=self.root).decode("utf-8")
        content_list = mineru_markdown_service.normalize_content_list(read_json_artifact(bundle["content_ref"], root=self.root))
        artifacts = {"markdown_text": markdown, "content_list": content_list, "parser_mode": "mineru_online",
            "artifact_dir": bundle["artifact_dir"], "source_markdown_path": bundle["markdown_ref"]["path"],
            "sidecar_path": bundle["content_ref"]["path"], "normalized_page_count": bundle["normalized_page_count"]}
        document = document_parse_service.parse_pdf_artifacts(Path(self.source["path"]), self.runner.execution.paper_id, artifacts)
        for chunk in document["chunks"]:
            resource = str(chunk.get("asset_ref") or "").removeprefix("./")
            if resource and resource not in bundle["resources"]:
                raise ArtifactIntegrityError(f"MinerU 结果缺少被正文引用的资源: {resource}")
        source_items = {key: {**item, "artifact_ref": bundle[
            "content_ref" if item["source_kind"] == "content_list" else "markdown_ref"]}
            for key, item in document["source_items"].items()}
        return {**document, "source_items": source_items}
