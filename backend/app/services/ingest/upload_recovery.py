"""上传阶段的持久检查点和有限短暂重试，作者：xxx。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any
import zipfile

import requests

from app.models.upload_models import (
    ArtifactIntegrityError, UploadCancelledError, UploadExecution, UploadFailure,
    UploadLeaseLostError, UploadPausedError,
)
from app.repositories.upload_repository import UploadRepository
from app.services.ingest.mineru_client import (
    MinerUAuthenticationError, MinerUConfigurationError, MinerUQuotaError,
    MinerURateLimitError, MinerUResultError, MinerUTemporaryError, MinerUTimeoutError, MinerUTransferError,
)
from app.services.ingest.upload_artifacts import fingerprint

MAX_STAGE_ATTEMPTS = 3
SHORT_RETRY_DELAYS = (1.0, 3.0)
MAX_SHORT_RETRY_WAIT_SECONDS = 5.0


def _network_failure(error: Exception, stage: str) -> UploadFailure | None:
    """识别网络和元信息 HTTP 边界，返回真实可重试类别或不适用。"""
    if isinstance(error, requests.Timeout):
        return UploadFailure(stage, "network_timeout", f"当前阶段连接或读取超时（{type(error).__name__}）", retryable=True)
    if isinstance(error, requests.ConnectionError):
        return UploadFailure(stage, "network_connection", "当前阶段无法连接外部服务", retryable=True)
    if not isinstance(error, requests.HTTPError):
        return None
    from app.services.ingest.mineru_client import _retry_after_seconds

    response = error.response
    status = response.status_code if response is not None else 0
    if status in {401, 403}:
        return UploadFailure(stage, "credentials", f"元信息服务凭证被拒绝：HTTP {status}")
    retry_after = _retry_after_seconds(response.headers.get("Retry-After", "")) if response is not None else None
    return UploadFailure(stage, "rate_limit" if status == 429 else "provider_error",
        f"元信息服务请求失败：HTTP {status}", retryable=status == 429 or status >= 500, retry_after=retry_after)


def classify_upload_failure(error: Exception, stage: str) -> UploadFailure:
    """接收真实阶段异常，区分可短暂重试、需修正配置及损坏产物。"""
    if isinstance(error, UploadPausedError):
        return error.failure
    categories = ((MinerUQuotaError, "quota", False),
        ((MinerUAuthenticationError, MinerUConfigurationError), "credentials", False),
        (MinerUTemporaryError, "provider_unavailable", True), (MinerUTimeoutError, "parse_wait_timeout", False))
    for types, kind, retryable in categories:
        if isinstance(error, types):
            return UploadFailure(stage, kind, str(error), retryable=retryable)
    if isinstance(error, MinerURateLimitError):
        return UploadFailure(stage, "rate_limit", str(error), retryable=True, retry_after=error.retry_after)
    if isinstance(error, MinerUTransferError):
        return UploadFailure(stage, "signed_url_expired", str(error),
            can_resume=stage == "mineru_artifacts", retryable=stage == "mineru_artifacts")
    network = _network_failure(error, stage)
    if network is not None:
        return network
    if isinstance(error, (ArtifactIntegrityError, zipfile.BadZipFile)):
        return UploadFailure(stage, "invalid_artifact", str(error), can_resume=stage == "mineru_artifacts")
    if isinstance(error, ValueError):
        return UploadFailure(stage, "invalid_result", str(error), can_resume=stage.startswith(("metadata_", "graph_")))
    if isinstance(error, MinerUResultError):
        return UploadFailure(stage, error.code or "invalid_result", str(error))
    return UploadFailure(stage, "unexpected", str(error))


def _request_was_rejected(error: Exception) -> bool:
    """只把明确拒绝当作未受理；普通网络故障和 HTTP 5xx 的创建结果仍未知。"""
    if isinstance(error, (MinerUAuthenticationError, MinerUConfigurationError, MinerURateLimitError, MinerUTransferError)):
        return True
    return isinstance(error, MinerUTemporaryError) and bool(error.code)


class UploadStageRunner:
    """用上传仓储保存阶段事实，不重跑已确认的上游操作。"""

    def __init__(self, repository: UploadRepository, execution: UploadExecution, *, sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        """接收仓储、执行身份及可测试的等待端口。"""
        self.repository = repository
        self.execution = execution
        self._sleep = sleeper

    async def run(
        self, stage: str, action: Callable[[], Awaitable[dict[str, Any]]], *, inputs: Any, repeatable: bool = True,
        budget_stage: str | None = None,
    ) -> dict[str, Any]:
        """【上传】【恢复步骤】核对输入指纹，复用完成记录，失败后有限重试或持久暂停。"""
        input_hash = fingerprint(inputs)
        while True:
            checkpoint = await self.repository.start_checkpoint(self.execution, stage=stage, input_hash=input_hash,
                repeatable=repeatable, attempt_limit=MAX_STAGE_ATTEMPTS, budget_stage=budget_stage)
            if checkpoint["state"] == "completed":
                return dict(checkpoint["payload"])
            attempt = int(checkpoint["attempt"])
            try:
                payload = await action()
                if not isinstance(payload, dict):
                    raise ArtifactIntegrityError("上传阶段返回值不是有效对象")
            except (UploadCancelledError, UploadLeaseLostError, UploadPausedError):
                raise
            except Exception as error:
                await self._failed(checkpoint, error, attempt=attempt, repeatable=repeatable, budget_stage=budget_stage)
                continue
            await self.repository.complete_checkpoint(self.execution, checkpoint, payload)
            await self.repository.check_execution(self.execution)
            return payload

    async def _failed(self, checkpoint: dict[str, Any], error: Exception, *, attempt: int, repeatable: bool, budget_stage: str | None) -> None:
        """【上传】【失败恢复】保留真实错误，未知创建不重发，长等待提示转为手动继续。"""
        failure = classify_upload_failure(error, budget_stage or checkpoint["stage"])
        if not repeatable and _request_was_rejected(error):
            await self.repository.reject_checkpoint(self.execution, checkpoint, failure)
        elif not repeatable:
            failure = replace(failure, kind="remote_result_unknown", retryable=False,
                can_resume=checkpoint["stage"] != "mineru_slot",
                detail="解析创建结果无法核实，已保存原文件并暂停，避免重复创建；" + failure.detail)
        await self.repository.record_retry(self.execution, failure)
        if not failure.retryable or attempt >= MAX_STAGE_ATTEMPTS:
            raise UploadPausedError(failure) from error
        delay = max(SHORT_RETRY_DELAYS[attempt - 1], failure.retry_after or 0)
        if delay > MAX_SHORT_RETRY_WAIT_SECONDS:
            raise UploadPausedError(replace(failure, detail=failure.detail + "；服务端要求等待较长时间，请稍后手动继续")) from error
        await self._sleep(delay)
        await self.repository.check_execution(self.execution)
