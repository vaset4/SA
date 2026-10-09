"""上传暂停与传输恢复边界回归，作者：xxx。"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.models.upload_models import ArtifactIntegrityError, UploadExecution, UploadFailure, UploadPausedError
from app.services.ingest.upload_recovery import UploadStageRunner, classify_upload_failure


@pytest.mark.asyncio
async def test_nested_pause_never_retries_outer_stage() -> None:
    """内层已暂停时外层必须保留原阶段原因且不再次调用内层。"""
    repository = AsyncMock()
    repository.start_checkpoint.return_value = {"state": "pending", "stage": "document", "attempt": 1}
    repository.reserve_attempt.side_effect = [1, 2, 3]
    execution = UploadExecution("task", "worker", "paper", "owner", "parse", "batch", None)
    failure = UploadFailure("mineru_artifacts", "network_timeout", "下载已用完短暂重试", retryable=True)
    action = AsyncMock(side_effect=UploadPausedError(failure))
    runner = UploadStageRunner(repository, execution, sleeper=AsyncMock())
    with pytest.raises(UploadPausedError) as error:
        await runner.run("document", action, inputs="source")
    assert error.value.failure == failure
    assert action.await_count == 1
    repository.record_retry.assert_not_awaited()


def test_corrupt_download_can_be_downloaded_again_without_reparsing() -> None:
    """结果 ZIP 未通过核验时保留原任务，并允许显式继续下载。"""
    failure = classify_upload_failure(ArtifactIntegrityError("ZIP 缺少正文"), "mineru_artifacts")
    assert failure.can_resume is True
    assert failure.retryable is False


def test_corrupt_source_is_not_reported_as_resumable_download() -> None:
    """原文件损坏不能靠重复下载结果掩盖。"""
    failure = classify_upload_failure(ArtifactIntegrityError("原文件 SHA 不一致"), "receive")
    assert failure.can_resume is False
