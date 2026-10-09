"""持久上传 handler 的身份与发布边界测试，作者：xxx。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
import pytest

from app.tasks import upload_task


def test_handler_returns_prepared_publication_without_early_completion(monkeypatch):
    """handler 只交回待发布合同，源文件和终态交由外层事务管理。"""
    prepared = {"_upload_publication": {"paper_id": "paper-a"}}
    prepare = AsyncMock(return_value=prepared)
    monkeypatch.setattr(upload_task, "prepare_upload_task", prepare)
    task = {"task_id": "task-a", "user_id": "owner-a", "worker_id": "worker-a", "payload": {"paper_id": "paper-a"}}
    assert asyncio.run(upload_task.run_upload_ingest_task(task)) == prepared
    assert prepare.await_args.args[0]["user_id"] == "owner-a"
    assert task["payload"]["paper_id"] == "paper-a"


def test_legacy_task_keeps_source_and_requires_explicit_new_upload(tmp_path, monkeypatch):
    """没有持久受理记录的旧任务不伪造检查点，也不删除历史文件。"""
    source = tmp_path / "legacy.pdf"
    source.write_bytes(b"%PDF-1.4")
    prepare = AsyncMock()
    monkeypatch.setattr(upload_task, "prepare_upload_task", prepare)
    task = {"task_id": "legacy", "user_id": "owner", "payload": {"paper_id": "paper", "staging_file_path": str(source)}}
    with pytest.raises(ValueError, match="旧上传任务"):
        asyncio.run(upload_task.run_upload_ingest_task(task))
    prepare.assert_not_awaited()
    assert source.exists()


def test_missing_paper_id_fails_before_processing(monkeypatch):
    """缺失身份时不临时生成新论文。"""
    prepare = AsyncMock()
    monkeypatch.setattr(upload_task, "prepare_upload_task", prepare)
    with pytest.raises(ValueError, match="paper_id"):
        asyncio.run(upload_task.run_upload_ingest_task({"task_id": "t", "user_id": "owner", "payload": {}}))
    prepare.assert_not_awaited()
