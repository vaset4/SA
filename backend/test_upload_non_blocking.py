"""上传受理不等待长任务的回归，作者：xxx。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.services import embedded_worker_service, library_ingest, task_service


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_requested", [False, True])
async def test_upload_receipt_does_not_wait_for_processing(monkeypatch, graph_requested):
    """文件受理返回原任务；即使 worker 尚未完成，也不等待解析、分析或可选图谱。"""
    worker = asyncio.get_running_loop().create_future()
    receipt = {"task_id": "upload-task", "status": "queued", "payload": {"paper_id": "paper"}}
    create = AsyncMock(return_value=receipt)
    monkeypatch.setattr(task_service, "create_upload_task", create)
    monkeypatch.setattr(embedded_worker_service, "ensure_embedded_worker_running", Mock(return_value=worker))
    try:
        result = await asyncio.wait_for(library_ingest.add_uploaded_paper("paper.pdf", b"%PDF",
            user_id="owner", request_id="intent", graph_requested=graph_requested), timeout=1)
        assert result == receipt
        assert not worker.done()
        assert create.await_args.kwargs["graph_requested"] is graph_requested
    finally:
        worker.cancel()
