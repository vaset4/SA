"""同步上传兼容入口的持久任务合同，作者：xxx。"""
from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from app.api.facades import paper_facade
from app.services import embedded_worker_service, task_service


@pytest.mark.asyncio
async def test_upload_facade_passes_intent_owner_and_project_to_one_task(monkeypatch):
    """上传 facade 只转交稳定意图及项目关系，受理结果不冒充已完成论文。"""
    receipt = {"task_id": "task", "status": "queued", "payload": {"paper_id": "paper"}}
    create = AsyncMock(return_value=receipt)
    wake = Mock()
    monkeypatch.setattr(task_service, "create_upload_task", create)
    monkeypatch.setattr(embedded_worker_service, "ensure_embedded_worker_running", wake)
    result = await paper_facade.ingest_uploaded_paper("paper.pdf", b"%PDF", None,
        project_id="project", user_id="owner", request_id="stable-intent")
    assert result == receipt
    create.assert_awaited_once_with("paper.pdf", b"%PDF", library_id=None,
        project_id="project", user_id="owner", request_id="stable-intent", graph_requested=False)
    wake.assert_called_once_with(task_types=["upload_ingest"])


@pytest.mark.asyncio
async def test_rejected_upload_does_not_start_worker(monkeypatch):
    """受理冲突或归属错误必须原样暴露，不能唤醒未受理的任务。"""
    create = AsyncMock(side_effect=ValueError("intent conflict"))
    wake = Mock()
    monkeypatch.setattr(task_service, "create_upload_task", create)
    monkeypatch.setattr(embedded_worker_service, "ensure_embedded_worker_running", wake)
    with pytest.raises(ValueError, match="intent conflict"):
        await paper_facade.ingest_uploaded_paper("paper.pdf", b"%PDF", None,
            user_id="owner", request_id="stable-intent")
    wake.assert_not_called()


@pytest.mark.asyncio
async def test_old_metadata_route_uses_same_durable_operation(monkeypatch):
    """旧补全入口不能直接写业务库，必须携带 owner 排队元信息任务。"""
    from app.services import upload_service

    request = AsyncMock(return_value={"task_id": "metadata-task", "status": "queued"})
    wake = Mock()
    monkeypatch.setattr(upload_service, "request_upload_operation", request)
    monkeypatch.setattr(embedded_worker_service, "ensure_embedded_worker_running", wake)
    result = await paper_facade.autofill_paper_metadata_or_raise("paper", user_id="owner")
    assert result["status"] == "queued"
    request.assert_awaited_once_with("paper", "metadata", user_id="owner")
    wake.assert_called_once_with(task_types=["upload_ingest"])
