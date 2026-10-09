"""上传受理与业务身份回归，作者：xxx。"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from app.services import embedded_worker_service, library_ingest, task_service


def test_upload_acceptance_does_not_report_completed_ingest(monkeypatch):
    """受理只返回 queued，完成事实必须由后续核验发布事务产生。"""
    receipt = {"task_id": "task", "status": "queued", "payload": {"paper_id": "paper"}}
    monkeypatch.setattr(task_service, "create_upload_task", AsyncMock(return_value=receipt))
    monkeypatch.setattr(embedded_worker_service, "ensure_embedded_worker_running", Mock())
    result = asyncio.run(library_ingest.add_uploaded_paper("paper.pdf", b"%PDF",
        "library", user_id="owner", request_id="intent"))
    assert result["status"] == "queued"
    assert "searchable" not in result
    assert result["payload"]["paper_id"] == "paper"


def test_explicit_upload_paper_id_uses_atomic_owner_aware_upsert(monkeypatch) -> None:
    """同一任务并发首次写入时，两次调用都通过同一条 owner-aware upsert 收敛。"""
    from app.repositories import paper_repository

    calls: list[tuple[str, tuple]] = []

    async def fake_fetchval(query, *args):  # noqa: ANN001
        calls.append((" ".join(query.split()), args))
        return "paper-stable"

    monkeypatch.setattr(paper_repository, "fetchval", fake_fetchval)
    paper = {
        "paper_id": "paper-stable", "user_id": "user-1", "library_id": "lib-1",
        "arxiv_id": "arxiv-existing", "doi": "doi-existing",
        "title": "Paper", "abstract": "", "authors": [], "categories": [],
    }

    async def run_concurrent_upserts():
        """并发执行两次相同论文身份的仓储写入。"""
        repository = paper_repository.PaperRepository()
        return await asyncio.gather(
            repository.add_or_update_paper(paper),
            repository.add_or_update_paper(paper),
        )

    assert asyncio.run(run_concurrent_upserts()) == ["paper-stable", "paper-stable"]
    assert len(calls) == 2
    assert all("ON CONFLICT (paper_id) DO UPDATE" in query for query, _ in calls)
    assert all("WHERE papers.user_id=EXCLUDED.user_id" in query for query, _ in calls)
    assert all(args[2] == "paper-stable" for _, args in calls)
