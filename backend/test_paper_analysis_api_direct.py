from __future__ import annotations

import asyncio
import os

os.environ["SSLKEYLOGFILE"] = ""

import app.api.routes.papers_api as papers_api


def test_create_paper_analysis_task_route_enqueues_background_job(monkeypatch) -> None:
    worker_calls: list[list[str] | None] = []

    monkeypatch.setattr(papers_api, "get_user_id", lambda user: "user-1")
    monkeypatch.setattr(
        papers_api.paper_facade,
        "get_paper_or_none",
        lambda paper_id, user_id="anonymous": {"paper_id": paper_id, "title": "Test Paper"},
    )

    async def fake_create_paper_analysis_task(paper_id: str, *, user_id: str = "anonymous"):
        return {
            "task_id": "task-analysis-1",
            "task_type": "paper_analysis",
            "status": "queued",
            "progress": 0,
            "message": "论文 AI 分析任务已入队",
            "payload": {"paper_id": paper_id, "user_id": user_id},
            "cancel_requested": False,
            "created_at": "2026-04-11T00:00:00+00:00",
            "updated_at": "2026-04-11T00:00:00+00:00",
        }

    monkeypatch.setattr(papers_api.task_service, "create_paper_analysis_task", fake_create_paper_analysis_task)
    monkeypatch.setattr(
        papers_api.embedded_worker_service,
        "ensure_embedded_worker_running",
        lambda *, task_types=None: worker_calls.append(task_types),
    )

    payload = asyncio.run(papers_api.create_paper_analysis_task("paper-1", user={}))

    assert payload["task_type"] == "paper_analysis"
    assert payload["payload"] == {"paper_id": "paper-1", "user_id": "user-1"}
    assert worker_calls == [["paper_analysis"]]
