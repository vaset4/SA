from __future__ import annotations

import asyncio

import app.services.task_service as task_service


def test_create_upload_task_preserves_intent_and_owner(monkeypatch) -> None:
    """控制层把稳定意图及服务端 owner 交给受理模块，不另外创建身份。"""
    from unittest.mock import AsyncMock
    from app.services import upload_service

    receive = AsyncMock(return_value={"task_id": "original", "payload": {"paper_id": "paper-a"}})
    monkeypatch.setattr(upload_service, "receive_upload", receive)
    task = asyncio.run(task_service.create_upload_task("paper.pdf", b"%PDF", user_id="owner-a",
        request_id="intent-a", library_id="library-a", project_id="project-a"))
    assert task["task_id"] == "original"
    assert receive.await_args.kwargs["user_id"] == "owner-a"
    assert receive.await_args.kwargs["request_id"] == "intent-a"
    assert receive.await_args.kwargs["project_id"] == "project-a"


def test_cancel_task_emits_cancel_requested_event(monkeypatch) -> None:
    events: list[tuple[str, str, str, dict]] = []

    def fake_request_task_cancel(task_id, message, *, user_id):  # noqa: ANN001
        return {
            "task_id": task_id,
            "task_type": "deep_research",
            "status": "running",
            "progress": 35,
            "message": message,
            "payload": {},
            "user_id": user_id,
            "cancel_requested": True,
            "created_at": "2026-03-23T00:00:00+00:00",
            "updated_at": "2026-03-23T00:00:01+00:00",
        }

    def fake_add_task_event(task_id, event_type, message, payload):  # noqa: ANN001
        events.append((task_id, event_type, message, payload))
        return {
            "event_id": "evt-1",
            "task_id": task_id,
            "event_type": event_type,
            "message": message,
            "payload": payload,
            "created_at": "2026-03-23T00:00:00+00:00",
        }

    monkeypatch.setattr(task_service.paper_db, "request_task_cancel", fake_request_task_cancel)
    monkeypatch.setattr(task_service.paper_db, "add_task_event", fake_add_task_event)

    task = asyncio.run(task_service.cancel_task("task-1", user_id="user-1"))

    assert task["cancel_requested"] is True
    assert events == [("task-1", "cancel_requested", "已请求取消任务", {})]


def test_create_paper_analysis_task_enqueues_backend_job(monkeypatch) -> None:
    created: dict[str, object] = {}
    status_updates: list[tuple[str, str, dict]] = []

    async def fake_create_task(task_type, payload=None, *, message="任务已入队", task_id=None, user_id):  # noqa: ANN001
        created.update(
            {
                "task_type": task_type,
                "payload": payload or {},
                "message": message,
                "task_id": task_id,
                "user_id": user_id,
            }
        )
        return {
            "task_id": "task-analysis-1",
            "task_type": task_type,
            "status": "queued",
            "progress": 0,
            "message": message,
            "payload": payload or {},
            "cancel_requested": False,
            "created_at": "2026-04-11T00:00:00+00:00",
            "updated_at": "2026-04-11T00:00:00+00:00",
        }

    monkeypatch.setattr(task_service, "create_task", fake_create_task)
    monkeypatch.setattr(
        task_service.paper_db,
        "update_analysis_status",
        lambda paper_id, status, **kwargs: status_updates.append((paper_id, status, kwargs)),
    )

    task = asyncio.run(task_service.create_paper_analysis_task("paper-1", user_id="user-1"))

    assert task["task_type"] == "paper_analysis"
    assert created == {
        "task_type": "paper_analysis",
        "payload": {"paper_id": "paper-1", "user_id": "user-1"},
        "message": "论文 AI 分析任务已入队",
        "task_id": None,
        "user_id": "user-1",
    }
    assert status_updates == [("paper-1", "queued", {"clear_analysis": True, "user_id": "user-1"})]
