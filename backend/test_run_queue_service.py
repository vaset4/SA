from __future__ import annotations

import asyncio

from app.services.run_queue_service import RunQueueService


def test_claim_next_task_appends_claimed_event(monkeypatch) -> None:
    claimed_payloads: list[tuple[str, str, dict]] = []

    def fake_claim_next_task(worker_id, lease_seconds, task_types):  # noqa: ANN001
        assert worker_id == "worker-a"
        assert lease_seconds == 123
        assert task_types == ["upload_ingest"]
        return {
            "task_id": "task-1",
            "task_type": "upload_ingest",
            "status": "running",
            "progress": 0,
            "message": "任务运行中",
            "payload": {},
            "cancel_requested": False,
            "created_at": "2026-03-23T00:00:00+00:00",
            "updated_at": "2026-03-23T00:00:00+00:00",
        }

    async def fake_append_event(task_id, event_type, message, payload=None):  # noqa: ANN001
        claimed_payloads.append((task_id, event_type, payload or {}))
        return {
            "event_id": "evt-1",
            "task_id": task_id,
            "event_type": event_type,
            "message": message,
            "payload": payload or {},
            "created_at": "2026-03-23T00:00:00+00:00",
        }

    monkeypatch.setattr("app.services.run_queue_service.paper_db.claim_next_task", fake_claim_next_task)
    monkeypatch.setattr("app.services.run_queue_service.task_service.append_event", fake_append_event)

    service = RunQueueService(lease_seconds=123)
    task = asyncio.run(service.claim_next_task("worker-a", ["upload_ingest"]))

    assert task is not None
    assert claimed_payloads == [("task-1", "claimed", {"worker_id": "worker-a"})]
