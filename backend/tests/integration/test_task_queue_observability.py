from __future__ import annotations

import asyncio
import importlib
import sys
import types


class FakeQueueService:
    def __init__(self) -> None:
        self.lease_seconds = 0.03
        self.completed: list[tuple[str, str, dict]] = []
        self.failed: list[tuple[str, str, str]] = []
        self.cancelled: list[tuple[str, str]] = []

    async def is_cancel_requested(self, task_id: str) -> bool:
        return False

    async def heartbeat(self, task_id: str, worker_id: str):  # noqa: ANN201
        return {"task_id": task_id, "worker_id": worker_id, "status": "running"}

    async def mark_completed(self, task_id: str, result, *, worker_id: str, message="任务执行完成"):  # noqa: ANN001
        self.completed.append((task_id, worker_id, result))
        return {"task_id": task_id, "status": "completed", "result": result}

    async def mark_failed(self, task_id: str, error: str, *, worker_id: str, message="任务执行失败"):  # noqa: ANN001
        self.failed.append((task_id, worker_id, error))
        return {"task_id": task_id, "status": "failed", "error": error}

    async def mark_cancelled(self, task_id: str, *, worker_id: str, message="任务已取消"):  # noqa: ANN001
        self.cancelled.append((task_id, worker_id))
        return {"task_id": task_id, "status": "cancelled"}


def test_execute_task_emits_task_execution_completed_observation(monkeypatch) -> None:
    fake_tasks = types.ModuleType("app.tasks")
    fake_tasks.get_task_handler = lambda task_type: None  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "app.tasks", fake_tasks)
    sys.modules.pop("app.workers.worker_main", None)
    worker_main = importlib.import_module("app.workers.worker_main")

    queue = FakeQueueService()
    observations: list[tuple[str, str]] = []

    async def fake_handler(task_run):  # noqa: ANN001
        return {"ok": True, "task_id": task_run["task_id"]}

    monkeypatch.setattr(worker_main, "get_task_handler", lambda task_type: fake_handler)
    monkeypatch.setattr(
        worker_main.observability,
        "record_timed_event",
        lambda event_type, node_name, started_at, **kwargs: observations.append(  # noqa: ARG005
            (event_type, kwargs.get("query", ""))
        ),
    )

    task = {"task_id": "task-1", "task_type": "deep_research", "cancel_requested": False}
    result = asyncio.run(worker_main.execute_task(task, worker_id="worker-a", queue_service=queue))

    assert result["status"] == "completed"
    assert queue.completed == [("task-1", "worker-a", {"ok": True, "task_id": "task-1"})]
    assert observations == [("task_execution_completed", "deep_research:task-1")]


def test_terminal_updates_reject_worker_without_active_lease(monkeypatch) -> None:
    """旧 worker 的 completed、failed、cancelled CAS 均应返回空。"""
    from app.repositories import task_repository

    calls: list[tuple[str, tuple]] = []

    async def fake_fetchrow(query, *args):  # noqa: ANN001
        calls.append((" ".join(query.split()), args))
        return None

    monkeypatch.setattr(task_repository, "fetchrow", fake_fetchrow)
    repository = task_repository.TaskRepository()

    async def run_terminal_updates() -> list[dict | None]:
        return [
            await repository.complete_task_run(
                "task-1", {"ok": True}, "完成", worker_id="worker-old"
            ),
            await repository.fail_task_run(
                "task-1", "boom", "失败", worker_id="worker-old"
            ),
            await repository.mark_task_cancelled(
                "task-1", "取消", worker_id="worker-old"
            ),
        ]

    results = asyncio.run(run_terminal_updates())

    assert results == [None, None, None]
    assert len(calls) == 3
    for query, args in calls:
        assert "worker_id=$" in query
        assert "status='running'" in query
        assert "lease_until IS NOT NULL" in query
        assert "lease_until > NOW()" in query
        assert args[0] == "task-1"
        assert args[-1] == "worker-old"
