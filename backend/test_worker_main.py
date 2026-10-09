from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

os.environ["SSLKEYLOGFILE"] = ""

import app.workers.worker_main as worker_main


def test_worker_loop_uses_new_lease_identity_for_each_claim(monkeypatch) -> None:
    """同一 worker 循环再次领取任务时使用新身份，避免旧线程复用后续租约。"""
    claims = []

    class Queue:
        """记录领取者身份的队列端口。"""

        async def claim_next_task(self, worker_id, task_types):
            """接收当前独立执行身份，返回一条测试任务。"""
            claims.append(worker_id)
            return {"task_id": str(len(claims)), "worker_id": worker_id}

    execute = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(worker_main, "execute_task", execute)
    count = asyncio.run(worker_main.run_worker_loop(worker_id="daemon", queue_service=Queue(), max_jobs=2))
    assert count == 2
    assert len(set(claims)) == 2
    assert all(identity.startswith("daemon:") for identity in claims)
    assert [call.kwargs["worker_id"] for call in execute.await_args_list] == claims


class FakeQueueService:
    def __init__(self, task=None, *, lease_seconds: float = 0.03) -> None:
        self.task = task
        self.lease_seconds = lease_seconds
        self.completed: list[tuple[str, str, dict]] = []
        self.failed: list[tuple[str, str, str]] = []
        self.cancelled: list[tuple[str, str]] = []
        self.heartbeats: list[tuple[str, str]] = []
        self.heartbeat_result: dict | None = {"status": "running"}
        self.heartbeat_error: Exception | None = None

    async def claim_next_task(self, worker_id, task_types=None):  # noqa: ANN001
        task, self.task = self.task, None
        return task

    async def is_cancel_requested(self, task_id: str) -> bool:
        return False

    async def heartbeat(self, task_id: str, worker_id: str):  # noqa: ANN201
        self.heartbeats.append((task_id, worker_id))
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        return self.heartbeat_result

    async def mark_completed(self, task_id: str, result, *, worker_id: str, message="任务执行完成"):  # noqa: ANN001
        self.completed.append((task_id, worker_id, result))
        return {"task_id": task_id, "status": "completed", "result": result}

    async def mark_failed(self, task_id: str, error: str, *, worker_id: str, message="任务执行失败"):  # noqa: ANN001
        self.failed.append((task_id, worker_id, error))
        return {"task_id": task_id, "status": "failed", "error": error}

    async def mark_cancelled(self, task_id: str, *, worker_id: str, message="任务已取消"):  # noqa: ANN001
        self.cancelled.append((task_id, worker_id))
        return {"task_id": task_id, "status": "cancelled"}


def test_execute_task_marks_completed(monkeypatch) -> None:
    queue = FakeQueueService()

    async def fake_handler(task_run):  # noqa: ANN001
        return {"ok": True, "task_id": task_run["task_id"]}

    monkeypatch.setattr(worker_main, "get_task_handler", lambda task_type: fake_handler)

    task = {"task_id": "task-1", "task_type": "deep_research", "cancel_requested": False}
    result = asyncio.run(worker_main.execute_task(task, worker_id="worker-a", queue_service=queue))

    assert result["status"] == "completed"
    assert queue.completed == [("task-1", "worker-a", {"ok": True, "task_id": "task-1"})]


def test_execute_task_renews_short_lease_while_handler_runs(monkeypatch) -> None:
    """短租约任务执行期间应持续 heartbeat，并由同一 worker 写终态。"""
    queue = FakeQueueService(lease_seconds=0.03)

    async def slow_handler(task_run):  # noqa: ANN001
        await asyncio.sleep(0.035)
        return {"ok": True, "task_id": task_run["task_id"]}

    monkeypatch.setattr(worker_main, "get_task_handler", lambda task_type: slow_handler)
    task = {"task_id": "task-lease", "task_type": "deep_research", "cancel_requested": False}

    result = asyncio.run(worker_main.execute_task(task, worker_id="worker-a", queue_service=queue))

    assert result["status"] == "completed"
    assert queue.heartbeats
    assert set(queue.heartbeats) == {("task-lease", "worker-a")}
    assert queue.completed[0][:2] == ("task-lease", "worker-a")


def test_execute_task_cancels_handler_without_terminal_write_when_lease_is_lost(monkeypatch) -> None:
    """heartbeat 返回空表示失租，旧 worker 必须取消 handler 且不写终态。"""
    queue = FakeQueueService(lease_seconds=0.03)
    queue.heartbeat_result = None
    handler_cancelled = asyncio.Event()

    async def blocked_handler(task_run):  # noqa: ANN001, ARG001
        try:
            await asyncio.Event().wait()
        finally:
            handler_cancelled.set()

    monkeypatch.setattr(worker_main, "get_task_handler", lambda task_type: blocked_handler)
    task = {"task_id": "task-lost", "task_type": "deep_research", "cancel_requested": False}

    result = asyncio.run(worker_main.execute_task(task, worker_id="worker-old", queue_service=queue))

    assert result["status"] == "lease_lost"
    assert handler_cancelled.is_set()
    assert queue.completed == []
    assert queue.failed == []
    assert queue.cancelled == []


def test_execute_task_treats_heartbeat_error_as_lease_lost(monkeypatch) -> None:
    """heartbeat 异常表示租约不再可靠，不得把 handler 写成 failed。"""
    queue = FakeQueueService(lease_seconds=0.03)
    queue.heartbeat_error = RuntimeError("database unavailable")

    async def blocked_handler(task_run):  # noqa: ANN001, ARG001
        await asyncio.Event().wait()

    monkeypatch.setattr(worker_main, "get_task_handler", lambda task_type: blocked_handler)
    task = {"task_id": "task-error", "task_type": "deep_research", "cancel_requested": False}

    result = asyncio.run(worker_main.execute_task(task, worker_id="worker-old", queue_service=queue))

    assert result["status"] == "lease_lost"
    assert "database unavailable" in result["error"]
    assert queue.completed == []
    assert queue.failed == []
    assert queue.cancelled == []


def test_run_worker_loop_stops_when_idle_after_one_job(monkeypatch) -> None:
    queue = FakeQueueService(
        {
            "task_id": "task-1",
            "task_type": "deep_research",
            "cancel_requested": False,
        }
    )

    async def fake_handler(task_run):  # noqa: ANN001
        return {"report": "done", "task_id": task_run["task_id"]}

    monkeypatch.setattr(worker_main, "get_task_handler", lambda task_type: fake_handler)

    processed = asyncio.run(
        worker_main.run_worker_loop(
            worker_id="worker-a",
            queue_service=queue,
            stop_when_idle=True,
            max_jobs=1,
        )
    )

    assert processed == 1
    assert len(queue.completed) == 1
    assert queue.completed[0][0] == "task-1"
    assert queue.completed[0][1].startswith("worker-a:")
    assert queue.completed[0][2] == {"report": "done", "task_id": "task-1"}
