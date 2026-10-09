from __future__ import annotations

import asyncio

import pytest

import app.repositories.paper_deletion_repository as deletion_repository_module
import app.repositories.paper_repository as paper_repository_module
import app.repositories.task_repository as task_repository_module


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
        return False


class _Connection:
    def __init__(self) -> None:
        self.paper = {
            "paper_id": "paper-1",
            "user_id": "user-1",
            "deletion_status": "active",
            "deletion_task_id": None,
        }

    def transaction(self):
        return _Transaction()

    async def fetchrow(self, query, *args):  # noqa: ANN001
        if "FROM papers" in query:
            return self.paper if args == ("paper-1", "user-1") else None
        raise AssertionError(f"unexpected query: {query}")

    async def execute(self, query, *args):  # noqa: ANN001
        assert "UPDATE papers" in query
        self.paper = {
            **self.paper,
            "deletion_status": "deleting",
            "deletion_task_id": args[2],
        }


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
        return False


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self):
        return _Acquire(self.connection)


class _TaskRepository:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.events: list[dict] = []

    async def create_task_run(self, task_type, payload, message, **kwargs):  # noqa: ANN001
        task = {
            "task_id": kwargs["task_id"],
            "user_id": kwargs["user_id"],
            "task_type": task_type,
            "status": "queued",
            "payload": payload,
            "message": message,
        }
        self.created.append(task)
        return task

    async def add_task_event(self, task_id, event_type, message, payload, **kwargs):  # noqa: ANN001
        self.events.append({"task_id": task_id, "event_type": event_type, "message": message, "payload": payload})

    async def get_task_run(self, task_id, *, user_id, **kwargs):  # noqa: ANN001
        return next(
            (task for task in self.created if task["task_id"] == task_id and task["user_id"] == user_id),
            None,
        )


def test_delete_request_is_idempotent_for_deleting_paper(monkeypatch) -> None:
    connection = _Connection()
    task_repository = _TaskRepository()

    async def fake_get_pool():
        return _Pool(connection)

    monkeypatch.setattr(deletion_repository_module, "get_pool", fake_get_pool)
    monkeypatch.setattr(deletion_repository_module, "get_task_repository", lambda: task_repository)
    repository = deletion_repository_module.PaperDeletionRepository()

    first = asyncio.run(repository.request_deletion("paper-1", user_id="user-1"))
    second = asyncio.run(repository.request_deletion("paper-1", user_id="user-1"))

    assert first is not None
    assert second is not None
    assert first["task_id"] == second["task_id"]
    assert len(task_repository.created) == 1
    assert len(task_repository.events) == 1


def test_deleting_papers_are_excluded_from_list_and_retrieval_candidates(monkeypatch) -> None:
    queries: list[str] = []

    async def fake_fetch(query, *args):  # noqa: ANN001
        queries.append(query)
        if "deletion_status" not in query or "active" not in query:
            return [{"paper_id": "deleting-paper"}]
        if "SELECT paper_id" in query:
            return [{"paper_id": "active-paper"}]
        return [{"paper_id": "active-paper", "title": "Active", "abstract": "body"}]

    monkeypatch.setattr(paper_repository_module, "fetch", fake_fetch)
    repository = paper_repository_module.PaperRepository()

    papers = asyncio.run(repository.list_papers(None, None, 20, 0, user_id="user-1"))
    paper_ids = asyncio.run(repository.list_paper_ids_by_library(user_id="user-1"))

    assert [paper["paper_id"] for paper in papers] == ["active-paper"]
    assert paper_ids == ["active-paper"]
    assert all("deletion_status" in query and "active" in query for query in queries)


def test_paper_delete_task_cannot_be_cancelled(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_fetchrow(query, *args):  # noqa: ANN001
        calls.append(query)
        return {
            "task_id": "task-delete-1",
            "user_id": "user-1",
            "task_type": "paper_delete",
            "status": "queued",
        }

    monkeypatch.setattr(task_repository_module, "fetchrow", fake_fetchrow)
    repository = task_repository_module.TaskRepository()

    with pytest.raises(ValueError, match="不可取消"):
        asyncio.run(repository.request_task_cancel("task-delete-1", "取消", user_id="user-1"))

    assert len(calls) == 1
