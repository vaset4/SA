"""论文删除任务阶段 D 聚焦测试。

Author: xxx
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import app.repositories.paper_deletion_repository as deletion_repository_module
import app.repositories.research_memory_repository as memory_repository_module
import app.tasks.paper_delete_task as paper_delete_task_module
from app.services.paper_file_cleanup_service import delete_managed_paper_files


DELETE_STEP_FUNCTIONS = (
    ("_delete_milvus", "milvus"),
    ("_delete_lightrag", "lightrag"),
    ("_delete_neo4j", "neo4j"),
    ("_delete_database_references", "database_references"),
    ("_delete_files", "files"),
    ("_delete_paper_record", "paper_record"),
)


class _FakeDeletionRepository:
    """记录删除状态变化，供 handler 故障重试测试使用。"""

    def __init__(self) -> None:
        self.task_id = "task-delete-1"
        self.failures: list[dict[str, str]] = []

    async def get_deletion_state(self, paper_id: str, *, user_id: str) -> dict[str, str]:
        """返回当前任务绑定的 deleting 论文。"""
        return {
            "paper_id": paper_id,
            "user_id": user_id,
            "library_id": "library-1",
            "pdf_path": "",
            "deletion_status": "deleting",
            "deletion_task_id": self.task_id,
        }

    async def mark_deletion_failed(
        self,
        paper_id: str,
        *,
        user_id: str,
        task_id: str,
        failed_step: str,
        error: str,
    ) -> bool:
        """记录失败步骤并模拟状态持久化成功。"""
        self.failures.append(
            {
                "paper_id": paper_id,
                "user_id": user_id,
                "task_id": task_id,
                "failed_step": failed_step,
                "error": error,
            }
        )
        return True


def _task_run(task_id: str) -> dict:
    """构造 owner-aware 论文删除任务。"""
    return {
        "task_id": task_id,
        "task_type": "paper_delete",
        "user_id": "user-1",
        "payload": {"paper_id": "paper-1", "user_id": "user-1"},
    }


@pytest.mark.parametrize("failed_step", [name for _, name in DELETE_STEP_FUNCTIONS])
def test_each_delete_step_failure_is_persisted_and_retry_converges(monkeypatch, failed_step: str) -> None:
    """每个存储步骤首次失败后都应保留步骤，新任务重试同一 paper_id 可完成。"""
    repository = _FakeDeletionRepository()
    call_order: list[str] = []
    should_fail = {"value": True}

    def build_step(step_name: str):
        """创建一个仅在指定步骤首次调用时失败的异步步骤。"""
        async def step(context):  # noqa: ANN001
            call_order.append(step_name)
            if step_name == failed_step and should_fail["value"]:
                should_fail["value"] = False
                raise RuntimeError(f"{step_name} unavailable")
            return {"status": "done", "paper_id": context.paper_id}

        return step

    monkeypatch.setattr(paper_delete_task_module, "get_paper_deletion_repository", lambda: repository)
    for function_name, step_name in DELETE_STEP_FUNCTIONS:
        monkeypatch.setattr(paper_delete_task_module, function_name, build_step(step_name))

    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(paper_delete_task_module.run_paper_delete_task(_task_run(repository.task_id)))

    assert repository.failures[-1]["failed_step"] == failed_step
    repository.task_id = "task-delete-2"
    call_order.clear()
    result = asyncio.run(paper_delete_task_module.run_paper_delete_task(_task_run(repository.task_id)))

    assert call_order == [name for _, name in DELETE_STEP_FUNCTIONS]
    assert result["paper_id"] == "paper-1"
    assert result["status"] == "deleted"


class _Transaction:
    """记录仓储是否使用同一事务边界。"""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
        self.exited = True
        return False


class _ReferenceConnection:
    """记录引用清理 SQL，并返回一条受影响记录。"""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.tx = _Transaction()

    def transaction(self) -> _Transaction:
        """返回可追踪的事务上下文。"""
        return self.tx

    async def fetch(self, query, *args):  # noqa: ANN001
        """记录参数化 SQL，并模拟一条受影响行。"""
        self.queries.append(query)
        assert args == ("user-1", "paper-1")
        return [{"id": "affected"}]


class _Acquire:
    """模拟 asyncpg pool acquire 上下文。"""

    def __init__(self, connection) -> None:  # noqa: ANN001
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
        return False


class _Pool:
    """向仓储提供固定连接。"""

    def __init__(self, connection) -> None:  # noqa: ANN001
        self.connection = connection

    def acquire(self) -> _Acquire:
        """返回固定连接的 acquire 上下文。"""
        return _Acquire(self.connection)


class _MemoryRepository:
    """确认 memory evidence 清理复用相同数据库连接。"""

    def __init__(self) -> None:
        self.connection = None

    async def remove_paper_references(self, paper_id: str, *, user_id: str, connection) -> dict:  # noqa: ANN001
        """记录事务连接并返回记忆清理统计。"""
        self.connection = connection
        assert (paper_id, user_id) == ("paper-1", "user-1")
        return {"deleted_evidence_refs": 1, "stale_memory_ids": ["memory-1"]}


def test_database_reference_cleanup_uses_one_transaction(monkeypatch) -> None:
    """项目、会话、对话、笔记和记忆引用必须在同一连接事务内清理。"""
    connection = _ReferenceConnection()
    memory_repository = _MemoryRepository()

    async def fake_get_pool():
        """返回隔离的 fake pool。"""
        return _Pool(connection)

    monkeypatch.setattr(deletion_repository_module, "get_pool", fake_get_pool)
    monkeypatch.setattr(
        deletion_repository_module,
        "get_research_memory_repository",
        lambda: memory_repository,
    )
    result = asyncio.run(
        deletion_repository_module.PaperDeletionRepository().cleanup_database_references(
            "paper-1",
            user_id="user-1",
        )
    )

    joined_sql = "\n".join(connection.queries)
    assert connection.tx.entered and connection.tx.exited
    assert memory_repository.connection is connection
    assert all(
        table in joined_sql
        for table in ("project_paper_links", "project_scopes", "sessions", "conversations", "notes")
    )
    assert result["memory"]["stale_memory_ids"] == ["memory-1"]


def test_failure_state_and_hard_delete_are_bound_to_current_task(monkeypatch) -> None:
    """失败状态和最终硬删除都必须校验当前 owner、deleting 状态和 task_id。"""
    queries: list[str] = []

    class _StateConnection:
        """模拟状态更新与最终删除的 RETURNING 行。"""

        def transaction(self):
            """最终删除连同上传 tombstone 使用同一事务。"""
            return _Transaction()

        async def execute(self, query, *args):
            """记录上传来源和批次的清理，保持 owner 条件。"""
            queries.append(query)
            assert args == ("paper-1", "user-1")

        async def fetchrow(self, query, *args):  # noqa: ANN001
            queries.append(query)
            assert args[:3] == ("paper-1", "user-1", "task-delete-1")
            return {"paper_id": "paper-1"}

    async def fake_get_pool():
        """返回状态写入使用的 fake pool。"""
        return _Pool(_StateConnection())

    monkeypatch.setattr(deletion_repository_module, "get_pool", fake_get_pool)
    repository = deletion_repository_module.PaperDeletionRepository()
    failed = asyncio.run(
        repository.mark_deletion_failed(
            "paper-1",
            user_id="user-1",
            task_id="task-delete-1",
            failed_step="milvus",
            error="down",
        )
    )
    deleted = asyncio.run(
        repository.hard_delete_paper(
            "paper-1",
            user_id="user-1",
            task_id="task-delete-1",
        )
    )

    assert failed and deleted
    assert "deletion_status='delete_failed'" in queries[0]
    assert "deletion_status='deleting'" in queries[0]
    assert "DELETE FROM papers" in queries[1]
    assert "deletion_status='deleting'" in queries[1]
    assert all("user_id=$2" in query for query in queries[2:])


class _MemoryConnection:
    """模拟两条记忆分别失去全部证据和保留其他证据。"""

    def __init__(self) -> None:
        self.refs = {
            "memory-1": ["paper-1"],
            "memory-2": ["paper-1", "paper-2"],
        }
        self.stale_ids: list[str] = []
        self.version_ids: list[str] = []

    async def fetch(self, query, *args):  # noqa: ANN001
        """执行锁定或证据删除，并返回对应记录。"""
        if "SELECT item.memory_id" in query:
            paper_id = args[1]
            return [
                {"memory_id": memory_id}
                for memory_id, paper_ids in self.refs.items()
                if paper_id in paper_ids
            ]
        if "DELETE FROM memory_evidence_refs" in query:
            paper_id = args[1]
            deleted = []
            for memory_id, paper_ids in self.refs.items():
                if paper_id in paper_ids:
                    paper_ids.remove(paper_id)
                    deleted.append({"ref_id": f"{memory_id}:{paper_id}"})
            return deleted
        raise AssertionError(f"unexpected fetch: {query}")

    async def execute(self, query, *args):  # noqa: ANN001
        """记录保留其他证据的记忆版本推进。"""
        if "related_paper_ids" in query:
            return "UPDATE"
        if "version=version+1" in query:
            self.version_ids.append(args[0])
            return "UPDATE 1"
        raise AssertionError(f"unexpected execute: {query}")

    async def fetchval(self, query, *args):  # noqa: ANN001
        """返回目标记忆是否仍有其他证据。"""
        assert "SELECT EXISTS" in query
        return bool(self.refs[args[0]])

    async def fetchrow(self, query, *args):  # noqa: ANN001
        """记录失去全部证据后进入 needs_rebind 的记忆。"""
        assert "stale_state='needs_rebind'" in query
        self.stale_ids.append(args[0])
        self.version_ids.append(args[0])
        return {"memory_id": args[0]}


def test_memory_evidence_cleanup_marks_only_unbacked_memory_stale(monkeypatch) -> None:
    """删除证据引用后，仅失去全部证据的记忆进入 needs_rebind，重试不重复写版本。"""
    connection = _MemoryConnection()
    snapshots: list[str] = []
    repository = memory_repository_module.ResearchMemoryRepository()

    async def fake_snapshot(connection_arg, memory_id, *args):  # noqa: ANN001
        """记录版本快照目标并确认沿用调用方事务连接。"""
        assert connection_arg is connection
        snapshots.append(memory_id)

    monkeypatch.setattr(repository, "_snapshot", fake_snapshot)
    first = asyncio.run(
        repository.remove_paper_references(
            "paper-1",
            user_id="user-1",
            connection=connection,
        )
    )
    second = asyncio.run(
        repository.remove_paper_references(
            "paper-1",
            user_id="user-1",
            connection=connection,
        )
    )

    assert first["deleted_evidence_refs"] == 2
    assert first["stale_memory_ids"] == ["memory-1"]
    assert connection.stale_ids == ["memory-1"]
    assert connection.version_ids == ["memory-1", "memory-2"]
    assert snapshots == ["memory-1", "memory-2"]
    assert second["deleted_evidence_refs"] == 0
    assert second["updated_memory_ids"] == []


def test_managed_file_cleanup_is_idempotent_and_rejects_outside_path(tmp_path: Path) -> None:
    """受控上传和 MinerU 产物可重复删除，DATA_DIR 外路径必须显式拒绝。"""
    data_dir = tmp_path / "data"
    uploads = data_dir / "uploads"
    uploads.mkdir(parents=True)
    pdf_path = uploads / "paper-1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    artifact_dir = uploads / "paper-1.mineru"
    artifact_dir.mkdir()
    (artifact_dir / "source.md").write_text("body", encoding="utf-8")

    first = delete_managed_paper_files(
        "paper-1",
        pdf_path=str(pdf_path),
        data_dir=data_dir,
    )
    second = delete_managed_paper_files(
        "paper-1",
        pdf_path=str(pdf_path),
        data_dir=data_dir,
    )

    assert first["deleted_count"] == 2
    assert second["deleted_count"] == 0
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"keep")
    with pytest.raises(ValueError, match="DATA_DIR"):
        delete_managed_paper_files(
            "paper-1",
            pdf_path=str(outside),
            data_dir=data_dir,
        )
    assert outside.exists()
