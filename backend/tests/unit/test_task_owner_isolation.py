"""任务与通知 owner 隔离的定向回归测试。

覆盖冻结合同：
  - 用户可达的 create/get/list/events/cancel 都以服务端 owner 作为 SQL 过滤条件
  - 跨用户与历史无 owner 的单项资源在 SQL 层不可匹配（路由统一 404 的上游）
  - 通知列表与读状态按 user 过滤，不允许只按 log_id 篡改他人通知
  - worker 内部方法（claim/heartbeat/终态）签名不含浏览器 owner
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
import asyncpg

from app.api.facades import system_facade
from app.repositories import notification_repository, observability_repository, paper_repository, task_repository
from app.repositories.notification_repository import NotificationRepository
from app.repositories.task_repository import TaskRepository
from app.services import task_service
from app.services import paper_db
from app.config import get_settings
from app.services.task_service import create_research_task


# ---------- 任务仓储：owner 必填与 SQL 过滤 ----------


def test_create_task_run_requires_user_id() -> None:
    """创建任务必须显式传服务端 user_id，不允许缺省或匿名默认。"""
    params = inspect.signature(TaskRepository.create_task_run).parameters
    assert "user_id" in params
    assert params["user_id"].default is inspect.Parameter.empty


def test_create_task_run_writes_user_id_column(monkeypatch) -> None:
    """INSERT 必须把 owner 写入独立 user_id 列。"""
    captured: dict[str, Any] = {}

    async def fake_fetchrow(sql: str, *args: Any) -> Any:
        captured["sql"] = sql
        captured["args"] = args
        return {
            "task_id": args[0], "user_id": args[1], "task_type": args[2],
            "status": "queued", "progress": 0, "message": "", "payload": "{}",
            "cancel_requested": False, "created_at": "now", "updated_at": "now",
        }

    monkeypatch.setattr(task_repository, "fetchrow", fake_fetchrow)

    result = asyncio.run(
        TaskRepository().create_task_run(
            "upload_ingest",
            {},
            "入队",
            task_id=None,
            user_id="user-1",
        )
    )

    assert "user_id" in captured["sql"]
    assert captured["args"][1] == "user-1"
    assert result["user_id"] == "user-1"


def test_create_task_run_rejects_blank_user_id(monkeypatch) -> None:
    """任务 owner 为空白时必须在 SQL 执行前失败。"""
    called = False

    async def fail_fetchrow(sql: str, *args: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("空 owner 不应写入数据库")

    monkeypatch.setattr(task_repository, "fetchrow", fail_fetchrow)

    with pytest.raises(ValueError, match="非空 user_id"):
        asyncio.run(TaskRepository().create_task_run("upload_ingest", {}, "入队", user_id=" "))
    assert called is False


def test_list_task_runs_filters_by_user(monkeypatch) -> None:
    """任务列表必须按 user_id 过滤，参数第一个是当前用户。"""
    captured: dict[str, Any] = {}

    async def fake_fetch(sql: str, *args: Any) -> list[Any]:
        captured["sql"] = sql
        captured["args"] = args
        return []

    monkeypatch.setattr(task_repository, "fetch", fake_fetch)

    asyncio.run(TaskRepository().list_task_runs(None, user_id="user-1", limit=50, offset=0))

    assert "user_id=$1" in captured["sql"]
    assert captured["args"][0] == "user-1"


def test_get_task_run_filters_by_user(monkeypatch) -> None:
    """单项任务读取必须按 (task_id, user_id) 匹配；他人/无 owner 任务取不到。"""
    captured: dict[str, Any] = {}

    async def fake_fetchrow(sql: str, *args: Any) -> Any:
        captured["sql"] = sql
        captured["args"] = args
        return None

    monkeypatch.setattr(task_repository, "fetchrow", fake_fetchrow)

    result = asyncio.run(TaskRepository().get_task_run("task-a", user_id="user-1"))

    assert result is None
    assert "user_id=$2" in captured["sql"]
    assert captured["args"] == ("task-a", "user-1")


def test_request_task_cancel_filters_by_user(monkeypatch) -> None:
    """取消必须先按 (task_id, user_id) 命中；跨用户返回 None，running 分支 UPDATE 也带 user 条件。"""
    calls: list[tuple[str, tuple[Any, ...]]] = []
    rows: list[Any] = [
        None,  # 请求1 SELECT：他人任务 → 不可见
        {"task_id": "task-a", "user_id": "user-1", "task_type": "deep_research", "status": "running",
         "progress": 30, "message": "", "payload": "{}", "result": None, "error": None,
         "cancel_requested": False, "worker_id": None, "lease_until": None,
         "created_at": "now", "updated_at": "now", "started_at": "now", "finished_at": None},
        {"task_id": "task-a", "user_id": "user-1", "task_type": "deep_research", "status": "running",
         "progress": 30, "message": "已请求取消任务", "payload": "{}", "result": None, "error": None,
         "cancel_requested": True, "worker_id": None, "lease_until": None,
         "created_at": "now", "updated_at": "now", "started_at": "now", "finished_at": None},
    ]

    async def fake_fetchrow(sql: str, *args: Any) -> Any:
        calls.append((sql, args))
        return rows.pop(0)

    monkeypatch.setattr(task_repository, "fetchrow", fake_fetchrow)

    repo = TaskRepository()

    # 他人任务：SELECT 带 user 条件 → 返回 None（路由 404）
    result = asyncio.run(repo.request_task_cancel("task-a", "取消", user_id="user-2"))
    assert result is None
    assert "user_id=$2" in calls[0][0]
    assert calls[0][1] == ("task-a", "user-2")

    # 自己任务：running 分支 UPDATE 也带 user 条件
    result = asyncio.run(repo.request_task_cancel("task-a", "取消", user_id="user-1"))
    assert result is not None
    assert result["cancel_requested"] is True
    assert "user_id=$2" in calls[1][0]
    assert calls[1][1] == ("task-a", "user-1")
    assert "user_id=$2" in calls[2][0]


def test_list_task_events_filters_by_task_owner(monkeypatch) -> None:
    """任务事件必须 JOIN task_runs 按 owner 过滤，避免从 event 侧泄露他人任务。"""
    captured: dict[str, Any] = {}

    async def fake_fetch(sql: str, *args: Any) -> list[Any]:
        captured["sql"] = sql
        captured["args"] = args
        return []

    monkeypatch.setattr(task_repository, "fetch", fake_fetch)

    asyncio.run(TaskRepository().list_task_events("task-a", user_id="user-1", limit=80))

    assert "JOIN task_runs" in captured["sql"]
    assert "user_id" in captured["sql"]
    assert captured["args"][:2] == ("task-a", "user-1")


# ---------- 任务服务：owner 贯穿与禁止匿名默认 ----------


def test_create_research_task_writes_owner(monkeypatch) -> None:
    """deep research 任务创建必须携带服务端 user_id（当前实现完全缺失）。"""
    captured: dict[str, Any] = {}

    def fake_create_task_run(task_type, payload, message, *, task_id, user_id):  # noqa: ANN001
        captured.update(task_type=task_type, payload=payload, user_id=user_id)
        return {
            "task_id": "t-1", "task_type": task_type, "user_id": user_id,
            "status": "queued", "progress": 0, "message": message, "payload": payload,
            "cancel_requested": False, "created_at": "now", "updated_at": "now",
        }

    def fake_add_task_event(task_id, event_type, message, payload=None):  # noqa: ANN001
        return {}

    monkeypatch.setattr(task_service.paper_db, "create_task_run", fake_create_task_run)
    monkeypatch.setattr(task_service.paper_db, "add_task_event", fake_add_task_event)

    task = asyncio.run(create_research_task("量子纠缠", "session-1", user_id="user-1"))

    assert captured["user_id"] == "user-1"
    assert captured["payload"]["user_id"] == "user-1"
    assert task["user_id"] == "user-1"


def test_create_lightrag_task_writes_owner(monkeypatch) -> None:
    """旧建图地址也必须经过 owner-aware 上传操作，不能携带任意正文直写图谱。"""
    from app.services import upload_service
    captured: dict[str, Any] = {}

    async def fake_operation(paper_id, operation, *, user_id):
        """记录服务端传递的原论文及账号。"""
        captured.update(paper_id=paper_id, operation=operation, user_id=user_id)
        return {"task_id": "t-2", "user_id": user_id, "payload": {"paper_id": paper_id}}

    monkeypatch.setattr(upload_service, "request_upload_operation", fake_operation)

    task = asyncio.run(
        task_service.create_lightrag_task(
            "paper-1",
            user_id="user-1",
        )
    )

    assert captured == {"paper_id": "paper-1", "operation": "graph", "user_id": "user-1"}
    assert task["user_id"] == "user-1"


def test_create_upload_task_rejects_missing_owner() -> None:
    """上传任务不允许 anonymous 默认，缺 user_id 必须 TypeError。"""
    with pytest.raises(TypeError):
        asyncio.run(task_service.create_upload_task("a.pdf", b"%PDF"))


# ---------- 通知仓储：user 过滤 ----------


def test_notification_list_filters_by_user(monkeypatch) -> None:
    """通知列表必须按 user_id 过滤，他人/无 owner 的通知不可见。"""
    captured: dict[str, Any] = {}

    async def fake_fetch(sql: str, *args: Any) -> list[Any]:
        captured["sql"] = sql
        captured["args"] = args
        return []

    monkeypatch.setattr(notification_repository, "fetch", fake_fetch)

    asyncio.run(NotificationRepository().list_notifications("user-1"))

    assert "user_id=$2" in captured["sql"]
    assert captured["args"] == ("push_notification", "user-1", 50)


def test_notification_set_read_state_filters_by_user(monkeypatch) -> None:
    """通知读状态必须按 (log_id, user_id) 更新；跨用户返回 False（路由 404）。"""
    captured: dict[str, Any] = {}

    async def fake_fetchval(sql: str, *args: Any) -> Any:
        captured["sql"] = sql
        captured["args"] = args
        return None

    monkeypatch.setattr(notification_repository, "fetchval", fake_fetchval)

    result = asyncio.run(NotificationRepository().set_read_state("log-b", "user-1", True))

    assert result is False
    assert "user_id=$3" in captured["sql"]
    assert captured["args"] == (True, "log-b", "user-1")


def test_push_notification_requires_owner(monkeypatch) -> None:
    """面向用户的通知写入缺少 owner 时必须显式失败。"""
    with pytest.raises(ValueError, match="user_id"):
        paper_db.write_log("push_notification", "recommendation", "论文")
    with pytest.raises(ValueError, match="user_id"):
        paper_db.write_log("push_notification", "recommendation", "论文", user_id=" ")

    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(
            observability_repository.ObservabilityRepository().write_log(
                "push_notification", "recommendation", "论文", 1, 0, None
            )
        )
    with pytest.raises(ValueError, match="user_id"):
        asyncio.run(
            observability_repository.ObservabilityRepository().write_log(
                "push_notification", "recommendation", "论文", 1, 0, None, user_id=" "
            )
        )

    captured: dict[str, Any] = {}

    async def fake_execute(sql: str, *args: Any) -> None:
        captured["sql"] = sql
        captured["args"] = args

    monkeypatch.setattr(observability_repository, "execute", fake_execute)
    asyncio.run(
        observability_repository.ObservabilityRepository().write_log(
            "push_notification", "recommendation", "论文", 1, 0, None, user_id="user-1"
        )
    )
    assert captured["args"][1] == "user-1"


def test_notification_list_database_error_is_visible(monkeypatch) -> None:
    """通知数据库故障不能静默伪装为空列表。"""
    def fail_list_notifications(user_id):  # noqa: ANN001
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(system_facade.paper_db, "list_notifications", fail_list_notifications)
    with pytest.raises(RuntimeError, match="database unavailable"):
        system_facade.list_notifications_payload("user-1")


# ---------- PostgreSQL：迁移可重复性与真实 owner 隔离 ----------


async def _create_legacy_owner_schema(conn: Any, schema_name: str) -> None:
    """创建不含 owner 列的历史表，用于验证可重复迁移不会篡改旧记录。"""
    await conn.execute(f'CREATE SCHEMA "{schema_name}"')
    await conn.execute(f'SET LOCAL search_path TO "{schema_name}", public')
    await conn.execute(
        """
        CREATE TABLE task_runs (
            task_id TEXT PRIMARY KEY, task_type TEXT NOT NULL, status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '',
            payload JSONB NOT NULL DEFAULT '{}'::jsonb, result JSONB, error TEXT,
            cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, worker_id TEXT,
            lease_until TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(), started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ
        );
        CREATE TABLE logs (
            log_id TEXT PRIMARY KEY, session_id TEXT, event_type TEXT, node_name TEXT,
            query TEXT, result_count INTEGER, latency_ms INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(), is_read BOOLEAN DEFAULT FALSE
        );
        INSERT INTO task_runs (task_id, task_type, status) VALUES ('legacy-task', 'deep_research', 'queued');
        INSERT INTO logs (log_id, event_type) VALUES ('legacy-log', 'push_notification');
        """
    )


async def _assert_owner_migration(conn: Any, schema_name: str) -> None:
    """验证迁移重复执行后 owner 列和索引存在，历史 NULL 记录保持未归属。"""
    await conn.execute(paper_repository.SCHEMA_SQL)
    await conn.execute(paper_repository.SCHEMA_SQL)
    columns = await conn.fetch(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema=$1 AND table_name = ANY($2::text[]) AND column_name='user_id'",
        schema_name,
        ["task_runs", "logs"],
    )
    assert {(row["table_name"], row["column_name"]) for row in columns} == {
        ("task_runs", "user_id"),
        ("logs", "user_id"),
    }
    indexes = await conn.fetch(
        "SELECT indexname FROM pg_indexes WHERE schemaname=$1 AND indexname = ANY($2::text[])",
        schema_name,
        ["idx_task_runs_user_created", "idx_task_runs_user_status_created", "idx_logs_user_created"],
    )
    assert {row["indexname"] for row in indexes} == {
        "idx_task_runs_user_created",
        "idx_task_runs_user_status_created",
        "idx_logs_user_created",
    }
    assert await conn.fetchval("SELECT user_id FROM task_runs WHERE task_id='legacy-task'") is None
    assert await conn.fetchval("SELECT user_id FROM logs WHERE log_id='legacy-log'") is None


async def _seed_owner_isolation_records(conn: Any) -> None:
    """写入 A、B 和历史 NULL owner 记录，覆盖用户可达的隔离查询路径。"""
    await conn.execute(
        """
        INSERT INTO task_runs (task_id, user_id, task_type, status) VALUES
        ('task-a', 'user-a', 'deep_research', 'queued'),
        ('task-b', 'user-b', 'deep_research', 'queued');
        INSERT INTO task_events (event_id, task_id, event_type, message) VALUES
        ('event-a', 'task-a', 'queued', 'A 的任务'),
        ('event-b', 'task-b', 'queued', 'B 的任务'),
        ('event-legacy', 'legacy-task', 'queued', '历史任务');
        INSERT INTO logs (log_id, user_id, event_type, node_name, query) VALUES
        ('log-a', 'user-a', 'push_notification', 'task', 'A 的通知'),
        ('log-b', 'user-b', 'push_notification', 'task', 'B 的通知');
        """
    )


def _bind_owner_repositories(monkeypatch: pytest.MonkeyPatch, conn: Any) -> None:
    """将 repository 的数据库原语定向到同一事务连接，避免测试逃逸到真实 schema。"""
    async def fetch(sql: str, *args: Any) -> list[Any]:
        return await conn.fetch(sql, *args)

    async def fetchrow(sql: str, *args: Any) -> Any:
        return await conn.fetchrow(sql, *args)

    async def fetchval(sql: str, *args: Any) -> Any:
        return await conn.fetchval(sql, *args)

    monkeypatch.setattr(task_repository, "fetch", fetch)
    monkeypatch.setattr(task_repository, "fetchrow", fetchrow)
    monkeypatch.setattr(notification_repository, "fetch", fetch)
    monkeypatch.setattr(notification_repository, "fetchval", fetchval)


async def _assert_task_repository_isolation() -> None:
    """验证任务列表、详情、事件和取消都无法穿透 A/B/NULL owner 边界。"""
    repository = TaskRepository()
    assert [task["task_id"] for task in await repository.list_task_runs(None, user_id="user-a", limit=10, offset=0)] == ["task-a"]
    assert [task["task_id"] for task in await repository.list_task_runs(None, user_id="user-b", limit=10, offset=0)] == ["task-b"]
    assert await repository.get_task_run("task-b", user_id="user-a") is None
    assert await repository.get_task_run("legacy-task", user_id="user-a") is None
    assert [event["event_id"] for event in await repository.list_task_events("task-a", user_id="user-a", limit=10)] == ["event-a"]
    assert await repository.list_task_events("task-b", user_id="user-a", limit=10) == []
    assert await repository.list_task_events("legacy-task", user_id="user-a", limit=10) == []
    assert await repository.request_task_cancel("task-b", "取消", user_id="user-a") is None
    assert await repository.request_task_cancel("legacy-task", "取消", user_id="user-a") is None
    assert (await repository.request_task_cancel("task-a", "取消", user_id="user-a"))["status"] == "cancelled"


async def _assert_notification_repository_isolation(conn: Any) -> None:
    """验证通知列表和已读状态均按 owner 过滤，NULL 历史通知不会暴露。"""
    repository = NotificationRepository()
    assert [item["log_id"] for item in await repository.list_notifications("user-a")] == ["log-a"]
    assert [item["log_id"] for item in await repository.list_notifications("user-b")] == ["log-b"]
    assert not await repository.set_read_state("log-b", "user-a", True)
    assert not await repository.set_read_state("legacy-log", "user-a", True)
    assert await repository.set_read_state("log-a", "user-a", True)
    assert await conn.fetchval("SELECT is_read FROM logs WHERE log_id='log-a'") is True


async def _verify_postgresql_owner_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """使用本测试独占连接和回滚 schema，事件循环结束前关闭全部数据库传输。"""
    conn = await asyncpg.connect(get_settings().database_url, timeout=5, command_timeout=5)
    schema_name = f"owner_isolation_{uuid.uuid4().hex}"
    try:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await _create_legacy_owner_schema(conn, schema_name)
            await _assert_owner_migration(conn, schema_name)
            await _seed_owner_isolation_records(conn)
            _bind_owner_repositories(monkeypatch, conn)
            await _assert_task_repository_isolation()
            await _assert_notification_repository_isolation(conn)
        finally:
            await transaction.rollback()
    finally:
        await conn.close()


def test_postgresql_owner_isolation_migration_and_repository(monkeypatch) -> None:
    """真实 PostgreSQL 迁移与仓储隔离测试，整个 schema 通过事务回滚清理。"""
    asyncio.run(_verify_postgresql_owner_isolation(monkeypatch))


# ---------- 路由：当前用户贯穿 ----------


def test_task_routes_pass_current_user(monkeypatch) -> None:
    """task_api 全部用户可达入口都必须把 get_user_id(user) 贯穿到 service。"""
    from fastapi import HTTPException

    from app.api.routes import task_api as task_route

    calls: list[str] = []

    async def fake_list_tasks(statuses, *, limit, offset, user_id):  # noqa: ANN001
        calls.append(user_id)
        return []

    async def fake_get_task_or_raise(task_id, *, user_id):  # noqa: ANN001
        calls.append(user_id)
        raise LookupError("任务不存在")

    async def fake_cancel_task(task_id, *, user_id, message="已请求取消任务"):  # noqa: ANN001
        calls.append(user_id)
        return {}

    async def fake_create_research_task(query, session_id, *, user_id):  # noqa: ANN001
        calls.append(user_id)
        return {}

    async def fake_list_task_events(task_id, *, limit, user_id):  # noqa: ANN001
        calls.append(user_id)
        return []

    monkeypatch.setattr(task_route.task_service, "list_tasks", fake_list_tasks)
    monkeypatch.setattr(task_route.task_service, "get_task_or_raise", fake_get_task_or_raise)
    monkeypatch.setattr(task_route.task_service, "cancel_task", fake_cancel_task)
    monkeypatch.setattr(task_route.task_service, "create_research_task", fake_create_research_task)
    monkeypatch.setattr(task_route.task_service, "list_task_events", fake_list_task_events)
    async def fake_resolve_identity(**kwargs):  # noqa: ANN003
        return SimpleNamespace(session_id=kwargs["session_id"])

    monkeypatch.setattr(task_route.session_identity_facade, "resolve_conversation_identity", fake_resolve_identity)

    user = {"user_id": "user-1", "role": "user"}
    req = SimpleNamespace(query="q", session_id="s")

    asyncio.run(task_route.list_tasks(None, 50, 0, user))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(task_route.get_task("task-a", user))
    assert exc_info.value.status_code == 404
    asyncio.run(task_route.cancel_task("task-a", user))
    asyncio.run(task_route.create_research_task(req, user))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(task_route.get_task_events("task-a", 80, user))
    assert exc_info.value.status_code == 404

    assert calls == ["user-1"] * 5


def test_notification_routes_pass_current_user(monkeypatch) -> None:
    """通知读状态与列表都必须把当前用户贯穿到 facade。"""
    from app.api.routes import system_api as system_route

    calls: list[str] = []

    def fake_list_notifications_payload(user_id):  # noqa: ANN001
        calls.append(user_id)
        return {"notifications": []}

    def fake_mark_notification_read_payload(log_id, user_id, read):  # noqa: ANN001
        calls.append(user_id)
        return {"log_id": log_id, "read": read}

    monkeypatch.setattr(system_route.system_facade, "list_notifications_payload", fake_list_notifications_payload)
    monkeypatch.setattr(system_route.system_facade, "mark_notification_read_payload", fake_mark_notification_read_payload)

    user = {"user_id": "user-1", "role": "user"}
    body = SimpleNamespace(read=True)

    asyncio.run(system_route.get_notifications(user))
    asyncio.run(system_route.mark_notification_read("log-a", body, user))

    assert calls == ["user-1"] * 2


# ---------- worker 内部方法签名不变量 ----------


def test_worker_internal_methods_have_no_user_param() -> None:
    """worker 内部能力不得接受浏览器 owner 参数，防止用户查询意外退化。"""
    for name in (
        "claim_next_task",
        "heartbeat_task",
        "complete_task_run",
        "fail_task_run",
        "update_task_progress",
        "mark_task_cancelled",
        "is_task_cancel_requested",
    ):
        sig = inspect.signature(getattr(TaskRepository, name))
        assert "user_id" not in sig.parameters, name
