"""【研究交付】【离线端口】真实仓储的事务内存替身，作者：xxx，2026-10-04 13:00:00。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
import pytest

from app.repositories.result_delivery_repository import ResultDeliveryRepository


@pytest.fixture(autouse=True)
def offline_memory_callback(monkeypatch):
    """参数为补丁工具；交付组隔离候选模块端口，候选正式流程由记忆组独立验证。"""
    from app.services.internal_research import research_memory_writeback

    async def committed_candidate(committed):
        """参数为提交回执；只在保存成功后返回离线候选事实，不访问服务。"""
        assert committed["persistence_status"] == "saved"
        return committed["run"]["metadata"].get("memory_writeback") or {"status": "skipped", "candidate_ids": []}

    monkeypatch.setattr(research_memory_writeback, "after_result_committed", committed_candidate)


class Transaction:
    """保存连接内数据快照，异常时回滚所有投影。"""

    def __init__(self, connection):
        """参数为内存连接；记录事务快照，无返回值。"""
        self.connection = connection

    async def __aenter__(self):
        """进入事务，无参数；记录全部表的副本。"""
        self.snapshot = deepcopy(self.connection.data)
        self.connection.transactions += 1
        return self

    async def __aexit__(self, kind, error, traceback):
        """参数为事务异常；异常时恢复所有表，无返回值。"""
        if error is not None:
            self.connection.data = self.snapshot


class Connection:
    """只实现被测真实仓储使用的 SQL 边界，未知语句直接失败。"""

    def __init__(self, run=None, *, fail_at=None):
        """参数为目标和故障步骤；建立用户、消息、会话及运行表。"""
        target = {
            "run_id": "target", "user_id": "user", "conversation_id": "conversation", "session_id": "session",
            "operation_kind": "research", "status": "running", "query": "原问题", "metadata": {},
            "delivery_revision": 0, "pending_artifact": None, "created_at": datetime.now(timezone.utc),
            **(run or {}),
        }
        self.data = {"runs": {target["run_id"]: deepcopy(target)}, "messages": {}, "events": [],
                     "sessions": {"session": {"current_run_id": target["run_id"], "status": "running"}},
                     "conversations": {target["conversation_id"]: {"id": target["conversation_id"],
                                          "user_id": target["user_id"], "message_count": 0}}}
        self.fail_at = fail_at
        self.transactions = 0
        self.term_connections = []

    async def __aenter__(self):
        """获取连接，无参数；返回当前连接。"""
        return self

    async def __aexit__(self, *_args):
        """释放连接，无返回值。"""

    def transaction(self):
        """无参数；返回能回滚的内存事务。"""
        return Transaction(self)

    def check(self, point):
        """参数为写入步骤；在指定真实边界注入失败。"""
        if self.fail_at == point:
            raise OSError(f"offline {point} failed")

    async def fetch(self, query, *args):
        """参数为仓储锁行查询；返回 owner 范围内的目标和操作。"""
        assert "FROM rag_runs" in query
        return [deepcopy(self.data["runs"][key]) for key in args[1]
                if key in self.data["runs"] and self.data["runs"][key]["user_id"] == args[0]]

    async def fetchrow(self, query, *args):
        """参数为真实仓储语句；按明确边界读取或写入一行。"""
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT * FROM rag_runs"):
            return deepcopy(self.data["runs"].get(args[1]))
        if normalized.startswith("SELECT * FROM sessions"):
            return deepcopy(self.data["sessions"].get(args[1]))
        if normalized.startswith("SELECT id FROM conversations"):
            return deepcopy(self.data["conversations"].get(args[1]))
        if normalized.startswith(("SELECT id FROM messages", "SELECT m.* FROM messages")):
            return next((deepcopy(row) for row in self.data["messages"].values()
                         if row["role"] == "assistant" and row["metadata"]["run_id"] == args[1]), None)
        if normalized.startswith("UPDATE rag_runs SET answer"):
            self.check("target")
            target = self.data["runs"][args[1]]
            target.update(answer=args[2], metadata=json.loads(args[3]), delivery_revision=args[4], pending_artifact=None)
            return deepcopy(target)
        if normalized.startswith("INSERT INTO messages"):
            self.check("assistant")
            row = {"id": args[0], "conversation_id": args[1], "content": args[2], "role": "assistant",
                   "render_mode": "markdown", "metadata": json.loads(args[3]), "created_at": args[4]}
            self.data["messages"][row["id"]] = row
            return deepcopy(row)
        raise AssertionError(normalized)

    async def execute(self, query, *args):
        """参数为真实写语句；更新内存表，未知 SQL 显式失败。"""
        normalized = " ".join(query.split())
        if normalized.startswith("UPDATE rag_runs SET pending_artifact"):
            self.check("stage")
            self.data["runs"][args[1]]["pending_artifact"] = json.loads(args[2])
        elif normalized.startswith("INSERT INTO messages"):
            self.check("user")
            self.data["messages"].setdefault(args[0], {"id": args[0], "role": "user", "content": args[2],
                "conversation_id": args[1], "metadata": json.loads(args[3]), "created_at": args[4]})
        elif normalized.startswith("UPDATE conversations"):
            self.check("conversation")
            self.data["conversations"][args[1]]["message_count"] = len(self.data["messages"])
        elif normalized.startswith("UPDATE sessions SET current_result=$3"):
            self.check("session")
            self.data["sessions"][args[1]].update(current_result=json.loads(args[2]), result_run_id=args[3],
                delivery_revision=args[4], status=args[5] or self.data["sessions"][args[1]]["status"])
        elif normalized.startswith("UPDATE sessions SET status=$5"):
            for row in self.data["sessions"].values():
                if row["current_run_id"] == args[1]:
                    row.update(status="failed", last_error=args[2])
        elif normalized.startswith("UPDATE rag_runs SET metadata"):
            self.data["runs"][args[1]]["metadata"].update(json.loads(args[2]))
        elif normalized.startswith("UPDATE messages m SET metadata"):
            self.check("memory_message")
            for row in self.data["messages"].values():
                metadata = row.get("metadata", {})
                if row["role"] == "assistant" and metadata.get("run_id") == args[1] and metadata.get("delivery_revision") == args[2]:
                    metadata.update(json.loads(args[3]))
        elif normalized.startswith("UPDATE sessions SET current_result=current_result"):
            for row in self.data["sessions"].values():
                if row.get("result_run_id") == args[1] and row.get("delivery_revision") == args[2]:
                    row["current_result"].update(json.loads(args[3]))
        else:
            raise AssertionError(normalized)


class RunPort:
    """只替换 terminal SQL，断言使用结果事务的同一个连接。"""

    async def finish_run(self, *, connection, **kwargs):
        """参数为真实仓储 terminal 请求；按状态 CAS 保存终态和唯一事件。"""
        connection.check("terminal")
        connection.term_connections.append(connection)
        row = connection.data["runs"][kwargs["run_id"]]
        if row["status"] != kwargs["expected_status"]:
            return None
        row.update(status=kwargs["status"], error=kwargs.get("error"))
        row["metadata"].update(deepcopy(kwargs.get("metadata") or {}))
        connection.data["events"].append({"run_id": row["run_id"], "payload": deepcopy(kwargs["payload"])})
        return deepcopy(row)


class Pool:
    """只提供一个可追踪连接。"""

    def __init__(self, connection):
        """参数为连接；保存替身端口。"""
        self.connection = connection

    def acquire(self):
        """无参数；返回同连接上下文。"""
        return self.connection


def repository(connection):
    """参数为内存连接；返回真实结果仓储，SQL 端口无外联能力。"""
    async def provide_pool():
        """无参数；提供测试连接池。"""
        return Pool(connection)

    return ResultDeliveryRepository(pool_provider=provide_pool, run_repository=RunPort())


def final_operation(connection, kind="persist_only", operation_id="save"):
    """参数为连接和轻量类型；建立指向同一目标的新操作，返回其副本。"""
    target = connection.data["runs"]["target"]
    operation = {**deepcopy(target), "run_id": operation_id, "target_run_id": "target", "status": "running",
                 "operation_kind": kind, "expected_delivery_revision": target["delivery_revision"], "metadata": {},
                 "pending_artifact": None, "answer": None}
    connection.data["runs"][operation_id] = operation
    return deepcopy(operation)


class OperationPort:
    """保留真正操作服务，只替换它的持久化端口。"""

    def __init__(self, connection):
        """参数为内存连接；绑定运行表。"""
        self.connection = connection

    async def get_run(self, *, user_id, run_id):
        """参数为已鉴权身份；返回对应运行。"""
        row = self.connection.data["runs"].get(run_id)
        return deepcopy(row) if row and row["user_id"] == user_id else None

    async def create_operation(self, *, operation, session_command):
        """参数为真实操作服务请求；保存创建合同，唯一性由执行组独立验证。"""
        created = {**deepcopy(operation), "status": "running", "delivery_revision": 0, "pending_artifact": None,
                   "created_at": datetime.now(timezone.utc)}
        self.connection.data["runs"][created["run_id"]] = created
        return deepcopy(created), True


class InlineExecutor:
    """执行器依赖端口；实际执行器的停止和排他另有正式回归。"""

    async def execute(self, *, runner_factory, **_kwargs):
        """参数为已取得执行权的 runner；调用真实业务链并返回结果。"""
        return await runner_factory()
