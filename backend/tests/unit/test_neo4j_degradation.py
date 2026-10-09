"""Phase 1 — Slice C: Neo4j 优雅降级（R2）契约测试。

红测先行：本文件先于实现编写。覆盖：

1. `_classify_exception` 对白名单异常返回 "failed"，未知异常返回 None
2. `Neo4jClient.execute_read` 捕获白名单异常 → 抛 `Neo4jDegradedError(reason=类名)`
3. `Neo4jClient.execute_read` 未知异常照样向上抛（不被静默吞）
4. `neo4j_graph_search_tool` 捕获 `Neo4jDegradedError`：
   - `Command.update["source_status"]["neo4j"] == "failed"`
   - `Command.update["source_status_reasons"]["neo4j"] == 异常类名`
   - 不向上抛
5. 设置超时时传给 driver 的参数等于 settings 字段（不允许裸数字）。
6. `execute_read` / `execute_write` 把读超时真正作为 `begin_transaction(timeout=...)` 传入
   （用 `session.run` 的 kwargs 传会退化成 Cypher 参数、静默失效），且结果在事务退出前取完。
"""

from __future__ import annotations

from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired

# 私有模块导入：SocketDeadlineExceededError 实际定义在 neo4j._exceptions
from neo4j._exceptions import SocketDeadlineExceededError  # type: ignore[import]

from app.services.graph.neo4j_client import (
    Neo4jClient,
    Neo4jDegradedError,
    _NEO4J_FAILURE_STATUS_MAP,
    _classify_exception,
)


# ----------------------------- _classify_exception -----------------------------


@pytest.mark.parametrize(
    "exc",
    [
        SocketDeadlineExceededError("deadline"),
        ServiceUnavailable("unavailable"),
        SessionExpired("expired"),
        AuthError("auth"),
    ],
)
def test_classify_known_exceptions_returns_failed(exc: BaseException) -> None:
    result = _classify_exception(exc)
    assert result == "failed", f"{type(exc).__name__} 应被分类为 failed，实际 {result!r}"


def test_unknown_exception_returns_none() -> None:
    assert _classify_exception(ValueError("not in whitelist")) is None
    assert _classify_exception(RuntimeError("also not")) is None


def test_failure_map_covers_all_required_classes() -> None:
    required = {SocketDeadlineExceededError, ServiceUnavailable, SessionExpired, AuthError}
    assert required.issubset(set(_NEO4J_FAILURE_STATUS_MAP.keys())), (
        f"_NEO4J_FAILURE_STATUS_MAP 必须显式覆盖 {required}，"
        f"实际 {set(_NEO4J_FAILURE_STATUS_MAP.keys())}"
    )
    assert all(status == "failed" for status in _NEO4J_FAILURE_STATUS_MAP.values())


# ----------------------------- Neo4jDegradedError ------------------------------


def test_degraded_error_carries_reason() -> None:
    exc = Neo4jDegradedError(reason="SocketDeadlineExceededError")
    assert exc.reason == "SocketDeadlineExceededError"
    assert "SocketDeadlineExceededError" in str(exc)


# ----------------------------- execute_read 包装 -------------------------------


class _FakeSessionRaising:
    """模拟 session() 上下文管理器，进入时抛指定异常。"""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __enter__(self) -> "_FakeSessionRaising":
        raise self._exc

    def __exit__(self, *args: Any) -> bool:
        return False


class _FakeDriver:
    """模拟 neo4j driver；session() 调用时返回会抛错的上下文管理器。"""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.session_calls: list[dict[str, Any]] = []

    def session(self, **kwargs: Any) -> _FakeSessionRaising:
        self.session_calls.append(kwargs)
        return _FakeSessionRaising(self._exc)

    def close(self) -> None:
        pass

    def verify_connectivity(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_cls,reason_name",
    [
        (SocketDeadlineExceededError, "SocketDeadlineExceededError"),
        (ServiceUnavailable, "ServiceUnavailable"),
        (SessionExpired, "SessionExpired"),
        (AuthError, "AuthError"),
    ],
)
async def test_execute_read_raises_degraded_on_known_exception(
    exc_cls: type[BaseException], reason_name: str
) -> None:
    client = Neo4jClient()
    # 已知异常应被翻译成 Neo4jDegradedError，reason 取自 __class__.__name__
    client._driver = _FakeDriver(exc_cls("trigger"))

    with pytest.raises(Neo4jDegradedError) as exc_info:
        await client.execute_read("MATCH (n) RETURN n")

    assert exc_info.value.reason == reason_name


@pytest.mark.asyncio
async def test_execute_read_propagates_unknown_exception() -> None:
    """白名单外的异常必须原样向上抛，避免静默吞错。"""
    client = Neo4jClient()
    client._driver = _FakeDriver(ValueError("not a neo4j error"))

    with pytest.raises(ValueError, match="not a neo4j error"):
        await client.execute_read("MATCH (n) RETURN n")


@pytest.mark.asyncio
async def test_execute_write_raises_degraded_on_known_exception() -> None:
    client = Neo4jClient()
    client._driver = _FakeDriver(ServiceUnavailable("down"))

    with pytest.raises(Neo4jDegradedError) as exc_info:
        await client.execute_write("CREATE (n)")

    assert exc_info.value.reason == "ServiceUnavailable"


@pytest.mark.asyncio
async def test_execute_write_propagates_unknown_exception() -> None:
    client = Neo4jClient()
    client._driver = _FakeDriver(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await client.execute_write("CREATE (n)")


# ----------------------------- Driver 超时参数 ---------------------------------


def test_settings_timeout_values_passed_to_driver() -> None:
    """driver 构造时必须使用 settings 字段，不允许裸数字。"""
    from app.config import get_settings

    settings = get_settings()
    expected_connect = settings.neo4j_connect_timeout_seconds

    captured: dict[str, Any] = {}

    class _RecordingDriver:
        def close(self) -> None:
            pass

    def _fake_graph_database_driver(uri: str, *, auth: Any = None, **kwargs: Any) -> _RecordingDriver:
        captured["uri"] = uri
        captured["auth"] = auth
        captured["kwargs"] = kwargs
        return _RecordingDriver()

    client = Neo4jClient()
    # 强制启用并避开真实 driver
    client._driver = None
    with patch("app.services.graph.neo4j_client.GraphDatabase") as fake_module:
        fake_module.driver = _fake_graph_database_driver
        client._ensure_driver()

    # 至少 connection_timeout 必须等于 settings 字段，不能是裸数字
    assert "connection_timeout" in captured["kwargs"], (
        f"driver 必须接收 connection_timeout 参数；实际 kwargs={list(captured['kwargs'].keys())}"
    )
    assert captured["kwargs"]["connection_timeout"] == expected_connect


# ----------------------- 事务超时参数（begin_transaction） ----------------------


class _FakeRecord:
    """假记录：只交出 .data()。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _FakeResult:
    """假 Result：可迭代出 _FakeRecord，并把「结果已被取用」记到所属事务上。"""

    def __init__(self, rows: list[dict[str, Any]], transaction: "_FakeTransaction") -> None:
        self._rows = rows
        self._transaction = transaction

    def __iter__(self) -> Iterator[_FakeRecord]:
        self._transaction.result_consumed = True
        return iter(_FakeRecord(row) for row in self._rows)

    def consume(self) -> None:
        """写路径用 .consume() 把结果取完，同样算取用。"""
        self._transaction.result_consumed = True


class _FakeTransaction:
    """假事务：进入返回自身，退出时记录异常参数与「退出前是否已取完结果」。"""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.run_calls: list[tuple[str, dict[str, Any]]] = []
        self.result_consumed = False
        self.exit_args: tuple[Any, ...] | None = None
        self.consumed_before_exit: bool | None = None

    def __enter__(self) -> "_FakeTransaction":
        return self

    def __exit__(self, *args: Any) -> bool:
        self.exit_args = args
        self.consumed_before_exit = self.result_consumed
        return False

    def run(self, query: str, parameters: dict[str, Any]) -> _FakeResult:
        self.run_calls.append((query, parameters))
        return _FakeResult(self._rows, self)


class _FakeSessionRecording:
    """假 session：进入不抛错，begin_transaction 记录关键字参数并返回假事务。"""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.begin_calls: list[dict[str, Any]] = []
        self.transaction: _FakeTransaction | None = None

    def __enter__(self) -> "_FakeSessionRecording":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def begin_transaction(self, *, metadata: Any = None, timeout: Any = None) -> _FakeTransaction:
        self.begin_calls.append({"metadata": metadata, "timeout": timeout})
        self.transaction = _FakeTransaction(self._rows)
        return self.transaction


class _FakeDriverRecording:
    """假 driver（照 _FakeDriver 的写法）：session() 返回不抛错的假 session。"""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = list(rows or [])
        self.session_calls: list[dict[str, Any]] = []
        self.session_obj: _FakeSessionRecording | None = None

    def session(self, **kwargs: Any) -> _FakeSessionRecording:
        self.session_calls.append(kwargs)
        self.session_obj = _FakeSessionRecording(self._rows)
        return self.session_obj

    def close(self) -> None:
        pass

    def verify_connectivity(self) -> None:
        pass


@pytest.mark.asyncio
async def test_execute_read_passes_read_timeout_to_begin_transaction() -> None:
    """读超时必须作为事务 timeout 进入 begin_transaction；用 session.run 的 kwargs 传只会变成 Cypher 参数。"""
    from app.config import get_settings

    rows = [{"name": "alice"}, {"name": "bob"}]
    driver = _FakeDriverRecording(rows)
    client = Neo4jClient()
    client._driver = driver

    records = await client.execute_read("MATCH (n) RETURN n", {"limit": 2})

    session = driver.session_obj
    transaction = session.transaction if session else None
    assert session is not None and transaction is not None
    assert session.begin_calls == [
        {"metadata": None, "timeout": get_settings().neo4j_read_timeout_seconds}
    ]
    assert transaction.run_calls == [("MATCH (n) RETURN n", {"limit": 2})]
    assert records == rows
    # 结果必须在事务退出前取完，否则显式事务里拿不到数据
    assert transaction.consumed_before_exit is True


@pytest.mark.asyncio
async def test_execute_write_commits_cleanly_with_read_timeout() -> None:
    """写路径必须干净退出事务（提交而非回滚），且同样带上读超时。"""
    from app.config import get_settings

    driver = _FakeDriverRecording()
    client = Neo4jClient()
    client._driver = driver

    await client.execute_write("CREATE (n)", {"name": "carol"})

    session = driver.session_obj
    transaction = session.transaction if session else None
    assert session is not None and transaction is not None
    assert session.begin_calls == [
        {"metadata": None, "timeout": get_settings().neo4j_read_timeout_seconds}
    ]
    # 异常参数为空 = 无异常退出 = 走提交
    assert transaction.exit_args == (None, None, None)
    assert transaction.result_consumed is True


# ----------------------------- neo4j_graph_search_tool 降级 --------------------


@pytest.mark.asyncio
async def test_tool_catches_degraded_and_marks_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """tool 必须捕获 Neo4jDegradedError 并把 source_status.neo4j 标 failed，不向上抛。"""
    from app.nodes import graph_tools

    class _FakeService:
        async def query_project_subgraph(self, paper_ids: list[str]) -> dict:
            raise Neo4jDegradedError(reason="SocketDeadlineExceededError")

    class _FakeAnswerService:
        async def build_bundle(self, **kwargs: Any) -> dict:
            return {"graph_evidence": []}

    monkeypatch.setattr(graph_tools, "get_research_object_query_service", lambda: _FakeService())
    monkeypatch.setattr(graph_tools, "get_explainable_answer_service", lambda: _FakeAnswerService())

    fake_runtime = MagicMock()
    fake_runtime.tool_call_id = "tc-1"

    structured_query = {"paper_ids": ["upload_a", "upload_b"]}
    # 直接调用底层协程函数，绕开 LangGraph runtime 注入逻辑
    cmd = await graph_tools.neo4j_graph_search_tool.coroutine(
        query="test",
        structured_query=structured_query,
        graph_evidence=[],
        source_status={},
        source_status_reasons={},
        retrieval_mode="",
        runtime=fake_runtime,
    )

    # langgraph tool 调用结果是 Command；其 update 字段必须包含降级标记
    assert cmd.update["source_status"]["neo4j"] == "failed"


@pytest.mark.asyncio
async def test_tool_writes_reason_to_source_status_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """tool 必须把异常类名写入 source_status_reasons.neo4j。"""
    from app.nodes import graph_tools

    class _FakeService:
        async def query_project_subgraph(self, paper_ids: list[str]) -> dict:
            raise Neo4jDegradedError(reason="SocketDeadlineExceededError")

    class _FakeAnswerService:
        async def build_bundle(self, **kwargs: Any) -> dict:
            return {"graph_evidence": []}

    monkeypatch.setattr(graph_tools, "get_research_object_query_service", lambda: _FakeService())
    monkeypatch.setattr(graph_tools, "get_explainable_answer_service", lambda: _FakeAnswerService())

    fake_runtime = MagicMock()
    fake_runtime.tool_call_id = "tc-2"

    structured_query = {"paper_ids": ["upload_a"]}
    cmd = await graph_tools.neo4j_graph_search_tool.coroutine(
        query="test",
        structured_query=structured_query,
        graph_evidence=[],
        source_status={},
        source_status_reasons={},
        retrieval_mode="",
        runtime=fake_runtime,
    )

    reasons = cmd.update.get("source_status_reasons") or {}
    assert reasons.get("neo4j") == "SocketDeadlineExceededError"


@pytest.mark.asyncio
async def test_tool_does_not_raise_on_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """tool 必须吞 Neo4jDegradedError 不向上抛（主图继续）。"""
    from app.nodes import graph_tools

    class _FakeService:
        async def query_project_subgraph(self, paper_ids: list[str]) -> dict:
            raise Neo4jDegradedError(reason="ServiceUnavailable")

    class _FakeAnswerService:
        async def build_bundle(self, **kwargs: Any) -> dict:
            return {"graph_evidence": []}

    monkeypatch.setattr(graph_tools, "get_research_object_query_service", lambda: _FakeService())
    monkeypatch.setattr(graph_tools, "get_explainable_answer_service", lambda: _FakeAnswerService())

    fake_runtime = MagicMock()
    fake_runtime.tool_call_id = "tc-3"

    # 不应抛
    structured_query = {"paper_ids": ["upload_a"]}
    cmd = await graph_tools.neo4j_graph_search_tool.coroutine(
        query="",
        structured_query=structured_query,
        graph_evidence=[],
        source_status={},
        source_status_reasons={},
        retrieval_mode="",
        runtime=fake_runtime,
    )

    assert cmd.update["source_status"]["neo4j"] == "failed"
    # 同时验证 source_status_reasons 也被写入
    assert (cmd.update.get("source_status_reasons") or {}).get("neo4j") == "ServiceUnavailable"
