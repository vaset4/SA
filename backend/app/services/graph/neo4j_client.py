"""
Neo4j 客户端封装。

这个文件给上层 service 提供一个“看起来是 async 的 Neo4j 访问接口”，
但底层实际上仍然是同步驱动，通过 `asyncio.to_thread` 包装。

这样做的原因通常是：
- 当前项目只需要少量读写，不必立刻引入 async 原生驱动复杂度
- 但上层 service 又希望保持统一的 async 调用风格

Phase 1（05-23-cross-paper-qa-retrieval-fix R2）— 优雅降级:
- 顶层维护 `_NEO4J_FAILURE_STATUS_MAP` 白名单
- `execute_read` / `execute_write` 捕获到白名单内异常时抛 `Neo4jDegradedError`
- 白名单外异常原样向上抛，避免静默吞错
- 上层工具（neo4j_graph_search_tool）负责把 Neo4jDegradedError 翻译成
  `source_status.neo4j = "failed"`
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from app.config import get_settings

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired
    # SocketDeadlineExceededError 实际定义在 neo4j._exceptions（私有模块，但属于稳定 API）
    from neo4j._exceptions import SocketDeadlineExceededError  # type: ignore[import]
except ImportError:  # pragma: no cover - 依赖未安装时的运行时保护
    GraphDatabase = None  # type: ignore[assignment]
    AuthError = ServiceUnavailable = SessionExpired = SocketDeadlineExceededError = Exception  # type: ignore[misc,assignment]


_client: "Neo4jClient | None" = None


# 已知失败异常类 → source_status 状态映射。
# 白名单显式可控；新增异常类必须在这里登记，禁止在 except 块里写死字符串名。
_NEO4J_FAILURE_STATUS_MAP: Mapping[type[BaseException], str] = {
    SocketDeadlineExceededError: "failed",  # 已知超时
    ServiceUnavailable: "failed",            # 连接拒绝 / 服务不可用
    SessionExpired: "failed",                # 会话失效
    AuthError: "failed",                     # 认证失败（按失败处理，不强制宕机）
}


class Neo4jDegradedError(Exception):
    """Neo4j 调用因白名单内已知失败被优雅降级时抛出。

    Attributes:
        reason: 触发降级的原异常类名（取自 exc.__class__.__name__），
                供上层日志和 source_status_reasons 写入。
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Neo4j degraded: {reason}")
        self.reason = reason


def _classify_exception(exc: BaseException) -> str | None:
    """按 `_NEO4J_FAILURE_STATUS_MAP` 显式白名单分类异常。

    Returns:
        命中白名单时返回对应状态字符串；未命中返回 None。
        调用方据此决定是抛 `Neo4jDegradedError` 还是原样向上抛。
    """
    for exc_type, status in _NEO4J_FAILURE_STATUS_MAP.items():
        if isinstance(exc, exc_type):
            return status
    return None


class Neo4jClient:
    """基于同步驱动的轻量异步封装。"""

    def __init__(self) -> None:
        self._driver = None
        self._settings = get_settings()

    @property
    def enabled(self) -> bool:
        """读取 Neo4j 功能总开关。"""
        return bool(self._settings.neo4j_enabled)

    def _ensure_driver(self):
        """
        惰性初始化底层 driver。

        只有真正发起 Neo4j 访问时才会创建连接，避免应用启动期强依赖 Neo4j。
        超时参数从 settings 读取，禁止裸数字。
        """
        if not self.enabled:
            raise RuntimeError("Neo4j is disabled")
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver is not installed")
        if self._driver is None:
            auth = None
            if self._settings.neo4j_username and self._settings.neo4j_password:
                auth = (self._settings.neo4j_username, self._settings.neo4j_password)
            # neo4j-driver 的 `connection_timeout` 参数控制 socket 建链阶段超时。
            # 读取超时按事务配置传递：`session.run()` 的 **kwargs 是 Cypher 参数不是事务配置，
            # 传进去只会变成查询变量，必须走 begin_transaction(timeout=...)。
            self._driver = GraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=auth,
                connection_timeout=self._settings.neo4j_connect_timeout_seconds,
            )
        return self._driver

    async def verify_connectivity(self) -> bool:
        """验证 Neo4j 连通性；通常用于启动预热或健康检查。"""
        if not self.enabled:
            return False
        driver = self._ensure_driver()
        await asyncio.to_thread(driver.verify_connectivity)
        return True

    async def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        """执行写查询，并通过线程池桥接同步驱动。

        已知失败（_NEO4J_FAILURE_STATUS_MAP 白名单）会被翻译成 `Neo4jDegradedError`；
        未知异常照样向上抛，避免静默吞错。
        """
        driver = self._ensure_driver()
        read_timeout = self._settings.neo4j_read_timeout_seconds

        def _run() -> None:
            with driver.session(database=self._settings.neo4j_database) as session:
                with session.begin_transaction(timeout=read_timeout) as transaction:
                    # 显式事务里结果不取完不算执行完，consume 同时把查询期异常在事务内抛出来
                    transaction.run(query, parameters or {}).consume()

        try:
            await asyncio.to_thread(_run)
        except BaseException as exc:  # noqa: BLE001 - 故意捕获后分类再抛
            if _classify_exception(exc) is not None:
                raise Neo4jDegradedError(reason=exc.__class__.__name__) from exc
            raise

    async def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """执行读查询，并把结果统一转成 `list[dict]`。

        已知失败（_NEO4J_FAILURE_STATUS_MAP 白名单）会被翻译成 `Neo4jDegradedError`；
        未知异常照样向上抛，避免静默吞错。
        """
        driver = self._ensure_driver()
        read_timeout = self._settings.neo4j_read_timeout_seconds

        def _run() -> list[dict[str, Any]]:
            with driver.session(database=self._settings.neo4j_database) as session:
                with session.begin_transaction(timeout=read_timeout) as transaction:
                    # 结果必须在事务关闭前取完，事务正常退出即提交
                    result = transaction.run(query, parameters or {})
                    return [record.data() for record in result]

        try:
            return await asyncio.to_thread(_run)
        except BaseException as exc:  # noqa: BLE001 - 故意捕获后分类再抛
            if _classify_exception(exc) is not None:
                raise Neo4jDegradedError(reason=exc.__class__.__name__) from exc
            raise

    async def close(self) -> None:
        """关闭底层 driver，通常在应用生命周期结束时调用。"""
        if self._driver is None:
            return
        driver = self._driver
        self._driver = None
        await asyncio.to_thread(driver.close)


def get_neo4j_client() -> Neo4jClient:
    """返回 Neo4j client 单例。"""
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
