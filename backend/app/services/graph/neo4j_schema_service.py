"""
Neo4j schema 初始化。

这个服务负责 Neo4j 的“基础设施准备”，而不是业务读写：
- 为 Paper 节点建立唯一约束
- 为各类研究对象节点建立 `entity_key` 唯一约束
- 暴露健康检查结果

因此它通常在应用启动预热或运维接口里被调用。
"""

from __future__ import annotations

from app.models.graph_models import GRAPH_NODE_LABELS
from app.services.graph.neo4j_client import Neo4jClient, get_neo4j_client

_schema_service: "Neo4jSchemaService | None" = None


class Neo4jSchemaService:
    """统一维护约束与索引。"""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or get_neo4j_client()

    async def initialize(self) -> None:
        """
        初始化 Neo4j 约束。

        采用 `IF NOT EXISTS`，所以可以安全地在启动阶段重复调用。
        """
        if not self._client.enabled:
            return
        await self._client.execute_write(
            "CREATE CONSTRAINT paper_paper_id IF NOT EXISTS FOR (n:Paper) REQUIRE n.paper_id IS UNIQUE"
        )
        for label in GRAPH_NODE_LABELS:
            if label == "Paper":
                continue
            await self._client.execute_write(
                f"CREATE CONSTRAINT {label.lower()}_entity_key IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.entity_key IS UNIQUE"
            )

    async def health(self) -> dict[str, object]:
        """返回 Neo4j 是否启用、连通以及 schema 是否可视为 ready。"""
        if not self._client.enabled:
            return {"enabled": False, "connected": False, "schema_ready": False}
        connected = await self._client.verify_connectivity()
        return {"enabled": True, "connected": connected, "schema_ready": connected}


def get_neo4j_schema_service() -> Neo4jSchemaService:
    """返回 schema service 单例。"""
    global _schema_service
    if _schema_service is None:
        _schema_service = Neo4jSchemaService()
    return _schema_service
