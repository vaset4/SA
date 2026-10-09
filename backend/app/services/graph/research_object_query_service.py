"""
研究对象图谱查询服务。

这是图谱的“读取侧”服务，和 `research_object_graph_service` 的写入侧相对。
当前职责比较聚焦：
- 按 paper_id 读取该论文的最小子图
- 暴露图谱健康状态

它返回的是偏前端友好的轻量结构，不是原始 Neo4j record。
"""

from __future__ import annotations

from typing import Any

from app.services.graph.neo4j_client import Neo4jClient, get_neo4j_client
from app.services.graph.neo4j_schema_service import get_neo4j_schema_service

_query_service: "ResearchObjectQueryService | None" = None
_VISIBLE_RELATION_TYPES = ("USES_METHOD", "USES_DATASET", "REPORTS_METRIC", "STUDIES_TASK")


class ResearchObjectQueryService:
    """提供最小子图读取与健康检查。"""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or get_neo4j_client()

    async def get_paper_subgraph(self, paper_id: str) -> dict[str, object]:
        """
        读取单篇论文的最小子图。

        当前定义的“最小子图”是：
        Paper 节点，以及它直接连出的实体节点和关系边。
        """
        if not self._client.enabled:
            return self._empty_payload(paper_id)
        paper_rows = await self._client.execute_read(
            """
            MATCH (p:Paper {paper_id: $paper_id})
            RETURN p
            LIMIT 1
            """,
            {"paper_id": paper_id},
        )
        if not paper_rows:
            return self._empty_payload(paper_id)
        paper_node = paper_rows[0]["p"]
        rows = await self._client.execute_read(
            """
            MATCH (p:Paper {paper_id: $paper_id})-[r]->(n)
            WHERE type(r) IN $relation_types
            RETURN p, r, n
            """,
            {"paper_id": paper_id, "relation_types": list(_VISIBLE_RELATION_TYPES)},
        )
        if not rows:
            return self._empty_payload(
                paper_id,
                title=paper_node.get("title", ""),
                library_id=paper_node.get("library_id"),
            )
        nodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []
        paper_payload = {"paper_id": paper_id, "title": paper_node.get("title", "")}
        seen_nodes: set[str] = set()
        for row in rows:
            paper_node = row["p"]
            target_node = row["n"]
            relation = row["r"]
            paper_payload = {"paper_id": paper_node["paper_id"], "title": paper_node.get("title", "")}
            node_id = target_node.get("entity_key", "")
            if node_id and node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append(
                    {
                        "id": node_id,
                        "label": self._node_label(target_node),
                        "name": target_node.get("name", ""),
                    }
                )
            edges.append(
                {
                    "source": f"paper:{paper_id}",
                    "target": node_id,
                    "type": self._relation_type(relation),
                }
            )
        library_id = rows[0]["p"].get("library_id")
        return {
            "paper": paper_payload,
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "library_id": library_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        }

    async def health(self) -> dict[str, object]:
        """复用 schema service 的健康检查结果。"""
        return await get_neo4j_schema_service().health()

    async def query_project_subgraph(self, paper_ids: list[str]) -> dict[str, Any]:
        """
        聚合多篇论文的最小可见子图。

        这里优先服务项目内研究时的“范围内对象线索”，
        不追求完整大图，只返回去重后的可见节点和边。
        """
        normalized_paper_ids = [str(item).strip() for item in paper_ids if str(item).strip()]
        if not normalized_paper_ids:
            return {"papers": [], "nodes": [], "edges": [], "meta": {"paper_count": 0, "node_count": 0, "edge_count": 0}}
        papers: list[dict[str, object]] = []
        nodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []
        seen_nodes: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()
        library_ids: set[str] = set()
        for paper_id in normalized_paper_ids:
            payload = await self.get_paper_subgraph(paper_id)
            papers.append(payload["paper"])
            library_id = str((payload.get("meta") or {}).get("library_id") or "").strip()
            if library_id:
                library_ids.add(library_id)
            for node in payload.get("nodes", []):
                node_id = str(node.get("id") or "").strip()
                if not node_id or node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                nodes.append(node)
            for edge in payload.get("edges", []):
                edge_key = (
                    str(edge.get("source") or "").strip(),
                    str(edge.get("target") or "").strip(),
                    str(edge.get("type") or "").strip(),
                )
                if not all(edge_key) or edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append(edge)
        return {
            "papers": papers,
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "paper_count": len(papers),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "library_ids": sorted(library_ids),
            },
        }

    def _node_label(self, node: object) -> str:
        """
        从 Neo4j 返回的节点值中恢复首个标签。

        节点值有三种真实形状：驱动 Node 对象（带 `labels` 属性）、显式带 `_label` 的字典，
        以及 `record.data()` 摊平后的属性字典——后者没有 `_label`，标签只能从
        `entity_key` 的 `库:标签:名` 中段还原（写入侧见 `build_entity_key`）。

        Args:
            node: Neo4j 返回的节点值，可为驱动对象、字典或其他形状。

        Returns:
            恢复出的标签字符串；三种形状都取不到标签时返回 "Unknown"。
        """
        labels = getattr(node, "labels", None)
        if labels:
            return str(next(iter(labels)))
        if isinstance(node, dict):
            if "_label" in node:
                return str(node["_label"])
            key_parts = str(node.get("entity_key") or "").split(":", 2)
            if len(key_parts) == 3 and key_parts[1]:
                return key_parts[1]
        return "Unknown"

    def _relation_type(self, relation: object) -> str:
        """
        从 Neo4j 返回的关系值中恢复关系类型。

        关系值有三种真实形状：驱动 Relationship 对象（带 `type` 属性）、显式带 `type`
        的字典，以及 `record.data()` 摊平成 `[起点属性, 关系类型, 终点属性]` 的三元列表
        ——后者中段才是关系类型，且形状不符时不许硬转成字符串。

        Args:
            relation: Neo4j 返回的关系值，可为驱动对象、字典或三元列表。

        Returns:
            恢复出的关系类型字符串；三种形状都取不到时返回空串。
        """
        relation_type = getattr(relation, "type", None)
        if relation_type:
            return str(relation_type)
        if isinstance(relation, dict):
            return str(relation.get("type", ""))
        if (
            isinstance(relation, (list, tuple))
            and len(relation) == 3
            and isinstance(relation[1], str)
        ):
            return relation[1]
        return ""

    def _empty_payload(self, paper_id: str, *, title: str = "", library_id: str | None = None) -> dict[str, object]:
        return {
            "paper": {"paper_id": paper_id, "title": title},
            "nodes": [],
            "edges": [],
            "meta": {"library_id": library_id, "node_count": 0, "edge_count": 0},
        }


def get_research_object_query_service() -> ResearchObjectQueryService:
    """返回查询 service 单例。"""
    global _query_service
    if _query_service is None:
        _query_service = ResearchObjectQueryService()
    return _query_service
