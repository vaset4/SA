"""
论文关系图查询服务。

这个服务把 Neo4j 中“论文通过共享研究对象连接”的关系，
压缩成适合前端可视化的 paper-to-paper 图。

它和 `evidence_path_service` 的区别是：
- `evidence_path_service` 关注回答解释
- 本服务关注图谱浏览和项目/库范围内的关系展示
"""

from __future__ import annotations

import re
from typing import Any

from app.services.graph.neo4j_client import Neo4jClient, get_neo4j_client

_service: "PaperGraphService | None" = None
_PAPER_GRAPH_RELATIONS = ("USES_METHOD", "USES_DATASET")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


class PaperGraphService:
    """把 Neo4j 研究对象关系聚合成论文到论文的可视化边。"""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or get_neo4j_client()

    async def get_paper_graph(
        self,
        *,
        project_id: str = "",
        library_id: str = "",
        resolved_scope: dict[str, Any] | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """
        读取指定范围内的论文关系图。

        优先使用 resolved_scope.paper_ids 收窄范围；若未提供，则按 library 或全局范围查询。
        """
        scoped_paper_ids = self._normalize_paper_ids((resolved_scope or {}).get("paper_ids", []))
        paper_limit = self._resolve_paper_limit(limit, scoped_paper_ids)
        if not self._client.enabled:
            return self._empty_payload(
                project_id=project_id,
                library_id=library_id,
                scoped_paper_count=len(scoped_paper_ids),
            )

        relation_types = await self._load_existing_relation_types()
        paper_keys = await self._load_paper_property_keys()
        supported_relations = [relation for relation in _PAPER_GRAPH_RELATIONS if relation in relation_types]
        query_params = {
            "paper_ids": scoped_paper_ids,
            "library_id": library_id,
            "paper_limit": paper_limit,
            "limit": max(1, limit),
        }
        edge_rows: list[dict[str, Any]] = []
        # 先查边，再按边里出现的论文构造节点；若没有边，再兜底查孤立节点。
        if supported_relations:
            edge_rows = await self._client.execute_read(
                self._build_edge_query(supported_relations, self._paper_order_by_clause(paper_keys)),
                query_params,
            )

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        relation_counts: dict[str, int] = {}
        for row in edge_rows:
            source = row.get("source")
            target = row.get("target")
            if not source or not target:
                continue
            source_payload = self._paper_node_payload(source)
            target_payload = self._paper_node_payload(target)
            nodes[source_payload["id"]] = source_payload
            nodes[target_payload["id"]] = target_payload
            relation = str(row.get("relation") or "")
            shared_items = self._normalize_shared_items(row.get("shared_items"))
            edges.append(
                {
                    "source": source_payload["id"],
                    "target": target_payload["id"],
                    "type": relation,
                    "weight": int(row.get("weight") or len(shared_items) or 1),
                    "shared_items": shared_items,
                }
            )
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        if not nodes:
            node_rows = await self._client.execute_read(
                self._build_node_query(self._paper_order_by_clause(paper_keys)),
                {
                    "paper_ids": scoped_paper_ids,
                    "library_id": library_id,
                    "paper_limit": paper_limit,
                },
            )
            for row in node_rows:
                paper = row.get("p")
                if not paper:
                    continue
                payload = self._paper_node_payload(paper)
                nodes[payload["id"]] = payload

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "meta": {
                "project_id": project_id,
                "library_id": library_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "scoped_paper_count": len(scoped_paper_ids) if scoped_paper_ids else len(nodes),
                "relation_counts": relation_counts,
            },
        }

    async def query_project_subgraph(
        self,
        *,
        project_id: str = "",
        library_id: str = "",
        paper_ids: list[str] | None = None,
        resolved_scope: dict[str, Any] | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """
        以项目范围语义读取论文关系子图。

        `Phase 3` 里更强调“项目内研究”的产品语义，所以补一个更贴近业务名字的入口，
        底层仍复用现有 paper-to-paper 图实现。
        """
        scope = dict(resolved_scope or {})
        if paper_ids:
            scope["paper_ids"] = paper_ids
        return await self.get_paper_graph(
            project_id=project_id,
            library_id=library_id,
            resolved_scope=scope or None,
            limit=limit,
        )

    def _resolve_paper_limit(self, edge_limit: int, scoped_paper_ids: list[str]) -> int:
        """根据是否已有显式 paper 范围，推导节点查询上限。"""
        if scoped_paper_ids:
            return len(scoped_paper_ids)
        return min(max(edge_limit, 50), 300)

    def _normalize_paper_ids(self, paper_ids: list[Any]) -> list[str]:
        """去重并规范化 paper_id 列表。"""
        seen: set[str] = set()
        normalized: list[str] = []
        for item in paper_ids:
            paper_id = str(item or "").strip()
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            normalized.append(paper_id)
        return normalized

    def _normalize_shared_items(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        items: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            items.append(text)
        return items

    async def _load_existing_relation_types(self) -> set[str]:
        """动态探测当前图谱里真实存在的关系类型。"""
        rows = await self._client.execute_read(
            """
            MATCH ()-[r]->()
            RETURN collect(DISTINCT type(r)) AS relation_types
            """,
            {},
        )
        return self._read_string_set(rows, "relation_types")

    async def _load_paper_property_keys(self) -> set[str]:
        """动态探测 Paper 节点有哪些属性，用于生成更稳健的排序语句。"""
        rows = await self._client.execute_read(
            """
            MATCH (p:Paper)
            UNWIND keys(p) AS key
            RETURN collect(DISTINCT key) AS paper_keys
            """,
            {},
        )
        return self._read_string_set(rows, "paper_keys")

    def _read_string_set(self, rows: list[dict[str, Any]], key: str) -> set[str]:
        if not rows:
            return set()
        values = rows[0].get(key, [])
        if not isinstance(values, list):
            return set()
        return {str(value).strip() for value in values if str(value or "").strip()}

    def _paper_order_by_clause(self, paper_keys: set[str]) -> str:
        if "year" in paper_keys:
            return "coalesce(p.year, 0) DESC, p.paper_id"
        if "published_date" in paper_keys:
            return "coalesce(p.published_date, '') DESC, p.paper_id"
        return "p.paper_id"

    def _build_edge_query(self, relation_types: list[str], order_by_clause: str) -> str:
        """构造论文共享研究对象边的查询语句。"""
        relation_pattern = "|".join(relation_types)
        return f"""
            MATCH (p:Paper)
            WHERE ($paper_ids = [] OR p.paper_id IN $paper_ids)
              AND ($library_id = '' OR p.library_id = $library_id)
            WITH p
            ORDER BY {order_by_clause}
            LIMIT $paper_limit
            WITH collect(p) AS scoped_papers
            UNWIND scoped_papers AS source
            MATCH (source)-[rel:{relation_pattern}]->(shared)<-[:{relation_pattern}]-(target:Paper)
            WHERE source.paper_id < target.paper_id
              AND target IN scoped_papers
            WITH source, target, type(rel) AS relation,
                 collect(DISTINCT coalesce(shared.name, shared.title, shared.entity_key, 'unknown')) AS shared_items
            RETURN source, target, relation, shared_items, size(shared_items) AS weight
            ORDER BY weight DESC, source.paper_id, target.paper_id, relation
            LIMIT $limit
            """

    def _build_node_query(self, order_by_clause: str) -> str:
        return f"""
                MATCH (p:Paper)
                WHERE ($paper_ids = [] OR p.paper_id IN $paper_ids)
                  AND ($library_id = '' OR p.library_id = $library_id)
                RETURN p
                ORDER BY {order_by_clause}
                LIMIT $paper_limit
                """

    def _paper_node_payload(self, node: Any) -> dict[str, Any]:
        """把 Neo4j 里的 Paper 节点转成前端图节点 payload。"""
        year = self._extract_value(node, "year")
        published_date = self._extract_value(node, "published_date")
        if year in (None, ""):
            year = self._extract_year_from_date(published_date)
        return {
            "id": f"paper:{self._extract_value(node, 'paper_id')}",
            "paper_id": self._extract_value(node, "paper_id"),
            "title": self._extract_value(node, "title"),
            "year": int(year) if year not in (None, "") else None,
            "published_date": str(published_date).strip() or None,
            "library_id": self._extract_value(node, "library_id") or None,
        }

    def _extract_value(self, record: Any, key: str, default: Any = "") -> Any:
        if record is None:
            return default
        getter = getattr(record, "get", None)
        if callable(getter):
            value = getter(key, default)
            return default if value is None else value
        if isinstance(record, dict):
            value = record.get(key, default)
            return default if value is None else value
        return default

    def _extract_year_from_date(self, published_date: Any) -> int | None:
        match = _YEAR_PATTERN.search(str(published_date or ""))
        if not match:
            return None
        return int(match.group(0))

    def _empty_payload(self, *, project_id: str, library_id: str, scoped_paper_count: int) -> dict[str, Any]:
        """在 Neo4j 未启用或无结果时，返回稳定空图结构。"""
        return {
            "nodes": [],
            "edges": [],
            "meta": {
                "project_id": project_id,
                "library_id": library_id,
                "node_count": 0,
                "edge_count": 0,
                "scoped_paper_count": scoped_paper_count,
                "relation_counts": {},
            },
        }


def get_paper_graph_service() -> PaperGraphService:
    """返回 paper graph service 单例。"""
    global _service
    if _service is None:
        _service = PaperGraphService()
    return _service
