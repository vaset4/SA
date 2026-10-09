"""研究演化地图查询服务。"""

from __future__ import annotations

from typing import Any

from app.models.evolution_models import (
    ALLOWED_EVOLUTION_RELATIONS,
    EvolutionEdge,
    EvolutionFocus,
    EvolutionGraphPayload,
    EvolutionNode,
    EvolutionNodeType,
    EvolutionQuery,
)
from app.services.graph.evolution_export_service import get_evolution_export_service
from app.services.graph.neo4j_client import Neo4jClient, get_neo4j_client

_service: "EvolutionGraphService | None" = None


class EvolutionGraphService:
    """构建 paper / method / task 研究演化视图。"""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or get_neo4j_client()

    async def get_evolution(self, query: EvolutionQuery) -> EvolutionGraphPayload:
        if not self._client.enabled:
            return self._empty_payload(query)

        focus = self._build_focus(query)
        paper_nodes: dict[str, EvolutionNode] = {}
        object_nodes: dict[str, EvolutionNode] = {}
        edge_map: dict[tuple[str, str, str], EvolutionEdge] = {}

        if query["view_mode"] == "paper" and query["paper_id"]:
            paper_rows = await self._fetch_paper_neighbors(query["paper_id"], query["depth"])
            self._ingest_rows(paper_rows, paper_nodes, edge_map)
            related_paper_ids = list(paper_nodes)
            if query["paper_id"] and query["paper_id"] not in related_paper_ids:
                related_paper_ids.append(query["paper_id"])
        else:
            anchor_rows = await self._fetch_view_anchor_rows(query)
            self._ingest_rows(anchor_rows, paper_nodes, edge_map)
            related_paper_ids = list(paper_nodes)

        related_paper_ids = self._apply_paper_scope(related_paper_ids, query)
        object_rows = await self._fetch_object_links(related_paper_ids) if related_paper_ids else []
        self._ingest_rows(object_rows, paper_nodes, edge_map, object_nodes=object_nodes)

        nodes = list(paper_nodes.values()) + list(object_nodes.values())
        nodes = self._filter_nodes(nodes, query)
        allowed_ids = {node["id"] for node in nodes}
        edges = [
            edge
            for edge in edge_map.values()
            if edge["source"] in allowed_ids and edge["target"] in allowed_ids
        ]
        filters = self._build_filters(query, related_paper_ids)
        return get_evolution_export_service().build_payload(
            focus=focus,
            nodes=sorted(nodes, key=lambda item: (item["node_type"], item["year"] or 0, item["label"].lower(), item["id"])),
            edges=sorted(edges, key=lambda item: (item["relation"], item["year"] or 0, item["source"], item["target"])),
            filters=filters,
        )

    async def get_evolution_export(self, query: EvolutionQuery) -> EvolutionGraphPayload:
        return await self.get_evolution(query)

    async def get_evolution_entry(
        self,
        paper_id: str,
        *,
        project_id: str = "",
        resolved_scope: dict[str, object] | None = None,
    ) -> dict[str, object]:
        filters = {
            "paper_id": paper_id,
            "project_id": project_id,
            "from_year": (resolved_scope or {}).get("year_from"),
            "to_year": (resolved_scope or {}).get("year_to"),
            "task_filters": (resolved_scope or {}).get("task_filters", []),
            "method_filters": (resolved_scope or {}).get("method_filters", []),
        }
        target_url = "/graph"
        if project_id:
            target_url = f"/projects/{project_id}/graph"
        return get_evolution_export_service().build_entry(
            focus={"key": paper_id, "label": paper_id, "node_type": "paper", "view_mode": "paper"},
            filters=filters,
            target_url=target_url,
        )

    async def _fetch_paper_neighbors(self, paper_id: str, depth: int) -> list[dict[str, Any]]:
        normalized_depth = max(1, min(int(depth or 1), 3))
        query = f"""
        MATCH path=(source:Paper {{paper_id: $paper_id}})-[:CITES|EXTENDS*0..{normalized_depth}]-(target:Paper)
        UNWIND relationships(path) AS rel
        WITH DISTINCT rel
        RETURN startNode(rel) AS source, rel, endNode(rel) AS target
        """
        return await self._client.execute_read(query, {"paper_id": paper_id})

    async def _fetch_view_anchor_rows(self, query: EvolutionQuery) -> list[dict[str, Any]]:
        if query["view_mode"] == "method":
            cypher = """
            MATCH (source:Paper)-[rel:USES_METHOD]->(target:Method)
            WHERE toLower(target.name) = toLower($value)
            RETURN source, rel, target
            """
            value = query["method"]
        else:
            cypher = """
            MATCH (source:Paper)-[rel:STUDIES_TASK]->(target:Task)
            WHERE toLower(target.name) = toLower($value)
            RETURN source, rel, target
            """
            value = query["task"]
        return await self._client.execute_read(cypher, {"value": value})

    async def _fetch_object_links(self, paper_ids: list[str]) -> list[dict[str, Any]]:
        return await self._client.execute_read(
            """
            MATCH (source:Paper)-[rel:USES_METHOD|STUDIES_TASK|USES_DATASET]->(target)
            WHERE source.paper_id IN $paper_ids
            RETURN source, rel, target
            """,
            {"paper_ids": paper_ids},
        )

    def _apply_paper_scope(self, paper_ids: list[str], query: EvolutionQuery) -> list[str]:
        scope = query["resolved_scope"] or {}
        allowed = set(scope.get("paper_ids", []))
        filtered = list(dict.fromkeys(paper_ids))
        if allowed:
            filtered = [paper_id for paper_id in filtered if paper_id in allowed]
        return filtered

    def _ingest_rows(
        self,
        rows: list[dict[str, Any]],
        paper_nodes: dict[str, EvolutionNode],
        edge_map: dict[tuple[str, str, str], EvolutionEdge],
        *,
        object_nodes: dict[str, EvolutionNode] | None = None,
    ) -> None:
        extra_bucket = object_nodes if object_nodes is not None else {}
        for row in rows:
            relation = row.get("rel") or row.get("r")
            if relation is None:
                continue
            relation_type = getattr(relation, "type", None) or row.get("relation")
            if relation_type not in ALLOWED_EVOLUTION_RELATIONS:
                continue
            source_node = row.get("source") or row.get("p") or row.get("focus")
            target_node = row.get("target") or row.get("n")
            if source_node is None or target_node is None:
                continue
            source_payload = self._to_node(source_node)
            target_payload = self._to_node(target_node)
            self._store_node(source_payload, paper_nodes, extra_bucket)
            self._store_node(target_payload, paper_nodes, extra_bucket)
            edge_key = (source_payload["id"], target_payload["id"], relation_type)
            edge_map[edge_key] = {
                "source": source_payload["id"],
                "target": target_payload["id"],
                "relation": relation_type,
                "year": source_payload["year"] or target_payload["year"],
                "weight": 2.0 if relation_type in {"EXTENDS", "USES_METHOD", "STUDIES_TASK"} else 1.0,
            }

    def _store_node(
        self,
        node: EvolutionNode,
        paper_nodes: dict[str, EvolutionNode],
        object_nodes: dict[str, EvolutionNode],
    ) -> None:
        if node["node_type"] == "paper":
            paper_nodes[node["paper_id"]] = node
        else:
            object_nodes[node["id"]] = node

    def _filter_nodes(self, nodes: list[EvolutionNode], query: EvolutionQuery) -> list[EvolutionNode]:
        filtered: list[EvolutionNode] = []
        dataset = (query["dataset"] or "").strip().lower()
        for node in nodes:
            if node["node_type"] == "paper":
                if query["from_year"] is not None and node["year"] is not None and node["year"] < query["from_year"]:
                    continue
                if query["to_year"] is not None and node["year"] is not None and node["year"] > query["to_year"]:
                    continue
            if dataset and node["node_type"] == "dataset" and dataset not in node["label"].lower():
                continue
            if query["view_mode"] == "method" and query["method"]:
                if node["node_type"] == "method" and query["method"].strip().lower() not in node["label"].lower():
                    continue
            if query["view_mode"] == "task" and query["task"]:
                if node["node_type"] == "task" and query["task"].strip().lower() not in node["label"].lower():
                    continue
            filtered.append(node)
        return filtered

    def _build_filters(self, query: EvolutionQuery, paper_ids: list[str]) -> dict[str, object]:
        scope = query["resolved_scope"] or {}
        return {
            "view_mode": query["view_mode"],
            "paper_id": query["paper_id"],
            "method": query["method"],
            "task": query["task"],
            "dataset": query["dataset"],
            "from_year": query["from_year"],
            "to_year": query["to_year"],
            "depth": query["depth"],
            "project_id": query["project_id"],
            "resolved_scope": {
                "project_id": scope.get("project_id", ""),
                "library_ids": scope.get("library_ids", []),
                "paper_ids": paper_ids,
                "year_from": scope.get("year_from"),
                "year_to": scope.get("year_to"),
                "task_filters": scope.get("task_filters", []),
                "method_filters": scope.get("method_filters", []),
            },
        }

    def _build_focus(self, query: EvolutionQuery) -> EvolutionFocus:
        if query["view_mode"] == "method":
            return {
                "key": query["method"],
                "label": query["method"],
                "node_type": "method",
                "view_mode": "method",
            }
        if query["view_mode"] == "task":
            return {
                "key": query["task"],
                "label": query["task"],
                "node_type": "task",
                "view_mode": "task",
            }
        return {
            "key": query["paper_id"],
            "label": query["paper_id"],
            "node_type": "paper",
            "view_mode": "paper",
        }

    def _empty_payload(self, query: EvolutionQuery) -> EvolutionGraphPayload:
        return get_evolution_export_service().build_payload(
            focus=self._build_focus(query),
            nodes=[],
            edges=[],
            filters=self._build_filters(query, []),
        )

    def _to_node(self, node: Any) -> EvolutionNode:
        node_type = self._node_type(node)
        if node_type == "paper":
            paper_id = str(self._prop(node, "paper_id") or self._prop(node, "entity_key") or "")
            title = str(self._prop(node, "title") or paper_id)
            year = self._as_year(self._prop(node, "year"))
            return {
                "id": f"paper:{paper_id}",
                "label": title,
                "node_type": "paper",
                "year": year,
                "paper_id": paper_id,
                "importance_score": 1.0,
                "group": str(year or "unknown"),
            }
        name = str(self._prop(node, "name") or self._prop(node, "title") or self._label_of(node))
        entity_key = str(self._prop(node, "entity_key") or name.lower().replace(" ", "_"))
        return {
            "id": f"{node_type}:{entity_key}",
            "label": name,
            "node_type": node_type,
            "year": None,
            "paper_id": "",
            "importance_score": 0.7,
            "group": node_type,
        }

    def _label_of(self, node: Any) -> str:
        labels = getattr(node, "labels", None)
        if labels:
            return next(iter(labels))
        if isinstance(node, dict):
            return str(node.get("_label", "Unknown"))
        return "Unknown"

    def _node_type(self, node: Any) -> EvolutionNodeType:
        label = self._label_of(node).lower()
        if label == "paper":
            return "paper"
        if label == "method":
            return "method"
        if label == "task":
            return "task"
        return "dataset"

    def _prop(self, node: Any, key: str) -> Any:
        if isinstance(node, dict):
            return node.get(key)
        getter = getattr(node, "get", None)
        if callable(getter):
            return getter(key, None)
        try:
            return node[key]
        except Exception:
            return None

    def _as_year(self, value: Any) -> int | None:
        try:
            return int(value) if value is not None and str(value).strip() else None
        except (TypeError, ValueError):
            return None


def get_evolution_graph_service() -> EvolutionGraphService:
    """返回演化图 service 单例。"""
    global _service
    if _service is None:
        _service = EvolutionGraphService()
    return _service
