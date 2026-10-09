"""Neo4j 生命周期治理与统计服务。"""

from __future__ import annotations

import asyncio

import app.services.paper_db as paper_db
from app.services.graph.neo4j_client import Neo4jClient, get_neo4j_client
from app.services.graph.research_object_graph_service import get_research_object_graph_service

_service: "Neo4jMaintenanceService | None" = None
_ENTITY_LABELS = ("Author", "Venue", "Method", "Dataset", "Metric", "Claim", "Limitation", "Task")


class Neo4jMaintenanceService:
    """封装删除、重建、回填与统计能力。"""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or get_neo4j_client()

    async def list_paper_ids(self) -> list[str]:
        """
        列出 Neo4j 中全部 Paper 节点的论文主键。

        Returns:
            已去重并排序的非空 paper_id
        """
        if not self._client.enabled:
            return []
        rows = await self._client.execute_read(
            "MATCH (paper:Paper) RETURN DISTINCT paper.paper_id AS paper_id ORDER BY paper_id"
        )
        return [str(row.get("paper_id") or "").strip() for row in rows if str(row.get("paper_id") or "").strip()]

    async def delete_paper_subgraph(self, paper_id: str) -> dict[str, object]:
        if not self._client.enabled:
            return {"enabled": False, "deleted_paper_id": paper_id, "orphan_cleanup": 0}
        await self._client.execute_write(
            """
            MATCH (p:Paper {paper_id: $paper_id})
            OPTIONAL MATCH (p)-[r]-()
            DELETE r, p
            """,
            {"paper_id": paper_id},
        )
        orphan_cleanup = 0
        for label in _ENTITY_LABELS:
            rows = await self._client.execute_read(
                f"""
                MATCH (n:{label})
                WHERE NOT (n)--()
                WITH collect(n) AS nodes
                FOREACH (item IN nodes | DELETE item)
                RETURN size(nodes) AS deleted_count
                """
            )
            orphan_cleanup += int((rows[0] or {}).get("deleted_count", 0)) if rows else 0
        return {"enabled": True, "deleted_paper_id": paper_id, "orphan_cleanup": orphan_cleanup}

    async def rebuild_paper_subgraph(self, paper_id: str) -> dict[str, object]:
        paper = await asyncio.to_thread(paper_db.get_paper, paper_id)
        if not paper:
            raise LookupError(f"论文不存在: {paper_id}")
        await self.delete_paper_subgraph(paper_id)
        payload = await get_research_object_graph_service().upsert_paper_subgraph(paper, "")
        return {"paper_id": paper_id, "node_count": len(payload["nodes"]) + 1, "edge_count": len(payload["edges"])}

    async def backfill_library(self, library_id: str) -> dict[str, object]:
        papers = await asyncio.to_thread(
            paper_db.list_papers,
            limit=10000,
            offset=0,
            library_id=library_id,
        )
        rebuilt = 0
        for paper in papers:
            paper_id = paper.get("paper_id", "")
            if not paper_id:
                continue
            await self.rebuild_paper_subgraph(paper_id)
            rebuilt += 1
        return {"library_id": library_id, "paper_count": rebuilt}

    async def stats(self, library_id: str | None = None, paper_ids: list[str] | None = None) -> dict[str, object]:
        if not self._client.enabled:
            return {
                "enabled": False,
                "library_id": library_id,
                "paper_ids": paper_ids or [],
                "paper_count": 0,
                "node_types": {},
                "relation_types": {},
                "db_paper_count": 0,
                "paper_delta": 0,
            }
        params = {"library_id": library_id or "", "paper_ids": paper_ids or []}
        paper_where = "WHERE p.paper_id IN $paper_ids" if paper_ids else "WHERE p.library_id = $library_id" if library_id else ""
        paper_rows = await self._client.execute_read(
            f"MATCH (p:Paper) {paper_where} RETURN count(p) AS paper_count",
            params,
        )
        node_where = "WHERE n.paper_id IN $paper_ids OR n.library_id = $library_id" if (paper_ids or library_id) else ""
        node_rows = await self._client.execute_read(
            f"""
            MATCH (n) {node_where}
            UNWIND labels(n) AS label
            WITH label, count(n) AS count
            WHERE label <> 'Paper'
            RETURN label, count
            ORDER BY label
            """,
            params,
        )
        edge_where = "WHERE r.paper_id IN $paper_ids" if paper_ids else "WHERE r.library_id = $library_id" if library_id else ""
        edge_rows = await self._client.execute_read(
            f"""
            MATCH ()-[r]->() {edge_where}
            RETURN type(r) AS relation, count(r) AS count
            ORDER BY relation
            """,
            params,
        )
        db_paper_count = len(paper_ids or [])
        if not paper_ids:
            db_paper_count = await asyncio.to_thread(
                paper_db.count_papers,
                library_id=library_id,
            )
        paper_count = int((paper_rows[0] or {}).get("paper_count", 0)) if paper_rows else 0
        return {
            "enabled": True,
            "library_id": library_id,
            "paper_ids": paper_ids or [],
            "paper_count": paper_count,
            "node_types": {str(row["label"]): int(row["count"]) for row in node_rows},
            "relation_types": {str(row["relation"]): int(row["count"]) for row in edge_rows},
            "db_paper_count": int(db_paper_count),
            "paper_delta": paper_count - int(db_paper_count),
        }


def get_neo4j_maintenance_service() -> Neo4jMaintenanceService:
    global _service
    if _service is None:
        _service = Neo4jMaintenanceService()
    return _service
