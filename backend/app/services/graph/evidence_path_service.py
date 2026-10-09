"""
Neo4j 路径级证据查询服务。

这个服务的目标不是做复杂图推理，而是找出“足够解释回答来源”的最小共享路径。
它当前采用的是一种轻量策略：
- 在同一批论文之间寻找共享中间节点
- 把共享节点转成前端可展示的 path/nodes/edges 结构

因此它更像“解释层证据抽取”，而不是完整的图算法引擎。
"""

from __future__ import annotations

from app.models.evidence_models import GraphEvidencePath
from app.services.graph.neo4j_client import get_neo4j_client

_service: "EvidencePathService | None" = None


class EvidencePathService:
    """把论文集合转换为最小可解释图路径。"""

    def __init__(self) -> None:
        self._client = get_neo4j_client()

    async def find_paths(
        self,
        paper_ids: list[str],
        *,
        path_type: str = "support",
        max_paths: int = 3,
    ) -> list[GraphEvidencePath]:
        """
        在给定论文集合之间查找可解释路径。

        当前实现要求至少两篇论文，且 Neo4j 已启用；否则直接返回空列表。
        """
        clean_ids = [paper_id.strip() for paper_id in paper_ids if paper_id.strip()]
        if len(clean_ids) < 2 or not self._client.enabled:
            return []
        try:
            rows = await self._client.execute_read(
                """
                MATCH (p1:Paper)-[r1]->(n)<-[r2]-(p2:Paper)
                WHERE p1.paper_id IN $paper_ids
                  AND p2.paper_id IN $paper_ids
                  AND p1.paper_id < p2.paper_id
                RETURN p1.paper_id AS source_paper_id,
                       p2.paper_id AS target_paper_id,
                       labels(n)[0] AS node_label,
                       coalesce(n.name, n.title, n.entity_key) AS node_name,
                       n.entity_key AS entity_key,
                       type(r1) AS left_relation,
                       type(r2) AS right_relation
                LIMIT 12
                """,
                {"paper_ids": clean_ids},
            )
        except Exception:
            return []

        # 把原始 Neo4j 行结果转换成前端更容易消费的 path 结构。
        paths: list[GraphEvidencePath] = []
        for index, row in enumerate(rows[:max_paths]):
            source_paper_id = row["source_paper_id"]
            target_paper_id = row["target_paper_id"]
            entity_key = row.get("entity_key") or f"entity:{index}"
            node_label = row.get("node_label") or "Entity"
            node_name = row.get("node_name") or entity_key
            paths.append(
                {
                    "path_id": f"{path_type}-{index}",
                    "path_type": path_type,
                    "summary": f"{source_paper_id} 与 {target_paper_id} 通过 {node_label}“{node_name}”相关",
                    "support_paper_ids": [source_paper_id, target_paper_id],
                    "confidence": 0.7,
                    "nodes": [
                        {
                            "id": f"paper:{source_paper_id}",
                            "label": source_paper_id,
                            "node_type": "paper",
                            "paper_id": source_paper_id,
                            "display_meta": {},
                        },
                        {
                            "id": entity_key,
                            "label": node_name,
                            "node_type": node_label.lower(),
                            "entity_key": entity_key,
                            "display_meta": {"label": node_label},
                        },
                        {
                            "id": f"paper:{target_paper_id}",
                            "label": target_paper_id,
                            "node_type": "paper",
                            "paper_id": target_paper_id,
                            "display_meta": {},
                        },
                    ],
                    "edges": [
                        {
                            "source": f"paper:{source_paper_id}",
                            "target": entity_key,
                            "relation": row.get("left_relation") or "RELATED_TO",
                            "paper_id": source_paper_id,
                            "weight": 1.0,
                            "evidence_role": "left",
                        },
                        {
                            "source": f"paper:{target_paper_id}",
                            "target": entity_key,
                            "relation": row.get("right_relation") or "RELATED_TO",
                            "paper_id": target_paper_id,
                            "weight": 1.0,
                            "evidence_role": "right",
                        },
                    ],
                }
            )
        return paths


def get_evidence_path_service() -> EvidencePathService:
    """返回 evidence path service 单例。"""
    global _service
    if _service is None:
        _service = EvidencePathService()
    return _service
