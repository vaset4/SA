"""
研究对象图谱写入服务。

这个服务负责把一篇论文及其抽取结果，转换成 Neo4j 里可幂等写入的子图。
核心步骤是：
1. 从 paper/full_text 中抽取研究对象
2. 构造 Paper 节点
3. 构造 Method / Dataset / Metric / Task 等实体节点
4. 构造 Paper -> Entity 的关系边
5. 按幂等方式写入 Neo4j

如果你想理解“图谱里的节点和边从哪里来”，这个文件是最直接的入口。
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import NamedTuple

from app.config import get_settings
from app.models.graph_models import GraphEdgePayload, GraphNodePayload, ResearchGraphPayload
from app.models.models import Paper
from app.services.graph.neo4j_client import Neo4jClient, get_neo4j_client
from app.services.graph.research_object_extractor import extract_research_objects
from app.services.graph.research_object_llm_extractor import LlmExtractionAliasMap, extract_with_llm
from app.services.graph.research_object_normalizer import build_entity_key, dedupe_preserve_order, normalize_entity_name

_service: "ResearchObjectGraphService | None" = None
settings = get_settings()


def _utc_now_iso() -> str:
    """生成图谱边属性里使用的 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def _extract_year(published_date: str | None) -> int | None:
    """从发布日期文本里粗略提取年份。"""
    match = re.search(r"\b(19|20)\d{2}\b", str(published_date or ""))
    if not match:
        return None
    return int(match.group(0))


class EntityResolution(NamedTuple):
    requested_entity_key: str
    resolved_entity_key: str
    label: str
    aliases: list[str]


class ResearchObjectGraphService:
    """把论文与抽取结果转换成可幂等写入的子图。"""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or get_neo4j_client()

    async def build_payload(self, paper: Paper, full_text: str = "") -> ResearchGraphPayload:
        """
        构造一篇论文对应的图谱写入 payload。

        这里先在内存里把 paper node / entity nodes / relation edges 全部拼好，
        后续 `upsert_paper_subgraph()` 再决定是否真正写入 Neo4j。
        """
        library_id = paper.get("library_id", "default")
        extraction, alias_map = await self._extract_objects(paper, full_text)
        paper_node: GraphNodePayload = {
            "id": f"paper:{paper['paper_id']}",
            "label": "Paper",
            "name": paper["title"],
            "library_id": library_id,
            "properties": {
                "paper_id": paper["paper_id"],
                "library_id": library_id,
                "title": paper["title"],
                "source": paper.get("source", "upload"),
                "doi": paper.get("doi"),
                "arxiv_id": paper.get("arxiv_id"),
                "published_date": paper.get("published_date"),
                "year": _extract_year(paper.get("published_date")),
                "methodology_type": (paper.get("analysis") or {}).get("methodology_type", ""),
                "tldr": (paper.get("analysis") or {}).get("tldr", ""),
            },
        }
        nodes: list[GraphNodePayload] = []
        edges: list[GraphEdgePayload] = []

        def add_entity(label: str, values: list[str], relation_type: str) -> None:
            # 每种研究对象都规范化成独立实体节点，并与 Paper 建一条语义关系边。
            for value in values:
                entity_key = build_entity_key(library_id, label, value)
                aliases = _aliases_for(value, alias_map)
                node: GraphNodePayload = {
                    "id": entity_key,
                    "label": label,
                    "name": value,
                    "library_id": library_id,
                    "properties": {
                        "entity_key": entity_key,
                        "library_id": library_id,
                        "name": value.strip(),
                        "normalized_name": entity_key.split(":", 2)[-1],
                        "aliases": aliases,
                    },
                }
                edge: GraphEdgePayload = {
                    "source": paper_node["id"],
                    "target": entity_key,
                    "type": relation_type,
                    "paper_id": paper["paper_id"],
                    "library_id": library_id,
                    "source_type": "analysis",
                    "properties": {
                        "paper_id": paper["paper_id"],
                        "library_id": library_id,
                        "source": "analysis",
                        "created_at": _utc_now_iso(),
                    },
                }
                nodes.append(node)
                edges.append(edge)

        add_entity("Author", extraction["authors"], "AUTHORED_BY")
        if extraction["venue"]:
            add_entity("Venue", [extraction["venue"]], "PUBLISHED_IN")
        add_entity("Method", extraction["methods"], "USES_METHOD")
        add_entity("Dataset", extraction["datasets"], "USES_DATASET")
        add_entity("Metric", extraction["metrics"], "REPORTS_METRIC")
        add_entity("Claim", extraction["claims"], "MAKES_CLAIM")
        add_entity("Limitation", extraction["limitations"], "HAS_LIMITATION")
        add_entity("Task", extraction["tasks"], "STUDIES_TASK")
        return {"paper": paper_node, "nodes": nodes, "edges": edges}

    async def _extract_objects(
        self,
        paper: Paper,
        full_text: str,
    ) -> tuple[dict, LlmExtractionAliasMap]:
        if settings.graph_extraction_mode == "llm":
            return await extract_with_llm(paper, full_text)
        extraction = extract_research_objects(paper, full_text)
        return extraction, {}

    async def upsert_paper_subgraph(self, paper: Paper, full_text: str = "") -> ResearchGraphPayload:
        """
        把单篇论文子图幂等写入 Neo4j。

        写入顺序是：
        Paper -> Entity Nodes -> Paper-to-Entity Edges
        这样可以确保边建立时目标节点已经存在。
        """
        payload = await self.build_payload(paper, full_text)
        if not self._client.enabled:
            return payload
        await self._client.execute_write(
            """
            MERGE (p:Paper {paper_id: $paper_id})
            SET p += $properties
            """,
            {
                "paper_id": paper["paper_id"],
                "properties": payload["paper"]["properties"],
            },
        )
        resolutions: dict[str, EntityResolution] = {}
        for node in payload["nodes"]:
            resolution = await self._upsert_entity_node(node)
            resolutions[resolution.requested_entity_key] = resolution
        for edge in payload["edges"]:
            resolution = resolutions[edge["target"]]
            await self._client.execute_write(
                _edge_upsert_query(resolution.label, edge["type"]),
                {
                    "paper_id": edge["paper_id"],
                    "library_id": edge["library_id"],
                    "target_key": resolution.resolved_entity_key,
                    "properties": edge["properties"],
                },
            )
        return payload

    async def _upsert_entity_node(self, node: GraphNodePayload) -> EntityResolution:
        properties = node["properties"]
        entity_key = str(properties["entity_key"])
        aliases = [str(item) for item in properties.get("aliases", []) if str(item).strip()]
        candidates = _normalized_candidates(str(properties["name"]), aliases, entity_key)
        existing = await self._find_existing_entity(str(node["label"]), str(node["library_id"]), entity_key, candidates, aliases)
        if existing:
            resolved_key = str(existing.get("entity_key") or entity_key)
            merged_aliases = dedupe_preserve_order([*aliases, *existing.get("aliases", [])])
            await self._client.execute_write(
                _existing_node_update_query(str(node["label"])),
                {"entity_key": resolved_key, "properties": {**properties, "entity_key": resolved_key, "aliases": merged_aliases}},
            )
            return EntityResolution(entity_key, resolved_key, str(node["label"]), merged_aliases)
        await self._client.execute_write(
            _new_node_upsert_query(str(node["label"])),
            {"entity_key": entity_key, "properties": properties},
        )
        return EntityResolution(entity_key, entity_key, str(node["label"]), aliases)

    async def _find_existing_entity(
        self,
        label: str,
        library_id: str,
        entity_key: str,
        normalized_candidates: list[str],
        aliases: list[str],
    ) -> dict | None:
        rows = await self._client.execute_read(
            _existing_node_query(label),
            {
                "library_id": library_id,
                "entity_key": entity_key,
                "normalized_candidates": normalized_candidates,
                "aliases": aliases,
            },
        )
        if not rows:
            return None
        node = rows[0].get("n", rows[0])
        return node if isinstance(node, dict) else dict(node)


def _aliases_for(value: str, alias_map: LlmExtractionAliasMap) -> list[str]:
    aliases = alias_map.get(value, [])
    return dedupe_preserve_order([value, *aliases])


def _normalized_candidates(name: str, aliases: list[str], entity_key: str) -> list[str]:
    normalized_from_key = entity_key.split(":", 2)[-1]
    values = [name, *aliases, normalized_from_key]
    return dedupe_preserve_order([normalize_entity_name(value) for value in values])


def _existing_node_query(label: str) -> str:
    return f"""
    MATCH (n:{label})
    WHERE n.library_id = $library_id
      AND (
        n.entity_key = $entity_key
        OR n.normalized_name IN $normalized_candidates
        OR n.name IN $aliases
        OR any(alias IN coalesce(n.aliases, []) WHERE alias IN $aliases)
      )
    RETURN n
    LIMIT 1
    """


def _existing_node_update_query(label: str) -> str:
    return f"""
    MATCH (n:{label} {{entity_key: $entity_key}})
    SET n += $properties
    """


def _new_node_upsert_query(label: str) -> str:
    return f"""
    MERGE (n:{label} {{entity_key: $entity_key}})
    SET n += $properties
    """


def _edge_upsert_query(target_label: str, edge_type: str) -> str:
    return f"""
    MATCH (p:Paper {{paper_id: $paper_id}})
    MATCH (n:{target_label} {{entity_key: $target_key}})
    MERGE (p)-[r:{edge_type} {{paper_id: $paper_id, library_id: $library_id, target_key: $target_key}}]->(n)
    SET r += $properties
    """


def get_research_object_graph_service() -> ResearchObjectGraphService:
    """返回图写入 service 单例。"""
    global _service
    if _service is None:
        _service = ResearchObjectGraphService()
    return _service
