"""研究演化地图导出服务。"""

from __future__ import annotations

from collections import Counter

from app.models.evolution_models import EvolutionEdge, EvolutionFocus, EvolutionGraphPayload, EvolutionNode, TimelineEvent

_export_service: "EvolutionExportService | None" = None


class EvolutionExportService:
    """把原始节点边转换成前端友好的演化 payload。"""

    def build_payload(
        self,
        *,
        focus: EvolutionFocus,
        nodes: list[EvolutionNode],
        edges: list[EvolutionEdge],
        filters: dict[str, object],
    ) -> EvolutionGraphPayload:
        timeline = self._build_timeline(nodes, edges)
        stats = self._build_stats(nodes, edges, timeline)
        return {
            "focus": focus,
            "timeline": timeline,
            "nodes": nodes,
            "edges": edges,
            "stats": stats,
            "filters": filters,
        }

    def build_entry(
        self,
        *,
        focus: EvolutionFocus,
        filters: dict[str, object],
        target_url: str,
    ) -> dict[str, object]:
        return {
            "focus": focus,
            "recommended_filters": filters,
            "target_url": target_url,
        }

    def _build_timeline(
        self,
        nodes: list[EvolutionNode],
        edges: list[EvolutionEdge],
    ) -> list[TimelineEvent]:
        paper_nodes = [node for node in nodes if node["node_type"] == "paper" and node["year"] is not None]
        paper_nodes.sort(key=lambda item: (item["year"] or 0, item["label"].lower(), item["id"]))
        events: list[TimelineEvent] = []
        for node in paper_nodes:
            outbound = [
                edge["relation"]
                for edge in edges
                if edge["source"] == node["id"] and edge["relation"] in {"CITES", "EXTENDS"}
            ]
            inbound = [
                edge["relation"]
                for edge in edges
                if edge["target"] == node["id"] and edge["relation"] in {"CITES", "EXTENDS"}
            ]
            object_links = sum(
                1
                for edge in edges
                if edge["source"] == node["id"] and edge["relation"] not in {"CITES", "EXTENDS"}
            )
            summary_parts: list[str] = []
            if outbound:
                summary_parts.append(f"向外连接 {len(outbound)} 条论文关系")
            if inbound:
                summary_parts.append(f"被后续路径命中 {len(inbound)} 次")
            if object_links:
                summary_parts.append(f"关联 {object_links} 个研究对象")
            events.append(
                {
                    "year": int(node["year"] or 0),
                    "title": node["label"],
                    "event_type": "paper",
                    "paper_id": node["paper_id"],
                    "summary": "；".join(summary_parts) or "进入研究演化视图",
                }
            )
        return events

    def _build_stats(
        self,
        nodes: list[EvolutionNode],
        edges: list[EvolutionEdge],
        timeline: list[TimelineEvent],
    ) -> dict[str, object]:
        node_counter = Counter(node["node_type"] for node in nodes)
        relation_counter = Counter(edge["relation"] for edge in edges)
        years = sorted({event["year"] for event in timeline})
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "timeline_count": len(timeline),
            "node_types": dict(node_counter),
            "relations": dict(relation_counter),
            "year_range": [years[0], years[-1]] if years else [],
        }


def get_evolution_export_service() -> EvolutionExportService:
    """返回导出 service 单例。"""
    global _export_service
    if _export_service is None:
        _export_service = EvolutionExportService()
    return _export_service
