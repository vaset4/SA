"""争议与 gap 聚合服务。"""

from __future__ import annotations

from typing import Any

import app.services.paper_db as paper_db
from app.models.controversy_models import ControversyMapPayload, ControversyMapQuery
from app.services.graph.claim_conflict_service import get_claim_conflict_service
from app.services.graph.gap_detection_service import get_gap_detection_service

_service: "ControversyMapService | None" = None


class ControversyMapService:
    """聚合结构化冲突和 gap 结果。"""

    async def build_map(self, query: ControversyMapQuery) -> ControversyMapPayload:
        papers = self._load_papers(query)
        claim_service = get_claim_conflict_service()
        gap_service = get_gap_detection_service()

        conflicts = []
        gaps = []
        if query["include_conflicts"]:
            claims = claim_service.build_claim_triples(papers)
            conflicts = claim_service.detect_conflicts(claims)
        if query["include_gaps"]:
            gaps = gap_service.detect_gaps(papers, query["topic"])

        overview = self._overview(conflicts, gaps)
        scope = {
            "paper_ids": [paper.get("paper_id", "") for paper in papers if paper.get("paper_id")],
            "topic": query["topic"],
            "project_id": query["project_id"],
            "resolved_scope": query["resolved_scope"] or None,
        }
        return {"conflicts": conflicts, "gaps": gaps, "overview": overview, "scope": scope}

    def _load_papers(self, query: ControversyMapQuery) -> list[dict[str, Any]]:
        scope = query["resolved_scope"] or {}
        paper_ids = query["paper_ids"] or list(scope.get("paper_ids", []))
        papers: list[dict[str, Any]] = []
        for paper_id in paper_ids:
            paper = paper_db.get_paper(paper_id)
            if paper:
                papers.append(paper)
        return papers

    def _overview(self, conflicts: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> str:
        if not conflicts and not gaps:
            return "当前范围内未命中明确争议或 gap。"
        parts: list[str] = []
        if conflicts:
            parts.append(f"识别到 {len(conflicts)} 个争议簇")
        if gaps:
            parts.append(f"识别到 {len(gaps)} 个 gap 候选")
        return "，".join(parts) + "。"


def get_controversy_map_service() -> ControversyMapService:
    global _service
    if _service is None:
        _service = ControversyMapService()
    return _service
