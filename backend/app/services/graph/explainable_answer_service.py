"""
统一聚合 chunk 引用与 graph evidence。

RAG 最终回答阶段会同时产生两类证据：
1. chunk 级引用
2. 图谱级可解释路径

这个服务的职责就是把它们统一收敛成一个 `EvidenceBundle`，
让上层 API 不必分别拼装 paper_sources / chunk_citations / graph_evidence。
"""

from __future__ import annotations

from app.models.evidence_models import ChunkCitationRef, EvidenceBundle, PaperSourceRef
from app.services.graph.evidence_path_service import get_evidence_path_service

_service: "ExplainableAnswerService | None" = None


class ExplainableAnswerService:
    """生成统一 EvidenceBundle。"""

    async def build_bundle(
        self,
        *,
        citations: list[dict] | None = None,
        paper_ids: list[str] | None = None,
        query: str = "",
        path_type: str = "support",
    ) -> EvidenceBundle:
        """
        基于 citations 和 paper_ids 构造统一证据包。

        这里会先把 citation 中出现但调用方未显式传入的 paper_id 补齐，
        再去图谱里查跨论文的可解释路径。
        """
        citation_items = citations or []
        known_paper_ids = [paper_id for paper_id in (paper_ids or []) if paper_id]
        for citation in citation_items:
            paper_id = str(citation.get("paper_id", "")).strip()
            if paper_id and paper_id not in known_paper_ids:
                known_paper_ids.append(paper_id)

        graph_evidence = await get_evidence_path_service().find_paths(
            known_paper_ids,
            path_type=path_type,
        )
        paper_sources: list[PaperSourceRef] = [{"paper_id": paper_id} for paper_id in known_paper_ids]
        chunk_citations: list[ChunkCitationRef] = [
            {
                "paper_id": citation.get("paper_id", ""),
                "chunk_id": citation.get("chunk_id", ""),
                "section": citation.get("section", ""),
                "page_num": citation.get("page_num", 0),
                "excerpt": citation.get("excerpt", ""),
            }
            for citation in citation_items
        ]
        summary = self._build_summary(paper_sources, graph_evidence, query=query)
        return {
            "paper_sources": paper_sources,
            "chunk_citations": chunk_citations,
            "graph_evidence": graph_evidence,
            "evidence_summary": summary,
        }

    def _build_summary(
        self,
        paper_sources: list[PaperSourceRef],
        graph_evidence: list[dict],
        *,
        query: str,
    ) -> str:
        """生成面向前端展示的证据摘要文本。"""
        if not paper_sources:
            return ""
        summary = f"本次回答引用了 {len(paper_sources)} 篇论文。"
        if graph_evidence:
            summary += f" 图谱中找到 {len(graph_evidence)} 条可解释路径。"
        elif query:
            summary += " 图谱中未命中额外路径。"
        return summary


def get_explainable_answer_service() -> ExplainableAnswerService:
    """返回 explainable answer service 单例。"""
    global _service
    if _service is None:
        _service = ExplainableAnswerService()
    return _service
