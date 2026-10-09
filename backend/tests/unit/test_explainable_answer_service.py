import asyncio

from app.services.graph.explainable_answer_service import ExplainableAnswerService


class FakeEvidencePathService:
    async def find_paths(self, paper_ids: list[str], *, path_type: str = "support", max_paths: int = 3):
        return [
            {
                "path_id": "support-0",
                "path_type": path_type,
                "summary": "p1 与 p2 通过 Transformer 相关",
                "support_paper_ids": paper_ids,
                "confidence": 0.7,
                "nodes": [],
                "edges": [],
            }
        ]


def test_build_bundle_merges_citations_and_paths(monkeypatch) -> None:
    service = ExplainableAnswerService()
    monkeypatch.setattr(
        "app.services.graph.explainable_answer_service.get_evidence_path_service",
        lambda: FakeEvidencePathService(),
    )
    result = asyncio.run(
        service.build_bundle(
            citations=[{"paper_id": "p1", "chunk_id": "c1", "excerpt": "evidence"}],
            paper_ids=["p2"],
            query="why",
            path_type="support",
        )
    )
    assert [item["paper_id"] for item in result["paper_sources"]] == ["p2", "p1"]
    assert result["chunk_citations"][0]["chunk_id"] == "c1"
    assert result["graph_evidence"][0]["path_id"] == "support-0"
    assert "2 篇论文" in result["evidence_summary"]
