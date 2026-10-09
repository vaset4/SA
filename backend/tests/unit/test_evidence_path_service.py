import asyncio

from app.services.graph.evidence_path_service import EvidencePathService


class FakeNeo4jClient:
    def __init__(self, enabled: bool = True, rows: list[dict] | None = None) -> None:
        self.enabled = enabled
        self._rows = rows or []

    async def execute_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        return self._rows


def test_find_paths_returns_empty_when_disabled() -> None:
    service = EvidencePathService()
    service._client = FakeNeo4jClient(enabled=False)
    result = asyncio.run(service.find_paths(["p1", "p2"]))
    assert result == []


def test_find_paths_builds_shared_entity_paths() -> None:
    service = EvidencePathService()
    service._client = FakeNeo4jClient(
        rows=[
            {
                "source_paper_id": "p1",
                "target_paper_id": "p2",
                "node_label": "Method",
                "node_name": "Transformer",
                "entity_key": "default:Method:transformer",
                "left_relation": "USES_METHOD",
                "right_relation": "USES_METHOD",
            }
        ]
    )
    result = asyncio.run(service.find_paths(["p1", "p2"], path_type="comparison"))
    assert len(result) == 1
    assert result[0]["path_type"] == "comparison"
    assert result[0]["support_paper_ids"] == ["p1", "p2"]
