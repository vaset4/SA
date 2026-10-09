import asyncio

from app.services.graph.research_object_graph_service import ResearchObjectGraphService


class AliasMergeClient:
    def __init__(self) -> None:
        self.enabled = True
        self.reads: list[tuple[str, dict]] = []
        self.writes: list[tuple[str, dict]] = []

    async def execute_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        self.reads.append((query, parameters or {}))
        aliases = parameters.get("aliases", []) if parameters else []
        if "STGCN" in aliases:
            return [{"n": {"entity_key": "lib_1:Method:stgcn", "aliases": ["STGCN"]}}]
        return []

    async def execute_write(self, query: str, parameters: dict | None = None) -> None:
        self.writes.append((query, parameters or {}))


def _paper() -> dict:
    return {
        "paper_id": "paper_1",
        "library_id": "lib_1",
        "title": "ST-GCN Followup",
        "authors": [],
        "analysis": {},
    }


def test_alias_match_rewrites_edge_to_resolved_entity_key(monkeypatch) -> None:
    import app.services.graph.research_object_graph_service as graph_module

    async def fake_extract(paper, full_text):  # noqa: ANN001
        return {
            "authors": [],
            "venue": "",
            "methods": ["ST-GCN"],
            "datasets": [],
            "metrics": [],
            "claims": [],
            "limitations": [],
            "tasks": [],
        }, {"ST-GCN": ["STGCN"]}

    monkeypatch.setattr(graph_module.settings, "graph_extraction_mode", "llm", raising=False)
    monkeypatch.setattr(graph_module, "extract_with_llm", fake_extract)
    client = AliasMergeClient()
    service = ResearchObjectGraphService(client=client)

    asyncio.run(service.upsert_paper_subgraph(_paper(), "ST-GCN and STGCN describe the same model."))

    edge_write = next(item for item in client.writes if "MERGE (p)-[r:USES_METHOD" in item[0])
    assert edge_write[1]["target_key"] == "lib_1:Method:stgcn"
    update_write = next(item for item in client.writes if "MATCH (n:Method {entity_key: $entity_key})" in item[0])
    assert update_write[1]["properties"]["aliases"] == ["ST-GCN", "STGCN"]
