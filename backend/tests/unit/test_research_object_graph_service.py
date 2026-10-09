import asyncio

from app.services.graph.research_object_graph_service import ResearchObjectGraphService


class FakeNeo4jClient:
    def __init__(self) -> None:
        self.enabled = True
        self.writes: list[tuple[str, dict]] = []

    async def execute_write(self, query: str, parameters: dict | None = None) -> None:
        self.writes.append((query, parameters or {}))

    async def execute_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        return []


def test_build_payload_contains_task_and_paper_anchor(monkeypatch) -> None:
    import app.services.graph.research_object_graph_service as graph_module

    monkeypatch.setattr(graph_module.settings, "graph_extraction_mode", "regex", raising=False)
    service = ResearchObjectGraphService(client=FakeNeo4jClient())
    paper = {
        "paper_id": "upload_1",
        "library_id": "default",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "categories": ["cs.CL"],
        "published_date": "2017-06-12",
        "source": "upload",
        "analysis": {
            "problem": "Machine translation with attention.",
            "method": "Transformer",
            "contribution": ["Improves translation quality"],
            "limitation": "High compute cost",
            "keywords": ["machine translation"],
            "related_work": [],
            "tldr": "Transformer for MT",
            "methodology_type": "experimental",
            "methodology_detail": "Uses BLEU on WMT.",
        },
    }
    payload = asyncio.run(service.build_payload(paper, "We report BLEU."))
    assert payload["paper"]["properties"]["paper_id"] == "upload_1"
    assert payload["paper"]["properties"]["year"] == 2017
    assert any(node["label"] == "Task" for node in payload["nodes"])
    assert any(edge["type"] == "USES_METHOD" for edge in payload["edges"])
    method_node = next(node for node in payload["nodes"] if node["label"] == "Method")
    assert method_node["properties"]["aliases"] == ["Transformer"]


def test_upsert_paper_subgraph_emits_writes(monkeypatch) -> None:
    import app.services.graph.research_object_graph_service as graph_module

    monkeypatch.setattr(graph_module.settings, "graph_extraction_mode", "regex", raising=False)
    client = FakeNeo4jClient()
    service = ResearchObjectGraphService(client=client)
    paper = {
        "paper_id": "upload_1",
        "library_id": "default",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "categories": ["cs.CL"],
        "published_date": "2017-06-12",
        "source": "upload",
        "analysis": {
            "problem": "Machine translation with attention.",
            "method": "Transformer",
            "contribution": ["Improves translation quality"],
            "limitation": "High compute cost",
            "keywords": ["machine translation"],
            "related_work": [],
            "tldr": "Transformer for MT",
            "methodology_type": "experimental",
            "methodology_detail": "Uses BLEU on WMT.",
        },
    }
    asyncio.run(service.upsert_paper_subgraph(paper, "We report BLEU."))
    assert len(client.writes) >= 3
    assert client.writes[0][1]["properties"]["year"] == 2017
