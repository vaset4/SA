import asyncio

from app.services.graph.research_object_graph_service import ResearchObjectGraphService


class DisabledClient:
    enabled = False


def _paper() -> dict:
    return {
        "paper_id": "paper_1",
        "library_id": "lib_1",
        "title": "Graph WaveNet",
        "authors": [],
        "analysis": {
            "problem": "Traffic forecasting.",
            "method": "Graph WaveNet",
            "contribution": [],
            "limitation": "",
            "keywords": ["traffic forecasting"],
            "methodology_detail": "Evaluated on METR-LA with MAE.",
        },
    }


def test_mode_regex_calls_legacy_extractor(monkeypatch) -> None:
    import app.services.graph.research_object_graph_service as graph_module

    monkeypatch.setattr(graph_module.settings, "graph_extraction_mode", "regex", raising=False)
    payload = asyncio.run(ResearchObjectGraphService(client=DisabledClient()).build_payload(_paper(), ""))

    assert any(node["name"] == "Graph WaveNet" for node in payload["nodes"])


def test_mode_llm_calls_extract_with_llm(monkeypatch) -> None:
    import app.services.graph.research_object_graph_service as graph_module

    async def fake_extract(paper, full_text):  # noqa: ANN001
        return {
            "authors": [],
            "venue": "",
            "methods": ["STGCN"],
            "datasets": [],
            "metrics": [],
            "claims": [],
            "limitations": [],
            "tasks": [],
        }, {"STGCN": ["Spatio-Temporal Graph Convolutional Network"]}

    monkeypatch.setattr(graph_module.settings, "graph_extraction_mode", "llm", raising=False)
    monkeypatch.setattr(graph_module, "extract_with_llm", fake_extract)

    payload = asyncio.run(ResearchObjectGraphService(client=DisabledClient()).build_payload(_paper(), "text"))

    method_node = next(node for node in payload["nodes"] if node["label"] == "Method")
    assert method_node["name"] == "STGCN"
    assert method_node["properties"]["aliases"] == ["STGCN", "Spatio-Temporal Graph Convolutional Network"]


def test_mode_default_is_llm() -> None:
    import app.services.graph.research_object_graph_service as graph_module

    assert graph_module.settings.graph_extraction_mode == "llm"
