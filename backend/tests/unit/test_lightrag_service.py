from __future__ import annotations

import asyncio

import pytest

from app.services.lightrag_insert_status import LightRAGInsertFailure
from app.services.lightrag_domain_prompt import build_addon_params
import app.services.lightrag_service as lightrag_module


def test_lightrag_embedding_func_uses_tei_client(monkeypatch) -> None:
    class FakeEmbeddingClient:
        def __init__(self) -> None:
            self.dimension = 3
            self.calls: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)]

    def fake_wrap_embedding_func_with_attrs(**attrs):
        def decorator(func):
            for key, value in attrs.items():
                setattr(func, key, value)
            return func
        return decorator

    monkeypatch.setattr(lightrag_module, "wrap_embedding_func_with_attrs", fake_wrap_embedding_func_with_attrs)
    monkeypatch.setattr(lightrag_module, "get_embedding_client", lambda: FakeEmbeddingClient())
    monkeypatch.setattr(lightrag_module.settings, "embedding_batch_size", 4, raising=False)
    monkeypatch.setattr(lightrag_module.settings, "lightrag_embedding_batch_size", 2, raising=False)

    service = lightrag_module.LightRAGService()
    embedding_func = service._build_embedding_func()
    vectors = asyncio.run(embedding_func(["a", "bb", "ccc"]))

    assert vectors.shape == (3, 3)
    assert vectors.tolist() == [
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [3.0, 0.0, 1.0],
    ]
    assert getattr(embedding_func, "embedding_dim") == 3
    assert getattr(embedding_func, "max_token_size") == lightrag_module.MAX_EMBEDDING_TOKENS


def test_lightrag_query_disables_internal_rerank(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRag:
        async def aquery(self, query: str, param) -> str:  # noqa: ANN001
            captured["query"] = query
            captured["param"] = param
            return "graph answer"

    monkeypatch.setattr(lightrag_module.settings, "graph_rag_enabled", True, raising=False)
    monkeypatch.setattr(lightrag_module.settings, "graph_rerank_enabled", False, raising=False)

    service = lightrag_module.LightRAGService()
    runtime = service._get_runtime("lib_graph")
    runtime.rag = FakeRag()
    runtime.initialized = True

    result = asyncio.run(service.query("what is graph rag", mode="hybrid", top_k=3, library_id="lib_graph"))

    assert result == "graph answer"
    assert captured["query"] == "what is graph rag"
    assert getattr(captured["param"], "mode") == "hybrid"
    assert getattr(captured["param"], "top_k") == 3
    assert getattr(captured["param"], "enable_rerank") is False


def test_collect_library_graph_hint_reuses_query_contract(monkeypatch) -> None:
    class FakeService(lightrag_module.LightRAGService):
        async def query(self, query: str, mode="hybrid", top_k=5, *, library_id=None):  # noqa: ANN001
            return f"{query}|{mode}|{top_k}|{library_id}"

    service = FakeService()
    result = asyncio.run(
        service.collect_library_graph_hint(
            "survey transformers",
            mode="global",
            top_k=4,
            library_id="lib_hint",
        )
    )

    assert result == "survey transformers|global|4|lib_hint"


def test_lightrag_insert_raises_when_doc_status_failed(monkeypatch) -> None:
    class FakeDocStatus:
        async def get_by_id(self, paper_id: str):
            assert paper_id == "paper_failed"
            return {"status": "failed", "error_msg": "524 Server Error"}

    class FakeRag:
        doc_status = FakeDocStatus()

        async def ainsert(self, text: str, ids: list[str]):  # noqa: ARG002
            assert ids == ["paper_failed"]
            return "track-1"

    monkeypatch.setattr(lightrag_module.settings, "graph_rag_enabled", True, raising=False)
    service = lightrag_module.LightRAGService()
    runtime = service._get_runtime("lib_graph")
    runtime.rag = FakeRag()
    runtime.initialized = True

    with pytest.raises(LightRAGInsertFailure, match="524 Server Error"):
        asyncio.run(service.insert("paper text", paper_id="paper_failed", library_id="lib_graph"))


def test_build_addon_params_omits_entity_types_when_disabled() -> None:
    assert build_addon_params(False) == {"language": "Chinese"}
