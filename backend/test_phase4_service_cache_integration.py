from __future__ import annotations

import os
from typing import Any

os.environ["SSLKEYLOGFILE"] = ""

import app.services.cache as cache_module
import app.services.llm_service as llm_service
import app.services.milvus_store as milvus_store


def _reset_cache_singleton() -> None:
    cache_module._cache = None


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.dimension = 2

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload
        self.text = str(payload).replace("'", '"')
        self.content = self.text.encode("utf-8")

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_milvus_store_encode_reuses_cached_embeddings(monkeypatch) -> None:
    _reset_cache_singleton()
    monkeypatch.setattr(cache_module.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(milvus_store.settings, "embedding_batch_size", 8, raising=False)

    store = milvus_store.MilvusStore()
    fake_client = _FakeEmbeddingClient()
    monkeypatch.setattr(milvus_store, "get_embedding_client", lambda: fake_client)

    first = store._encode(["alpha", "beta", "alpha"])
    second = store._encode(["beta", "alpha"])

    assert fake_client.calls == [["alpha", "beta"]]
    assert first == [[5.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
    assert second == [[4.0, 1.0], [5.0, 0.0]]

    stats = cache_module.get_cache().get_stats()
    assert stats["embedding_hits"] == 2
    assert stats["embedding_misses"] == 3


def test_llm_invoke_reuses_cached_response(monkeypatch) -> None:
    _reset_cache_singleton()
    monkeypatch.setattr(cache_module.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(llm_service.time, "sleep", lambda _: None)

    requests_count = {"value": 0}

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        requests_count["value"] += 1
        return _FakeResponse({"choices": [{"message": {"content": "cached-answer"}}]})

    monkeypatch.setattr(llm_service.requests, "post", fake_post)

    client = llm_service.OpenAICompatibleLLM(
        model="test-model",
        api_key="test-key",
        base_url="http://test-host/v1",
        temperature=0.0,
        timeout=30,
    )

    first = client.invoke("hello")
    second = client.invoke("hello")

    assert requests_count["value"] == 1
    assert first.content == second.content == "cached-answer"

    stats = cache_module.get_cache().get_stats()
    assert stats["llm_hits"] == 1
    assert stats["llm_misses"] == 1


def test_service_cache_integration_skips_cache_when_disabled(monkeypatch) -> None:
    _reset_cache_singleton()
    monkeypatch.setattr(cache_module.settings, "cache_enabled", False, raising=False)
    monkeypatch.setattr(milvus_store.settings, "embedding_batch_size", 8, raising=False)
    monkeypatch.setattr(llm_service.time, "sleep", lambda _: None)

    store = milvus_store.MilvusStore()
    fake_client = _FakeEmbeddingClient()
    monkeypatch.setattr(milvus_store, "get_embedding_client", lambda: fake_client)

    requests_count = {"value": 0}

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        requests_count["value"] += 1
        return _FakeResponse({"choices": [{"message": {"content": "no-cache"}}]})

    monkeypatch.setattr(llm_service.requests, "post", fake_post)

    client = llm_service.OpenAICompatibleLLM(
        model="test-model",
        api_key="test-key",
        base_url="http://test-host/v1",
        temperature=0.0,
        timeout=30,
    )

    store._encode(["alpha"])
    store._encode(["alpha"])
    client.invoke("hello")
    client.invoke("hello")

    assert fake_client.calls == [["alpha"], ["alpha"]]
    assert requests_count["value"] == 2
