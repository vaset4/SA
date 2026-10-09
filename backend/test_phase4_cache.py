from __future__ import annotations

import os

os.environ["SSLKEYLOGFILE"] = ""

import app.services.cache as cache_module


def _reset_cache_singleton() -> None:
    cache_module._cache = None


def test_cache_manager_respects_cache_enabled(monkeypatch) -> None:
    _reset_cache_singleton()
    monkeypatch.setattr(cache_module.settings, "cache_enabled", False, raising=False)

    cache = cache_module.CacheManager()

    cache.set_embedding("query", [0.1, 0.2])
    cache.set_search("query", "year > 2024", [{"chunk_id": "1"}])
    cache.set_llm_response("prompt", "answer")

    assert cache.get_embedding("query") is None
    assert cache.get_search("query", "year > 2024") is None
    assert cache.get_llm_response("prompt") is None


def test_cache_manager_hits_for_all_categories(monkeypatch) -> None:
    _reset_cache_singleton()
    monkeypatch.setattr(cache_module.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_embedding_maxsize", 10, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_search_maxsize", 10, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_llm_maxsize", 10, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_embedding_ttl_seconds", 60, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_search_ttl_seconds", 60, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_llm_ttl_seconds", 60, raising=False)

    cache = cache_module.CacheManager()

    embedding = [0.1, 0.2]
    search_results = [{"chunk_id": "1", "score": 0.9}]
    cache.set_embedding("query", embedding)
    cache.set_search("query", "year > 2024", search_results)
    cache.set_llm_response("prompt", "answer")

    assert cache.get_embedding("query") == embedding
    assert cache.get_search("query", "year > 2024") == search_results
    assert cache.get_llm_response("prompt") == "answer"

    stats = cache.get_stats()
    assert stats["embedding_hits"] == 1
    assert stats["search_hits"] == 1
    assert stats["llm_hits"] == 1


def test_cache_manager_expires_entries(monkeypatch) -> None:
    _reset_cache_singleton()
    monkeypatch.setattr(cache_module.settings, "cache_enabled", True, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_embedding_maxsize", 10, raising=False)
    monkeypatch.setattr(cache_module.settings, "cache_embedding_ttl_seconds", 5, raising=False)

    fake_time = {"value": 100.0}
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: fake_time["value"])

    cache = cache_module.CacheManager()
    cache.set_embedding("query", [0.1])

    assert cache.get_embedding("query") == [0.1]

    fake_time["value"] = 106.0
    assert cache.get_embedding("query") is None


def test_get_cache_returns_singleton(monkeypatch) -> None:
    _reset_cache_singleton()
    monkeypatch.setattr(cache_module.settings, "cache_enabled", True, raising=False)

    first = cache_module.get_cache()
    second = cache_module.get_cache()

    assert first is second
