from __future__ import annotations

import os

import pytest

os.environ["SSLKEYLOGFILE"] = ""

import app.api.server as server_module
import app.api.facades.system_facade as system_facade_module
import app.services.cache as cache_module
import app.services.llm_service as llm_service_module


class _FakeConnection:
    async def fetchval(self, query: str) -> int:
        assert query == "SELECT 1"
        return 1


class _AcquireContext:
    async def __aenter__(self) -> _FakeConnection:
        return _FakeConnection()

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


class _FakePool:
    def acquire(self) -> _AcquireContext:
        return _AcquireContext()


class _FakeRedis:
    async def ping(self) -> bool:
        return True


class _FakeL2:
    async def _get_redis(self) -> _FakeRedis:
        return _FakeRedis()


class _FakeCache:
    def __init__(self, stats: dict, l2=None) -> None:  # noqa: ANN001
        self._stats = stats
        self.l2 = l2

    def get_stats(self) -> dict:
        return self._stats


@pytest.mark.asyncio
async def test_health_reports_all_checks_without_redis_probe(monkeypatch) -> None:
    class FakeMilvusStore:
        def is_healthy(self) -> bool:
            return True

    async def fake_get_pool() -> _FakePool:
        return _FakePool()

    monkeypatch.setattr(system_facade_module, "get_pool", fake_get_pool)
    monkeypatch.setattr(system_facade_module, "get_milvus_store", lambda: FakeMilvusStore())
    monkeypatch.setattr(llm_service_module, "get_llm", lambda: object())
    monkeypatch.setattr(
        cache_module,
        "get_cache",
        lambda: _FakeCache({"embedding_hits": 1, "search_hits": 2, "llm_hits": 3}),
    )
    monkeypatch.setattr(server_module.settings, "redis_cache_enabled", False, raising=False)
    monkeypatch.setattr(server_module.settings, "graph_rag_enabled", False, raising=False)
    monkeypatch.setattr(server_module.settings, "neo4j_enabled", False, raising=False)

    payload = await system_facade_module.build_health_payload()

    assert payload["status"] == "healthy"
    assert payload["checks"]["postgres"] is True
    assert payload["checks"]["milvus"] is True
    assert payload["checks"]["llm"] is True
    assert payload["checks"]["redis"] is False
    assert payload["checks"]["cache"]["embedding_hits"] == 1
    assert payload["checks"]["cache"]["search_hits"] == 2
    assert payload["checks"]["cache"]["llm_hits"] == 3


@pytest.mark.asyncio
async def test_health_pings_redis_and_degrades_on_core_failure(monkeypatch) -> None:
    class FakeMilvusStore:
        def is_healthy(self) -> bool:
            return False

    async def broken_pool():  # noqa: ANN202
        raise RuntimeError("postgres unavailable")

    monkeypatch.setattr(system_facade_module, "get_pool", broken_pool)
    monkeypatch.setattr(system_facade_module, "get_milvus_store", lambda: FakeMilvusStore())
    monkeypatch.setattr(
        llm_service_module,
        "get_llm",
        lambda: (_ for _ in ()).throw(RuntimeError("llm unavailable")),
    )
    monkeypatch.setattr(
        cache_module,
        "get_cache",
        lambda: _FakeCache({"embedding_hits": 0}, l2=_FakeL2()),
    )
    monkeypatch.setattr(server_module.settings, "redis_cache_enabled", True, raising=False)
    monkeypatch.setattr(server_module.settings, "graph_rag_enabled", False, raising=False)
    monkeypatch.setattr(server_module.settings, "neo4j_enabled", False, raising=False)

    payload = await system_facade_module.build_health_payload()

    assert payload["status"] == "degraded"
    assert payload["checks"]["postgres"] is False
    assert payload["checks"]["milvus"] is False
    assert payload["checks"]["llm"] is False
    assert payload["checks"]["redis"] is True


@pytest.mark.asyncio
async def test_health_includes_neo4j_when_enabled(monkeypatch) -> None:
    class FakeMilvusStore:
        def is_healthy(self) -> bool:
            return True

    class FakeGraphQueryService:
        async def health(self) -> dict:
            return {"enabled": True, "connected": True, "schema_ready": True}

    async def fake_get_pool() -> _FakePool:
        return _FakePool()

    monkeypatch.setattr(system_facade_module, "get_pool", fake_get_pool)
    monkeypatch.setattr(system_facade_module, "get_milvus_store", lambda: FakeMilvusStore())
    monkeypatch.setattr(system_facade_module, "get_research_object_query_service", lambda: FakeGraphQueryService())
    monkeypatch.setattr(llm_service_module, "get_llm", lambda: object())
    monkeypatch.setattr(cache_module, "get_cache", lambda: _FakeCache({"embedding_hits": 0}))
    monkeypatch.setattr(server_module.settings, "redis_cache_enabled", False, raising=False)
    monkeypatch.setattr(server_module.settings, "graph_rag_enabled", False, raising=False)
    monkeypatch.setattr(server_module.settings, "neo4j_enabled", True, raising=False)

    payload = await system_facade_module.build_health_payload()

    assert payload["status"] == "healthy"
    assert payload["checks"]["neo4j"]["connected"] is True
