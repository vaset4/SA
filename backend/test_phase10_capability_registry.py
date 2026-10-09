from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ["SSLKEYLOGFILE"] = ""

import app.api.facades.system_facade as system_facade
import app.services.capability_registry as capability_registry
import app.services.retrieval_provider_registry as retrieval_provider_registry


def _fake_settings(
    *,
    vector_backend: str = "milvus",
    graph_rag_enabled: bool = True,
    tavily_api_key: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        llm_model_name="test-model",
        llm_api_key="sk-test",
        llm_base_url="https://example.com/v1",
        llm_temperature=0.1,
        llm_timeout=30,
        vector_backend=vector_backend,
        graph_rag_enabled=graph_rag_enabled,
        tavily_api_key=tavily_api_key,
    )


def test_default_vector_provider_follows_settings(monkeypatch) -> None:
    monkeypatch.setattr(retrieval_provider_registry, "get_settings", lambda: _fake_settings(vector_backend="faiss"))
    monkeypatch.setattr(retrieval_provider_registry, "get_vector_store", lambda: "FAISS_PROVIDER")
    monkeypatch.setattr(retrieval_provider_registry, "get_milvus_store", lambda: "MILVUS_PROVIDER")

    assert retrieval_provider_registry.get_default_vector_provider_name() == "faiss"
    assert retrieval_provider_registry.get_default_vector_provider() == "FAISS_PROVIDER"


def test_graph_provider_respects_toggle(monkeypatch) -> None:
    monkeypatch.setattr(retrieval_provider_registry, "get_settings", lambda: _fake_settings(graph_rag_enabled=False))

    assert retrieval_provider_registry.get_graph_provider_name() is None
    assert retrieval_provider_registry.get_graph_provider() is None
    enabled = retrieval_provider_registry.list_enabled_retrieval_providers()
    assert all(item["name"] != "lightrag" for item in enabled)


def test_capability_snapshot_contains_registry_state(monkeypatch) -> None:
    fake_settings = _fake_settings(vector_backend="faiss", graph_rag_enabled=True, tavily_api_key="")
    monkeypatch.setattr(capability_registry, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(retrieval_provider_registry, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(system_facade, "capability_snapshot", lambda: capability_registry.capability_snapshot())

    snapshot = system_facade.capability_snapshot_payload()

    assert snapshot["models"][0]["name"] == "test-model"
    assert snapshot["retrieval"]["default_vector_provider"] == "faiss"
    assert snapshot["retrieval"]["graph_provider"] == "lightrag"
    assert snapshot["config"]["graph_rag_enabled"] is True
    assert "tools" in snapshot
