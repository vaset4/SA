from __future__ import annotations

import os

import app.config as config_module


def _reset_settings_singleton() -> None:
    config_module._settings_instance = None


def test_phase4_settings_expose_required_fields(monkeypatch) -> None:
    _reset_settings_singleton()
    monkeypatch.setenv("AUTH_ENABLED", "false")

    settings = config_module.Settings()

    assert settings.auth_enabled is False
    assert settings.jwt_secret_key == "change-me-in-production"
    assert settings.jwt_expire_hours == 24
    assert settings.api_key == ""
    assert settings.rate_limit == "60/minute"
    assert settings.cache_enabled is True
    assert settings.redis_cache_enabled is False
    assert settings.tracing_enabled is True
    assert settings.rag_similarity_threshold == 0.3
    assert settings.rag_rerank_threshold == 0.5
    assert settings.rerank_max_candidates == 20
    assert settings.rerank_provider == "bge"
    assert settings.rerank_api_key == ""
    assert settings.rerank_endpoint == ""
    assert settings.deep_research_max_parallel_units == 2
    assert settings.deep_research_max_researcher_iterations == 2
    assert settings.deep_research_max_react_tool_calls == 1
    assert settings.deep_research_max_search_queries_per_round == 1
    assert settings.deep_research_final_report_max_note_chars == 1200
    assert settings.deep_research_final_report_max_findings_chars == 6000
    assert settings.deep_research_final_report_raw_note_preview_count == 2
    assert settings.deep_research_final_report_raw_note_max_chars == 700


def test_phase4_settings_allow_explicit_auth_enable(monkeypatch) -> None:
    _reset_settings_singleton()
    monkeypatch.setenv("AUTH_ENABLED", "true")

    settings = config_module.Settings()

    assert settings.auth_enabled is True


def test_get_settings_still_returns_singleton() -> None:
    _reset_settings_singleton()

    first = config_module.get_settings()
    second = config_module.get_settings()

    assert first is second
