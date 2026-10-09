"""Phase 1 配置项契约测试。

对应任务 05-23-cross-paper-qa-retrieval-fix 的 Slice A。

只验证 11 个新增字段的存在、类型与默认值，不验证业务逻辑。
"""

from __future__ import annotations

from app.config import Settings, get_settings


def _fresh_settings() -> Settings:
    """绕过单例，构造一份默认实例，避免被环境变量污染影响断言。"""
    return Settings()


def test_enough_guard_keys_exist_with_defaults() -> None:
    s = _fresh_settings()
    assert s.rag_enough_guard_enabled is True
    assert s.rag_enough_guard_min_vector_calls == 1
    assert s.rag_enough_guard_min_graph_calls == 1


def test_neo4j_timeout_keys_exist_with_defaults() -> None:
    s = _fresh_settings()
    assert s.neo4j_connect_timeout_seconds == 5.0
    assert s.neo4j_read_timeout_seconds == 10.0


def test_paper_analysis_llm_timeout_stays_below_cloudflare_524_window() -> None:
    s = _fresh_settings()
    assert s.paper_analysis_llm_timeout_seconds == 180
    assert s.paper_analysis_llm_timeout_seconds < 200
    assert s.paper_analysis_llm_max_retries == 0


def test_runtime_llm_rate_limit_defaults_match_provider_limit() -> None:
    s = Settings(_env_file=None)
    assert s.llm_queue_requests_per_window == 5
    assert s.llm_queue_window_seconds == 60
    assert "llm_rate_limit_max_retries" not in Settings.model_fields


def test_unindexed_notice_template_has_placeholder() -> None:
    s = _fresh_settings()
    assert "{paper_titles_csv}" in s.rag_unindexed_paper_notice_template


def test_graph_hint_unavailable_template_has_reason_placeholder() -> None:
    s = _fresh_settings()
    assert "{reason}" in s.rag_graph_hint_unavailable_template


def test_graph_hint_skipped_template_nonempty() -> None:
    s = _fresh_settings()
    assert isinstance(s.rag_graph_hint_skipped_template, str)
    assert len(s.rag_graph_hint_skipped_template) > 0


def test_graph_quality_gate_keys_exist_with_defaults() -> None:
    s = _fresh_settings()
    assert s.rag_graph_quality_gate_enabled is True
    assert isinstance(s.rag_graph_hint_ok_but_unused_template, str)
    assert len(s.rag_graph_hint_ok_but_unused_template) > 0


def test_citation_regex_patterns_is_nonempty_list_of_str() -> None:
    s = _fresh_settings()
    assert isinstance(s.rag_citation_regex_patterns, list)
    assert len(s.rag_citation_regex_patterns) >= 3
    assert all(isinstance(p, str) for p in s.rag_citation_regex_patterns)


def test_citation_regex_patterns_have_capture_group() -> None:
    """每个正则的第一个捕获组就是 chunk_id（见 design §3 / §4.5.3）。"""
    import re

    s = _fresh_settings()
    for pattern in s.rag_citation_regex_patterns:
        compiled = re.compile(pattern)
        assert compiled.groups >= 1, f"pattern {pattern!r} must have at least 1 capture group"


def test_get_settings_returns_singleton_with_phase1_keys() -> None:
    """get_settings() 单例也必须能访问 Phase 1 字段。"""
    s = get_settings()
    assert hasattr(s, "rag_enough_guard_enabled")
    assert hasattr(s, "rag_graph_quality_gate_enabled")
    assert hasattr(s, "rag_citation_regex_patterns")
