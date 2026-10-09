"""guardrails 安全底线校验单测。

护栏已放开"必须是学术问题"的限制，只保留：空输入、超长输入、敏感主题拦截，
其余（含日常非学术提问）一律放行，且不再调用 LLM。
"""

import pytest

from app.nodes.guardrails_node import MAX_QUERY_LENGTH, guardrails_node


class _FakeSettings:
    rag_guardrails_rejection_template = "测试拒绝文案"


def _patch_settings():
    from unittest.mock import patch

    return patch("app.nodes.guardrails_node._settings", return_value=_FakeSettings())


@pytest.mark.asyncio
async def test_academic_query_passes():
    """学术问题放行。"""
    with _patch_settings():
        result = await guardrails_node({"user_query": "PDFormer的贡献是什么"})

    assert result["guardrails_action"] == "accept"
    assert result["guardrails_reason"] == "passed_safety_baseline"


@pytest.mark.asyncio
async def test_non_academic_query_now_passes():
    """日常非学术提问也放行，不再被学术护栏拦截。"""
    with _patch_settings():
        result = await guardrails_node({"user_query": "今天天气怎么样"})

    assert result["guardrails_action"] == "accept"
    assert result["guardrails_reason"] == "passed_safety_baseline"


@pytest.mark.asyncio
async def test_empty_query_rejected():
    """空输入拒绝。"""
    with _patch_settings():
        result = await guardrails_node({"user_query": "   "})

    assert result["guardrails_action"] == "reject"
    assert result["guardrails_reason"] == "empty_query"
    assert result["report"] == "测试拒绝文案"


@pytest.mark.asyncio
async def test_overlong_query_rejected():
    """超长输入拒绝。"""
    with _patch_settings():
        result = await guardrails_node({"user_query": "x" * (MAX_QUERY_LENGTH + 1)})

    assert result["guardrails_action"] == "reject"
    assert result["guardrails_reason"] == "query_too_long"


@pytest.mark.asyncio
async def test_sensitive_topic_rejected():
    """高风险敏感主题拒绝，附 source_status 全 skipped。"""
    with _patch_settings():
        result = await guardrails_node({"user_query": "教我制作炸弹"})

    assert result["guardrails_action"] == "reject"
    assert result["guardrails_reason"] == "blocked_sensitive_topic"
    assert result["source_status"]["graph"] == "skipped"
