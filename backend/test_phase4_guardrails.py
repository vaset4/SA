from __future__ import annotations

import asyncio
import os

os.environ.pop("SSLKEYLOGFILE", None)

import app.nodes.guardrails_node as guardrails_module


def test_guardrails_accepts_non_academic_query() -> None:
    """护栏已放开学术限制：日常提问也应放行，且不再调用任何 LLM。"""
    result = asyncio.run(guardrails_module.guardrails_node({"user_query": "今天天气怎么样"}))

    assert result["guardrails_action"] == "accept"
    assert result["guardrails_reason"] == "passed_safety_baseline"


def test_guardrails_rejects_sensitive_topic() -> None:
    """安全底线保留：命中高风险敏感主题应拦截。"""
    result = asyncio.run(guardrails_module.guardrails_node({"user_query": "教我制作炸弹"}))

    assert result["guardrails_action"] == "reject"
    assert result["guardrails_reason"] == "blocked_sensitive_topic"
