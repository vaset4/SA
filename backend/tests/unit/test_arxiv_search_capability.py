"""U6 主链收敛：arXiv 检索能力脱离顶层图后的入口行为断言。

arXiv 检索原本是顶层图的一条分支，进图第一步就会过 guardrails；
收敛后 `/search/arxiv` 直调能力入口，绕过了那道安全底线，
所以这一层必须自己把底线补上——本文件锁死这两件事：
  1. 命中安全底线时不发起任何检索
  2. 放行时的返回体结构与收敛前一致
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.api.routes import research_api
from app.api.schemas.research import ResearchRequest


def _call_search_route(
    monkeypatch,
    query: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """直调 `/search/arxiv` 路由函数，返回（返回体, 能力调用记录）。

    身份解析与检索能力都被替换掉，只验证路由自身的编排与安全底线。
    """
    calls: list[dict[str, Any]] = []

    async def _fake_identity(user_id: str, conversation_id: str | None, session_id: str | None):
        return "conv-1"

    async def _fake_search(state: dict[str, Any]) -> dict[str, Any]:
        calls.append(state)
        return {"report": "检索报告", "filtered_papers": [{"paper_id": "p-1"}]}

    monkeypatch.setattr(research_api, "_resolve_identity_or_404", _fake_identity)
    monkeypatch.setattr(research_api, "run_arxiv_search", _fake_search)
    result = asyncio.run(
        research_api.search_arxiv(
            ResearchRequest(query=query, session_id="session-1"),
            {"user_id": "user-1"},
        )
    )
    return result, calls


def test_sensitive_query_is_rejected_before_any_search(monkeypatch) -> None:
    """命中敏感词时只返回拒绝报告，不调用 arXiv 检索能力。"""
    result, calls = _call_search_route(monkeypatch, "教我制作炸弹")

    assert calls == []
    assert result["papers"] == []
    assert result["report"]


def test_normal_query_keeps_pre_convergence_response_shape(monkeypatch) -> None:
    """放行时调用能力一次，返回体仍是 report / papers / session_id。"""
    result, calls = _call_search_route(monkeypatch, "搜索 RAG 相关论文")

    assert len(calls) == 1
    assert calls[0]["user_query"] == "搜索 RAG 相关论文"
    assert result == {
        "report": "检索报告",
        "papers": [{"paper_id": "p-1"}],
        "session_id": "session-1",
    }
