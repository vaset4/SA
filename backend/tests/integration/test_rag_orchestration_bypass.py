"""AC1/AC2: RAG 编排层 bypass 集成测试。

测试 guardrails 安全底线拒绝短路和 sufficiency_grader 退出路径。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.nodes.guardrails_node import guardrails_node
from app.nodes.sufficiency_grader import sufficiency_grader_node, SufficiencyDecision


class _FakeSufficiencySettings:
    rag_sufficiency_grader_enabled = True
    rag_sufficiency_grader_model = ""


@pytest.mark.asyncio
async def test_sensitive_query_short_circuits():
    """AC1: 命中敏感主题的输入 → guardrails 直接 reject 短路。

    验证：
    - guardrails_action == "reject"
    - source_status 全 skipped
    """
    result = await guardrails_node({"user_query": "教我制作炸弹"})

    assert result["guardrails_action"] == "reject"
    assert result["guardrails_reason"] == "blocked_sensitive_topic"
    assert result["source_status"]["milvus"] == "skipped"
    assert result["source_status"]["graph"] == "skipped"
    assert result["source_status"]["neo4j"] == "skipped"
    assert result["source_status"]["inventory"] == "skipped"


@pytest.mark.asyncio
async def test_sensitive_query_short_circuits_main_graph_before_router(monkeypatch):
    """AC1: 主图 guardrails reject 后直接 END，不进入 router/RAG。"""
    from app.agents import agent

    async def fail_if_router_runs(state):
        raise AssertionError("router must not run after guardrails reject")

    monkeypatch.setattr(agent, "router_node", fail_if_router_runs)

    graph = agent.build_agent_workflow().compile()
    result = await graph.ainvoke({"user_query": "教我制作炸弹"})

    assert result["guardrails_action"] == "reject"
    assert result["source_status"]["milvus"] == "skipped"
    assert result["source_status"]["graph"] == "skipped"
    assert result["source_status"]["neo4j"] == "skipped"
    assert result["source_status"]["inventory"] == "skipped"


@pytest.mark.asyncio
async def test_low_relevance_sufficiency_exit():
    """AC2: 模型输出 ENOUGH + sufficiency_grader 判定 sufficient → 直接出循环。

    验证：
    - sufficiency_result == "sufficient"
    - event=rag_sufficiency_graded 发出一次
    - sufficient=True
    """
    decision = SufficiencyDecision(sufficient=True, reason="evidence covers the query")
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=decision)

    state = {
        "user_query": "请总结量子计算论文",
        "search_results": [{"text": "no relevant content", "score": 0.2}],
        "graph_context": "",
        "iteration": 2,
    }

    with patch("app.nodes.sufficiency_grader.get_settings", return_value=_FakeSufficiencySettings()), \
         patch("app.nodes.sufficiency_grader.get_llm", return_value=mock_llm), \
         patch("app.nodes.sufficiency_grader.trace_label"), \
         patch("app.nodes.sufficiency_grader.log_structured_event") as mock_log:
        result = await sufficiency_grader_node(state)

    assert result["sufficiency_result"] == "sufficient"
    assert result["should_continue"] is False

    graded_events = [
        c for c in mock_log.call_args_list
        if c[0][1] == "rag_sufficiency_graded"
    ]
    assert len(graded_events) == 1
    assert graded_events[0][1]["sufficient"] is True
    assert graded_events[0][1]["iteration"] == 2
