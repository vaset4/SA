"""U6 主链收敛：顶层图拓扑与 router 归一行为断言。

收敛后顶层图只剩 guardrails → router → rag_qa 一条主链，
router 不再分派执行分支，只把意图归一成产品层主语义。
"""

from __future__ import annotations

import asyncio

from app.agents import agent as agent_module

_GRAPH_ENDPOINT_NODES = {"__start__", "__end__"}


def _main_chain_drawable():
    """编译主图并返回可绘制的图结构，供节点与边断言复用。"""
    return agent_module.build_agent_workflow().compile().get_graph()


def test_main_chain_has_exactly_guardrails_router_rag_qa() -> None:
    drawable = _main_chain_drawable()

    assert set(drawable.nodes) - _GRAPH_ENDPOINT_NODES == {"guardrails", "router", "rag_qa"}


def test_main_chain_edges_are_linear_with_guardrails_reject_shortcut() -> None:
    drawable = _main_chain_drawable()

    edges = {(edge.source, edge.target) for edge in drawable.edges}
    assert edges == {
        ("__start__", "guardrails"),
        ("guardrails", "router"),
        ("guardrails", "__end__"),
        ("router", "rag_qa"),
        ("rag_qa", "__end__"),
    }


def test_guardrails_reject_branch_maps_to_end() -> None:
    workflow = agent_module.build_agent_workflow()

    branch = workflow.branches["guardrails"]["route_after_guardrails"]
    assert branch.ends == {"router": "router", "reject": "__end__"}
    assert agent_module.route_after_guardrails({"guardrails_action": "reject"}) == "reject"
    assert agent_module.route_after_guardrails({"guardrails_action": "accept"}) == "router"


def test_router_normalizes_preset_intent_to_core_intent() -> None:
    result = asyncio.run(agent_module.router_node({"user_query": "随便问问", "intent": "citation"}))

    assert result == {"intent": "citation", "core_intent": "graph_query"}


def test_router_falls_back_to_scope_fact_without_preset_intent() -> None:
    scoped = asyncio.run(
        agent_module.router_node(
            {
                "user_query": "这篇论文说了什么",
                "structured_query": {"project_id": "project-1", "paper_ids": []},
            }
        )
    )
    unscoped = asyncio.run(agent_module.router_node({"user_query": "介绍一下 RAG"}))

    assert scoped == {"intent": "rag_qa", "core_intent": "internal_research"}
    assert unscoped == {"intent": "deep_research", "core_intent": "external_research"}