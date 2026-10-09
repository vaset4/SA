"""analyze 之后的深度分流路由测试：终态 / 探测回边 / minimal 直答短路 / 正常进 validate。

判定顺序本身是契约（见 route_after_planner_analyze 文档），本文件用对照用例锁住它，
颠倒判定序必红。

作者：LKX
"""

from __future__ import annotations

from app.nodes import rag_node


def test_minimal_without_draft_short_circuits_to_synthesize() -> None:
    """minimal 且没产草案就是真短路：跳过 validate/execute 直接收口。"""
    assert rag_node.route_after_planner_analyze({"turn_depth": "minimal"}) == "planner_synthesize"


def test_minimal_with_draft_still_goes_to_validate() -> None:
    """minimal 但有草案说明这是正常 analyze 产出，计划仍要过校验。"""
    state = {"turn_depth": "minimal", "planner_draft": {"units": []}}

    assert rag_node.route_after_planner_analyze(state) == "planner_validate"


def test_terminate_fail_beats_minimal_short_circuit() -> None:
    """深度评估/记忆读取这类硬失败优先于 minimal，失败态不许被短路吞掉。"""
    state = {"turn_depth": "minimal", "planner_loop_decision": "terminate_fail"}

    assert rag_node.route_after_planner_analyze(state) == "write_back"


def test_lookup_and_research_never_short_circuit() -> None:
    """短路只给 minimal：另两档要检索论文，缺草案也得走原路暴露问题。"""
    for depth in ("lookup", "research"):
        assert rag_node.route_after_planner_analyze({"turn_depth": depth}) == "planner_validate"


def test_missing_or_blank_depth_is_not_minimal() -> None:
    """深度没判出来不等于 minimal，保持默认成功边。"""
    assert rag_node.route_after_planner_analyze({}) == "planner_validate"
    assert rag_node.route_after_planner_analyze({"turn_depth": ""}) == "planner_validate"


def test_probe_decision_returns_to_analyze() -> None:
    """探测轮只取了上下文、没产草案：回 analyze 带着结果重出计划。

    落到默认成功边就会带着没有草案的态进 validate，探测白取还多报一次错。
    """
    state = {"turn_depth": "lookup", "planner_loop_decision": "probe_context"}

    assert rag_node.route_after_planner_analyze(state) == "planner_analyze"


def test_probe_decision_beats_minimal_short_circuit() -> None:
    """探测边必须排在 minimal 之前——这条锁住两者的相对顺序，颠倒必红。

    判成 minimal 却仍要论文证据的回合照样产草案、也照样会去探测，探测后它没有草案。
    先认 minimal 会把这种回合送进 synthesize，那边的直答分支要求 _is_direct_answer_turn
    为真、常规综合分支又要求有研究计划，两边都不成立，最后落成「synthesis input invalid」
    的假失败：探测拿到的上下文白取，整轮报错收场。
    """
    state = {"turn_depth": "minimal", "planner_loop_decision": "probe_context"}

    assert rag_node.route_after_planner_analyze(state) == "planner_analyze"


def test_direct_answer_turn_is_not_stolen_by_probe_edge() -> None:
    """真直答回合认不到探测边：它的决策是空串（minimal 短路那一步清空的）。

    探测边只认决策动作，不能改判成「无草案即探测」——那样每个直答回合都会被
    抢去反复取数，永远开不了工。
    """
    state = {"turn_depth": "minimal", "planner_loop_decision": ""}

    assert rag_node.route_after_planner_analyze(state) == "planner_synthesize"


def test_analyze_conditional_edges_include_all_shortcut_targets() -> None:
    """图装配必须登记全部短路目标，LangGraph 条件边映射里没列的目标会直接报错。"""
    graph = rag_node.create_rag_node().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    analyze_targets = {target for source, target in edges if source == "planner_analyze"}

    assert {"planner_validate", "planner_synthesize", "planner_analyze", "write_back"} <= analyze_targets
