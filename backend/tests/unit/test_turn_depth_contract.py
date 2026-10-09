"""项目科研回合的深度与终态合同测试（单元 1a）。

覆盖点：
- 合同枚举：TurnDepth 三档、CapabilityStatus 四态的取值与 VALID_* 元组一致
- 双 State 声明：五个新字段在 AgentState 与 RAGState 都声明
- 真实图贯通（防假绿）：走真实 LangGraph 状态过滤，断言回合字段未在图入口被丢弃。
  跨图/子图传递的字段只要漏声明一个 State，进图入口就会被静默丢弃，
  手搓 dict 断言 __annotations__ 测不到这个真实断裂点。
"""

from __future__ import annotations

from typing import get_args

import pytest
from langgraph.graph import END, START, StateGraph

from app.models.models import AgentState, RAGState
from app.models.research_contracts import (
    VALID_CAPABILITY_STATUSES,
    VALID_TURN_DEPTHS,
    CapabilityStatus,
    TurnDepth,
)


# 回合字段必须同时活在主图与 RAG 子图的状态通道上，少一边就会在进图入口被静默丢弃。
TURN_STATE_FIELDS = (
    "project_research_context",
    "turn_depth",
    "depth_transitions",
    "turn_outcome",
)


def test_turn_depth_literal_matches_valid_tuple() -> None:
    """三档深度的 Literal 与运行时校验用的 VALID_TURN_DEPTHS 不得漂移。"""
    assert get_args(TurnDepth) == VALID_TURN_DEPTHS
    assert VALID_TURN_DEPTHS == ("minimal", "lookup", "research")


def test_capability_status_keeps_empty_and_failed_separate() -> None:
    """四态合同：empty 与 failed 必须都在且是两个不同状态。

    「项目内没有相关记忆」和「记忆存储不可用」是两件事，归并即违反四态合同。
    """
    assert get_args(CapabilityStatus) == VALID_CAPABILITY_STATUSES
    assert VALID_CAPABILITY_STATUSES == ("success", "empty", "skipped", "failed")


@pytest.mark.parametrize("state_schema", [AgentState, RAGState], ids=["AgentState", "RAGState"])
def test_turn_fields_declared_in_both_states(state_schema) -> None:
    """漏声明任一 State，字段会在进图入口被 LangGraph 静默丢弃。"""
    missing = set(TURN_STATE_FIELDS) - set(state_schema.__annotations__)
    assert not missing, f"{state_schema.__name__} missing turn fields: {missing}"


def _run_single_node_graph(state_schema, input_payload: dict) -> dict:
    # 用真实 State schema 编译单节点图：LangGraph 只把「已声明的通道」写进节点入参，
    # 未声明字段在图入口即被丢弃，所以节点读到什么就是该字段真实能不能穿透。
    seen: dict = {}

    def _capture_node(state):
        for field in TURN_STATE_FIELDS:
            seen[field] = state.get(field)
        return {}

    workflow = StateGraph(state_schema)
    workflow.add_node("capture", _capture_node)
    workflow.add_edge(START, "capture")
    workflow.add_edge("capture", END)
    graph = workflow.compile()
    graph.invoke(input_payload)
    return seen


def _turn_payload() -> dict:
    context = {
        "user_id": "user-1",
        "project_id": "proj-1",
        "authorized_paper_ids": ["paper-1"],
        "turn_focus_paper_ids": [],
        "session_id": "sess-1",
        "conversation_id": "conv-1",
        "query": "这三篇的实验设置有什么差别",
        "history_summary": "",
    }
    transitions = [
        {
            "from_depth": "lookup",
            "to_depth": "research",
            "trigger_observation": "单篇证据无法覆盖跨论文比较",
            "reason_code": "cross_paper_comparison_required",
            "loop_step": 2,
        }
    ]
    outcome = {
        "status": "completed",
        "final_depth": "research",
        "depth_reason_code": "cross_paper_comparison_required",
        "depth_transitions": transitions,
        "capability_results": {"retrieve_paper_evidence": "success", "read_project_memory": "empty"},
        "memory_item_ids": [],
        "evidence_ids": ["ev-1"],
        "missing_evidence": [],
        "loop_steps": 3,
        "termination_reason": "converged",
    }
    return {
        "user_query": "这三篇的实验设置有什么差别",
        "project_research_context": context,
        "turn_depth": "research",
        "depth_transitions": transitions,
        "turn_outcome": outcome,
    }


@pytest.mark.parametrize("state_schema", [AgentState, RAGState], ids=["AgentState", "RAGState"])
def test_turn_fields_survive_real_graph_entry(state_schema) -> None:
    """回合字段经真实图入口后仍可被节点读到，值不被改写。"""
    payload = _turn_payload()
    seen = _run_single_node_graph(state_schema, payload)

    for field in TURN_STATE_FIELDS:
        assert seen[field] == payload[field], f"{state_schema.__name__}.{field} 未穿过图入口"


@pytest.mark.parametrize("state_schema", [AgentState, RAGState], ids=["AgentState", "RAGState"])
def test_depth_transitions_survive_without_reducer_duplication(state_schema) -> None:
    """depth_transitions 无 reducer：节点整体回写即覆盖，不与入口值累加成重复轨迹。

    锁定这条是因为若给该字段挂 operator.add，主图与子图桥接时同一批变更会被追加两次。
    """
    seen: dict = {}
    appended = [
        {
            "from_depth": "minimal",
            "to_depth": "lookup",
            "trigger_observation": "用户追问论文原文事实",
            "reason_code": "paper_fact_requested",
            "loop_step": 1,
        }
    ]

    def _rewrite_node(state):
        seen["entry"] = state.get("depth_transitions")
        return {"depth_transitions": appended}

    workflow = StateGraph(state_schema)
    workflow.add_node("rewrite", _rewrite_node)
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", END)
    graph = workflow.compile()
    final = graph.invoke({"user_query": "q", "depth_transitions": []})

    assert seen["entry"] == []
    assert final["depth_transitions"] == appended
