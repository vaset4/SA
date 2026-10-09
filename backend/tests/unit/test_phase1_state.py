"""Phase 1 — RAGState 新字段契约测试。

对应任务 05-23-cross-paper-qa-retrieval-fix 的 Slice B。

只验证 7 个新字段的类型注解 + 计数字段的 reducer 元数据，
不验证业务逻辑（业务逻辑在 Slice F/F.5/H 测试覆盖）。
"""

from __future__ import annotations

import operator
import typing
from typing import get_type_hints

from app.models.models import RAGState


_REQUIRED_PHASE1_FIELDS: set[str] = {
    "unindexed_paper_ids",
    "vector_call_count",
    "graph_call_count",
    "graph_hint_section",
    "graph_citation_intersection_count",
    "source_status_reasons",
}

_COUNTER_FIELDS_WITH_ADD_REDUCER: set[str] = {
    "vector_call_count",
    "graph_call_count",
}

_REQUIRED_ROUND2_PHASE_B_FIELDS: set[str] = {
    "selected_evidence",
    "evidence_status",
    "evidence_selection_stats",
    "retrieval_channels",
    "retrieval_effective_query",
    "retrieval_effective_paper_ids",
    "retrieval_effective_sections",
    "retrieval_used_route_hints",
    "error_kind",
}

_REQUIRED_PLANNER_REPAIR_FIELDS: set[str] = {
    "evidence_sufficiency_report",
    "repair_query_plans",
    "planner_repair_trace",
    "planner_repair_merge_report",
    "final_answer_eligibility_report",
}

_REQUIRED_PLANNER_ERROR_FIELDS: set[str] = {
    "error_code",
    "error_summary",
}

# 规划 ReAct 循环跨 analyze/validate/execute 回边流转的字段，未声明会被 LangGraph 静默丢弃
_REQUIRED_PLANNER_LOOP_FIELDS: set[str] = {
    "planner_budget",
    "planner_observation",
    "planner_recovery_trace",
    "planner_loop_decision",
}


def test_all_phase1_fields_present_in_annotations() -> None:
    annotations = RAGState.__annotations__
    missing = _REQUIRED_PHASE1_FIELDS - set(annotations.keys())
    assert not missing, f"RAGState missing Phase 1 fields: {missing}"


def test_round2_phase_b_fields_present_in_annotations() -> None:
    annotations = RAGState.__annotations__
    missing = _REQUIRED_ROUND2_PHASE_B_FIELDS - set(annotations.keys())
    assert not missing, f"RAGState missing Round 2 Phase B fields: {missing}"


def test_planner_repair_fields_present_in_annotations() -> None:
    annotations = RAGState.__annotations__
    missing = _REQUIRED_PLANNER_REPAIR_FIELDS - set(annotations.keys())
    assert not missing, f"RAGState missing planner repair fields: {missing}"


def test_planner_error_fields_present_in_annotations() -> None:
    """规划失败 error_code/error_summary 必须声明在 RAGState，否则子图返回会被静默丢弃。"""
    annotations = RAGState.__annotations__
    missing = _REQUIRED_PLANNER_ERROR_FIELDS - set(annotations.keys())
    assert not missing, f"RAGState missing planner error fields: {missing}"


def test_counter_fields_use_operator_add_reducer() -> None:
    """计数字段必须用 Annotated[int, operator.add]，否则并发更新会丢累加。

    见 design.md §2 + §9 风险表"vector_call_count / graph_call_count 在并发
    tool_call 场景下计数不准"——当前是顺序，但 reducer 必须先到位。
    """
    hints = get_type_hints(RAGState, include_extras=True)
    for field in _COUNTER_FIELDS_WITH_ADD_REDUCER:
        hint = hints[field]
        origin = typing.get_origin(hint)
        assert origin is typing.Annotated or hasattr(hint, "__metadata__"), (
            f"{field!r} must be Annotated[int, operator.add]; got {hint!r}"
        )
        metadata = getattr(hint, "__metadata__", ())
        assert operator.add in metadata, (
            f"{field!r} Annotated metadata must contain operator.add; got {metadata!r}"
        )


def test_non_counter_fields_are_plain_types() -> None:
    """非计数字段不带 reducer，避免误伤后续覆盖语义。"""
    hints = get_type_hints(RAGState, include_extras=True)
    plain_fields = _REQUIRED_PHASE1_FIELDS - _COUNTER_FIELDS_WITH_ADD_REDUCER
    for field in plain_fields:
        hint = hints[field]
        metadata = getattr(hint, "__metadata__", ())
        assert operator.add not in metadata, (
            f"{field!r} should NOT use operator.add reducer; got metadata {metadata!r}"
        )


def test_state_dict_accepts_phase1_keys_at_runtime() -> None:
    """TypedDict total=False 下，构造时不强制字段；但运行时赋值/读取必须畅通。"""
    state: RAGState = {  # type: ignore[typeddict-item]
        "unindexed_paper_ids": ["upload_b49aabbf"],
        "vector_call_count": 2,
        "graph_call_count": 1,
        "graph_hint_section": "hint",
        "graph_citation_intersection_count": 1,
        "source_status_reasons": {"graph": "evidence_not_cited"},
    }
    assert state["unindexed_paper_ids"] == ["upload_b49aabbf"]
    assert state["vector_call_count"] == 2
    assert state["graph_call_count"] == 1
    assert state["graph_hint_section"] == "hint"
    assert state["graph_citation_intersection_count"] == 1
    assert state["source_status_reasons"]["graph"] == "evidence_not_cited"
