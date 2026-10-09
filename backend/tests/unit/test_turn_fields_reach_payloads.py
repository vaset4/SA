"""回合级字段的对外投影测试：trace summary 与两条 API 载荷必须原值透传，缺失不得补假值。

作者：xxx
2026-08-30 14:20:00
"""

from __future__ import annotations
from result_delivery_test_utils import offline_memory_callback  # noqa: F401

from test_chat_api import route_ports  # noqa: F401

import pytest

from app.middleware.trace_context import (
    TraceContext,
    build_full_trace,
    build_trace_summary,
)

TURN_LEVEL_KEYS = (
    "turn_depth",
    "depth_transitions",
    "memory_read_result",
    "turn_outcome",
    "memory_writeback",
)


def test_trace_summary_carries_turn_level_fields_with_contracts_intact():
    ctx = TraceContext(request_id="req-turn", session_id="sess-turn")

    summary = build_trace_summary(ctx, _turn_trace())

    assert summary["turn_depth"] == "research"
    assert summary["depth_transitions"] == [
        {
            "from_depth": "lookup",
            "to_depth": "research",
            "trigger_observation": "单篇证据不足以比较",
            "reason_code": "evidence_insufficient",
            "loop_step": 2,
        }
    ]
    # contract 对象保持原状，不被拍平成一堆顶层键
    assert summary["memory_read_result"]["status"] == "success"
    assert summary["memory_read_result"]["items"][0]["memory_id"] == "mem-1"
    assert summary["memory_read_result"]["excluded_counts"] == {"budget_truncated": 2}
    assert summary["turn_outcome"]["capability_results"] == {
        "memory_read": "success",
        "paper_retrieval": "empty",
    }
    assert summary["turn_outcome"]["memory_item_ids"] == ["mem-1"]
    assert summary["turn_outcome"]["loop_steps"] == 4
    assert summary["turn_outcome"]["termination_reason"] == "converged"
    assert summary["memory_writeback"]["candidate_ids"] == ["cand-1"]


def test_trace_summary_omits_turn_level_fields_absent_from_trace():
    ctx = TraceContext(request_id="req-plain", session_id="sess-plain")

    summary = build_trace_summary(ctx, {"retrieval": {}, "generation": {}})

    for key in TURN_LEVEL_KEYS:
        assert key not in summary


def test_full_trace_summary_carries_turn_level_fields():
    ctx = TraceContext(request_id="req-turn", session_id="sess-turn")

    full_trace = build_full_trace(ctx, _turn_trace())

    assert full_trace["summary"]["turn_depth"] == "research"
    assert full_trace["summary"]["turn_outcome"]["status"] == "completed"
    assert full_trace["summary"]["memory_writeback"]["status"] == "success"


def test_full_trace_payload_keeps_turn_fields_out_of_debug_block():
    ctx = TraceContext(request_id="req-turn", session_id="sess-turn")

    full_trace = build_full_trace(ctx, _turn_trace())

    # 超过 OBS_TRACE_MAX_BYTES 时整个 debug 块被替换，回合级字段不能只存在于 debug
    assert set(full_trace["debug"]) == {"full_model_inputs_outputs"}


def test_completed_run_metadata_carries_turn_level_fields():
    import app.api.routes.chat_api as chat_api

    metadata = chat_api._completed_run_metadata(
        {
            "turn_depth": "lookup",
            "depth_transitions": [{"from_depth": "minimal", "to_depth": "lookup"}],
            "memory_writeback": {"status": "empty", "reason_codes": ["no_reusable_judgment"]},
        },
        {"run_id": "run-1", "run_status": "completed"},
    )

    assert metadata["turn_depth"] == "lookup"
    assert metadata["depth_transitions"] == [{"from_depth": "minimal", "to_depth": "lookup"}]
    assert metadata["memory_writeback"] == {"status": "empty", "reason_codes": ["no_reusable_judgment"]}


def test_completed_run_metadata_reads_turn_fields_from_response_when_result_lacks_them():
    import app.api.routes.chat_api as chat_api

    metadata = chat_api._completed_run_metadata(
        {},
        {"turn_depth": "research", "memory_writeback": {"status": "success", "candidate_ids": ["cand-1"]}},
    )

    assert metadata["turn_depth"] == "research"
    assert metadata["memory_writeback"]["candidate_ids"] == ["cand-1"]


def test_completed_run_metadata_leaves_absent_turn_fields_none():
    import app.api.routes.chat_api as chat_api

    metadata = chat_api._completed_run_metadata({}, {})

    assert metadata["turn_depth"] is None
    assert metadata["depth_transitions"] is None
    assert metadata["memory_writeback"] is None
    assert metadata["memory_read_result"] is None


def test_completed_run_metadata_carries_memory_read_result():
    """历史消息重新打开时，回答卡要能重现本轮用了哪几条项目历史判断。"""
    import app.api.routes.chat_api as chat_api

    metadata = chat_api._completed_run_metadata(
        {
            "memory_read_result": {
                "status": "success",
                "items": [{"memory_id": "mem-1", "provenance_label": "倾向"}],
                "excluded_counts": {},
            }
        },
        {},
    )

    assert metadata["memory_read_result"]["status"] == "success"
    assert metadata["memory_read_result"]["items"][0]["provenance_label"] == "倾向"


def test_completed_run_metadata_keeps_memory_read_empty_and_failed_apart():
    """empty 与 failed 不得归并：一个是项目内没有相关判断，一个是记忆存储不可用。"""
    import app.api.routes.chat_api as chat_api

    empty = chat_api._completed_run_metadata({"memory_read_result": {"status": "empty", "items": []}}, {})
    failed = chat_api._completed_run_metadata(
        {"memory_read_result": {"status": "failed", "items": [], "error_kind": "repository_error"}},
        {},
    )

    assert empty["memory_read_result"]["status"] == "empty"
    assert failed["memory_read_result"]["status"] == "failed"
    assert failed["memory_read_result"]["error_kind"] == "repository_error"


def test_cross_paper_done_payload_carries_turn_level_fields():
    from app.api.facades.graph_facade import _cross_paper_done_payload

    payload = _cross_paper_done_payload(
        {
            "report": "答案",
            "turn_depth": "research",
            "depth_transitions": [{"from_depth": "lookup", "to_depth": "research", "loop_step": 2}],
            "memory_writeback": {"status": "success", "candidate_ids": ["cand-1"]},
        },
        "sess-stream",
    )

    assert payload["turn_depth"] == "research"
    assert payload["depth_transitions"][0]["loop_step"] == 2
    assert payload["memory_writeback"]["candidate_ids"] == ["cand-1"]


def test_cross_paper_done_payload_leaves_absent_turn_fields_none():
    from app.api.facades.graph_facade import _cross_paper_done_payload

    payload = _cross_paper_done_payload({"report": "答案"}, "sess-stream")

    assert payload["turn_depth"] is None
    assert payload["depth_transitions"] is None
    assert payload["memory_writeback"] is None
    assert payload["memory_read_result"] is None


def test_cross_paper_done_payload_carries_memory_read_result():
    """流式回答卡与同步同构：记忆使用情况随 done 事件一起下发。"""
    from app.api.facades.graph_facade import _cross_paper_done_payload

    payload = _cross_paper_done_payload(
        {
            "report": "答案",
            "memory_read_result": {
                "status": "success",
                "items": [{"memory_id": "mem-1"}, {"memory_id": "mem-2"}],
                "excluded_counts": {"budget_truncated": 1},
            },
        },
        "sess-stream",
    )

    assert payload["memory_read_result"]["status"] == "success"
    assert len(payload["memory_read_result"]["items"]) == 2


@pytest.mark.asyncio
async def test_cross_paper_sync_response_carries_turn_level_fields(monkeypatch, route_ports) -> None:
    import app.api.routes.chat_api as chat_api
    from app.api.schemas.research import CrossPaperQARequest

    async def fake_resolve_identity_or_404(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        return None

    async def fake_build_project_context(project_id, requested_paper_ids=None, user_id="anonymous"):  # noqa: ARG001
        return {
            "project_id": project_id,
            "resolved_scope": {"paper_ids": ["paper-a"]},
            "filter_expr": 'paper_id == "paper-a"',
            "paper_ids": ["paper-a"],
        }

    async def fake_invoke_graph(query, extra):  # noqa: ARG001
        return (
            {
                "report": "跨论文答案",
                "citations": [],
                "turn_depth": "research",
                "depth_transitions": [{"from_depth": "lookup", "to_depth": "research", "loop_step": 2}],
                "memory_writeback": {"status": "success", "candidate_ids": ["cand-1"]},
                "memory_read_result": {"status": "empty", "items": [], "excluded_counts": {}},
            },
            "session-1",
        )

    monkeypatch.setattr(chat_api, "_resolve_identity_or_404", fake_resolve_identity_or_404)
    monkeypatch.setattr(chat_api, "build_project_context", fake_build_project_context)
    monkeypatch.setattr(chat_api, "invoke_graph", fake_invoke_graph)

    response = await chat_api.cross_paper_qa(
        CrossPaperQARequest(query="比较这两篇的方法", paper_ids=[], project_id="proj-1"),
        user={"user_id": "tester"},
    )

    assert response["turn_depth"] == "research"
    assert response["depth_transitions"][0]["loop_step"] == 2
    assert response["memory_writeback"] == {"status": "success", "candidate_ids": ["cand-1"]}
    # 同步响应也带记忆读取终态，两条路径的回答卡才显示同一份记忆使用情况
    assert response["memory_read_result"]["status"] == "empty"


def _turn_trace() -> dict:
    """构造一条已带回合级字段的序列化 trace，字段形状照 research_contracts 的合同。"""
    return {
        "query": "两篇论文的方法差异",
        "retrieval": {"selected": 2},
        "generation": {"citations_count": 2},
        "turn_depth": "research",
        "depth_transitions": [
            {
                "from_depth": "lookup",
                "to_depth": "research",
                "trigger_observation": "单篇证据不足以比较",
                "reason_code": "evidence_insufficient",
                "loop_step": 2,
            }
        ],
        "memory_read_result": {
            "status": "success",
            "items": [{"memory_id": "mem-1", "version": 3, "maturity": "leaning"}],
            "excluded_counts": {"budget_truncated": 2},
        },
        "turn_outcome": {
            "status": "completed",
            "final_depth": "research",
            "depth_reason_code": "multi_paper_comparison",
            "capability_results": {"memory_read": "success", "paper_retrieval": "empty"},
            "memory_item_ids": ["mem-1"],
            "evidence_ids": ["ev-1", "ev-2"],
            "missing_evidence": [],
            "loop_steps": 4,
            "termination_reason": "converged",
        },
        "memory_writeback": {
            "status": "success",
            "candidate_ids": ["cand-1"],
            "reason_codes": ["validated_claim_with_evidence"],
        },
    }
