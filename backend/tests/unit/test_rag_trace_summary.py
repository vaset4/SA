import json

from app.middleware.trace_context import (
    ModelCallRecord,
    TraceContext,
    build_full_trace,
    build_trace_summary,
    model_call_to_summary,
)


def test_trace_summary_exposes_retrieved_reranked_selected_and_cited_counts():
    ctx = TraceContext(request_id="req-1", session_id="sess-1")
    rag_trace = {
        "query": "q",
        "mode": "fusion",
        "graph_mode": "hybrid",
        "dialogue_mode": "research",
        "source_status": {"graph": "failed"},
        "retrieval_reasoning": "reason",
        "latency_ms": {"total": 10},
        "retrieval": {
            "hybrid_results": 8,
            "reranked": 5,
            "selected": 3,
            "generation_status": "ok",
        },
        "generation": {
            "citations_count": 3,
            "final_quality_score": 0.8,
        },
    }

    summary = build_trace_summary(ctx, rag_trace)

    assert summary["retrieved_count"] == 8
    assert summary["reranked_count"] == 5
    assert summary["selected_count"] == 3
    assert summary["cited_count"] == 3
    assert summary["evidence_count"] == 3
    assert summary["citation_count"] == 3
    assert summary["evidence_consistent"] is True


def test_trace_summary_flags_selected_citation_mismatch():
    ctx = TraceContext(request_id="req-1", session_id="sess-1")
    summary = build_trace_summary(
        ctx,
        {
            "retrieval": {
                "selected": 2,
                "generation_status": "failed",
                "generation_error_kind": "generation_evidence_mismatch",
                "generation_reason": "selected=2 citations=1",
            },
            "generation": {"citations_count": 1},
        },
    )

    assert summary["evidence_consistent"] is False
    assert summary["generation_status"] == "failed"
    assert summary["generation_error_kind"] == "generation_evidence_mismatch"
    assert summary["generation_reason"] == "selected=2 citations=1"


def test_trace_summary_separates_logical_calls_from_http_attempts():
    ctx = TraceContext(request_id="req-1", session_id="sess-1")
    ctx.model_calls = [
        ModelCallRecord(call_id="call-1", retry_count=0, http_attempt_count=1),
        ModelCallRecord(call_id="call-2", retry_count=3, http_attempt_count=4),
    ]

    summary = build_trace_summary(ctx, {"retrieval": {}, "generation": {}})

    assert summary["model_call_count"] == 2
    assert summary["model_http_attempt_count"] == 5


def test_model_call_summary_includes_retry_attempt_metadata():
    call = ModelCallRecord(
        call_id="call-1",
        retry_count=1,
        http_attempt_count=2,
        retry_events=[{"kind": "http_error", "status_code": 429, "delay_seconds": 12.0}],
    )

    summary = model_call_to_summary(call)

    assert summary["retry_count"] == 1
    assert summary["http_attempt_count"] == 2
    assert summary["retry_events"][0]["status_code"] == 429


def test_trace_summary_exposes_planner_native_metrics():
    ctx = TraceContext(request_id="req-planner", session_id="sess-planner")
    summary = build_trace_summary(ctx, _planner_trace())

    assert summary["runtime_chain"] == "planner"
    assert summary["source_status"]["planner"] == "ok"
    assert summary["source_status"]["planner_final"] == "failed"
    assert summary["source_status_reasons"]["planner_final"] == "provider failed"
    assert summary["planner_units_total"] == 3
    assert summary["planner_units_success"] == 1
    assert summary["planner_units_blocked"] == 1
    assert summary["planner_units_insufficient"] == 1
    assert summary["planner_evidence_count"] == 2
    assert summary["planner_claim_count"] == 2
    assert summary["planner_final_status"] == "failed"
    assert summary["planner_final_accepted_block_count"] == 1
    assert summary["planner_final_rejected_block_count"] == 1
    assert summary["planner_final_block_roles"] == ["recommendation"]
    assert summary["planner_final_error_codes"] == ["unknown_claim_id"]


def test_full_trace_adds_planner_stage_without_rejected_text():
    ctx = TraceContext(request_id="req-planner", session_id="sess-planner")
    full_trace = build_full_trace(ctx, _planner_trace())
    planner_stage = next(stage for stage in full_trace["stages"] if stage["name"] == "planner")
    serialized = json.dumps(planner_stage, ensure_ascii=False)

    assert planner_stage["metrics"]["planner_evidence_count"] == 2
    assert planner_stage["planner_final"]["accepted_blocks"] == [
        {"role": "recommendation", "block_id": "block-final", "content_kind": "inference",
         "claim_ids": ["claim-final"], "evidence_ids": ["ev-1"]}
    ]
    assert planner_stage["planner_final"]["validation_errors"] == [
        {
            "code": "unknown_claim_id",
            "block_role": "rationale",
            "claim_ids": ["missing-claim"],
            "evidence_ids": [],
            "message": "unknown referenced claim id",
        }
    ]
    assert "accepted final answer text" not in serialized
    assert "rejected block text" not in serialized


def _planner_trace():
    return {
        "query": "q",
        "runtime_chain": "planner",
        "source_status": {"planner": "ok", "planner_final": "failed"},
        "source_status_reasons": {"planner_final": "provider failed"},
        "research_plan": {
            "units": [
                {"unit_id": "u1"},
                {"unit_id": "u2"},
                {"unit_id": "u3"},
            ]
        },
        "research_unit_results": [
            {"unit_id": "u1", "status": "evidence_found", "claims": [{"claim_id": "claim-1"}]},
            {"unit_id": "u2", "status": "blocked", "claims": []},
            {"unit_id": "u3", "status": "insufficient_evidence", "claims": [{"claim_id": "claim-3"}]},
        ],
        "evidence_records": [{"evidence_id": "ev-1"}, {"evidence_id": "ev-2"}],
        "planner_final_blocks": [
            {
                "role": "recommendation",
                "block_id": "block-final",
                "content_kind": "inference",
                "text": "accepted final answer text",
                "claim_ids": ["claim-final"],
                "evidence_ids": ["ev-1"],
            }
        ],
        "planner_final_validation_errors": [
            {
                "code": "unknown_claim_id",
                "block_role": "rationale",
                "claim_ids": ["missing-claim"],
                "evidence_ids": [],
                "message": "unknown referenced claim id",
                "rejected_text": "rejected block text",
            }
        ],
    }
