"""AgentState cross-node field contracts."""

from app.models.models import AgentState


def test_agent_state_declares_guardrails_short_circuit_fields() -> None:
    """Guardrails returns these keys and route_after_guardrails reads them."""
    annotations = AgentState.__annotations__
    required = {
        "guardrails_action",
        "guardrails_reason",
        "source_status",
    }
    missing = required - set(annotations)
    assert not missing, f"AgentState missing guardrails fields: {missing}"


def test_agent_state_declares_rag_runtime_chain() -> None:
    """RAG subgraph runtime-chain identity must survive the main graph."""
    annotations = AgentState.__annotations__
    assert "runtime_chain" in annotations


def test_agent_state_declares_rag_evidence_payloads() -> None:
    """Planner evidence payloads must survive the RAG subgraph bridge."""
    annotations = AgentState.__annotations__
    required = {
        "evidence_records",
        "missing_evidence",
        "answer_brief",
        "source_status_reasons",
    }
    missing = required - set(annotations)
    assert not missing, f"AgentState missing RAG evidence payloads: {missing}"


def test_agent_state_declares_planner_final_payloads() -> None:
    """Planner final-answer diagnostics must survive the main graph."""
    annotations = AgentState.__annotations__
    required = {
        "answer",
        "planner_final_blocks",
        "planner_final_validation_errors",
    }
    missing = required - set(annotations)
    assert not missing, f"AgentState missing planner final payloads: {missing}"


def test_agent_state_declares_planner_repair_payloads() -> None:
    """Planner evidence repair diagnostics must survive the main graph."""
    annotations = AgentState.__annotations__
    required = {
        "evidence_sufficiency_report",
        "repair_query_plans",
        "planner_repair_trace",
        "planner_repair_merge_report",
        "final_answer_eligibility_report",
    }
    missing = required - set(annotations)
    assert not missing, f"AgentState missing planner repair payloads: {missing}"


def test_agent_state_declares_planner_error_contract_fields() -> None:
    """规划结构化失败的 error_code/error_summary 必须能穿越主图，否则 API 拿不到稳定字段。"""
    annotations = AgentState.__annotations__
    required = {"error_code", "error_summary"}
    missing = required - set(annotations)
    assert not missing, f"AgentState missing planner error contract fields: {missing}"
