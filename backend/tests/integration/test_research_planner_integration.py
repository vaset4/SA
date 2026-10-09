"""Research planner RAG integration tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from app.api.routes import chat_api
from app.agents import agent as agent_module
from app.agents.agent import AgentState
from app.models.models import RAGState
from app.models.research_planner_contracts import ResearchUnit, ResearchUnitResult, ResearchClaim
from app.services.internal_research.research_planner_claim_provider import build_claim_prompt
from app.services.internal_research.research_planner_plan_prompt import build_planner_prompt
from research_planner_test_utils import make_unit
from o2_evidence_test_utils import bind_publication, scope
from app.nodes import rag_node
from app.services.internal_research import research_planner_pipeline


def test_coarse_intent_focus_preserved() -> None:
    state = {
        "intent": "compare",
        "focus": {"turn_focus": ["p1", "p2"]},
    }
    payload = rag_node._planner_context_metadata(state)

    assert payload["intent"] == "compare"
    assert payload["focus"]["turn_focus"] == ["p1", "p2"]


def test_rag_init_goes_straight_to_planner_analyze() -> None:
    """意图与焦点并入 Planner 后，rag_init 直连 planner_analyze，不存在前置判断节点。"""
    graph = rag_node.create_rag_node().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    rag_init_targets = {target for source, target in edges if source == "rag_init"}

    assert rag_init_targets == {"planner_analyze"}
    assert "intent_focus" not in set(graph.nodes)
    assert not hasattr(rag_node, "intent_focus_node")
    assert not hasattr(rag_node, "route_after_intent_focus")
    # ReAct 整链已摘除：节点、路由、工具节点与未注册的第二条 analyze 路径都不复存在
    for removed in (
        "react_decision_node",
        "synthesize_node",
        "compress_node",
        "analyze_node",
        "evaluate_node",
        "refine_node",
        "route_react",
        "route_compress",
        "route_after_synthesize",
        "route_after_evaluate",
        "tools_node",
        "research_planner_node",
    ):
        assert not hasattr(rag_node, removed), f"{removed} 应随 ReAct 整链一起摘除"


def test_planner_graph_has_no_fallback_route() -> None:
    edges = {(edge.source, edge.target) for edge in rag_node.create_rag_node().get_graph().edges}

    assert ("planner_analyze", "planner_validate") in edges
    assert ("planner_validate", "planner_execute") in edges
    assert ("planner_execute", "planner_synthesize") in edges
    assert ("planner_synthesize", "write_back") in edges
    assert ("planner_synthesize", "analyze") not in edges
    assert ("planner_execute", "react_decision") not in edges
    assert ("planner_execute", "graph_prefetch") not in edges
    assert ("planner_execute", "write_back") in edges


def test_planner_synthesis_success_writes_back_without_generic_analyze() -> None:
    assert rag_node.route_after_planner_synthesize({"source_status": {"planner": "ok"}}) == "write_back"
    assert rag_node.route_after_planner_synthesize({"source_status": {"planner": "partial"}}) == "write_back"


def test_default_planner_adapters_exclude_structure_hint() -> None:
    adapters = rag_node._planner_adapters({})

    assert set(adapters) == {"local_direct_text", "local_metadata"}
    assert "local_structure_hint" not in adapters


def test_plan_driven_optional_structure_adapter() -> None:
    base = {"local_direct_text": object()}
    optional = {"local_structure_hint": object()}
    direct_plan = _plan_with_sources(["local_direct_text"])
    structure_plan = _plan_with_sources(["local_direct_text", "local_structure_hint"])

    assert research_planner_pipeline._adapters_for_plan(direct_plan, base, optional) == base
    selected = research_planner_pipeline._adapters_for_plan(structure_plan, base, optional)
    assert set(selected) == {"local_direct_text", "local_structure_hint"}


def test_planner_prompt_requires_json_only() -> None:
    prompt = build_planner_prompt({
        "user_query": "compare traffic forecasting methods",
        "project_scope": {"paper_ids": ["paper-a", "paper-b"]},
    })

    assert "Return exactly one JSON object" in prompt
    assert "Do not write Markdown" in prompt
    assert '"question"' in prompt
    assert '"units"' in prompt
    assert '"evidence_policy"' in prompt
    assert '"output_role": "evidence"' in prompt
    assert "Do not put output_role inside expected_output" in prompt
    assert "jointly cover every required paper" in prompt


def test_direct_claim_prompt_requires_json_only() -> None:
    """无参数；验证直接事实提示只允许 JSON，并明确原文及支持片段合同。"""
    unit = ResearchUnit.model_validate(make_unit("u1", allowed_claims=["paper_supported"]))
    prompt = build_claim_prompt(
        unit,
        [{"evidence_id": "ev-1", "layer": "direct_text", "paper_id": "p1", "excerpt": "method text"}],
        [],
    )

    assert "Return one JSON object validating ResearchClaimDraftBatch" in prompt
    assert "No prose outside JSON" in prompt
    assert "Evidence records:\n" in prompt
    assert "text: method text" in prompt
    assert "supporting_evidence_ids AND supporting_spans" in prompt
    assert "evidence_id, evidence_version, field, start, end and quote" in prompt
    assert "Visible ranges:" in prompt


def test_inference_claim_prompt_uses_dependency_claims() -> None:
    """无参数；验证推断明确选择上游主张，来源单元由服务端派生。"""
    unit = ResearchUnit.model_validate(make_unit(
        "u2", source_type="inference_from_units", allowed_claims=["evidence_grounded_inference"],
        output_role="final_recommendation", deps=[{"unit_id": "u1", "required_product": "claims"}],
    ))
    claim = ResearchClaim(
        claim_id="u1.c1",
        claim_type="paper_supported",
        validation_status="validated",
        supporting_evidence_ids=["ev-1"],
        source_unit_ids=[],
        reasoning_summary="",
        text="Paper A supports adaptive temporal modeling.",
    )
    dependency = ResearchUnitResult(unit_id="u1", status="evidence_found", claims=[claim])

    prompt = build_claim_prompt(unit, [], [dependency])

    assert "Dependency claims:\n" in prompt
    assert "source_claim_ids must select the specific eligible claims from direct dependencies" in prompt
    assert "source_unit_ids must stay empty: ownership is derived by the server" in prompt
    assert "Never inherit a whole unit" in prompt
    assert '"claim_id": "u1.c1"' in prompt
    assert '"supporting_evidence_ids": ["ev-1"]' in prompt
    assert claim.text in prompt


@pytest.mark.asyncio
async def test_lightrag_expansion_does_not_enter_planner_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_queries: list[str] = []
    unit = ResearchUnit.model_validate(make_unit(
        "u1", operation_family="comparison", goal="Compare model relationships", action="compare relationship evidence",
    ))
    expected_queries = ["attention mechanism", "adaptive adjacency", "temporal attention"]
    rows = [{"chunk_id": f"v3:{'b' * 32}:{index}", "parent_id": f"p{index}", "paper_id": "paper-a",
             "text": query, "full_text": query, "preview": query, "score": 0.9}
            for index, query in enumerate(expected_queries)]
    bind_publication(monkeypatch, rows)

    def fake_search(_store: object, query: str, *_args: object) -> list[dict[str, object]]:
        captured_queries.append(query)
        index = len(captured_queries)
        return [rows[index - 1]]

    monkeypatch.setattr(rag_node, "get_graph_provider", lambda: _LightRAGProvider())
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: object())
    monkeypatch.setattr(rag_node, "_search_chunks", fake_search)
    monkeypatch.setattr(rag_node.settings, "rag_top_k", 2)
    monkeypatch.setattr(rag_node.settings, "rag_planner_lightrag_locator_enabled", True)

    records = await rag_node._planner_direct_text_records(
        {"user_query": "compare traffic methods", "intent": "compare", "project_scope": scope(),
         "structured_query": {"library_id": "library-a"}},
        unit,
        None,
    )

    assert captured_queries == ["attention mechanism", "adaptive adjacency", "temporal attention"]
    assert {record["source"] for record in records.records} == {"milvus"}
    assert {record["layer"] for record in records.records} == {"direct_text"}


@pytest.mark.asyncio
async def test_lightrag_expansion_provider_missing_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    unit = SimpleNamespace(
        evidence_policy=SimpleNamespace(locator_queries=["attention mechanism"]),
        operation_family="comparison",
        goal="Compare model relationships",
        research_action="compare relationship evidence",
    )

    monkeypatch.setattr(rag_node, "get_graph_provider", lambda: None)
    monkeypatch.setattr(rag_node.settings, "rag_planner_lightrag_locator_enabled", True)

    with pytest.raises(ValueError, match="graph provider is unavailable"):
        await rag_node._planner_locator_queries({"intent": "compare"}, unit)


def test_planner_payload_state_fields_declared() -> None:
    required = {
        "research_plan",
        "research_unit_results",
        "coverage_review",
        "planner_trace",
        "planner_final_blocks",
        "planner_final_validation_errors",
    }
    assert "planner_draft" in set(get_type_hints(RAGState, include_extras=True))
    assert required <= set(get_type_hints(RAGState, include_extras=True))
    assert required <= set(get_type_hints(AgentState, include_extras=True))


def test_planner_synthesize_node_is_coroutine() -> None:
    assert asyncio.iscoroutinefunction(rag_node.planner_synthesize_node)


@pytest.mark.asyncio
async def test_agent_bridge_preserves_planner_evidence_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_rag_graph_stream(_graph: object, _state: dict) -> dict:
        return {
            "answer": "planner answer",
            "runtime_chain": "planner",
            "source_status": {"planner": "partial"},
            "citations": [{"evidence_id": "ev-1"}],
            "chunk_citations": [{"evidence_id": "ev-1"}],
            "paper_sources": [{"paper_id": "paper-a"}],
            "evidence_records": [{"evidence_id": "ev-1", "layer": "direct_text"}],
            "missing_evidence": [{"paper_id": "paper-b", "reason": "no claim"}],
            "answer_brief": {"intent": "compare"},
            "planner_final_blocks": [{"role": "recommendation"}],
            "planner_final_validation_errors": [{"code": "malformed_provider_output"}],
            "source_status_reasons": {"planner_final": "provider failed"},
        }

    monkeypatch.setattr(agent_module, "create_rag_node", lambda: object())
    monkeypatch.setattr(agent_module, "_run_rag_graph_stream", fake_run_rag_graph_stream)

    result = await agent_module.rag_node({"user_query": "q", "session_id": "s"})

    assert result["runtime_chain"] == "planner"
    assert result["evidence_records"][0]["evidence_id"] == "ev-1"
    assert result["missing_evidence"][0]["paper_id"] == "paper-b"
    assert result["answer_brief"]["intent"] == "compare"
    assert result["planner_final_blocks"][0]["role"] == "recommendation"
    assert result["planner_final_validation_errors"][0]["code"] == "malformed_provider_output"
    assert result["source_status_reasons"]["planner_final"] == "provider failed"


def test_planner_scope_uses_resolved_focus_not_structured_query() -> None:
    allowed = research_planner_pipeline._allowed_paper_ids(
        {
            "focus": {"project_scope": ["paper-a"]},
            "structured_query": {"paper_ids": ["paper-b"]},
        }
    )

    assert allowed == frozenset({"paper-a"})


def test_planner_scope_ignores_structured_query_without_resolved_scope() -> None:
    allowed = research_planner_pipeline._allowed_paper_ids(
        {"structured_query": {"paper_ids": ["paper-b"]}}
    )

    assert allowed is None


def test_completed_run_metadata_preserves_planner_payloads() -> None:
    result = {
        "request_id": "req-1",
        "runtime_chain": "planner",
        "source_status": {"planner": "ok"},
        "evidence_records": [{"evidence_id": "ev-1", "layer": "direct_text", "usable_as_citation": True}],
    }
    response = {
        "research_plan": {"units": [{"unit_id": "u1"}]},
        "research_unit_results": [{"unit_id": "u1", "status": "evidence_found"}],
        "coverage_review": {"covered_facet_ids": ["f1"]},
        "planner_trace": {"analyzer_plan_type": "single_unit"},
        "planner_final_blocks": [{"role": "recommendation"}],
        "planner_final_validation_errors": [{"code": "unknown_claim_id"}],
    }

    metadata = chat_api._completed_run_metadata(result, response)

    assert metadata["runtime_chain"] == "planner"
    assert metadata["evidence_records"][0]["evidence_id"] == "ev-1"
    assert metadata["research_plan"]["units"][0]["unit_id"] == "u1"
    assert metadata["research_unit_results"][0]["status"] == "evidence_found"
    assert metadata["coverage_review"]["covered_facet_ids"] == ["f1"]
    assert metadata["planner_trace"]["analyzer_plan_type"] == "single_unit"
    assert metadata["planner_final_blocks"][0]["role"] == "recommendation"
    assert metadata["planner_final_validation_errors"][0]["code"] == "unknown_claim_id"


class _LightRAGProvider:
    async def collect_library_graph_hint(self, query: str, mode: str, top_k: int, *, library_id: str | None = None) -> str:
        assert query == "compare traffic methods"
        assert mode == "hybrid"
        assert top_k == 2
        assert library_id == "library-a"
        return "adaptive adjacency\ntemporal attention"


def _plan_with_sources(source_types: list[str]) -> SimpleNamespace:
    units = [SimpleNamespace(evidence_policy=SimpleNamespace(source_type=source)) for source in source_types]
    return SimpleNamespace(units=units)
