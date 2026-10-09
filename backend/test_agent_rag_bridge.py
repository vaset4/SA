from __future__ import annotations

import asyncio

import app.agents.agent as main_agent


def test_main_rag_node_passes_scope_and_runtime_fields_to_subgraph(monkeypatch) -> None:
    sentinel_graph = object()
    captured: dict[str, object] = {}

    async def fake_run_rag_graph_stream(rag_graph, rag_state):  # noqa: ANN001
        captured["rag_graph"] = rag_graph
        captured["rag_state"] = dict(rag_state)
        return {"answer": "ok", "citations": []}

    monkeypatch.setattr(main_agent, "create_rag_node", lambda: sentinel_graph)
    monkeypatch.setattr(main_agent, "_run_rag_graph_stream", fake_run_rag_graph_stream)

    asyncio.run(
        main_agent.rag_node(
            {
                "user_query": "stgcn解决了什么问题",
                "filter_expr": 'paper_id == "paper-1"',
                "structured_query": {
                    "paper_ids": ["paper-1"],
                    "resolved_scope": {"paper_ids": ["paper-1"]},
                },
                "session_id": "session-1",
                "conversation_id": "conv-1",
                "context_summary": "existing summary",
                "chat_history": [{"role": "user", "content": "earlier"}],
            }
        )
    )

    assert captured["rag_graph"] is sentinel_graph
    assert captured["rag_state"] == {
        "user_query": "stgcn解决了什么问题",
        "iteration": 0,
        "should_continue": True,
        "filter_expr": 'paper_id == "paper-1"',
        "structured_query": {
            "paper_ids": ["paper-1"],
            "resolved_scope": {"paper_ids": ["paper-1"]},
        },
        "session_id": "session-1",
        "conversation_id": "conv-1",
        "context_summary": "existing summary",
        "chat_history": [{"role": "user", "content": "earlier"}],
    }


def test_main_rag_node_bridges_answer_citations_and_retrieval_metadata(monkeypatch) -> None:
    async def fake_run_rag_graph_stream(rag_graph, rag_state):  # noqa: ANN001, ARG001
        return {
            "answer": "STGCN 主要解决交通预测中的时空依赖建模问题。",
            "citations": [{"paper_id": "paper-1", "chunk_id": "chunk-1"}],
            "research_scope": {"mode": "single_paper", "summary": "来自子图整理。"},
            "research_objects": [{"key": "methods", "label": "方法", "items": [{"label": "STGCN-runtime"}]}],
            "note_actions": [{"key": "runtime-copy", "type": "copy_note", "label": "复制运行时整理", "note": "runtime"}],
            "synthesis_state": {"status": "ready", "paper_focus": {"mode": "single_paper"}},
            "paper_sources": [{"paper_id": "paper-1", "title": "STGCN"}],
            "chunk_citations": [{"paper_id": "paper-1", "chunk_id": "chunk-1"}],
            "graph_evidence": [{"type": "method", "value": "STGCN"}],
            "evidence_summary": "1 条 chunk 证据，0 条图证据。",
            "answer_confidence": "high",
            "search_results": [{"paper_id": "paper-1", "chunk_id": "chunk-1"}],
            "graph_context": "",
            "source_status": {"milvus": "ok", "graph": "skipped"},
            "retrieval_mode": "milvus",
            "retrieval_reasoning": "存在 paper 范围约束，已强制回退到 Milvus",
            "graph_mode": "hybrid",
            "failure_reason": "",
            "rag_result": {"chunks": ["preview"], "sources": ["paper-1"], "scores": [0.9], "is_sufficient": True},
        }

    monkeypatch.setattr(main_agent, "create_rag_node", lambda: object())
    monkeypatch.setattr(main_agent, "_run_rag_graph_stream", fake_run_rag_graph_stream)
    contract_calls = {"count": 0}

    def _unexpected_contract(**kwargs):  # noqa: ANN003
        contract_calls["count"] += 1
        return {
            "research_scope": {"mode": "fallback"},
            "research_objects": [],
            "note_actions": [],
            "synthesis_state": {},
            "dialogue_mode": "research",
        }

    monkeypatch.setattr(main_agent, "build_internal_research_contract", _unexpected_contract)

    result = asyncio.run(main_agent.rag_node({"user_query": "stgcn解决了什么问题"}))

    assert result["answer"] == "STGCN 主要解决交通预测中的时空依赖建模问题。"
    assert result["report"] == "STGCN 主要解决交通预测中的时空依赖建模问题。"
    assert result["citations"] == [{"paper_id": "paper-1", "chunk_id": "chunk-1"}]
    assert result["paper_sources"] == [{"paper_id": "paper-1", "title": "STGCN"}]
    assert result["chunk_citations"] == [{"paper_id": "paper-1", "chunk_id": "chunk-1"}]
    assert result["graph_evidence"] == [{"type": "method", "value": "STGCN"}]
    assert result["evidence_summary"] == "1 条 chunk 证据，0 条图证据。"
    assert result["answer_confidence"] == "high"
    assert result["research_scope"] == {"mode": "single_paper", "summary": "来自子图整理。"}
    assert result["research_objects"] == [{"key": "methods", "label": "方法", "items": [{"label": "STGCN-runtime"}]}]
    assert result["note_actions"] == [{"key": "runtime-copy", "type": "copy_note", "label": "复制运行时整理", "note": "runtime"}]
    assert result["source_status"] == {"milvus": "ok", "graph": "skipped"}
    assert result["retrieval_mode"] == "milvus"
    assert result["retrieval_reasoning"] == "存在 paper 范围约束，已强制回退到 Milvus"
    assert result["graph_mode"] == "hybrid"
    assert result["rag_result"] == {
        "chunks": ["preview"],
        "sources": ["paper-1"],
        "scores": [0.9],
        "is_sufficient": True,
    }
    assert contract_calls["count"] == 0
