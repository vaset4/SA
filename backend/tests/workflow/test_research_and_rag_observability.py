from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

from app.agents import agent as agent_module
from app.api.facades.research_contract_facade import resolve_session_bridge
import app.services.rag_trace_service as rag_trace_service


def test_invoke_graph_emits_research_completed_observation(monkeypatch) -> None:
    fake_agent_singleton = types.ModuleType("app.api.agent_singleton")

    class FakeAgent:
        async def ainvoke(self, state: dict, config: dict):  # noqa: ARG002
            return {"report": f"answer:{state['user_query']}"}

    async def fake_get_shared_agent():
        return FakeAgent()

    fake_agent_singleton.get_shared_agent = fake_get_shared_agent

    fake_checkpoint_service = types.ModuleType("app.services.checkpoint_service")
    fake_checkpoint_service.build_checkpoint_config = lambda session_id: {"configurable": {"thread_id": session_id}}

    fake_session_mapping = types.ModuleType("app.services.session_mapping_service")
    fake_session_mapping.ensure_session_id = lambda session_id=None: session_id or "session-generated"
    fake_session_mapping.resolve_thread_id = lambda session_id, conversation_id=None, thread_id=None: (
        thread_id or session_id
    )
    fake_session_mapping.build_session_identity = lambda session_id, conversation_id=None: {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "thread_id": session_id,
    }

    monkeypatch.setitem(sys.modules, "app.api.agent_singleton", fake_agent_singleton)
    monkeypatch.setitem(sys.modules, "app.services.checkpoint_service", fake_checkpoint_service)
    monkeypatch.setitem(sys.modules, "app.services.session_mapping_service", fake_session_mapping)
    sys.modules.pop("app.api.facades.graph_facade", None)
    graph_facade = importlib.import_module("app.api.facades.graph_facade")
    monkeypatch.setattr(graph_facade, "get_shared_agent", fake_get_shared_agent)

    observations: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        graph_facade.observability,
        "record_timed_event",
        lambda event_type, node_name, started_at, **kwargs: observations.append(  # noqa: ARG005
            (event_type, node_name, kwargs.get("session_id", ""))
        ),
    )

    result, session_id = asyncio.run(graph_facade.invoke_graph("what is rag", {"session_id": "session-x"}))

    assert result["report"] == "answer:what is rag"
    assert session_id == "session-x"
    assert observations == [("research_completed", "graph_facade", "session-x")]


def _guardrails_reject_agent():
    class FakeAgent:
        async def ainvoke(self, state: dict, config: dict):  # noqa: ARG002
            return {
                "guardrails_action": "reject",
                "guardrails_reason": "retry_exhausted",
                "report": "这次输入没有明确可分析的研究对象",
                "source_status": {
                    "milvus": "skipped",
                    "graph": "skipped",
                    "neo4j": "skipped",
                    "inventory": "skipped",
                },
            }

    return FakeAgent()


def _patch_graph_facade_basics(monkeypatch, graph_facade, observations):
    async def fake_get_shared_agent():
        return _guardrails_reject_agent()

    monkeypatch.setattr(graph_facade, "get_shared_agent", fake_get_shared_agent)
    monkeypatch.setattr(graph_facade, "get_settings", lambda: types.SimpleNamespace(
        graph_timeout_seconds=5,
        deep_research_graph_timeout_seconds=5,
    ))
    monkeypatch.setattr(
        graph_facade,
        "ensure_session_id",
        lambda session_id=None, conversation_id=None: session_id or "session-generated",
    )
    monkeypatch.setattr(graph_facade, "build_session_identity", lambda session_id, conversation_id=None: {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "thread_id": conversation_id or session_id,
    })
    monkeypatch.setattr(graph_facade, "build_checkpoint_config", lambda session_id, conversation_id=None: {
        "configurable": {"thread_id": conversation_id or session_id},
    })
    monkeypatch.setattr(
        graph_facade.observability,
        "record_timed_event",
        lambda event_type, node_name, started_at, **kwargs: observations.append(  # noqa: ARG005
            (event_type, node_name)
        ),
    )


@pytest.mark.asyncio
async def test_invoke_graph_writes_trace_for_guardrails_short_circuit(monkeypatch) -> None:
    from app.api.facades import graph_facade
    from app.services import guardrails_trace_service

    written: list[dict] = []
    observations: list[tuple[str, str]] = []

    async def fake_safe_write_trace(**kwargs):
        written.append(kwargs)

    _patch_graph_facade_basics(monkeypatch, graph_facade, observations)
    monkeypatch.setattr(guardrails_trace_service, "safe_write_trace", fake_safe_write_trace)

    result, session_id = await graph_facade.invoke_graph(
        "19*88=",
        {"intent": "rag_qa", "session_id": "session-guard", "user_id": "user-1"},
    )

    assert session_id == "session-guard"
    assert result["guardrails_action"] == "reject"
    assert result["request_id"]
    assert result["trace_summary"]["model_call_count"] == 0
    assert result["trace_summary"]["mode"] == "guardrails_reject"
    assert written[0]["session_id"] == "session-guard"
    assert written[0]["query"] == "19*88="
    assert written[0]["trace_data"]["summary"]["source_status"]["graph"] == "skipped"
    assert observations == [("research_completed", "graph_facade")]


@pytest.mark.asyncio
async def test_invoke_graph_writes_trace_for_non_rag_qa_intent(monkeypatch) -> None:
    """收敛后所有入口跑同一条主链，非 rag_qa 意图同样要落 trace，并归属到发起用户。

    这是「主链跑了却不落 trace」那条失效的端到端锁：按 preset intent 关门时，
    这里的 trace_summary 与落库调用都不会出现。
    """
    from app.api.facades import graph_facade
    from app.services import guardrails_trace_service

    written: list[dict] = []
    observations: list[tuple[str, str]] = []

    async def fake_safe_write_trace(**kwargs):
        written.append(kwargs)

    _patch_graph_facade_basics(monkeypatch, graph_facade, observations)
    monkeypatch.setattr(guardrails_trace_service, "safe_write_trace", fake_safe_write_trace)

    result, session_id = await graph_facade.invoke_graph(
        "19*88=",
        {"intent": "deep_research", "session_id": "session-research", "user_id": "user-research"},
    )

    assert session_id == "session-research"
    assert result["trace_summary"]["mode"] == "guardrails_reject"
    assert written[0]["session_id"] == "session-research"
    assert written[0]["user_id"] == "user-research"


@pytest.mark.asyncio
async def test_execute_write_back_emits_rag_answer_completed_observation() -> None:
    observations: list[tuple[str, str, str, int]] = []
    persisted: list[str] = []
    updated: list[tuple[int | None, float | None]] = []

    class FakeTrace:
        def __init__(self) -> None:
            self.query = "Explain agent memory"
            self.session_id = "session-rag"
            self.request_id = "req-fake"
            self.mode = "fusion"
            self.dialogue_mode = "research"
            self.graph_mode = "hybrid"
            self.retrieval_reasoning = ""
            self.source_status = {}
            self.outcome = ""
            self.project_id = ""
            self.paper_ids = []
            self.paper_focus_mode = ""
            self.synthesis_status = ""
            self.evidence_layers = {}
            self.retrieval = {"hybrid_results": 0, "reranked": 0, "graded_relevant": 0, "graded_irrelevant": 0}
            self.self_rag_iterations = 0
            self.final_quality_score = None
            self.citations_count = 0
            self.rewritten_queries = []

        def checkpoint(self, name: str) -> None:  # noqa: ARG002
            persisted.append("checkpoint")

        def to_dict(self) -> dict:
            return {
                "request_id": self.request_id,
                "session_id": self.session_id,
                "query": self.query,
                "mode": self.mode,
                "dialogue_mode": self.dialogue_mode,
                "graph_mode": self.graph_mode,
                "retrieval_reasoning": self.retrieval_reasoning,
                "source_status": self.source_status,
                "rewritten_queries": self.rewritten_queries,
                "retrieval": self.retrieval,
                "generation": {"self_rag_iterations": self.self_rag_iterations, "final_quality_score": self.final_quality_score, "citations_count": self.citations_count},
                "latency_ms": {"total": 100},
                "scope": {"project_id": self.project_id, "paper_ids": self.paper_ids, "paper_focus_mode": self.paper_focus_mode},
                "synthesis": {"status": self.synthesis_status, "evidence_layers": self.evidence_layers},
            }

    class FakeObservability:
        async def persist_rag_trace(self, trace) -> None:  # noqa: ANN001
            persisted.append(trace.session_id)

        def record_timed_event(self, event_type, node_name, started_at, **kwargs) -> None:  # noqa: ANN001, ARG002
            observations.append(
                (event_type, node_name, kwargs.get("session_id", ""), kwargs.get("result_count", 0))
            )

    state = {
        "trace": FakeTrace(),
        "citations": [{"paper_id": "p1"}, {"paper_id": "p2"}],
        "evaluation": {"quality_score": 0.82},
        "self_rag_iteration": 1,
        "rag_result": {"sources": []},
        "dialogue_mode": "research",
        "graph_mode": "hybrid",
        "retrieval_reasoning": "项目范围内保留图谱线索。",
        "synthesis_state": {
            "status": "ready",
            "dialogue_mode": "research",
            "project_scope": {"project_id": "project-1", "paper_ids": ["p1", "p2"]},
            "paper_focus": {"mode": "multi_paper"},
            "evidence_layers": {"milvus": "原文证据", "lightrag": "全局线索"},
        },
    }

    result = await rag_trace_service.execute_write_back(
        state,
        paper_db=object(),
        settings=type("S", (), {"push_score_threshold": 0.6})(),
        observability_module=FakeObservability(),
        with_trace_fn=lambda payload, trace: {**payload, "trace": trace},
        update_trace_generation_fn=lambda trace, **kwargs: updated.append(  # noqa: ARG005
            (kwargs.get("iteration"), kwargs.get("quality_score"))
        ),
    )

    assert result["trace"].session_id == "session-rag"
    assert result["synthesis_state"]["status"] == "ready"
    assert result["dialogue_mode"] == "research"
    assert persisted[-1] == "session-rag"
    assert observations == [("rag_answer_completed", "rag_write_back", "session-rag", 2)]
    assert updated == [(1, 0.82)]
    assert result["trace"].dialogue_mode == "research"
    assert result["trace"].project_id == "project-1"
    assert result["trace"].paper_focus_mode == "multi_paper"


def test_router_only_exposes_core_research_intents() -> None:
    assert agent_module.core_route_intents() == {
        "paper_analysis",
        "internal_research",
        "external_research",
        "graph_query",
    }


def test_resolve_session_bridge_keeps_session_conversation_and_thread_roles() -> None:
    bridge = resolve_session_bridge(session_id="session-1", conversation_id="conv-1")

    assert bridge.session_id == "session-1"
    assert bridge.conversation_id == "conv-1"
    assert bridge.thread_id == "session-1"


def test_begin_stream_trace_context_visible_then_reset() -> None:
    """流式端点在迭代协程里建立的 trace ctx 必须在迭代期间可见、迭代后可释放。

    这是 R1 的核心保证：图节点 execute_write_back 在外层 async for 迭代 stream 时
    才执行，必须能通过 get_trace_ctx() 取到本次请求的 ctx；释放后不得泄漏。
    """
    from app.api.facades import graph_facade
    from app.middleware.trace_context import get_trace_ctx, reset_trace_ctx

    assert get_trace_ctx() is None

    token = graph_facade.begin_stream_trace_context(
        extra={"intent": "rag_qa", "request_id": "req-stream", "user_id": "user-1"},
        session_id="session-stream",
    )
    try:
        ctx = get_trace_ctx()
        assert ctx is not None
        assert ctx.request_id == "req-stream"
        assert ctx.session_id == "session-stream"
        assert ctx.user_id == "user-1"
    finally:
        assert token is not None
        reset_trace_ctx(token)

    assert get_trace_ctx() is None


def test_begin_stream_trace_context_builds_for_non_rag_intent() -> None:
    """收敛后所有入口跑同一条主链，非 rag_qa 意图同样要建立 trace ctx。"""
    from app.api.facades import graph_facade
    from app.middleware.trace_context import get_trace_ctx, reset_trace_ctx

    token = graph_facade.begin_stream_trace_context(
        extra={"intent": "free_chat", "request_id": "req-free", "user_id": "user-free"},
        session_id="session-free",
    )
    try:
        ctx = get_trace_ctx()
        assert ctx is not None
        assert ctx.request_id == "req-free"
        assert ctx.session_id == "session-free"
    finally:
        assert token is not None
        reset_trace_ctx(token)

    assert get_trace_ctx() is None


def test_begin_stream_trace_context_does_not_overwrite_existing_ctx() -> None:
    """请求内已有 trace ctx 时不重复建立，避免覆盖外层协程的归属。"""
    from app.api.facades import graph_facade
    from app.middleware.trace_context import (
        TraceContext,
        get_trace_ctx,
        reset_trace_ctx,
        set_trace_ctx,
    )

    outer = TraceContext(request_id="req-outer", session_id="session-outer", user_id="user-outer")
    outer_token = set_trace_ctx(outer)
    try:
        inner_token = graph_facade.begin_stream_trace_context(
            extra={"intent": "free_chat", "request_id": "req-inner", "user_id": "user-inner"},
            session_id="session-inner",
        )

        assert inner_token is None
        assert get_trace_ctx() is outer
    finally:
        reset_trace_ctx(outer_token)


def test_trace_ctx_user_id_comes_from_extra() -> None:
    """user_id 必须从 extra 带入 trace，否则按用户查不到自己的 trace。"""
    from app.api.facades import graph_facade
    from app.middleware.trace_context import get_trace_ctx, reset_trace_ctx

    token = graph_facade.begin_stream_trace_context(
        extra={"intent": "deep_research", "user_id": "user-attributed"},
        session_id="session-attr",
    )
    try:
        ctx = get_trace_ctx()
        assert ctx is not None
        assert ctx.user_id == "user-attributed"
    finally:
        assert token is not None
        reset_trace_ctx(token)
