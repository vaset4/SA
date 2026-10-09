from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

from langgraph.graph import END, START, StateGraph

os.environ["SSLKEYLOGFILE"] = ""

import app.agents.agent as agent_module
import app.api.facades.graph_facade as graph_facade_module
from app.models.models import AgentState


async def _event_stream(events):
    for event in events:
        yield event


async def _collect_sse(generator) -> list[dict]:
    payloads: list[dict] = []
    async for chunk in generator:
        for frame in chunk.split("\n\n"):
            if not frame.strip():
                continue
            for line in frame.splitlines():
                if line.startswith("data: "):
                    payloads.append(json.loads(line[6:]))
    return payloads


def _agent_state(query: str) -> AgentState:
    return {
        "user_query": query,
        "intent": None,
        "structured_query": None,
        "research_plan": [],
        "completed_tasks": [],
        "report": None,
        "papers": [],
        "filtered_papers": [],
        "rag_result": None,
        "chat_history": [],
        "session_id": "session-rag",
        "context_summary": None,
    }


def test_stream_sse_emits_route_and_rag_stage_events():
    generator = graph_facade_module.stream_research_sse(
        _event_stream(
            [
                {"event": "on_chain_end", "name": "router", "data": {"output": {"intent": "rag_qa"}}},
                {
                    "event": "on_chain_stream",
                    "name": "LangGraph",
                    "data": {"chunk": ("custom", {"stage": "tool", "node": "tools", "tool_name": "read_chunk"})},
                },
                {
                    "event": "on_chain_stream",
                    "name": "LangGraph",
                    "data": {"chunk": ("custom", {"stage": "analyze", "node": "analyze"})},
                },
                {
                    "event": "on_chain_stream",
                    "name": "LangGraph",
                    "data": {"chunk": ("values", {"intent": "rag_qa", "report": "Answer", "session_id": "sess-events"})},
                },
            ]
        ),
        stream_kind="events",
        session_id="sess-events",
    )

    events = asyncio.run(_collect_sse(generator))

    assert events == [
        {"type": "event", "stage": "session", "session_id": "sess-events"},
        {"type": "event", "stage": "route", "node": "router", "intent": "rag_qa"},
        {"type": "event", "stage": "tool", "node": "tools", "tool_name": "read_chunk"},
        {"type": "event", "stage": "analyze", "node": "analyze"},
        {"type": "token", "content": "Answer"},
        {
            "type": "event",
            "stage": "done",
            "intent": "rag_qa",
            "session_id": "sess-events",
            "research_mode": None,
            "key_findings": [],
            "uncertainties": [],
            "idea_candidates": [],
            "evidence_count": 0,
        },
        {
            "type": "done",
            "intent": "rag_qa",
            "session_id": "sess-events",
            "research_mode": None,
            "key_findings": [],
            "uncertainties": [],
            "idea_candidates": [],
            "evidence_count": 0,
        },
    ]


def test_rag_node_bridges_inner_graph_updates_to_custom_stream(monkeypatch):
    class FakeInnerRagGraph:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def astream(self, state, **kwargs):
            self.calls.append({"state": state, "kwargs": kwargs})
            yield ("updates", {"tools": {"messages": [SimpleNamespace(tool_call_id="read_chunk")]}})
            yield ("updates", {"analyze": {"answer": "Final answer with citations"}})
            yield (
                "values",
                {
                    "answer": "Final answer with citations",
                    "rag_result": {"sources": ["paper-1"], "chunks": [], "scores": [], "is_sufficient": True},
                },
            )

    fake_inner = FakeInnerRagGraph()
    monkeypatch.setattr(agent_module, "create_rag_node", lambda: fake_inner)

    workflow = StateGraph(AgentState)
    workflow.add_node("rag_qa", agent_module.rag_node)
    workflow.add_edge(START, "rag_qa")
    workflow.add_edge("rag_qa", END)
    graph = workflow.compile()

    async def collect():
        result = []
        async for mode, chunk in graph.astream(_agent_state("研究 RAG"), stream_mode=["custom", "values"]):
            result.append((mode, chunk))
        return result

    outputs = asyncio.run(collect())
    custom_events = [chunk for mode, chunk in outputs if mode == "custom"]
    value_events = [chunk for mode, chunk in outputs if mode == "values"]

    assert custom_events == [
        {"stage": "tool", "node": "tools", "tool_name": "read_chunk"},
        {"stage": "analyze", "node": "analyze"},
    ]
    assert value_events[-1]["report"] == "Final answer with citations"
    assert value_events[-1]["rag_result"]["sources"] == ["paper-1"]
    assert fake_inner.calls[0]["kwargs"]["stream_mode"] == ["updates", "values"]
