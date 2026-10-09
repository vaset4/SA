from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

os.environ["SSLKEYLOGFILE"] = ""

import app.api.server as server_module
import app.api.facades.graph_facade as graph_facade_module

class FakeResearchEndpointAgent:
    def __init__(self) -> None:
        self.last_state = None

    async def ainvoke(self, state, config=None):
        self.last_state = state
        return {
            "report": "# Routed report",
            "intent": state.get("intent") or "arxiv_search",
        }


async def _call_research(query: str, session_id: str | None = None):
    req = server_module.ResearchRequest(
        query=query,
        session_id=session_id,
    )
    return await server_module.research(req, user={})


def test_research_endpoint_returns_direct_report(monkeypatch):
    class DirectReportAgent:
        async def ainvoke(self, state, config=None):
            return {"report": "# Final research report", "intent": state.get("intent")}

    async def fake_get_shared_agent():
        return DirectReportAgent()

    monkeypatch.setattr(graph_facade_module, "get_shared_agent", fake_get_shared_agent)

    payload = asyncio.run(_call_research("研究 LangGraph interrupt"))
    assert payload["report"] == "# Final research report"
    assert payload["session_id"]
    assert "awaiting_confirmation" not in payload
    assert "pending_plan" not in payload


def test_research_endpoint_returns_clarification_payload(monkeypatch):
    question = "请先说明你更关心 API 用法还是工程接入。"

    class ClarifyAgent:
        async def ainvoke(self, state, config=None):
            return {
                "__interrupt__": [
                    SimpleNamespace(
                        value={
                            "kind": "deep_research_clarification",
                            "question": question,
                            "report": question,
                        }
                    )
                ],
                "intent": state.get("intent"),
            }

    async def fake_get_shared_agent():
        return ClarifyAgent()

    monkeypatch.setattr(graph_facade_module, "get_shared_agent", fake_get_shared_agent)

    payload = asyncio.run(_call_research("研究 LangGraph interrupt"))
    assert payload["awaiting_clarification"] is True
    assert payload["clarification_question"] == question
    assert payload["report"] == payload["clarification_question"]


def test_research_endpoint_returns_external_research_structure(monkeypatch):
    candidate_paper = {
        "title": "LangGraph interrupt docs",
        "url": "https://example.com/langgraph-interrupt",
        "reason_to_import": "适合作为后续内部研究的补库入口。",
    }

    class StructuredAgent:
        async def ainvoke(self, state, config=None):
            return {
                "report": "# Final report",
                "intent": state.get("intent") or "deep_research",
                "research_mode": "external_only",
                "key_findings": ["外部资料已覆盖 interrupt / resume 主路径。"],
                "uncertainties": ["仍需补充生产环境恢复案例。"],
                "idea_candidates": ["把 checkpoint 恢复与前端 session 语义统一验证。"],
                "external_gaps": ["缺少生产环境 session 恢复的外部案例对照。"],
                "follow_up_questions": ["哪些资料值得补库后回到内部研究继续比对？"],
                "candidate_papers_to_import": [candidate_paper],
            }

    async def fake_get_shared_agent():
        return StructuredAgent()

    monkeypatch.setattr(graph_facade_module, "get_shared_agent", fake_get_shared_agent)

    payload = asyncio.run(_call_research("研究 LangGraph interrupt"))
    assert payload["session_id"]
    assert payload["research_mode"] == "external_only"
    assert payload["key_findings"] == ["外部资料已覆盖 interrupt / resume 主路径。"]
    assert payload["uncertainties"] == ["仍需补充生产环境恢复案例。"]
    assert payload["idea_candidates"] == ["把 checkpoint 恢复与前端 session 语义统一验证。"]
    assert payload["external_gaps"] == ["缺少生产环境 session 恢复的外部案例对照。"]
    assert payload["follow_up_questions"] == ["哪些资料值得补库后回到内部研究继续比对？"]
    assert payload["candidate_papers_to_import"] == [candidate_paper]


def test_research_endpoint_forces_deep_research_intent(monkeypatch):
    agent = FakeResearchEndpointAgent()

    async def fake_get_shared_agent():
        return agent

    monkeypatch.setattr(graph_facade_module, "get_shared_agent", fake_get_shared_agent)

    response = asyncio.run(_call_research("RAG的发展"))
    assert response["intent"] == "deep_research"
    assert agent.last_state["intent"] == "deep_research"


def test_research_endpoint_maps_llm_empty_response_to_bad_gateway(monkeypatch):
    class FailingAgent:
        async def ainvoke(self, state, config=None):
            raise ValueError("LLM 返回空响应体: status=200, content_type=")

    async def fake_get_shared_agent():
        return FailingAgent()

    monkeypatch.setattr(graph_facade_module, "get_shared_agent", fake_get_shared_agent)

    with pytest.raises(server_module.HTTPException) as exc_info:
        asyncio.run(_call_research("研究 LangGraph interrupt"))

    assert exc_info.value.status_code == 502
    assert "deep research failed: LLM 返回空响应体" in exc_info.value.detail


def test_research_endpoint_preserves_http_timeout_from_graph(monkeypatch):
    class TimeoutAgent:
        async def ainvoke(self, state, config=None):
            raise server_module.HTTPException(status_code=504, detail="Deep Research 超时（300s），请缩小研究范围或稍后重试")

    async def fake_get_shared_agent():
        return TimeoutAgent()

    monkeypatch.setattr(graph_facade_module, "get_shared_agent", fake_get_shared_agent)

    with pytest.raises(server_module.HTTPException) as exc_info:
        asyncio.run(_call_research("RAG的发展"))

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "Deep Research 超时（300s），请缩小研究范围或稍后重试"


def test_invoke_graph_uses_extended_timeout_for_deep_research(monkeypatch):
    captured = {"timeout": None}

    class FastAgent:
        async def ainvoke(self, state, config=None):
            return {"report": "# ok", "intent": state.get("intent")}

    async def fake_get_shared_agent():
        return FastAgent()

    async def fake_wait_for(awaitable, timeout):  # noqa: ANN001
        captured["timeout"] = timeout
        return await awaitable

    monkeypatch.setattr(graph_facade_module, "get_shared_agent", fake_get_shared_agent)
    monkeypatch.setattr(
        graph_facade_module,
        "get_settings",
        lambda: SimpleNamespace(graph_timeout_seconds=180.0, deep_research_graph_timeout_seconds=300.0),
    )
    monkeypatch.setattr(graph_facade_module.asyncio, "wait_for", fake_wait_for)

    result, _session_id = asyncio.run(
        graph_facade_module.invoke_graph(
            "研究 LangGraph interrupt",
            {"intent": "deep_research", "session_id": "session-timeout"},
        )
    )

    assert result["report"] == "# ok"
    assert captured["timeout"] == 300.0
