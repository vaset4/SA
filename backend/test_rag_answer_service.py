from __future__ import annotations

import asyncio

from app.services import rag_answer_service


class _FakeLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def ainvoke(self, messages):  # noqa: ANN001, ARG002
        return type("LLMResponse", (), {"content": self._payload})()


class _CapturingLLM(_FakeLLM):
    def __init__(self, payload: str) -> None:
        super().__init__(payload)
        self.messages = None

    async def ainvoke(self, messages):  # noqa: ANN001
        self.messages = messages
        return await super().ainvoke(messages)


class _FakeEvidenceService:
    def __init__(self, bundle: dict | None = None) -> None:
        self.bundle = bundle or {
            "paper_sources": [],
            "chunk_citations": [],
            "graph_evidence": [],
            "evidence_summary": "",
        }

    async def build_bundle(self, **kwargs):  # noqa: ANN003
        return self.bundle


class _FailingLLM:
    async def ainvoke(self, messages):  # noqa: ANN001, ARG002
        raise ValueError("LLM natural response failed")


def _identity_with_trace(payload, trace):  # noqa: ANN001
    if trace is None:
        return payload
    return {**payload, "trace": trace}


def _noop_update_trace(trace, **kwargs):  # noqa: ANN001, ARG001, ANN003
    return None


def _never_react_action(message) -> bool:  # noqa: ANN001
    return False


def test_build_synthesis_brief_prefers_structured_research_state() -> None:
    brief = rag_answer_service.build_synthesis_brief(
        {
            "dialogue_mode": "research",
            "structured_query": {"paper_ids": ["paper-1", "paper-2"]},
            "search_results": [
                {"paper_id": "paper-1", "parent_id": "parent-1"},
                {"paper_id": "paper-2", "parent_id": "parent-2"},
            ],
            "graph_context": "graph hint",
            "synthesis_state": {
                "project_scope": {"summary": "当前项目范围覆盖 2 篇论文。"},
                "paper_focus": {"summary": "当前问题聚焦 2 篇论文。"},
                "supported_dimensions": ["methods", "comparisons"],
                "unsupported_dimensions": ["limitations"],
                "open_questions": ["局限性证据不足。"],
            },
        }
    )

    assert "Dialogue mode: research" in brief
    assert "Project scope: 当前项目范围覆盖 2 篇论文。" in brief
    assert "Paper focus: 当前问题聚焦 2 篇论文。" in brief
    assert "graph_hint=yes" in brief
    assert "Supported dimensions: methods, comparisons" in brief
    assert "Missing dimensions: limitations" in brief


def test_execute_analysis_returns_natural_answer_and_summary(monkeypatch) -> None:
    llm = _FakeLLM("STGCN 解决了交通预测中的时空依赖建模问题。[paper_id]")
    monkeypatch.setattr(
        rag_answer_service,
        "get_explainable_answer_service",
        lambda: _FakeEvidenceService(
            {
                "paper_sources": [{"paper_id": "paper-1"}],
                "chunk_citations": [],
                "graph_evidence": [],
                "evidence_summary": "本次回答引用了 1 篇论文。",
            }
        ),
    )
    monkeypatch.setattr(
        rag_answer_service.internal_research_service,
        "build_internal_research_contract",
        lambda **kwargs: {
            "research_scope": {"mode": "single_paper"},
            "research_objects": [{"key": "methods"}],
            "note_actions": [{"key": "copy"}],
            "synthesis_state": {"status": "ready"},
            "dialogue_mode": "lookup",
        },
    )

    result = asyncio.run(
        rag_answer_service.execute_analysis(
            {
                "user_query": "stgcn解决了什么问题",
                "context_summary": "summary",
                "messages": [],
                "search_results": [
                    {
                        "paper_id": "paper-1",
                        "chunk_id": "chunk-1",
                        "section": "Method",
                        "page_num": 4,
                        "preview": "时空依赖建模",
                    }
                ],
            },
            llm=llm,
            analyze_prompt="Question: {question}\nSynthesis: {synthesis}\nSummary: {summary}",
            with_trace=_identity_with_trace,
            update_trace_generation=_noop_update_trace,
            is_react_action_message=_never_react_action,
        )
    )

    assert result["answer"] == "STGCN 解决了交通预测中的时空依赖建模问题。"
    assert result["citations"] == []
    assert result["research_scope"] == {"mode": "single_paper"}
    assert result["research_objects"] == [{"key": "methods"}]
    assert result["note_actions"] == [{"key": "copy"}]
    assert result["synthesis_state"] == {"status": "ready"}
    assert result["dialogue_mode"] == "lookup"
    assert result["answer_confidence"] == "medium"
    assert result["chunk_citations"] == [
        {
            "paper_id": "paper-1",
            "chunk_id": "chunk-1",
            "section": "Method",
            "page_num": 4,
            "excerpt": "时空依赖建模",
        }
    ]
    assert result["evidence_summary"] == "本次回答引用了 1 篇论文。"


def test_execute_analysis_returns_safe_message_when_natural_generation_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        rag_answer_service,
        "get_explainable_answer_service",
        lambda: _FakeEvidenceService(),
    )

    result = asyncio.run(
        rag_answer_service.execute_analysis(
            {
                "user_query": "stgcn解决了什么问题",
                "context_summary": "summary",
                "messages": [],
            },
            llm=_FailingLLM(),
            analyze_prompt="Question: {question}\nSynthesis: {synthesis}\nSummary: {summary}",
            with_trace=_identity_with_trace,
            update_trace_generation=_noop_update_trace,
            is_react_action_message=_never_react_action,
        )
    )

    assert result["answer"] == "当前已检索到部分证据，但最终回答生成失败，请稍后重试或缩小问题范围。"
    assert result["citations"] == []
    assert result["paper_sources"] == []
    assert result["chunk_citations"] == []
    assert result["graph_evidence"] == []
    assert result["evidence_summary"] == "最终回答生成阶段失败，未能产出结构化引用证据。"
    assert result["answer_confidence"] == "low"


def test_execute_analysis_appends_graph_context_when_messages_do_not_contain_it(monkeypatch) -> None:
    llm = _CapturingLLM("图谱线索显示两篇论文都在做时空建模。")
    monkeypatch.setattr(
        rag_answer_service,
        "get_explainable_answer_service",
        lambda: _FakeEvidenceService(),
    )

    asyncio.run(
        rag_answer_service.execute_analysis(
            {
                "user_query": "这两个论文的方法有什么共同点",
                "context_summary": "",
                "messages": [],
                "graph_context": "Graph says both papers model spatiotemporal dependencies.",
            },
            llm=llm,
            analyze_prompt="Question: {question}\nSynthesis: {synthesis}\nSummary: {summary}",
            with_trace=_identity_with_trace,
            update_trace_generation=_noop_update_trace,
            is_react_action_message=_never_react_action,
        )
    )

    assert llm.messages is not None
    assert any(
        "=== Graph Context ===\nGraph says both papers model spatiotemporal dependencies."
        in str(getattr(message, "content", ""))
        for message in llm.messages
    )
