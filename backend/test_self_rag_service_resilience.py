from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from app.services import self_rag_service


def _with_trace(payload: dict, trace) -> dict:  # noqa: ANN001
    if trace is None:
        return payload
    return {**payload, "trace": trace}


def _dedupe_hits(groups: list[list[dict]]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for group in groups:
        for item in group or []:
            key = (item.get("paper_id", ""), item.get("chunk_id", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def test_execute_evaluation_falls_back_to_write_back_when_llm_fails() -> None:
    async def _fail_evaluate(question: str, answer: str, evidence: str):  # noqa: ANN001
        raise ValueError("bad structured output")

    state = {
        "user_query": "Chain-of-Thought 是什么？",
        "answer": "初始回答",
        "answer_confidence": "medium",
        "self_rag_iteration": 1,
    }

    result = asyncio.run(
        self_rag_service.execute_evaluation(
            state,
            evaluate_answer=_fail_evaluate,
            should_refine=lambda evaluation, iteration: True,
            build_evidence_text_fn=lambda current: "evidence text",
            with_trace=_with_trace,
            update_trace_generation=lambda *args, **kwargs: None,
        )
    )

    assert result["self_rag_action"] == "write_back"
    assert result["self_rag_iteration"] == 1
    assert result["answer_confidence"] == "medium"
    assert "self_rag evaluation failed" in result["failure_reason"]


def test_execute_refinement_preserves_answer_when_refine_fails(monkeypatch) -> None:
    async def _fail_refine(**kwargs):  # noqa: ANN003, ANN001
        raise ValueError("invalid cited answer")

    monkeypatch.setattr(self_rag_service.settings, "self_rag_max_iterations", 2, raising=False)

    state = {
        "user_query": "Chain-of-Thought 是什么？",
        "answer": "已有回答",
        "citations": [
            {
                "paper_id": "paper-1",
                "chunk_id": "chunk-1",
                "section": "Method",
                "page_num": 4,
                "excerpt": "CoT improves reasoning.",
            }
        ],
        "answer_confidence": "medium",
        "self_rag_iteration": 1,
        "evaluation": {
            "relevance": "high",
            "faithfulness": "medium",
            "completeness": "partial",
            "quality_score": 0.6,
            "grounded": True,
            "missing_info": ["缺少定义"],
            "follow_up_queries": [],
            "has_hallucination": False,
        },
        "search_results": [
            {
                "paper_id": "paper-1",
                "chunk_id": "chunk-1",
                "preview": "old",
            }
        ],
    }

    result = asyncio.run(
        self_rag_service.execute_refinement(
            state,
            build_supplementary_queries=lambda evaluation: ["CoT 定义"],
            refine_answer=_fail_refine,
            get_vector_store=lambda: object(),
            search_chunks_fn=lambda store, query, top_k, filter_expr: [
                {
                    "paper_id": "paper-2",
                    "chunk_id": "chunk-2",
                    "preview": "new evidence",
                }
            ],
            dedupe_hits_fn=_dedupe_hits,
            format_search_results_fn=lambda results: "supplementary evidence",
            with_trace=_with_trace,
            update_trace_generation=lambda *args, **kwargs: None,
        )
    )

    assert result["answer"] == "已有回答"
    assert result["citations"] == state["citations"]
    assert result["answer_confidence"] == "medium"
    assert result["self_rag_iteration"] == 2
    assert "self_rag refine failed" in result["failure_reason"]
    assert result["search_results"][-1]["chunk_id"] == "chunk-2"
    assert isinstance(result["messages"][0], HumanMessage)
    assert "supplementary evidence" in result["messages"][0].content
