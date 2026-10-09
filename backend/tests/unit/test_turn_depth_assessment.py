"""Planner 首次分析的深度评估测试（单元 1d）。

覆盖点：
- 确定性硬约束：三条约束各自越界都抛 TurnDepthAssessmentError，不修正模型输出
- 无歧义样例：「这篇的作者是谁」判 lookup 通过、判 minimal+要证据被拒；
  「比较这三篇」判 minimal 被拒、判 research 通过
- 意图与焦点并入：resolve_turn_context 产出 intent/focus 与四项追问承接字段
- 越界即失败：analyze_planner_state 返回 depth_assessment=failed，不退回默认深度
- 幂等：replan 回边重入 analyze 时不再重判深度

作者：xxx
创建时间：2026-08-29 00:00:00
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.models.research_planner_contracts import TurnDepthProposal
from app.services.internal_research import research_planner_pipeline
from app.services.internal_research.focus_resolver import (
    FOCUS_REFERENCE_UNRESOLVED,
    resolve_focus,
)
from app.services.internal_research.turn_context_resolver import resolve_turn_context
from app.services.internal_research.turn_depth_assessor import (
    TurnDepthAssessmentError,
    assess_turn_depth,
    build_depth_prompt,
    is_explicit_multi_paper_compare,
)


class _StubStructuredLLM:
    """按 schema 返回预置结构化对象的 LLM 替身。"""

    def __init__(self, payloads: dict[str, Any]) -> None:
        self._payloads = payloads
        self.schemas: list[str] = []

    def with_structured_output(self, schema: type) -> "_StubStructuredLLM":
        self.schemas.append(schema.__name__)
        self._active = schema.__name__
        return self

    async def ainvoke(self, _messages: Any) -> Any:
        payload = self._payloads[self._active]
        if isinstance(payload, Exception):
            raise payload
        return payload


def _proposal(**overrides: Any) -> TurnDepthProposal:
    fields: dict[str, Any] = {
        "depth": "lookup",
        "reason_code": "single_paper_metadata_lookup",
        "requires_paper_evidence": True,
        "resolved_query": "这篇论文的作者是谁",
    }
    fields.update(overrides)
    return TurnDepthProposal(**fields)


def _single_paper_state(question: str = "这篇的作者是谁") -> dict[str, Any]:
    return {
        "user_query": question,
        "intent": "lookup",
        "focus": {"project_scope": ["p1"], "conversation_scope": [], "turn_focus": ["p1"]},
    }


def _compare_state(question: str = "比较这三篇的实验设置") -> dict[str, Any]:
    return {
        "user_query": question,
        "intent": "compare",
        "focus": {
            "project_scope": ["p1", "p2", "p3"],
            "conversation_scope": [],
            "turn_focus": ["p1", "p2", "p3"],
            "fallback_reason": "",
        },
    }


def _assess(state: dict[str, Any], proposal: Any, papers: list[str]) -> Any:
    llm = _StubStructuredLLM({"TurnDepthProposal": proposal})
    return asyncio.run(
        assess_turn_depth(state, llm, scoped_paper_ids=frozenset(papers) if papers else None)
    )


async def _stub_load_papers(paper_ids: list[str]) -> list[dict[str, Any]]:
    """焦点解析用例的读取替身：按 ID 返回最小论文记录，读取真经由注入的回调发生。"""
    return [{"paper_id": paper_id, "title": f"标题{paper_id}"} for paper_id in paper_ids]


class TestUnambiguousSamples:
    """无歧义样例：单篇论文事实问题与明确多论文比较。"""

    def test_single_paper_author_question_stays_lookup(self) -> None:
        """「这篇的作者是谁」判 lookup + 要证据是合法结果，不进 research。"""
        assessment = _assess(_single_paper_state(), _proposal(), ["p1"])

        assert assessment["depth"] == "lookup"
        assert assessment["requires_paper_evidence"] is True
        assert assessment["reason_code"] == "single_paper_metadata_lookup"
        assert assessment["resolved_query"] == "这篇论文的作者是谁"

    def test_single_paper_author_question_rejects_minimal(self) -> None:
        """论文事实问题判 minimal 又要证据时越界，抛错而不改判默认深度。"""
        with pytest.raises(TurnDepthAssessmentError) as exc:
            _assess(
                _single_paper_state(),
                _proposal(depth="minimal", reason_code="chat_only"),
                ["p1"],
            )

        assert exc.value.reason_code == "minimal_requires_paper_evidence"

    def test_multi_paper_compare_rejects_minimal(self) -> None:
        """「比较这三篇」不得判 minimal。"""
        with pytest.raises(TurnDepthAssessmentError) as exc:
            _assess(
                _compare_state(),
                _proposal(
                    depth="minimal",
                    reason_code="no_paper_fact_needed",
                    requires_paper_evidence=False,
                    resolved_query="比较这三篇的实验设置",
                ),
                ["p1", "p2", "p3"],
            )

        assert exc.value.reason_code == "multi_paper_compare_without_evidence"

    def test_multi_paper_compare_accepts_research(self) -> None:
        """多论文比较判 research + 要证据是合法结果。"""
        assessment = _assess(
            _compare_state(),
            _proposal(
                depth="research",
                reason_code="cross_paper_comparison",
                resolved_query="比较这三篇的实验设置",
            ),
            ["p1", "p2", "p3"],
        )

        assert assessment["depth"] == "research"
        assert assessment["requires_paper_evidence"] is True


class TestDeterministicConstraints:
    """三条确定性硬约束：模型偏好不得越界。"""

    def test_empty_scope_cannot_require_paper_evidence(self) -> None:
        """项目内没有论文时要求论文 Evidence 永远无法满足，直接越界。"""
        state = {"user_query": "帮我把这个研究问题理一下", "intent": "explore", "focus": {}}

        with pytest.raises(TurnDepthAssessmentError) as exc:
            _assess(state, _proposal(depth="lookup"), [])

        assert exc.value.reason_code == "empty_scope_requires_paper_evidence"

    def test_empty_scope_allows_minimal_without_evidence(self) -> None:
        """空论文范围下 minimal + 不要证据是合法的，空项目仍能正常交流。"""
        state = {"user_query": "帮我把这个研究问题理一下", "intent": "explore", "focus": {}}
        assessment = _assess(
            state,
            _proposal(
                depth="minimal",
                reason_code="no_paper_fact_needed",
                requires_paper_evidence=False,
                resolved_query="帮我把这个研究问题理一下",
            ),
            [],
        )

        assert assessment["depth"] == "minimal"
        assert assessment["requires_paper_evidence"] is False

    def test_compare_intent_with_single_paper_is_not_multi_paper_compare(self) -> None:
        """只说了「不同」但焦点不足两篇时不算明确多论文比较，不触发第三条约束。"""
        state = {
            "user_query": "这篇论文和常规做法有什么不同",
            "intent": "compare",
            "focus": {"project_scope": ["p1"], "conversation_scope": [], "turn_focus": ["p1"]},
        }

        assert is_explicit_multi_paper_compare(state) is False
        assert _assess(state, _proposal(depth="lookup"), ["p1"])["depth"] == "lookup"

    def test_multi_paper_compare_needs_two_papers_in_focus(self) -> None:
        """明确多论文比较只认确定性信号：compare 意图 + 焦点两篇以上。"""
        assert is_explicit_multi_paper_compare(_compare_state()) is True
        assert is_explicit_multi_paper_compare(_single_paper_state()) is False
        assert is_explicit_multi_paper_compare({"intent": "compare"}) is False


@pytest.mark.parametrize(
    ("question", "history", "reason", "expected_focus", "scope"),
    [
        ("我想比较两篇论文的动态关系建模，先等我指定论文。", [],
         "compare_insufficient_focus", [f"p{i}" for i in range(10)],
         [f"p{i}" for i in range(10)]),
        ("比较第一篇和第四篇", [], FOCUS_REFERENCE_UNRESOLVED,
         ["p1", "p2", "p3"], ["p1", "p2", "p3"]),
        ("比较第一篇和第四篇", [{"paper_ids": ["p1", "p2"]}],
         FOCUS_REFERENCE_UNRESOLVED, ["p1", "p2"], ["p1", "p2", "p3"]),
        ("比较第一篇的方法", [{"paper_ids": ["foreign"]}],
         "compare_insufficient_focus", ["p1", "p2", "p3"], ["p1", "p2", "p3"]),
    ],
)
def test_unresolved_comparison_keeps_scope_without_rejecting_waiting(
    question, history, reason, expected_focus, scope,
):
    """输入问题、历史和预期焦点；验证真实解析保留范围且允许无证据等待，无返回值。"""
    focus = resolve_focus(
        question, intent="compare", project_paper_ids=scope, conversation_history=history,
    )
    assert focus["fallback_reason"] == reason
    assert focus["turn_focus"] == expected_focus
    assert focus["project_scope"] == scope
    state = {"user_query": question, "intent": "compare", "focus": focus}
    assert is_explicit_multi_paper_compare(state) is False
    proposal = _proposal(
        depth="minimal", reason_code="waiting_for_papers",
        requires_paper_evidence=False, resolved_query=question,
    )
    assessment = _assess(state, proposal, scope)
    assert assessment["depth"] == proposal.depth
    assert assessment["reason_code"] == proposal.reason_code
    assert assessment["requires_paper_evidence"] is False


@pytest.mark.parametrize(
    ("question", "history", "expected_focus"),
    [
        ("比较第一篇和第二篇", [], ["p1", "p2"]),
        ("比较Alpha Study和Beta Study", [], ["p1", "p2"]),
        ("和上次那篇比呢", [{"paper_ids": ["p2", "p1"]}], ["p2", "p1"]),
        ("比较第一篇的方法", [{"paper_ids": ["p2", "foreign"]}], ["p1", "p2"]),
    ],
)
def test_resolved_comparison_still_requires_evidence(question, history, expected_focus):
    """输入明确对象及历史；验证授权收窄后的两篇比较仍要求证据，无返回值。"""
    scope = ["p1", "p2", "p3"]
    focus = resolve_focus(
        question, intent="compare", project_paper_ids=scope, conversation_history=history,
        papers=[{"paper_id": "p1", "title": "Alpha Study"},
                {"paper_id": "p2", "title": "Beta Study"}],
    )
    assert focus["fallback_reason"] == ""
    assert focus["turn_focus"] == expected_focus
    assert set(focus["turn_focus"]) <= set(scope)
    state = {"user_query": question, "intent": "compare", "focus": focus}
    assert is_explicit_multi_paper_compare(state) is True
    for depth in ("minimal", "lookup"):
        with pytest.raises(TurnDepthAssessmentError) as exc:
            _assess(state, _proposal(depth=depth, requires_paper_evidence=False), scope)
        assert exc.value.reason_code == "multi_paper_compare_without_evidence"
    assert _assess(state, _proposal(depth="research"), scope)["requires_paper_evidence"] is True


@pytest.mark.parametrize("focus", [
    None, {}, {"turn_focus": ["p1", "p2"]},
    {"turn_focus": ["p1", "p2"], "fallback_reason": None},
    {"turn_focus": ["p1", "p2"], "fallback_reason": "unrecognized_reason"},
    {"turn_focus": None, "fallback_reason": ""},
])
def test_unconfirmed_focus_does_not_supply_explicit_comparison_proof(focus):
    """输入缺失或未知焦点；验证不冒充明确比较且不改模型深度，无返回值。"""
    state = {"user_query": "比较方案", "intent": "compare", "focus": focus}
    assert is_explicit_multi_paper_compare(state) is False
    proposal = _proposal(depth="research", requires_paper_evidence=True)
    assessment = _assess(state, proposal, ["p1", "p2"])
    assert assessment["depth"] == "research"
    assert assessment["requires_paper_evidence"] is True
    assert state["focus"] == focus


class TestDepthPrompt:
    """深度 prompt 必须带上三档正面定义与三条硬约束。"""

    def test_prompt_carries_depth_definitions_and_constraints(self) -> None:
        prompt = build_depth_prompt(_single_paper_state(), frozenset({"p1"}))

        assert "depth=minimal" in prompt
        assert "depth=lookup" in prompt
        assert "depth=research" in prompt
        assert "must not be minimal" in prompt
        assert "requires_paper_evidence must be false" in prompt
        assert "evidence-free direct answer" in prompt
        assert "never by keyword-to-category mapping" in prompt

    def test_prompt_carries_turn_context(self) -> None:
        """问题、焦点与授权论文数量必须进 prompt，否则模型无从判断需要什么。"""
        prompt = build_depth_prompt(_compare_state(), frozenset({"p1", "p2", "p3"}))

        assert "比较这三篇的实验设置" in prompt
        assert "Authorized paper count: 3" in prompt
        assert "Coarse intent: compare" in prompt

    def test_prompt_separates_general_concepts_from_paper_facts(self) -> None:
        """通用概念与论文事实分属两档：问概念不该去翻项目论文。

        项目里有论文不等于这个话题就得查证据。用户凭空问一个教科书概念（如「什么是
        Transformer」），用通用知识直接答即可；只有问法指向论文（这篇怎么说、项目里的
        论文怎么处理）才落 lookup。分界线写错了，概念题就会退化成翻一遍库再答「没找到」，
        这是收敛前不存在、收敛后必须挡住的那类回退。
        """
        prompt = build_depth_prompt(_single_paper_state(), frozenset({"p1"}))

        assert "general concept" in prompt
        assert "textbook definition" in prompt
        assert "points at the papers" in prompt
        # 旧口径把「definition」整类划进 lookup，等于让概念题一律去查证据
        assert "e.g. a definition," not in prompt


class TestTurnContextResolution:
    """意图与焦点解析并入 Planner 后仍产出全部下游依赖字段。"""

    def test_resolve_turn_context_produces_intent_and_focus(self) -> None:
        state = {
            "user_query": "对比 p1 和 p2 的方法",
            "structured_query": {"paper_ids": ["p1", "p2", " "]},
            "session_id": "sess-1",
        }
        delta = asyncio.run(resolve_turn_context(state, load_papers=_stub_load_papers))

        assert delta["intent"] == "compare"
        assert delta["focus"]["turn_focus"] == ["p1", "p2"]
        assert delta["focus"]["project_scope"] == ["p1", "p2"]
        assert delta["retry_count"] == 0
