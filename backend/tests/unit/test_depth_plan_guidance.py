"""草案 prompt 里三档处理深度的规划指引测试（单元 V1 深度分流）。

覆盖点：
- 三档各自渲出该档的规划原则：minimal 不建正式 Research Plan、lookup 只铺最小检索、
  research 先立目标与子任务
- 三档文案不串味：minimal 里不能出现 research 那套多阶段计划要求，否则模型判了
  minimal 也照样铺完整研究计划
- 指引里不含任何数值上限：深度由模型判，证据不足的出路是升级深度或明确报缺口，
  不是用「最多 N 次检索」截断
- 深度没评估出来时 prompt 保持原样，不塞默认深度与默认指引
"""

from __future__ import annotations

import re

import pytest

from app.services.internal_research.research_planner_plan_prompt import (
    _DEPTH_PLAN_GUIDANCE,
    _depth_assessment_lines,
    build_planner_prompt,
)


# 计数式边界的中英文写法：出现任意一种即说明有人给某一档加了检索或子任务数量上限
_NUMERIC_CAP_PATTERNS = ("最多", "至多", "不超过", "at most", "no more than", "maximum", "max ", "up to ")


def _render(depth: str) -> str:
    """按给定深度渲染深度段文本，供各档指引断言使用。

    Args:
        depth: 处理深度取值，如 minimal / lookup / research

    Returns:
        str: 深度标签与该档规划指引拼成的多行文本
    """
    state = {
        "user_query": "论文 A 的对比学习损失怎么定义",
        "turn_depth_assessment": {
            "depth": depth,
            "reason_code": "single_paper_fact",
            "requires_paper_evidence": depth != "minimal",
        },
    }
    return "\n".join(_depth_assessment_lines(state))


def test_minimal_guidance_forbids_formal_research_plan() -> None:
    """minimal 只做一次理解与回答动作，且论文事实主张必须改走 lookup 或 research。"""
    text = _render("minimal")
    assert "do not create a formal Research Plan" in text
    assert "One understanding-and-answer action is enough" in text
    assert "must move to lookup or research" in text


def test_lookup_guidance_asks_for_smallest_retrieval() -> None:
    """lookup 只铺回答该问题所需的最小检索，不为形式增加无关检索，也不铺多阶段计划。"""
    text = _render("lookup")
    assert "only the retrieval and evidence checking that answering this one question needs" in text
    assert "unrelated retrieval for the sake of form" in text
    assert "do not lay out a full multi-stage research plan" in text
    # 证据不足的出路写成升级或报缺口，这是不加数值上限的前提
    assert "escalate this turn to research or report the gap" in text


def test_lookup_guidance_requires_this_turn_evidence() -> None:
    """lookup 的论文事实必须带本轮 Evidence，且能定位到论文与原文片段。"""
    text = _render("lookup")
    assert "carries this-turn Evidence" in text
    assert "locates both the paper" in text


def test_research_guidance_asks_goals_then_subtasks() -> None:
    """research 先形成可执行研究目标与子任务，结论要能回溯本轮 Evidence 且分论文呈现。"""
    text = _render("research")
    assert "First state executable research goals and the sub-tasks" in text
    assert "trace back to this-turn Evidence" in text
    assert "different papers stays presented separately" in text


def test_minimal_guidance_does_not_borrow_research_requirements() -> None:
    """三档文案不许串味：minimal 里不能带 research 那套多阶段计划要求。"""
    minimal_text = _render("minimal")
    for leaked in ("sub-task", "research goals", "trace back", "multi-stage", "presented separately"):
        assert leaked not in minimal_text


@pytest.mark.parametrize("depth", ["minimal", "lookup", "research"])
def test_no_numeric_cap_in_depth_guidance(depth: str) -> None:
    """任何一档都不许出现数值上限：既无「最多 N 次」式措辞，也不出现裸数字。

    深度分流已由模型判定，证据不够还有重规划与补证据回边兜底；再加一个计数上限
    就变成仅为掩盖问题设的最大轮数，会把本该升级深度的轮次直接截断。
    """
    for line in _DEPTH_PLAN_GUIDANCE[depth]:
        lowered = line.lower()
        for pattern in _NUMERIC_CAP_PATTERNS:
            assert pattern not in lowered, f"{depth} 档指引出现计数上限措辞: {line}"
        assert not re.search(r"\d", line), f"{depth} 档指引出现数字上限: {line}"


def test_unassessed_depth_renders_nothing() -> None:
    """深度没评估出来时返回空列表，不编一个默认深度与默认指引塞进 prompt。"""
    assert _depth_assessment_lines({}) == []
    assert _depth_assessment_lines({"turn_depth_assessment": None}) == []
    assert _depth_assessment_lines({"turn_depth_assessment": {}}) == []
    assert _depth_assessment_lines({"turn_depth_assessment": {"depth": ""}}) == []


def test_unknown_depth_keeps_labels_without_guidance() -> None:
    """深度取值不在三档内时只渲标签，不造第四套指引，也不改原有渲染行为。"""
    lines = _depth_assessment_lines({"turn_depth_assessment": {"depth": "exhaustive"}})
    assert lines[0] == "Turn depth: exhaustive"
    assert not any("Plan guidance" in line for line in lines)


def test_guidance_reaches_planner_prompt() -> None:
    """指引要真进到发给模型的草案 prompt，而不是只在辅助函数里拼出来。"""
    state = {
        "user_query": "论文 A 的对比学习损失怎么定义",
        "turn_depth_assessment": {
            "depth": "lookup",
            "reason_code": "single_paper_fact",
            "requires_paper_evidence": True,
        },
    }
    prompt = build_planner_prompt(state)
    assert "Plan guidance for depth=lookup" in prompt
    assert "do not lay out a full multi-stage research plan" in prompt
