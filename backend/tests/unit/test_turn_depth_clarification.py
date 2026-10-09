"""澄清提问作为 Planner 可调用能力（单元 V2，补 AC :366）。

作者：xxx
创建时间：2026-09-03 00:00:00

目标没确认、且不同理解会导致明显不同动作时，这一轮该先问用户一句，而不是按其中
一种理解把活干完（prd.md:100）。本文件盯住这条出口上五件会静默失效的事：

1. 契约上「判要问」却不带问题或原因码：这一轮没有可返回给用户的内容，trace 上也查
   不出为什么问，澄清于是退化成一个无从追查的偷懒默认值。
2. 判据只能是 needs_clarification：光看有没有问题文本，分不出「模型判要问」与
   「模型顺手写了一句、但判的是不用问」。
3. 澄清必须排在 minimal 直答与产草案之前：排在后面等于先按一种理解把活干完，
   再回头问用户是不是这个意思。
4. 决策是 terminate_honest 而不是 terminate_fail，且 analyze 阶段的它必须直奔
   write_back：进 synthesize 会被那边的诚实收尾分支改写成「范围内查过但没找到」。
5. 澄清回合不产记忆候选（prd.md §7.2 的「固定不触发」清单）：靠既有终态闸，
   不为澄清另加一道闸门。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.middleware.trace_builders import build_trace_summary_payload
from app.models.research_contracts import TurnDepthAssessment
from app.models.research_planner_contracts import (
    ResearchPlanDraft,
    TurnClarificationRequest,
    TurnDepthProposal,
)
from app.nodes import rag_node
from app.services.internal_research import research_memory_writeback as writeback
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research import research_planner_plan_prompt as plan_prompt
from app.services.internal_research.turn_depth_assessor import (
    assess_turn_depth,
    build_depth_prompt,
)

# ─── 跨文件约定本身就是这些字面量 ───

_QUESTION = "帮我把这篇的实验部分理一下"
# 澄清问句：说人话、点明两种理解，且不含任何内部组件名
_CLARIFY_TEXT = "你是想按小节顺序把实验部分整理成要点，还是只挑出与结论相关的那几处？"
_CLARIFY_REASON = "unclear_requested_output"
_DEPTH_REASON = "single_paper_section_walkthrough"

# 深度取值
_DEPTH_MINIMAL = "minimal"
_DEPTH_LOOKUP = "lookup"
_DEPTH_RESEARCH = "research"

# 状态与载荷键名
_DRAFT_KEY = "planner_draft"
_DECISION_KEY = "planner_loop_decision"
_BUDGET_KEY = "planner_budget"
_TRACE_KEY = "planner_recovery_trace"
_ASSESSMENT_KEY = "turn_depth_assessment"
# 澄清判断在深度评估结果里的三个键
_NEEDS_CLARIFY_KEY = "needs_clarification"
_CLARIFY_QUESTION_KEY = "clarification_question"
_CLARIFY_REASON_KEY = "clarification_reason_code"

# 澄清是正常终止而不是硬失败
_TERMINATE_HONEST = "terminate_honest"
_TERMINATE_FAIL = "terminate_fail"
# 本轮没取论文证据
_PLANNER_SKIPPED = "skipped"
_PLANNER_ANALYZED = "analyzed"
# 四态能力表里澄清这一项的名字，取 design.md §6.2 循环里的动作名
_CAPABILITY_CLARIFY = "ask_clarification"


class _StubStructuredLLM:
    """按 schema 名返回预置结构化对象的 LLM 替身，做法照 test_minimal_depth_short_circuit。"""

    def __init__(self, payloads: dict[str, Any]) -> None:
        self._payloads = payloads
        self._active = ""

    def with_structured_output(self, schema: type) -> "_StubStructuredLLM":
        self._active = schema.__name__
        return self

    async def ainvoke(self, _messages: Any) -> Any:
        if self._active not in self._payloads:
            raise AssertionError(f"本用例不该走到 {self._active} 的结构化调用")
        return self._payloads[self._active]


@dataclass(frozen=True)
class _StubDraft:
    """草案替身：本文件只关心草案有没有被产出来，以及它有没有表达澄清。

    clarification 对齐 ResearchPlanDraft 契约：analyze 在产草案之后读它判断这一轮
    是不是「先问用户」。替身少了这个字段，用例会红在 AttributeError 上，那不是被测行为。
    """

    clarification: Any = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"question": _QUESTION}


class _DraftRecorder:
    """记下 _analyze_plan 调用次数：零调用才是「本轮没铺计划」的直接证据。

    clarification 透传给草案替身，供「出计划那一步判要问」的用例驱动那条分支。
    """

    def __init__(self, clarification: Any = None) -> None:
        self.calls = 0
        self._clarification = clarification

    async def __call__(self, _state: dict[str, Any], _llm: Any) -> _StubDraft:
        self.calls += 1
        return _StubDraft(clarification=self._clarification)


@dataclass(frozen=True)
class _LoopSettings:
    """最小配置对象：循环额度已取消，只保留计划单元数上限。"""

    rag_research_planner_max_units: int = 8


@dataclass(frozen=True)
class _StubTraceCtx:
    """trace summary 只读这几项，其余不构造。"""

    request_id: str = "req-1"
    session_id: str = "sess-1"
    model_calls: list[Any] = field(default_factory=list)
    evidence_observations: list[Any] = field(default_factory=list)
    evidence_capture_errors: list[Any] = field(default_factory=list)


def _clarify_request(
    *,
    needed: bool = True,
    question: str = _CLARIFY_TEXT,
    reason_code: str = _CLARIFY_REASON,
) -> TurnClarificationRequest:
    """造一份澄清请求。

    参数:
        needed: 本轮是否要先问用户
        question: 要问的那句话
        reason_code: 为什么非问不可
    返回:
        TurnClarificationRequest
    """
    return TurnClarificationRequest(needed=needed, question=question, reason_code=reason_code)


def _proposal(
    *,
    depth: str = _DEPTH_RESEARCH,
    requires_paper_evidence: bool = True,
    clarification: TurnClarificationRequest | None = None,
) -> TurnDepthProposal:
    """造一份模型深度偏好，供真实深度评估链跑通首次分析。

    参数:
        depth: 模型建议的处理深度
        requires_paper_evidence: 模型是否要求论文 Evidence
        clarification: 澄清请求；None 表示模型没提这件事
    返回:
        TurnDepthProposal
    """
    return TurnDepthProposal(
        depth=depth,
        reason_code=_DEPTH_REASON,
        requires_paper_evidence=requires_paper_evidence,
        resolved_query=_QUESTION,
        clarification=clarification,
    )


def _fresh_state() -> dict[str, Any]:
    """造一个深度未定的首轮状态：analyze 会真跑一次意图解析与深度评估。

    范围里放一篇论文：要论文 Evidence 的那几档才过得了「空范围不得要求证据」这条硬约束，
    同时一篇又构不成明确多论文比较，不会撞上第三条硬约束。
    """
    return {
        "user_query": _QUESTION,
        "structured_query": {"paper_ids": ["p1"]},
        "project_scope": {"paper_ids": ["p1"]},
    }


def _run_analyze(
    state: dict[str, Any],
    llm: Any,
    *,
    draft_clarification: Any = None,
) -> tuple[dict[str, Any], _DraftRecorder]:
    """跑一次 analyze 节点，同时记下草案是否被生成。

    参数:
        state: 交给节点的状态字典
        llm: 深度评估用的 LLM 替身
        draft_clarification: 草案替身要表达的澄清判断；缺省不澄清
    返回:
        (节点返回的状态增量, 草案调用记录器)
    """
    recorder = _DraftRecorder(clarification=draft_clarification)
    with patch.object(pipeline, "_analyze_plan", recorder):
        delta = asyncio.run(pipeline.analyze_planner_state(state, llm, _LoopSettings()))
    return delta, recorder


def _assess(proposal: TurnDepthProposal, *, paper_ids: tuple[str, ...] = ("p1",)) -> TurnDepthAssessment:
    """走真实深度评估链（含三条确定性硬约束）把模型偏好落成本轮判断。

    参数:
        proposal: 模型偏好
        paper_ids: 本轮授权论文
    返回:
        TurnDepthAssessment
    """
    llm = _StubStructuredLLM({TurnDepthProposal.__name__: proposal})
    return asyncio.run(
        assess_turn_depth({"user_query": _QUESTION}, llm, scoped_paper_ids=frozenset(paper_ids))
    )


def _clarify_state(*, depth: str = _DEPTH_RESEARCH, needs_clarification: bool = True) -> dict[str, Any]:
    """造一个已判过澄清的终态前状态，供写回与 trace 侧断言复用。

    参数:
        depth: 本轮深度，research 档用来验「写回被终态闸挡住而不是被深度闸挡住」
        needs_clarification: 本轮是否判了要问
    返回:
        写回与终态组装能直接吃的状态字典
    """
    return {
        "answer": _CLARIFY_TEXT,
        "turn_depth": depth,
        _DECISION_KEY: _TERMINATE_HONEST,
        _ASSESSMENT_KEY: {
            "depth": depth,
            "reason_code": _DEPTH_REASON,
            "requires_paper_evidence": False,
            "resolved_query": _QUESTION,
            _NEEDS_CLARIFY_KEY: needs_clarification,
            _CLARIFY_QUESTION_KEY: _CLARIFY_TEXT if needs_clarification else "",
            _CLARIFY_REASON_KEY: _CLARIFY_REASON,
        },
        "source_status": {"planner": _PLANNER_SKIPPED},
    }


class TestClarificationContract:
    """信任边界：判要问就必须带齐问题与原因码。"""

    @pytest.mark.parametrize(
        ("question", "reason_code"),
        [("", _CLARIFY_REASON), (_CLARIFY_TEXT, ""), ("", "")],
        ids=("missing_question", "missing_reason_code", "missing_both"),
    )
    def test_asking_without_question_or_reason_is_rejected(
        self, question: str, reason_code: str
    ) -> None:
        """判要问却缺问题或缺原因码，在信任边界就被拒。

        缺问题这一轮没有任何可返回给用户的内容；缺原因码则 trace 上查不出为什么问，
        澄清于是成了一个无从追查的偷懒默认值。两者都不许在下游补默认值兜圆。
        """
        with pytest.raises(ValidationError):
            TurnClarificationRequest(needed=True, question=question, reason_code=reason_code)

    def test_not_asking_needs_nothing_else(self) -> None:
        """判不用问是正常结论：不强求问题文本，也不算越界。"""
        request = TurnClarificationRequest(needed=False)

        assert request.needed is False
        assert request.question == ""

    def test_proposal_defaults_to_no_clarification(self) -> None:
        """模型没提澄清这件事时字段为 None，按不澄清处理，不是缺字段报错。"""
        assert _proposal().clarification is None


class TestAssessmentMapping:
    """深度评估结果里的澄清三项：什么时候带问题，什么时候只留原因码。"""

    def test_clarification_lands_in_assessment(self) -> None:
        """判要问：三项齐落，原因码与深度原因码分开记。"""
        assessment = _assess(_proposal(clarification=_clarify_request()))

        assert assessment[_NEEDS_CLARIFY_KEY] is True
        assert assessment[_CLARIFY_QUESTION_KEY] == _CLARIFY_TEXT
        assert assessment[_CLARIFY_REASON_KEY] == _CLARIFY_REASON
        assert assessment["reason_code"] == _DEPTH_REASON

    def test_not_needed_keeps_reason_code_but_drops_question(self) -> None:
        """判不用问：原因码照样留下，问题文本不带出去。

        原因码留着，trace 才分得清「判过、不用问」和「根本没判过要不要问」；
        问题文本必须丢掉，否则模型顺手写的一句话会被下游当成本轮回答返回给用户。
        """
        assessment = _assess(
            _proposal(
                clarification=TurnClarificationRequest(
                    needed=False, question=_CLARIFY_TEXT, reason_code="goal_is_explicit"
                )
            )
        )

        assert assessment[_NEEDS_CLARIFY_KEY] is False
        assert assessment[_CLARIFY_QUESTION_KEY] == ""
        assert assessment[_CLARIFY_REASON_KEY] == "goal_is_explicit"

    def test_missing_clarification_maps_to_empty(self) -> None:
        """模型压根没判这件事：不澄清、无问题、原因码空，不补默认值。"""
        assessment = _assess(_proposal())

        assert assessment[_NEEDS_CLARIFY_KEY] is False
        assert assessment[_CLARIFY_QUESTION_KEY] == ""
        assert assessment[_CLARIFY_REASON_KEY] == ""


class TestClarificationPrompt:
    """提示词里的产品判定标准：正反两侧都要在，缺哪一侧澄清就走偏。"""

    @pytest.mark.parametrize(
        "wording",
        [
            # 反面：判得准就不许问 —— 缺这句，澄清会变成模型拿不准时的安全默认值
            "you must not ask",
            # 正面：prd.md:100 的判据，不同理解会导致明显不同动作才值得问
            "would lead to visibly different work",
            # 澄清不是「证据可能不够」的出口：那种回合该先查、查完报缺口
            "Do not ask because the evidence may turn out to be thin",
            # 上一轮已经问过就别再问，免得用户被反复追问、这一轮永远开不了工
            "already asked a clarifying question on the previous turn",
            # AC :366 后半：用户不需要理解内部组件，问句里不许出现它们
            "never mention depths, retrieval",
            # 原因码是硬要求，提示词也得说
            "clarification.reason_code",
        ],
        ids=(
            "must_not_ask_when_determinable",
            "different_readings_different_actions",
            "not_an_exit_for_thin_evidence",
            "no_double_ask",
            "no_internal_component_names",
            "reason_code_required",
        ),
    )
    def test_prompt_carries_clarification_rules(self, wording: str) -> None:
        """澄清的判定标准必须逐条在提示词里，这段就是这件事的产品口径本身。"""
        prompt = build_depth_prompt({"user_query": _QUESTION}, frozenset({"p1"}))

        assert wording in prompt

    def test_prompt_keeps_depth_fields_required_while_asking(self) -> None:
        """澄清时深度三项照填：硬约束仍然管着它们，不给「问就不必判」的口子。"""
        prompt = build_depth_prompt({"user_query": _QUESTION}, frozenset({"p1"}))

        assert "Keep filling depth, requires_paper_evidence and resolved_query as usual" in prompt


class TestAnalyzeClarificationBranch:
    """analyze 侧：澄清回合到此为止，那句问题就是本轮回答。"""

    def test_clarification_returns_question_as_answer(self) -> None:
        """判要问：不铺计划，问句原样作为回答，决策是正常终止而不是失败。"""
        llm = _StubStructuredLLM({
            TurnDepthProposal.__name__: _proposal(clarification=_clarify_request())
        })

        delta, recorder = _run_analyze(_fresh_state(), llm)

        assert recorder.calls == 0
        assert _DRAFT_KEY not in delta
        assert delta["answer"] == _CLARIFY_TEXT
        assert delta[_DECISION_KEY] == _TERMINATE_HONEST
        assert delta[_DECISION_KEY] != _TERMINATE_FAIL
        # 澄清不是错误：错误码留空，否则前端与写回都会把这一轮读成出了故障
        assert delta["error_code"] == ""
        assert delta["error_summary"] == ""

    def test_clarification_reports_no_paper_retrieval(self) -> None:
        """本轮一次检索都没发起：planner 落 skipped，引用相关键全空。

        skipped 会被四态换算表读成 retrieve_paper_evidence: skipped；写 analyzed 是谎报
        产出了草案，而整条不投状态位会让评测把「没跑过」当成「跑过且为空」。
        """
        llm = _StubStructuredLLM({
            TurnDepthProposal.__name__: _proposal(clarification=_clarify_request())
        })

        delta, _ = _run_analyze(_fresh_state(), llm)

        assert delta["source_status"]["planner"] == _PLANNER_SKIPPED
        assert delta["source_status"]["planner"] != _PLANNER_ANALYZED
        assert delta["source_status"]["local_evidence_execution"] == "not_run"
        assert delta["research_plan"] is None
        assert delta["planner_final_blocks"] == []
        assert delta["evidence_records"] == []

    def test_clarification_still_advances_budget_and_keeps_assessment(self) -> None:
        """步数照记、判断照留：这一步真判过深度，原因码得留在状态里供 trace 取数。"""
        llm = _StubStructuredLLM({
            TurnDepthProposal.__name__: _proposal(clarification=_clarify_request())
        })

        delta, _ = _run_analyze(_fresh_state(), llm)

        assert delta[_BUDGET_KEY]["total_steps"] == 1
        assert [step["stage"] for step in delta[_TRACE_KEY]] == ["analyze"]
        assert delta[_ASSESSMENT_KEY][_CLARIFY_REASON_KEY] == _CLARIFY_REASON
        assert delta[_ASSESSMENT_KEY][_NEEDS_CLARIFY_KEY] is True

    @pytest.mark.parametrize(
        ("depth", "requires_paper_evidence"),
        [(_DEPTH_MINIMAL, False), (_DEPTH_LOOKUP, True), (_DEPTH_RESEARCH, True)],
        ids=("beats_minimal_direct_answer", "beats_lookup_draft", "beats_research_draft"),
    )
    def test_clarification_precedes_every_other_branch(
        self, depth: str, requires_paper_evidence: bool
    ) -> None:
        """三档深度下澄清都排在最前：既不产草案，也不落 minimal 直答短路。

        排在后面就等于先按某一种理解把活干完（直答给一段正文、或铺计划发检索），
        再回头问用户是不是这个意思 —— 用户看到的是一次跑偏的回答加一个问题。
        """
        llm = _StubStructuredLLM({
            TurnDepthProposal.__name__: _proposal(
                depth=depth,
                requires_paper_evidence=requires_paper_evidence,
                clarification=_clarify_request(),
            )
        })

        delta, recorder = _run_analyze(_fresh_state(), llm)

        assert recorder.calls == 0
        assert _DRAFT_KEY not in delta
        assert delta["answer"] == _CLARIFY_TEXT
        assert delta[_DECISION_KEY] == _TERMINATE_HONEST

    def test_no_clarification_keeps_drafting(self) -> None:
        """判不用问：照旧产草案，澄清这条分支不许顺手拦下正常回合。"""
        llm = _StubStructuredLLM({
            TurnDepthProposal.__name__: _proposal(
                clarification=_clarify_request(needed=False, question="", reason_code="goal_is_explicit")
            )
        })

        delta, recorder = _run_analyze(_fresh_state(), llm)

        assert recorder.calls == 1
        assert _DRAFT_KEY in delta
        assert delta["source_status"]["planner"] == _PLANNER_ANALYZED
        assert "answer" not in delta

    def test_replan_reentry_does_not_reask(self) -> None:
        """replan 回边重入 analyze 不重判澄清：深度已定，首次分析整段早返回。

        重判会让一次重规划突然变成向用户提问，用户上一轮的问题就此丢在半路。
        """
        state = {
            **_fresh_state(),
            "turn_depth": _DEPTH_RESEARCH,
            _ASSESSMENT_KEY: {"depth": _DEPTH_RESEARCH, "requires_paper_evidence": True},
        }

        delta, recorder = _run_analyze(state, _StubStructuredLLM({}))

        assert recorder.calls == 1
        assert _DRAFT_KEY in delta
        assert delta[_DECISION_KEY] == ""


class TestClarificationRouting:
    """路由侧：澄清回合直奔终态，既有两条边不受影响。"""

    def test_analyze_terminate_honest_goes_straight_to_write_back(self) -> None:
        """analyze 标 terminate_honest（澄清）→ write_back。

        进 synthesize 会被那边第一支诚实收尾抢走，把「先问你一句」改写成
        「在你指定的 N 篇论文里检索过了，没找到」——一句关于检索的假话。
        """
        state = {"turn_depth": _DEPTH_RESEARCH, _DECISION_KEY: _TERMINATE_HONEST}

        assert rag_node.route_after_planner_analyze(state) == "write_back"

    def test_minimal_clarification_is_not_stolen_by_short_circuit(self) -> None:
        """minimal 档的澄清同样直奔终态：判定序里终态必须先于 minimal 短路。"""
        state = {"turn_depth": _DEPTH_MINIMAL, _DECISION_KEY: _TERMINATE_HONEST}

        assert rag_node.route_after_planner_analyze(state) == "write_back"

    def test_terminate_fail_still_goes_to_write_back(self) -> None:
        """硬失败那条边照旧（回归）。"""
        state = {"turn_depth": _DEPTH_MINIMAL, _DECISION_KEY: _TERMINATE_FAIL}

        assert rag_node.route_after_planner_analyze(state) == "write_back"

    def test_execute_stage_honest_termination_still_synthesizes(self) -> None:
        """execute 之后的 terminate_honest 仍进 synthesize（回归）。

        那一支是「范围内查过但没找到」的模板收尾，本单元不许把它一起改道，
        否则证据不足的回合会失去用户可读的收尾正文。
        """
        state = {"turn_depth": _DEPTH_LOOKUP, _DECISION_KEY: _TERMINATE_HONEST}

        assert rag_node.route_after_planner_execute(state) == "planner_synthesize"

    def test_no_decision_keeps_default_success_edge(self) -> None:
        """没有决策时照旧进 validate：新分支不许改默认边（回归）。"""
        assert rag_node.route_after_planner_analyze({"turn_depth": _DEPTH_LOOKUP}) == "planner_validate"


class TestClarificationCompliance:
    """合规：澄清回合不产记忆候选，且这一轮为什么问得留在 trace 里。"""

    def test_clarification_turn_proposes_no_memory_candidate(self) -> None:
        """深度判到 research 也不产候选：终态闸挡在模型提议之前。

        prd.md §7.2 的「固定不触发」清单里含澄清。这里刻意用 research 档 —— 深度闸对它
        是放行的，所以它验的正是终态闸真挡住了；换成 minimal 只会验到深度闸，
        澄清那一档一旦被判成 research 就漏了。
        """
        async def _never(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("澄清回合不该进候选评估")

        with patch.object(writeback, "_write_back_research_turn", _never):
            result = asyncio.run(
                writeback.write_back_turn_memory(_clarify_state(), request_id="req-1")
            )

        assert result["status"] == "skipped"
        assert result["candidate_ids"] == []
        assert result["reason_codes"] == [f"skipped_terminal_{_TERMINATE_HONEST}"]

    def test_turn_outcome_records_clarification_capability(self) -> None:
        """四态能力表里澄清投 success，论文取证投 skipped，终止原因是正常终止。

        不投澄清这一位，这一轮的能力表只剩三条 skipped，看不出澄清这个能力跑过 ——
        AC :366 要的正是「澄清表现为 Planner 可调用能力」。
        """
        state = _clarify_state()
        outcome = writeback.build_turn_outcome(
            state, memory_writeback={"status": "skipped", "reason_codes": []}
        )

        assert outcome["capability_results"][_CAPABILITY_CLARIFY] == "success"
        assert outcome["capability_results"]["retrieve_paper_evidence"] == "skipped"
        assert outcome["termination_reason"] == _TERMINATE_HONEST
        assert outcome["depth_reason_code"] == _DEPTH_REASON

    def test_judged_not_needed_shows_up_as_skipped(self) -> None:
        """判过不用问：澄清这一位落 skipped，而不是整条从能力表消失。

        整条消失会让评测把「判过、不用问」读成「压根没判过要不要问」。
        """
        outcome = writeback.build_turn_outcome(
            _clarify_state(needs_clarification=False),
            memory_writeback={"status": "skipped", "reason_codes": []},
        )

        assert outcome["capability_results"][_CAPABILITY_CLARIFY] == "skipped"

    def test_clarification_reason_code_reaches_trace_summary(self) -> None:
        """为什么问这一句要能在 trace summary 里查到，不只活在内存状态里。"""
        state = _clarify_state()
        summary = build_trace_summary_payload(
            _StubTraceCtx(), {"turn_depth_assessment": state[_ASSESSMENT_KEY]}
        )

        assert summary["turn_depth_assessment"][_CLARIFY_REASON_KEY] == _CLARIFY_REASON
        assert summary["turn_depth_assessment"][_NEEDS_CLARIFY_KEY] is True










class TestDraftClarificationExit:
    """出计划那一步的澄清出口：判出「得先问用户」时真能问出口。

    首次分析那个入口发生在探测与规划之前，计划铺起来之后才判出的关键缺失信息
    没有别的出路 —— 此前只能交一份空计划（契约拒绝）或硬塞一个无关单元
    （等于先按某种理解把活干完）。这一组用例盯住这条出口的四个会静默失效处。
    """

    def test_draft_clarification_returns_question_as_answer(self) -> None:
        """草案判要问：问句作为本轮回答，不产草案、不发检索、正常终止。"""
        state = {**_fresh_state(), "turn_depth": _DEPTH_RESEARCH}
        delta, recorder = _run_analyze(
            state,
            _StubStructuredLLM({}),
            draft_clarification=_clarify_request(),
        )

        assert recorder.calls == 1
        assert _DRAFT_KEY not in delta
        assert delta["answer"] == _CLARIFY_TEXT
        assert delta[_DECISION_KEY] == _TERMINATE_HONEST
        assert delta["error_code"] == ""
        assert delta["source_status"]["planner"] == _PLANNER_SKIPPED
        assert delta["research_plan"] is None
        assert delta["evidence_records"] == []

    def test_draft_not_asking_keeps_the_draft(self) -> None:
        """草案判不用问：照旧产草案，这条分支不许拦下正常回合。"""
        state = {**_fresh_state(), "turn_depth": _DEPTH_RESEARCH}
        delta, recorder = _run_analyze(
            state,
            _StubStructuredLLM({}),
            draft_clarification=_clarify_request(needed=False, question="", reason_code="goal_is_explicit"),
        )

        assert recorder.calls == 1
        assert _DRAFT_KEY in delta
        assert delta["source_status"]["planner"] == _PLANNER_ANALYZED
        assert "answer" not in delta

    def test_missing_draft_clarification_is_not_a_question(self) -> None:
        """模型没提澄清这件事：按不澄清处理，不许拿缺字段当「要问」。"""
        state = {**_fresh_state(), "turn_depth": _DEPTH_RESEARCH}
        delta, _ = _run_analyze(state, _StubStructuredLLM({}))

        assert _DRAFT_KEY in delta
        assert "answer" not in delta

    def test_draft_requires_units_unless_it_asks_the_user(self) -> None:
        """契约层：不澄清就必须有单元；判要问才允许空单元。"""
        with pytest.raises(ValidationError):
            ResearchPlanDraft.model_validate({
                "question": _QUESTION,
                "plan_type": "single_unit",
                "units": [],
            })

        asking = ResearchPlanDraft.model_validate({
            "question": _QUESTION,
            "plan_type": "single_unit",
            "units": [],
            "clarification": {
                "needed": True,
                "question": _CLARIFY_TEXT,
                "reason_code": _CLARIFY_REASON,
            },
        })

        assert asking.units == []
        assert asking.clarification is not None and asking.clarification.needed

    def test_clarification_needing_nothing_is_rejected_like_the_other_entry(self) -> None:
        """与首次分析那个入口同一份契约：判要问却缺问题或缺原因码在信任边界就拒。"""
        with pytest.raises(ValidationError):
            ResearchPlanDraft.model_validate({
                "question": _QUESTION,
                "plan_type": "single_unit",
                "units": [],
                "clarification": {"needed": True, "question": "", "reason_code": ""},
            })

    def test_plan_prompt_describes_the_draft_clarification_exit(self) -> None:
        """出计划提示词里必须把澄清出口写出来，不能只靠契约字段名暗示。

        探满告知里那句「也可以问用户一句」已随循环额度一起删掉：额度没了，就没有
        「探满」这个时刻可指。留下的这条规则句才是模型判要问时的唯一依据，
        它若丢了，模型只能交空计划（被契约拒）或硬塞无关单元。
        """
        state = {**_fresh_state(), "turn_depth": _DEPTH_RESEARCH}
        prompt = plan_prompt.build_planner_prompt(state)

        assert "clarification field" in prompt
        assert "empty units array" in prompt
        assert "Do not use clarification for a mere missing paper detail" in prompt
