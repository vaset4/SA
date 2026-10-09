"""minimal 深度短路：analyze 不铺计划、synthesize 直答收口（单元 V1 批 2）。

作者：xxx
创建时间：2026-08-31 10:00:00

判为 minimal 的回合不需要论文事实，整条链应短成 analyze → synthesize。本文件盯住
这条短路上四件会静默失效的事：

1. analyze 侧不能产 planner_draft：路由据「深度 minimal 且无草案」认短路，产了草案
   照旧被送进 validate 与 execute，短路等于没发生，而且没有任何异常或日志。
2. 短路只对「不要论文 Evidence」的回合成立：要出现论文事实主张的回合必须升到
   lookup/research 去真取证据，不能以 minimal 直答蒙一个看着像事实的答案。
3. 短路了步数照记：analyze 这一步真判过深度、也可能真读过记忆，漏记会让 total_steps
   这道兜底上界少算一步。
4. 直答不能兜底成模板正文：模型没出正文就得结构化失败，否则模型故障被伪装成一次
   正常回答留在会话里，用户与评测都看不出这轮其实没答上。
5. 本轮授权论文数要真传到直答（单元 V5）：空项目问论文事实时得说清缺可验证材料，
   不传论文数就只能靠模型自觉，还会把用户指回一个空书架。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from app.models.research_contracts import TurnDepthAssessment
from app.models.research_planner_contracts import DirectAnswerDraft, TurnDepthProposal
from app.services.internal_research import research_planner_pipeline as pipeline

# ─── 短路两端的键名与取值：这些字面量就是跨文件约定本身 ───

# 三档处理深度
_DEPTH_MINIMAL = "minimal"
_DEPTH_LOOKUP = "lookup"
_DEPTH_RESEARCH = "research"
# 各深度的稳定原因码，按深度取，免得样例出现深度与原因码互相矛盾的组合
_DEPTH_REASON_CODES = {
    _DEPTH_MINIMAL: "no_paper_fact_needed",
    _DEPTH_LOOKUP: "single_paper_metadata_lookup",
    _DEPTH_RESEARCH: "cross_paper_comparison",
}

# 路由认短路靠这个键不存在
_DRAFT_KEY = "planner_draft"
# 短路增量与直答载荷上 planner 的取值：本轮没铺计划也没取论文证据
_PLANNER_SKIPPED = "skipped"
# 正常产出草案时 planner 的取值，短路增量里不该出现
_PLANNER_ANALYZED = "analyzed"
# 循环预算与留痕在状态里的键名
_BUDGET_KEY = "planner_budget"
_TRACE_KEY = "planner_recovery_trace"
_DECISION_KEY = "planner_loop_decision"
# 硬失败与诚实收尾的对外契约
_TERMINATE_FAIL = "terminate_fail"
_TERMINATE_HONEST = "terminate_honest"
_EXECUTION_FAILED_CODE = "planner_execution_failed"
_EVIDENCE_NOT_FOUND_CODE = "evidence_not_found_in_scope"
# 直答自己的失败码：这一轮一次检索都没发起，不能借用表示「取证已开始」的执行失败码
_DIRECT_ANSWER_FAILED_CODE = "minimal_direct_answer_failed"

# 记忆读取四态里的成功态与失败态
_READ_SUCCESS = "success"
_READ_FAILED = "failed"
# 记忆段段首标记，用来验记忆守卫在直答这一侧同样生效
_MEMORY_BLOCK_MARKER = "【项目研究记忆】"

_QUESTION = "我们之前定的研究方向还算数吗"
_ANSWER_TEXT = "按项目此前的判断，这个方向仍然成立，只是把成熟度往前推了一档。"

# 直答两档口径在 prompt 里的判据（单元 V5）：项目里有没有论文，说法必须不一样
_MISSING_MATERIAL_WORDING = "缺少可验证材料"
_GO_CHECK_PAPERS_WORDING = "去查项目里的论文"


class _StubStructuredLLM:
    """按 schema 名返回预置结构化对象的 LLM 替身，做法照 test_planner_memory_wiring。"""

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


class _StubDirectAnswerLLM:
    """直答用的结构化 LLM 替身：按契约造草案，记下收到的 prompt。

    正文为空时照真实解析路径那样被契约挡下（`DirectAnswerDraft` 的 `min_length=1`），
    而不是让替身自己抛错——要测的是「空正文进不了正常回答」这件事本身。
    来源说明那一档由 test_research_planner_direct_answer 专门覆盖，这里只关心走没走对分支。
    """

    def __init__(self, content: str = _ANSWER_TEXT, error: Exception | None = None,
                 knowledge_bearing: bool = False) -> None:
        self._content = content
        self._error = error
        self._knowledge_bearing = knowledge_bearing
        self.prompts: list[str] = []

    def with_structured_output(self, _schema: type) -> "_StubDirectAnswerLLM":
        return self

    async def ainvoke(self, messages: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        if self._error is not None:
            raise self._error
        return DirectAnswerDraft(answer=self._content, knowledge_bearing=self._knowledge_bearing)


@dataclass(frozen=True)
class _StubDraft:
    """草案替身：只需能 model_dump，本文件只关心草案有没有被产出来。

    clarification 对齐 ResearchPlanDraft 契约：analyze 产完草案要读它判断这一轮
    是不是「先问用户」。替身少了这个字段，用例会红在 AttributeError 上，那不是被测行为。
    """

    clarification: Any = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"question": _QUESTION}


@dataclass(frozen=True)
class _LoopSettings:
    """最小配置对象：循环额度已取消，只保留计划单元数上限。"""

    rag_research_planner_max_units: int = 8


class _DraftRecorder:
    """记下 _analyze_plan 的调用次数，返回草案替身。

    「一次都没被调用」就是「本轮没生成正式 Research Plan」的直接证据：光看返回增量里
    没有 planner_draft，分不出是压根没铺计划，还是铺完又被丢掉。
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, _state: dict[str, Any], _llm: Any) -> _StubDraft:
        self.calls += 1
        return _StubDraft()


def _assessment(*, depth: str, requires_paper_evidence: bool) -> TurnDepthAssessment:
    """造一份本轮深度评估结果，键名走契约声明而不是手写字面量。

    参数:
        depth: 本轮处理深度
        requires_paper_evidence: 本轮是否要求论文 Evidence
    返回:
        TurnDepthAssessment 形状的字典
    """
    return TurnDepthAssessment(
        depth=depth,
        reason_code=_DEPTH_REASON_CODES[depth],
        requires_paper_evidence=requires_paper_evidence,
        resolved_query=_QUESTION,
    )


def _proposal(*, depth: str, requires_paper_evidence: bool) -> TurnDepthProposal:
    """造一份模型深度偏好，供真实深度评估链跑通首次分析。

    参数:
        depth: 模型建议的处理深度
        requires_paper_evidence: 模型是否要求论文 Evidence
    返回:
        TurnDepthProposal
    """
    return TurnDepthProposal(
        depth=depth,
        reason_code=_DEPTH_REASON_CODES[depth],
        requires_paper_evidence=requires_paper_evidence,
        resolved_query=_QUESTION,
    )


def _resolved_state(*, depth: str, requires_paper_evidence: bool) -> dict[str, Any]:
    """造一个深度已定的状态：首次分析据 turn_depth 早返回，不再调深度评估。

    参数:
        depth: 已定的本轮处理深度
        requires_paper_evidence: 本轮是否要求论文 Evidence
    返回:
        可直接交给 analyze_planner_state 的状态字典
    """
    return {
        "user_query": _QUESTION,
        "structured_query": {"paper_ids": []},
        "turn_depth": depth,
        "turn_depth_assessment": _assessment(
            depth=depth, requires_paper_evidence=requires_paper_evidence
        ),
    }


def _run_analyze(state: dict[str, Any], llm: Any) -> tuple[dict[str, Any], _DraftRecorder]:
    """跑一次 analyze 节点，同时记下草案是否被生成。

    参数:
        state: 交给节点的状态字典
        llm: 深度评估用的 LLM 替身
    返回:
        (节点返回的状态增量, 草案调用记录器)
    """
    recorder = _DraftRecorder()
    with patch.object(pipeline, "_analyze_plan", recorder):
        delta = asyncio.run(pipeline.analyze_planner_state(state, llm, _LoopSettings()))
    return delta, recorder


def _memory_result(status: str) -> dict[str, Any]:
    """造一份记忆读取结果，success 档带一条真实条目。

    参数:
        status: 记忆读取四态之一
    返回:
        memory_read_result 形状的字典
    """
    if status != _READ_SUCCESS:
        return {"status": status, "items": [], "error_kind": "RuntimeError"}
    return {
        "status": status,
        "items": [
            {
                "memory_id": "mem-1",
                "title": "先做召回再谈重排",
                "body": "上轮定的方向：先把召回补齐。",
                "memory_type": "decision_note",
                "maturity": "verified",
            }
        ],
    }


class TestAnalyzeShortCircuit:
    """analyze 侧：什么样的回合才短路，短路时留下什么。"""

    def test_minimal_without_paper_evidence_skips_draft_entirely(self) -> None:
        """minimal + 不要论文证据：整条草案生成一次都没发生，增量里也没有草案。

        这一条盯的是短路的事实判据本身：路由认「深度 minimal 且无 planner_draft」，
        所以增量里多出一个草案键，这一轮就照旧走 validate 与 execute。
        """
        llm = _StubStructuredLLM({
            TurnDepthProposal.__name__: _proposal(
                depth=_DEPTH_MINIMAL, requires_paper_evidence=False
            )
        })
        state = {"user_query": _QUESTION, "structured_query": {"paper_ids": []}}

        delta, recorder = _run_analyze(state, llm)

        assert recorder.calls == 0
        assert _DRAFT_KEY not in delta
        assert delta["turn_depth"] == _DEPTH_MINIMAL
        # planner 落 skipped 而非 analyzed：本轮压根没产出草案，写 analyzed 是谎报
        assert delta["source_status"]["planner"] == _PLANNER_SKIPPED
        assert delta["source_status"]["planner"] != _PLANNER_ANALYZED
        assert delta["runtime_chain"] == "planner"

    def test_short_circuit_still_advances_budget_and_leaves_trace(self) -> None:
        """短路了步数照记：analyze 真判过深度，不能因为跳过计划就不留痕。"""
        state = _resolved_state(depth=_DEPTH_MINIMAL, requires_paper_evidence=False)

        delta, _ = _run_analyze(state, _StubStructuredLLM({}))

        assert delta[_BUDGET_KEY]["total_steps"] == 1
        assert [step["stage"] for step in delta[_TRACE_KEY]] == ["analyze"]
        # 纯思考步清空决策：留着上一轮的陈旧决策会误导 analyze 之后的路由选边
        assert delta[_DECISION_KEY] == ""

    def test_minimal_requiring_paper_evidence_still_drafts(self) -> None:
        """minimal 但本轮要论文 Evidence：不短路，照旧产草案。

        答案里要出现论文事实主张的回合得去真取证据（升 lookup/research 或如实报缺口），
        以 minimal 直答等于凭印象编一个看着像事实的答案。
        """
        state = _resolved_state(depth=_DEPTH_MINIMAL, requires_paper_evidence=True)

        delta, recorder = _run_analyze(state, _StubStructuredLLM({}))

        assert recorder.calls == 1
        assert _DRAFT_KEY in delta
        assert delta["source_status"]["planner"] == _PLANNER_ANALYZED

    @pytest.mark.parametrize("depth", [_DEPTH_LOOKUP, _DEPTH_RESEARCH])
    def test_deeper_depths_never_short_circuit(self, depth: str) -> None:
        """lookup / research 照旧铺计划：短路只属于不需要论文事实的那一档。"""
        state = _resolved_state(depth=depth, requires_paper_evidence=True)

        delta, recorder = _run_analyze(state, _StubStructuredLLM({}))

        assert recorder.calls == 1
        assert _DRAFT_KEY in delta


class TestSynthesizeDirectAnswer:
    """synthesize 侧：无计划的 minimal 回合怎么收口，三条分支谁先判。"""

    def test_missing_plan_at_minimal_answers_directly(self) -> None:
        """无 research_plan + minimal：直答收口，载荷不带计划也不带引用。

        常规综合那一支缺计划会直接抛 ValueError，所以这一支必须自己认出短路来的回合。
        """
        llm = _StubDirectAnswerLLM()
        state = _resolved_state(depth=_DEPTH_MINIMAL, requires_paper_evidence=False)

        payload = asyncio.run(pipeline.synthesize_planner_state(state, llm))

        assert payload["answer"] == _ANSWER_TEXT
        assert payload["source_status"]["planner"] == _PLANNER_SKIPPED
        assert payload["research_plan"] is None
        # 本轮没取任何论文证据，带引用的定稿块必须是空的
        assert payload["planner_final_blocks"] == []
        assert payload["evidence_records"] == []
        assert payload["error_code"] == ""

    @pytest.mark.parametrize(
        ("scope_paper_ids", "expect_missing_material"),
        [([], True), (["p1"], False)],
        ids=("empty_project_states_missing_material", "papers_in_scope_keeps_wording"),
    )
    def test_direct_answer_knows_the_turn_paper_count(
        self, scope_paper_ids: list[str], expect_missing_material: bool
    ) -> None:
        """本轮授权论文数从 state 传到直答：空项目那一档才说缺可验证材料。

        断这一条的理由：不传论文数时，空项目问论文事实说不说清缺材料全靠模型自觉，
        而项目里明明有论文时又不该说缺——两档口径只有在真拿到论文数时才分得开。
        """
        llm = _StubDirectAnswerLLM()
        state = {
            **_resolved_state(depth=_DEPTH_MINIMAL, requires_paper_evidence=False),
            "project_scope": {"paper_ids": scope_paper_ids},
        }

        asyncio.run(pipeline.synthesize_planner_state(state, llm))

        prompt = llm.prompts[0]
        assert (_MISSING_MATERIAL_WORDING in prompt) is expect_missing_material
        assert (_GO_CHECK_PAPERS_WORDING in prompt) is not expect_missing_material

    @pytest.mark.parametrize(
        ("status", "block_expected"),
        [(_READ_SUCCESS, True), (_READ_FAILED, False)],
        ids=("read_success_renders", "read_failed_stays_silent"),
    )
    def test_direct_answer_reuses_memory_guard(self, status: str, block_expected: bool) -> None:
        """记忆段守卫走同一份实现：只有真读到条目才进 prompt。

        直答这一侧另写一遍守卫就会漏掉 status/items 判断，模型于是拿着一个失败或空的
        读取结果编一段项目此前的判断，而这段话看上去与真读到的记忆一模一样。
        """
        llm = _StubDirectAnswerLLM()
        state = {
            **_resolved_state(depth=_DEPTH_MINIMAL, requires_paper_evidence=False),
            "memory_read_result": _memory_result(status),
        }

        asyncio.run(pipeline.synthesize_planner_state(state, llm))

        assert (_MEMORY_BLOCK_MARKER in llm.prompts[0]) is block_expected

    def test_honest_termination_keeps_precedence(self) -> None:
        """terminate_honest 仍走既有诚实收尾，没被无计划直答分支抢走。

        诚实收尾是「范围内查过但没找到」的正常终止路径，模板确定性组装、不调 LLM；
        被直答抢走就变成模型凭印象补一段正文，把查过没找到说成答上了。
        """
        llm = _StubDirectAnswerLLM()
        state = {
            **_resolved_state(depth=_DEPTH_MINIMAL, requires_paper_evidence=False),
            _DECISION_KEY: _TERMINATE_HONEST,
        }

        payload = asyncio.run(pipeline.synthesize_planner_state(state, llm))

        assert payload["error_code"] == _EVIDENCE_NOT_FOUND_CODE
        assert llm.prompts == []

    @pytest.mark.parametrize(
        ("content", "error"),
        [("", None), (_ANSWER_TEXT, RuntimeError("模型连接失败"))],
        ids=("empty_answer_text", "model_raises"),
    )
    def test_direct_answer_failure_is_structured_not_templated(
        self, content: str, error: Exception | None
    ) -> None:
        """模型没出正文就结构化失败：带 terminate_fail，不是一句模板正文。

        直答是这一轮唯一的出口，补一句「暂时无法回答」会把模型故障伪装成正常回答，
        写回与评测都读成这轮答上了。
        """
        llm = _StubDirectAnswerLLM(content=content, error=error)
        state = _resolved_state(depth=_DEPTH_MINIMAL, requires_paper_evidence=False)

        payload = asyncio.run(pipeline.synthesize_planner_state(state, llm))

        assert payload[_DECISION_KEY] == _TERMINATE_FAIL
        assert payload["error_code"] == _DIRECT_ANSWER_FAILED_CODE
        assert payload["error_summary"]
        assert payload["research_plan"] is None
        assert _ANSWER_TEXT not in payload["answer"]
        # 直答失败也不许虚报取证：这一轮压根没发起检索，local_evidence_execution 必须是 not_run
        assert payload["source_status"]["local_evidence_execution"] == "not_run"

    def test_missing_plan_at_lookup_is_not_answered_directly(self) -> None:
        """lookup 缺计划是真故障：结构化失败，不许借直答这一支糊过去。

        直答只对 minimal 成立。深度门要是漏了，取证失败的回合会退化成无证据直答，
        而载荷看上去是一次正常回答。
        """
        llm = _StubDirectAnswerLLM()
        state = _resolved_state(depth=_DEPTH_LOOKUP, requires_paper_evidence=True)

        payload = asyncio.run(pipeline.synthesize_planner_state(state, llm))

        assert payload[_DECISION_KEY] == _TERMINATE_FAIL
        assert payload["error_code"] == _EXECUTION_FAILED_CODE
        assert llm.prompts == []
