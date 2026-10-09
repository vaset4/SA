"""Planner 记忆接线的三处跨文件接缝测试（单元 2c）。

作者：xxx
创建时间：2026-08-30 11:30:00

记忆读取服务本体的合规性由姊妹文件 test_project_memory_read.py 钉住。本文件只管
三处接缝，它们全靠字段名逐字对齐，错一个字整条链静默失效，没有异常也没有日志：

1. 模型偏好 → 深度评估结果（turn_depth_assessor）：模型给的收窄键被改名或被无条件
   收进表，收窄值就到不了 SQL。记忆层的白名单按键名取值，取不到即视为「这一维不
   收窄」，于是读回一份范围完全不对的记忆，读取状态照样是 success。
2. 深度评估结果 → 按需读取（research_planner_pipeline）：needs_project_memory 读错
   会变成每轮都读或永远不读；memory_read_result 这个 State 键名拼错，记忆读到了也
   落不进状态；失败两分支的判据写错，「记忆是唯一来源」的回合会空着手继续，模型只能
   编出一份项目此前的判断。
3. 读到的记忆 → 草案 prompt（research_planner_plan_prompt）：状态键或成功态判错，
   记忆读到了却渲染不进 prompt；自己拼字符串绕过注入服务，会丢掉「不得作为引用来源」
   这句身份约束，模型于是把用户自己的判断当文献证据写进带引用的结论。

三处接缝的入口都是下划线私有函数（_to_assessment / _memory_narrowing /
_read_memory_delta / _memory_is_only_source / _project_memory_lines）。直接测私有函数
是刻意的：这些接缝没有公开入口覆盖得到（公开入口要么要真实 LLM、要么要真库），而它们
错了整条链静默失效，比「测试不碰私有函数」这条洁癖重要得多。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app.middleware.trace_builders import build_trace_summary_payload
from app.models.research_contracts import TurnDepthAssessment
from app.models.research_memory_contracts import VALID_MEMORY_TYPES, MemoryReadNarrowing
from app.models.research_planner_contracts import (
    ProjectMemoryReadRequest,
    TurnDepthProposal,
)
from app.services import research_memory_injector_service as injector
from app.services.internal_research import project_memory_reader as reader
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research import research_planner_plan_prompt as plan_prompt
from app.services.internal_research import turn_depth_assessor as depth_assessor


# ─── 接缝上的键名与取值：这些字面量就是跨文件约定本身，测试必须点名 ───

# 记忆读取结果落进 State 的键名：Planner 写它、草案 prompt 读它
_MEMORY_RESULT_STATE_KEY = "memory_read_result"
# 深度评估结果里承载记忆判断的三个键名
_NEEDS_MEMORY_KEY = "needs_project_memory"
_NARROWING_KEY = "memory_narrowing"
_MEMORY_REASON_KEY = "memory_read_reason_code"

# 收窄条件的四个键名
_KEY_PAPER_ID = "paper_id"
_KEY_QUESTION_ID = "question_id"
_KEY_MEMORY_TYPE = "memory_type"
_KEY_FOLLOW_LINKS = "follow_links"

# ProjectMemoryReadRequest 里不参与收窄的两个字段：一个是判定位、一个是原因码。
# 收窄键集合用「全部字段减去这两个」反推，日后新增收窄维度本文件自动跟着测到，
# 而手写字面量列表会随产品改名一起改，改完就等于没测
_REQUEST_NON_NARROWING_FIELDS = frozenset({"needed", "reason_code"})

# 记忆读取的终态取值；skipped 不该出现在 Planner 这条按需读取路径上
_READ_SUCCESS = "success"
_READ_EMPTY = "empty"
_READ_FAILED = "failed"
_READ_SKIPPED = "skipped"

# 三档处理深度
_DEPTH_MINIMAL = "minimal"
_DEPTH_LOOKUP = "lookup"
_DEPTH_RESEARCH = "research"
# 各深度的稳定原因码，按深度取，免得样例里出现深度与原因码互相矛盾的组合
_DEPTH_REASON_CODES = {
    _DEPTH_MINIMAL: "no_paper_fact_needed",
    _DEPTH_LOOKUP: "single_paper_metadata_lookup",
    _DEPTH_RESEARCH: "cross_paper_comparison",
}

# 失败载荷上的对外契约：稳定错误码、可见状态位与循环决策
_MEMORY_READ_FAILED_CODE = "memory_read_failed"
_DEPTH_ASSESSMENT_FAILED_CODE = "planner_depth_assessment_failed"
_MEMORY_READ_STATUS_KEY = "memory_read"
_DEPTH_ASSESSMENT_STATUS_KEY = "depth_assessment"
_TERMINATE_FAIL = "terminate_fail"
_DRAFT_KEY = "planner_draft"

# 记忆存储不可用时的失败类型，要一路透传到 MemoryOnlyTurnReadError.error_kind
_ERROR_KIND = "RuntimeError"

# 读取身份与项目归属都从 project_scope 取
_PROJECT_SCOPE = {"user_id": "wiring_owner", "project_id": "proj-wiring"}

# 模型把四个收窄键全给满的样例，用来验一个都不许在路上掉
_FULL_NARROWING = {
    _KEY_PAPER_ID: "paper-1",
    _KEY_QUESTION_ID: "rq-1",
    _KEY_MEMORY_TYPE: "decision_note",
    _KEY_FOLLOW_LINKS: True,
}

# 记忆段的身份约束句：记忆是用户自己的判断，不是可引用的文献证据
_IDENTITY_CONSTRAINT = "不得作为引用来源"
# 记忆段段首标记，用来验非成功态时 prompt 里连一行噪音都没有
_MEMORY_BLOCK_MARKER = "【项目研究记忆】"
# 替换注入服务渲染函数时的返回值，用来验落进 prompt 的就是它的返回值
_RENDER_SENTINEL = "<记忆段由注入服务渲染>"

class _StubStructuredLLM:
    """按 schema 名返回预置结构化对象的 LLM 替身，做法照 test_turn_depth_assessment。"""

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


class _ReadRecorder:
    """记下 read_project_memory 每次调用的实参，并返回预置读取结果。

    两个形参声明为仅限关键字：产品代码若改成按位置传，这里直接 TypeError 而不是让
    context 与 query_narrowing 悄悄错位——那种错位会读到别人的记忆或读错范围。
    """

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *, context: Any, query_narrowing: Any) -> dict[str, Any]:
        self.calls.append({"context": context, "query_narrowing": query_narrowing})
        return self._result


@dataclass(frozen=True)
class _LoopSettings:
    """最小配置对象：循环额度已取消，只保留计划单元数上限。"""

    rag_research_planner_max_units: int = 8


def _narrowing_fields() -> frozenset[str]:
    """反射取出模型能给的收窄键集合。

    返回:
        ProjectMemoryReadRequest 的字段集减去判定位与原因码后剩下的收窄键
    """
    return frozenset(ProjectMemoryReadRequest.model_fields) - _REQUEST_NON_NARROWING_FIELDS


def _proposal(
    *,
    depth: str = _DEPTH_LOOKUP,
    requires_paper_evidence: bool = True,
    memory: dict[str, Any] | None = None,
) -> TurnDepthProposal:
    """造一份模型深度偏好。

    参数:
        depth: 本轮处理深度
        requires_paper_evidence: 本轮是否要求论文 Evidence
        memory: 记忆读取请求的字段；None 表示模型完全没提要不要翻记忆
    返回:
        TurnDepthProposal，字段取值已过信任边界校验
    """
    return TurnDepthProposal(
        depth=depth,
        reason_code=_DEPTH_REASON_CODES[depth],
        requires_paper_evidence=requires_paper_evidence,
        resolved_query="我们之前定的研究方向是什么",
        project_memory=None if memory is None else ProjectMemoryReadRequest(**memory),
    )


def _assessment(**proposal_kwargs: Any) -> TurnDepthAssessment:
    """走真实映射函数产出深度评估结果，不手搓 dict。

    三个记忆键的名字正是被验的接缝：手搓 dict 等于把两端的键名各写一遍，产品改名时
    测试跟着改，接缝就再也没人盯着了。

    参数:
        proposal_kwargs: 透传给 _proposal 的深度与记忆请求取值
    返回:
        TurnDepthAssessment，由 turn_depth_assessor 真实落值
    """
    return depth_assessor._to_assessment(_proposal(**proposal_kwargs))


def _memory_only_assessment() -> TurnDepthAssessment:
    """造「项目记忆是本轮唯一信息来源」的深度评估结果。

    返回:
        minimal 深度 + 不要求论文证据 + 判定要翻记忆的评估结果
    """
    return _assessment(
        depth=_DEPTH_MINIMAL,
        requires_paper_evidence=False,
        memory={"needed": True, "reason_code": "recall_prior_decision"},
    )


def _read_result(status: str, *, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """造一份记忆读取结果。

    参数:
        status: 读取终态
        items: 读到的记忆条目，缺省为空列表
    返回:
        MemoryReadResult 形状的字典；failed 态带 error_kind
    """
    result: dict[str, Any] = {"status": status, "items": list(items or []), "excluded_counts": {}}
    if status == _READ_FAILED:
        result["error_kind"] = _ERROR_KIND
    return result


def _memory_items() -> list[dict[str, Any]]:
    """造一条投影后的记忆条目，字段与 MemoryReadItem 一致。"""
    return [
        {
            "memory_id": "mem-1",
            "version": 1,
            "title": "先按检索增强这条路走",
            "body_summary": "比过纯微调，检索增强更合本项目的数据量",
            "memory_type": "decision_note",
            "maturity": "leaning",
            "source_kind": "manual",
            "provenance_label": "倾向",
        }
    ]


def _read_delta(assessment: Any, recorder: _ReadRecorder) -> dict[str, Any]:
    """按深度评估结果跑一次按需读取，返回状态增量。

    patch 打在 pipeline 模块里的那个名字上，而不是 project_memory_reader 的定义处：
    pipeline 早已 from ... import 过来，打在定义处对它无效，用例会假绿。

    参数:
        assessment: 深度评估结果，可以是真实评估结果或缺键的旧态
        recorder: 记录实参并返回预置结果的读取替身
    返回:
        _read_memory_delta 产出的状态增量
    """
    with patch.object(pipeline, "read_project_memory", recorder):
        return asyncio.run(
            pipeline._read_memory_delta({"project_scope": _PROJECT_SCOPE}, assessment)
        )


def _depth_prompt_state() -> dict[str, Any]:
    """造一份深度评估 prompt 的输入态。"""
    return {"user_query": "我们之前定了什么方向", "intent": "explore", "focus": {}}


def _prompt_state(result: Any) -> dict[str, Any]:
    """造一份草案 prompt 的输入态。

    参数:
        result: 本轮记忆读取结果；None 表示状态里根本没有这个键（本轮没读记忆）
    返回:
        含深度评估结果、按需含记忆读取结果的状态字典
    """
    state: dict[str, Any] = {
        "user_query": "我们之前定的方向还算数吗",
        "turn_depth_assessment": _memory_only_assessment(),
    }
    if result is not None:
        state[_MEMORY_RESULT_STATE_KEY] = result
    return state


def _analyze_with_memory_only_failure() -> dict[str, Any]:
    """跑一次「记忆是唯一来源却读取失败」的 analyze 节点，返回节点增量。

    深度 minimal + 不要求论文证据 + 判定要翻记忆三者同时成立，本轮没有第二条信息来源；
    读取替身返回 failed，于是 _read_memory_delta 抛 MemoryOnlyTurnReadError。

    返回:
        analyze_planner_state 的状态增量
    """
    llm = _StubStructuredLLM({TurnDepthProposal.__name__: _proposal(
        depth=_DEPTH_MINIMAL,
        requires_paper_evidence=False,
        memory={"needed": True, "reason_code": "recall_prior_decision"},
    )})
    state = {
        "user_query": "我们之前定的研究方向是什么",
        "structured_query": {"paper_ids": []},
        "project_scope": _PROJECT_SCOPE,
    }
    with patch.object(pipeline, "read_project_memory", _ReadRecorder(_read_result(_READ_FAILED))), \
         patch.object(pipeline, "read_project_research_questions", AsyncMock(return_value=[])):
        return asyncio.run(pipeline.analyze_planner_state(state, llm, _LoopSettings()))


# 判定「本轮不必翻记忆」的四种态：模型判过不需要、模型没提、旧态没有这几个键
_NO_READ_ASSESSMENTS: tuple[Any, ...] = (
    _assessment(memory={"needed": False, "reason_code": "no_project_history_needed"}),
    _assessment(memory=None),
    {},
    {"depth": _DEPTH_LOOKUP},
)
_NO_READ_IDS = ("needed_false", "model_silent", "empty_assessment", "key_missing")

# 读取失败但本轮还有论文证据可走的三种深度组合。
# minimal + 要求论文证据这一档专盯唯一来源判据里的 requires_paper_evidence 那一条
_EVIDENCE_LEFT_DEPTHS: tuple[dict[str, Any], ...] = (
    {"depth": _DEPTH_LOOKUP, "requires_paper_evidence": True},
    {"depth": _DEPTH_RESEARCH, "requires_paper_evidence": True},
    {"depth": _DEPTH_MINIMAL, "requires_paper_evidence": True},
)
_EVIDENCE_LEFT_IDS = ("lookup", "research", "minimal_but_requires_evidence")

# empty 要在两种情形下都不触发失败分支
_EMPTY_CASE_ASSESSMENTS: tuple[Any, ...] = (
    _memory_only_assessment(),
    _assessment(memory={"needed": True}),
)
_EMPTY_CASE_IDS = ("memory_is_only_source", "paper_evidence_left")

# 不该渲染出记忆段的五种态
_NON_RENDERING_RESULTS: tuple[Any, ...] = (
    _read_result(_READ_EMPTY),
    _read_result(_READ_FAILED),
    _read_result(_READ_SUCCESS),
    {"status": _READ_SKIPPED, "items": [], "excluded_counts": {}},
    None,
)
_NON_RENDERING_IDS = ("empty", "failed", "success_without_items", "skipped", "key_missing")


# ─── 接缝一：模型偏好 → 深度评估结果的映射 ───


class TestNarrowingKeyAlignment:
    """验模型给的收窄键与记忆层逐字对齐，这是本文件最重要的一组。"""

    def test_assessment_declares_the_three_memory_keys(self) -> None:
        """三个记忆键在 TurnDepthAssessment 契约里都有声明，落值与契约不漂移。"""
        declared = set(TurnDepthAssessment.__annotations__)

        assert {_NEEDS_MEMORY_KEY, _NARROWING_KEY, _MEMORY_REASON_KEY} <= declared

    def test_request_fields_match_memory_layer_narrowing_contract(self) -> None:
        """模型请求的收窄字段集与记忆层收窄契约的键集完全相同。

        防的是两边各自加维：模型侧多一个键，记忆层不认、静默丢弃；记忆层多一个键，
        模型永远给不出来，那一维的收窄能力等于不存在。
        """
        assert _narrowing_fields() == frozenset(MemoryReadNarrowing.__annotations__)

    def test_produced_narrowing_carries_every_requested_key(self) -> None:
        """模型四个键全给时，memory_narrowing 一个不少、也不改名。"""
        assessment = _assessment(memory={"needed": True, **_FULL_NARROWING})

        assert set(assessment[_NARROWING_KEY]) == _narrowing_fields()

    def test_narrowing_values_reach_repository_kwargs(self) -> None:
        """收窄值真能穿过记忆层白名单变成仓储过滤参数。

        白名单 _NARROWING_FILTER_KEYS 按键名取值，取不到就当「这一维不收窄」：把
        paper_id 写成 paperId 不会报错，只会让收窄整维消失，读回一份范围完全不对的
        记忆，状态还是 success，模型照着不相干的历史判断组织回答。
        """
        narrowing = _assessment(memory={"needed": True, **_FULL_NARROWING})[_NARROWING_KEY]

        kwargs = injector._narrowing_kwargs(narrowing)

        assert set(kwargs) == set(injector._NARROWING_FILTER_KEYS)
        assert kwargs == {key: _FULL_NARROWING[key] for key in injector._NARROWING_FILTER_KEYS}
        # follow_links 不进 SQL，读取服务按这个键名决定要不要沿记忆关系扩一跳
        assert narrowing[reader._FOLLOW_LINKS_KEY] is True

    def test_whitelist_keys_all_come_from_the_model_request(self) -> None:
        """记忆层白名单里的每个键，模型侧都给得出来，不存在读不到的死键。"""
        assert frozenset(injector._NARROWING_FILTER_KEYS) <= _narrowing_fields()


class TestMemoryRequestToAssessment:
    """验模型的记忆读取请求怎么落成本回合的三个记忆键。"""

    def test_blank_and_false_values_stay_out_of_narrowing(self) -> None:
        """空串 / None / False 一律不入表。

        记忆层会把「键在、值为空」当成「指定为空值」，按空值过滤直接读空，于是
        「模型没指定这一维」被读成「项目里没有这类判断」。
        """
        assessment = _assessment(memory={
            "needed": True,
            _KEY_PAPER_ID: "   ",
            _KEY_QUESTION_ID: "",
            _KEY_MEMORY_TYPE: None,
            _KEY_FOLLOW_LINKS: False,
        })

        assert assessment[_NEEDS_MEMORY_KEY] is True
        assert assessment[_NARROWING_KEY] == {}

    def test_only_the_keys_the_model_gave_enter(self) -> None:
        """模型只给一维时收窄表里就只有那一维，其余键不补空值占位。"""
        assessment = _assessment(memory={"needed": True, _KEY_QUESTION_ID: "rq-7"})

        assert assessment[_NARROWING_KEY] == {_KEY_QUESTION_ID: "rq-7"}

    def test_needed_false_reads_nothing_but_keeps_the_reason_code(self) -> None:
        """模型判过「本轮用不上」时不翻记忆、不收窄，但原因码留下来。

        原因码是「判过不需要」与「根本没判过」的唯一区别。丢了它，trace 上分不清模型
        判断用不上，还是这次结构化调用压根没做这个判断，读取时机也就无从评测。
        """
        assessment = _assessment(memory={
            "needed": False,
            "reason_code": "no_project_history_needed",
            **_FULL_NARROWING,
        })

        assert assessment[_NEEDS_MEMORY_KEY] is False
        assert assessment[_NARROWING_KEY] == {}
        assert assessment[_MEMORY_REASON_KEY] == "no_project_history_needed"

    def test_missing_project_memory_gives_a_blank_reason_code(self) -> None:
        """模型完全没提这件事：不翻、不收窄、原因码为空串而不是编一个码。"""
        assessment = _assessment(memory=None)

        assert assessment[_NEEDS_MEMORY_KEY] is False
        assert assessment[_NARROWING_KEY] == {}
        assert assessment[_MEMORY_REASON_KEY] == ""

    def test_invented_memory_type_is_rejected_at_the_trust_boundary(self) -> None:
        """模型编出的记忆类型在信任边界即被拒。

        带着无效类型往下走会查出零条，再被读成「项目里没这类判断」，模型于是在
        「本项目还没形成过这类判断」这个错误前提上作答，全程没有任何异常。
        """
        with pytest.raises(ValidationError):
            ProjectMemoryReadRequest(needed=True, memory_type="not_a_type")


class TestDepthPromptListsMemoryTypes:
    """验合法记忆类型全部列进提示词：模型看不到 Python 类型，漏一个它只能猜。"""

    @pytest.mark.parametrize("memory_type", VALID_MEMORY_TYPES)
    def test_prompt_lists_every_valid_memory_type(self, memory_type: str) -> None:
        """每一个合法 memory_type 都出现在深度评估提示词里。

        漏列的那一类，模型要么猜一个近似词、在信任边界被拒连带整次深度评估失败，
        要么干脆不用它，那一类判断从此再没有被收窄读取过。
        """
        prompt = depth_assessor.build_depth_prompt(_depth_prompt_state(), frozenset({"p1"}))

        assert memory_type in prompt

    def test_prompt_lists_them_as_one_enumeration(self) -> None:
        """合法取值成段列出，而不是零散夹在别处的举例句里。"""
        prompt = depth_assessor.build_depth_prompt(_depth_prompt_state(), frozenset({"p1"}))

        assert ", ".join(VALID_MEMORY_TYPES) in prompt


# ─── 接缝二：按需读取 + 失败两分支 ───


class TestMemoryReadHappensOnlyOnDemand:
    """验读记忆是按需动作：该读时读一次、实参不错位，不该读时一次都不读。"""

    @pytest.mark.parametrize("assessment", _NO_READ_ASSESSMENTS, ids=_NO_READ_IDS)
    def test_not_needed_reads_nothing_and_leaves_no_fake_state(self, assessment: Any) -> None:
        """判定不翻记忆时零调用、零增量：不落 skipped 假状态，也不塞空结果。

        「没问过」和「问过被跳过」不是一回事：塞个 skipped 结果进去，下游看到的是一次
        真发生过的读取动作；反过来每轮无条件读，白花一次数据库往返，还给记忆是唯一
        来源的回合多出一个能整轮中断的失败点。
        """
        recorder = _ReadRecorder(_read_result(_READ_SUCCESS, items=_memory_items()))

        delta = _read_delta(assessment, recorder)

        assert recorder.calls == []
        assert delta == {}

    def test_narrowing_and_scope_pass_through_verbatim(self) -> None:
        """收窄条件原样成 query_narrowing，project_scope 原样成 context。

        一个决定读什么范围、一个决定读谁的记忆：前者错位读回不相干的记忆，后者错位会
        让读取服务的归属门拿不到 user_id，本该正常的回合直接抛错。
        """
        assessment = _assessment(memory={"needed": True, **_FULL_NARROWING})
        recorder = _ReadRecorder(_read_result(_READ_SUCCESS, items=_memory_items()))

        _read_delta(assessment, recorder)

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["query_narrowing"] == dict(_FULL_NARROWING)
        assert recorder.calls[0]["context"] == _PROJECT_SCOPE

    def test_empty_narrowing_passes_none_not_an_empty_dict(self) -> None:
        """模型没给任何收窄键时传 None，不是 {}。

        读取服务的签名与一跳判据都按 None 这个口径写：None 是「本次不收窄」，{} 读起来
        是「收窄过、条件为空」。传 {} 会让两种情形在日志与行为上再也分不开。
        """
        assessment = _assessment(memory={"needed": True, "reason_code": "recall_prior_decision"})
        recorder = _ReadRecorder(_read_result(_READ_EMPTY))

        _read_delta(assessment, recorder)

        assert assessment[_NARROWING_KEY] == {}
        assert recorder.calls[0]["query_narrowing"] is None

    def test_result_lands_exactly_on_the_memory_read_result_key(self) -> None:
        """读取结果落在 memory_read_result 这个键上，增量里没有别的东西。

        键名拼错不会报错：状态里查不到这个键，prompt 那侧就当本轮没读过记忆，
        记忆读到了也进不了 prompt，整条链静默失效。
        """
        result = _read_result(_READ_SUCCESS, items=_memory_items())

        delta = _read_delta(_assessment(memory={"needed": True}), _ReadRecorder(result))

        assert delta == {_MEMORY_RESULT_STATE_KEY: result}


class TestMemoryReadFailureBranches:
    """验读取失败分两条路走，判据错一项就走错整轮；empty 一条都不触发。"""

    @pytest.mark.parametrize("depth_kwargs", _EVIDENCE_LEFT_DEPTHS, ids=_EVIDENCE_LEFT_IDS)
    def test_failed_with_paper_evidence_left_keeps_the_main_chain(
        self, depth_kwargs: dict[str, Any]
    ) -> None:
        """还有论文证据可走时读取失败不抛异常，failed 与 error_kind 留在状态里可见。

        普通论文问答不该因为记忆读不出来整轮中断；但失败也不能被抹掉，否则「本轮没用
        记忆」看起来和「项目里没有相关记忆」一模一样。minimal + 要求论文证据这一档专门
        盯住唯一来源判据里的 requires_paper_evidence 那一条：删掉它，一次本可继续的回合
        会被判成整轮失败。
        """
        assessment = _assessment(memory={"needed": True}, **depth_kwargs)

        delta = _read_delta(assessment, _ReadRecorder(_read_result(_READ_FAILED)))

        assert delta[_MEMORY_RESULT_STATE_KEY]["status"] == _READ_FAILED
        assert delta[_MEMORY_RESULT_STATE_KEY]["error_kind"] == _ERROR_KIND

    def test_failed_when_memory_is_the_only_source_raises(self) -> None:
        """记忆是唯一来源时读取失败必须抛错，error_kind 一起带出去。

        minimal 不检索论文、本轮又不要求论文证据、模型却判定必须翻历史判断：三者同时
        成立时继续往下走，模型只能编一份项目此前的判断出来，而用户会把它当成自己当初
        真做过的决定。
        """
        recorder = _ReadRecorder(_read_result(_READ_FAILED))

        with pytest.raises(pipeline.MemoryOnlyTurnReadError) as exc:
            _read_delta(_memory_only_assessment(), recorder)

        assert exc.value.error_kind == _ERROR_KIND

    @pytest.mark.parametrize("assessment", _EMPTY_CASE_ASSESSMENTS, ids=_EMPTY_CASE_IDS)
    def test_empty_never_triggers_the_failure_branch(self, assessment: Any) -> None:
        """empty 在两种情形下都不走失败分支：项目内没有相关记忆是正常结果。

        把 empty 也当失败，「这个项目还没攒下这方面判断」的正常回合会直接报一次可重试
        失败，用户看到的是错误提示，而不是一句「项目内暂无相关记忆」。
        """
        delta = _read_delta(assessment, _ReadRecorder(_read_result(_READ_EMPTY)))

        assert delta[_MEMORY_RESULT_STATE_KEY]["status"] == _READ_EMPTY


class TestAnalyzeStateSurfacesMemoryFailure:
    """验记忆-唯一-来源失败在 analyze 节点变成可观测、可重试的显式失败。"""

    def test_failure_delta_is_observable_and_carries_no_draft(self) -> None:
        """错误码、可见状态位、循环决策三样齐全，且增量里没有 planner_draft。

        带失败载荷又没有 draft 的态一旦被路由送进 validate，用户最终看到的是一个校验
        错误，而真实原因是记忆读不出来，排查从第一眼就跑偏。
        """
        delta = _analyze_with_memory_only_failure()

        assert delta["error_code"] == _MEMORY_READ_FAILED_CODE
        assert delta["source_status"][_MEMORY_READ_STATUS_KEY] == _READ_FAILED
        assert delta["planner_loop_decision"] == _TERMINATE_FAIL
        assert _DRAFT_KEY not in delta

    def test_memory_failure_is_caught_before_the_broad_handler(self) -> None:
        """MemoryOnlyTurnReadError 的分支要在宽 except Exception 之前生效。

        两个 except 顺序写反时记忆读取失败会被记成深度评估失败：错误码与可见状态位
        全指向另一个部件，深度评估本身其实好着，排查一开始就查错了地方。
        """
        delta = _analyze_with_memory_only_failure()

        assert delta["error_code"] == _MEMORY_READ_FAILED_CODE
        assert delta["error_code"] != _DEPTH_ASSESSMENT_FAILED_CODE
        assert _DEPTH_ASSESSMENT_STATUS_KEY not in delta["source_status"]


# ─── 接缝三：读到的记忆 → 草案 prompt ───


class TestMemoryRendersIntoPlannerPrompt:
    """验记忆只在真读到时进 prompt，且必须由注入服务渲染。"""

    def test_success_with_items_renders_a_memory_block(self) -> None:
        """真读到记忆时记忆段进 prompt，并带着身份约束句。

        身份约束是产品硬约束：记忆是用户自己的判断，不是文献证据。缺了这句，模型会把
        它当论文原文写进带引用的结论，用户点开引用发现论文里没有这句话。
        """
        state = _prompt_state(_read_result(_READ_SUCCESS, items=_memory_items()))

        prompt = plan_prompt.build_planner_prompt(state)

        assert _MEMORY_BLOCK_MARKER in prompt
        assert _IDENTITY_CONSTRAINT in prompt
        assert _memory_items()[0]["title"] in prompt

    @pytest.mark.parametrize("result", _NON_RENDERING_RESULTS, ids=_NON_RENDERING_IDS)
    def test_non_success_renders_nothing_at_all(self, result: Any) -> None:
        """empty / failed / skipped / 无条目 / 缺键都不渲染，也不留噪音行。

        改成 truthy 判断（有 memory_read_result 就渲染）会把 failed 那档也渲染出来，
        而那档 items 是空的，模型看到一个空的记忆段标题、把它当成「项目里什么判断都
        没有」；反过来补一行「本轮没有记忆」是同一种噪音。
        """
        state = _prompt_state(result)

        assert plan_prompt._project_memory_lines(state) == []
        assert _MEMORY_BLOCK_MARKER not in plan_prompt.build_planner_prompt(state)

    def test_memory_block_sits_after_depth_lines_and_before_focus(self) -> None:
        """记忆段夹在深度那几行之后、Focus 之前，位置不飘。

        prompt 是按位置读的：先交代本轮该铺多少检索，紧接着给项目已有判断，再往下才是
        焦点与范围。记忆段跑到 Focus 之后会被读成范围声明的一部分。
        """
        state = _prompt_state(_read_result(_READ_SUCCESS, items=_memory_items()))

        prompt = plan_prompt.build_planner_prompt(state)

        assert prompt.index("Turn depth:") < prompt.index(_MEMORY_BLOCK_MARKER)
        assert prompt.index(_MEMORY_BLOCK_MARKER) < prompt.index("\nFocus: ")

    def test_rendering_goes_through_the_injector_service(self) -> None:
        """记忆段由注入服务渲染，实参就是读到的 items。

        自己拼一套渲染会同时绕掉两件事：段首那句身份约束，以及按字符预算截断——记忆一多
        就把整份草案 prompt 顶爆，模型连任务本身都读不完整。
        """
        items = _memory_items()
        state = _prompt_state(_read_result(_READ_SUCCESS, items=items))

        with patch.object(
            plan_prompt, "render_memory_prompt_block", return_value=_RENDER_SENTINEL
        ) as render:
            prompt = plan_prompt.build_planner_prompt(state)

        render.assert_called_once_with(items)
        assert _RENDER_SENTINEL in prompt


# ─── 接缝四：研究问题清单 → 深度评估 prompt（单元 V9，补 AC :375）───

# 项目里已接纳的两个研究问题，作清单渲染与透传的样例
_QUESTIONS = [
    {"question_id": "rq-1", "title": "先补召回还是先做重排"},
    {"question_id": "rq-2", "title": "时空预测里图结构该怎么给"},
]
# 清单段的段首标记
_QUESTION_SECTION = "Research question ids (id -> title):"


class _QuestionListRecorder:
    """记下研究问题清单被读了几次、拿的是哪个项目范围，并返回预置清单。"""

    def __init__(self, questions: list[dict[str, str]]) -> None:
        self._questions = questions
        self.calls: list[Any] = []

    async def __call__(self, context: Any) -> list[dict[str, str]]:
        self.calls.append(context)
        return self._questions


class _PromptCapturingLLM(_StubStructuredLLM):
    """在结构化替身之上记下收到的 prompt 正文，用来验清单真进了提示词。"""

    def __init__(self, payloads: dict[str, Any]) -> None:
        super().__init__(payloads)
        self.prompts: list[str] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        return await super().ainvoke(messages)


class TestResearchQuestionNarrowingIsUsable:
    """验「按研究问题收窄」这一维真的可用：模型手上得有本项目的 id 清单。

    清单不给，模型只能编 id；编出来的 id 关联不到任何记忆，读取落 empty，
    Planner 于是把「id 编错了」读成「项目里还没有这个问题下的判断」——
    安全（只收窄）但静默错空，AC :375 卡在这里。
    """

    def test_prompt_lists_project_research_questions(self) -> None:
        """清单成段进 prompt，id 与标题都在：只给 id 模型没法判断该挑哪个。"""
        prompt = depth_assessor.build_depth_prompt(
            _depth_prompt_state(), frozenset({"p1"}), research_questions=_QUESTIONS
        )

        assert _QUESTION_SECTION in prompt
        for question in _QUESTIONS:
            assert question["question_id"] in prompt
            assert question["title"] in prompt

    def test_empty_list_renders_no_section_at_all(self) -> None:
        """项目里没有研究问题时整段不出现，不摆一个空清单。

        空清单会被模型读成「这个项目没有研究问题」，而它分不出这与「这一轮没查到」，
        提示词那条规则也正是按「没有这一段就留空」写的。
        """
        prompt = depth_assessor.build_depth_prompt(_depth_prompt_state(), frozenset({"p1"}))

        assert _QUESTION_SECTION not in prompt

    def test_prompt_rule_binds_question_id_to_the_listed_ids(self) -> None:
        """规则把 question_id 限定在清单内，并保留「编 id 会静默读空」这条反面理由。"""
        prompt = depth_assessor.build_depth_prompt(
            _depth_prompt_state(), frozenset({"p1"}), research_questions=_QUESTIONS
        )

        assert "Set question_id only to one of the Research question ids listed below" in prompt
        assert "never invent one" in prompt
        # 旧措辞「除非上文逐字给了 id」等于叫模型永远留空：清单已经给了，这句必须换掉
        assert "unless a research question id is given to you verbatim" not in prompt

    def test_analyze_reads_the_list_and_feeds_it_to_the_prompt(self) -> None:
        """analyze 首次分析真去读清单，取的是 project_scope，且清单进了深度 prompt。"""
        llm = _PromptCapturingLLM({TurnDepthProposal.__name__: _proposal(
            depth=_DEPTH_MINIMAL, requires_paper_evidence=False, memory=None
        )})
        recorder = _QuestionListRecorder(_QUESTIONS)

        with patch.object(pipeline, "read_project_research_questions", recorder):
            asyncio.run(pipeline._first_analysis_delta(
                {"user_query": "我们之前定了什么方向", "project_scope": _PROJECT_SCOPE}, llm
            ))

        assert recorder.calls == [_PROJECT_SCOPE]
        assert _QUESTION_SECTION in llm.prompts[0]
        assert "rq-1" in llm.prompts[0]

    def test_blank_project_never_hits_the_repository(self) -> None:
        """没有项目范围时不查库：这一维压根无从收窄，白跑一次往返。"""
        with patch.object(reader, "get_research_memory_repository") as repo_factory:
            questions = asyncio.run(reader.read_project_research_questions({}))

        assert questions == []
        assert repo_factory.call_count == 0

    def test_list_read_failure_degrades_visibly_without_killing_the_turn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """清单读失败：记 error 日志后交空清单，不抛错中断整轮问答。

        这份清单只是「可以按哪个研究问题收窄」的提示，不是本轮的信息来源。读不出来
        只是少一个收窄维度、范围更宽而四道门照旧管着，按 design §9 普通问答该继续；
        但失败必须留痕，否则与「这个项目没有研究问题」再也分不开。
        """
        class _BoomRepo:
            async def list_questions(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
                raise RuntimeError("connection refused")

        with caplog.at_level("ERROR"):
            with patch.object(reader, "get_research_memory_repository", lambda: _BoomRepo()):
                questions = asyncio.run(
                    reader.read_project_research_questions(_PROJECT_SCOPE)
                )

        assert questions == []
        assert any(record.levelname == "ERROR" for record in caplog.records)
        assert any("研究问题清单" in record.getMessage() for record in caplog.records)

    def test_memory_read_reason_code_reaches_trace_summary(self) -> None:
        """「调出原因可解释」那一半：为什么翻记忆要能在 trace summary 里查到。

        原因码只活在内存状态里等于不可解释 —— 事后排查「这一轮凭什么去翻记忆」时
        什么也看不到，AC :375 的后半句就落不了地。
        """
        @dataclass(frozen=True)
        class _TraceCtx:
            request_id: str = "req-1"
            session_id: str = "sess-1"
            model_calls: tuple[Any, ...] = ()
            evidence_observations: tuple[Any, ...] = ()
            evidence_capture_errors: tuple[Any, ...] = ()

        assessment = _assessment(
            memory={"needed": True, _KEY_QUESTION_ID: "rq-1", "reason_code": "recall_prior_decision"}
        )

        summary = build_trace_summary_payload(_TraceCtx(), {"turn_depth_assessment": assessment})

        assert summary["turn_depth_assessment"][_MEMORY_REASON_KEY] == "recall_prior_decision"
        assert summary["turn_depth_assessment"][_NARROWING_KEY] == {_KEY_QUESTION_ID: "rq-1"}
