"""单元 V5 空项目缺证据的独立合规断言：一篇论文都没有时，两条可达路径的可观测终态。

作者：xxx
创建时间：2026-09-01 12:00:00

守的 AC（prd.md:392）：空项目请求论文事实时明确说明缺少可验证材料，而不是返回
无上下文的猜测或统一拒绝。这条 AC 有两半，正面（说清缺材料）与反面（既不猜也不
整轮挡回去）都在这里各有断言。

空项目问论文事实有两条可达路径，各自都要守：assessor 的空范围硬约束只强制
requires_paper_evidence=false，模型仍可判 minimal（走直答）或 lookup（走完整链，
检索无命中后诚实收尾）。

全部走真实图装配（create_rag_node()）：论文数是从 state 流到 prompt / 收尾文案的，
而本项目有过「跨图传的字段没声明在 RAGState 就被静默丢弃」的坑，手搓 dict 的
函数级测试对这类丢字段一律假绿。validate / execute / synthesize 三个节点用只计数
的探针包住真实实现，既能证明两条路各自走到哪，也能证明空项目与有论文走的是同一条边。

路径 2 的覆盖范围要说清，别当成「空项目的 lookup 回合都验过了」：能走到诚实收尾的
计划形状是「取证单元 + 依赖它的定向建议单元」。取证单元的目标论文为空表时产的是
unit 级 direct_text_missing 缺口，终答资格不把它当拦路缺口，那种形状的回合会走常规
LLM 综合而不是诚实收尾（详见 _honest_ending_draft 的说明）。
"""

from __future__ import annotations

import ast
import asyncio
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from app.models import evidence_layers as layers
from app.models.research_contracts import TurnDepthAssessment
from app.models.research_planner_contracts import (
    DirectAnswerDraft,
    ResearchClaimDraftBatch,
    ResearchPlanDraft,
)
from app.nodes import rag_node
from app.services.internal_research import research_planner_conversion as conversion
from app.services.internal_research import research_planner_direct_answer as direct_answer
from app.services.internal_research.research_memory_writeback import (
    build_turn_outcome,
    write_back_turn_memory,
)
from research_planner_test_utils import make_claim_policy, make_draft, make_unit

# ─── 被审对象的对外契约字面量 ───

_DEPTH_MINIMAL = "minimal"
_DEPTH_LOOKUP = "lookup"
_DEPTH_REASON_CODES = {
    _DEPTH_MINIMAL: "no_paper_fact_needed",
    _DEPTH_LOOKUP: "single_paper_metadata_lookup",
}
_DECISION_HONEST = "terminate_honest"
_DECISION_FAIL = "terminate_fail"
_EVIDENCE_NOT_FOUND_CODE = "evidence_not_found_in_scope"
_REQUEST_ID = "req-v5-compliance"

# 论文事实问题：答准它必须有论文原文，正是这条 AC 说的「请求论文事实」
_PAPER_FACT_QUESTION = "这篇论文的实验用了多少条数据"
# 直答替身返回的正文，用来分辨「模型自己答的」与「被模板拒答句顶替」
_MODEL_ANSWER_TEXT = "这个项目里还没有论文，我没有可核对的材料。"
_PAPER_ID = "paper-a"


@dataclass(frozen=True)
class _StubMessage:
    """模型返回的消息替身，只带直答读的 content。"""

    content: str


@dataclass(frozen=True)
class _LoopSettings:
    """最小配置对象：只带计划单元数上限与循环总超时，供节点读配置。

    循环额度已取消：放行回边看的是「这一步有没有带来新东西」，不再有可设的恢复次数。
    检索落空的硬缺口头一次出现会放行一轮补检索，缺口集不变后落到诚实收尾——本文件
    审的是收尾那一刻的说法，路径形状的节点次数由用例据实断言。
    """

    rag_research_planner_max_units: int = 8
    rag_planner_loop_total_timeout_seconds: float = 0.0


class _PlannerLLM:
    """图里唯一的 LLM 替身：直答出正文，结构化调用按 schema 分派。

    计划草案照给（lookup 那条路要真过一遍校验与执行），claim 草案恒空（检索落空，
    本来就无从产 claim）。直答自己也是结构化调用（正文 + 这段是不是知识性内容），
    按非知识性内容走——本文件审的是论文事实那条口径，来源说明那一档不参与。
    其余 schema 一律报错：本文件的回合不该发起别的结构化调用。
    """

    def __init__(self, draft: dict[str, Any] | None = None) -> None:
        self.prompts: list[str] = []
        self._draft = draft

    def with_structured_output(self, schema: type) -> Any:
        if schema is ResearchPlanDraft:
            return _PlanDraftCall(self._draft)
        if schema is ResearchClaimDraftBatch:
            return _EmptyClaimCall()
        if schema is DirectAnswerDraft:
            return _DirectAnswerCall(self)
        raise AssertionError(f"本轮不该发起 {schema.__name__} 结构化调用")


class _DirectAnswerCall:
    """直答结构化调用替身：记下 prompt 供口径断言取证，返回预置正文。"""

    def __init__(self, owner: "_PlannerLLM") -> None:
        self._owner = owner

    async def ainvoke(self, messages: Any) -> Any:
        self._owner.prompts.append(str(messages[0].content))
        return DirectAnswerDraft(answer=_MODEL_ANSWER_TEXT, knowledge_bearing=False)


@dataclass(frozen=True)
class _PlanDraftCall:
    """计划草案结构化调用替身：返回预置草案，未预置说明这一轮不该产计划。"""

    draft: dict[str, Any] | None

    async def ainvoke(self, _messages: Any) -> ResearchPlanDraft:
        if self.draft is None:
            raise AssertionError("本轮不该产计划草案")
        return ResearchPlanDraft.model_validate(self.draft)


class _EmptyClaimCall:
    """claim 草案结构化调用替身：检索落空，没有证据可支撑任何 claim。"""

    async def ainvoke(self, _messages: Any) -> ResearchClaimDraftBatch:
        return ResearchClaimDraftBatch(claims=[])


@dataclass
class _NodeSpy:
    """包住真实节点的计数探针：记调用次数，再把状态交给真实实现。

    不替换实现，只数次数。空项目与有论文走没走同一条边，看的就是这三个数。
    """

    name: str
    inner: Any
    calls: int = 0

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        result = self.inner(state)
        if asyncio.iscoroutine(result):
            return await result
        return result


@dataclass
class _GraphRun:
    """一次真实子图运行的产物：终态 + 三个节点各自被走到几次。"""

    final_state: dict[str, Any]
    validate: _NodeSpy
    execute: _NodeSpy
    synthesize: _NodeSpy
    llm: _PlannerLLM

    @property
    def node_calls(self) -> tuple[int, int, int]:
        """(validate, execute, synthesize) 的调用次数，用来比对控制流形状。"""
        return (self.validate.calls, self.execute.calls, self.synthesize.calls)


def _assessment(*, depth: str, requires_paper_evidence: bool) -> Any:
    """造一份本轮深度评估结果，键名走契约声明而不是手抄字面量。

    参数:
        depth: 本轮处理深度
        requires_paper_evidence: 本轮回答是否必须有论文 Evidence 支撑
    返回:
        TurnDepthAssessment 形状的字典
    """
    return TurnDepthAssessment(
        depth=depth,
        reason_code=_DEPTH_REASON_CODES[depth],
        requires_paper_evidence=requires_paper_evidence,
        resolved_query=_PAPER_FACT_QUESTION,
        needs_project_memory=False,
    )


def _graph_input(*, depth: str, paper_ids: list[str]) -> dict[str, Any]:
    """造进图的初始状态：深度已定，项目论文范围按参数给。

    空项目那一档 project_scope.paper_ids 是空表，_allowed_paper_ids 会返回 None
    （不是空集合），论文数取数踩不踩这个坑正是本文件要验的事之一。

    requires_paper_evidence 一律 false：assessor 的空范围硬约束在空项目下强制它为
    false，这里照那条约束的结果造状态，不去造一个真实运行里不可能出现的组合。

    参数:
        depth: 本轮已定的处理深度
        paper_ids: 项目里的论文 id 表，空表代表这个项目一篇论文都没有
    返回:
        可直接交给已编译子图 ainvoke 的状态字典
    """
    return {
        "user_query": _PAPER_FACT_QUESTION,
        "structured_query": {"paper_ids": list(paper_ids)},
        "project_scope": {"paper_ids": list(paper_ids)},
        "turn_depth": depth,
        "turn_depth_assessment": _assessment(depth=depth, requires_paper_evidence=False),
    }


def _init_stub(_state: dict[str, Any]) -> dict[str, Any]:
    """rag_init 替身：真实实现要探 Milvus 索引状态，本文件只审 planner 段的说法。"""
    return {}


async def _write_back_stub(_state: dict[str, Any]) -> dict[str, Any]:
    """write_back 替身：真实实现要写库与观测层，四态派生在用例里单独真实调用。"""
    return {}


async def _empty_adapter(_unit: Any, _context: Any) -> list[dict[str, Any]]:
    """证据适配器替身：检索恒空。

    空项目本来就没有任何可检索内容，这是它在检索层的真实形状；有论文那一档也用它，
    这样两档的差别只剩「论文数」这一个变量，别的都一样。
    """
    return []


async def _not_in_index_probe(_scoped_ids: list[str]) -> bool:
    """库内容探针替身：恒报库内无可检索内容，替掉真实实现里的向量库查询。"""
    return False


def _titles_stub(_scoped_ids: list[str]) -> dict[str, str]:
    """标题解析替身：真实实现要读 Postgres，与本文件审的措辞无关。"""
    return {}


def _honest_ending_draft(*, paper_ids: list[str]) -> dict[str, Any]:
    """造一份能走到诚实收尾的两单元草案：取证单元 + 依赖它的定向建议单元。

    为什么必须带定向建议单元：终答资格只把「落在定向建议依赖链上的硬缺口」与
    target_coverage_missing 算成拦路缺口。取证单元的目标论文为空表时（空项目下计划
    的自然形状）产的是 unit 级 direct_text_missing，不在这两类里，资格照旧放行，
    这一轮就走常规 LLM 综合而不是诚实收尾。空项目与有论文两档都用同一个形状，
    差别只留「论文数」这一个变量。

    参数:
        paper_ids: 取证单元的目标论文 id 表，空表代表项目里没有论文可指定
    返回:
        ResearchPlanDraft 形状的字典
    """
    evidence = make_unit("unit-evidence", output_role="evidence")
    # make_unit 的目标论文缺省会兜成一篇样例论文，空项目这一档必须显式清空：
    # 兜着一个不存在的 id 走下去，收尾文案就会按「有 1 篇」说话，等于测试自己造了假绿
    evidence["evidence_policy"]["target_paper_ids"] = list(paper_ids)
    recommend = make_unit(
        "unit-recommend",
        goal="Give the next research direction",
        action="infer next direction from prior units",
        source_type="inference_from_units",
        output_role="final_recommendation",
        deps=[{"unit_id": "unit-evidence", "reason": "needs the located evidence"}],
    )
    recommend["evidence_policy"] = {
        "source_type": "inference_from_units",
        "target_paper_ids": [],
        "required_sections": [],
        "evidence_dimensions": [],
        "locator_queries": [],
        "min_citations_per_target": 0,
    }
    # 推理单元不落引用：引用要挂在论文原文上，这一步只是据前序单元推结论
    recommend["claim_policy"] = make_claim_policy(
        ["evidence_grounded_inference"], citation_required=False
    )
    draft = make_draft([evidence, recommend])
    draft["question"] = _PAPER_FACT_QUESTION
    return draft


def _run_graph(
    state: dict[str, Any],
    *,
    draft: dict[str, Any] | None = None,
) -> _GraphRun:
    """用真实装配的子图跑一轮，三个 planner 节点用计数探针包住真实实现。

    探针必须在 create_rag_node() 之前替换：add_node 装配时就把函数对象抓走了，
    编译之后再打补丁改不到图里那份引用，调用次数会永远是 0。

    参数:
        state: 进图的初始状态
        draft: 预置计划草案，None 表示这一轮不该产计划（minimal 直答）
    返回:
        _GraphRun：图的终态、三个节点的调用次数与 LLM 替身
    """
    llm = _PlannerLLM(draft)
    validate = _NodeSpy("planner_validate", rag_node.planner_validate_node)
    execute = _NodeSpy("planner_execute", rag_node.planner_execute_node)
    synthesize = _NodeSpy("planner_synthesize", rag_node.planner_synthesize_node)
    replacements = {
        "rag_init_node": _init_stub,
        "write_back_node": _write_back_stub,
        "planner_validate_node": validate,
        "planner_execute_node": execute,
        "planner_synthesize_node": synthesize,
        "_llm": lambda: llm,
        "_settings": _LoopSettings,
        # 检索、探库、读标题三处 I/O 换成替身：都在被审说法的上游，且单元测里不该连库
        "_planner_adapters": lambda _state: {layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT: _empty_adapter},
        "_planner_optional_adapters": lambda _state: {},
        "_scoped_index_probe": _not_in_index_probe,
        "_resolve_paper_titles": _titles_stub,
    }
    with ExitStack() as stack:
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(rag_node, name, replacement))
        final_state = asyncio.run(rag_node.create_rag_node().ainvoke(state))
    return _GraphRun(
        final_state=dict(final_state),
        validate=validate,
        execute=execute,
        synthesize=synthesize,
        llm=llm,
    )


def _direct_answer_run(*, paper_ids: list[str]) -> _GraphRun:
    """跑一轮 minimal 直答回合（路径 1）：不要论文证据，不产计划。"""
    return _run_graph(_graph_input(depth=_DEPTH_MINIMAL, paper_ids=paper_ids))


def _honest_ending_run(*, paper_ids: list[str]) -> _GraphRun:
    """跑一轮 lookup 回合（路径 2）：真过校验与执行，检索落空后诚实收尾。"""
    return _run_graph(
        _graph_input(depth=_DEPTH_LOOKUP, paper_ids=paper_ids),
        draft=_honest_ending_draft(paper_ids=paper_ids),
    )


class TestEmptyProjectStatesMissingMaterialOnBothPaths:
    """AC 正面那半：空项目问论文事实，两条可达路径都要说清缺少可验证材料。

    两条路的说法出自不同地方：路径 1 由 prompt 规则约束模型怎么说，路径 2 是模板
    确定性组装的正文。两处都在真实图上验，因为论文数是从 state 一路流过来的——
    provider 层单测能过、图上照样可能因为字段没进图而拿到 0 或拿不到。
    """

    def test_direct_answer_prompt_switches_to_missing_material_wording(self) -> None:
        """路径 1：空项目回合进模型的 prompt 带的是「缺可验证材料」那一档规则。

        破了会怎样：模型收到的还是「去查项目里的论文」，而项目里一篇都没有，
        用户照着去翻一个空书架；说不说清缺材料全凭模型自觉，本 AC 没有任何着力点。
        """
        run = _direct_answer_run(paper_ids=[])

        prompt = run.llm.prompts[0]
        # 断的是真实规则常量本身，不是另抄一段近似措辞：换掉常量内容这条会红
        assert direct_answer._PAPER_FACT_RULE_EMPTY_PROJECT in prompt
        assert direct_answer._PAPER_FACT_RULE_WITH_PAPERS not in prompt

    def test_project_with_papers_keeps_pointing_at_the_papers(self) -> None:
        """有论文时不许改口：材料确实在项目里，指回去查是条真出路。

        破了会怎样：论文数没真从 state 流到 prompt，两档口径糊成一档——
        项目里明明有论文却被告知「本项目缺少可验证材料」，这是另一种错报。
        """
        run = _direct_answer_run(paper_ids=[_PAPER_ID])

        prompt = run.llm.prompts[0]
        assert direct_answer._PAPER_FACT_RULE_WITH_PAPERS in prompt
        assert direct_answer._PAPER_FACT_RULE_EMPTY_PROJECT not in prompt

    def test_honest_ending_answer_states_missing_material(self) -> None:
        """路径 2：空项目走到诚实收尾，正文说清项目里没有可核对的论文材料。

        破了会怎样：用户读到「在你指定的 0 篇论文里检索过了，可能这几篇论文没涉及」
        —— 指代一批不存在的论文，既没说清缺什么，也没给下一步动作。
        """
        run = _honest_ending_run(paper_ids=[])

        answer = run.final_state["answer"]
        assert answer == conversion._EMPTY_PROJECT_ANSWER.format(question=_PAPER_FACT_QUESTION)
        # 不许指代不存在的论文，也不许出现「0 篇」这种把空项目说成一次检索范围的表述
        assert "这几篇论文" not in answer
        assert "0 篇" not in answer
        assert "0 篇" not in run.final_state["error_summary"]

    def test_honest_ending_keeps_scoped_wording_when_papers_exist(self) -> None:
        """有论文走到诚实收尾时保持原措辞：确实查过了，只是没命中。

        破了会怎样：项目里有论文却被告知「这个项目里还没有论文」，
        用户会去重新上传已经在项目里的论文。
        """
        run = _honest_ending_run(paper_ids=[_PAPER_ID])

        answer = run.final_state["answer"]
        assert answer == conversion._SCOPED_MISS_ANSWER.format(
            paper_count=1, question=_PAPER_FACT_QUESTION
        )
        assert conversion._EMPTY_PROJECT_SUMMARY not in run.final_state["error_summary"]

    def test_honest_ending_trace_reason_does_not_invent_a_search(self) -> None:
        """空项目的 trace 归因不说「范围内检索无命中」。

        破了会怎样：排查的人照这句去查检索式为什么没召回。空项目这一轮 execute 确实
        跑过、检索式也确实发出去过，但压根没有 scoped 论文范围可检索——「范围内没命中」
        会把人带去调检索式，而真正的原因是这个项目里没有材料。
        """
        empty = _honest_ending_run(paper_ids=[])
        scoped = _honest_ending_run(paper_ids=[_PAPER_ID])

        empty_reason = empty.final_state["source_status_reasons"]["planner_final"]
        assert empty_reason == conversion._EMPTY_PROJECT_FINAL_REASON
        assert "检索无命中" not in empty_reason
        # 有论文那条归因保持原样：检索真发起过
        assert scoped.final_state["source_status_reasons"]["planner_final"] == (
            conversion._SCOPED_MISS_FINAL_REASON
        )


class TestEmptyProjectTurnIsNotABlanketRefusal:
    """AC 反面那半之一：缺材料不等于把整轮问题挡回去。

    「统一拒绝」在载荷上有形状：这一轮落到失败/终止分支、或者正文被一句模板拒答
    顶替掉。真实图上验的就是它没落到那种形状——直答回合照旧是成功路径，正文是
    模型自己写的，而 prompt 明确留着「不依赖论文原文的部分照常回答」这条口子。
    """

    def test_empty_project_direct_answer_is_still_a_success_turn(self) -> None:
        """空项目直答回合仍是正常成功路径：错误位留空，也没落终止失败决策。

        破了会怎样：空项目被当成一种失败挡在门外，前端按错误态渲染，
        用户问的那些本来不需要论文的部分（改写、澄清、延续既有思路）一并被拒掉。
        """
        run = _direct_answer_run(paper_ids=[])

        assert run.final_state["error_code"] == ""
        assert run.final_state["error_summary"] == ""
        assert run.final_state["planner_loop_decision"] != _DECISION_FAIL
        # 走到了综合节点，不是在半路被截回终态
        assert run.synthesize.calls == 1

    def test_empty_project_answer_comes_from_the_model_not_a_refusal_template(self) -> None:
        """正文是模型自己写的那句，不是被一句固定拒答文案顶替。

        破了会怎样：不管用户问什么，空项目一律回同一句「请先上传论文」——
        这正是 AC 说的统一拒绝，而且从载荷上看它跟正常回答长得一模一样。
        """
        run = _direct_answer_run(paper_ids=[])

        assert run.final_state["answer"] == _MODEL_ANSWER_TEXT

    def test_empty_project_rule_still_orders_answering_the_rest(self) -> None:
        """空项目那条规则里必须同时留着「照常回答不依赖论文的部分」。

        破了会怎样：规则被删成只剩「说明缺少可验证材料」，模型照着只说缺材料，
        缺证据的说明就变成了另一种统一拒绝，AC 的反面那半在提示词侧先失守。
        """
        run = _direct_answer_run(paper_ids=[])

        rule = direct_answer._PAPER_FACT_RULE_EMPTY_PROJECT
        assert "照常正常回答" in rule
        assert "不要因此把整个问题挡回去" in rule
        assert rule in run.llm.prompts[0]
        # 这一轮的身份仍是「直接回答用户」，不是「拒绝这一轮」
        assert "请直接用中文回答用户" in run.llm.prompts[0]

    def test_honest_ending_is_evidence_not_found_not_a_system_failure(self) -> None:
        """路径 2 的收尾是「查过没找到」，不是规划失败，而且给了下一步动作。

        破了会怎样：空项目被记成一次 planner 失败，评测把产品的正常诚实收尾算成
        故障；用户读完也不知道该干什么，只知道被拒了。
        """
        run = _honest_ending_run(paper_ids=[])

        assert run.final_state["error_code"] == _EVIDENCE_NOT_FOUND_CODE
        assert run.final_state["source_status"]["planner"] == "no_evidence"
        # 正文带「往项目里加论文」这个可执行的下一步，不是只有一句拒绝
        assert "加进这个项目" in run.final_state["answer"]


class TestEmptyProjectNeverFabricatesCitedAnswer:
    """AC 反面那半之二：空项目不许产出「看上去有出处」的答案。

    项目里一篇论文都没有，任何引用块或证据 id 都必然指不到原文。留下一个，
    用户点开依据是空的，评测却把这一轮算成有文献支撑的回答。
    """

    def test_direct_answer_turn_carries_no_plan_no_blocks_no_evidence(self) -> None:
        """路径 1：计划为空、定稿引用块为空、证据池为空。

        破了会怎样：一轮没检索过任何论文的回答挂上了引用，
        「无上下文的猜测」被包装成有依据的结论。
        """
        run = _direct_answer_run(paper_ids=[])

        assert run.final_state["research_plan"] is None
        assert run.final_state["planner_final_blocks"] == []
        assert run.final_state["research_unit_results"] == []
        assert run.final_state["evidence_records"] == []

    def test_direct_answer_turn_yields_no_evidence_ids_downstream(self) -> None:
        """路径 1 的真实派生终态里也没有证据 id。

        破了会怎样：终态派生侧凭空多出一个证据 id，前端点开查不到出处，
        而只看短路增量一切正常，问题只在派生侧显形。
        """
        run = _direct_answer_run(paper_ids=[])

        writeback = asyncio.run(write_back_turn_memory(run.final_state, request_id=_REQUEST_ID))
        outcome = build_turn_outcome(run.final_state, memory_writeback=writeback)

        assert outcome["evidence_ids"] == []
        assert outcome["missing_evidence"] == []

    def test_honest_ending_carries_no_blocks_and_no_evidence(self) -> None:
        """路径 2：诚实收尾同样不留引用块与证据记录。

        破了会怎样：一句「没找到」的收尾里挂着引用，用户以为查到了东西。
        """
        run = _honest_ending_run(paper_ids=[])

        assert run.final_state["planner_final_blocks"] == []
        assert run.final_state["evidence_records"] == []
        assert run.final_state["research_plan"] is None
        # 缺口记录可以留（确实缺了这些维度），但一条都不许挂到某篇论文上：
        # 空项目里任何论文归属都是凭空造的
        attributed = [
            item for item in run.final_state["missing_evidence"] if item.get("paper_id")
        ]
        assert attributed == []


# 表示「空项目 / 无论文」的命名片段
_EMPTY_TOKENS = ("empty_scope", "empty_project", "no_paper", "zero_paper", "paperless")
# 表示「提前拦下这一轮」的命名片段；与上一组同时命中即为空项目专用的拦截门
_GATE_TOKENS = (
    "termination",
    "terminate",
    "block",
    "gate",
    "guard",
    "reject",
    "refuse",
    "abort",
    "short_circuit",
    "bail",
    "early_return",
)
# 被审模块：V5 改过的三个，加上装配图与选边的 rag_node
_AUDITED_PLANNER_MODULES = (
    "research_planner_direct_answer.py",
    "research_planner_conversion.py",
    "research_planner_pipeline.py",
)
_INTERNAL_RESEARCH_DIR = (
    Path(__file__).resolve().parents[2] / "app" / "services" / "internal_research"
)
_RAG_NODE_PATH = Path(__file__).resolve().parents[2] / "app" / "nodes" / "rag_node.py"


def _identifiers(path: Path) -> set[str]:
    """收集一个模块里所有可能承载拦截门的标识符名。

    参数:
        path: 模块文件路径
    返回:
        名字、属性名、函数名与关键字实参名的集合
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


class TestNoEmptyProjectInterceptionGate:
    """本单元的硬约束：不许为空项目新加一道拦截门。

    本卡 1e 加过 `_empty_scope_termination` 提前拦空项目，被用户按「不要仅为掩盖问题
    添加边界规则和护栏」叫停并全部撤回。V5 的做法必须是「让回答路径知道论文数这个
    事实」，而不是在它前面架门——门会把一整轮问题挡掉，正是 AC 反面那半禁的事。

    断言按符号扫源码 + 比控制流形状，不是在注释里写一句约定：注释拦不住下一个人加门。
    """

    @pytest.mark.parametrize("module_name", _AUDITED_PLANNER_MODULES)
    def test_no_empty_project_gate_identifier_in_planner_modules(self, module_name: str) -> None:
        """三个模块里不许出现同时带「空项目」与「拦截」语义的标识符。

        破了会怎样：空项目在回答路径之前被一道门截住，用户问的所有问题（包括压根
        不需要论文的那些）一起被挡回去，而载荷上看起来只是「这一轮提前结束了」。
        """
        offenders = {
            name
            for name in _identifiers(_INTERNAL_RESEARCH_DIR / module_name)
            if any(token in name.lower() for token in _EMPTY_TOKENS)
            and any(token in name.lower() for token in _GATE_TOKENS)
        }

        assert offenders == set()

    def test_no_empty_project_edge_in_graph_assembly(self) -> None:
        """装配图与选边那侧也不许出现空项目专用的分支符号。

        破了会怎样：门挪到了路由层，planner 模块扫得干干净净，
        而空项目在图上被一条专用边直接送去终态。
        """
        offenders = {
            name
            for name in _identifiers(_RAG_NODE_PATH)
            if any(token in name.lower() for token in _EMPTY_TOKENS)
            and any(token in name.lower() for token in _GATE_TOKENS)
        }

        assert offenders == set()

    def test_direct_answer_path_is_identical_with_and_without_papers(self) -> None:
        """路径 1：空项目与有论文经过同样的节点、落同样的决策与状态位。

        破了会怎样：空项目多走一条早返回或新终止分支，两档的差别就不再只是「说法
        不同」，而是「走的路不同」——那就是加了一道门，只是没起名字。
        """
        empty = _direct_answer_run(paper_ids=[])
        scoped = _direct_answer_run(paper_ids=[_PAPER_ID])

        # analyze 短路直达 synthesize：两档都不进 validate 与 execute
        assert empty.node_calls == scoped.node_calls == (0, 0, 1)
        assert empty.final_state["planner_loop_decision"] == scoped.final_state["planner_loop_decision"]
        assert empty.final_state["source_status"] == scoped.final_state["source_status"]
        assert empty.final_state["error_code"] == scoped.final_state["error_code"] == ""

    def test_honest_ending_path_is_identical_with_and_without_papers(self) -> None:
        """路径 2：空项目与有论文一样真走校验与执行，落同一个收尾决策与错误码。

        破了会怎样：空项目被提前判死，不再进 execute——「确实查过了」这个前提没了，
        诚实收尾变成一句没做过检索的托辞，而 trace 里看不出差别。

        execute 走两次不是分叉：取消恢复额度后，检索落空的硬缺口头一次出现会按判等
        放行一轮补检索，缺口集不变后落诚实收尾。两档走的是同一条（1,2,1），差别
        仍然只在说法上。
        """
        empty = _honest_ending_run(paper_ids=[])
        scoped = _honest_ending_run(paper_ids=[_PAPER_ID])

        assert empty.node_calls == scoped.node_calls == (1, 2, 1)
        assert empty.final_state["planner_loop_decision"] == _DECISION_HONEST
        assert scoped.final_state["planner_loop_decision"] == _DECISION_HONEST
        assert empty.final_state["error_code"] == scoped.final_state["error_code"]
        assert empty.final_state["source_status"] == scoped.final_state["source_status"]

    @pytest.mark.parametrize(
        "runner",
        [_direct_answer_run, _honest_ending_run],
        ids=("direct_answer", "honest_ending"),
    )
    def test_payload_key_set_does_not_change_with_paper_count(self, runner: Any) -> None:
        """两档载荷键集合一致：论文数只换措辞，不换下游要消费的形状。

        破了会怎样：空项目多出或少掉一个键，前端与评测按论文数分叉读载荷，
        「同一条路径」就只是说法上的同一条。
        """
        empty = runner(paper_ids=[])
        scoped = runner(paper_ids=[_PAPER_ID])

        assert set(empty.final_state) == set(scoped.final_state)
        assert set(empty.final_state["source_status"]) == set(scoped.final_state["source_status"])
