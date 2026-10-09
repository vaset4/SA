"""单元 V1 深度分流的独立合规断言：minimal 短路在真实图与真实派生链上的可观测终态。

作者：xxx
创建时间：2026-08-31 23:00:00

这份断言由未参与实现的一方写，盯的是「实现侧写对了字段」之外的那一层：
字段写对了，下游派生仍可能把它丢掉，而丢掉的时候既不报错也不留日志。

六条各自守什么、破了会怎样，写在每个测试类与用例的文档注释里。全部走真实实现：
图用 create_rag_node() 真实装配（手搓 dict 冒充状态时，未在 RAGState 声明的字段
会在图入口被静默丢弃，测试假绿），四态表走 build_turn_outcome 真实派生，
记忆段走 render_project_memory_block 真实渲染。
"""

from __future__ import annotations

import ast
import asyncio
import re
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from app.models.research_contracts import (
    VALID_CAPABILITY_STATUSES,
    TurnDepthAssessment,
)
from app.models.research_planner_contracts import DirectAnswerDraft
from app.nodes import rag_node
from app.services.internal_research import research_memory_writeback as writeback
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research import research_planner_plan_prompt as plan_prompt
from app.services.internal_research.research_memory_writeback import (
    build_turn_outcome,
    write_back_turn_memory,
)

# ─── 跨文件约定本身：这些字面量就是被审对象的对外契约 ───

_DEPTH_MINIMAL = "minimal"
_DEPTH_LOOKUP = "lookup"
_DEPTH_RESEARCH = "research"
_DEPTH_REASON_CODES = {
    _DEPTH_MINIMAL: "no_paper_fact_needed",
    _DEPTH_LOOKUP: "single_paper_metadata_lookup",
    _DEPTH_RESEARCH: "cross_paper_comparison",
}

# 四态能力表里论文取证那一项的能力名，与 writeback._CAPABILITY_PAPER_RETRIEVAL 同值
_CAPABILITY_PAPER_RETRIEVAL = "retrieve_paper_evidence"
_CAPABILITY_MEMORY_READ = "read_project_memory"
_STATUS_SKIPPED = "skipped"
_READ_SUCCESS = "success"
_READ_EMPTY = "empty"
_READ_FAILED = "failed"

# 记忆段段首的身份约束句标记，真实渲染函数产出
_MEMORY_BLOCK_MARKER = "【项目研究记忆】"
_MEMORY_ID = "mem-1"

_QUESTION = "我们之前定的方向还算数吗"
_ANSWER_TEXT = "按项目此前的判断，这个方向仍然成立。"
_REQUEST_ID = "req-v1-compliance"

@dataclass(frozen=True)
class _StubMessage:
    """模型返回的消息替身，只带直答读的 content。"""

    content: str


@dataclass(frozen=True)
class _LoopSettings:
    """最小配置对象：循环额度已取消，只保留计划单元数上限。"""

    rag_research_planner_max_units: int = 8


class _StructuredCall:
    """结构化调用替身：本文件的用例都不该走到这里，走到就是短路没生效。"""

    def __init__(self, schema_name: str) -> None:
        self._schema_name = schema_name

    async def ainvoke(self, _messages: Any) -> Any:
        raise AssertionError(f"本轮不该发起 {self._schema_name} 结构化调用")


class _DirectAnswerCall:
    """直答自己的结构化调用替身：记下 prompt 供记忆段断言取证，返回预置草案。

    直答改成结构化输出（正文 + 这段是不是知识性内容）后，它不再走纯文本调用；
    这里按契约返回，来源说明那一档不参与本文件的判断，统一按非知识性内容走。
    """

    def __init__(self, owner: "_DirectAnswerLLM") -> None:
        self._owner = owner

    async def ainvoke(self, messages: Any) -> Any:
        self._owner.prompts.append(str(messages[0].content))
        return DirectAnswerDraft(answer=_ANSWER_TEXT, knowledge_bearing=False)


class _DirectAnswerLLM:
    """直答用的模型替身，记下收到的 prompt 供记忆段断言取证。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type) -> Any:
        # 直答只认自己那份契约；本文件其他 schema 都表明短路没生效，走断言替身
        if schema is DirectAnswerDraft:
            return _DirectAnswerCall(self)
        return _StructuredCall(schema.__name__)


@dataclass
class _NodeSpy:
    """记录某个图节点被调用了几次，本文件用来证明它一次都没被调用。"""

    name: str
    calls: int = 0
    seen_states: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.seen_states.append(dict(state))
        return {}


def _assessment(*, depth: str, requires_paper_evidence: bool, needs_memory: bool = False) -> Any:
    """造一份本轮深度评估结果，键名走契约声明而不是手抄字面量。

    参数:
        depth: 本轮处理深度
        requires_paper_evidence: 本轮回答是否必须有论文 Evidence 支撑
        needs_memory: 本轮要不要翻项目研究记忆
    返回:
        TurnDepthAssessment 形状的字典
    """
    return TurnDepthAssessment(
        depth=depth,
        reason_code=_DEPTH_REASON_CODES[depth],
        requires_paper_evidence=requires_paper_evidence,
        resolved_query=_QUESTION,
        needs_project_memory=needs_memory,
    )


def _graph_input(*, depth: str, requires_paper_evidence: bool, memory: dict[str, Any] | None = None) -> dict[str, Any]:
    """造进图的初始状态：深度已定，首次分析据此早返回，不再调意图解析与深度评估。

    深度与深度评估结果都必须在 RAGState 里声明过，否则进图入口就被丢掉，
    analyze 会当成首轮去调真实意图解析链，本文件的替身会立刻报错——
    这正是「手搓 dict 假绿」那个坑的反向保险。

    参数:
        depth: 本轮已定的处理深度
        requires_paper_evidence: 本轮回答是否必须有论文 Evidence 支撑
        memory: 本轮记忆读取结果，None 表示本轮没读记忆
    返回:
        可直接交给已编译子图 ainvoke 的状态字典
    """
    state: dict[str, Any] = {
        "user_query": _QUESTION,
        "structured_query": {"paper_ids": []},
        "turn_depth": depth,
        "turn_depth_assessment": _assessment(
            depth=depth,
            requires_paper_evidence=requires_paper_evidence,
            needs_memory=memory is not None,
        ),
    }
    if memory is not None:
        state["memory_read_result"] = memory
    return state


def _init_stub(_state: dict[str, Any]) -> dict[str, Any]:
    """rag_init 替身：真实实现要探 Milvus 索引状态，本文件只审 planner 段的分流。"""
    return {}


async def _write_back_stub(_state: dict[str, Any]) -> dict[str, Any]:
    """write_back 替身：真实实现要写库与观测层，四态派生在用例里单独真实调用。"""
    return {}


@dataclass
class _GraphRun:
    """一次真实子图运行的产物：终态 + 两个不该被走到的节点的调用记录。"""

    final_state: dict[str, Any]
    validate: _NodeSpy
    execute: _NodeSpy


def _run_graph(state: dict[str, Any], *, llm: Any) -> _GraphRun:
    """用真实装配的子图跑一轮，validate 与 execute 换成只计数的探针。

    探针必须在 create_rag_node() 之前替换：add_node 装配时就把函数对象抓走了，
    编译之后再打补丁改不到图里那份引用，「一次都没被调用」会变成永真断言。

    参数:
        state: 进图的初始状态
        llm: analyze 与 synthesize 共用的 LLM 替身
    返回:
        _GraphRun：图的终态与两个探针
    """
    validate = _NodeSpy("planner_validate")
    execute = _NodeSpy("planner_execute")
    replacements = {
        "rag_init_node": _init_stub,
        "write_back_node": _write_back_stub,
        "planner_validate_node": validate,
        "planner_execute_node": execute,
        "_llm": lambda: llm,
        "_settings": _LoopSettings,
        # 本文件只审规划前分流：探测适配器映射换成空表关掉探测，别让探测那条支线
        # 把「该进 validate 的回合」的判据带进 Pydantic 之外的真实取数路径
        "_planner_optional_adapters": lambda _state: {},
    }
    with ExitStack() as stack:
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(rag_node, name, replacement))
        final_state = asyncio.run(rag_node.create_rag_node().ainvoke(state))
    return _GraphRun(final_state=dict(final_state), validate=validate, execute=execute)


def _memory_result(status: str, *, with_items: bool = True) -> dict[str, Any]:
    """造一份记忆读取结果。

    参数:
        status: 记忆读取四态之一
        with_items: 是否带一条真实条目，用来分开「读成功且有条目」与「读成功但空表」
    返回:
        memory_read_result 形状的字典
    """
    if not with_items:
        return {"status": status, "items": []}
    return {
        "status": status,
        "items": [
            {
                "memory_id": _MEMORY_ID,
                "title": "先补召回再谈重排",
                "body_summary": "上轮定的方向：先把召回补齐。",
                "provenance_label": "项目判断",
                "memory_type": "decision_note",
                "maturity": "verified",
            }
        ],
    }


def _derive_turn_outcome(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """走真实派生链算出本轮终态：先真实评写回，再由写回终态组装四态表。

    两步都用产品实现：写回那侧 minimal 深度在深度闸门就返回 skipped，不碰库也不调模型，
    所以这里拿到的是真链路结果，不是替身。

    参数:
        state: 图跑完的终态
    返回:
        (turn_outcome, memory_writeback)
    """
    memory_writeback = asyncio.run(write_back_turn_memory(state, request_id=_REQUEST_ID))
    outcome = build_turn_outcome(state, memory_writeback=memory_writeback)
    return dict(outcome), dict(memory_writeback)


def _minimal_run() -> _GraphRun:
    """跑一轮标准 minimal 直答回合：不要论文证据、不读记忆。"""
    return _run_graph(
        _graph_input(depth=_DEPTH_MINIMAL, requires_paper_evidence=False),
        llm=_DirectAnswerLLM(),
    )


class TestCapabilityTableStaysCompleteOnMinimalTurn:
    """约束 1：minimal 直答回合的四态能力表里，论文取证那一项必须真的在表里。

    换算表 `_PAPER_RETRIEVAL_CAPABILITY` 是白名单式的：`source_status.planner` 的取值
    不在表里，`_capability_results` 就整条不写这一项，而且不报错、不留日志。
    结果是「本轮没取论文证据」在评测里读成「压根没有这个能力」，
    而「没跑过」与「跑过且为空」是两件不同的事（AC :361 要的正是四态可见）。

    实现侧只写了 `source_status.planner="skipped"`，写没写对要由派生侧说话，
    所以这里跑真实图产出终态，再交给真实的 build_turn_outcome 派生。
    """

    def test_paper_retrieval_capability_present_and_skipped(self) -> None:
        """论文取证这一项必须出现在四态表里，取值为 skipped。

        破了会怎样：这一项从表里消失，评测无法区分「没检索」与「检索到空」，
        AC :361 的四态可见就是假的，且没有任何运行时信号提示它消失了。
        """
        run = _minimal_run()

        outcome, memory_writeback = _derive_turn_outcome(run.final_state)
        capabilities = outcome["capability_results"]

        assert _CAPABILITY_PAPER_RETRIEVAL in capabilities
        assert capabilities[_CAPABILITY_PAPER_RETRIEVAL] == _STATUS_SKIPPED
        # 四态词表之外的取值同样会让下游读不懂，顺带钉住取值合法
        assert capabilities[_CAPABILITY_PAPER_RETRIEVAL] in VALID_CAPABILITY_STATUSES
        # 写回那侧照旧按深度闸门 skipped：minimal 回合不评估候选
        assert memory_writeback["status"] == _STATUS_SKIPPED

    def test_planner_status_is_a_key_the_conversion_table_knows(self) -> None:
        """终态里 planner 的取值必须在换算表里有对应项。

        破了会怎样：改一个词（例如把 skipped 写成 not_executed）就让这一项静默消失，
        而单看短路增量里的 source_status 一切正常，问题只在派生侧显形。
        """
        run = _minimal_run()

        planner_status = run.final_state["source_status"]["planner"]

        assert planner_status in writeback._PAPER_RETRIEVAL_CAPABILITY

    def test_memory_read_capability_reports_skipped_when_never_asked(self) -> None:
        """本轮压根没问记忆时，记忆读取这一项按 skipped 进表，不与 empty 混淆。

        破了会怎样：「没问过记忆」被记成「问过、项目里没有相关记忆」，
        评测会把一次没发生的读取算成一次空命中。
        """
        run = _minimal_run()

        outcome, _ = _derive_turn_outcome(run.final_state)

        assert outcome["capability_results"][_CAPABILITY_MEMORY_READ] == _STATUS_SKIPPED
        assert outcome["final_depth"] == _DEPTH_MINIMAL


class TestMinimalTurnNeverTouchesValidateOrExecute:
    """约束 2：端到端跑真实图，minimal 直答回合一次都不进 validate 与 execute。

    只断言路由函数的返回值证明不了这件事：路由说去 synthesize，图里那张条件边映射表
    要是没登记这个目标，或者别处还留着一条通往 validate 的边，运行时照样会走过去。
    所以这里用 create_rag_node() 真实装配，把两个节点换成只计数的探针。
    """

    def test_validate_and_execute_are_never_called(self) -> None:
        """两个节点的调用次数必须是 0。

        破了会怎样：本轮仍会发出论文检索、仍会校验一份不存在的计划，
        minimal 这一档等于没生效，而用户侧只看到答案，看不出多跑了两段。
        """
        run = _minimal_run()

        assert run.validate.calls == 0
        assert run.execute.calls == 0

    def test_direct_answer_actually_produced_the_answer(self) -> None:
        """短路不是把这一轮丢掉：直答仍要产出正文与 skipped 的取证状态。

        破了会怎样：跳过了 validate/execute 却也没答上，短路变成静默丢回合。
        """
        run = _minimal_run()

        assert run.final_state["answer"] == _ANSWER_TEXT
        assert run.final_state["source_status"]["planner"] == _STATUS_SKIPPED
        assert run.final_state["runtime_chain"] == "planner"

    def test_analyze_step_is_still_counted(self) -> None:
        """短路了 analyze 那一步照记：它真判过深度，也可能真读过记忆。

        破了会怎样：total_steps 这道兜底上界少算一步，循环步数在 trace 里与实际不符。
        """
        run = _minimal_run()

        assert run.final_state["planner_budget"]["total_steps"] == 1


class TestNoFakeResearchArtifactsOnMinimalTurn:
    """约束 3：AC :362 的外部可观测形式——不生成正式 Research Plan、不摆论文引用。

    这一档的产品口径是「普通交流不检索论文、不生成正式 Research Plan」。
    终态里只要留下计划或引用，前端与评测就会把这一轮显示成做过研究工作。
    """

    def test_no_research_plan_and_no_citation_bearing_blocks(self) -> None:
        """research_plan 为 None，定稿答案块为空。

        破了会怎样：回答卡把一份没执行过的计划显示成本轮研究过程，
        「不生成正式 Research Plan」这条 AC 只在提示词里成立、在终态里不成立。
        """
        run = _minimal_run()

        assert run.final_state["research_plan"] is None
        assert run.final_state["planner_final_blocks"] == []
        assert run.final_state["research_unit_results"] == []
        assert run.final_state["coverage_review"] is None

    def test_no_evidence_ids_anywhere_in_the_turn(self) -> None:
        """本轮没有任何论文证据 id：证据池空、终态引用列表空、尝试过的检索式为空。

        破了会怎样：出现一个查不到出处的引用 id，用户点开依据是空的，
        而评测会把这一轮算成有证据支撑的回答。
        """
        run = _minimal_run()
        outcome, _ = _derive_turn_outcome(run.final_state)

        assert run.final_state["evidence_records"] == []
        assert outcome["evidence_ids"] == []
        assert outcome["missing_evidence"] == []
        # attempted_queries / scoped_papers 两个键不在这里断言：载荷确实写了它们，
        # 但 AgentState 与 RAGState 都没声明这两个字段，进图就被丢掉，state 里压根取不到。
        # 这是早于本单元的存量情况（诚实收尾载荷一直这么写），要治得先定「折叠追溯」这个
        # 功能还要不要，属产品决策。改成断言它们不存在等于把这个缺陷锁死，所以两条都不写


class _DraftRecorder:
    """记下草案生成被调了几次，并返回一份可 model_dump 的草案替身。"""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, _state: dict[str, Any], _llm: Any) -> Any:
        self.calls += 1
        return _StubDraft()


@dataclass(frozen=True)
class _StubDraft:
    """草案替身：本文件只关心草案有没有被产出来。

    clarification 对齐 ResearchPlanDraft 契约：analyze 产完草案要读它判断这一轮
    是不是「先问用户」。替身少了这个字段，用例会红在 AttributeError 上，那不是被测行为。
    """

    clarification: Any = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"question": _QUESTION}


class TestPaperEvidenceRequirementBlocksShortCircuit:
    """约束 4：minimal 但本轮要论文 Evidence 时不许短路（prd :113）。

    深度门要是漏了这一半，取证失败的回合会退化成一次无证据直答：
    模型凭印象说出论文事实，而载荷看上去是一次正常回答，没有任何缺证据的信号。
    """

    def test_minimal_requiring_paper_evidence_goes_to_validate(self) -> None:
        """真实图上照旧进 validate，并且草案真的被生成了。

        破了会怎样：本该升到 lookup/research 去取证的回合被 minimal 直答截住，
        答案里的论文事实主张一条证据都没有。
        """
        recorder = _DraftRecorder()
        state = _graph_input(depth=_DEPTH_MINIMAL, requires_paper_evidence=True)

        with patch.object(pipeline, "_analyze_plan", recorder):
            run = _run_graph(state, llm=_DirectAnswerLLM())

        assert recorder.calls == 1
        assert run.validate.calls == 1

    @pytest.mark.parametrize("depth", [_DEPTH_LOOKUP, _DEPTH_RESEARCH])
    def test_deeper_depths_never_short_circuit(self, depth: str) -> None:
        """另两档照旧走计划链：短路只属于不需要论文事实的那一档。

        破了会怎样：查证与研究两档也跳过检索，整个取证闭环失效。
        """
        recorder = _DraftRecorder()
        state = _graph_input(depth=depth, requires_paper_evidence=True)

        with patch.object(pipeline, "_analyze_plan", recorder):
            run = _run_graph(state, llm=_DirectAnswerLLM())

        assert recorder.calls == 1
        assert run.validate.calls == 1


class TestMemoryIdentityConstraintOnDirectAnswer:
    """约束 5：直答用到记忆时，记忆是项目历史判断，不是引用来源。

    记忆段段首那句身份约束（非论文原文、不得作为引用来源）是唯一挡住
    「把用户自己的判断当文献证据写进带引用结论」的东西。另写一套渲染就绕掉它。
    另一半是准入：只有真读到条目才允许出现记忆段，读失败或读空时模型不能拿着
    一个空结果编一段「项目此前的判断」。
    """

    def test_memory_block_carries_identity_constraint(self) -> None:
        """读到记忆时，进模型的 prompt 里必须带真实渲染出的身份约束段。

        破了会怎样：记忆以无标识的裸文本进 prompt，模型分不清这是项目判断还是文献，
        于是把它写成带出处的结论。
        """
        llm = _DirectAnswerLLM()
        state = _graph_input(
            depth=_DEPTH_MINIMAL,
            requires_paper_evidence=False,
            memory=_memory_result(_READ_SUCCESS),
        )

        run = _run_graph(state, llm=llm)

        prompt = llm.prompts[0]
        assert _MEMORY_BLOCK_MARKER in prompt
        # 身份约束句必须是真实渲染函数产出的那一份，不是另写的近似措辞
        assert plan_prompt.render_project_memory_block(run.final_state) in prompt

    def test_memory_never_becomes_a_citation_source(self) -> None:
        """用了记忆也不产生引用：定稿块与证据 id 依旧为空，记忆只进记忆清单。

        破了会怎样：项目自己的判断被算成一条论文证据，
        用户看到一个点不开原文的引用，评测把它算成有文献支撑。
        """
        state = _graph_input(
            depth=_DEPTH_MINIMAL,
            requires_paper_evidence=False,
            memory=_memory_result(_READ_SUCCESS),
        )

        run = _run_graph(state, llm=_DirectAnswerLLM())
        outcome, _ = _derive_turn_outcome(run.final_state)

        assert run.final_state["planner_final_blocks"] == []
        assert run.final_state["evidence_records"] == []
        assert outcome["evidence_ids"] == []
        # 记忆命中记在自己那一栏，与证据分开
        assert outcome["memory_item_ids"] == [_MEMORY_ID]

    @pytest.mark.parametrize(
        ("status", "with_items"),
        [(_READ_FAILED, False), (_READ_EMPTY, False), (_READ_SUCCESS, False)],
        ids=("read_failed", "read_empty", "success_but_no_items"),
    )
    def test_memory_block_absent_unless_success_with_items(self, status: str, with_items: bool) -> None:
        """只有 status==success 且 items 非空才允许出现记忆段。

        破了会怎样：模型拿着一个失败或空的读取结果，仍被要求参考「项目此前的判断」，
        于是编一段出来，而这段话与真读到的记忆看上去一模一样。
        """
        llm = _DirectAnswerLLM()
        state = _graph_input(
            depth=_DEPTH_MINIMAL,
            requires_paper_evidence=False,
            memory=_memory_result(status, with_items=with_items),
        )

        _run_graph(state, llm=llm)

        assert _MEMORY_BLOCK_MARKER not in llm.prompts[0]


# 表示「上限 / 计数」的命名片段，出现在标识符里就说明那里有个数值闸门
_CAP_TOKENS = ("max", "min_", "limit", "cap_", "quota", "counter", "_times", "rounds")
# 表示「检索取证」的命名片段；与上一组同时命中即为 lookup 侧的检索次数上限
_LOOKUP_TOKENS = ("lookup", "retriev", "search")
# 被审的两个模块，路径按 app 包相对本文件定位，免得写死绝对路径
_AUDITED_MODULES = (
    "research_planner_pipeline.py",
    "research_planner_plan_prompt.py",
)
_INTERNAL_RESEARCH_DIR = (
    Path(__file__).resolve().parents[2] / "app" / "services" / "internal_research"
)


def _identifiers(path: Path) -> set[str]:
    """收集一个模块里所有可能承载数值闸门的标识符名。

    参数:
        path: 模块文件路径
    返回:
        模块级常量名、函数名、关键字实参名与属性名的集合
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


class TestNoLookupRetrievalCap:
    """约束 6：lookup 侧不许有任何检索次数上限或计数器（用户 2026-08-30 当场拍板）。

    这属全局规则明禁的「仅为掩盖问题添加边界规则和最大轮数」：证据不够的出路是
    升级深度或如实报缺口，不是数到第 N 次就截断。截断会把「没查够」伪装成「查完了」。

    断言按符号扫源码，不是在注释里写一句约定——注释拦不住下一个人加常量。
    """

    @pytest.mark.parametrize("module_name", _AUDITED_MODULES)
    def test_no_lookup_scoped_cap_identifier(self, module_name: str) -> None:
        """两个模块里不许出现同时带「上限/计数」与「检索」语义的标识符。

        破了会怎样：lookup 回合数到上限就停手，缺证据的回合看上去像正常收敛，
        用户拿到一个没查够的答案，而 trace 里没有任何缺口信号。
        """
        offenders = {
            name
            for name in _identifiers(_INTERNAL_RESEARCH_DIR / module_name)
            if any(token in name.lower() for token in _CAP_TOKENS)
            and any(token in name.lower() for token in _LOOKUP_TOKENS)
        }

        assert offenders == set()

    def test_depth_plan_guidance_carries_no_numeric_bound(self) -> None:
        """三档规划指引全是自然语言原则，不含任何数字。

        破了会怎样：指引里出现「最多 3 次检索」这类数字，模型按数字截断计划，
        数值上限就从提示词这一侧偷偷回来了。
        """
        for depth, lines in plan_prompt._DEPTH_PLAN_GUIDANCE.items():
            joined = " ".join(lines)
            assert not re.search(r"\d", joined), f"{depth} 档指引里出现了数字：{joined}"

    def test_lookup_guidance_keeps_escalation_as_the_way_out(self) -> None:
        """lookup 档的出路必须写成升级深度或报缺口，而不是停手。

        破了会怎样：这一档没有出路描述，模型遇到证据不够只能自己编一个收尾，
        「证据不足要如实报」这条产品口径在提示词里就不存在了。
        """
        joined = " ".join(plan_prompt._DEPTH_PLAN_GUIDANCE[_DEPTH_LOOKUP]).lower()

        assert "escalate" in joined
        assert "report the gap" in joined
