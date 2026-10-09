"""回合处理深度评估：模型给偏好，确定性硬约束说最终算不算数。

深度是循环内的策略参数，决定初始动作预算与证据要求强度，不是三条独立入口路径
（design.md §6.2）。判断依据是「完成本轮回答需要什么」，不是按关键词套固定类别，
所以深度由模型判断；但模型偏好越界时直接暴露决策失败，不静默退回默认深度
（design.md §5.2）。

同一次结构化调用还顺带判断两件事：「这一轮要不要翻项目研究记忆」与「要不要先问用户
一句再往下走」。读记忆与向用户澄清都是循环内的按需动作（design.md §6.2），判它们要不要
发生本身也是模型判断，搭在这次调用上是为了不给用户多加一次模型等待。

作者：xxx
创建时间：2026-08-29 00:00:00
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage

from app.middleware.trace_context import trace_label
from app.models.research_contracts import TurnDepthAssessment
from app.models.research_memory_contracts import VALID_MEMORY_TYPES
from app.models.research_planner_contracts import (
    ProjectMemoryReadRequest,
    TurnDepthProposal,
)
from app.services.internal_research.research_planner_plan_prompt import (
    render_conversation_context,
)

# 构成「明确的多论文比较」的最小论文数：两篇起才谈得上比较
_MULTI_PAPER_COMPARE_MIN = 2
# 规则意图分类里代表比较诉求的意图值
_COMPARE_INTENT = "compare"


class TurnDepthAssessmentError(ValueError):
    """模型给出的深度偏好越过确定性硬约束。

    携带稳定 reason_code，供上层映射成可观测的 depth_assessment=failed。
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


async def assess_turn_depth(
    state: dict[str, Any],
    llm: Any,
    *,
    scoped_paper_ids: frozenset[str] | None,
    research_questions: Sequence[Mapping[str, Any]] = (),
) -> TurnDepthAssessment:
    """评估本回合处理深度，并用确定性硬约束校验模型偏好。

    Args:
        state: 当前 RAGState 字典态，须已完成意图与焦点解析
        llm: 支持 with_structured_output 的 LLM
        scoped_paper_ids: 本回合授权可用的论文集合，None 或空表示项目内没有可用论文
        research_questions: 本项目已接纳的研究问题（question_id + title），
            供模型判断要不要按研究问题收窄记忆读取；空表示项目内没有研究问题可选

    Returns:
        TurnDepthAssessment: 深度、原因码、论文 Evidence 要求与归一化问题，
            外加本轮是否翻项目记忆、是否先向用户澄清及各自的原因码

    Raises:
        TurnDepthAssessmentError: 模型偏好越过硬约束时抛出，由调用方暴露为决策失败
    """
    proposal = await _propose_depth(state, llm, scoped_paper_ids, research_questions)
    _enforce_depth_constraints(
        proposal,
        scoped_paper_count=len(scoped_paper_ids or frozenset()),
        explicit_multi_paper_compare=is_explicit_multi_paper_compare(state),
    )
    return _to_assessment(proposal)


def _to_assessment(proposal: TurnDepthProposal) -> TurnDepthAssessment:
    """把过了硬约束的模型偏好落成本回合的深度、记忆读取与澄清判断。

    Args:
        proposal: 已通过确定性硬约束校验的模型偏好

    Returns:
        TurnDepthAssessment: 深度四项，加上本轮是否翻记忆、收窄条件与记忆读取原因码，
            再加本轮要不要先问用户、问哪一句、为什么问
    """
    memory = proposal.project_memory
    # 1. 模型没提 project_memory 或判 needed=False，都按「本轮不翻」落值：
    #    这是模型判断的正常结果，不是评估失败，所以不抛错也不补默认收窄条件
    needed = memory is not None and memory.needed
    clarification = proposal.clarification
    # 2. 澄清同理：没提或判 needed=False 都是「目标已经足够明确」这个正常结论。
    #    问题文本只在真要问时才带出去，免得模型顺手写了一句却没判要问，
    #    下游把它当成本轮回答返回给用户
    asking = clarification is not None and clarification.needed
    return TurnDepthAssessment(
        depth=proposal.depth,
        reason_code=proposal.reason_code,
        requires_paper_evidence=proposal.requires_paper_evidence,
        resolved_query=proposal.resolved_query,
        needs_project_memory=needed,
        memory_narrowing=_memory_narrowing(memory) if needed else {},
        # 3. 不翻时模型给的原因码照样留下（例如 no_project_history_needed），
        #    否则 trace 里分不清「模型判断用不上」和「模型根本没判过这件事」
        memory_read_reason_code=memory.reason_code if memory is not None else "",
        needs_clarification=asking,
        clarification_question=clarification.question if asking else "",
        # 4. 判不用问时的原因码同样留下：与记忆那一项同一条理由，
        #    trace 上要分得清「判过、不用问」和「根本没判过要不要问」
        clarification_reason_code=clarification.reason_code if clarification is not None else "",
    )


def _memory_narrowing(request: ProjectMemoryReadRequest) -> dict[str, Any]:
    """收集模型真正指定了的记忆读取收窄条件。

    Args:
        request: 模型给出的记忆读取请求，needed 已为 True

    Returns:
        dict[str, Any]: 只含有真值的收窄键；空串、None、False 一律不入表，
            免得记忆层把「模型没指定这一维」当成「指定为空值」而查不到任何判断
    """
    candidates = (
        ("paper_id", request.paper_id),
        ("question_id", request.question_id),
        ("memory_type", request.memory_type),
        ("follow_links", request.follow_links),
    )
    return {key: value for key, value in candidates if value}


def is_explicit_multi_paper_compare(state: dict[str, Any]) -> bool:
    """判断本回合是不是明确的多论文比较。

    只认确定性信号：compare 意图、焦点解析未退回或失败，且对象至少两篇。
    项目检索范围与未解析指代不是明确对象，缺少解析原因也不提供确认依据。

    Args:
        state: 已完成意图与焦点解析的状态

    Returns:
        bool: 是明确的多论文比较则为 True
    """
    if str(state.get("intent") or "") != _COMPARE_INTENT:
        return False
    focus = state.get("focus")
    if not isinstance(focus, dict):
        return False
    # 1. 【研究规划】【比较对象】只有空原因码确认对象已解析，保留项目范围供后续判断
    if focus.get("fallback_reason") != "":
        return False
    turn_focus = focus.get("turn_focus")
    if not isinstance(turn_focus, list):
        return False
    return len(turn_focus) >= _MULTI_PAPER_COMPARE_MIN


def _enforce_depth_constraints(
    proposal: TurnDepthProposal,
    *,
    scoped_paper_count: int,
    explicit_multi_paper_compare: bool,
) -> None:
    """校验三条确定性硬约束，越界抛错而不修正模型输出。

    Args:
        proposal: 模型给出的深度偏好
        scoped_paper_count: 本回合授权论文数量
        explicit_multi_paper_compare: 本回合是否为明确的多论文比较

    Raises:
        TurnDepthAssessmentError: 命中任一硬约束
    """
    # 1. 明确的论文事实问题不得判 minimal：要 Evidence 就不能走不检索论文的深度
    if proposal.depth == "minimal" and proposal.requires_paper_evidence:
        raise TurnDepthAssessmentError(
            "minimal_requires_paper_evidence",
            "minimal 深度不检索论文，不能同时要求论文 Evidence",
        )
    # 2. 空论文范围不得要求论文 Evidence：项目里没有论文时这个要求永远无法满足
    if scoped_paper_count == 0 and proposal.requires_paper_evidence:
        raise TurnDepthAssessmentError(
            "empty_scope_requires_paper_evidence",
            "本回合授权论文范围为空，不能要求论文 Evidence",
        )
    # 3. 明确的多论文比较不得降为无证据直答
    if not explicit_multi_paper_compare:
        return
    if proposal.depth == "minimal" or not proposal.requires_paper_evidence:
        raise TurnDepthAssessmentError(
            "multi_paper_compare_without_evidence",
            "明确的多论文比较不能降为无证据直接回答",
        )


async def _propose_depth(
    state: dict[str, Any],
    llm: Any,
    scoped_paper_ids: frozenset[str] | None,
    research_questions: Sequence[Mapping[str, Any]] = (),
) -> TurnDepthProposal:
    """调模型产出结构化深度偏好。结构化失败按原样抛出，由上层映射失败码。"""
    assessor = llm.with_structured_output(TurnDepthProposal)
    with trace_label("planner_analyze", "turn_depth"):
        return await assessor.ainvoke([
            HumanMessage(
                content=build_depth_prompt(
                    state, scoped_paper_ids, research_questions=research_questions
                )
            )
        ])


def build_depth_prompt(
    state: dict[str, Any],
    scoped_paper_ids: frozenset[str] | None,
    *,
    research_questions: Sequence[Mapping[str, Any]] = (),
) -> str:
    """构造深度评估 prompt。

    Args:
        state: 已完成意图与焦点解析的状态
        scoped_paper_ids: 本回合授权论文集合
        research_questions: 本项目已接纳的研究问题，空则整段不出现

    Returns:
        str: 交给结构化输出模型的完整提示词
    """
    scoped = sorted(scoped_paper_ids or frozenset())
    return "\n".join([
        *_DEPTH_RULE_LINES,
        "",
        f"Question: {state.get('user_query', '')}",
        "Conversation context:",
        render_conversation_context(state),
        f"Coarse intent: {state.get('intent', '')}",
        f"Focus: {state.get('focus') or {}}",
        *_paper_list_lines(state),
        f"Authorized paper count: {len(scoped)}",
        f"Authorized paper ids: {scoped}",
        *_research_question_lines(research_questions),
    ])


def _research_question_lines(research_questions: Sequence[Mapping[str, Any]]) -> list[str]:
    """渲染本项目的研究问题清单，供模型按研究问题收窄记忆读取。

    项目里没有研究问题时整段不出现：摆一个空清单等于告诉模型「这个项目没有研究问题」，
    而它区分不出「没有」和「这一轮没查到」。

    Args:
        research_questions: 已接纳的研究问题，每条带 question_id 与 title

    Returns:
        list[str]: 段首说明 + 每条一行 `id -> title`；无问题时为空列表
    """
    listed = [
        f"- {str(item.get('question_id') or '')} -> {str(item.get('title') or '')}"
        for item in research_questions
        if str(item.get("question_id") or "").strip()
    ]
    if not listed:
        return []
    return ["Research question ids (id -> title):", *listed]


def _paper_list_lines(state: dict[str, Any]) -> list[str]:
    """读取预装载的论文清单并渲染为提示词行。

    Args:
        state: 已并入 _paper_list_lines 的状态

    Returns:
        list[str]: 论文清单行，无数据时为空列表
    """
    lines = state.get("_paper_list_lines")
    if not lines:
        return []
    return ["", "Authorized papers (id | title | index status):", *lines, ""]


# 深度定义与硬约束的正面表述，来自 prd.md §5-§6；这段同时是「这一轮要不要翻项目记忆」
# 与「要不要先向用户澄清」的判定标准，改它等于改这三件事的产品判定标准
_DEPTH_RULE_LINES = (
    "Choose the processing depth for this turn of a project-scoped research assistant.",
    "Judge by what answering this turn actually requires, never by keyword-to-category mapping.",
    "",
    "depth=minimal: no paper fact is needed. Plain conversation, task clarification, "
    "organizing a research question, rewriting text, project management, recalling "
    "what the project already concluded, or a general concept the user brings up on its "
    "own - a textbook definition, how a well-known method works - which is answered from "
    "general knowledge without the project's papers.",
    "depth=lookup: a paper fact is needed and one local question can be answered with a "
    "small amount of direct evidence, e.g. how the project's papers define or use "
    "something, a method detail, a metric, a paragraph location, or paper metadata such "
    "as title or authors. A question belongs here only when it points at the papers - "
    "what this paper says, how the project's papers handle something - not merely "
    "because the topic also appears in them.",
    "depth=research: the turn must be split into interdependent sub-questions, compares "
    "several papers, resolves conflicting evidence, or needs extra evidence rounds.",
    "",
    "Hard constraints. A proposal violating any of them is rejected as a decision failure, "
    "so satisfy them instead of guessing a safe default:",
    "- A concrete paper-fact question must not be minimal.",
    "- With zero authorized papers, requires_paper_evidence must be false.",
    "- An explicit comparison over two or more authorized papers must not become an "
    "evidence-free direct answer: it must not be minimal and requires_paper_evidence "
    "must be true. Prefer research for it.",
    "",
    "Set requires_paper_evidence=true only when the answer must cite verified paper "
    "evidence from this turn.",
    "Set reason_code to a short stable snake_case code describing why this depth fits, "
    "e.g. single_paper_metadata_lookup, cross_paper_comparison, no_paper_fact_needed.",
    "Set resolved_query to a self-contained rewrite of Question: resolve pronouns and "
    "elliptical follow-ups using Conversation context and Focus, and keep the user's "
    "language. Never widen or narrow what the user asked.",
    "",
    "Also decide whether this turn has to read the project research memory and set "
    "project_memory.needed accordingly. That memory holds the judgments the user has "
    "built up inside this project - decisions, leanings, working hypotheses and open "
    "questions. It is not paper text and must never be used as a citation source.",
    "Read it when the user asks what direction, conclusion or decision this project "
    "already settled on, when a follow-up continues an earlier judgment, when the answer "
    "has to build on what the project already believes, or when answering blind could "
    "contradict a decision already made.",
    "Do not read it for a plain paper-fact question, for a one-off question unrelated to "
    "this project's own judgments, or for pure text rewriting.",
    "All three depths may read it. A minimal turn asking what direction we settled on "
    "still needs it: minimal only means no paper retrieval, not no project history.",
    "Narrow the read only where the narrowing is certain, and leave the other keys out: "
    "paper_id must be one of the Authorized paper ids listed below, memory_type is set "
    "only when a single kind of judgment is wanted (past decisions only -> decision_note), "
    "follow_links is set only when the answer needs one more hop over "
    "based_on / conflicts_with / supersedes relations.",
    "Set question_id only to one of the Research question ids listed below, and only when "
    "the user is clearly asking about that research question. Leave it empty otherwise and "
    "never invent one: an id that is not on that list matches no memory and silently turns "
    "into an empty read, which reads back as 'this project has no such judgment yet'. "
    "When no Research question ids section appears below, this project has none, so leave "
    "question_id empty.",
    f"Valid memory_type values: {', '.join(VALID_MEMORY_TYPES)}.",
    "Set project_memory.reason_code to a short stable snake_case code for why this turn "
    "does or does not need the project memory, e.g. recall_prior_decision, "
    "continue_prior_direction, no_project_history_needed.",
    "",
    "Also decide whether this turn has to ask the user one clarifying question before "
    "doing anything else, and set clarification.needed accordingly.",
    "Ask only when you cannot determine what the user is after AND the readings you are "
    "torn between would lead to visibly different work - a different paper searched, a "
    "different question answered, a different thing produced.",
    "If you can determine the goal, you must not ask. Asking is itself a decision, not a "
    "safe default: an unneeded question costs the user a whole round trip and stalls work "
    "they already described well enough to act on.",
    "Do not ask because the evidence may turn out to be thin, because the project has no "
    "papers, or because the question is broad. A thin-evidence turn looks first and then "
    "reports the gap; a broad question is answered at the level it was asked.",
    "Do not ask for plain conversation, for text rewriting, or for a straightforward "
    "recall of what this project already decided.",
    "If Conversation context shows you already asked a clarifying question on the "
    "previous turn, do not ask again: work with what the user has given you since.",
    "When you do ask, write clarification.question as one short question in the user's "
    "own language, naming the concrete readings you are choosing between. It is returned "
    "to the user verbatim as this turn's whole reply, so never mention depths, retrieval, "
    "project memory, research plans or any other internal component.",
    "Set clarification.reason_code to a short stable snake_case code for what exactly is "
    "undetermined, e.g. ambiguous_target_paper, unclear_comparison_dimension, "
    "unclear_requested_output.",
    "Keep filling depth, requires_paper_evidence and resolved_query as usual even when "
    "you ask: the hard constraints above still apply to them.",
)
