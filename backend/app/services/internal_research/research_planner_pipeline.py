"""
内部 RAG planner 分支总编排层。

这层不直接决定图路由，也不关心 HTTP 返回结构，
只负责把 planner 主流程稳定串起来：
1. analyze：让 LLM 产出草案计划
2. validate：验证策略并派生可执行 research_plan
3. execute：调度证据适配器执行研究单元
4. synthesize：综合最终回答和 planner 元数据
"""

from __future__ import annotations

import asyncio
import logging
import json
from urllib.parse import urlsplit
from typing import Any, Awaitable, Callable, Mapping

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.middleware.trace_context import evidence_scope, record_evidence_observation, trace_label
from app.models.research_planner_contracts import (
    CURRENT_RESEARCH_CONTRACT_VERSION,
    EvidenceSufficiencyReport,
    PlannerFailure,
    PlannerCoverageReview,
    PlannerValidationResult,
    ResearchPlan,
    ResearchUnitResult,
)
from app.repositories.paper_repository import get_paper_repository
from app.services.graph.research_object_query_service import get_research_object_query_service
from app.services.milvus_store import get_milvus_store
from app.services.llm_service import ProviderHTTPError, provider_failure_message
from app.services.internal_research.project_memory_reader import (
    read_project_memory,
    read_project_research_questions,
)
from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft, ResearchReplanStrategyDraft
from app.models.research_action_contracts import InitialResearchDecisionOutput, ResearchDecision, ResearchDecisionOutput
from app.services.internal_research.research_action_validation import (
    action_work_keys, project_decision_output, record_action_validation, validate_actions,
)
from app.services.internal_research.research_planner_plan_prompt import build_action_prompt
from app.services.research_memory_stale_service import MemoryCheckedModel
from app.services.internal_research.research_planner_claim_provider import (
    LLMClaimDraftProvider,
)
from app.services.internal_research.research_planner_consumer_budget import CLAIM_PROMPT_VERSION
from app.services.internal_research.research_planner_conversion import (
    build_clarification_response,
    build_planner_failure_response,
    migrate_execution_checkpoint,
    PlannerFailureReadFacts,
    project_planner_failure_read_facts,
)
from app.services.internal_research.research_planner_executor import (
    EvidenceAdapter,
    EvidenceCapabilityError,
    PlannerExecutionContext,
    ResearchPlannerExecutor,
)
from app.services.internal_research.research_planner_evidence_repair import (
    execute_with_repair,
    build_evidence_sufficiency_report,
    PlannerEvidenceRepairOutcome,
)
from app.services.internal_research.research_planner_execution_state import (
    EXECUTION_STATE_VERSION, ExecutionIdentity, PlannerExecutionState, ReadReceipt,
)
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from app.services.internal_research.research_planner_final_answer import LLMPlannerFinalAnswerProvider
from app.services.internal_research.research_answer_review import (
    LLMResearchAnswerReviewer, bind_rendered_review_answer, review_requires_continuation,
)
from app.services.internal_research.research_planner_final_delivery import (
    delivery_state_fields,
    make_terminal_delivery_report,
    project_final_delivery,
)
from app.services.internal_research.research_planner_direct_answer import (
    build_direct_answer_response,
)
from app.services.internal_research.research_planner_plan_prompt import (
    render_conversation_context,
    render_project_memory_block,
)
from app.services.internal_research.research_planner_synthesis import (
    synthesize_planner_answer_async,
)
from app.services.internal_research.turn_context_resolver import TURN_PAPERS_KEY, resolve_turn_context
from app.services.internal_research.turn_depth_assessor import (
    TurnDepthAssessmentError,
    assess_turn_depth,
)
from app.services.internal_research.research_planner_validation_rules import fulfillment_contract_issues
from app.services.internal_research.research_planner_validator import (
    PlannerValidationOptions,
    validate_or_repair_plan,
)
from app.models.research_planner_loop_contracts import (
    PlannerDecision,
)
from app.services.internal_research.research_planner_loop_controller import (
    decide_next,
    decide_action_admission,
    observe_execution,
    observe_validation,
    fact_digest,
    material_fact_keys,
)
from app.services.internal_research.research_planner_loop_state import (
    advance_and_serialize,
    read_budget,
    read_decision,
)

logger = logging.getLogger(__name__)


class MemoryOnlyTurnReadError(RuntimeError):
    """项目记忆是本轮唯一信息来源，但记忆读取失败。

    只在「不检索论文、也不要求论文 Evidence、却必须翻历史判断」的回合抛出：这时没有
    第二条信息来源，继续往下走只会让模型编出项目此前的判断。普通论文问答的记忆读取
    失败不走这里，由主链带着 failed 状态继续。

    Attributes:
        error_kind: 记忆读取返回的失败类型，随失败载荷一起暴露供重试判断
    """

    def __init__(self, error_kind: str) -> None:
        """参数为记忆读取真实错误类型；初始化唯一来源不可用异常，无返回值。"""
        super().__init__(f"项目记忆是本轮唯一信息来源，读取失败（error_kind={error_kind}）")
        self.error_kind = error_kind


class ContextLoadError(RuntimeError):
    """预装载上下文时发生基础设施级故障，重试后仍不成立。

    只有库连不上、查询超时这类真故障走这里。数据本身为空（论文没摘要、没章节、
    图库无关系）是如实的空，照常返回空段，不是失败。图谱面也不抛这个错——
    它单独降级为该段写「无」。抛出即意味着这一轮不能继续规划：带着「没有论文」
    的假清单往下走，模型会据一份错的项目现状铺计划。

    Attributes:
        face: 失败面（论文库 / 章节查询 / 图谱），让调用方一眼看出是哪一层塌了
        detail: 可读失败原因，随失败载荷暴露给上层
    """

    def __init__(self, face: str, detail: str) -> None:
        """参数为上下文失败面及真实原因；初始化可追查的装载异常，无返回值。"""
        super().__init__(f"预装载上下文失败（{face}）：{detail}")
        self.face = face
        self.detail = detail


async def analyze_planner_state(
    state: dict[str, Any],
    llm: Any,
    settings: Any,
) -> dict[str, Any]:
    """参数为本次图状态、模型与配置；返回当前动作或澄清及完整状态增量。

    首次解析焦点与深度；回边保留稳定要求、材料及尝试。每次只选择当前动作，
    执行结果回填后再继续研究。失败按原身份终止，不用空计划隐藏问题。
    """
    # 1. 读旧步数快照（首轮为空快照）
    migration = migrate_execution_checkpoint(state)
    state = {**state, **migration}
    budget = read_budget(state)
    try:
        # 首轮先解析意图与焦点、判本轮处理深度；replan 重入时深度已定，不再重判
        turn_delta = await _first_analysis_delta(state, llm)
    except Exception as exc:
        # 深度评估越界、记忆读不出来、意图解析炸掉都在这里落成显式失败载荷
        return _research_failure_payload(state, _first_analysis_failure_delta(budget, exc), stage="analyze",
                                         snapshot=getattr(exc, "planner_execution_snapshot", {}))
    turn_delta = {**migration, **turn_delta}
    resolved_state = {**state, **turn_delta}
    # 2. 目标没确认：先把问题问回去，不铺计划也不发检索 —— 这一步必须在直答与产草案之前，
    #    落到后面就等于先按某一种理解把活干完，再问用户是不是这个意思
    clarification = _clarification_question(resolved_state)
    if clarification:
        return _clarification_delta(budget, turn_delta, clarification)
    # 3. 本轮不需要论文事实：analyze 到此为止，不铺计划、不发检索，交给 synthesize 直答。
    #    预装载好的上下文段跟着 turn_delta 一起带出去：调用方要能看出本轮范围里有哪些论文
    if _is_direct_answer_turn(resolved_state):
        return _minimal_short_circuit_delta(budget, turn_delta)
    # 4. 【研究循环】【选择行动】正式读取不再等待完整未来计划，模型始终消费累计实际结果
    try:
        # 4. 【研究循环】【统一决策】读取与分析都通过正式动作入口执行，不再强制预探测
        draft = await _analyze_plan(resolved_state, llm)
    except Exception as exc:
        # 显式失败载荷 + terminate_fail 决策，直接走 write_back 终态，
        # 不能只清空决策——那会让路由把带失败载荷、无 draft 的态误送进 validate
        failure = _failure_from_exc(exc, "planner_analysis_failed")
        payload = _analyze_failure_delta(budget, failure, reason="planner analysis structured failure")
        return _research_failure_payload(resolved_state, payload, stage="analyze",
                                         snapshot=getattr(exc, "planner_execution_snapshot", {}))
    # 6. 出计划那一步判出「得先问用户」：这句问题就是本轮回答，不铺计划也不发检索。
    #    它排在产草案之后是因为只有草案能表达这个判断；与首次分析那个澄清走同一条
    #    载荷与终止决策，不另造一套
    draft_clarification = (draft.decision.clarification.question
                           if isinstance(draft, ResearchDecision) and draft.decision.kind == "clarify"
                           else "" if isinstance(draft, ResearchDecision) else _draft_clarification_question(draft))
    if draft_clarification:
        delta = _clarification_delta(budget, turn_delta, draft_clarification)
        if resolved_state.get("research_plan"):
            # 5. 【研究循环】【途中澄清】保留已经执行的事实，澄清不清空累计成果
            preserved = {key: resolved_state[key] for key in ("research_plan", "research_unit_results", "coverage_review",
                         "evidence_records", "planner_read_receipts", "evidence_sufficiency_report") if key in resolved_state}
            delta = {**delta, **preserved, "planner_draft": draft.model_dump(mode="json"),
                     "source_status": {**(resolved_state.get("source_status") or {}), "planner": "clarification"}}
        return delta
    # 7. 思考步产出 draft 成功：只 +1 总步数并留痕（decision=None 清空上一轮决策，避免陈旧值误导路由）
    loop_delta = advance_and_serialize(
        budget=budget, stage="analyze", decision=None, observation=None,
    )
    return {
        **turn_delta,
        "planner_draft": draft.model_dump(mode="json"),
        "runtime_chain": "planner",
        "source_status": {"planner": "analyzed"},
        **loop_delta,
    }


def _draft_clarification_question(draft: ResearchPlanStrategyDraft | ResearchReplanStrategyDraft) -> str:
    """取草案里要问用户的那一句；草案判不用问时返回空串。

    判据只认 clarification.needed：那是模型在信任边界内给出的判断，问题文本与原因码
    由 TurnClarificationRequest 一并校验。光看有没有问题文本分不出「模型判要问」与
    「模型顺手写了一句但判的是不用问」。

    参数:
        draft: 本次产出的计划草案
    返回:
        str: 要问用户的问题；草案判不澄清时为空串
    """
    clarification = draft.clarification
    if clarification is None or not clarification.needed:
        return ""
    return str(clarification.question or "")


def _clarification_question(state: Mapping[str, Any]) -> str:
    """取本轮要问用户的那一句澄清问题；不澄清时返回空串。

    判据只认深度评估结果里的 needs_clarification：那是模型在信任边界内给出的判断，
    带问题文本与原因码一起过校验。光看有没有问题文本分不出「模型判要问」和
    「模型顺手写了一句但判的是不用问」。

    参数:
        state: 已并入首次分析增量的状态，取 turn_depth_assessment
    返回:
        str: 要问用户的问题；本轮不澄清则为空串
    """
    assessment = state.get("turn_depth_assessment")
    if not isinstance(assessment, Mapping) or not assessment.get("needs_clarification"):
        return ""
    return str(assessment.get("clarification_question") or "")


def _clarification_delta(
    budget: Any,
    turn_delta: Mapping[str, Any],
    question: str,
) -> dict[str, Any]:
    """组装澄清回合的 analyze 增量：那句问题作为本轮回答 + terminate_honest 正常终止。

    决策用 terminate_honest 而不是 terminate_fail：目标没确认是正常判断结果，不是故障，
    这一轮该正常结束在一个问题上。它同时让记忆写回的终态闸把本轮挡在候选评估之外
    （prd.md §7.2 的「固定不触发」清单含澄清），不必为澄清另加一道闸。

    有意不产 planner_draft：本轮没铺计划，产了就得走校验与执行，澄清等于没发生。
    步数照记一步：analyze 这一步真判过深度、也可能真读过记忆。

    参数:
        budget: 推进前的循环步数快照
        turn_delta: 首次分析增量（意图与焦点、深度评估结果、按需读到的项目记忆）
        question: 要问用户的那一句问题
    返回:
        dict[str, Any]: 可直接作为节点返回值的澄清增量
    """
    loop_delta = advance_and_serialize(
        budget=budget,
        stage="analyze",
        decision=PlannerDecision(action="terminate_honest", reason="planner needs user clarification"),
        observation=None,
    )
    # turn_delta 放最前面：澄清载荷自带 source_status，后写的键覆盖前面的（2c 坑 3 同理）
    return {**turn_delta, **build_clarification_response(question=question), **loop_delta,
            "final_delivery_applicability": "clarification"}


def _is_direct_answer_turn(state: Mapping[str, Any]) -> bool:
    """判断本轮是不是「不需要论文事实」的直答回合。

    minimal 深度本就不检索论文，本轮又没要求论文 Evidence：这时铺研究计划、发检索、
    校验引用全是空转，analyze 之后直接进综合把答案说出来即可。

    只要本轮还要论文 Evidence 就不算直答回合，哪怕深度判成 minimal 也照旧产草案：
    答案里要出现论文事实主张的回合得升到 lookup/research 去真取证据，不能凭印象直接答。

    参数:
        state: 已并入首次分析增量的状态，取 turn_depth 与 turn_depth_assessment
    返回:
        bool: 本轮可以直答则为 True
    """
    if str(state.get("turn_depth") or "") != _MINIMAL_DEPTH:
        return False
    assessment = state.get("turn_depth_assessment")
    # 论文 Evidence 要求只落在深度评估结果里，没评估过（无 assessment）视为不要求
    return not (isinstance(assessment, Mapping) and assessment.get("requires_paper_evidence"))


def _minimal_short_circuit_delta(budget: Any, turn_delta: Mapping[str, Any]) -> dict[str, Any]:
    """组装 minimal 直答短路的 analyze 增量：带首次分析结果，但不带计划草案。

    有意不产 planner_draft：路由据「深度 minimal 且无草案」认出这一轮短路了，
    产了草案照旧会被送进 validate 与 execute，直答等于没发生。

    步数照记一步：analyze 这一步真判过深度、也可能真读过记忆，短路不等于没发生，
    漏记会让本回合的步数统计少算一步。

    参数:
        budget: 推进前的循环步数快照
        turn_delta: 首次分析增量（意图与焦点、深度评估结果、预装载上下文段、
            按需读到的项目记忆）
    返回:
        dict[str, Any]: 可直接作为节点返回值的短路增量
    """
    loop_delta = advance_and_serialize(budget=budget, stage="analyze", decision=None, observation=None)
    return {
        **turn_delta,
        "runtime_chain": "planner",
        "final_delivery_applicability": "minimal",
        # 本轮既没铺计划也没取论文证据，planner 落 skipped：写 analyzed 是谎报产出了草案，
        # 而 skipped 与直答载荷、四态换算表同取值，trace 里能读成「本轮没取论文证据」
        "source_status": {"planner": _PLANNER_SKIPPED_STATUS},
        **loop_delta,
    }


def _first_analysis_failure_delta(budget: Any, exc: Exception) -> dict[str, Any]:
    """把首次分析（意图解析 / 深度评估 / 记忆读取）抛出的异常映射成对应的失败增量。

    三类失败各带自己的可见状态位，都不退回默认深度、也不猜一份项目历史判断，
    一律显式失败让调用方能据错误码重试。

    参数:
        budget: 推进前的循环步数快照
        exc: 首次分析抛出的异常
    返回:
        dict[str, Any]: 可直接作为节点返回值的失败增量
    """
    # 1. 深度评估越过确定性硬约束：暴露决策失败，不退回任何默认深度（design §9）
    if isinstance(exc, TurnDepthAssessmentError):
        return _analyze_failure_delta(
            budget,
            _failure(_DEPTH_ASSESSMENT_FAILED_CODE, f"{exc.reason_code}: {exc}"),
            reason="planner depth assessment out of bounds",
            extra_source_status={"depth_assessment": "failed"},
        )
    # 2. 记忆是本轮唯一信息来源却读不出来：给出可重试失败，不让模型猜一份项目历史判断出来
    if isinstance(exc, MemoryOnlyTurnReadError):
        return _analyze_failure_delta(
            budget,
            _failure(_MEMORY_READ_FAILED_CODE, str(exc)),
            reason="project memory unreadable and it is the only source this turn",
            extra_source_status={"memory_read": "failed"},
        )
    # 3. 论文库或章节查询重试后仍读不出来：带失败面与可读原因显式失败。
    #    这里绝不能落成「本轮没有论文」——那等于把基础设施故障说成用户项目是空的
    if isinstance(exc, ContextLoadError):
        return _analyze_failure_delta(
            budget,
            _failure(_CONTEXT_LOAD_FAILED_CODE, f"{exc.face}：{exc.detail}"),
            reason=f"planner context preload failed on {exc.face}",
            extra_source_status={"context_load": "failed"},
            extra_source_status_reasons={"context_load": f"{exc.face}：{exc.detail}"},
        )
    # 4. 其它首次分析异常：沿用深度评估失败码，深度状态位同样标失败
    return _analyze_failure_delta(
        budget,
        _failure_from_exc(exc, _DEPTH_ASSESSMENT_FAILED_CODE),
        reason="planner depth assessment failure",
        extra_source_status={"depth_assessment": "failed"},
    )


async def _first_analysis_delta(state: dict[str, Any], llm: Any) -> dict[str, Any]:
    """产出首次分析增量：意图与焦点解析 + 处理深度评估 + 按需读项目记忆。

    深度只在首次分析时判一次。replan 回边重入 analyze 时深度已在 state 里，
    此后的深度变化走 DepthTransition，不在这里重新评估。研究问题清单也只在这里读一次，
    它是深度评估的输入（模型据它决定要不要按研究问题收窄记忆读取）。

    Args:
        state: 当前 RAGState 字典态
        llm: 深度评估使用的 LLM

    Returns:
        dict[str, Any]: 首轮返回意图/焦点/承接字段 + turn_depth_assessment + turn_depth，
            深度评估要求翻记忆时另带 memory_read_result；replan 重入时返回空增量

    Raises:
        MemoryOnlyTurnReadError: 记忆读取失败且记忆是本轮唯一信息来源
    """
    if str(state.get("turn_depth") or ""):
        return {}
    turn_delta = await resolve_turn_context(state, load_papers=_load_turn_papers_with_retry(state))
    # 焦点解析已按同一份 paper_ids 保序读过一次论文记录，顺着内存态传给装载器复用，
    # 本回合不再为同样的 ID 重跑一次库查询。这份记录不进图状态：它是整篇论文的原始记录，
    # 装载器把它压成清单行之后就没有第二个消费方
    turn_papers = turn_delta.pop(TURN_PAPERS_KEY, [])
    resolved_state = {**state, **turn_delta, TURN_PAPERS_KEY: turn_papers}
    # 预装载上下文：深度判档与规划看到同源的论文清单、章节结构与图谱状态
    context_delta = await _load_context(resolved_state)
    resolved_state = {**resolved_state, **context_delta}
    # 研究问题清单先读出来交给深度评估：模型只能从这份清单里挑 question_id，
    # 不给清单它就只能编，编出的 id 关联不到记忆、读取静默落空（2c 坑 2）
    assessment = await assess_turn_depth(
        resolved_state,
        llm,
        scoped_paper_ids=_allowed_paper_ids(resolved_state),
        research_questions=await read_project_research_questions(state.get("project_scope") or {}),
    )
    # 记忆读取跟着首次分析走：上面 turn_depth 那道早返回让整段只在首轮执行，
    # 所以 replan 换角度重规划时沿用首轮读到的记忆快照，不再重读记忆库
    memory_delta = await _read_memory_delta(resolved_state, assessment)
    return {
        **turn_delta,
        **context_delta,
        "turn_depth_assessment": assessment,
        "turn_depth": assessment["depth"],
        **memory_delta,
    }


async def _read_memory_delta(state: dict[str, Any], assessment: Mapping[str, Any]) -> dict[str, Any]:
    """深度评估判定本轮要翻项目记忆时读一次记忆，把读取结果作为状态增量返回。

    Args:
        state: 已并入意图与焦点解析结果的状态，project_scope 提供读取身份与项目范围
        assessment: 本轮深度评估结果，取 needs_project_memory 与 memory_narrowing

    Returns:
        dict[str, Any]: 读了记忆时返回 {"memory_read_result": MemoryReadResult}；
            模型判断本轮不必翻记忆时返回空增量

    Raises:
        MemoryOnlyTurnReadError: 读取失败且记忆是本轮唯一信息来源
    """
    # 1. 模型判断本轮不必翻记忆：不读库也不落 skipped 结果，「没问过」与「问过被跳过」不是一回事
    if not assessment.get("needs_project_memory"):
        return {}
    # 2. 读取身份与项目范围取自 project_scope；user_id 为空时 read_project_memory 抛
    #    ValueError，那是接线错误，不在这里包成 failed 掩盖成记忆存储故障
    result = await read_project_memory(
        context=state.get("project_scope") or {},
        query_narrowing=assessment.get("memory_narrowing") or None,
    )
    # 3. 读取失败且本轮既不检索论文、又要求历史判断：没有第二条信息来源，继续下去只能编，
    #    交给 analyze 层转成可重试失败
    if result.get("status") == _MEMORY_READ_FAILED_STATUS and _memory_is_only_source(assessment):
        raise MemoryOnlyTurnReadError(str(result.get("error_kind") or ""))
    # 4. success / empty / 尚有论文证据可走的 failed 都原样带进状态：failed 与 error_kind
    #    留在 memory_read_result 里可见，本轮不使用记忆但主链继续推进
    return {"memory_read_result": result}


def _memory_is_only_source(assessment: Mapping[str, Any]) -> bool:
    """判断项目记忆是不是本轮唯一的信息来源。

    minimal 深度本就不检索论文，本轮又没要求论文 Evidence，而模型判定必须翻历史判断：
    三者同时成立时记忆读不出来就无从作答，只能靠猜，所以调用方要暴露失败而不是继续。

    Args:
        assessment: 本轮深度评估结果

    Returns:
        bool: 记忆是唯一信息来源则为 True
    """
    return (
        assessment.get("depth") == _MINIMAL_DEPTH
        and not assessment.get("requires_paper_evidence")
        and bool(assessment.get("needs_project_memory"))
    )


def _analyze_failure_delta(
    budget: Any,
    failure: PlannerFailure,
    *,
    reason: str,
    extra_source_status: dict[str, str] | None = None,
    extra_source_status_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    """组装 analyze 阶段失败增量：失败载荷 + terminate_fail 决策 + 推进后的步数增量。

    Args:
        budget: 推进前的循环步数快照
        failure: 已映射好的 PlannerFailure
        reason: 写进 terminate_fail 决策的原因文本
        extra_source_status: 追加到 source_status 的可见状态位，例如 depth_assessment
        extra_source_status_reasons: 追加到 source_status_reasons 的可读原因，
            让「哪一层塌了」跟着状态位一起到上层，光有 failed 排查不了方向

    Returns:
        dict[str, Any]: 可直接作为节点返回值的失败增量
    """
    payload = build_planner_failure_response(failure)
    if extra_source_status:
        payload["source_status"] = {**payload["source_status"], **extra_source_status}
    if extra_source_status_reasons:
        payload["source_status_reasons"] = {
            **payload["source_status_reasons"],
            **extra_source_status_reasons,
        }
    loop_delta = advance_and_serialize(
        budget=budget,
        stage="analyze",
        decision=PlannerDecision(action="terminate_fail", reason=reason),
        observation=None,
    )
    return {**payload, **loop_delta, "final_delivery_applicability": "pre_execution_failure"}


def validate_planner_state(state: dict[str, Any], settings: Any) -> dict[str, Any]:
    """
    LangGraph planner_validate 阶段函数（循环里的「行动+观察」步）。

    策略声明必须通过领域校验，不补造 facet 或删除依赖。失败时不直接返回终态失败载荷，
    而是产观察 + 调决策器 + 推进步数，把决策/观察/步数写回 state，由路由据
    planner_loop_decision 选边（replan 回 analyze / terminate_fail 走 write_back）。

    Args:
        state: 当前 RAGState 字典态，必须包含 planner_draft
        settings: 全局配置（本函数只读单元数上限）

    Returns:
        dict[str, Any]: 可执行 research_plan（成功）或失败载荷 + 循环决策/步数增量
    """
    # 1. 【研究循环】【合同接续】已建立的增量账本不能因缺失字段切回整份计划合同
    incremental = ((state.get("research_plan") or {}).get("metadata") or {}).get("execution_mode") == "incremental"
    if incremental or "decision" in (state.get("planner_draft") or {}):
        return _validate_action_state(state, settings)
    # 1. 读旧步数快照，供本步推进（无 reducer 的 Dict 键，必须显式读旧值回写）
    budget = read_budget(state)
    validation = validate_or_repair_plan(
        state.get("planner_draft") or {},
        options=PlannerValidationOptions(
            question=state.get("user_query", ""),
            user_texts=tuple(item["content"] for item in (state.get("chat_history") or [])
                             if isinstance(item, dict) and item.get("role") == "user"
                             and isinstance(item.get("content"), str)),
            memory_grounded_allowed=bool(render_project_memory_block(state)),
            allowed_paper_ids=_allowed_paper_ids(state),
            max_units=int(getattr(settings, "rag_research_planner_max_units", 8)),
            requirements_baseline=(_plan_from_state(state).coverage_review if state.get("research_plan") else None),
        ),
    )
    # 2. 把校验结果归一成观察，据「失败面有没有变化」决策下一步
    observation = observe_validation(validation)
    decision = decide_next(observation=observation, budget=budget)
    loop_delta = advance_and_serialize(
        budget=budget, stage="validate", decision=decision, observation=observation,
    )
    # 3. 校验失败：仅 terminate_fail 时才落失败载荷；replan 只带决策增量回 analyze
    #    失败载荷会写 error_code/source_status.planner=failed，这些是无 reducer 的键，
    #    若在 replan 路径也写，重规划成功后旧值不会被清、chat_api 会误记 planner_failed
    if validation.failure is not None or validation.plan is None:
        if decision.action == "terminate_fail":
            failure = validation.failure or _failure("planner_validation_failed", "planner validation failed")
            payload = {**build_planner_failure_response(failure), **loop_delta,
                       "final_delivery_applicability": "pre_execution_failure"}
            return _research_failure_payload(state, payload, stage="validate")
        # replan 回边：不落失败载荷，清掉可能残留的旧错误码，避免污染最终态
        return {**_clear_planner_error(), **loop_delta}
    # 4. 校验通过：research_plan 就绪，决策为 continue，路由进 execute
    plan = validation.plan
    return {
        "research_plan": plan.model_dump(mode="json"),
        "planner_execution_history": _replaced_plan_history(state),
        "coverage_review": plan.coverage_review.model_dump(mode="json"),
        "planner_trace": plan.metadata.model_dump(mode="json"),
        "runtime_chain": "planner",
        "source_status": {"planner": "validated"},
        **loop_delta,
    }


def _validate_action_state(state: dict[str, Any], settings: Any) -> dict[str, Any]:
    """参数为图状态及配置；接纳当前动作或反馈真实缺口，完整交付另行检查累计成果。"""
    budget = read_budget(state)
    raw = state.get("planner_draft") or {}
    previous = _plan_from_state(state) if state.get("research_plan") else None
    baseline = PlannerCoverageReview.model_validate(state["coverage_review"]) if state.get("coverage_review") else None
    validation, review = validate_actions(raw, options=PlannerValidationOptions(
        question=state.get("user_query", ""), requirements_baseline=baseline,
        user_texts=tuple(item["content"] for item in state.get("chat_history") or []
                         if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str)),
        memory_grounded_allowed=bool(render_project_memory_block(state)), allowed_paper_ids=_allowed_paper_ids(state),
        max_units=int(getattr(settings, "rag_research_planner_max_units", 8))),
        previous_plan=previous, results=state.get("research_unit_results") or [])
    work = action_work_keys(raw, state)
    kind = (raw.get("decision") or {}).get("kind")
    if validation.failure is None and kind == "execute":
        accepted = {key for step in budget.history if step.stage == "execute"
                    for key in step.observation.get("attempted_work_keys", [])}
        if len(set(work)) != len(work) or set(work) & accepted:
            validation = PlannerValidationResult(failure=_failure("repeated_action", "same action inputs already accepted; inspect existing results"))
    if validation.failure is None and kind == "finish":
        if previous is None and raw["decision"].get("allow_partial"):
            assert review is not None
            record_action_validation(raw, validation, initial=baseline is None)
            report = make_terminal_delivery_report(status="insufficient", reason="尚未执行研究动作，未取得可回答的材料",
                reason_code="no_research_material", stage="task_execution", entrypoint="research_honest")
            return {"answer": "目前尚未取得可用于回答这个问题的研究材料。", "coverage_review": review.model_dump(mode="json"),
                    "final_delivery_applicability": "research", **delivery_state_fields(report),
                    "source_status": {"planner": "no_evidence", "local_evidence_execution": "not_run"},
                    **advance_and_serialize(budget=budget, stage="validate",
                        decision=PlannerDecision(action="terminate_honest", reason=raw["decision"]["reason"]), observation=None)}
        validation = _validate_finish(state, validation, raw["decision"])
    record_action_validation(raw, validation, initial=baseline is None)
    observation = observe_validation(validation).model_copy(update={"pending_work_keys": work})
    decision = decide_action_admission(observation, budget)
    delta = advance_and_serialize(budget=budget, stage="validate", decision=decision, observation=observation)
    if review is not None:
        delta["coverage_review"] = review.model_dump(mode="json")
    if validation.failure is not None:
        if decision.action == "terminate_fail":
            payload = {**build_planner_failure_response(validation.failure), **delta}
            # 1. 【研究循环】【失败保全】错误不能删除此前真正完成的动作与产物
            payload.update({key: state[key] for key in ("research_plan", "research_unit_results", "evidence_records",
                            "planner_read_receipts") if key in state})
            return _research_failure_payload(state, payload, stage="validate")
        return {**_clear_planner_error(), **delta}
    plan = validation.plan
    if plan is None:
        raise ValueError("accepted execution or finish requires an action ledger")
    return {**_clear_planner_error(), **delta, "research_plan": plan.model_dump(mode="json"),
            "coverage_review": plan.coverage_review.model_dump(mode="json"),
            "planner_trace": plan.metadata.model_dump(mode="json"), "runtime_chain": "planner",
            "source_status": {**(state.get("source_status") or {}), "planner": "validated"}}


def _validate_finish(state: dict[str, Any], validation: PlannerValidationResult, decision: dict[str, Any]) -> PlannerValidationResult:
    """参数为累计成果及交付提议；未满足要求反馈缺口，明确部分交付沿用实际支持检查。"""
    if validation.plan is None or not state.get("evidence_sufficiency_report"):
        return PlannerValidationResult(failure=_failure("finish_without_execution", "no executed action results are available"))
    report = build_evidence_sufficiency_report(plan=validation.plan, results=_results_from_state(state),
        evidence_records=state.get("evidence_records") or [], pass_index=1)
    if (report.missing_required_facet_ids or report.blocking_unit_ids) and not decision.get("allow_partial"):
        return PlannerValidationResult(failure=_failure("finish_requirements_missing",
            f"required facets {report.missing_required_facet_ids}, blocked units {report.blocking_unit_ids}; continue or explicitly finish partial"))
    return validation


async def execute_planner_state(
    state: dict[str, Any], *, llm: Any, settings: Any,
    adapters: Mapping[str, EvidenceAdapter], optional_adapters: Mapping[str, EvidenceAdapter] | None = None,
    index_probe: Any = None, request_resolver: Any = None,
) -> dict[str, Any]:
    """参数为当前图状态、模型及取证依赖；接续本次成果，返回当前产物与累计读取快照。"""
    migration = migrate_execution_checkpoint(state)
    if migration:
        return migration
    budget = read_budget(state)
    context = None
    report = None
    try:
        plan = _plan_from_state(state)
        context = _execution_context(state, plan, llm=llm, request_resolver=request_resolver)
        executor = ResearchPlannerExecutor(
            adapters=_adapters_for_plan(plan, adapters, optional_adapters or {}),
            reasoner=ResearchUnitReasoner(draft_provider=LLMClaimDraftProvider(llm=MemoryCheckedModel(llm, state),
                                         coverage_review=plan.coverage_review, question=plan.question),
                                         memory_grounded_allowed=bool(render_project_memory_block(state))),
        )
        # 1. 【研究执行】【定向续接】首次只执行计划，补证回边仅处理统一报告中已获准的缺口
        with evidence_scope("planner_execute"):
            prior = state.get("evidence_sufficiency_report")
            report = await execute_with_repair(
                plan=plan, executor=executor, context=context,
                previous_results=_results_from_state(state) if state.get("research_unit_results") else [],
                mode="repair" if read_decision(state) == "repair_evidence" else "initial",
                prior_sufficiency=EvidenceSufficiencyReport.model_validate(prior) if prior else None,
            )
            record_evidence_observation("execution_evidence_pool", {
                "records": report.evidence_records, "read_receipts": report.read_receipts,
                "repair_status": report.repair_status, "unit_ids": [result.unit_id for result in report.results],
            })
        return await _execution_success_delta(state, plan, report, budget=budget, index_probe=index_probe)
    except asyncio.CancelledError:
        if context is not None:
            record_evidence_observation("cancelled_execution_snapshot", _execution_snapshot(context))
        raise
    except Exception as exc:
        snapshot = _execution_snapshot(context) if context is not None else {}
        previous_count = len(state.get("planner_read_receipts") or [])
        attempted = any(item["actual_call_count"] for item in snapshot.get("planner_read_receipts", [])[previous_count:])
        code = "planner_execution_failed" if attempted else "planner_execution_before_read_failed"
        # 1. 【研究交付】【失败适用性】权限/输入失败尚未进入执行；读取、复用与纯推理由工作入口记账
        if context is not None and context.execution_state.execution_started:
            state = {**state, "final_delivery_applicability": "research"}
        delta = _execution_failure_state(state, exc, code=code, capability_failure=_capability_failure(exc),
                                          snapshot=snapshot)
        failed = getattr(exc, "planner_execution_report", None) or report
        if failed is not None:
            delta["research_unit_results"] = [result.model_dump(mode="json") for result in failed.results]
            delta["planner_evidence_capabilities"] = {**(state.get("planner_evidence_capabilities") or {}),
                                                        **failed.capability_results,
                                                        **delta.get("planner_evidence_capabilities", {})}
        return delta
async def _execution_success_delta(
    state: dict[str, Any], plan: ResearchPlan, report: PlannerEvidenceRepairOutcome, *, budget: Any, index_probe: Any,
) -> dict[str, Any]:
    """参数为已完成成果、原状态和循环上下文；核对索引事实并生成整份成功回写增量。"""
    # 2. 【研究循环】【满足观察】循环只消费统一产物事实，不再将非空状态当成任务完成
    scoped_ids = _scoped_paper_ids(plan)
    has_answerable = report.final_sufficiency_report.has_usable_results
    observation = observe_execution(
        eligibility=report.final_answer_eligibility_report, sufficiency=report.final_sufficiency_report,
        missing_evidence=report.missing_evidence, scoped_paper_ids=scoped_ids,
        has_answerable_evidence=has_answerable,
        scoped_in_index=(None if plan.metadata.execution_mode == "incremental"
                         else await _probe_scoped_in_index(index_probe, scoped_ids, has_answerable)),
        evidence_records=report.evidence_records,
        locator_keys=_locator_fact_keys({**state, "evidence_records": report.evidence_records}, report.read_receipts),
    )
    observation = observation.model_copy(update={"attempted_work_keys": [
        "explore:" + fact_digest({"question": str(state.get("user_query") or ""), "papers": sorted(scoped_ids),
                                   "facts": material_fact_keys(state.get("evidence_records") or [], scoped_ids)})
    ]})
    if plan.metadata.execution_mode == "incremental":
        observation = observation.model_copy(update={"attempted_work_keys": action_work_keys(state["planner_draft"], state)})
    decision = (PlannerDecision(action="replan", reason="当前动作完成，回填实际结果供模型选择下一步")
                if plan.metadata.execution_mode == "incremental" else decide_next(observation=observation, budget=budget))
    return {
        "final_delivery_applicability": "research",
        "research_plan": plan.model_copy(update={"metadata": plan.metadata.model_copy(update={"pending_unit_ids": []})}).model_dump(mode="json"),
        "research_unit_results": [result.model_dump(mode="json") for result in report.results],
        "evidence_records": list(report.evidence_records), "planner_read_receipts": list(report.read_receipts),
        "runtime_chain": "planner", "source_status": {**_state_str_map(state, "source_status"), "planner": "executed", "planner_repair": report.repair_status,
                                                         "local_evidence_execution": "attempted"},
        "source_status_reasons": {"planner_repair": report.repair_reason} if report.repair_reason else {},
        "planner_evidence_capabilities": {**(state.get("planner_evidence_capabilities") or {}), **report.capability_results},
        **_state_repair_payload(report.synthesis_context()),
        **advance_and_serialize(budget=budget, stage="execute", decision=decision, observation=observation),
    }


def _replaced_plan_history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """参数为重规划前状态；返回追加旧计划与当前产物的只读历史，旧结论不继续充当新成果。"""
    history = list(state.get("planner_execution_history") or [])
    if state.get("research_plan"):
        history.append({"reason": "plan_replaced", "research_plan": state["research_plan"],
                        "research_unit_results": state.get("research_unit_results") or [],
                        "receipt_ids": [item["receipt_id"] for item in state.get("planner_read_receipts") or []]})
    return history


def _scoped_paper_ids(plan: ResearchPlan) -> list[str]:
    """收集本轮 plan 覆盖的 scoped 论文范围（保序去重），供诚实收尾列出范围。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for unit in plan.units:
        for paper_id in unit.evidence_policy.target_paper_ids:
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                ordered.append(paper_id)
    return ordered


async def _probe_scoped_in_index(index_probe: Any, scoped_ids: list[str], has_answerable: bool) -> bool:
    """仅在检索空（无可用证据）时探库判 scoped 论文是否有内容，避免 happy path 额外 I/O。"""
    # 已有可用证据时 scoped_in_index 不影响决策，直接视为在库省去探针
    if has_answerable or index_probe is None or not scoped_ids:
        return True
    return await index_probe(scoped_ids)


async def synthesize_planner_state(
    state: dict[str, Any],
    llm: Any,
    *,
    paper_title_resolver: Any = None,
) -> dict[str, Any]:
    """【研究交付】【最终组织】按实际成果形成同次正文、引用和交付报告。

    参数:
        state: 当前研究成果及明确入口；诚实收尾仍允许交付已有产物
        llm: 最终组织或 minimal 直答所用模型
        paper_title_resolver: 既有调用参数；最终来源标题由保存材料提供
    返回:
        同一次尝试的完整交付增量；minimal 保持独立入口语义
    """
    # 1. 【研究交付】【直接回答】未进入研究执行的 minimal 路径不制造研究完成报告
    if not state.get("research_plan") and _is_direct_answer_turn(state):
        return await _build_minimal_direct_answer(state, llm)
    # 2. 【研究交付】【诚实收尾】动作只说明研究停止，实际成果仍交给统一最终组织入口
    research_state = {**state, "final_delivery_applicability": "research"}
    try:
        plan = _plan_from_state(state)
        results = _results_from_state(state)
    except Exception as exc:
        return _execution_failure_state(research_state, f"planner synthesis input invalid: {exc}", stage="synthesize")
    payload = await synthesize_planner_answer_async(
        plan,
        results,
        project_paper_count=len(_allowed_paper_ids(state) or frozenset()),
        evidence_records=state.get("evidence_records") if isinstance(state.get("evidence_records"), list) else [],
        final_provider=LLMPlannerFinalAnswerProvider(llm=MemoryCheckedModel(llm, state)),
        answer_reviewer=LLMResearchAnswerReviewer(llm=MemoryCheckedModel(llm, state)),
        repair_context=_repair_context_from_state(state),
        final_entrypoint="research_honest" if read_decision(state) == "terminate_honest" else "normal",
    )
    # 3. 【研究交付】【缺口回边】候选未完成时保留核对与成果，先让规划者选择补分析、补读或重写
    report = payload.get("final_delivery_report") or {}
    action = payload.get("planner_loop_decision")
    if action in {"revise_answer", "replan"} or (action is None and review_requires_continuation(report)):
        reason = "; ".join(item["message"] for item in report.get("validation_errors", []))
        reason = reason or "; ".join(item["message"] for item in report.get("reasons", []))
        delta = advance_and_serialize(budget=read_budget(state), stage="synthesize",
            decision=PlannerDecision(action=action or "replan", reason=reason or "修正当前正文的具体问题"), observation=None)
        history = [*(state.get("planner_execution_history") or []), {"reason": "candidate_content_review",
                    "answer": payload["answer"], "final_delivery_report": report,
                    "planner_final_blocks": payload.get("planner_final_blocks", [])}]
        return {**payload, **delta, "answer": "", "planner_execution_history": history,
                "source_status": {**payload["source_status"], "planner": "reviewing"}}
    payload = {**payload, **advance_and_serialize(budget=read_budget(state), stage="synthesize",
        decision=PlannerDecision(action="terminate_fail" if report.get("status") == "failed" else "continue",
                                 reason="最终正文校验及核对已结束"), observation=None)}
    if (payload.get("final_delivery_report") or {}).get("status") == "failed":
        payload = project_planner_failure_read_facts(payload, _failure_read_facts(research_state, stage="synthesize"))
        return bind_rendered_review_answer(payload)
    return payload


async def _build_minimal_direct_answer(state: dict[str, Any], llm: Any) -> dict[str, Any]:
    """生成 minimal 直答载荷：拿问题、对话上下文与项目历史判断直接出正文。

    对话上下文与记忆段都复用草案 prompt 的那两处渲染。记忆段那道守卫（只有真读到
    条目才渲染）必须走同一份实现，另写一遍就会漏掉 status/items 判断，模型能拿着
    失败或空的读取结果编一段项目此前的判断。

    本轮授权论文数一并交给直答：项目里一篇论文都没有时，直答得说清缺的是可核对的
    论文材料，而不是把用户指回一个空书架。这个事实只有 state 知道，不传就只能靠模型自觉。

    参数:
        state: 当前 RAGState 字典态，取 user_query / chat_history / memory_read_result
            与本轮授权论文范围
        llm: 直答使用的 LLM
    返回:
        dict[str, Any]: 直答载荷；模型报错或返回空正文时是带 terminate_fail 的失败载荷
    """
    try:
        payload = await build_direct_answer_response(
            question=str(state.get("user_query") or ""),
            conversation_context=render_conversation_context(state),
            memory_block=render_project_memory_block(state),
            # 授权范围为空时 _allowed_paper_ids 返回 None，取数前先兜成空集合
            scoped_paper_count=len(_allowed_paper_ids(state) or frozenset()),
            llm=MemoryCheckedModel(llm, state),
        )
        return {**payload, "final_delivery_applicability": "minimal"}
    except Exception as exc:
        # 直答是这一轮唯一的出口，模型没出正文就如实失败：补一句「暂时无法回答」的模板正文
        # 会把模型故障伪装成一次正常回答留在会话里，用户与评测都看不出这轮其实没答上
        return _execution_failure_state(
            {**state, "final_delivery_applicability": "minimal"},
            f"minimal direct answer failed: {provider_failure_message(exc)}",
            code=_DIRECT_ANSWER_FAILED_CODE,
        )


def _plan_from_state(state: dict[str, Any]) -> ResearchPlan:
    """从当前状态里恢复强类型 research_plan。"""
    payload = state.get("research_plan")
    if not isinstance(payload, dict):
        raise ValueError("research_plan missing from planner state")
    plan = ResearchPlan.model_validate(payload)
    if plan.contract_version != CURRENT_RESEARCH_CONTRACT_VERSION:
        raise ValueError("继续执行需要新版研究合同，旧 checkpoint 必须先重新规划")
    issues = fulfillment_contract_issues(plan.coverage_review, plan.units, allowed_paper_ids=_allowed_paper_ids(state),
                                         require_complete=plan.metadata.execution_mode != "incremental")
    if issues:
        raise ValueError("保存的新版研究合同无效：" + "; ".join(issue.message for issue in issues))
    return plan


def _results_from_state(state: dict[str, Any]) -> list[ResearchUnitResult]:
    """从当前状态里恢复强类型 research_unit_results。"""
    raw_results = state.get("research_unit_results")
    if not isinstance(raw_results, list):
        raise ValueError("research_unit_results missing from planner state")
    return [ResearchUnitResult.model_validate(result) for result in raw_results]


def execution_content_identity(value: Any) -> Any:
    """参数为范围或请求；返回去除租约标识的内容身份，续租不会伪造新的内容版本。"""
    if isinstance(value, Mapping):
        return {key: execution_content_identity(item) for key, item in value.items() if key != "lease_id"}
    if isinstance(value, (tuple, list)):
        return [execution_content_identity(item) for item in value]
    return value


def _execution_context(state: dict[str, Any], plan: ResearchPlan | None = None, *,
                       llm: Any = None, request_resolver: Any = None) -> PlannerExecutionContext:
    """参数为本次图状态、有效计划和依赖；返回核对运行/范围/内容身份后的执行上下文。"""
    identity = _execution_identity(state)
    model = getattr(llm, "model", None)
    reasoning = {}
    if isinstance(model, str) and isinstance(getattr(llm, "temperature", None), (int, float)):
        reasoning = {"model": model, "temperature": llm.temperature,
                     "provider": _provider_identity(str(getattr(llm, "base_url", ""))),
                     "prompt_version": CLAIM_PROMPT_VERSION, "memory": state.get("memory_read_result"),
                     "memory_prompt": render_project_memory_block(state),
                     "question": plan.question if plan is not None else str(state.get("user_query") or ""),
                     "context_metadata": _context_metadata(state),
                     "memory_grounded_allowed": bool(render_project_memory_block(state))}
    return PlannerExecutionContext(
        question=plan.question if plan is not None else str(state.get("user_query") or ""), metadata=_context_metadata(state),
        execution_state=PlannerExecutionState(identity, evidence_records=state.get("evidence_records") or [],
                                               read_receipts=state.get("planner_read_receipts") or []),
        request_resolver=request_resolver, result_context=reasoning, plan=plan,
    )


def _provider_identity(base_url: str) -> str:
    """参数为模型实际基础地址；返回去除凭据、查询和片段的提供方路由身份。"""
    parsed = urlsplit(base_url)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname or ''}{port}{parsed.path.rstrip('/')}"


def _execution_snapshot(context: PlannerExecutionContext) -> dict[str, Any]:
    """参数为拥有实际工作的上下文；返回材料与读取凭据的同一快照。"""
    snapshot = context.execution_state.snapshot()
    return {"evidence_records": snapshot.evidence_records, "planner_read_receipts": snapshot.read_receipts}


def _execution_identity(state: Mapping[str, Any]) -> ExecutionIdentity:
    """参数为服务端当前图状态；返回与读取账本一致的运行身份，缺失身份显式报错。"""
    scope = state.get("project_scope") or {}
    run_id, user_id = str(state.get("run_id") or ""), str(scope.get("user_id") or "")
    if not run_id or not user_id:
        raise ValueError("研究执行缺少服务端 run_id 或用户身份")
    return ExecutionIdentity(run_id=run_id, user_id=user_id, scope=execution_content_identity(scope),
                             publication_snapshot=execution_content_identity(scope.get("publication") or {}))


def _merged_failure_snapshot(state: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """参数为本轮状态与异常累计快照；返回去重且同身份的材料/回执增量，冲突显式报错。"""
    if not snapshot:
        return {}
    # 1. 【研究失败】【累计快照】累计快照可重复携带旧回执，同一回执不能被覆盖成另一事实
    receipts: dict[str, Any] = {}
    for item in [*(state.get("planner_read_receipts") or []), *(snapshot.get("planner_read_receipts") or [])]:
        receipt_id = item["receipt_id"]
        if receipt_id in receipts and receipts[receipt_id] != item:
            raise ValueError("失败快照中同一读取回执的事实不一致")
        receipts[receipt_id] = item
    completed = PlannerExecutionState(
        _execution_identity(state), read_receipts=list(receipts.values()),
        evidence_records=[*(state.get("evidence_records") or []), *(snapshot.get("evidence_records") or [])],
    ).snapshot()
    return {**snapshot, "planner_read_receipts": completed.read_receipts, "evidence_records": completed.evidence_records}


def _failure_read_facts(state: Mapping[str, Any], *, stage: str) -> PlannerFailureReadFacts:
    """参数为完整本轮失败状态及阶段；返回已发本地调用事实，不由计划、材料或错误码反推。"""
    started = state.get("final_delivery_applicability") == "research"
    if not state.get("run_id") or not (state.get("project_scope") or {}).get("user_id"):
        return PlannerFailureReadFacts(known=False, stage=stage, execution_started=started,
                                       unknown_reason="missing_execution_identity")
    if "planner_read_receipts" not in state:
        return PlannerFailureReadFacts(known=False, stage=stage, execution_started=started,
                                       unknown_reason="missing_read_receipts")
    identity = _execution_identity(state)
    receipts: dict[str, ReadReceipt] = {}
    # 1. 【研究失败】【事实来源】沿用既有回执合同核对身份，只统计实际发出的本地读取
    for raw in state.get("planner_read_receipts") or []:
        if type(raw.get("contract_version")) is not int or raw["contract_version"] != EXECUTION_STATE_VERSION:
            raise ValueError("读取凭据缺少当前执行合同版本，需要重新校验规划")
        receipt = ReadReceipt.model_validate(raw)
        if receipt.identity != identity:
            raise ValueError("读取凭据的研究运行、用户、范围或发布快照不一致")
        if receipt.receipt_id in receipts and receipts[receipt.receipt_id] != receipt:
            raise ValueError("失败快照中同一读取回执的事实不一致")
        receipts[receipt.receipt_id] = receipt
    attempted = [receipt for receipt in receipts.values()
                 if receipt.source_type.startswith("local_") and receipt.actual_call_count > 0]
    return PlannerFailureReadFacts(known=True, stage=stage, execution_started=started,
        actual_call_count=sum(receipt.actual_call_count for receipt in attempted),
        has_successful_read=any(receipt.outcome == "success" for receipt in attempted))


def _capability_failure(exc: Exception) -> tuple[str, str] | None:
    """从执行异常里取取证失败面与可读原因；非取证异常（计划校验、综合输入等）不产四态。

    参数:
        exc: 从执行阶段冒出来的异常
    返回:
        (source_type, 可读原因)；不是取证调用失败时返回 None
    """
    if isinstance(exc, EvidenceCapabilityError):
        return (exc.source_type, exc.reason)
    return None


def _execution_failure_state(
    state: dict[str, Any],
    message: str | Exception,
    *,
    code: str = "planner_execution_failed",
    capability_failure: tuple[str, str] | None = None,
    stage: str = "execute",
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """参数为实际状态、错误及能力失败；返回保留执行原因的明确终止增量。"""
    # 执行/综合输入异常是系统性硬失败：显式写 terminate_fail 决策，路由据此直达 write_back
    # 不留旧决策（如 repair_evidence）否则会被误路由回 execute 形成回边死循环
    #
    # 1. 【研究失败】【错误身份】错误码仅保留故障身份，实际读取由累计回执统一投影
    error = _failure_from_exc(message, code) if isinstance(message, ProviderHTTPError) else _failure(code, str(message))
    failure = build_planner_failure_response(error)
    payload = {
        **failure,
        "research_plan": state.get("research_plan"),
        "coverage_review": state.get("coverage_review"),
        "planner_loop_decision": "terminate_fail",
    }
    if capability_failure is not None:
        source_type, reason = capability_failure
        # 取证失败只如实记 failed 与原因，错误码、路由与终止行为仍是上面那套硬失败
        payload["planner_evidence_capabilities"] = {**(state.get("planner_evidence_capabilities") or {}), source_type: "failed"}
        payload["source_status_reasons"] = {**_state_str_map(state, "source_status_reasons"), **failure["source_status_reasons"], source_type: reason}
    if stage == "synthesize":
        payload["source_status"] = dict(state.get("source_status") or {})
    return _research_failure_payload(state, payload, stage=stage, snapshot=snapshot)



def _research_failure_payload(
    state: Mapping[str, Any], payload: dict[str, Any], *, stage: str,
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """【研究交付】【失败收尾】参数为入口、失败增量与阶段；返回同次失败交付或不适用标记。"""
    completed = _merged_failure_snapshot(state, snapshot or {})
    if state.get("research_plan") and not payload.get("research_plan"):
        # 1. 【研究循环】【错误保全】后续模型失败不能抹掉已经执行的动作和原始成果
        payload = {**payload, **{key: state[key] for key in ("research_plan", "research_unit_results", "coverage_review")
                                if key in state}}
    payload = project_planner_failure_read_facts(
        {**payload, **completed}, _failure_read_facts({**state, **completed}, stage=stage),
    )
    if state.get("final_delivery_applicability") != "research":
        return {**payload, "final_delivery_applicability": state.get("final_delivery_applicability")
                or "pre_execution_failure"}
    reason = str(payload.get("error_summary") or payload.get("error_code") or "research execution failed")
    report = make_terminal_delivery_report(
        status="failed", reason=reason, reason_code=str(payload.get("error_code") or "execution_failed"),
        stage=stage, entrypoint="research_execution_failure",
    )
    projection = project_final_delivery(report)
    # 1. 【研究交付】【失败事实】失败说明与空引用属于当前尝试，不能携带前一尝试的接受记录
    return {**payload, **delivery_state_fields(report),
            "planner_final_blocks": [], "planner_final_validation_errors": [],
            "source_status": {**payload.get("source_status", {}), "planner_final": projection["planner_final_status"]},
            "source_status_reasons": {**payload.get("source_status_reasons", {}), "planner_final": reason}}


def _clear_planner_error(state: dict[str, Any] | None = None) -> dict[str, Any]:
    # replan 回边前清空上一轮失败痕迹：error_code/error_summary 是无 reducer 的 Dict 键，
    # 不显式清空则成功重规划后仍残留，chat_api 会据陈旧 error_code 误记 planner_failed
    return {
        "error_code": "",
        "error_summary": "",
        "source_status": {"planner": "replanning"},
    }


def _state_repair_payload(repair_context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evidence_sufficiency_report": repair_context.get("evidence_sufficiency_report"),
        "repair_query_plans": repair_context.get("repair_query_plans", []),
        "planner_repair_trace": repair_context.get("planner_repair_trace", {}),
        "planner_repair_merge_report": repair_context.get("planner_repair_merge_report", {}),
        "final_answer_eligibility_report": repair_context.get("final_answer_eligibility_report", {}),
        "missing_evidence": repair_context.get("missing_evidence", []),
    }


def _repair_context_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """参数为当前研究状态；透传原问题、实际读取凭据与取证缺口供最终交付说明。"""
    status = state.get("source_status") if isinstance(state.get("source_status"), dict) else {}
    reasons = state.get("source_status_reasons") if isinstance(state.get("source_status_reasons"), dict) else {}
    return {
        "user_query": state.get("user_query") or "",
        "planner_read_receipts": state.get("planner_read_receipts") or [],
        "final_delivery_report": state.get("final_delivery_report") or {},
        "planner_final_candidate": state.get("planner_final_candidate") or {},
        "evidence_sufficiency_report": state.get("evidence_sufficiency_report") or {},
        "repair_query_plans": state.get("repair_query_plans") or [],
        "planner_repair_trace": state.get("planner_repair_trace") or {},
        "planner_repair_merge_report": state.get("planner_repair_merge_report") or {},
        "final_answer_eligibility_report": state.get("final_answer_eligibility_report") or {},
        "missing_evidence": state.get("missing_evidence") or [],
        "planner_repair_status": status.get("planner_repair") or "",
        "planner_repair_reason": reasons.get("planner_repair") or "",
    }








def _locator_fact_keys(state: Mapping[str, Any], receipts: list[dict[str, Any]]) -> list[str]:
    """参数为实际目录、材料和读取凭据；返回带论文作用域的真实定位键，不采信模型新写的筛选值。"""
    keys = {json.dumps([item["paper_id"], "section", item["section_path"]], ensure_ascii=False)
            for item in state.get("planner_locator_facts") or []}
    records = {item["evidence_id"]: item for item in state.get("evidence_records") or []}
    for receipt in receipts:
        report = receipt.get("read_report") or {}
        request = receipt.get("request") or {}
        papers = request.get("paper_ids") or (request.get("policy") or {}).get("target_paper_ids") or []
        for item in report.get("remaining") or []:
            paper = item.get("paper_id") or (papers[0] if len(papers) == 1 else "")
            if not paper:
                continue
            for field, kind in (("section_path", "section"), ("page_num", "page"), ("chunk_id", "chunk")):
                if item.get(field):
                    keys.add(json.dumps([paper, kind, str(item[field])], ensure_ascii=False))
        for evidence_id in receipt.get("evidence_ids") or []:
            record = records.get(evidence_id, {})
            paper = record.get("paper_id")
            if not paper:
                continue
            if receipt.get("source_type") == "local_section_structure":
                keys.update(json.dumps([paper, "section", value.strip()], ensure_ascii=False)
                            for value in str(record.get("summary") or "").split(" | ") if value.strip())
            for field, kind in (("section_path", "section"), ("page_num", "page"), ("chunk_id", "chunk")):
                if record.get(field):
                    keys.add(json.dumps([paper, kind, str(record[field])], ensure_ascii=False))
    return sorted(keys)


































async def _analyze_plan(state: dict[str, Any], llm: Any) -> ResearchDecision:
    """参数为本轮状态和模型；首次识别要求，每次只选择当前动作并保留记忆资格复核。"""
    initial = not bool(state.get("coverage_review"))
    schema = InitialResearchDecisionOutput if initial else ResearchDecisionOutput
    analyzer = MemoryCheckedModel(llm, state).with_structured_output(schema)
    with trace_label("planner_analyze", "plan_draft"):
        raw = await analyzer.ainvoke([HumanMessage(content=build_action_prompt(state))])
        return project_decision_output(raw, initial=initial)




def _allowed_paper_ids(state: dict[str, Any]) -> frozenset[str] | None:
    ids = _paper_ids_from_focus(state) or _paper_ids_from_project_scope(state)
    return frozenset(ids) if ids else None


def _paper_ids_from_focus(state: dict[str, Any]) -> list[str]:
    focus = state.get("focus")
    if isinstance(focus, dict):
        return _clean_ids(focus.get("turn_focus") or focus.get("project_scope"))
    paper_focus = state.get("paper_focus")
    if isinstance(paper_focus, dict):
        return _clean_ids(paper_focus.get("paper_ids"))
    return []


def _paper_ids_from_project_scope(state: dict[str, Any]) -> list[str]:
    project_scope = state.get("project_scope")
    if isinstance(project_scope, dict):
        return _clean_ids(project_scope.get("paper_ids"))
    return []


def _clean_ids(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item or "").strip()]


def _context_metadata(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": state.get("intent"),
        "focus": state.get("focus"),
        "structured_query": state.get("structured_query") or {},
    }


async def _load_context(state: dict[str, Any]) -> dict[str, Any]:
    """
    预装载本轮项目上下文：论文清单 + 章节结构 + 图谱状态。

    论文清单与章节结构共用一次章节窄查询：清单里的索引状态要看「检索侧到底有没有行」，
    章节段要看「有哪些章节」，两者问的是同一份数据，查两遍等于在性能卡上新建一处慢点。

    图谱面是软失败：查不到只让该段写「无」，清单与章节段照常返回。论文库与章节查询
    是硬失败，向上抛给 analyze 落成显式失败载荷，不静默退回「无上下文规划」。

    Args:
        state: 当前 RAGState 字典态，取 project_scope 身份与本轮授权论文
    Returns:
        dict[str, Any]: 含 _paper_list_lines / _section_structure_lines /
            _graph_status_lines 三段的状态增量
    Raises:
        ContextLoadError: 论文库或章节查询发生基础设施级故障且重试后仍不成立
    """
    papers = await _load_context_papers(state)
    section_paths = await _load_section_paths(state, papers)
    graph_status_lines, graph_failure_reason = await _load_graph_status(state)
    delta: dict[str, Any] = {
        "_paper_list_lines": _render_paper_list(papers, section_paths),
        "_section_structure_lines": _render_section_structure(papers, section_paths),
        "_graph_status_lines": graph_status_lines,
        "planner_locator_facts": [{"paper_id": paper, "section_path": path, "source": "loaded_directory"}
                                  for paper, paths in section_paths.items() for path in paths],
    }
    # 图谱查不到时把原因记进可见状态位，沿用 graph_tools 那套 neo4j=failed + reasons 的写法：
    # 段里只写「无」看不出是图库空还是连不上，排查方向完全不同
    if graph_failure_reason:
        delta["source_status"] = {**_state_str_map(state, "source_status"), "neo4j": "failed"}
        delta["source_status_reasons"] = {
            **_state_str_map(state, "source_status_reasons"),
            "neo4j": graph_failure_reason,
        }
    return delta


def _load_turn_papers_with_retry(
    state: Mapping[str, Any],
) -> Callable[[list[str]], Awaitable[list[dict[str, Any]]]]:
    """组装「按论文 ID 取记录」的读取能力，注入给焦点解析当第一次读取用。

    这是本轮真正读论文库的第一跳，所以读取动作必须包在上下文装载既有的重试里：首次查询
    撞上一次瞬时连接故障时按同一份配置重试，重试用尽抛 ContextLoadError，由 analyze 归因成
    上下文装载失败。裸调一次会让首次故障直接打断整轮，还会被记成「深度评估失败」，
    把排查方向带偏。

    Args:
        state: 当前 RAGState 字典态，读取身份在回调被调用时才从 project_scope 取

    Returns:
        Callable[[list[str]], Awaitable[list[dict[str, Any]]]]: 读取回调；RAG 子图内取不到
            用户身份时原样返回空列表，不往库里发一次无身份的查询

    Raises:
        ContextLoadError: 重试次数用尽仍读不出来，由调用方落成上下文装载失败
    """

    async def _read(paper_ids: list[str]) -> list[dict[str, Any]]:
        # RAG 子图内取用户身份只能从 project_scope 读，state["user_id"] 在 rag_node 桥接时被丢弃
        user_id = _scope_user_id(state)
        if not user_id:
            return []
        return await _retry_load(
            lambda: get_paper_repository().get_papers_by_ids(paper_ids, user_id=user_id),
            face=_CONTEXT_FACE_PAPER_DB,
            action="论文清单",
        )

    return _read


async def _load_context_papers(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    取回本轮授权论文的库记录，供清单段与索引状态判定使用。

    优先复用焦点解析那一次已经读到的记录（焦点解析按同一份 paper_ids 保序读过一次，
    读取动作本身就包在 `_load_turn_papers_with_retry` 里），同一回合不再为同样的 ID
    重复往库里跑一趟。只有当本轮授权集合超出已读记录（焦点降级到项目全量、或 replan
    重入时记录没带下来）才补一次库查询。

    Args:
        state: 当前 RAGState 字典态，取 project_scope 身份、_turn_papers 与授权论文
    Returns:
        list[dict[str, Any]]: 按本轮清单顺序排列的论文记录；本段关闭或无授权论文时为空列表
    Raises:
        ContextLoadError: 论文库重试后仍读不出来
    """
    if not get_settings().planner_paper_context_enabled:
        return []
    user_id = _scope_user_id(state)
    paper_ids = _ordered_allowed_paper_ids(state)
    if not user_id or not paper_ids:
        return []
    cached = _cached_turn_papers(state, paper_ids)
    if cached is not None:
        return cached
    papers = await _retry_load(
        lambda: get_paper_repository().get_papers_by_ids(paper_ids, user_id=user_id),
        face=_CONTEXT_FACE_PAPER_DB,
        action="论文清单",
    )
    logger.info(f"【规划上下文】【论文清单】读到 {len(papers)} 篇：user_id={user_id} 授权={len(paper_ids)}")
    return papers


def _cached_turn_papers(state: dict[str, Any], paper_ids: list[str]) -> list[dict[str, Any]] | None:
    """
    命中焦点解析已读记录时按本轮授权集合裁剪复用，否则返回 None 交给库查询。

    Args:
        state: 当前 RAGState 字典态，取 _turn_papers
        paper_ids: 本轮授权论文 ID，顺序即清单顺序
    Returns:
        list[dict[str, Any]] | None: 覆盖得上时返回按 paper_ids 保序的记录，否则 None
    """
    records = state.get(TURN_PAPERS_KEY)
    if not isinstance(records, list) or not records:
        return None
    by_id = {
        str(item.get("paper_id") or ""): item
        for item in records
        if isinstance(item, Mapping) and str(item.get("paper_id") or "")
    }
    # 授权集合里只要有一篇不在已读记录里，就说明这份缓存不是本轮该看的清单，整份作废重查
    if any(paper_id not in by_id for paper_id in paper_ids):
        return None
    return [by_id[paper_id] for paper_id in paper_ids]


async def _load_section_paths(
    state: dict[str, Any],
    papers: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """
    按论文取回已索引的章节路径，一篇一次窄查询。

    只投影 `section_path`：章节名单是轻量信息，为它把 `full_text` 等正文字段整篇拉进
    内存，在性能卡上等于新建一处慢点。零行是如实的空（论文没索引），不是失败。

    Args:
        state: 当前 RAGState 字典态，取 project_scope 里的读取身份
        papers: 已读到的论文记录，决定要查哪几篇
    Returns:
        dict[str, list[str]]: paper_id -> 章节路径列表；查不到章节的论文映射到空列表
    Raises:
        ContextLoadError: 章节查询重试后仍不成立
    """
    user_id = _scope_user_id(state)
    if not user_id or not papers:
        return {}
    from app.services.upload_read_service import publication_from_scope

    max_papers = get_settings().planner_paper_context_max_papers
    # 章节名单是直读的第一跳：盖上本轮发布快照，后续读正文才不会落到新版本上
    store = get_milvus_store().for_user(str(user_id)).for_snapshot(publication_from_scope(state.get("project_scope")))
    section_paths: dict[str, list[str]] = {}
    # 只查会展开进提示词的前 N 篇：未展开的那几篇提示词里只报篇数，查了也没人看
    for paper in papers[:max_papers]:
        paper_id = str(paper.get("paper_id") or "")
        if not paper_id:
            continue
        section_paths[paper_id] = await _retry_load(
            lambda pid=paper_id: asyncio.to_thread(store.list_section_paths, pid),
            face=_CONTEXT_FACE_SECTION,
            action="章节结构",
        )
    return section_paths


def _render_paper_list(
    papers: list[dict[str, Any]],
    section_paths: Mapping[str, list[str]],
) -> list[str]:
    """
    渲染论文清单段：每篇一行 ID + 标题 + 索引状态。

    Args:
        papers: 按本轮清单顺序排列的论文记录
        section_paths: 每篇的章节路径查询结果，用于判索引状态
    Returns:
        list[str]: 清单行；超过展开上限时末行写明未展开篇数；无论文时为空列表
    """
    if not papers:
        return []
    max_papers = get_settings().planner_paper_context_max_papers
    displayed = papers[:max_papers]
    lines = [
        f"- {str(paper.get('paper_id') or '')} | {str(paper.get('title') or '无标题')} | "
        f"{_readable_detail(paper)} | "
        f"{_index_status(paper, section_paths.get(str(paper.get('paper_id') or '')))}"
        for paper in displayed
    ]
    # 超上限只展开前 N 篇，但必须写明还剩几篇在范围内，不许静默截断
    remaining = len(papers) - len(displayed)
    if remaining > 0:
        lines.append(f"另有 {remaining} 篇在本轮范围内、本段未展开")
    return lines


def _readable_detail(paper: Mapping[str, Any]) -> str:
    """
    标明这篇论文还能额外读哪些详情，让模型知道该不该去要。

    只有摘要、尚未索引的论文，光看索引状态会被模型当成「没什么可读」而跳过；这里
    如实说明库里到底有摘要、有已存分析，模型才可能点名去读。两者都没有时写「无」，
    不留空字段冒充有值。

    Args:
        paper: 论文库记录，取 abstract 与 analysis_status
    Returns:
        str: 形如「可读详情: 摘要+分析」的标记；都没有时为「可读详情: 无」
    """
    has_abstract = bool(str(paper.get("abstract") or "").strip())
    has_analysis = str(paper.get("analysis_status") or "") == _ANALYSIS_READY_STATUS
    readable = "+".join(
        name for name, present in (("摘要", has_abstract), ("分析", has_analysis)) if present
    )
    return f"可读详情: {readable or '无'}"


def _index_status(paper: Mapping[str, Any], sections: list[str] | None) -> str:
    """
    判断一篇论文的索引状态，四态如实区分。

    判据是检索侧到底有没有章节行，不是「有没有摘要」：库里 3 篇都有摘要时按摘要分档
    会把三种真实状态渲染成同一态。库记录说已向量化却查不到行是第四种真实状态
    （`knowledge_inspector_service` 的 vectorized_paper_without_rows 盯的就是这个漂移），
    不并进「未索引」——一个要重新入库，一个要查索引为什么丢了。

    Args:
        paper: 论文库记录，取 analysis_status 与 is_vectorized
        sections: 该篇章节查询结果；None 表示本轮没查（未展开的论文）
    Returns:
        str: 已分析 / 仅有摘要 / 未索引 / 库记录已索引、取不到内容 / 未查询
    """
    if sections is None:
        return _INDEX_STATUS_NOT_QUERIED
    if sections:
        if str(paper.get("analysis_status") or "") == _ANALYSIS_READY_STATUS:
            return _INDEX_STATUS_ANALYZED
        return _INDEX_STATUS_INDEXED
    if paper.get("is_vectorized"):
        return _INDEX_STATUS_VECTORIZED_NO_ROWS
    if str(paper.get("abstract") or "").strip():
        return _INDEX_STATUS_ABSTRACT_ONLY
    return _INDEX_STATUS_NOT_INDEXED


def _render_section_structure(
    papers: list[dict[str, Any]],
    section_paths: Mapping[str, list[str]],
) -> list[str]:
    """
    渲染章节结构段：每篇论文的章节路径清单。

    模型据此直接指名章节，省掉「先发一次检索问章节存不存在、再出计划」的一次往返。
    查不到章节的论文如实写空，不影响其余论文。

    Args:
        papers: 按本轮清单顺序排列的论文记录
        section_paths: 每篇的章节路径查询结果
    Returns:
        list[str]: 每篇一段的章节清单文本；一篇都没查到时为空列表
    """
    if not section_paths:
        return []
    lines: list[str] = []
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "")
        if paper_id not in section_paths:
            continue
        sections = section_paths[paper_id]
        lines.append(f"- {paper_id}: {' | '.join(sections) if sections else _SECTION_EMPTY_TEXT}")
    return lines


async def _load_graph_status(state: dict[str, Any]) -> tuple[list[str], str]:
    """
    装载图谱状态：一行「有多少对象、多少关系」或「无数据」。

    关系三元组本身是大块内容，由工具按需取，这里只给规划一行全局视野。
    图谱面单独降级：无数据、查不到、图库不可用都只让该段写「无」，装载其余部分照常。

    Args:
        state: 当前 RAGState 字典态，取本轮授权论文
    Returns:
        tuple[list[str], str]: (状态段文本行, 失败原因)；查询成立时失败原因为空串
    """
    if not get_settings().planner_graph_context_enabled:
        return [], ""
    paper_ids = _ordered_allowed_paper_ids(state)
    if not paper_ids:
        return [], ""
    try:
        result = await _retry_load(
            lambda: get_research_object_query_service().query_project_subgraph(paper_ids),
            face=_CONTEXT_FACE_GRAPH,
            action="图谱状态",
        )
    except ContextLoadError as exc:
        logger.info(f"【规划上下文】【图谱状态】查不到，该段写「无」：{exc.detail}")
        return [_GRAPH_STATUS_EMPTY_TEXT], exc.detail
    meta = result.get("meta") or {}
    node_count = int(meta.get("node_count") or 0)
    edge_count = int(meta.get("edge_count") or 0)
    if not node_count and not edge_count:
        return [_GRAPH_STATUS_EMPTY_TEXT], ""
    return [f"图谱状态: 有数据（{node_count} 个节点，{edge_count} 条关系）"], ""


async def _retry_load(load: Any, *, face: str, action: str) -> Any:
    """
    对可重试故障重试装载动作，重试仍不成立才抛出带失败面的错误。

    次数与退避走配置。Postgres 与 Milvus 两层现在都不带任何重试，这层由装载器自己补。

    Args:
        load: 无参协程工厂，每次重试重新调用它拿一个新协程
        face: 失败面标识（论文库 / 章节查询 / 图谱），随失败一起暴露
        action: 日志里的动作名
    Returns:
        Any: 装载动作的返回值
    Raises:
        ContextLoadError: 重试次数用尽仍失败
    """
    cfg = get_settings()
    max_retries = cfg.planner_context_max_retries
    for attempt in range(max_retries + 1):
        try:
            return await load()
        except Exception as exc:
            if attempt >= max_retries:
                logger.error(f"【规划上下文】【{action}】重试 {max_retries} 次后仍失败：{exc}")
                raise ContextLoadError(face, f"{type(exc).__name__}: {exc}") from exc
            logger.info(f"【规划上下文】【{action}】失败，重试 {attempt + 1}/{max_retries}：{exc}")
            await asyncio.sleep(cfg.planner_context_retry_backoff_seconds)


def _scope_user_id(state: Mapping[str, Any]) -> str:
    """取本轮读取身份。

    RAG 子图内只能从 project_scope 读：state["user_id"] 在 rag_node 桥接时已被丢掉。

    Args:
        state: 当前 RAGState 字典态
    Returns:
        str: 用户 ID；取不到时为空串
    """
    project_scope = state.get("project_scope")
    if not isinstance(project_scope, Mapping):
        return ""
    return str(project_scope.get("user_id") or "")


def _ordered_allowed_paper_ids(state: dict[str, Any]) -> list[str]:
    """取本轮授权论文 ID，保留清单顺序。

    这里刻意不复用 `_allowed_paper_ids`：那个返回 frozenset，遍历顺序是哈希序，
    清单里的「第二篇」会和后端序数规则里的「第二篇」指到不同论文上。

    Args:
        state: 当前 RAGState 字典态
    Returns:
        list[str]: 去重后保序的论文 ID；无授权论文时为空列表
    """
    ordered = _paper_ids_from_focus(state) or _paper_ids_from_project_scope(state)
    seen: set[str] = set()
    unique: list[str] = []
    for paper_id in ordered:
        if paper_id not in seen:
            seen.add(paper_id)
            unique.append(paper_id)
    return unique


def _state_str_map(state: Mapping[str, Any], key: str) -> dict[str, str]:
    """读出 state 上的字符串映射字段，供节点读后整份写回。

    `source_status` / `source_status_reasons` 没有 reducer，节点返回的是整份替换，
    只写自己那一条会把同回合别处写的状态位抹掉。

    Args:
        state: 当前 RAGState 字典态
        key: 要读的字段名
    Returns:
        dict[str, str]: 字段现值的浅拷贝；不存在或类型不符时为空字典
    """
    current = state.get(key)
    if not isinstance(current, Mapping):
        return {}
    return {str(name): str(value) for name, value in current.items()}


def _adapters_for_plan(
    plan: Any,
    adapters: Mapping[str, EvidenceAdapter],
    optional_adapters: Mapping[str, EvidenceAdapter],
) -> dict[str, EvidenceAdapter]:
    """按 research_plan 实际需要的 source_type 选择本轮要启用的适配器集合。"""
    needed = {unit.evidence_policy.source_type for unit in plan.units}
    selected = {source: adapter for source, adapter in optional_adapters.items() if source in needed}
    return {**adapters, **selected}


# 上下文预装载失败的稳定错误码，与 source_status.context_load=failed 一同暴露
_CONTEXT_LOAD_FAILED_CODE = "planner_context_load_failed"

# 三个装载面标识，随失败一起暴露：哪一层塌了决定往哪儿查
_CONTEXT_FACE_PAPER_DB = "论文库"
_CONTEXT_FACE_SECTION = "章节查询"
_CONTEXT_FACE_GRAPH = "图谱"

# 论文库记录里表示「分析已就绪」的 analysis_status 取值
_ANALYSIS_READY_STATUS = "analysis_ready"

# 论文清单里的索引状态取值。四态如实区分，缺字段不用空串冒充有值：
# 「库记录已索引、取不到内容」是真实的索引漂移，不并进「未索引」
_INDEX_STATUS_ANALYZED = "已分析"
_INDEX_STATUS_INDEXED = "已索引"
_INDEX_STATUS_ABSTRACT_ONLY = "仅有摘要"
_INDEX_STATUS_NOT_INDEXED = "未索引"
_INDEX_STATUS_VECTORIZED_NO_ROWS = "库记录已索引、取不到内容"
_INDEX_STATUS_NOT_QUERIED = "本段未查询"

# 某篇论文查不到章节时清单段里的如实写法
_SECTION_EMPTY_TEXT = "无章节（未索引或零行）"

# 图谱段无数据时的写法。图库空、查不到都写这行，具体原因进 source_status_reasons
_GRAPH_STATUS_EMPTY_TEXT = "图谱状态: 无"

# 深度评估失败的稳定错误码，与 source_status.depth_assessment=failed 一同暴露
_DEPTH_ASSESSMENT_FAILED_CODE = "planner_depth_assessment_failed"

# 记忆读取失败的稳定错误码，与 source_status.memory_read=failed 一同暴露
_MEMORY_READ_FAILED_CODE = "memory_read_failed"

# 记忆读取四态里的失败态取值，只有它触发失败分支：empty 是「项目内确实没有相关记忆」，
# 属正常结果，Planner 据此继续推进而不猜测
_MEMORY_READ_FAILED_STATUS = "failed"

# 不检索论文的最浅处理深度取值
_MINIMAL_DEPTH = "minimal"

# 本轮没走计划与取证时 planner 的状态取值。与直答载荷同取值，
# 也在 research_memory_writeback 的论文取证换算表里，能读成「本轮没取论文证据」
_PLANNER_SKIPPED_STATUS = "skipped"

# minimal 直答保留独立失败身份；本地读取状态由实际回执派生
_DIRECT_ANSWER_FAILED_CODE = "minimal_direct_answer_failed"

# 探测单元的三处占位文案：适配器只读筛选条件，这些字段不参与取数，仅为满足契约下限

# 探测回边的决策理由：进 planner_recovery_trace，排查时据此区分「这一轮回边是探测还是重规划」

# 一条探测结果最多回灌多少条发现行：探测是「补上下文」不是「搬运正文」，条数上限防
# 单次探测把提示词撑爆；条目里已含人类可读的摘要与来源，模型不需要全部正文


def _failure(code: str, message: str) -> PlannerFailure:
    """参数为错误码和原因；返回错误身份，读取事实留给拥有本轮状态的失败边界投影。"""
    return PlannerFailure(
        code=code,
        message=message or "planner failed",
        trace={"code": code, "message": message},
    )


def _failure_from_exc(exc: Exception, default_code: str) -> PlannerFailure:
    """
    把分析阶段异常映射为 PlannerFailure。

    StructuredOutputError 使用稳定 structured 码与用户摘要；其它异常沿用 default_code。

    参数:
        exc: 捕获的异常
        default_code: 非结构化失败时的失败码
    返回:
        PlannerFailure
    """
    # 1. 【研究失败】【结构错误】这里只映射模型错误身份，读取与否由本轮累计事实判定
    from app.services.llm_service import StructuredOutputError

    if isinstance(exc, StructuredOutputError):
        code = "planner_structured_output_failed"
        message = exc.user_summary or str(exc)
        return PlannerFailure(
            code=code,
            message=message,
            local_evidence_execution="not_run",
            trace={
                "code": code,
                "message": message,
                "structured_output": {
                    "code": exc.code,
                    "technical_message": exc.technical_message,
                    "details": dict(exc.details or {}),
                },
            },
        )
    # 2. 【研究失败】【供应商说明】用户摘要沿用客户端脱敏结果，技术上下文仅留在诊断内
    if isinstance(exc, ProviderHTTPError):
        return _failure(default_code, exc.user_summary).model_copy(update={"trace": {
            "code": default_code, "message": exc.user_summary, "technical_message": str(exc),
            "provider_error_type": exc.provider_error_type, "provider_error_code": exc.provider_error_code,
        }})
    # 3. 【研究失败】【其他错误】保留默认故障身份，不据错误类型推测是否读过资料
    return _failure(default_code, provider_failure_message(exc))
