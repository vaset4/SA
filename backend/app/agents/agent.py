"""
主图 Agent 定义。

主链收敛后，顶层图只剩一条执行路径：
  guardrails（安全底线）→ router（意图归一）→ rag_qa（Agentic RAG 子图）→ END

收敛前按意图分派的 arxiv_search / deep_research / citation / literature_review /
free_chat 五条并列答题分支已下线：入口不再选择“走哪条链”，所有问题都进 RAG 子图，
由子图内部规划决定怎么检索、怎么作答。这些能力后续作为主链可调用的工具接线。
arXiv 检索已移出顶层图，见 `app/nodes/arxiv_search.py`。

这份主图的核心职责不是“自己实现所有业务”，而是：
1. 统一维护顶层 AgentState
2. 先经过 guardrails，再在 router 里把意图归一成产品层主语义
3. 在 compile 时接入 LangGraph checkpointer，让所有路径共享同一套 checkpoint 机制
"""

import asyncio
import logging

from langgraph.config import get_stream_writer
# BaseCheckpointSaver 是 LangGraph checkpointer 的抽象接口。
# 本项目运行时默认由 API 层传入 PostgreSQL 实现，但这里保留抽象类型以便测试替换。
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END

from app.models.models import (
    AgentState,
    RAGState,
)
from app.models.structured_outputs import RouteDecision
from app.nodes import create_paper_analyzer, create_rag_node
from app.nodes.guardrails_node import guardrails_node
from app.services.internal_research_service import build_internal_research_contract
from app.config import get_settings
from app.middleware.observability import log_structured_event

settings = None
logger = logging.getLogger(__name__)


INTENT_PAPER_ANALYSIS = "paper_analysis"
INTENT_INTERNAL_RESEARCH = "internal_research"
INTENT_EXTERNAL_RESEARCH = "external_research"
INTENT_GRAPH_QUERY = "graph_query"


def _settings():
    override = globals().get("settings")
    if override is not None:
        return override
    return get_settings()


def _log_agent_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    """统一输出主图内部的结构化事件。"""
    log_structured_event(logger, event, level=level, **fields)


# ============================================================
# 意图路由
# ============================================================

# 支持的意图类型（用于条件边路由和类型提示）
INTENT_ARXIV_SEARCH  = "arxiv_search"
INTENT_RAG_QA        = "rag_qa"
INTENT_DEEP_RESEARCH = "deep_research"
INTENT_CITATION      = "citation"
INTENT_LITERATURE_REVIEW = "literature_review"
INTENT_FREE_CHAT     = "free_chat"

CORE_INTENTS = {
    INTENT_PAPER_ANALYSIS: INTENT_PAPER_ANALYSIS,
    INTENT_INTERNAL_RESEARCH: INTENT_INTERNAL_RESEARCH,
    INTENT_EXTERNAL_RESEARCH: INTENT_EXTERNAL_RESEARCH,
    INTENT_GRAPH_QUERY: INTENT_GRAPH_QUERY,
}

LEGACY_INTENT_TO_CORE_INTENT = {
    INTENT_ARXIV_SEARCH: INTENT_EXTERNAL_RESEARCH,
    INTENT_RAG_QA: INTENT_INTERNAL_RESEARCH,
    INTENT_DEEP_RESEARCH: INTENT_EXTERNAL_RESEARCH,
    INTENT_CITATION: INTENT_GRAPH_QUERY,
    INTENT_LITERATURE_REVIEW: INTENT_EXTERNAL_RESEARCH,
    INTENT_FREE_CHAT: INTENT_INTERNAL_RESEARCH,
}

# 所有合法意图的集合，用于验证 LLM 输出
VALID_INTENTS = {
    INTENT_ARXIV_SEARCH,
    INTENT_RAG_QA,
    INTENT_DEEP_RESEARCH,
    INTENT_CITATION,
    INTENT_LITERATURE_REVIEW,
    INTENT_FREE_CHAT,
}


def core_route_intents() -> set[str]:
    """返回产品层主语义集合，供测试和外层契约复用。"""
    return set(CORE_INTENTS.values())


def normalize_route(route: str | RouteDecision | None) -> str:
    """把旧执行意图归并到产品层四类主语义。"""
    if isinstance(route, RouteDecision):
        route = route.route
    normalized = str(route or "").strip()
    if normalized in CORE_INTENTS:
        return normalized
    return LEGACY_INTENT_TO_CORE_INTENT.get(normalized, INTENT_EXTERNAL_RESEARCH)


def resolve_execution_intent(intent: str | None) -> str:
    """把产品层主语义映射回当前可执行节点名称。"""
    normalized = str(intent or "").strip()
    if normalized == INTENT_PAPER_ANALYSIS:
        return INTENT_RAG_QA
    if normalized == INTENT_INTERNAL_RESEARCH:
        return INTENT_RAG_QA
    if normalized == INTENT_EXTERNAL_RESEARCH:
        return INTENT_DEEP_RESEARCH
    if normalized == INTENT_GRAPH_QUERY:
        return INTENT_CITATION
    return normalized or INTENT_DEEP_RESEARCH


def _scope_core_intent(state: AgentState) -> str:
    """
    按本轮范围事实判定产品层主语义：有没有锁定项目和论文。

    Args:
        state: 主图状态，读取 structured_query

    Returns:
        str: 锁定了 project_id 或 paper_ids 时为 internal_research，否则为 external_research
    """
    structured_query = state.get('structured_query')
    if not isinstance(structured_query, dict):
        return INTENT_EXTERNAL_RESEARCH
    if structured_query.get('project_id') or structured_query.get('paper_ids'):
        return INTENT_INTERNAL_RESEARCH
    return INTENT_EXTERNAL_RESEARCH


async def router_node(state: AgentState) -> AgentState:
    """
    意图归一节点：把本轮意图折算成产品层主语义，不再分派执行分支。

    收敛后顶层图只有一条主链，所以这里只回答两个事实问题：
      1. 调用方是否显式预设了 intent
      2. 否则本轮有没有锁定项目或论文范围

    结果写回 `core_intent`（产品层主语义）与 `intent`（执行语义标签）。
    后者仍被图超时口径与会话摘要消费（graph_facade、research_api），但不决定链路。

    Args:
        state: 主图状态，读取 intent 与 structured_query

    Returns:
        AgentState: 包含 intent 与 core_intent 的状态更新
    """
    preset_intent = str(state.get('intent') or '').strip()
    if preset_intent in VALID_INTENTS or preset_intent in CORE_INTENTS:
        # 1. API 层显式传入意图时以它为准，例如 /chat 与 /research
        core_intent = normalize_route(preset_intent)
        outcome = 'preset'
    else:
        # 2. 没有预设意图时，按本轮范围事实判定
        core_intent = _scope_core_intent(state)
        outcome = 'scope'
    execution_intent = resolve_execution_intent(core_intent)
    _log_agent_event(
        "agent_router_completed",
        outcome=outcome,
        intent=execution_intent,
        core_intent=core_intent,
    )
    return {'intent': execution_intent, 'core_intent': core_intent}


def route_after_guardrails(state: AgentState) -> str:
    """根据 guardrails 结果决定继续进入 router 还是直接结束。"""
    if state.get("guardrails_action") == "reject":
        return "reject"
    return "router"


# ============================================================
# RAG 路径
# ============================================================

# RAG 子图 → 主图的反向回填白名单。
# 跨图字段必须同时声明在 RAGState 与 AgentState 并列在这里，漏一处就在主图入口被静默丢弃。
# 末尾六个是项目科研回合的回合级字段，由统一终结节点产出，API 与 SSE 的 done payload 都取自它们。
_RAG_BRIDGE_KEYS = (
    'citations',
    'research_scope',
    'research_objects',
    'note_actions',
    'paper_sources',
    'chunk_citations',
    'graph_evidence',
    'evidence_summary',
    'evidence_records',
    'missing_evidence',
    'answer_brief',
    'synthesis_state',
    'dialogue_mode',
    'answer_confidence',
    'search_results',
    'graph_context',
    'source_status',
    'source_status_reasons',
    'runtime_chain',
    'retrieval_mode',
    'retrieval_reasoning',
    'graph_mode',
    'failure_reason',
    'request_id',
    'research_plan',
    'research_unit_results',
    'run_id',
    'planner_read_receipts',
    'planner_execution_history',
    'planner_locator_facts',
    'planner_draft',
    'planner_budget',
    'planner_probe_results',
    'planner_observation',
    'planner_loop_decision',
    'planner_recovery_trace',

    'coverage_review',
    'planner_trace',
    'planner_final_blocks',
    'planner_final_candidate',
    'planner_final_validation_errors',
    'final_delivery_report',
    'final_citation_bindings',
    'final_generation_attempt',
    'final_delivery_applicability',
    'evidence_sufficiency_report',
    'repair_query_plans',
    'planner_repair_trace',
    'planner_repair_merge_report',
    'final_answer_eligibility_report',
    'error_code',
    'error_summary',
    'turn_depth',
    'turn_depth_assessment',
    'depth_transitions',
    'memory_read_result',
    'turn_outcome',
    'memory_writeback',
)


def _rag_input_state(state: AgentState) -> RAGState:
    """参数为主图状态；返回保留运行身份、累计成果、作用域及会话上下文的 RAG 输入。"""
    rag_state: RAGState = {
        'user_query': state.get('user_query', ''),
        'iteration':  0,
        'should_continue': True,
        # LangGraph 的 InjectedState("source_status_reasons") 在 tool 调用时要求 key 存在；
        # 若任何工具（含 forced scoped_paper_search / graph_search）在 source_status_reasons
        # 尚未被任何节点写入前先被触发，会抛 KeyError。预置空 dict 避免这条尾部依赖。
        'source_status_reasons': {},
    }
    # 1. 【研究执行】【子图接续】主图恢复的材料、凭据和尝试历史随同一运行一起进入 RAG
    for key in (
        'run_id',
        'research_plan',
        'research_unit_results',
        'evidence_records',
        'planner_read_receipts',
        'planner_execution_history',
        'planner_final_candidate',
        'planner_locator_facts',
        'planner_draft',
        'planner_budget',
        'planner_probe_results',
        'planner_observation',
        'planner_loop_decision',
        'planner_recovery_trace',
        'final_delivery_report',
        'final_citation_bindings',
        'final_generation_attempt',
        'final_delivery_applicability',
        'coverage_review',
        'evidence_sufficiency_report',
        'turn_depth',
        'turn_depth_assessment',
        'memory_read_result',
        '_paper_list_lines',
        '_section_structure_lines',
        '_graph_status_lines',
        'filter_expr',
        'structured_query',
        'dialogue_mode',
        'project_scope',
        'paper_focus',
        'session_id',
        'conversation_id',
        'context_summary',
        'chat_history',
    ):
        value = state.get(key)
        if value is not None:
            rag_state[key] = value
    return rag_state


async def rag_node(state: AgentState) -> AgentState:
    """
    RAG 问答路径入口。

    这里会创建并执行 Agentic RAG 子图，再把子图产出的回答写回主图 `report`。
    主图本身不直接展开 RAG 内部节点，只负责把用户问题桥接到子图状态。

    Args:
        state: 主图 AgentState，至少要包含 user_query，并可能携带作用域和会话上下文

    Returns:
        AgentState: 已回填最终回答、研究合同、证据元数据和 trace 字段的主图状态更新
    """
    _log_agent_event("agent_rag_started", query=state.get("user_query", ""), session_id=state.get("session_id"))
    rag_graph = create_rag_node()
    rag_state = _rag_input_state(state)
    # 2. 执行 RAG 子图，拿到最终 values 输出
    result = await _run_rag_graph_stream(rag_graph, rag_state)
    answer = result.get('answer', '')
    citations = result.get('citations', [])
    if not answer:
        answer = '知识库中未找到相关内容，请先将相关论文加入知识库。'
    if all(key in result for key in ("research_scope", "research_objects", "note_actions", "synthesis_state")):
        research_contract = {
            "research_scope": result.get("research_scope", {}),
            "research_objects": result.get("research_objects", []),
            "note_actions": result.get("note_actions", []),
            "synthesis_state": result.get("synthesis_state", {}),
            "dialogue_mode": result.get("dialogue_mode"),
        }
    else:
        research_contract = build_internal_research_contract(
            query=state.get("user_query", ""),
            answer=answer,
            citations=citations,
            structured_query=state.get("structured_query"),
            answer_confidence=str(result.get("answer_confidence") or ""),
            failure_reason=str(result.get("failure_reason") or ""),
        )
    # 3. 把子图结果整理回主图统一输出结构，供 API 层和 observability 复用
    result_count = max(
        len(result.get("citations", [])),
        len(result.get("chunk_citations", [])),
        len(result.get("graph_evidence", [])),
        len((result.get("rag_result") or {}).get("sources", [])) if isinstance(result.get("rag_result"), dict) else 0,
        len(result.get("paper_sources", [])),
    )
    # 规划失败有用户文案 answer，不得记成无差别 answered
    source_status = result.get('source_status') if isinstance(result.get('source_status'), dict) else {}
    from app.services.internal_research.research_planner_final_delivery import (
        delivery_report_from_state, project_final_delivery,
    )
    delivery = delivery_report_from_state(result)
    if delivery is not None:
        outcome = project_final_delivery(delivery)['chat_result_status']
    elif source_status.get('planner') == 'failed' or result.get('error_code'):
        outcome = 'planner_failed'
    elif not result.get('answer'):
        outcome = 'fallback_empty_answer'
    else:
        outcome = 'answered'
    _log_agent_event(
        "agent_rag_completed",
        request_id=result.get('request_id'),
        session_id=state.get('session_id'),
        mode=result.get('retrieval_mode'),
        source_status=result.get('source_status'),
        outcome=outcome,
        result_count=result_count,
        error_code=result.get('error_code'),
        error_summary=result.get('error_summary'),
    )
    bridged_result: AgentState = {
        'answer': answer,
        'report': answer,
        'rag_result': result.get('rag_result'),
        **research_contract,
    }
    for key in _RAG_BRIDGE_KEYS:
        if key in result:
            bridged_result[key] = result.get(key)
    return bridged_result


# RAG 子图节点 → 面向用户的阶段进度文案；决定哪些节点会向前端透出进度
_RAG_STAGE_LABELS = {
    'planner_analyze': '正在分析问题，拆解研究单元',
    'planner_validate': '正在校验研究计划',
    'planner_execute': '正在检索论文证据',
    'planner_synthesize': '正在综合最终答案',
}


def _planner_stage_detail(node_name: str, update: dict) -> dict:
    """给 planner 各阶段补充关键细节：计划单元数、抽到的主张与证据数量、证据是否充分。"""
    detail: dict = {}
    if node_name == 'planner_analyze':
        # 重规划回边时透出「重规划中（第 N 次）」可区分信号，让前端把"进度回到拆解"
        # 解释成"换思路重新规划"而非倒退（grill 决策 4；展示样式归前端）
        budget = update.get('planner_budget')
        replan_count = budget.get('replan_count') if isinstance(budget, dict) else 0
        if isinstance(replan_count, int) and replan_count > 0:
            detail['replan_count'] = replan_count
            detail['replanning'] = True
    elif node_name == 'planner_validate':
        plan = update.get('research_plan')
        if isinstance(plan, dict) and isinstance(plan.get('units'), list):
            detail['unit_count'] = len(plan['units'])
    elif node_name == 'planner_execute':
        results = update.get('research_unit_results')
        if isinstance(results, list):
            detail['claim_count'] = sum(len(r.get('claims') or []) for r in results if isinstance(r, dict))
        records = update.get('evidence_records')
        if isinstance(records, list):
            detail['evidence_count'] = len(records)
        report = update.get('evidence_sufficiency_report')
        if isinstance(report, dict):
            detail['evidence_sufficient'] = not report.get('blocking_unit_ids')
    return detail


def _emit_rag_stage_event(node_name: str, update: dict) -> None:
    """把 RAG 子图的关键阶段翻译成主图可转发的流式事件。"""
    writer = get_stream_writer()
    label = _RAG_STAGE_LABELS.get(node_name)
    if not label:
        return
    payload = {'stage': node_name, 'node': node_name, 'detail': label}
    payload.update(_planner_stage_detail(node_name, update))
    writer(payload)


async def _consume_rag_graph_stream(rag_graph, rag_state: RAGState, sink: dict) -> None:
    """消费 RAG 子图流：透出阶段事件并把最新 values 落进 sink（供超时兜底读取部分结果）。"""
    async for mode, chunk in rag_graph.astream(rag_state, stream_mode=['updates', 'values']):
        if mode == 'updates' and isinstance(chunk, dict):
            for node_name, update in chunk.items():
                if isinstance(update, dict):
                    _emit_rag_stage_event(node_name, update)
        elif mode == 'values' and isinstance(chunk, dict):
            sink.clear()
            sink.update(chunk)


async def _run_rag_graph_stream(rag_graph, rag_state: RAGState) -> dict:
    """运行 RAG 子图流，并同时采集阶段事件与最终 values。

    规划 ReAct 循环最坏会跑 replan×N 轮检索，裸 astream 无硬总超时可能长时间无人管。
    这里加一层 config 可调、可禁用的整图兜底超时（grill 决策 5b，已获用户同意）：
    超时不静默吞掉，返回带 error_code 的显式失败载荷，并保留已采集的部分 values 供追溯。

    递归上限是另一道兜底（图的固有属性，挂在子图 compile 出口）：探测回边在取消计数
    额度后靠「申请不重复」收敛，判据失效时只能由它截停。超限时 GraphRecursionError 会
    逃出 astream，按与超时同一套写法接成可读失败，不让框架错误直接冒给用户。
    """
    result: dict = {}
    timeout = _rag_graph_total_timeout()
    consume = _consume_rag_graph_stream(rag_graph, rag_state, result)
    try:
        # 1. 总超时未配置或已禁用时直接跑，递归上限兜底不受超时开关影响
        if timeout is None:
            await consume
        else:
            await asyncio.wait_for(consume, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(
            "【规划循环】【超时】RAG 子图执行超过 %.0fs 兜底上限，强制收敛并返回显式失败",
            timeout,
        )
        return _rag_graph_timeout_result(result, timeout)
    except GraphRecursionError:
        limit = getattr(rag_graph, 'config', None) or {}
        logger.error(
            "【规划循环】【递归上限】RAG 子图递归超过 %s 兜底上限，强制收敛并返回显式失败",
            limit.get('recursion_limit'),
        )
        return _rag_graph_recursion_result(result, limit.get('recursion_limit'))
    return result


def _rag_graph_total_timeout() -> float | None:
    """读取整图兜底超时秒数；<=0 或未配置视为禁用（返回 None）。"""
    raw = getattr(_settings(), 'rag_planner_loop_total_timeout_seconds', 0.0)
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _rag_graph_timeout_result(partial: dict, timeout: float) -> dict:
    """整图超时兜底载荷：保留已采集的部分 values，叠加显式超时错误码，不伪装成功。"""
    result = dict(partial)
    result.setdefault('answer', '本次研究耗时超过系统上限，已中止；请缩小论文范围或稍后重试。')
    result['error_code'] = 'planner_graph_timeout'
    result['error_summary'] = f'研究流程超过 {timeout:.0f} 秒兜底上限被中止'
    source_status = dict(result.get('source_status') or {})
    source_status['planner'] = 'failed'
    result['source_status'] = source_status
    return _rag_graph_terminal_delivery(result)


def _rag_graph_recursion_result(partial: dict, limit: int | None) -> dict:
    """递归上限兜底载荷：保留已采集的部分 values，叠加显式错误码，不伪装成功。"""
    result = dict(partial)
    result.setdefault('answer', '本次研究反复回到同一步而未能收敛，已中止；请缩小论文范围或把问题问得更具体。')
    result['error_code'] = 'planner_graph_recursion_limit'
    result['error_summary'] = f'研究流程递归超过 {limit if limit else "配置"} 步上限被中止'
    source_status = dict(result.get('source_status') or {})
    source_status['planner'] = 'failed'
    result['source_status'] = source_status
    return _rag_graph_terminal_delivery(result)


def _rag_graph_terminal_delivery(result: dict) -> dict:
    """【研究交付】【图终止】参数为真实图失败载荷；返回执行后失败报告或明确不适用入口。"""
    if result.get('final_delivery_applicability') != 'research':
        return {**result, 'final_delivery_applicability': result.get('final_delivery_applicability')
                or 'pre_execution_failure'}
    from app.services.internal_research.research_planner_final_delivery import (
        delivery_state_fields, make_terminal_delivery_report,
    )
    attempt = result.get('final_generation_attempt') or {}
    report = make_terminal_delivery_report(
        status='failed', reason=result['error_summary'], reason_code=result['error_code'],
        stage='graph', entrypoint='research_execution_failure',
        attempt_id=attempt.get('attempt_id', ''), exposure=attempt.get('exposure'),
    )
    return {**result, **delivery_state_fields(report),
            'source_status': {**result.get('source_status', {}), 'planner_final': 'failed'},
            'source_status_reasons': {**result.get('source_status_reasons', {}), 'planner_final': result['error_summary']},
            'planner_final_blocks': [], 'citations': [], 'chunk_citations': []}


# ============================================================
# 主图构建
# ============================================================

def build_agent_workflow() -> StateGraph:
    """
    构造尚未 compile 的主图 workflow。

    图结构（唯一主链）：
      START → guardrails → router → rag_qa → END
                  └─ reject ─────────────→ END

    三步含义：
      guardrails 只做安全底线判断（空/超长/敏感词），命中敏感主题时直接 END
      router     把意图归一成产品层主语义，不再分派执行分支
      rag_qa     进入 Agentic RAG 子图，检索与作答全部由子图内部规划决定

    这里只负责节点和边定义，不直接承担 compile 入口职责。

    Returns:
        StateGraph: 尚未 compile 的主图 workflow
    """
    # 顶层 workflow 只关心 AgentState。更细粒度的局部状态由各子图自己维护。
    workflow = StateGraph(AgentState)

    # 1. 注册三个主链节点：安全护栏、意图归一、RAG 子图入口
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("router", router_node)
    workflow.add_node("rag_qa", rag_node)

    # 2. 护栏结果决定是继续进主链还是直接结束（reject 不进 router）
    workflow.add_edge(START, "guardrails")
    workflow.add_conditional_edges(
        "guardrails",
        route_after_guardrails,
        {"router": "router", "reject": END},
    )

    # 3. 意图归一后无条件进 RAG 子图，子图内部已完成作答，直接收束到 END
    workflow.add_edge("router", "rag_qa")
    workflow.add_edge("rag_qa", END)

    return workflow


def create_agent(checkpointer: BaseCheckpointSaver | None = None):
    """
    兼容包装层：创建带路由的主图 Agent。

    compile 的真实入口已经收敛到 `app.agents.graph_factory`，这里保留
    `create_agent()` 仅用于兼容现有调用点与测试。
    """
    from app.agents.graph_factory import create_compiled_graph

    return create_compiled_graph(checkpointer=checkpointer)


# ============================================================
# 主函数（交互式测试）
# ============================================================

async def main():
    """
    主函数：交互式测试主图工作流。

    这里只是本地调试入口，生产环境通常通过 FastAPI + agent_singleton 调用本图。
    运行：python -m app.agents.agent
    """
    print("=" * 60)
    print("  AI 论文研究助手 - Phase A")
    print("=" * 60)
    print("支持的功能：")
    print("  - 论文问答（主链：guardrails → router → rag_qa）")
    print("  - 项目/论文范围内的研究讨论")
    print("=" * 60)

    agent = create_agent()

    user_input = input("\n请输入你的问题: ").strip()
    if not user_input:
        user_input = "搜索2024年最新的RAG论文"
        print(f"使用默认查询: {user_input}")

    # 构建初始状态
    initial_state: AgentState = {
        'user_query': user_input,
        'intent': None,
        'structured_query': None,
        'report': None,
        'papers': [],
        'filtered_papers': [],
        'rag_result': None,
    }

    print(f"\n问题: {user_input}")
    print("=" * 60)

    result = await agent.ainvoke(initial_state)

    print("\n" + "=" * 60)
    print(f"意图: {result.get('intent', 'unknown')}")

    if result.get('papers'):
        print(f"找到论文数: {len(result['papers'])}")

    if result.get('report'):
        print(f"\n--- 报告 ---\n{result['report']}")
    else:
        print("未生成报告")


if __name__ == "__main__":
    asyncio.run(main())
