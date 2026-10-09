"""
Project-scoped RAG subgraph assembly.

There is exactly one chain, and it is planner-first:
`rag_init -> planner_analyze -> planner_validate
-> planner_execute -> planner_synthesize -> write_back`.

Turn depth shortens this same chain rather than forking a second one: a `minimal`
turn needs no paper fact, so analyze hands off straight to `planner_synthesize`,
skipping plan validation and retrieval. Still one runtime model, one set of nodes.

Intent classification and focus resolution live inside the planner's first analysis;
there is no separate pre-planner node judging them a second time.

No feature flag selects a second runtime model; rollback is `git revert`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from functools import partial
from typing import Any, Mapping, TypedDict

from langgraph.graph import END, START, StateGraph

import app.middleware.observability as observability
import app.services.paper_db as paper_db
import app.services.rag_answer_service as rag_answer_service
import app.services.rag_trace_service as rag_trace_service
import app.services.retrieval_service as retrieval_service
from app.config import get_settings, get_vector_rerank_top_k
from app.middleware.trace_context import evidence_scope, record_evidence_observation, record_retrieval_hits
from app.models import evidence_layers as layers
from app.models.models import RAGState
from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport, KnowledgeReadRequest
from app.services.knowledge_access_service import read_knowledge, resolve_section_paths, published_chunk_order
from app.models.research_contracts import EvidenceRecord, RetrievalScopeError
from app.nodes import rag_evaluate, rag_search
from app.services.graph.research_object_query_service import get_research_object_query_service
from app.services.llm_service import get_llm
from app.services.internal_research.research_planner_pipeline import (
    analyze_planner_state,
    execute_planner_state,
    synthesize_planner_state,
    validate_planner_state,
    execution_content_identity,
)
from app.services.internal_research.research_planner_direct_text import rank_direct_text_hits
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError,
    EvidenceCapabilitySoftError,
)
from app.services.internal_research.research_planner_lifecycle import settle_owned_work
from app.services.milvus_store import get_milvus_store
from app.services.source_io_service import run_source_io
from app.services.retrieval_provider_registry import get_default_vector_provider, get_graph_provider
from app.services.retrieval_query_expansion import expand_lightrag_locator_queries

logger = logging.getLogger(__name__)
llm = None

# 不检索论文的最浅处理深度取值，路由据它把直答回合从 validate/execute 上摘下来
_MINIMAL_DEPTH = "minimal"
# 研究对象图里论文侧端点的 id 前缀，形如 "paper:<paper_id>"，节点侧端点是裸节点 id
_PAPER_NODE_ID_PREFIX = "paper:"
# analyze 阶段的两个终态决策：硬失败，以及「先问用户一句」这种正常终止。
# 两者的用户可见载荷都已在 analyze 里组装好，后面没有可综合的东西，直奔终态节点
_ANALYZE_TERMINAL_DECISIONS = frozenset({"terminate_fail", "terminate_honest"})
# 出计划前申请补充上下文的决策动作：这一轮没产草案，路由据它回 analyze 重出计划。
# 取值须与 PlannerDecisionAction 词表一致：本文件与邻居一样按词表写字面量，
# 写岔了回边认不出来，探测态会带着没草案的态落到默认成功边进校验、当场报错
_PROBE_DECISION_ACTION = "probe_context"
# 规划子图自身的递归上限兜底值，与配置默认保持一致：旧替身 settings 未声明该字段时，
# 读配置会落空，回落到这个值。它是基础设施护栏不是研究额度，只保证「回边多走几步仍然终止」
_DEFAULT_GRAPH_RECURSION_LIMIT = 50
# 论文详情按作者顺序并列的字段与中文标签：字段名取自 analysis 的实际产出，
# 库里没有该字段时整行不出现，不拿空标签占位
_ANALYSIS_FIELD_LABELS = (
    ("problem", "问题"),
    ("method", "方法"),
    ("contribution", "贡献"),
    ("limitation", "局限"),
    ("conclusion", "结论"),
    ("keywords", "关键词"),
)


class _SettingsProxy:
    """
    延迟代理全局配置，并保留旧测试直接 patch `rag_node.settings.*` 的入口。

    这是一种兼顾新实现与旧测试兼容性的折中写法：
    运行时按需读最新 settings，测试里仍可像旧版一样直接 monkeypatch。
    """

    def __init__(self):
        object.__setattr__(self, "_overrides", {})

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(get_settings(), name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_overrides")[name] = value

    def __delattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            del overrides[name]
            return
        raise AttributeError(name)


settings = _SettingsProxy()


def _settings():
    return settings


def _llm():
    return llm or get_llm()


def get_vector_store():
    return get_default_vector_provider()


def _search_chunks(store, query, top_k, filter_expr="", structured_query=None):
    """参数为检索器、查询和既有范围；返回业务过滤结果，同时记录真实查询顺序与范围。"""
    scope = retrieval_service.require_local_scope_from_query(structured_query)
    base_filter_expr = str(scope["filter_expr"]).strip()
    requested_filter_expr = str(filter_expr or "").strip()
    execution_filter_expr = base_filter_expr
    if requested_filter_expr and requested_filter_expr != base_filter_expr:
        execution_filter_expr = f"({base_filter_expr}) and ({requested_filter_expr})"
    execution_scope = {**scope, "filter_expr": execution_filter_expr}
    source = _retrieval_observation_source(store)
    with evidence_scope("retrieval_query", details={"query": query, "scope": execution_scope, **source}):
        results = retrieval_service.search_chunks(store, query, top_k, scope=execution_scope)
        record_retrieval_hits("query_raw", results, query=query, scope=execution_scope, **source)
        filtered = retrieval_service.filter_hits_by_business_state(
            results, paper_db=paper_db, structured_query=structured_query,
        )
        # 【评测观察】【范围过滤】论文子集描述授权与业务筛选的交集，原文存在性仍由业务过滤器判定
        business_ids = retrieval_service.resolve_allowed_paper_ids(structured_query)
        filtered_scope = {
            **execution_scope, "business_allowed_paper_ids": sorted(business_ids),
            "filter_applied": "filter_hits_by_business_state",
        }
        if business_ids:
            authorized_ids = {str(item or "").strip() for item in execution_scope["paper_ids"]}
            filtered_scope["paper_ids"] = sorted(authorized_ids & business_ids)
        record_retrieval_hits("query_filtered", filtered, query=query, scope=filtered_scope, **source)
        return filtered


class _RetrievalObservationSource(TypedDict):
    """实际检索器的受控身份投影，未知后端名称保持缺失。"""

    provider_type: str
    retrieval_backend: str | None


def _retrieval_observation_source(store: Any) -> _RetrievalObservationSource:
    """参数为实际检索器实例；返回准确类型和已知后端身份，未知替身不冒充真实服务。"""
    provider_type = f"{type(store).__module__}.{type(store).__qualname__}"
    known_backends = {
        "app.services.milvus_store.MilvusStore": "milvus",
        "app.services.vector_store.VectorStore": "faiss",
    }
    return {"provider_type": provider_type, "retrieval_backend": known_backends.get(provider_type)}


def _dedupe_hits(groups):
    return retrieval_service.dedupe_hits(groups)


def _run_rewritten_search(store, queries, scope):
    settings = _settings()
    candidate_limit = max(settings.rag_top_k, get_vector_rerank_top_k(settings)) * 3
    return retrieval_service.run_rewritten_search(
        store,
        queries,
        scope,
        candidate_limit=candidate_limit,
    )


def _similarity_threshold():
    return retrieval_service.similarity_threshold(_settings())


def _rerank_threshold():
    return retrieval_service.rerank_threshold(_settings())


def _ensure_trace(state):
    return rag_trace_service.ensure_trace(
        state,
        settings=_settings(),
        observability_module=observability,
    )


def _with_trace(payload, trace):
    return rag_trace_service.with_trace(payload, trace)


def _update_trace_generation(
    trace,
    *,
    citations=None,
    selected_count=None,
    evidence_selection_stats=None,
    generation_status=None,
    error_kind=None,
    reason=None,
    quality_score=None,
    iteration=None,
):
    rag_trace_service.update_trace_generation(
        trace,
        citations=citations,
        selected_count=selected_count,
        evidence_selection_stats=evidence_selection_stats,
        generation_status=generation_status,
        error_kind=error_kind,
        reason=reason,
        quality_score=quality_score,
        iteration=iteration,
    )


def rag_init_node(state: RAGState) -> dict[str, Any]:
    """
    初始化 RAG 子图状态。

    Args:
        state: 主图桥接进来的初始 RAGState

    Returns:
        dict[str, Any]: 已补齐默认字段、trace 基线和未索引论文提示的子图初始状态
    """
    payload = rag_search.run_rag_init(
        state,
        ensure_trace=_ensure_trace,
        with_trace=_with_trace,
    )
    # 1. 先生成后续所有节点都会依赖的默认字段和 trace 初始态
    project_scope = state.get("project_scope") if isinstance(state.get("project_scope"), dict) else {}
    paper_ids = [
        str(pid).strip()
        for pid in (project_scope.get("paper_ids") or [])
        if str(pid or "").strip()
    ]
    unindexed = rag_answer_service.detect_unindexed_papers(
        paper_ids,
        milvus_store=get_milvus_store(),
    )
    # 2. 再补一层“哪些论文尚未完成索引”的运行时提示，便于后续回答解释空结果
    payload["unindexed_paper_ids"] = unindexed
    return payload


async def planner_analyze_node(state: RAGState) -> dict[str, Any]:
    """参数为当前图状态；模型消费累计事实并选择下一动作，读取统一由执行节点负责。"""
    return await analyze_planner_state(state, _llm(), _settings())


def planner_validate_node(state: RAGState) -> dict[str, Any]:
    """Planner 第 2 阶段：校验并标准化草案计划，生成可执行 research_plan。"""
    return validate_planner_state(state, _settings())


async def planner_execute_node(state: RAGState) -> dict[str, Any]:
    """Planner 第 3 阶段：按 research_plan 调度证据适配器，执行研究单元。"""
    return await execute_planner_state(
        state,
        llm=_llm(),
        settings=_settings(),
        adapters=_planner_adapters(state),
        optional_adapters=_planner_optional_adapters(state),
        request_resolver=partial(_planner_read_request, state),
        index_probe=_scoped_index_probe,
    )


async def planner_synthesize_node(state: RAGState) -> dict[str, Any]:
    """Planner 第 4 阶段：综合最终答案，或诚实收尾（范围内查过没找到）。"""
    return await synthesize_planner_state(state, _llm(), paper_title_resolver=_resolve_paper_titles)


async def _scoped_index_probe(scoped_ids: list[str]) -> bool:
    """探针：scoped 论文在向量库里是否确有可检索内容，供决策器判是否值得 replan。

    只在执行检索落空时被 execute 阶段调用（happy path 不触发）。任一 scoped 论文
    在库里有 chunk 即算库内有内容；全无内容 → 决策器直接诚实收尾不浪费一轮 replan。

    参数:
        scoped_ids: 本轮 scoped 论文 id 列表
    返回:
        库内是否至少有一篇 scoped 论文有可检索 chunk
    """
    if not scoped_ids:
        return False
    store = get_vector_store()
    for paper_id in scoped_ids:
        exists = getattr(store, "exists", None)
        if callable(exists) and await asyncio.to_thread(exists, paper_id):
            return True
    return False


def _resolve_paper_titles(scoped_ids: list[str]) -> dict[str, str]:
    """把 scoped 论文 id 解析成标题映射，供诚实收尾列出人可读的论文范围。

    参数:
        scoped_ids: 本轮 scoped 论文 id 列表
    返回:
        paper_id -> 标题；解析不到的条目留空由调用方回退 paper_id
    """
    from app.services import paper_db

    titles: dict[str, str] = {}
    for paper_id in scoped_ids:
        try:
            paper = paper_db.get_paper(paper_id)
        except Exception:
            continue
        title = getattr(paper, "title", "") if paper else ""
        if title:
            titles[paper_id] = str(title)
    return titles


def route_after_planner_analyze(state: RAGState) -> str:
    """analyze 之后选边：终态直奔 write_back，探测回合回本节点，minimal 直答跳 synthesize，其余进 validate。

    minimal 深度本轮不检索论文，analyze 已经把答案材料备齐、不产计划草案，
    再走 validate（校验计划）和 execute（发检索）就是空转，因此直接进 synthesize 收口。

    澄清回合（analyze 标 terminate_honest）连 synthesize 都不必进：要问用户的那句话
    已经是本轮回答，进 synthesize 反而会被那边的诚实收尾分支改写成「范围内查过没找到」。

    探测回合（analyze 标 probe_context）只取回了补充上下文、同样没产草案，回本节点重出计划：
    出草案那时提示词才带着探测结果，计划第一次真正用得上它。

    判定顺序是契约，三条短路的先后不能换：
    1. 终态必须最先认：minimal 短路与探测回边那两步 planner_loop_decision 分别是空串与
       probe_context，在 _route_by_decision 里都会落成功边；若先认它们，深度评估失败、
       记忆读不出来这类带 terminate_fail 的态会被抢走，失败被静默吞掉。
    2. 探测必须排在 minimal 之前：判成 minimal 却仍要求论文 Evidence 的回合照样产草案、
       也照样会去探测（见 _is_direct_answer_turn），探测后它没有草案。先认 minimal 会把
       这种回合送进 synthesize，而那边的直答分支要求 _is_direct_answer_turn 为真、
       常规综合分支又要求有研究计划，两边都不成立，最后落成「synthesis input invalid」
       的假失败——探测拿到的上下文白取，整轮报错收场。
       真正的直答回合不受影响：它们在第 3 步就早返回了，决策为空串，认不到探测这条边。

    参数:
        state: 当前 RAGState
    返回:
        下一个节点名：write_back / planner_analyze / planner_synthesize / planner_validate
    """
    # 1. 终态优先：analyze 已标 terminate_fail（硬失败）或 terminate_honest（先问用户），
    #    无论什么深度都直奔终态节点
    if _read_loop_decision(state) in _ANALYZE_TERMINAL_DECISIONS:
        return "write_back"
    # 2. 探测回边：这一轮只申请并取回了补充上下文，回本节点带着结果重出计划
    if _read_loop_decision(state) == _PROBE_DECISION_ACTION:
        return "planner_analyze"
    # 3. minimal 且无草案才是真短路：正常 analyze 成功必产 planner_draft，
    #    光看深度分不出「这一轮到底短路了没有」
    if str(state.get("turn_depth") or "") == _MINIMAL_DEPTH and not state.get("planner_draft"):
        return "planner_synthesize"
    return _route_by_decision(state, success_target="planner_validate")


def route_after_planner_validate(state: RAGState) -> str:
    # 校验后据决策选边：continue→execute / replan→回 analyze / terminate_fail→write_back
    if _read_loop_decision(state) == "terminate_honest" and not state.get("research_plan"):
        return "write_back"
    if (_read_loop_decision(state) == "continue"
            and ((state.get("planner_draft") or {}).get("decision") or {}).get("kind") == "finish"):
        return "planner_synthesize"
    return _route_by_decision(
        state,
        success_target="planner_execute",
        replan_target="planner_analyze",
    )


def route_after_planner_execute(state: RAGState) -> str:
    # 执行后据决策选边：continue/terminate_honest→synthesize / replan→回 analyze / repair→回 execute / terminate_fail→write_back
    decision = _read_loop_decision(state)
    if decision == "replan":
        return "planner_analyze"
    if decision == "repair_evidence":
        return "planner_execute"
    if decision == "terminate_fail":
        return "write_back"
    # continue 与 terminate_honest 都进 synthesize（诚实收尾在 synthesize 内分流模板组装）
    return "planner_synthesize"


def route_after_planner_synthesize(state: RAGState) -> str:
    """参数为已核对的候选状态；实质缺口回原规划循环，其余同次结果才进入保存。"""
    if _read_loop_decision(state) == "replan":
        return "planner_analyze"
    if _read_loop_decision(state) == "revise_answer":
        return "planner_synthesize"
    return "write_back"


def _read_loop_decision(state: RAGState) -> str:
    """读回上一步循环决策动作，供路由选边。缺省空串（走默认成功边）。"""
    return str(state.get("planner_loop_decision") or "")


def _route_by_decision(
    state: RAGState,
    *,
    success_target: str,
    replan_target: str | None = None,
) -> str:
    """据 planner_loop_decision 选边：终态失败→write_back，replan→回 analyze，否则成功边。

    路由一律只读循环决策，不再看 source_status.planner（无 reducer 的 Dict 键会被
    回边往返改写，不能作路由判据，design §5.1）。

    参数:
        state: 当前 RAGState
        success_target: 决策为 continue（或无决策）时的成功后继节点
        replan_target: 决策为 replan 时的回边目标；None 表示该阶段不支持 replan
    返回:
        下一个节点名
    """
    decision = _read_loop_decision(state)
    if decision == "terminate_fail":
        return "write_back"
    if decision == "replan" and replan_target is not None:
        return replan_target
    return success_target


def _planner_context_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """Expose coarse context passed to the planner for tests and trace."""
    return {
        "intent": state.get("intent"),
        "focus": state.get("focus"),
        "structured_query": state.get("structured_query") or {},
    }


def _planner_adapters(state: RAGState) -> dict[str, Any]:
    return {
        layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT: partial(_planner_direct_text_records, state),
        layers.PLANNER_SOURCE_LOCAL_METADATA: partial(_planner_metadata_records, state),
    }


def _planner_optional_adapters(state: RAGState) -> dict[str, Any]:
    """
    规划循环可按需取数的适配器表，供探测申请与执行取数两条路径共用。

    探测路径：出计划前的探测轮里，模型按 PlannerProbeSourceType 申请补充上下文，
    这里按申请的来源取数；表里没有来源就落 failed 并写明没接线。
    执行路径：execute 的 _adapters_for_plan 按计划单元声明的来源从同一张表挑适配器。

    Args:
        state: 当前 RAGState 字典态，各适配器闭包捕获它做归属校验与取数范围
    Returns:
        dict[str, Any]: 来源类型到适配器函数的映射
    """
    return {
        layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT: partial(_planner_structure_records, state),
        layers.PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE: partial(_planner_section_structure_records, state),
        layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT: partial(_planner_section_text_records, state),
        layers.PLANNER_SOURCE_LOCAL_CHUNK_METADATA: partial(_planner_chunk_metadata_records, state),
        # 【规划取证】【论文详情】探测枚举里模型申请得到，与执行侧同一实现，不另写适配器
        layers.PLANNER_SOURCE_LOCAL_METADATA: partial(_planner_metadata_records, state),
        # 【规划取证】【外部上下文】探测枚举覆盖不到：它只服务 execute 的按需取数，留着不是死条目
        layers.PLANNER_SOURCE_EXTERNAL_CONTEXT: partial(_planner_arxiv_records, state),
    }


async def _planner_read_request(state: RAGState, unit, _context) -> Mapping[str, Any] | None:
    """参数为授权状态、单元与执行上下文；核验读取权后返回影响正文内容的完整请求身份。"""
    source = unit.evidence_policy.source_type
    if source not in {layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT, layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT}:
        return None
    from app.services.upload_read_service import authorize_publication_read, require_publication_scope

    scope = state.get("project_scope") or {}
    snapshot = _planner_scope_publication(state)
    papers = _unit_or_state_paper_ids(state, unit.evidence_policy.target_paper_ids)
    if not snapshot or snapshot.get("user_id") != _planner_reader_user_id(state) or not snapshot.get("lease_id"):
        raise RetrievalScopeError("正文复用缺少本轮已授权的发布快照")
    if not papers or not set(papers).issubset(scope.get("paper_ids") or []):
        raise RetrievalScopeError("正文复用目标超出本轮授权范围")
    # 1. 【研究执行】【复用核验】命中前仍检查当前项目范围、论文归属、删除状态和读取占用
    if source == layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT:
        await authorize_publication_read(snapshot, user_id=_planner_reader_user_id(state), paper_ids=tuple(papers))
        request = _planner_section_request(state, unit, _context)
        return {"operation": "sections", "paper_ids": papers,
                "section_paths": list(request.section_paths), "continuations": list(request.continuations),
                "max_chars": request.max_chars}
    # 2. 【研究执行】【动态定位】图扩展查询没有冻结输出，不把原始措辞当成真实查询身份
    store = get_vector_store()
    retrieval_source = _retrieval_observation_source(store)
    if _should_expand_planner_queries(state, unit) or retrieval_source["retrieval_backend"] != "milvus":
        return None
    structured = _unit_structured_query(state, papers)
    trusted = retrieval_service.require_local_scope_from_query(structured)
    await asyncio.to_thread(require_publication_scope, trusted, collection_name=store.collection_name)
    return {
        "operation": "search_chunks", "ranking_contract": "planner-direct-text-v1", "queries": list(unit.evidence_policy.locator_queries),
        "structured_query": execution_content_identity(structured),
        "filter_expr": str(state.get("filter_expr") or ""), "top_k": _settings().rag_top_k,
        "policy": unit.evidence_policy.model_dump(mode="json", exclude={"min_citations_per_target"}),
        "parent_limit": int(_settings().rag_auto_expand_parent_limit),
        "retrieval_source": retrieval_source, "collection_name": store.collection_name,
    }


async def _planner_direct_text_records(state: RAGState, unit, _context) -> EvidenceReadResult:
    """参数为授权回合、检索单元及上下文；返回候选按原粒度读取的权威正文与独立完整性报告。"""
    store = get_vector_store()
    source = _retrieval_observation_source(store)
    with evidence_scope("direct_text_adapter", details=source):
        queries = await _planner_locator_queries(state, unit)
        # 1. 【规划取证】【定位查询】整批并行查询全部收尾后才交还执行权，避免旧批次与重试重叠
        batch = asyncio.gather(*(
            asyncio.to_thread(
                _search_chunks,
                store,
                query,
                _settings().rag_top_k,
                str(state.get("filter_expr") or ""),
                _unit_structured_query(state, unit.evidence_policy.target_paper_ids),
            )
            for query in queries
        ), return_exceptions=True)
        groups, cancellation = await settle_owned_work(batch)
        if isinstance(groups, BaseException):
            raise cancellation or groups
        # 2. 【规划取证】【定位查询】失败按查询声明序归因；取消优先，且不再合并或回读父块
        for group in groups:
            if isinstance(group, BaseException):
                if cancellation is not None:
                    raise cancellation from group
                raise group
        if cancellation is not None:
            raise cancellation
        merged = _dedupe_hits(groups)
        record_retrieval_hits("merged_hits", merged, **source)
        ranked = rank_direct_text_hits(merged, unit.evidence_policy)
        record_retrieval_hits("ranked_hits", ranked, **source)
        result = await _planner_read_direct_materials(state, unit, ranked)
        record_evidence_observation("adapter_evidence_pool", {
            "records": list(result.records), "read_report": result.read_report.to_dict(), **source,
        })
        return result


async def _planner_read_direct_materials(state: RAGState, unit, ranked: list[dict[str, Any]]) -> EvidenceReadResult:
    """参数为授权回合、单元和已排序候选；按原粒度读取发布正文并明确未读取定位。"""
    from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials

    parent = unit.evidence_policy.evidence_granularity == "parent_chunk"
    limit = int(_settings().rag_auto_expand_parent_limit) if parent else len(ranked)
    selected, pending = ranked[:limit], ranked[limit:]
    if not selected:
        if pending:
            raise ValueError("父块读取额度未允许读取任何已选材料")
        return EvidenceReadResult((), KnowledgeReadReport(operation="chunks", status="complete"))
    # 1. 【知识取证】【权威正文】按搜索实际定位回读，同批材料由知识模块共享授权与清单核验
    scope: Mapping[str, Any] = state.get("project_scope") or {}
    requested = tuple(str(hit["chunk_id"]) for hit in selected)
    paper_ids = tuple(dict.fromkeys(str(hit["paper_id"]) for hit in selected))
    result = await read_knowledge(KnowledgeReadRequest(
        operation="chunks", scope=scope, paper_ids=paper_ids, chunk_ids=requested,
        granularity="parent" if parent else "child",
    ))
    report = result.read_report
    if pending:
        remaining = tuple({"chunk_id": hit["chunk_id"], "paper_id": hit["paper_id"],
                           "reason": "parent_read_limit"} for hit in pending)
        report = replace(report, status="partial", remaining=(*report.remaining, *remaining),
                         requested=tuple(str(hit["chunk_id"]) for hit in ranked))
    records = convert_knowledge_materials(result.materials, hits=ranked)
    return EvidenceReadResult(tuple(records), report)


async def _planner_locator_queries(state: RAGState, unit) -> list[str]:
    queries = list(unit.evidence_policy.locator_queries)
    if not _should_expand_planner_queries(state, unit):
        return queries
    provider = get_graph_provider()
    if provider is None:
        raise ValueError("LightRAG query expansion requested but graph provider is unavailable")
    expansion = await expand_lightrag_locator_queries(
        query=_planner_expansion_query(state, unit),
        queries=queries,
        provider=provider,
        library_id=retrieval_service.resolve_graph_library_id(state.get("structured_query")),
        top_k=_settings().rag_top_k,
    )
    return expansion.locator_queries


def _should_expand_planner_queries(state: RAGState, unit) -> bool:
    if not getattr(_settings(), "rag_planner_lightrag_locator_enabled", False):
        return False
    text = " ".join([
        str(state.get("intent") or ""),
        str(getattr(unit, "operation_family", "") or ""),
        str(getattr(unit, "goal", "") or ""),
        str(getattr(unit, "research_action", "") or ""),
    ]).casefold()
    hints = ("explore", "overview", "synthesis", "compare", "relationship", "综述", "概览", "对比", "关系")
    return any(hint in text for hint in hints)


def _planner_expansion_query(state: RAGState, unit) -> str:
    return str(state.get("user_query") or getattr(unit, "goal", "") or "").strip()


async def _planner_metadata_records(state: RAGState, unit, _context) -> EvidenceReadResult:
    """参数为授权回合、论文详情单元及上下文；返回拥有权限的论文元数据与非正文读取说明。"""
    user_id = _planner_reader_user_id(state)
    if not user_id:
        raise RetrievalScopeError("local_metadata 取证缺少读取身份，无法校验论文归属")
    authorized = _planner_authorized_paper_ids(state)
    records: list[EvidenceRecord] = []
    paper_ids = _unit_or_state_paper_ids(state, unit.evidence_policy.target_paper_ids)
    for paper_id in paper_ids:
        # 1. 目标与授权范围求交：越界目标不读库，也不放宽本轮授权
        if paper_id not in authorized:
            continue
        paper = await asyncio.to_thread(paper_db.get_paper, paper_id, user_id)
        if not paper:
            continue
        records.append(_metadata_record(paper_id, paper))
    return _nontext_read_result(records, "paper_metadata", paper_ids)


def _nontext_read_result(
    records: list[EvidenceRecord], operation: str, requested: list[str], *,
    remaining: list[dict[str, Any]] | None = None,
) -> EvidenceReadResult:
    """参数为非正文材料、操作、请求定位和未返回部分；返回材料与独立读取说明。"""
    report = KnowledgeReadReport(
        operation=operation, status="partial" if remaining else "not_applicable",
        requested=tuple(requested), remaining=tuple(remaining or ()),
        limitations=("此来源提供定位或上下文材料，未读取本地论文正文",),
    )
    return EvidenceReadResult(tuple(records), report)


def _planner_authorized_paper_ids(state: RAGState) -> set[str]:
    """取本轮授权论文集合，供按 ID 直读的适配器在真读之前核对目标。"""
    project_scope = state.get("project_scope") if isinstance(state.get("project_scope"), dict) else {}
    return {str(pid).strip() for pid in (project_scope.get("paper_ids") or []) if str(pid or "").strip()}


async def _planner_structure_records(state: RAGState, unit, _context) -> EvidenceReadResult:
    """参数为授权回合、图谱关系单元及上下文；返回真实关系记录与独立条数额度说明，服务失败明确上抛。"""
    paper_ids = _unit_or_state_paper_ids(state, unit.evidence_policy.target_paper_ids)
    # 1. 真查之前先校验目标落在授权范围内：图谱按 ID 直查、不带归属条件，范围外的 ID 会读出别人的关系
    _reject_unauthorized_graph_targets(state, paper_ids)
    max_rows = int(_settings().rag_planner_graph_relation_max_rows)
    try:
        payload = await get_research_object_query_service().query_project_subgraph(paper_ids)
    except Exception as exc:
        # 2. 图库不可用是软失败：带上失败面如实抛出，执行层只阻塞该单元、整轮继续
        raise EvidenceCapabilitySoftError(
            layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT,
            f"{type(exc).__name__}: {exc}",
        ) from exc
    # 3. 边端点是节点 id，先映射成人可读名字再拼三元组文案，模型看不懂裸 id
    edges = list(payload.get("edges") or [])
    names = _graph_endpoint_names(payload.get("papers") or [], payload.get("nodes") or [])
    records = [_graph_relation_record(edge, names) for edge in edges[:max_rows]]
    # 4. 【知识取证】【图谱额度】命中总量和未返回部分独立报告，不创建伪证据
    remaining = [{"reason": "graph_row_limit", "hit_count": len(edges), "returned_count": len(records)}] if len(edges) > max_rows else []
    return _nontext_read_result(records, "graph_relations", paper_ids, remaining=remaining)


def _reject_unauthorized_graph_targets(state: RAGState, paper_ids: list[str]) -> None:
    """
    校验图谱取证目标落在本轮授权范围内，越界即拒，不放给底层查询。

    模型给出的 target_paper_ids 是计划里的候选，不是授权凭据；图谱按论文 ID 直查、
    不带归属条件，拿范围外的 ID 去查会读到本轮授权之外的论文关系。越界与「图库无数据」
    是两回事：这里如实写明目标越界，便于与连接失败、查询超时、零边区分开；
    既不退回重规划，也不静默剔除目标，更不转成空查询冒充「查过没有」。

    Args:
        state: 当前 RAGState 字典态，授权范围取 project_scope.paper_ids
        paper_ids: 本次要查的目标论文 ID

    Returns:
        None: 目标为空或全部落在授权范围内时正常返回

    Raises:
        EvidenceCapabilitySoftError: 目标越界；辅助来源的失败只阻塞该单元、整轮继续
    """
    if not paper_ids:
        return
    project_scope = state.get("project_scope")
    scoped = project_scope if isinstance(project_scope, dict) else {}
    authorized = {str(paper_id) for paper_id in (scoped.get("paper_ids") or [])}
    unauthorized = [paper_id for paper_id in paper_ids if paper_id not in authorized]
    # 1. 【规划取证】【图谱授权】越界目标一律拒掉，失败面与原因都写清楚
    if unauthorized:
        logger.info(
            f"【规划取证】【图谱授权】目标不在本轮授权范围，拒绝查图：越界 {unauthorized}，"
            f"本轮授权 {len(authorized)} 篇"
        )
        raise EvidenceCapabilitySoftError(
            layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT,
            f"目标论文不在本轮授权范围，已拒绝图谱取证：{unauthorized}",
        )


def _graph_endpoint_names(papers: list[Any], nodes: list[Any]) -> dict[str, str]:
    """
    把图端点 id 映射成人可读名字，模型看不懂裸 id。

    论文端点（"paper:<paper_id>"）取论文标题、无标题回退 paper_id；其余端点取节点
    name、取不到回退节点 id；同一端点取先出现的名字。

    Args:
        papers: 子图返回的论文记录，含 paper_id 与 title
        nodes: 子图返回的节点记录，含 id 与 name

    Returns:
        dict[str, str]: 端点 id → 展示名映射
    """
    names: dict[str, str] = {}
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "").strip()
        if paper_id:
            key = f"{_PAPER_NODE_ID_PREFIX}{paper_id}"
            names.setdefault(key, str(paper.get("title") or "").strip() or paper_id)
    for node in nodes:
        node_id = str(node.get("id") or "").strip()
        if node_id:
            names.setdefault(node_id, str(node.get("name") or "").strip() or node_id)
    return names


def _graph_relation_record(edge: dict[str, Any], names: dict[str, str]) -> EvidenceRecord:
    """
    把一条图谱边拼成不可引用的关系三元组记录。

    证据 ID 基于边内容确定性生成，同一关系重复查到仍是同一个 ID（AC 28 要求稳定）。

    Args:
        edge: 子图返回的边，source/target 是节点 id、type 是关系类型
        names: 端点 id → 展示名映射

    Returns:
        dict[str, Any]: structure_hint 层、不可引用的关系记录；excerpt 与 summary 同文案
    """
    from app.services.internal_research.evidence_identity import make_evidence_id

    source_id = str(edge.get("source") or "").strip()
    target_id = str(edge.get("target") or "").strip()
    relation = str(edge.get("type") or "").strip()
    paper_id = _edge_paper_id(source_id, target_id)
    summary = f"{names.get(source_id, source_id)} {relation} {names.get(target_id, target_id)}".strip()
    return {
        "evidence_id": make_evidence_id(paper_id=paper_id, excerpt=summary, namespace="graph_relation"),
        "source": layers.EVIDENCE_SOURCE_NEO4J,
        "layer": layers.EVIDENCE_LAYER_STRUCTURE_HINT,
        "paper_id": paper_id,
        "summary": summary,
        "excerpt": summary,
        "usable_as_citation": False,
        "provenance": "graph_relation_query",
    }


def _edge_paper_id(source_id: str, target_id: str) -> str:
    """取边所属论文的 ID：优先起点，起点不是论文端点时再认终点，都不是返回空串。"""
    for endpoint in (source_id, target_id):
        if endpoint.startswith(_PAPER_NODE_ID_PREFIX):
            return endpoint[len(_PAPER_NODE_ID_PREFIX):]
    return ""


def _planner_reader_user_id(state: RAGState) -> str:
    """
    取本轮取证读取的用户身份。

    RAG 子图内只能从 project_scope 读：state["user_id"] 在 rag_node 桥接进子图时已被丢掉。

    Args:
        state: 当前 RAGState 字典态
    Returns:
        str: 用户 ID；取不到时为空串
    """
    project_scope = state.get("project_scope") if isinstance(state.get("project_scope"), dict) else {}
    return str(project_scope.get("user_id") or "").strip()


def _planner_scoped_store(state: RAGState, source_type: str):
    """
    取绑好本轮用户身份的向量库句柄，供按论文 ID 直读的取证适配器共用。

    这几个适配器读的论文 ID 来自模型起草的 target_paper_ids，而仓储层的归属谓词在
    user_id 为空时会整条跳过（内部审计不限定 owner）。不绑身份就等于拿模型给的 ID
    直接读库，所以缺身份时直接失败，不降级成空结果冒充「查得到但是空」。

    Args:
        state: 当前 RAGState 字典态，身份取自 project_scope
        source_type: 调用方的取证能力，失败原因里点明是哪个能力缺身份
    Returns:
        绑定了读取身份的向量库句柄
    Raises:
        RetrievalScopeError: 本轮取不到用户身份，论文归属无从校验
    """
    user_id = _planner_reader_user_id(state)
    if not user_id:
        raise RetrievalScopeError(f"{source_type} 取证缺少读取身份，无法校验论文归属")
    snapshot = _planner_scope_publication(state)
    if snapshot is None or snapshot.get("user_id") != user_id or not snapshot.get("lease_id"):
        raise RetrievalScopeError(f"{source_type} 取证缺少本轮已授权发布快照")
    # 本轮发布快照随身份一起盖到句柄上：直读与向量检索必须读同一版本，
    # 否则「先列章节、后发布新版、再读正文」会用旧版章节路径去筛新版块
    return get_vector_store().for_user(user_id).for_snapshot(snapshot)


def _planner_scope_publication(state: RAGState) -> dict[str, Any] | None:
    """
    取本轮请求固定的发布快照，供直读取证沿用同一版本。

    快照由请求入口在授权校验后固定；取不到时返回 None，正式取证调用方据此明确拒绝读取。

    Args:
        state: 当前 RAGState 字典态，快照取自 project_scope.publication
    Returns:
        dict[str, Any] | None: 本轮发布快照；未固定时返回 None
    """
    from app.services.upload_read_service import publication_from_scope

    project_scope = state.get("project_scope") if isinstance(state.get("project_scope"), dict) else {}
    return publication_from_scope(project_scope)


async def _planner_section_structure_records(state: RAGState, unit, _context) -> EvidenceReadResult:
    """参数为授权回合、目录单元及上下文；返回窄查询的章节定位材料，正文读取另行请求。"""
    records: list[EvidenceRecord] = []
    store = _planner_scoped_store(state, layers.PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE)
    # 1. 论文 ID 口径与元数据适配器一致：单元绑定优先、本轮焦点兜底
    paper_ids = _unit_or_state_paper_ids(state, unit.evidence_policy.target_paper_ids)
    for paper_id in paper_ids:
        # 2. 章节窄查询是同步 IO，放线程池执行，不阻塞事件循环
        sections = await run_source_io(store.list_section_paths, paper_id)
        # 3. 查得到但零章节是如实的空：该篇不产记录，也不用空串冒充有值
        if not sections:
            continue
        records.append(_section_structure_record(paper_id, sections))
    return _nontext_read_result(records, "section_structure", paper_ids)


def _unit_structured_query(state: RAGState, paper_ids: list[str]) -> dict[str, Any]:
    """
    把研究单元的取证目标收窄成一份派生检索范围，不改兄弟单元在用的原始范围。

    单元声明了 target_paper_ids 时，它是查询硬范围，必须在真实 top-k 之前生效：
    只改顶层 paper_ids 而 project_scope 保持项目全量，向量检索仍按全量取 top-k，
    项目里其他论文会把目标论文的相关证据挤出去，目标论文反而被判成没证据。

    收窄只往授权范围内的交集走，不放宽；取不到交集时保持原范围不动。

    Args:
        state: 当前 RAGState 字典态，原始范围取自 structured_query
        paper_ids: 本单元的目标论文 ID，空表示不限定

    Returns:
        dict[str, Any]: 收窄后的派生 structured_query；目标为空或交集为空时返回原范围
    """
    structured = state.get("structured_query") if isinstance(state.get("structured_query"), dict) else {}
    if not paper_ids:
        return structured
    project_scope = structured.get("project_scope") if isinstance(structured.get("project_scope"), dict) else {}
    authorized = [str(pid).strip() for pid in (project_scope.get("paper_ids") or []) if str(pid or "").strip()]
    targets = [str(pid).strip() for pid in paper_ids if str(pid or "").strip()]
    if not authorized:
        return structured
    # 1. 目标与授权范围求交：越界目标既不进查询条件，也不放宽授权
    narrowed = [pid for pid in targets if pid in set(authorized)]
    if not narrowed:
        return structured
    # 2. 顶层与 project_scope 都要改：底层过滤条件读的是后者，只改前者等于没改
    scoped_project = _narrowed_project_scope(project_scope, narrowed)
    scoped = {**structured, "paper_ids": list(narrowed), "project_scope": scoped_project}
    # 3. 顶层 resolved_scope 同样要收窄：业务过滤器按它取本轮允许命中的论文
    resolved = structured.get("resolved_scope")
    if isinstance(resolved, dict):
        scoped["resolved_scope"] = {**resolved, "paper_ids": list(narrowed)}
    return scoped


def _narrowed_project_scope(project_scope: dict[str, Any], paper_ids: list[str]) -> dict[str, Any]:
    """
    生成收窄论文范围后的派生项目范围，保留 owner、发布快照与年份等原有约束。

    派生对象不就地改原范围：兄弟单元共用同一份 project_scope，就地收窄会污染它们的范围。

    Args:
        project_scope: 本轮请求级项目范围合同
        paper_ids: 收窄后的目标论文 ID，必须已是授权范围的子集

    Returns:
        dict[str, Any]: 新的项目范围对象，filter_expr 与 resolved_scope 同步收窄
    """
    import json

    narrowed_scope = {**project_scope, "paper_ids": list(paper_ids)}
    resolved = project_scope.get("resolved_scope")
    if isinstance(resolved, dict):
        narrowed_scope["resolved_scope"] = {**resolved, "paper_ids": list(paper_ids)}
    # 底部过滤条件必须跟着收窄：底层取 top-k 用的是它，只改 paper_ids 等于没改
    clause = "paper_id in " + json.dumps(list(paper_ids), ensure_ascii=True)
    base = str(project_scope.get("filter_expr") or "").strip()
    narrowed_scope["filter_expr"] = f"({base}) and ({clause})" if base else clause
    return narrowed_scope


def _unit_or_state_paper_ids(state: RAGState, unit_paper_ids: list[str]) -> list[str]:
    if unit_paper_ids:
        return [str(pid).strip() for pid in unit_paper_ids if str(pid or "").strip()]
    focus = state.get("focus") if isinstance(state.get("focus"), dict) else {}
    project_scope = state.get("project_scope") if isinstance(state.get("project_scope"), dict) else {}
    return [
        str(pid).strip()
        for pid in (focus.get("turn_focus") or project_scope.get("paper_ids") or [])
        if str(pid or "").strip()
    ]


def _metadata_record(paper_id: str, paper: dict[str, Any]) -> EvidenceRecord:
    """
    把一篇论文的库记录包成规划用元数据记录，供模型在出计划前了解这篇论文。

    摘要是完整原文，不是预览：模型判「要不要读这篇」时，摘要前 500 字之外的结论
    恰恰是决定性的。已存分析按作者与分析产物顺序逐字段拼接；库里没有分析时如实
    不带该字段，不拿摘要冒充分析。

    Args:
        paper_id: 论文 ID
        paper: 论文库记录，取 title / abstract / analysis

    Returns:
        dict[str, Any]: 元数据层、不可引用的论文详情记录
    """
    title = str(paper.get("title") or paper_id).strip()
    abstract = str(paper.get("abstract") or "").strip()
    excerpt = "\n\n".join(part for part in [abstract, _analysis_text(paper.get("analysis"))] if part)
    return {
        "evidence_id": f"metadata-{paper_id}",
        "source": layers.EVIDENCE_SOURCE_INVENTORY,
        "layer": layers.EVIDENCE_LAYER_METADATA,
        "paper_id": paper_id,
        "summary": title,
        "excerpt": excerpt,
        "usable_as_citation": False,
    }


def _analysis_text(analysis: Any) -> str:
    """
    把已存论文分析拼成可读文本，按作者产出的字段顺序逐段并列。

    只做拼接与空白归一，不按字数裁剪：裁剪会静默吃掉结论，要控制长度由提示词层
    如实说明截断，不在数据层动手脚。

    Args:
        analysis: 库里的 analysis 字段，可能是 dict、字符串或空

    Returns:
        str: 形如「问题：… \n 方法：…」的分析文本；没有分析时返回空串
    """
    if isinstance(analysis, str):
        return analysis.strip()
    if not isinstance(analysis, dict):
        return ""
    parts: list[str] = []
    for key, label in _ANALYSIS_FIELD_LABELS:
        value = analysis.get(key)
        if isinstance(value, list):
            rendered = "; ".join(str(item).strip() for item in value if str(item or "").strip())
        else:
            rendered = str(value or "").strip()
        if rendered:
            parts.append(f"{label}: {rendered}")
    return "\n".join(parts)


def _section_structure_record(paper_id: str, sections: list[str]) -> EvidenceRecord:
    """
    把一篇论文的章节路径拼成一条规划用证据记录。

    Args:
        paper_id: 论文 ID
        sections: 该论文已索引的章节路径，按论文内先后顺序去重

    Returns:
        dict[str, Any]: 章节结构记录；章节名单按既有「 | 」分隔习惯拼成一行，不塞正文
    """
    return {
        "evidence_id": f"section-structure-{paper_id}",
        "source": layers.EVIDENCE_SOURCE_MILVUS,
        "layer": layers.EVIDENCE_LAYER_STRUCTURE_HINT,
        "paper_id": paper_id,
        "summary": " | ".join(sections),
        "usable_as_citation": False,
    }


def _planner_section_request(state: RAGState, unit, context) -> KnowledgeReadRequest:
    """参数为本轮授权状态、章节动作及执行账本；生成实际读取和复用共用的同版本定位。"""
    paper_ids = tuple(_unit_or_state_paper_ids(state, unit.evidence_policy.target_paper_ids))
    continuations: tuple[dict[str, Any], ...] = ()
    sections = tuple(unit.evidence_policy.required_sections)
    if unit.evidence_policy.continuation_ids:
        if sections or context is None:
            raise ValueError("续读需要本轮执行账本，且不能同时指定另一份章节定位")
        continuations = context.execution_state.resolve_continuations(unit.evidence_policy.continuation_ids, paper_ids)
        sections = tuple(dict.fromkeys(entry["locator"] for entry in continuations))
    budget = int(_settings().rag_planner_section_text_max_chars)
    return KnowledgeReadRequest(operation="sections", scope=state.get("project_scope") or {}, paper_ids=paper_ids,
        section_paths=sections, continuations=continuations, max_chars=None if budget == 0 else budget)


async def _planner_section_text_records(state: RAGState, unit, _context) -> EvidenceReadResult:
    """参数为授权回合、章节读取单元及上下文；返回权威章节正文与独立完整性报告。"""
    from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials

    result = await read_knowledge(_planner_section_request(state, unit, _context))
    records = convert_knowledge_materials(result.materials, provenance="section_path_read")
    return EvidenceReadResult(tuple(records), result.read_report)


async def _planner_chunk_metadata_records(state: RAGState, unit, _context) -> EvidenceReadResult:
    """参数为授权回合、筛块单元及上下文；返回定位元数据，未匹配/多义及条数额度独立报告。"""
    policy = unit.evidence_policy
    max_rows = int(_settings().rag_planner_chunk_metadata_max_rows)
    store = _planner_scoped_store(state, layers.PLANNER_SOURCE_LOCAL_CHUNK_METADATA)
    collected: list[tuple[str, dict[str, Any]]] = []
    remaining: list[dict[str, Any]] = []
    paper_ids = _unit_or_state_paper_ids(state, policy.target_paper_ids)
    for paper_id in paper_ids:
        section_paths: list[str] = []
        if policy.required_sections:
            paths = await run_source_io(store.list_section_paths, paper_id)
            resolutions = resolve_section_paths(paths, policy.required_sections)
            for resolution in resolutions:
                if resolution.status == "matched":
                    section_paths.extend(resolution.paths)
                else:
                    remaining.append({"paper_id": paper_id, "requested": resolution.requested,
                                      "reason": resolution.status, "candidates": list(resolution.paths)})
            if not section_paths:
                continue
        # 1. 【知识取证】【定位元数据】保持窄查询，不因目录筛选自动读取正文
        found = await run_source_io(
            store.list_chunk_rows_by_metadata, paper_id, section_paths=section_paths,
            block_types=list(policy.required_block_types), page_nums=list(policy.required_pages),
        )
        collected.extend((paper_id, row) for row in sorted(found, key=published_chunk_order))
    records = [_chunk_metadata_record(paper_id, row) for paper_id, row in collected[:max_rows]]
    remaining.extend({"paper_id": paper_id, "chunk_id": row["chunk_id"], "reason": "metadata_row_limit"}
                     for paper_id, row in collected[max_rows:])
    return _nontext_read_result(records, "chunk_metadata", paper_ids, remaining=remaining)


def _chunk_metadata_record(paper_id: str, row: dict[str, Any]) -> EvidenceRecord:
    """
    把一条块行包成不可引用的定位元数据记录，不含任何正文字段。

    本能力只回答「哪篇 / 哪节 / 第几页 / 什么类型 / 哪一块」，要正文是 local_section_text 的活。
    `section` 直接取块行的 section 列（全路径口径），summary 拼出页码、块类型与章节路径。

    Args:
        paper_id: 该块所属论文 ID，块行窄投影里不带 paper_id，由调用方传入
        row: Milvus 块行，来自 list_chunk_rows_by_metadata 的窄投影
    Returns:
        dict[str, Any]: metadata 层、不可引用的块定位记录
    """
    section = str(row.get("section") or "").strip()
    section_path = str(row.get("section_path") or "").strip()
    block_type = str(row.get("block_type") or "").strip()
    page_num = int(row.get("page_num") or 0)
    return {
        "evidence_id": f"chunk-metadata-{str(row.get('chunk_id') or '').strip()}",
        "source": layers.EVIDENCE_SOURCE_MILVUS,
        "layer": layers.EVIDENCE_LAYER_METADATA,
        "paper_id": paper_id,
        "chunk_id": str(row.get("chunk_id") or "").strip(),
        "parent_id": str(row.get("parent_id") or "").strip(),
        "section": section,
        "page_num": page_num,
        "summary": f"第 {page_num} 页 · {block_type} · {section_path or section}",
        "usable_as_citation": False,
        "provenance": "chunk_metadata_filter",
    }


async def _planner_arxiv_records(state: RAGState, unit, _context) -> EvidenceReadResult:
    """参数为回合、外部查询单元及上下文；复用检索和过滤能力，返回非引用外部材料与读取说明。"""
    from app.nodes.arxiv_search import arxiv_filter_node, arxiv_node

    # 1. 计划没给检索词到不了这里（executor 侧已先记 blocked），到这就是装配缺陷，明说
    queries = [str(q).strip() for q in unit.evidence_policy.locator_queries if str(q or "").strip()]
    if not queries:
        raise EvidenceCapabilityError(
            layers.PLANNER_SOURCE_EXTERNAL_CONTEXT, "external_context_needed 缺少 locator_queries",
        )
    # 2. 复用能力模块两步节点：搜索读 user_query，过滤读 papers + user_query
    arxiv_state = {"user_query": queries[0]}
    search_update = await arxiv_node(arxiv_state)
    arxiv_state.update(search_update)
    filter_update = await arxiv_filter_node(arxiv_state)
    papers = list(filter_update.get("filtered_papers") or [])
    return _nontext_read_result([_arxiv_record(paper) for paper in papers], "arxiv", queries)


def _arxiv_record(paper: dict[str, Any]) -> EvidenceRecord:
    """
    把一篇 arXiv 论文包成不可引用的元数据证据记录。

    摘要只取前 200 字进 summary，控制进上下文的长度；evidence_id 用 arXiv 命名空间
    确定性生成，同一篇论文重复查到仍是同一个 ID。

    Args:
        paper: arXiv 能力返回的论文 dict（paper_id / title / authors / abstract 等）

    Returns:
        dict[str, Any]: metadata 层、不可引用的外部论文记录
    """
    from app.services.internal_research.evidence_identity import make_evidence_id

    paper_id = str(paper.get("paper_id") or "").strip()
    title = str(paper.get("title") or "").strip()
    authors = [str(name).strip() for name in (paper.get("authors") or []) if str(name or "").strip()]
    abstract = str(paper.get("abstract") or "").strip()
    summary = f"{title}（作者：{', '.join(authors) or '未知'}）{abstract[:200]}"
    return {
        "evidence_id": make_evidence_id(paper_id=paper_id, excerpt=summary, namespace="arxiv"),
        "source": layers.EVIDENCE_SOURCE_ARXIV,
        "layer": layers.EVIDENCE_LAYER_METADATA,
        "paper_id": paper_id,
        "summary": summary,
        "excerpt": summary,
        "usable_as_citation": False,
        "provenance": "arxiv_search",
    }


async def write_back_node(state: RAGState) -> dict[str, Any]:
    """把最终结果与 trace 写回持久层/观测层。"""
    return await rag_evaluate.run_write_back(
        state,
        paper_db=paper_db,
        settings=_settings(),
        observability_module=observability,
        with_trace_fn=_with_trace,
        update_trace_generation_fn=_update_trace_generation,
    )


def _graph_recursion_limit() -> int:
    """
    读取规划子图自身的递归上限。

    它挂在子图的 compile 出口上（不是每次 astream 的调用 config），理由：
    上限是这张图的固有属性，任何入口跑这份制品都受同一道护栏——三条入口
    （agent.py 主链子图调用、eval_pipeline、rag_adapter）不必各自记得传 config。

    参数:
        无
    返回:
        子图递归上限（正整数）；配置缺字段或取值非法时回落到默认值
    """
    raw = getattr(_settings(), "rag_planner_graph_recursion_limit", _DEFAULT_GRAPH_RECURSION_LIMIT)
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_GRAPH_RECURSION_LIMIT
    return limit if limit >= 1 else _DEFAULT_GRAPH_RECURSION_LIMIT


def create_rag_node():
    """
    构造并 compile 研究型子图。

    唯一运行链（planner-first，无开关分叉）：
      rag_init -> planner_analyze -> planner_validate
      -> planner_execute -> planner_synthesize -> write_back

    判为 minimal 的回合不需要论文事实，analyze 之后直接进 planner_synthesize
    收口，跳过计划校验与检索。这是同一条链上的短路，不是第二套运行模型：
    节点集合与运行模型都只有这一份。

    意图与焦点解析已并入 planner_analyze 的首次分析，Planner 之前不再单独判一次。

    这张图在主图里作为一个"黑盒能力节点"被调用。

    Returns:
        CompiledStateGraph: 已 compile 的 RAG 子图，可被主图直接调用
    """
    workflow = StateGraph(RAGState)
    # 1. 先注册唯一运行链上的全部节点
    workflow.add_node("rag_init", rag_init_node)
    workflow.add_node("planner_analyze", planner_analyze_node)
    workflow.add_node("planner_validate", planner_validate_node)
    workflow.add_node("planner_execute", planner_execute_node)
    workflow.add_node("planner_synthesize", planner_synthesize_node)
    workflow.add_node("write_back", write_back_node)
    # 2. 再定义从 rag_init 直达 planner 的唯一主路径
    workflow.add_edge(START, "rag_init")
    workflow.add_edge("rag_init", "planner_analyze")
    workflow.add_conditional_edges(
        "planner_analyze",
        route_after_planner_analyze,
        # 三条短路边：探测回合回 planner_analyze 自己（这一轮只取了上下文，计划要带着
        # 结果重出），minimal 直答进 synthesize（没计划可校验可执行）
        {
            "planner_validate": "planner_validate",
            "planner_synthesize": "planner_synthesize",
            "planner_analyze": "planner_analyze",
            "write_back": "write_back",
        },
    )
    workflow.add_conditional_edges(
        "planner_validate",
        route_after_planner_validate,
        # replan 回边：校验失败且有额度时回 analyze 重规划（design §5.1）
        {
            "planner_execute": "planner_execute",
            "planner_analyze": "planner_analyze",
            "planner_synthesize": "planner_synthesize",
            "write_back": "write_back",
        },
    )
    workflow.add_conditional_edges(
        "planner_execute",
        route_after_planner_execute,
        # replan 回边回 analyze，repair 自环回 execute 补检索（design §5.2）
        {
            "planner_synthesize": "planner_synthesize",
            "planner_analyze": "planner_analyze",
            "planner_execute": "planner_execute",
            "write_back": "write_back",
        },
    )
    workflow.add_conditional_edges(
        "planner_synthesize",
        route_after_planner_synthesize,
        {"write_back": "write_back", "planner_analyze": "planner_analyze", "planner_synthesize": "planner_synthesize"},
    )
    workflow.add_edge("write_back", END)
    # 3. 把图自身的递归上限显式挂在 compile 出口：探测回边（planner_analyze 自环）在取消
    # 计数额度后唯一剩下的收敛保证就是它，否则超限抛的是框架错误、用户与排查者都想不到
    return workflow.compile().with_config({"recursion_limit": _graph_recursion_limit()})
