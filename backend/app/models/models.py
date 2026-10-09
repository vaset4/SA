"""
数据模型定义模块

定义整个系统中所有节点、子图、服务之间传递的数据结构。
所有模型使用 TypedDict，与 LangGraph State 机制完全兼容。

阅读这个文件时，最重要的是区分三层概念：
1. 业务实体：Paper / Chunk / PaperAnalysis 这类“系统里有什么数据”
2. 图状态：AgentState / RAGState 这类“某条工作流现在跑到哪”
3. 前端会话模型：Conversation / Message 这类“UI 要怎么展示历史记录”

这些类型名称都叫“模型”或“状态”，但它们不在同一抽象层。
不要把 LangGraph 的运行态 state 和 PostgreSQL 中的业务/会话记录混为一谈。

模型分组：
  1. 基础业务模型 - PaperAnalysis / Paper / Chunk / RAGResult
  2. 主图 State   - AgentState（扩展了 intent / papers / rag_result）
  3. 子图 State   - ArxivState / PaperAnalyzerState / RAGState / ComparatorState / CitationState
"""

# 导入 TypedDict，用于定义字典形态的静态类型。
# 导入 List/Dict/Any/Optional，描述字段结构。
# 导入 Annotated，用于给 LangGraph 的消息聚合器附加元信息。
import operator
from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict
# 导入 add_messages 聚合器，让 messages 字段支持 LangGraph 追加合并。
from langgraph.graph.message import add_messages

from app.models.research_contracts import (
    AnswerBrief,
    CapabilityStatus,
    DepthTransition,
    EvidenceRecord,
    EvidenceSelectionStats,
    FocusResolution,
    MemoryWritebackResult,
    MissingEvidence,
    PaperFocusContract,
    ProjectResearchTurnContext,
    ProjectScopeContract,
    ResearchTurnOutcome,
    SelectedEvidence,
    SynthesisState,
    TurnDepth,
    TurnDepthAssessment,
)
from app.models.research_memory_contracts import MemoryReadResult
from app.models.knowledge_access_contracts import SourceFragment


# ============================================================
# 1. 基础业务模型
# ============================================================

class PaperAnalysis(TypedDict):
    """
    论文结构化分析结果。

    由 paper_analyzer 子图生成，对论文摘要或全文做语义提炼。

    懒加载策略：
      - 批量搜索时 analysis 字段为 None，不触发此结构的生成
      - 只有用户明确选中某篇论文，或将论文加入知识库时，才调用 LLM 生成此结构
      - 这样可避免对10篇搜索结果全部调用 LLM，节省成本和时间

    Attributes:
        problem:      该论文要解决的核心问题描述
        method:       核心方法或模型的名称与简述
        contribution: 主要贡献列表，通常2-4条，每条一句话
        limitation:   论文明确指出或隐含的局限性
        keywords:     关键词列表，用于兴趣匹配和推荐打分
        related_work: 论文中提及的相关工作名称列表，用于引用链分析
        tldr:         1-2 句话的快速总结
        methodology_type: 方法论类型（experimental/theoretical/survey/hybrid）
        methodology_detail: 方法论细节描述
    """
    # 论文要解决的问题。
    problem: str
    # 论文核心方法或模型。
    method: str
    # 论文主要贡献点列表。
    contribution: List[str]
    # 论文局限性说明。
    limitation: str
    # 用于推荐与检索增强的关键词。
    keywords: List[str]
    # 论文相关工作名称列表。
    related_work: List[str]
    # 1-2 句话的快速总结。
    tldr: str
    # 方法论类型。
    methodology_type: str
    # 方法论细节。
    methodology_detail: str


class Paper(TypedDict):
    """
    论文主体数据模型，统一表示来自不同来源的论文。

    来源类型与 paper_id 规则：
      - arXiv 来源：paper_id = 规范化的 arXiv ID，如 "2301.00001"
      - 用户上传：paper_id = "upload_{uuid4}"

    去重策略（在 paper_db 层执行，此模型本身不负责去重）：
      1. 优先用 arxiv_id 匹配
      2. 其次用 doi 匹配
      3. 都没有则允许写入（上传的无 ID 论文）

    这个模型会同时出现在两层：
    - 业务库层：paper_db 负责把它落到 papers 表
    - 图状态层：arxiv / compare / citation / related_work 等节点会直接在 state 中传递它

    Attributes:
        library_id:     所属知识库 ID，未显式指定时归入默认库
        paper_id:       系统内唯一标识符（主键）
        arxiv_id:       规范化 arXiv ID，不含 "arxiv:" 前缀
                        例："2301.00001"，无法提取时为 None
        doi:            DOI 字符串，用于跨来源去重，无则为 None
        title:          论文标题
        authors:        作者姓名列表
        abstract:       摘要原文
        pdf_url:        PDF 直链（arXiv 提供的官方下载链接）
        pdf_path:       本地缓存路径，PDF 下载后填充，否则为 None
        categories:     arXiv 分类标签列表，例：["cs.AI", "cs.LG"]
        published_date: 首次发表日期，格式 "YYYY-MM-DD"
        source:         数据来源，固定值 "arxiv" 或 "upload"
        analysis:       结构化分析，懒加载，搜索时默认为 None
        tags:           用户自定义标签列表，初始为空列表
    """
    # 所属知识库 ID。
    library_id: str
    # 系统内部的唯一主键。
    paper_id: str
    # arXiv 论文 ID，可为空。
    arxiv_id: Optional[str]
    # DOI，可为空。
    doi: Optional[str]
    # 论文标题。
    title: str
    # 作者列表。
    authors: List[str]
    # 摘要原文。
    abstract: str
    # PDF 下载地址。
    pdf_url: str
    # 本地缓存 PDF 路径，可为空。
    pdf_path: Optional[str]
    # 学科分类标签列表。
    categories: List[str]
    # 发表日期字符串。
    published_date: str
    # 数据来源标记。
    source: str
    # 结构化分析结果，可为空。
    analysis: Optional[PaperAnalysis]
    # 用户标签列表。
    tags: List[str]
    user_id: NotRequired[str]
    is_vectorized: NotRequired[bool]
    index_batch_id: NotRequired[str | None]
    metadata_status: NotRequired[str]
    metadata_error: NotRequired[str]
    metadata_missing: NotRequired[List[str]]


class Chunk(TypedDict):
    """
    PDF 全文分块后的文本片段，采用 Parent-Child 两级分块策略。

    chunk_id 命名规则：{paper_id}_{parent_idx}_{child_idx}
    parent_id 命名规则：{paper_id}_{parent_idx}

    这里的 chunk 是“知识检索层”的主数据单位，不会直接进入前端会话库。
    paper_id 把它和业务库中的论文元数据关联起来，parent_id 则把它和父块全文关联起来。

    Attributes:
        chunk_id:  子块唯一标识符（用于向量化和检索定位）
        parent_id: 父块标识符（同父块的子块共享，用于 read_chunk 取全文）
        text:      子块文本，约 400-600 字符，存入 FAISS 向量索引
        full_text: 父块完整文本，约 2000-4000 字符，read_chunk 时返回
        page_num:  所在页码，从 1 开始
        section:   所属章节标题，未能识别时为 "unknown"
    """
    chunk_id: str
    parent_id: str
    text: str
    full_text: str
    page_num: int
    section: str
    # v2 新增字段
    embedding_text: NotRequired[str]
    display_text: NotRequired[str]
    block_type: NotRequired[str]
    section_path: NotRequired[str]
    asset_ref: NotRequired[str]
    parser_mode: NotRequired[str]
    chunk_token_count: NotRequired[int]
    full_text_token_count: NotRequired[int]
    chunk_warnings: NotRequired[List[str]]
    # 新内容格式保存子块与父块各自的来源映射，不能互相替代
    source_fragments: NotRequired[List[SourceFragment]]
    parent_source_fragments: NotRequired[List[SourceFragment]]


class RAGResult(TypedDict):
    """
    RAG 检索结果，汇总多个命中的文本块及其来源信息。

    由 rag_node 的检索阶段生成并累积更新。
    is_sufficient 由 react_decision_node 的 LLM 判断后填充。

    Attributes:
        chunks:        命中的文本块内容列表（已排序，最相关在前）
        sources:       各 chunk 对应的 paper_id 列表，与 chunks 等长
        scores:        各 chunk 的相似度分数列表，0-1，越高越相关
        is_sufficient: True 表示当前信息足够回答用户问题，False 表示需继续检索
    """
    # 命中的文本片段或 preview 列表。
    chunks: List[str]
    # 每个片段对应的论文 ID。
    sources: List[str]
    # 每个片段对应的分数。
    scores: List[float]
    # 当前证据是否足够回答。
    is_sufficient: bool


# ============================================================
# 2. 主图 State（扩展）
# ============================================================

class AgentState(TypedDict):
    """
    主图的全局状态，在所有顶层节点之间传递。

    LangGraph 的节点只需返回需要更新的字段，框架自动合并到全局 State，
    因此节点函数可以只返回 {"intent": "arxiv_search"} 而不需要返回全部字段。

    当前主图里最常见的字段分组：
      - 路由字段：`intent / core_intent`
      - deep_research 字段：`research_brief / research_notes / raw_notes / report`
      - arXiv 路径字段：`papers / filtered_papers`
      - RAG / chat 字段：`rag_result / chat_history / context_summary`
      - 线程字段：`session_id`

    AgentState 是“顶层主图的共享上下文”。
    几乎所有 HTTP / WebSocket 请求最后都会先被整理成它，再交给主图 router 分发。

    其中最容易混淆的字段有：
    - session_id：用于映射到 LangGraph thread_id，是运行线程 ID，不是前端 conversation_id
    - chat_history：图内短期对话上下文，不等于 PostgreSQL 会话库中的 messages 表
    - papers / filtered_papers / rag_result：这些是某些功能路径的中间结果，不代表都会持久化

    Attributes:
        user_query:       用户的原始输入文本
        intent:           实际执行路由意图，可选值：
                          "arxiv_search" / "rag_qa" / "deep_research" /
                          "compare" / "citation" / "literature_review" /
                          "related_work" / "free_chat"
        core_intent:      产品层主语义，可选值：
                          "paper_analysis" / "internal_research" /
                          "external_research" / "graph_query"
        structured_query: 结构化查询补充信息，当前已被 compare / citation /
                          cross-paper QA 等路径使用
        report:           主图多数“报告型”路径的统一文本输出
        research_brief:   deep research 新内核生成的研究简报
        research_notes:   deep research 并发 researcher 汇总后的压缩笔记
        raw_notes:        deep research 保留的原始检索笔记
        papers:           arXiv 原始搜索结果，过滤前的完整列表
        filtered_papers:  LLM 过滤后的相关论文列表，供 reporter 使用
        rag_result:       rag_qa 路径的检索汇总结果
        guardrails_action: 顶层 guardrails 的放行/拒绝动作，供条件边短路使用
        guardrails_reason: 顶层 guardrails 的判定原因
        source_status:    guardrails 短路时记录各检索源 skipped 状态
        runtime_chain:    RAG 实际运行链路，现在只有 planner 一条
        evidence_records: RAG / planner 返回的统一证据记录
        missing_evidence: RAG / planner 返回的缺失证据说明
        answer_brief:     RAG / planner 返回的回答简报
    """
    # 用户原始问题。
    user_query: str
    # 路由意图。
    intent: Optional[str]
    # 产品层主语义。
    core_intent: Optional[str]
    # 结构化查询补充对象，可承载 paper_ids / paper_id 等辅助信息。
    structured_query: Optional[Dict[str, Any]]
    # 当前对话模式，区分普通对话、查证和研究。
    dialogue_mode: Optional[str]
    # 当前项目范围合同。
    project_scope: Optional[ProjectScopeContract]
    # 当前问题焦点合同。
    paper_focus: Optional[PaperFocusContract]
    # 当前研究整理状态。
    synthesis_state: Optional[SynthesisState]
    # Research Planner additive payloads; serialized dict/list only.
    research_plan: Optional[Dict[str, Any]]
    research_unit_results: List[Dict[str, Any]]
    # 【研究执行】【成果接续】服务端运行身份与成对保存的读取凭据，只在同一次运行接续
    run_id: str
    planner_read_receipts: List[Dict[str, Any]]
    planner_execution_history: List[Dict[str, Any]]
    planner_locator_facts: List[Dict[str, Any]]
    planner_draft: Optional[Dict[str, Any]]
    planner_budget: Dict[str, Any]
    planner_probe_results: List[Dict[str, Any]]
    planner_observation: Dict[str, Any]
    planner_loop_decision: str
    planner_recovery_trace: List[Dict[str, Any]]
    coverage_review: Optional[Dict[str, Any]]
    planner_trace: Optional[Dict[str, Any]]
    planner_final_blocks: List[Dict[str, Any]]
    planner_final_candidate: Dict[str, Any]
    planner_final_validation_errors: List[Dict[str, Any]]
    # 【研究交付】【同次结果】正文、引用坐标及交付报告跨图保持同一尝试身份
    final_delivery_report: Optional[Dict[str, Any]]
    final_citation_bindings: List[Dict[str, Any]]
    final_generation_attempt: Optional[Dict[str, Any]]
    final_delivery_applicability: Optional[str]
    evidence_sufficiency_report: Optional[Dict[str, Any]]
    repair_query_plans: List[Dict[str, Any]]
    planner_repair_trace: Optional[Dict[str, Any]]
    planner_repair_merge_report: Optional[Dict[str, Any]]
    final_answer_eligibility_report: Optional[Dict[str, Any]]
    # 规划结构化/硬失败的稳定错误码与用户摘要；不声明会被主图入口静默丢弃
    error_code: Optional[str]
    error_summary: Optional[str]
    # 项目科研回合的服务端只读输入（授权论文范围、会话身份、归一化问题）。
    project_research_context: Optional[ProjectResearchTurnContext]
    # Planner 首次分析的深度判断：深度、原因码、Evidence 要求与归一化问题。
    turn_depth_assessment: Optional[TurnDepthAssessment]
    # 本回合最终生效的处理深度，决定动作预算与证据要求强度。
    turn_depth: Optional[TurnDepth]
    # 按深度判断里的记忆读取决定翻项目记忆后的结果：读到哪些判断、排除了多少、失败在哪。
    # 由 planner 分析步产出、同图内消费；不声明会在主图入口被静默丢弃成 None。
    memory_read_result: Optional[MemoryReadResult]
    # 深度变更轨迹，循环内每次改深度追加一条；无 reducer，节点须读旧值再整体回写，
    # 否则子图与主图各持一份会重复累加。
    depth_transitions: List[DepthTransition]
    # 回合终态：深度、四态能力结果、Evidence 与缺口、循环步数与终止原因。
    turn_outcome: Optional[ResearchTurnOutcome]
    # 候选记忆写回终态；与 turn_outcome 一样由统一终结节点产出，两处 State 都要声明才桥接得回主图。
    memory_writeback: Optional[MemoryWritebackResult]
    # RAG 实际运行链路，必须声明在主图 State 中，否则子图返回值会被 LangGraph 丢弃。
    runtime_chain: Optional[str]
    # Planner 规划前预装载的论文清单（ID + 标题 + 索引状态三态）
    _paper_list_lines: List[str]
    # Planner 规划前预装载的图谱状态（有数据/无数据/查询失败）
    _graph_status_lines: List[str]
    # Planner 规划前预装载的章节结构（每篇论文已索引的章节路径清单）
    _section_structure_lines: List[str]
    # RAG 子图返回的统一证据记录，必须声明在主图 State 中才能进入 API metadata。
    evidence_records: List[EvidenceRecord]
    # RAG 子图返回的缺失证据说明。
    missing_evidence: List[MissingEvidence]
    # RAG 子图返回的回答简报。
    answer_brief: AnswerBrief
    # 主图统一文本输出，供 research/compare/related_work 等路径复用。
    report: Optional[str]
    # 【研究交付】【同次正文】保留 RAG 桥接的原正文，供最终报告、引用和保存共同使用
    answer: Optional[str]
    # 当前研究运行模式。
    research_mode: Optional[str]
    # deep research 新内核生成的研究简报。
    research_brief: Optional[str]
    # deep research 汇总后的压缩研究笔记。
    research_notes: List[str]
    # deep research 保留的原始检索笔记。
    raw_notes: List[str]
    # 本轮研究选中的本地论文快照。
    selected_papers: List[Dict[str, Any]]
    # 供研究台消费的论文对比矩阵。
    comparison_matrix: List[Dict[str, Any]]
    # 结构化证据卡片。
    evidence_cards: List[Dict[str, Any]]
    # 来源调用记录。
    source_records: List[Dict[str, Any]]
    # 推理摘要。
    reasoning_summaries: List[Dict[str, Any]]
    # 关键发现。
    key_findings: List[str]
    # 未决不确定性。
    uncertainties: List[str]
    # 后续 idea 候选。
    idea_candidates: List[str]
    # 外部证据仍未覆盖的 gap。
    external_gaps: List[str]
    # 继续深挖时建议优先追问的问题。
    follow_up_questions: List[str]
    # 建议回流知识库的候选论文。
    candidate_papers_to_import: List[Dict[str, Any]]
    # arXiv 原始论文结果。
    papers: List[Paper]
    # 过滤后的相关论文结果。
    filtered_papers: List[Paper]
    # RAG 检索结果。
    rag_result: Optional[RAGResult]
    # 顶层 guardrails 动作：accept 继续 router，reject 直接结束。
    guardrails_action: Optional[str]
    # 顶层 guardrails 判定原因，供调试与观测使用。
    guardrails_reason: Optional[str]
    # guardrails 短路或 RAG 路径返回的检索源状态。
    source_status: Dict[str, str]
    source_status_reasons: Dict[str, str]
    # 图内短期对话历史。
    chat_history: List[dict]
    # 运行线程级 session ID。
    session_id: Optional[str]
    # 前端展示层 conversation ID。
    conversation_id: Optional[str]
    # LangGraph checkpoint 使用的 thread ID。
    thread_id: Optional[str]
    # 历史上下文压缩摘要。
    context_summary: Optional[str]


# ============================================================
# 3. 子图 State
# ============================================================

class ArxivState(TypedDict):
    """
    ArXiv 搜索子图专属状态。

    Attributes:
        user_query: 用户的原始查询文本（输入）
        papers:     搜索返回的论文列表（输出），所有 Paper 的 analysis 字段均为 None
    """
    # 用户原始查询。
    user_query: str
    # arXiv 搜索结果。
    papers: List[Paper]


class PaperAnalyzerState(TypedDict):
    """
    PaperAnalyzer 子图专属状态。

    懒加载：只在需要深度分析时构建此 State 并调用子图，
    不在批量搜索流程中自动触发。

    Attributes:
        paper:     待分析的论文（输入），至少需要 abstract 字段有内容
        full_text: 论文全文字符串（可选输入），有则截取前3000字提升分析质量
        analysis:  生成的结构化分析结果（输出），初始为 None
        force_analysis: 是否忽略已有 analysis 并重新生成（可选）
    """
    # 待分析论文对象。
    paper: Paper
    # 可选全文。
    full_text: Optional[str]
    # 输出分析结果。
    analysis: Optional[PaperAnalysis]
    # 显式重跑时忽略已有 analysis。
    force_analysis: NotRequired[bool]


class RAGState(TypedDict, total=False):
    """
    RAG 子图专属状态，支持 Agentic 多轮检索循环。

    total=False 使节点可以只返回部分字段。

    RAGState 与 AgentState 的关系是：
    - AgentState 是主图顶层状态
    - RAGState 是 rag 子图内部的局部状态
    - 主图只在进入 rag_qa 路径时构造它，并在结束后把少量结果写回 AgentState

    这里的 `messages` 字段使用 `add_messages` 聚合器，是为了让 LangGraph 工具调用链
    能像对话消息一样逐步追加，而不是每个节点都手动拼接完整列表。

    当前 RAGState 承载三类信息：
      - 检索循环态：`iteration / should_continue / rewritten_query / retrieval_keys`
      - 答案收敛态：`evaluation / self_rag_action / self_rag_iteration / citations`
      - 运行时辅助态：`messages / context_summary / trace` 等

    注意：
      - `next_search_source` 属于早期设计遗留字段，当前主链主要依赖
        `react_decision` + 工具调用，而不是这个字段做分派
      - 运行时还可能临时携带 `retrieval_mode / graph_mode / hallucination_* / trace`
        等字段；这些字段不一定由所有节点读写，但会在子图内部流动

    Attributes:
        user_query:          用户原始问题
        rag_result:          当前累积的检索结果
        iteration:           当前迭代轮次
        should_continue:     是否继续检索
        next_search_source:  早期遗留的下一步检索来源标记，当前默认主链较少直接依赖
        rewritten_query:     LLM 改写后的检索查询
        rewritten_queries:   查询改写后的候选查询列表
        filter_expr:         检索过滤表达式
        failure_reason:      上一轮失败原因
        context_summary:     压缩后的上下文摘要
        already_read:        已读 chunk_id 集合
        retrieval_keys:      已执行查询 key 集合
        search_results:      最近一轮检索结果
        grading_action:      文档评分后的动作
        evaluation:          回答质量评估结果
        self_rag_action:     Self-RAG 控制流动作
        self_rag_iteration:  Self-RAG 精炼轮次
        answer_confidence:   回答置信度
        citations:           最终回答引用
        answer:              最终回答文本
        paper_sources:       最终回答关联到的论文来源列表
        inventory_contexts:  项目论文清单工具返回的稳定元数据上下文
        inventory_source_ids:项目论文清单工具返回的论文 ID 列表
        chunk_citations:     当前回答关联到的 chunk 级证据列表
        graph_evidence:      当前回答关联到的图谱/结构证据列表
        graph_context:       最近一轮图谱聚合上下文
        source_status:       最近一轮各检索源执行状态
        runtime_chain:       本次 RAG 实际运行链路，现在只有 planner 一条
    """
    # 用户原始问题。
    user_query: str
    # 当前累积的 RAG 结构化结果。
    rag_result: RAGResult
    # 当前迭代轮次。
    iteration: int
    # 是否继续检索循环。
    should_continue: bool
    # 早期遗留的下一步检索来源标记。
    next_search_source: str
    # LLM 改写后的查询词。
    rewritten_query: str
    # 查询改写后的候选查询列表。
    rewritten_queries: List[str]
    # 检索过滤表达式。
    filter_expr: Optional[str]
    # 请求级结构化范围，供作用域过滤、工具注入和研究契约使用。
    structured_query: Optional[Dict[str, Any]]
    # 当前对话模式。
    dialogue_mode: str
    # 当前项目范围合同。
    project_scope: ProjectScopeContract
    # 当前论文焦点合同。
    paper_focus: PaperFocusContract
    # 当前研究整理状态。
    synthesis_state: SynthesisState
    # 当前图会话 ID。
    session_id: str
    # 当前前端会话 ID。
    conversation_id: str
    # 当前对话历史，主要用于主图桥接。
    chat_history: List[Dict[str, Any]]
    # 上一轮失败原因。
    failure_reason: str
    # 压缩后的摘要。
    context_summary: str
    # 已读 chunk 集合。
    already_read: set
    # 已检索 key 集合。
    retrieval_keys: set
    # 最近一轮检索结果。
    search_results: List[Dict[str, Any]]
    # 文档评分后的动作，如 proceed / retry_search。
    grading_action: str
    # 回答质量评估结果。
    evaluation: Optional[Dict[str, Any]]
    # Self-RAG 动作。
    self_rag_action: str
    # Self-RAG 精炼轮次。
    self_rag_iteration: int
    # 回答置信度。
    answer_confidence: Optional[str]
    # 最终引用列表。
    citations: List[Dict[str, Any]]
    # LangGraph 聚合消息流，供 ToolNode/ReAct 链逐步追加。
    messages: Annotated[List, add_messages]
    # 最终回答文本。
    answer: str
    # 当前回答关联到的论文来源列表。
    paper_sources: List[Dict[str, Any]]
    # 项目论文清单工具返回的稳定元数据上下文。
    inventory_contexts: List[str]
    # 项目论文清单工具返回的论文 ID 列表。
    inventory_source_ids: List[str]
    # 当前回答关联到的 chunk 级证据列表。
    chunk_citations: List[Dict[str, Any]]
    # 当前回答关联到的图谱/结构证据列表。
    graph_evidence: List[Dict[str, Any]]
    # 最近一轮图谱聚合上下文。
    graph_context: str
    # 最近一轮各检索源执行状态。
    source_status: Dict[str, str]
    # 本次 RAG 实际运行链路（planner / legacy_react）。
    runtime_chain: str
    # 最近一轮最终生效的检索模式。
    retrieval_mode: str
    # 最近一轮检索模式决策原因。
    retrieval_reasoning: str
    # 最近一轮最终生效的 graph mode。
    graph_mode: str
    # 统一证据记录列表。
    evidence_records: List[EvidenceRecord]
    # 缺失证据记录。
    missing_evidence: List[MissingEvidence]
    # 回答简报。
    answer_brief: AnswerBrief
    # Planner analyzer 输出草案，仅在 RAG 子图内部跨 planner 阶段传递。
    planner_draft: Dict[str, Any]
    # Research Planner additive payloads; serialized dict/list only.
    research_plan: Dict[str, Any]
    research_unit_results: List[Dict[str, Any]]
    # 【研究执行】【成果接续】服务端运行身份与成对保存的读取凭据，只在同一次运行接续
    run_id: str
    planner_read_receipts: List[Dict[str, Any]]
    planner_execution_history: List[Dict[str, Any]]
    planner_locator_facts: List[Dict[str, Any]]
    coverage_review: Dict[str, Any]
    planner_trace: Dict[str, Any]
    planner_final_blocks: List[Dict[str, Any]]
    planner_final_candidate: Dict[str, Any]
    planner_final_validation_errors: List[Dict[str, Any]]
    # 【研究交付】【同次结果】正文、引用坐标及交付报告跨图保持同一尝试身份
    final_delivery_report: Optional[Dict[str, Any]]
    final_citation_bindings: List[Dict[str, Any]]
    final_generation_attempt: Optional[Dict[str, Any]]
    final_delivery_applicability: Optional[str]
    evidence_sufficiency_report: Dict[str, Any]
    repair_query_plans: List[Dict[str, Any]]
    planner_repair_trace: Dict[str, Any]
    planner_repair_merge_report: Dict[str, Any]
    final_answer_eligibility_report: Dict[str, Any]
    # Planner ReAct 循环的步数与动作计数快照；跨 analyze/validate/execute 回边流转
    # 只作观察记录（留痕与进度展示），无任何研究额度，不参与继续/停止判断
    # 无 reducer 的 Dict 键，计数只能靠节点显式「读旧值→advance→整体回写」推进
    # 漏回写则本回合步数统计失真（spec langgraph-state-discipline Rule 1）
    planner_budget: Dict[str, Any]
    # 上一轮阶段归一化观察（校验 issues / 充分性缺口 / attempted_queries / scoped 范围）
    # 兼作 replan 时回灌 analyzer 的反馈载体与诚实收尾载荷的取数来源
    planner_observation: Dict[str, Any]
    # 全循环留痕：每步 stage/action/observation 摘要，进 planner_trace 供追溯
    planner_recovery_trace: List[Dict[str, Any]]
    # 决策器上一步产出的动作枚举；路由函数只读此字段选边，不再靠 source_status 判活
    planner_loop_decision: str
    # 规划结构化/硬失败的稳定错误码与用户摘要；不声明会在子图入口被静默丢弃
    error_code: str
    error_summary: str
    # 项目科研回合的服务端只读输入，由主图桥接进子图供 Planner 首次分析取数。
    project_research_context: ProjectResearchTurnContext
    # Planner 首次分析的深度判断，含选择原因码与本回合归一化问题。
    turn_depth_assessment: TurnDepthAssessment
    # Planner 判定并在循环内可变更的处理深度。
    turn_depth: TurnDepth
    # 按深度判断里的记忆读取决定翻项目记忆后的结果，供后续分析步与生成取用；
    # 主图与子图两处都要声明，漏一处则子图返回值在图入口被静默丢弃。
    memory_read_result: MemoryReadResult
    # 深度变更轨迹；无 reducer，节点须读旧值再整体回写，漏回写则轨迹丢步。
    depth_transitions: List[DepthTransition]
    # 回合终态，由统一终结节点产出后桥接回主图。
    turn_outcome: ResearchTurnOutcome
    # 候选记忆写回终态，同由统一终结节点产出：skipped/empty/failed 各自独立，进 trace 与 done payload。
    memory_writeback: MemoryWritebackResult
    # 本轮各取证能力的四态结果（source_type 值 → CapabilityStatus），由 execute 阶段产出、
    # 终结节点并入 turn_outcome.capability_results。只声明在 RAGState：读它的人在子图内，
    # 出子图由 turn_outcome 承担（已在主图桥接白名单里）。
    planner_evidence_capabilities: Dict[str, CapabilityStatus]
    # 出计划前探测取回的补充上下文（探测申请 → 四态结果 + 发现行），由 analyze 阶段累积；
    # 只声明在 RAGState：读它的人是子图内的 analyze 与草案提示词渲染，不出子图
    planner_probe_results: List[Dict[str, Any]]
    # Planner 规划前预装载的论文清单（ID + 标题 + 索引状态三态）
    _paper_list_lines: List[str]
    # Planner 规划前预装载的图谱状态（有数据/无数据/查询失败）
    _graph_status_lines: List[str]
    # Planner 规划前预装载的章节结构（每篇论文已索引的章节路径清单）
    _section_structure_lines: List[str]
    # 第二轮 Phase B：最终生成可消费的系统入选证据。
    selected_evidence: List[SelectedEvidence]
    evidence_status: str
    evidence_selection_stats: EvidenceSelectionStats
    retrieval_channels: List[Dict[str, Any]]
    retrieval_effective_query: str
    retrieval_effective_paper_ids: List[str]
    retrieval_effective_sections: List[str]
    retrieval_used_route_hints: List[str]
    error_kind: str
    # 二阶段：意图识别结果（主链唯一一处意图判断，只由 turn_context_resolver 产出）。
    intent: str
    # 二阶段：焦点解析结果。
    focus: "FocusResolution"
    # 二阶段：取证闭环重试计数。
    retry_count: int
    # Phase 1（05-23-cross-paper-qa-retrieval-fix）— 运行时证据守门字段。
    # 三个计数字段必须用 operator.add reducer，否则未来若工具并行调用会丢累加。
    unindexed_paper_ids: List[str]
    vector_call_count: Annotated[int, operator.add]
    graph_call_count: Annotated[int, operator.add]
    graph_hint_section: str
    graph_citation_intersection_count: int
    source_status_reasons: Dict[str, str]
    # 05-24: sufficiency_grader 判定结果（"sufficient" / "insufficient"）。
    sufficiency_result: str
    # 请求级 RAG trace 对象，由 ensure_trace 初始化，随节点流转至 write_back。
    trace: Any


class ComparatorState(TypedDict):
    """
    论文对比子图专属状态。

    Attributes:
        paper_ids: 待对比的论文 paper_id 列表，2-5篇（输入）
        report:    生成的 Markdown 对比表格报告（输出）
    """
    # 待对比论文 ID 列表。
    paper_ids: List[str]
    # 最终对比报告。
    report: str


class CitationState(TypedDict):
    """
    引用链子图专属状态。

    Attributes:
        paper_id:   起始论文的 paper_id（输入）
        depth:      引用树展开深度，默认1，最大2，防止爆炸式增长（输入）
        references: 该论文引用的论文列表（输出）
        cited_by:   引用该论文的论文列表（输出）
    """
    # 起始论文 ID。
    paper_id: str
    # 展开深度。
    depth: int
    # 引用的论文列表。
    references: List[Paper]
    # 被引用的论文列表。
    cited_by: List[Paper]


# ============================================================
# 6. 对话管理模型（Phase C）
# ============================================================

class Message(TypedDict):
    """
    对话消息模型。

    Attributes:
        id:         消息唯一标识符
        role:       角色，"user" 或 "assistant"
        content:    消息内容
        title:      可选标题（用于 assistant 消息分类）
        tone:       可选语气标记（如 "accent", "error"）
        created_at: 创建时间戳（ISO 8601 格式）
    """
    # 消息唯一 ID。
    id: str
    # 角色。
    role: str
    # 消息正文。
    content: str
    # 可选标题。
    title: Optional[str]
    # 可选语气标记。
    tone: Optional[str]
    # 创建时间。
    created_at: str


class Conversation(TypedDict):
    """
    对话元数据模型。

    Attributes:
        id:                对话唯一标识符
        title:             对话标题
        created_at:        创建时间（ISO 8601 格式）
        updated_at:        更新时间（ISO 8601 格式）
        is_pinned:         是否置顶
        tags:              标签列表
        associated_papers: 关联论文 ID 列表
        message_count:     消息数量
    """
    # conversations.db 中的 conversation_id。
    id: str
    # 对话标题。
    title: str
    # 创建时间。
    created_at: str
    # 更新时间。
    updated_at: str
    # 是否置顶。
    is_pinned: bool
    # 标签列表。
    tags: List[str]
    # 关联论文 ID 列表。
    associated_papers: List[str]
    # 消息数量。
    message_count: int


class ConversationDetail(TypedDict):
    """
    对话完整详情模型（包含消息）。

    Attributes:
        id:                对话唯一标识符
        title:             对话标题
        messages:          消息列表
        associated_papers: 关联论文 ID 列表
        created_at:        创建时间
        updated_at:        更新时间
        is_pinned:         是否置顶
        tags:              标签列表
    """
    # 对话唯一 ID。
    id: str
    # 对话标题。
    title: str
    # 消息列表。
    messages: List[Message]
    # 关联论文 ID 列表。
    associated_papers: List[str]
    # 创建时间。
    created_at: str
    # 更新时间。
    updated_at: str
    # 是否置顶。
    is_pinned: bool
    # 标签列表。
    tags: List[str]
