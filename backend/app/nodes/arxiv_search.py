"""
arXiv 论文检索能力模块。

这里承载「按关键词检索 arXiv → LLM 相关性过滤 → 生成 Markdown 报告」三步流程。
它原本是主图里 arxiv_search 意图的三个节点；主链收敛后顶层图只保留
guardrails → router → rag_qa 一条主链，本流程改为可直接调用的能力模块，
由 `/search/arxiv` 路由直接调用（见 api/routes/research_api.py）。

本模块不再注册进 LangGraph，只提供顺序执行入口 `run_arxiv_search`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import List

from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.middleware.observability import log_structured_event
from app.models.models import AgentState, Paper
from app.services.llm_service import get_llm
from app.tools.arxiv_tool import get_arxiv_tool

llm = None
settings = None
logger = logging.getLogger(__name__)

# 相关性判断只取摘要前 300 字，足以覆盖论文主题，控制 Token 消耗
_FILTER_ABSTRACT_CHARS = 300
# 报告里每篇论文的摘要展示长度
_REPORT_ABSTRACT_CHARS = 200
# 报告里作者列表的展示条数，超出部分用 et al. 收尾
_REPORT_AUTHOR_LIMIT = 3


def _llm():
    override = globals().get("llm")
    if override is not None:
        return override
    return get_llm()


def _settings():
    override = globals().get("settings")
    if override is not None:
        return override
    return get_settings()


def _log_agent_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    """统一输出 arXiv 检索能力内部的结构化事件。"""
    log_structured_event(logger, event, level=level, **fields)


def _build_topic_extract_prompt(user_query: str) -> str:
    """
    构造「把用户查询拆成独立主题列表」的 LLM Prompt。

    Args:
        user_query: 用户原始查询

    Returns:
        str: 要求模型只回 JSON 的查询构造 Prompt
    """
    return f"""You are an arXiv search query builder.

Analyze the user's query and extract the core research topics.
Each topic should be a self-contained concept that could independently describe a paper.

User query: "{user_query}"

Rules:
1. If the query combines multiple distinct topics (e.g., "traffic prediction + RAG"), split them.
2. If the query is about a single topic, return only one item.
3. Use concise English phrases (2-5 words each).
4. Do NOT include generic words like "paper", "research", "latest", "related".

Examples:
  "搜索RAG相关论文"
    → {{"topics": ["retrieval augmented generation"]}}

  "交通流量预测和RAG结合的论文"
    → {{"topics": ["traffic flow prediction", "retrieval augmented generation"]}}

  "找transformer改进的最新论文"
    → {{"topics": ["transformer architecture improvements"]}}

  "视觉语言模型用于医学图像分析"
    → {{"topics": ["vision language model", "medical image analysis"]}}

Reply with ONLY the JSON object, no explanation:"""


def _build_filter_prompt(paper: Paper, user_query: str) -> str:
    """
    构造单篇论文的相关性判断 Prompt。

    Args:
        paper: 待判断的论文
        user_query: 用户原始查询

    Returns:
        str: 要求模型回 RELEVANT / REASON 两行的判断 Prompt
    """
    abstract_snippet = paper['abstract'][:_FILTER_ABSTRACT_CHARS]
    return f"""You are a relevance filter for academic paper search.

User's search intent: "{user_query}"

Paper to evaluate:
Title: {paper['title']}
Abstract: {abstract_snippet}

Is this paper relevant to the user's search intent?
A paper is relevant if it directly addresses the main topic(s) the user is looking for.
A paper is NOT relevant if it only incidentally mentions related words without addressing the core topic.

Reply in this exact format:
RELEVANT: YES/NO
REASON: one sentence explanation"""


async def _judge_paper(paper: Paper, user_query: str) -> tuple[Paper, bool, str]:
    """
    对单篇论文做相关性判断，返回 (paper, is_relevant, reason)。

    独立异步函数，供 gather 并发调用。

    Args:
        paper: 待判断的论文
        user_query: 用户原始查询

    Returns:
        tuple[Paper, bool, str]: 原论文、是否相关、模型给出的理由
    """
    prompt = _build_filter_prompt(paper, user_query)
    try:
        response = await _llm().ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()#从响应对象中提取文本内容（response.content），并使用 strip() 去除首尾空白字符

        # 解析 LLM 输出：提取 YES/NO
        is_relevant = 'RELEVANT: YES' in text.upper()

        # 提取理由（供日志输出）
        reason = ''
        for line in text.splitlines():
            if line.upper().startswith('REASON:'):
                reason = line.split(':', 1)[-1].strip()
                break

        return paper, is_relevant, reason
    except Exception as e:
        # 单篇判断失败时，默认保留该论文（宁可误留不漏掉）
        _log_agent_event(
            "arxiv_filter_item_failed",
            level=logging.ERROR,
            outcome="kept_by_default",
            paper_id=paper.get('arxiv_id'),
            error=str(e),
        )
        return paper, True, 'filter error, kept by default'


def _format_paper_entry(index: int, paper: Paper) -> str:
    """
    把单篇论文格式化为报告里的一个 Markdown 条目。

    Args:
        index: 论文在报告中的序号，从 1 开始
        paper: 待格式化的论文

    Returns:
        str: 含元信息、摘要、PDF 链接与可选结构化分析的 Markdown 片段
    """
    authors_str = ', '.join(paper['authors'][:_REPORT_AUTHOR_LIMIT])
    if len(paper['authors']) > _REPORT_AUTHOR_LIMIT:
        authors_str += ' et al.'

    # 摘要截取（避免报告过长）
    abstract_preview = paper['abstract'][:_REPORT_ABSTRACT_CHARS]
    if len(paper['abstract']) > _REPORT_ABSTRACT_CHARS:
        abstract_preview += '...'

    entry = f"""### {index}. {paper['title']}

**arXiv ID**: `{paper['arxiv_id']}`  |  **日期**: {paper['published_date']}  |  **分类**: {', '.join(paper['categories'])}

**作者**: {authors_str}

**摘要**: {abstract_preview}

**PDF**: [{paper['pdf_url']}]({paper['pdf_url']})
"""
    # 如果已有结构化分析（触发了懒加载），附上分析结果
    if paper.get('analysis'):
        analysis = paper['analysis']
        entry += f"""
**分析**:
- **问题**: {analysis['problem']}
- **方法**: {analysis['method']}
- **贡献**: {'; '.join(analysis['contribution'])}
- **关键词**: {', '.join(analysis['keywords'])}
"""
    return entry


def _empty_result_report(user_query: str, raw_count: int) -> str:
    """
    生成「没有可展示论文」时的提示报告。

    从原始搜索结果数量区分两种成因：搜到了但全部被过滤，还是 arXiv 本身没返回结果。

    Args:
        user_query: 用户原始查询
        raw_count: arXiv 原始返回的论文数量

    Returns:
        str: 带针对性建议的 Markdown 提示报告
    """
    if raw_count > 0:
        # 搜到了但全被过滤 → 该交叉领域论文极少
        return (
            f'# arXiv 搜索结果：{user_query}\n\n'
            f'**未找到同时涉及所有主题的论文。**\n\n'
            f'arXiv 搜索返回了 {raw_count} 篇结果，但经相关性过滤后全部不符合要求。\n'
            f'这通常意味着该交叉领域的研究论文目前较少。\n\n'
            f'**建议**：\n'
            f'- 拆开单独搜索各个主题，了解各自领域现状\n'
            f'- 适当放宽查询范围（如去掉某个限定词）\n'
            f'- 尝试不同的表述方式（如换用英文直接搜索）\n'
        )
    # arXiv API 本身没有返回结果
    return (
        f'# arXiv 搜索结果：{user_query}\n\n'
        f'**arXiv 未返回任何结果。**\n\n'
        f'**建议**：\n'
        f'- 检查关键词拼写\n'
        f'- 尝试更宽泛的查询词\n'
        f'- 使用英文关键词\n'
    )


async def arxiv_node(state: AgentState) -> AgentState:
    """
    arXiv 搜索节点：调用 ArxivTool 搜索论文，返回论文列表。

    此节点只负责搜索，不触发 PaperAnalysis（懒加载策略）。
    搜索结果存入 state['papers']，后续经 filter → reporter 流程处理。

    查询构造策略（结构化 AND 查询）：
      arXiv 搜索本质是 BM25 词频匹配。若用多个主题词直接拼接（OR 语义），
      只命中部分词的不相关论文也会排在前面。

      改进方案：LLM 将用户查询拆分为独立主题列表，
      arxiv_tool 用 AND 连接各主题，强制每篇结果同时包含所有主题。

      例：用户查询"交通预测+RAG"
        旧：traffic flow prediction retrieval augmented generation（OR，10篇混杂）
        新：(traffic flow prediction) AND (retrieval augmented generation)（AND，精准）

      单主题查询时只有一个元素，AND 退化为普通查询，不影响效果。

    Args:
        state: 包含 user_query 的状态字典

    Returns:
        AgentState: 包含 papers 字段的状态更新
    """
    user_query = state.get('user_query', '')
    _log_agent_event("arxiv_search_started", query=user_query)

    # LLM 将用户查询拆分为独立主题列表（JSON 格式）
    # 每个主题是一个独立的语义单元，最终用 AND 连接
    extract_prompt = _build_topic_extract_prompt(user_query)

    # 默认查询（LLM 失败时的兜底）
    search_query = user_query
    topics = []

    try:
        kw_response = await _llm().ainvoke([HumanMessage(content=extract_prompt)])
        raw = kw_response.content.strip()

        # 提取 JSON（部分模型会加 markdown 代码块）
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            topics = [t.strip() for t in data.get('topics', []) if t.strip()]

        if topics:
            if len(topics) == 1:
                # 单主题：直接搜索
                search_query = topics[0]
                _log_agent_event("arxiv_query_built", outcome="single_topic", query=search_query)
            else:
                # 多主题：用 AND 连接，强制同时包含所有主题
                # arXiv 语法：(topic1) AND (topic2)
                parts = [f'({t})' for t in topics]
                search_query = ' AND '.join(parts)
                _log_agent_event("arxiv_query_built", outcome="multi_topic_and", query=search_query)
        else:
            _log_agent_event("arxiv_query_built", outcome="empty_topics_fallback", query=search_query)

    except Exception as e:
        _log_agent_event(
            "arxiv_query_build_failed",
            level=logging.ERROR,
            outcome="fallback_original_query",
            query=user_query,
            error=str(e),
        )

    # 执行 arXiv 搜索
    # 多主题 AND 查询结果可能少于 max_results，这是正确行为（宁缺毋滥）
    arxiv_tool = get_arxiv_tool()
    papers = arxiv_tool.search(
        query=search_query,
        max_results=_settings().arxiv_max_results,
    )

    _log_agent_event("arxiv_search_completed", query=search_query, result_count=len(papers))
    return {'papers': papers}


async def arxiv_filter_node(state: AgentState) -> AgentState:
    """
    arXiv 结果过滤节点：用 LLM 对每篇论文打相关性判断，过滤掉不相关结果。

    问题背景：
      arXiv 原生搜索是 BM25 关键词匹配，复合查询（如"交通预测+RAG"）会把
      只包含部分关键词的论文也返回，导致结果充斥不相关内容。

    过滤策略：
      1. 将用户原始查询与每篇论文的标题+摘要一起送给 LLM
      2. LLM 判断该论文是否与用户意图相关，返回 YES 或 NO + 理由
      3. 只保留 YES 的论文写入 filtered_papers
      4. 若过滤后为空，不再回退原始列表（回退等于过滤器失效），交由报告节点提示

    并发执行：
      每篇论文的 LLM 判断相互独立，使用 asyncio.gather 并发请求，
      10篇论文的过滤延迟约等于单次 LLM 调用延迟，而非串行的10倍。

    Args:
        state: 包含 papers（原始搜索结果）和 user_query 的状态字典

    Returns:
        AgentState: 包含 filtered_papers 字段的状态更新
    """
    papers: List[Paper] = state.get('papers', [])
    user_query = state.get('user_query', '')

    if not papers:
        return {'filtered_papers': []}

    _log_agent_event("arxiv_filter_started", result_count=len(papers), query=user_query)

    # 并发判断所有论文
    judge_tasks = [_judge_paper(p, user_query) for p in papers]
    results = await asyncio.gather(*judge_tasks)

    # 收集过滤结果
    filtered = [paper for paper, is_relevant, _reason in results if is_relevant]

    if not filtered:
        _log_agent_event("arxiv_filter_completed", outcome="all_filtered", result_count=0, paper_count=len(papers))
        return {'filtered_papers': filtered}

    _log_agent_event(
        "arxiv_filter_completed",
        outcome="completed",
        paper_count=len(papers),
        result_count=len(filtered),
    )
    return {'filtered_papers': filtered}


async def arxiv_reporter_node(state: AgentState) -> AgentState:
    """
    arXiv 搜索结果报告节点：将过滤后的论文列表格式化为 Markdown 报告。

    报告格式：
      - 标题行：论文编号、arXiv ID、标题
      - 作者和日期
      - 摘要（截取前 200 字）
      - PDF 链接
      - 如果 analysis 不为 None，附上结构化分析（懒加载触发时）

    数据来源：
      优先使用 filtered_papers（过滤后），回退到 papers（原始）。
      报告头部注明过滤情况，让用户知道有多少论文被过滤掉了。

    Args:
        state: 包含 filtered_papers / papers 和 user_query 的状态字典

    Returns:
        AgentState: 包含 report 字段的状态更新
    """
    # 优先使用过滤后的论文列表
    filtered_papers: List[Paper] = state.get('filtered_papers', [])
    all_papers: List[Paper] = state.get('papers', [])
    papers = filtered_papers if filtered_papers else all_papers
    user_query = state.get('user_query', '')

    # 计算过滤信息（供报告头部显示）
    filtered_count = len(all_papers) - len(filtered_papers) if filtered_papers else 0

    if not papers:
        _log_agent_event("arxiv_report_completed", outcome="empty_result")
        return {'report': _empty_result_report(user_query, len(all_papers))}

    _log_agent_event("arxiv_report_started", result_count=len(papers), query=user_query)

    paper_entries = [_format_paper_entry(i, paper) for i, paper in enumerate(papers, 1)]

    # 过滤说明行（让用户知道有多少不相关结果被过滤）
    filter_note = ''
    if filtered_count > 0:
        filter_note = f'（已过滤 {filtered_count} 篇不相关结果，原始搜索返回 {len(all_papers)} 篇）'

    # 拼接完整报告
    separator = '---\n'
    papers_section = separator.join(paper_entries)
    report = f"""# arXiv 搜索结果：{user_query}

共找到 **{len(papers)}** 篇相关论文{filter_note}：

---

{papers_section}
---
*使用 arXiv API 搜索，结果经 LLM 相关性过滤*
"""

    _log_agent_event("arxiv_report_completed", outcome="completed", result_count=len(papers), report_length=len(report))
    return {'report': report}


async def run_arxiv_search(state: AgentState) -> AgentState:
    """
    顺序执行「搜索 → 过滤 → 报告」三步，返回合并后的状态增量。

    这是本能力对外的唯一入口，供 `/search/arxiv` 路由直接调用。

    Args:
        state: 至少包含 user_query 的状态字典

    Returns:
        AgentState: 依次合并 papers / filtered_papers / report 的状态增量
    """
    merged: AgentState = {}
    # 1. 每步都读「原始入参 + 前序步骤产出」，让 filter 拿得到 papers、reporter 拿得到 filtered_papers
    current: AgentState = dict(state)  # type: ignore[assignment]
    # 2. 三步存在严格串行依赖（后一步读前一步产出），不做并发
    for node in (arxiv_node, arxiv_filter_node, arxiv_reporter_node):
        update = await node(current)
        merged.update(update)
        current.update(update)
    return merged
