"""
PaperAnalyzer 子图：对单篇论文做结构化分析。

它处在“论文获取 -> 结构化理解 -> 后续入库/展示”这条链的中间层，
负责把一篇 `Paper` 从“只有标题/摘要/作者等元数据”变成“带 analysis 的论文对象”。

它不是主图里直接暴露给用户的功能节点，而是被 arXiv 获取、入库预处理、
论文展示等上游流程复用的单篇分析子图。

特性：
  - 懒加载：仅在需要时触发一次 LLM 调用
  - 输入可选全文：优先使用全文节选，其次使用摘要
  - 失败显式暴露：解析失败/调用失败抛给上层记录状态

职责边界：
  - 不负责搜索论文来源
  - 不负责写 `paper_db`
  - 不负责向量化或 PDF 分块
  - 只负责生成结构化 `PaperAnalysis`
"""

import asyncio
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from app.models.models import Paper, PaperAnalysis, PaperAnalyzerState
from app.models.structured_outputs import PaperAnalysisOutput
from app.services.llm_service import get_llm

# 保留可 monkeypatch 的模块级入口，默认延迟初始化。
llm = None


def _llm():
    """按需获取统一 LLM，避免模块导入阶段强依赖环境变量。"""
    return llm or get_llm()

# 全文截取长度（字符数）
# 摘要+引言+结论通常覆盖核心信息，控制长度节省 Token
FULL_TEXT_TRUNCATE = 3000
# 这里截取的是“分析用上下文”，不是分块入库长度；目的是控制单次分析 token 成本。

def _build_result_state(paper: Paper, analysis: PaperAnalysis) -> PaperAnalyzerState:
    """把 `analysis` 写回 `paper`，并组装子图返回状态。"""
    # 不原地修改传入 `paper`，而是构造一份带 analysis 的新对象。
    updated_paper = {**paper, 'analysis': analysis}
    return {
        # 顶层 `analysis` 方便子图消费者直接读取。
        'analysis': analysis,
        # 同时把 `paper.analysis` 补齐，便于下游继续沿用整篇论文对象。
        'paper': updated_paper,
    }


def _build_analysis_prompt(paper: Paper, full_text: Optional[str]) -> str:
    """
    构造分析 Prompt（优先全文节选，其次摘要）。

    这里的关键权衡是：
      - 全文更完整，但 token 更贵
      - 摘要更短，但信息可能不足

    所以当前策略是：如果传入了较完整全文，就截取前段；否则回退到摘要。
    """
    # 优先用全文前段；没有可靠全文时再用摘要。
    if full_text and len(full_text.strip()) > 100:
        # 截取全文前段，优先覆盖摘要、引言等最密集的信息区。
        content = full_text[:FULL_TEXT_TRUNCATE]
        content_label = "论文全文（节选）"
    else:
        # 没有可靠全文时，仍然允许仅基于摘要完成分析。
        content = paper.get('abstract', '')
        content_label = "论文摘要"

    # 作者列表最多展示前三位，避免 prompt 上下文被长作者表稀释。
    authors_str = ', '.join(paper.get('authors', [])[:3])
    if len(paper.get('authors', [])) > 3:
        authors_str += ' et al.'

    # Prompt 直接约束输出 JSON，对应结构化输出模型。
    prompt = f"""你是一位学术论文分析专家。请对以下论文进行结构化分析。

论文标题：{paper.get('title', '')}
作者：{authors_str}
发表日期：{paper.get('published_date', '')}
分类：{', '.join(paper.get('categories', []))}

{content_label}：
{content}

要求：
1. 所有字段必须填写，不能为空
2. 使用简洁自然的中文输出
3. keywords 提取5个最核心的技术关键词
4. related_work 列出论文明确提到的相关工作名称（方法名/模型名/论文简称）
5. tldr 用 1-2 句话概括整篇论文
6. methodology_type 按实际方法论选择对应类型
7. methodology_detail 说明数据集、实验设置、指标，或理论分析方式
8. 直接输出 JSON，不要用 markdown 代码块包裹"""
    return prompt


def _invoke_analysis(prompt: str, llm_client: Any | None = None) -> PaperAnalysis:
    """
    调用结构化输出接口并转换为 `PaperAnalysis`。

    这里把 LLM 输出先约束成 `PaperAnalysisOutput`，再投影成业务层 `PaperAnalysis`，
    目的是把“模型输出 schema”和“系统内部 analysis 结构”解耦。
    """
    # 这里通过结构化输出约束模型返回，减少手工 JSON 解析脆弱性。
    structured_llm = (llm_client or _llm()).with_structured_output(PaperAnalysisOutput)
    parsed = structured_llm.invoke([HumanMessage(content=prompt)])
    # `PaperAnalysisOutput` 再统一转换成业务层使用的 `PaperAnalysis`。
    return PaperAnalysis(**parsed.model_dump())


def _run_analysis(
    paper: Paper,
    full_text: Optional[str],
    *,
    force_analysis: bool = False,
    llm_client: Any | None = None,
) -> PaperAnalyzerState:
    """
    论文分析节点：生成并写回 `PaperAnalysis`。

    这是 PaperAnalyzer 子图唯一的核心节点。
    输入是 `PaperAnalyzerState`，输出仍然是同一个状态结构，只是把
    `paper["analysis"]` 和顶层 `analysis` 补齐。
    """
    title_short = paper.get('title', 'Unknown')[:50]
    if paper.get('analysis') is not None and not force_analysis:
        print(f"  [PaperAnalyzer] 跳过（已有分析）: {title_short}...")
        return {'analysis': paper['analysis'], 'paper': paper}
    print(f"  [PaperAnalyzer] 开始分析: {title_short}...")
    prompt = _build_analysis_prompt(paper, full_text)
    try:
        analysis = _invoke_analysis(prompt, llm_client)
    except Exception as e:
        print(f"  [PaperAnalyzer] 结构化分析失败: {e}")
        raise
    print(f"  [PaperAnalyzer] 分析完成: {title_short}...")
    print(f"    问题: {analysis['problem'][:80]}...")
    print(f"    方法: {analysis['method'][:60]}...")
    print(f"    TLDR: {analysis['tldr'][:80]}...")
    print(f"    关键词: {analysis['keywords']}")
    return _build_result_state(paper, analysis)


def analyze_node(state: PaperAnalyzerState) -> PaperAnalyzerState:
    """
    论文分析节点：生成并写回 `PaperAnalysis`。

    这是 PaperAnalyzer 子图唯一的核心节点。
    输入是 `PaperAnalyzerState`，输出仍然是同一个状态结构，只是把
    `paper["analysis"]` 和顶层 `analysis` 补齐。
    """
    return _run_analysis(
        state['paper'],
        state.get('full_text'),
        force_analysis=bool(state.get('force_analysis')),
    )


def create_paper_analyzer():
    """
    创建 PaperAnalyzer 子图。

    单节点子图，流程简单：`START -> analyze_node -> END`。
    单独封装成子图的意义是和其他节点保持一致的调用方式，
    这样 agent / ingest 链都可以统一用 `ainvoke()` 调它。

    Returns:
        编译后的 PaperAnalyzer 子图
    """
    # 这个子图没有路由分支，本质上是一个单节点工作流。
    # 但保留成子图后，主图或 ingest 链就可以用统一的 graph 调用方式复用它。
    workflow = StateGraph(PaperAnalyzerState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", END)
    return workflow.compile()


async def analyze_paper(
    paper: Paper,
    full_text: Optional[str] = None,
    *,
    force_analysis: bool = False,
    llm_client: Any | None = None,
) -> Paper:
    """
    便捷入口：分析论文并返回带 `analysis` 的 `Paper`。

    适合上层只关心“拿到一篇已分析论文”的场景，不需要自己手动构造状态对象。
    """
    result = await asyncio.to_thread(
        _run_analysis,
        paper,
        full_text,
        force_analysis=force_analysis,
        llm_client=llm_client,
    )
    return result['paper']
