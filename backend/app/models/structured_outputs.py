"""
结构化输出模型。

这里集中定义各类 LLM 结构化返回 schema，供主图路由、RAG ReAct 决策、
文档评分、Self-RAG 评估、带引用回答生成等节点复用。

这些模型本身不承载业务逻辑，它们的职责是：
  - 约束模型输出格式，降低解析歧义
  - 让调用侧明确知道每个字段的语义和边界
  - 作为节点之间“结构化结果”而不是“图状态”的契约
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReactDecision(BaseModel):
    """
    RAG 子图中 ReAct 决策节点的结构化返回。

    `action` 枚举需要与 rag 子图里实际暴露给模型的工具集合保持同步；
    调用侧会把它转换成 tool_call，或在 `SUFFICIENT` 时结束检索环。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal[
        "local_rag_search",
        "read_chunk",
        "graph_search",
        "SUFFICIENT",
    ] = Field(description="下一步动作")
    args: dict = Field(default_factory=dict, description="动作参数")
    reasoning: str = Field(default="", description="做出该动作的理由")


class RouteDecision(BaseModel):
    """主图 router 节点的结构化路由决策。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    route: Literal[
        "rag_qa",
        "arxiv_search",
        "deep_research",
        "compare",
        "citation",
        "literature_review",
        "related_work",
        "free_chat",
    ] = Field(description="匹配到的路由")
    confidence: float = Field(ge=0.0, le=1.0, description="路由置信度")
    reasoning: str = Field(description="路由原因")


class DocumentGrade(BaseModel):
    """首轮检索结果的相关性评分结果。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    binary_score: Literal["relevant", "irrelevant"] = Field(description="相关性")
    reasoning: str = Field(description="判断理由")


class AnswerEvaluation(BaseModel):
    """
    Self-RAG 回答质量评估结果。

    Self-RAG 的编排层与节点层都已随 ReAct 链摘除，这个模型现在只由
    公共结构化输出入口提供给调用方。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    relevance: Literal["high", "medium", "low"] = Field(description="回答相关性")
    faithfulness: Literal["high", "medium", "low"] = Field(description="回答忠实度")
    completeness: Literal["complete", "partial", "insufficient"] = Field(
        description="回答完整性"
    )
    quality_score: float = Field(ge=0.0, le=1.0, description="综合质量分")
    grounded: bool = Field(default=True, description="答案是否完全由证据支撑")
    missing_info: list[str] = Field(default_factory=list, description="缺失信息点")
    follow_up_queries: list[str] = Field(default_factory=list, description="建议继续检索的问题")
    has_hallucination: bool = Field(default=False, description="是否存在幻觉")


class PaperAnalysisOutput(BaseModel):
    """单篇论文分析节点的结构化输出。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    problem: str = Field(description="论文核心问题")
    method: str = Field(description="论文核心方法")
    contribution: list[str] = Field(description="主要贡献")
    limitation: str = Field(description="论文局限性")
    keywords: list[str] = Field(description="关键词")
    related_work: list[str] = Field(description="相关工作")
    tldr: str = Field(description="1-2 句话总结全文")
    methodology_type: Literal["experimental", "theoretical", "survey", "hybrid"] = Field(description="方法论类型")
    methodology_detail: str = Field(description="方法论细节: 数据集、实验设置、评估指标等")


class Citation(BaseModel):
    """单条证据引用。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    paper_id: str = Field(description="来源论文 ID")
    chunk_id: str = Field(default="", description="来源 chunk ID")
    section: str = Field(default="", description="来源章节")
    page_num: int = Field(default=0, description="来源页码")
    excerpt: str = Field(default="", description="支撑回答的摘录")


class CitedAnswer(BaseModel):
    """
    带引用的最终回答。

    产出方（Self-RAG 精炼节点）已随 ReAct 链摘除，这个模型现在只由
    公共结构化输出入口提供给调用方。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(description="回答正文")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    confidence: Literal["high", "medium", "low"] = Field(description="回答置信度")
    confidence_reasoning: str = Field(description="置信度判断依据")
