"""
GraphRAG 检索模式路由。

这个模块只负责决定“本轮检索应该走哪条路”，不直接执行 Milvus 或 LightRAG
 检索。当前可选模式有三种：
  - `milvus`: 纯 chunk 主检索链
  - `graph`: 纯图谱聚合检索
  - `fusion`: 先 Milvus、再 LightRAG、最后顺序合并
"""

from __future__ import annotations

# logging 用于记录路由决策过程。
import logging
# Literal 用于约束返回模式字符串。
from typing import Literal

# Pydantic 模型用于兼容现有检索执行层的 mode 结构。
from pydantic import BaseModel, Field, ConfigDict, model_validator

# 导入全局配置，读取 GraphRAG 开关。
from app.config import get_settings
from app.services.retrieval_strategy_service import build_retrieval_plan

# 模块级日志器。
logger = logging.getLogger(__name__)
# 保留可 monkeypatch 的模块级入口，默认延迟初始化。
settings = None


def _settings():
    """按需读取配置，避免模块导入阶段触发全局初始化。"""
    return settings or get_settings()


class RetrievalMode(BaseModel):
    """
    检索路由决策结果。

    这是“只负责策略，不负责执行”的路由产物。
    真正执行 Milvus / LightRAG 检索的是 `retrieval_service.py`。
    """

    # 允许 LLM 用 `retrieval_mode` 这个别名字段回填 `mode`。
    model_config = ConfigDict(populate_by_name=True)
    # 主检索模式：milvus / graph / fusion 三选一。
    mode: Literal["milvus", "graph", "fusion"] = Field(
        validation_alias="retrieval_mode",
        description="检索模式",
    )
    # 只有 graph/fusion 用得上，但为了结构统一，milvus 也保留该字段。
    graph_mode: Literal["local", "global", "hybrid"] = Field(default="hybrid")
    # 记录路由原因，便于日志、调试和前端展示。
    reasoning: str = Field(default="", description="路由理由")

    @model_validator(mode="before")
    @classmethod
    def _normalize_nullable_graph_mode(cls, value):
        """兼容上游偶发返回 `graph_mode: null` 的结构化结果。"""
        if isinstance(value, dict) and value.get("graph_mode") is None:
            value = {**value, "graph_mode": "hybrid"}
        return value


# 提示词只做“模式选择”，不让模型直接执行检索。
PROMPT = """You are a retrieval router for an academic paper assistant.

Choose exactly one retrieval mode:
- milvus: standard chunk retrieval for factual or excerpt-driven questions
- graph: entity/relation/community questions that should use LightRAG directly
- fusion: broad synthesis questions that benefit from both chunk retrieval and graph retrieval

Also choose the best graph_mode:
- local: entity-centric lookup
- global: community/topic overview
- hybrid: combine local and global

User query: {query}

Return fields:
- retrieval_mode: one of milvus / graph / fusion
- graph_mode: one of local / global / hybrid
- reasoning: short explanation"""

async def route_retrieval(query: str, *, structured_query: dict | None = None) -> RetrievalMode:
    """
    路由检索模式，支持 milvus/graph/fusion 三种模式。

    决策流程：
    1. 检查 graph_rag_enabled 配置
    2. 启发式规则匹配
    3. LLM 结构化输出
    4. 失败降级到 milvus

    返回的 RetrievalMode 仍保持旧接口，但内部决策已收口到产品语义：
    - 项目内问题：`Milvus` 主检索，图谱证据留给 Neo4j 解释层
    - 库级综述：`Milvus` 主检索，`LightRAG` 只做辅助提示
    """
    # GraphRAG 总开关关闭时，所有查询都强制回到 Milvus 主链。
    if not _settings().graph_rag_enabled:
        result = RetrievalMode(mode="milvus", graph_mode="hybrid", reasoning="GraphRAG 未启用")
        logger.info(f"检索路由: {query[:50]}... → {result.mode} ({result.reasoning})")
        return result
    structured = structured_query if isinstance(structured_query, dict) else {}
    plan = build_retrieval_plan(
        query=query,
        project_id=structured.get("project_id"),
        paper_ids=structured.get("paper_ids"),
        dialogue_mode=structured.get("dialogue_mode"),
        paper_focus_mode=((structured.get("paper_focus") or {}) if isinstance(structured.get("paper_focus"), dict) else {}).get("mode"),
    )
    mode = "fusion" if plan.enable_library_graph_hint else plan.primary_store
    result = RetrievalMode(mode=mode, graph_mode=plan.graph_mode, reasoning=plan.reason)
    logger.info(f"检索路由: {query[:50]}... → {result.mode} ({result.reasoning})")
    return result
