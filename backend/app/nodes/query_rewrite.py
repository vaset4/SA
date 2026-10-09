"""
检索前查询改写与 Multi-Query 扩展。

这个模块位于 RAG 检索前置阶段，负责把原始问题改写成更适合检索的主查询，
并按需补充若干 supplementary queries。

需要特别区分：
  - 它不是检索路由层，不负责决定走 milvus / graph / fusion
  - 它不直接执行向量检索或图谱检索
  - 它的输出只是“后续检索可用的 query 列表”
"""

from __future__ import annotations

# json + re 用于从 LLM 输出里提取并解析 JSON 对象。
import json
import re
# List 用于描述补充查询列表。
from typing import List

# Pydantic 模型用于约束查询改写结果结构。
from pydantic import BaseModel, Field

# 导入全局配置，读取 supplementary query 数量上限。
from app.config import get_settings
# 导入统一 LLM 服务。
from app.middleware.trace_context import trace_label
from app.services.llm_service import get_llm

# 保留可 monkeypatch 的模块级入口，默认延迟初始化。
settings = None


def _settings():
    """按需读取配置，避免模块导入阶段触发全局初始化。"""
    return settings or get_settings()


class QueryRewriteResult(BaseModel):
    """查询改写结果。"""

    # 原始用户查询。
    original: str = Field(description="原始查询")
    # 改写后的主查询，后续通常优先作为检索关键词。
    rewritten: str = Field(description="改写后的主查询")
    # 用于 Multi-Query 检索扩展的补充查询列表。
    supplementary: List[str] = Field(default_factory=list, description="补充查询列表")


def rewrite_query(query: str, context: str = "") -> QueryRewriteResult:
    """
    调用 LLM 生成主改写查询和补充查询。

    这个函数通常发生在真正检索前，因此它优化的是“检索召回表达”，不是面向用户展示的润色。
    """
    # 每次按需获取统一 LLM 实例。
    llm = get_llm()
    # 根据原始查询和当前上下文拼装提示词。
    prompt = _build_prompt(query, context)
    # 这里是同步调用，适用于当前在 RAG 检索前的快速改写路径。
    with trace_label("query_rewrite", "rewrite"):
        response = llm.invoke(prompt)
    # 从模型输出中提取 JSON 对象。
    payload = _parse_json_object(response.content)
    return QueryRewriteResult(
        # 原始查询原样保留。
        original=query,
        # 若模型没给出 rewritten，则回退原始查询。
        rewritten=str(payload.get("rewritten") or query).strip(),
        # supplementary 会做类型清洗、去重和数量裁剪。
        supplementary=_normalize_supplementary(payload.get("supplementary")),
    )


def get_all_queries(result: QueryRewriteResult) -> List[str]:
    """
    把 original / rewritten / supplementary 合并成去重后的查询列表。

    保留原始查询的意义在于：即使改写质量一般，也不会完全丢掉用户原话带来的召回机会。
    """
    queries: List[str] = []
    # 顺序保留：先原始查询，再主改写，再补充查询。
    for candidate in [result.original, result.rewritten, *result.supplementary]:
        normalized = candidate.strip()
        # 跳过空串，并保持顺序去重。
        if normalized and normalized not in queries:
            queries.append(normalized)
    return queries


def _build_prompt(query: str, context: str) -> str:
    """生成查询改写提示词；上下文只作为 disambiguation 辅助，不是强制检索范围。"""
    return (
        "你是学术论文检索系统的查询改写器。\n"
        "请输出 JSON，对象必须包含 rewritten 和 supplementary 两个字段。\n"
        f"supplementary 最多返回 {_settings().multi_query_count} 个字符串。\n"
        "不要输出解释文字，只输出 JSON。\n\n"
        f"原始查询: {query}\n"
        f"上下文: {context or '(none)'}\n"
    )


def _parse_json_object(content: str) -> dict:
    """从模型输出文本中提取第一个 JSON 对象。"""
    # 用正则截取最外层 `{...}` 片段，兼容模型偶尔输出前后缀文本。
    match = re.search(r"\{[\s\S]*\}", content or "")
    if not match:
        raise ValueError("查询改写结果不是合法 JSON")
    # 解析 JSON 文本。
    payload = json.loads(match.group())
    # 顶层必须是对象。
    if not isinstance(payload, dict):
        raise ValueError("查询改写结果不是 JSON 对象")
    return payload


def _normalize_supplementary(value) -> List[str]:
    """清洗 supplementary 查询列表，做类型过滤、去重和数量裁剪。"""
    # 不是列表时直接回退为空列表。
    if not isinstance(value, list):
        return []
    queries: List[str] = []
    for item in value:
        # 只接受字符串项。
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        # 去掉空串并保持顺序去重。
        if normalized and normalized not in queries:
            queries.append(normalized)
    # 按配置限制 supplementary query 数量。
    return queries[: _settings().multi_query_count]
