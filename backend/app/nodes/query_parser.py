"""
查询解析器：从用户查询中提取过滤条件。

支持从自然语言中提取：
- 年份范围：2020-2024、2020年后、2024年之前
- 分类：cs.AI、机器学习、NLP
- 作者：Hinton、Bengio
- 论文范围：特定论文 ID、标题关键词

这个模块只负责把自然语言约束解析成结构化过滤条件，不直接执行搜索。
它的典型调用方会拿 `clean_query` 去检索，再把结构化字段转换成 filter_expr。
"""

from __future__ import annotations

# logging 用于记录启发式/LLM 解析结果。
import json
import logging
# re 用于启发式提取年份、分类、论文 ID 等模式。
import re
# List / Optional 用于描述可选过滤条件。
from typing import List, Optional

# Pydantic 模型用于约束解析结果结构。
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# 导入统一 LLM 服务；启发式不确定时会交给 LLM 结构化解析。
from app.middleware.trace_context import trace_label
from app.services.llm_service import StructuredOutputError, get_llm

# 模块级 logger。
logger = logging.getLogger(__name__)


class QueryFilters(BaseModel):
    """查询过滤条件。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    # 起始年份约束。
    year_min: Optional[int] = Field(default=None, description="最小年份")
    # 截止年份约束。
    year_max: Optional[int] = Field(default=None, description="最大年份")
    # 论文分类约束。
    category: Optional[str] = Field(default=None, description="论文分类")
    # 作者名约束。
    # 注意：启发式分支目前几乎不主动抽作者，更多依赖 LLM fallback 来补齐。
    authors: Optional[str] = Field(default=None, description="作者名称")
    # 论文 ID 白名单约束。
    paper_ids: Optional[List[str]] = Field(default=None, description="论文 ID 列表")
    # 论文标题关键词约束；通常需要调用方进一步映射成 paper_ids。
    paper_keywords: Optional[List[str]] = Field(default=None, description="论文标题关键词")
    # 去掉过滤条件后的“纯净查询”，供后续真正检索使用。
    clean_query: str = Field(description="移除过滤条件后的纯净查询")


# 提示词只负责结构化提取，不负责真正发起搜索。
PROMPT = """Extract search filters from the user query.

Identify:
- Year range: "2020-2024", "after 2020", "before 2024", "2020年后", "2024年之前"
- Category: "cs.AI", "cs.LG", "machine learning", "NLP", "计算机视觉"
- Authors: author names like "Hinton", "Bengio"
- Paper scope: specific paper IDs (like "2301.00001"), paper titles, or keywords indicating specific papers
  * "in paper X", "in the paper about X", "根据论文 X", "在 X 论文中"
  * "compare papers X and Y", "对比论文 X 和 Y"

User query: {query}

Use the JSON Schema to express filters; absent filters are null."""

# 1. 【查询解析】【合同示例】示例从同一模型序列化，字段新增时不会留下第二份手写 JSON
QUERY_FILTER_EXAMPLES = (
    ("搜索 2020-2024 年的 Transformer 论文", QueryFilters(year_min=2020, year_max=2024, clean_query="Transformer 论文")),
    ("Find papers by Hinton about deep learning", QueryFilters(authors="Hinton", clean_query="deep learning")),
    ("What does the Transformer paper say about attention?", QueryFilters(paper_keywords=["Transformer"], clean_query="attention")),
    ("Compare 2301.00001 and 2302.00002", QueryFilters(paper_ids=["2301.00001", "2302.00002"], clean_query="compare")),
)

# arXiv ID 匹配模式。
ARXIV_ID_PATTERN = re.compile(r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b')


def _heuristic_parse(query: str) -> Optional[QueryFilters]:
    """启发式解析，快速提取明显的过滤条件。"""
    # 默认没有提取到任何过滤条件。
    year_min = None
    year_max = None
    category = None
    authors = None
    paper_ids = None
    paper_keywords = None
    clean_query = query

    # 提取 arXiv ID。
    arxiv_matches = ARXIV_ID_PATTERN.findall(query)
    if arxiv_matches:
        # 当前这里直接用 set 去重，返回顺序不保证稳定。
        paper_ids = list(set(arxiv_matches))
        for arxiv_id in arxiv_matches:
            clean_query = clean_query.replace(arxiv_id, '').strip()

    # 提取年份范围：2020-2024。
    year_range_match = re.search(r'(\d{4})\s*[-~到至]\s*(\d{4})', query)
    if year_range_match:
        year_min = int(year_range_match.group(1))
        year_max = int(year_range_match.group(2))
        clean_query = clean_query.replace(year_range_match.group(0), '').strip()

    # 提取单个年份：2020年、2020年后、2020年之前。
    if not year_min and not year_max:
        year_after_match = re.search(r'(\d{4})\s*年?\s*(后|以后|之后|after)', query)
        if year_after_match:
            year_min = int(year_after_match.group(1))
            clean_query = clean_query.replace(year_after_match.group(0), '').strip()

        year_before_match = re.search(r'(\d{4})\s*年?\s*(前|以前|之前|before)', query)
        if year_before_match:
            year_max = int(year_before_match.group(1))
            clean_query = clean_query.replace(year_before_match.group(0), '').strip()

    # 提取 arXiv 分类：cs.AI, cs.LG, cs.CV, cs.CL。
    category_match = re.search(r'\b(cs\.[A-Z]{2}|stat\.[A-Z]{2})\b', query)
    if category_match:
        category = category_match.group(1)
        clean_query = clean_query.replace(category_match.group(0), '').strip()

    # 提取论文范围关键词："in paper X", "in the X paper", "根据 X 论文"。
    # 这里抽出来的是“题目关键词/标题片段”，真正把它映射成 paper_ids 还需要上游或调用方继续处理。
    paper_scope_patterns = [
        r'(?:in|from|about|根据|在)\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:paper|论文)',
        r'(?:paper|论文)\s+(?:about|on|关于)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)',
    ]
    for pattern in paper_scope_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            keyword = match.group(1).strip()
            # 至少保留 3 个字符，避免无意义短词进入 paper_keywords。
            if keyword and len(keyword) > 2:
                paper_keywords = [keyword]
                clean_query = clean_query.replace(match.group(0), '').strip()
                break

    # 清理多余空格。
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()

    # 如果提取到任何过滤条件，则直接返回结构化结果。
    if year_min or year_max or category or authors or paper_ids or paper_keywords:
        return QueryFilters(
            year_min=year_min,
            year_max=year_max,
            category=category,
            authors=authors,
            paper_ids=paper_ids,
            paper_keywords=paper_keywords,
            clean_query=clean_query or query,
        )

    return None


async def parse_query(query: str) -> QueryFilters:
    """
    解析用户查询，提取过滤条件。

    决策流程：
    1. 启发式规则快速匹配
    2. LLM 结构化输出
    3. 结构合同失败原样暴露；其他历史故障保留原查询路径

    Returns:
        QueryFilters: 包含过滤条件和纯净查询
    """
    # 先尝试启发式解析，命中时不再走 LLM。
    heuristic = _heuristic_parse(query)
    if heuristic is not None:
        logger.info(f"查询解析（启发式）: {query[:50]}... → year_min={heuristic.year_min}, year_max={heuristic.year_max}, category={heuristic.category}, paper_ids={heuristic.paper_ids}")
        return heuristic

    # 启发式不足时，再使用 LLM 做结构化解析。
    # 这样做的目的，是把“显而易见的格式化约束”留给便宜稳定的规则，把歧义理解留给 LLM。
    parser = get_llm().with_structured_output(QueryFilters)
    examples = json.dumps([
        {"query": text, "output": example.model_dump(mode="json")} for text, example in QUERY_FILTER_EXAMPLES
    ], ensure_ascii=False)
    prompt = PROMPT.format(query=query) + "\nExamples:\n" + examples
    try:
        # 这里返回的是结构化过滤条件，不会直接发起任何检索。
        with trace_label("query_parser", "parse_filters"):
            result = await parser.ainvoke(prompt)
        logger.info(f"查询解析（LLM）: {query[:50]}... → year_min={result.year_min}, year_max={result.year_max}, category={result.category}, paper_ids={result.paper_ids}")
        return result
    except (ValidationError, StructuredOutputError):
        # 2. 【查询解析】【合同失败】格式违约不能变成无过滤条件的有效检索请求
        raise
    except Exception as exc:
        # 解析失败时降级为“无过滤条件 + 原始查询”。
        logger.warning(
            "查询解析失败: query=%s query_len=%d error=%s",
            query[:50],
            len(query),
            exc,
        )
        return QueryFilters(
            year_min=None,
            year_max=None,
            category=None,
            authors=None,
            paper_ids=None,
            paper_keywords=None,
            clean_query=query,
        )


def build_filter_expr(filters: QueryFilters) -> str:
    """
    根据过滤条件构建 Milvus filter_expr。

    Args:
        filters: 查询过滤条件

    Returns:
        str: Milvus 过滤表达式，如 "year >= 2020 and year <= 2024"
    """
    # 这里仅负责把结构化过滤条件转成 Milvus filter_expr 字符串。
    # 它本身不校验这些条件是否真的有结果，也不处理 title keyword -> paper_ids 的映射。
    conditions: list[str] = []

    if filters.year_min is not None:
        conditions.append(f"year >= {filters.year_min}")

    if filters.year_max is not None:
        conditions.append(f"year <= {filters.year_max}")

    if filters.category:
        # 转义引号，避免 filter_expr 字符串被破坏。
        escaped_category = filters.category.replace('"', '\\"')
        conditions.append(f'category == "{escaped_category}"')

    if filters.authors:
        # 转义特殊字符，避免字符串匹配条件出错。
        escaped_authors = filters.authors.replace('\\', '\\\\').replace('"', '\\"')
        conditions.append(f'authors like "%{escaped_authors}%"')

    if filters.paper_ids:
        # 论文 ID 过滤：paper_id in ["id1", "id2"]。
        escaped_ids = [pid.replace('"', '\\"') for pid in filters.paper_ids]
        ids_str = ', '.join(f'"{pid}"' for pid in escaped_ids)
        conditions.append(f'paper_id in [{ids_str}]')

    # 注意：paper_keywords 需要调用层先做标题匹配再换成 paper_ids，
    # 这里不会直接处理标题关键词搜索。

    # 有条件则用 `and` 连接；没有条件时返回空串。
    return " and ".join(conditions) if conditions else ""
