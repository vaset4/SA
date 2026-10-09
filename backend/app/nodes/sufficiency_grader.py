"""sufficiency_grader 节点：评估当前证据是否足够回答用户问题。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings
from app.middleware.observability import log_structured_event
from app.middleware.trace_context import trace_label
from app.services.llm_service import StructuredOutputError, get_llm

logger = logging.getLogger(__name__)


class SufficiencyDecision(BaseModel):
    """证据充分性判定结果。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    sufficient: bool = Field(description="当前证据是否足够充分回答用户问题")
    reason: str = Field(description="判断理由（一句话）")


SUFFICIENCY_PROMPT = """你是一个证据充分性评估器。

用户问题：{query}

当前已收集的证据摘要：
{evidence}

请判断：基于以上证据，能否生成一个有依据、有引用的高质量回答？
- 如果证据足够覆盖问题的核心要点，将充分性判定为真
- 如果证据明显不足（缺少关键信息、与问题无关、或只有片面信息），将充分性判定为假

只评估证据充分性，不要尝试回答问题本身。"""


def _build_evidence_summary(state: dict[str, Any]) -> str:
    """从 state 构建证据摘要供 grader 评估。"""
    parts: list[str] = []
    search_results = state.get("search_results") or []
    if search_results:
        parts.append(f"向量检索：{len(search_results)} 条结果")
        for i, item in enumerate(search_results[:3], 1):
            if isinstance(item, dict):
                score = item.get("score", 0)
                text = str(item.get("text", item))[:200]
            else:
                score = 0
                text = str(item)[:200]
            parts.append(f"  [{i}] score={score:.3f} {text}")
    graph_context = state.get("graph_context") or ""
    if graph_context.strip():
        parts.append(f"图谱上下文：{graph_context[:500]}")
    else:
        parts.append("图谱上下文：（空）")
    return "\n".join(parts) if parts else "（无证据）"


async def sufficiency_grader_node(state: dict[str, Any]) -> dict[str, Any]:
    """独立评估当前证据是否足够回答问题。

    - 返回 sufficiency_result 为 "sufficient" / "insufficient"
    - 结构合同错误直接暴露；其他历史故障保留 insufficient 路径

    这个节点已不在任何运行链上（`rag_sufficiency_grader_enabled` 的描述也这么写），
    仅保留给旧配置与离线校验使用。
    """
    settings = get_settings()
    if not settings.rag_sufficiency_grader_enabled:
        return {"sufficiency_result": "sufficient"}

    query = state.get("query") or state.get("question") or state.get("user_query") or ""
    evidence = _build_evidence_summary(state)
    iteration = state.get("iteration", 0)

    try:
        llm = get_llm()
        prompt = SUFFICIENCY_PROMPT.format(query=query, evidence=evidence)
        with trace_label("rag_answer", "sufficiency_grader"):
            result = await llm.with_structured_output(SufficiencyDecision).ainvoke(prompt)
        log_structured_event(
            logger,
            "rag_sufficiency_graded",
            sufficient=result.sufficient,
            iteration=iteration,
            reason=result.reason[:200],
        )
        if result.sufficient:
            return {"sufficiency_result": "sufficient", "should_continue": False}
        return {"sufficiency_result": "insufficient", "should_continue": True}
    except (ValidationError, StructuredOutputError):
        # 2. 【证据充分性】【合同失败】模型格式错误不能冒充真实证据不足
        raise
    except (OutputParserException, Exception) as exc:
        log_structured_event(
            logger,
            "rag_sufficiency_graded",
            sufficient=False,
            iteration=iteration,
            reason=f"grader_error_default_insufficient: {str(exc)[:100]}",
        )
        return {"sufficiency_result": "insufficient", "should_continue": True}
