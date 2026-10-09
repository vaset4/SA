"""回合上下文解析：意图与三层论文焦点。

这是 Planner 首次分析的第一步。它曾经是 Planner 之前的独立图节点 intent_focus，
现在并入 Planner，保证「解析意图与焦点」在整条链上只发生一次（design.md §10 切片 1）。

作者：xxx
创建时间：2026-08-29 00:00:00
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.services.internal_research.focus_resolver import resolve_focus
from app.services.internal_research.intent_classifier import classify_intent

# 规则意图分类无命中时的兜底意图，与 intent_classifier 的默认档一致
_DEFAULT_INTENT = "explore"

# 焦点解析读到的本轮论文记录在状态增量里的键名。规划上下文装载器顺着它复用这一次读取，
# 同一回合不再为同样的论文 ID 重跑一遍库查询
TURN_PAPERS_KEY = "_turn_papers"


async def resolve_turn_context(
    state: dict[str, Any],
    *,
    load_papers: Callable[[list[str]], Awaitable[list[dict[str, Any]]]],
) -> dict[str, Any]:
    """解析本回合的意图与论文焦点。

    读论文记录的能力由调用方注入，本层不碰仓储：这一轮真正读库的是装载层，它把重试
    与失败面归因一起带进来，焦点解析只负责拿记录做选择。注入方要保证按传入的 paper_ids
    顺序返回记录（`get_papers_by_ids` 用 array_position 保序），焦点解析的序数基准依赖它。

    Args:
        state: 当前 RAGState 字典态，取 user_query / structured_query / chat_history；
            不修改传入的 state
        load_papers: 「按论文 ID 取记录」的读取能力，由调用方（规划上下文装载层）提供

    Returns:
        dict[str, Any]: 可直接并进 Planner 节点返回值的状态增量，含 intent 与 focus 三层焦点；
            paper_ids 里带的记录同时按 TURN_PAPERS_KEY 带出供装载层复用

    Raises:
        Exception: 读取论文记录的基础设施故障原样上抛，由调用方按失败面归因
    """
    question = str(state.get("user_query") or "")
    paper_ids = _scoped_paper_ids(state)
    # 1. 先判断这轮问题更接近 lookup / compare / inventory / explore 哪种内部研究意图
    intent_result = classify_intent(question)
    intent = str(intent_result.get("intent") or _DEFAULT_INTENT)
    # 2. 取回本轮清单的标题：用户点名标题时要靠它收窄，没有标题就只剩序数指代可解
    turn_papers = await _load_turn_papers(paper_ids, load_papers)
    # 3. 再按项目范围和对话历史解析这轮真正聚焦的论文集合
    focus_result = resolve_focus(
        question,
        intent=intent,
        project_paper_ids=paper_ids,
        conversation_history=state.get("chat_history") or [],
        papers=turn_papers,
    )
    return {
        "intent": intent,
        "focus": focus_result,
        "retry_count": 0,
        # 第 2 步已按同一份 paper_ids 读过一次论文记录，随增量带出去给上下文预装载复用，
        # 同一回合不再为同样的 ID 重跑一次库查询。调用方用完即丢，不写进图状态
        TURN_PAPERS_KEY: turn_papers,
    }


async def _load_turn_papers(
    paper_ids: list[str],
    load_papers: Callable[[list[str]], Awaitable[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """取回本轮清单的论文记录，供按标题点名收窄使用。

    读取本身在注入进来的 `load_papers` 里：那一路带着装载层的重试与失败面归因，本层不兜库故障
    ——读不到就让异常冒出去，由装载层报「论文库读不出来」，不在这里假装「没有标题可匹配」。

    Args:
        paper_ids: 本轮结构化查询携带的论文范围，顺序即清单顺序
        load_papers: 调用方注入的读取能力，按 paper_ids 取记录

    Returns:
        list[dict[str, Any]]: 论文记录列表；本轮没有论文时为空列表，且不触发读取
    """
    # 本轮没有论文范围时短路：省掉一次注定空结果的库查询，也不惊动调用方的重试
    if not paper_ids:
        return []
    return await load_papers(paper_ids)


def _scoped_paper_ids(state: dict[str, Any]) -> list[str]:
    """取本回合结构化查询里携带的论文范围。"""
    structured_query = state.get("structured_query")
    if not isinstance(structured_query, dict):
        return []
    return _clean_ids(structured_query.get("paper_ids"))


def _clean_ids(raw: Any) -> list[str]:
    """把可能带空值和空白的 ID 列表规整成干净字符串列表。"""
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item or "").strip()]
