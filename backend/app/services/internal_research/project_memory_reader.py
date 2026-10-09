"""项目研究记忆层 · Planner 循环内的记忆读取动作，与论文检索平级。

Planner 判断本轮需要项目历史认识时才发起这个动作，并给出查询范围；不需要时不读，
所以这里没有「每轮无条件全翻」的路径。读出来的判断只用于理解背景与延续思路，
不进证据链、不产出 citations —— 记忆不是引用来源。

只读 Postgres，按 paper_id / question_id / memory_type / 关系一跳收窄，
不走向量召回、不碰图库：收窄维度全部由仓储既有的等值与关联过滤承担。
「按研究问题收窄」这一维还需要模型知道项目里有哪些研究问题，那份清单也从这里读
（`read_project_research_questions`），否则模型只能编 id、收窄静默读空。

四道资格门（当前用户 + 当前项目 + accepted + confirmed + fresh）由记忆注入服务的
共享查询函数强制，Planner 只能收窄不能放宽。读取终态分三种：读到条目 success、
门与收窄之后确实没有 empty、记忆存储不可用 failed。empty 与 failed 不归并：
Planner 拿到 empty 会据此继续推进而不猜测，拿到 failed 才知道该重试或提示用户。

作者：xxx
创建时间：2026-08-30 09:50:00
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.config import get_settings
from app.models.research_memory_contracts import (
    MemoryReadItem,
    MemoryReadNarrowing,
    MemoryReadResult,
)
from app.repositories.research_memory_repository import get_research_memory_repository
from app.services.research_memory_injector_service import (
    _excluded_counts,
    _maturity_rank,
    _narrowing_kwargs,
    _project_item,
    _query_gated_items,
    _require_owner,
)

logger = logging.getLogger(__name__)

# 收窄条件的键名，与合同 MemoryReadNarrowing 的键位一一对齐
_FOLLOW_LINKS_KEY = "follow_links"
_QUESTION_ID_KEY = "question_id"


async def read_project_memory(
    context: Mapping[str, Any],
    query_narrowing: MemoryReadNarrowing | None = None,
) -> MemoryReadResult:
    """
    读取当前用户在当前项目里的历史判断，供 Planner 本轮理解背景。

    参数:
        context: 只读的项目范围映射，从中取 user_id 与 project_id；运行时来自图状态的
                 project_scope，这里按键取值、不绑定具体类型
        query_narrowing: 可选收窄条件，None 或空字典表示不收窄，读本项目本人全部合格记忆
    返回:
        MemoryReadResult：读到条目为 success，门与收窄后无条目为 empty，
        记忆存储不可用为 failed 且带 error_kind
    """
    project_id = context.get("project_id")
    user_id = context.get("user_id")
    # 空身份不落 failed 而是抛错：failed 的语义是记忆存储不可用，空身份却是接线错误。
    # 仓储的过滤循环遇到 None 会跳过拼 WHERE，于是把该项目下所有人的记忆读进上下文。
    # 校验放在 try 之外，免得被 except Exception 收成一个看着像环境故障的 failed
    _require_owner(user_id, project_id)
    narrowing = query_narrowing or {}
    try:
        # 1. 种子查询：四道门叠加 Planner 给的收窄条件
        seeds = await _query_gated_items(project_id, user_id, narrowing=narrowing)
        rows = seeds
        # 2. 判定成立才扩一跳，判据见 _should_expand_one_hop
        if _should_expand_one_hop(seeds, narrowing):
            rows = await _expand_one_hop(project_id, user_id, seeds)
    except Exception as exc:
        # 3. 记忆存储不可用要显式落 failed：拿空列表冒充成功，Planner 会把「读不到」
        #    当成「项目里没有相关判断」，接着在错误前提上推进
        logger.error(
            "【研究记忆】【记忆读取】Planner 读取项目记忆失败 project_id=%s user_id=%s narrowing=%s",
            project_id,
            user_id,
            narrowing,
            exc_info=True,
        )
        return {"status": "failed", "items": [], "excluded_counts": {}, "error_kind": type(exc).__name__}
    if not rows:
        # 4. 门与收窄之后确实没有：正常空结果，Planner 据此继续推进而不是重试
        return {"status": "empty", "items": [], "excluded_counts": {}}
    seed_ids = frozenset(row["memory_id"] for row in seeds)
    return _build_success(rows, seed_ids, narrowing.get(_QUESTION_ID_KEY))


async def read_project_research_questions(context: Mapping[str, Any]) -> list[dict[str, str]]:
    """
    取本项目已接纳的研究问题清单，供深度评估把「按研究问题收窄」这一维交给模型判断。

    没有这份清单时模型只能凭印象编一个 question_id，编出来的 id 关联不到任何记忆，
    读取落成 empty，Planner 于是把「id 编错了」读成「项目里还没有这个问题下的判断」——
    安全（只收窄）但静默错空，所以这份清单必须真给。

    参数:
        context: 只读的项目范围映射，取 project_id；运行时来自图状态的 project_scope
    返回:
        list[dict[str, str]]: 已接纳的研究问题，按仓储既有排序（研究目的置顶），
            每条只带 question_id 与 title；无项目或项目内没有研究问题时为空列表
    """
    project_id = str(context.get("project_id") or "").strip()
    if not project_id:
        return []
    try:
        user_id = context.get("user_id")
        if not isinstance(user_id, str):
            raise ValueError("研究问题读取缺少可信用户身份")
        _require_owner(user_id, project_id)
        rows = await get_research_memory_repository().list_questions(project_id, user_id=user_id)
    except Exception:
        # 这份清单只是「可以按哪个研究问题收窄」的提示，不是本轮的信息来源：读不出来
        # 就少一个收窄维度，读取范围反而更宽、四道门照旧管着，不该把整轮问答打断
        # （design §9：记忆侧故障在普通问答里继续主链）。但失败必须留痕，不静默当作
        # 「这个项目没有研究问题」
        logger.error(
            "【研究记忆】【研究问题清单】读取失败，本轮无法按研究问题收窄 project_id=%s",
            project_id,
            exc_info=True,
        )
        return []
    return [
        {"question_id": str(row["question_id"]), "title": str(row.get("title") or "")}
        for row in rows
        if isinstance(row, dict) and row.get("question_id")
    ]


def _should_expand_one_hop(
    seeds: list[dict[str, Any]],
    narrowing: Mapping[str, Any],
) -> bool:
    """
    判断这次读取该不该沿记忆关系扩一跳。

    三个条件必须同时成立：有种子作起点、模型明确要求扩、且这次真的收窄过。
    没收窄时种子已是本项目本人全部合格记忆，合格邻居必然已在其中，扩展的结果注定不变，
    却要按种子逐条发关系查询 —— 白跑之外还多出一个失败点：那些查询在读取的 try 之内，
    链路一抖动就把一次结果注定不变的读取落成 failed，而记忆是唯一来源的回合会因此整轮中断。

    参数:
        seeds: 收窄后的过门条目，扩展的起点
        narrowing: 本次读取的收窄条件
    返回:
        该扩一跳则为 True
    """
    if not seeds or not narrowing.get(_FOLLOW_LINKS_KEY):
        return False
    return bool(_narrowing_kwargs(narrowing))


async def _expand_one_hop(
    project_id: str,
    user_id: str,
    seeds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    把收窄命中的记忆沿关系扩一跳，邻居必须自己也过四道门才进结果。

    参数:
        project_id: 归属项目
        user_id: 记忆归属人
        seeds: 收窄后的过门条目，作为扩展起点
    返回:
        种子 + 合格邻居；种子仍排在前，让收窄直接命中的判断在同成熟度档内优先进上下文
    """
    repo = get_research_memory_repository()
    seed_ids = {row["memory_id"] for row in seeds}
    linked_ids: set[str] = set()
    # 1. 逐条种子取关系行。based_on / supersedes 有方向，conflicts_with 单行存双向，
    #    所以两端都收进来、再整体剔掉种子自身，剩下的才是新邻居。
    #    种子已被收窄过、条数很小才敢按条查；若哪天种子上千，改成一次 list_links(project_id)
    #    取全项目关系再进程内筛
    for row in seeds:
        for link in await repo.list_links_of_memory(row["memory_id"]):
            linked_ids.add(link["src_memory_id"])
            linked_ids.add(link["dst_memory_id"])
    wanted = linked_ids - seed_ids
    if not wanted:
        return list(seeds)
    # 2. 邻居绝不按 memory_id 单独捞：那条查询不带门，等于谁跟合格记忆有关系谁就能进
    #    上下文，别人的记忆、待审草稿、已失效的判断都能顺着关系溜进来。改成重跑一次
    #    不带收窄的过门查询再与邻居 id 取交集 —— 能落在交集里的邻居，自己也被门放行过
    pool = await _query_gated_items(project_id, user_id)
    return list(seeds) + [row for row in pool if row["memory_id"] in wanted]


def _build_success(
    rows: list[dict[str, Any]],
    seed_ids: frozenset[str],
    question_id: str | None,
) -> MemoryReadResult:
    """
    把读到的条目排序、按条数上限截断、投影成读取成功的返回。

    参数:
        rows: 过门条目，含一跳邻居
        seed_ids: 收窄直接命中的条目 id，用于判断能不能回填归属问题
        question_id: 按问题收窄时的问题 id，未按问题收窄为 None
    返回:
        status=success 的 MemoryReadResult，excluded_counts 记录条数上限截掉了多少条
    """
    max_items = get_settings().research_memory_inject_max_items
    # 1. 稳定排序：仓储已按 updated_at DESC 返回，按成熟度排完，同一档内仍是最近更新在前。
    #    条数不够时先掉直觉级，而不是先掉刚写下的判断
    ordered = sorted(rows, key=_maturity_rank)
    items = [_project_read_item(row, seed_ids, question_id) for row in ordered[:max_items]]
    return {"status": "success", "items": items, "excluded_counts": _excluded_counts(len(ordered), len(items))}


def _project_read_item(
    row: dict[str, Any],
    seed_ids: frozenset[str],
    question_id: str | None,
) -> MemoryReadItem:
    """
    投影单条记忆，给按问题收窄直接命中的条目补上归属问题。

    参数:
        row: 过门的记忆条目行
        seed_ids: 收窄直接命中的条目 id
        question_id: 按问题收窄时的问题 id，未按问题收窄为 None
    返回:
        MemoryReadItem；只在归属确凿时才带 question_ids
    """
    item = _project_item(row)
    if not question_id or row["memory_id"] not in seed_ids:
        # 归属问题只在能确证时才写：一跳邻居只是与种子有关系，未必挂在同一个研究问题下，
        # 给它填这个问题 id 就是编造归属，Planner 会照着假归属组织回答
        return item
    # 种子是按 question_id 关联查出来的，归属由那次 join 保证，不必再回表确认
    return {**item, "question_ids": [question_id]}
