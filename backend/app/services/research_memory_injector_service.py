"""
项目研究记忆层 · 记忆读取动作的加载与渲染实现。

Planner 循环内按需读记忆时，从 Postgres 取出四道资格门都满足的判断，渲染成一段
可注入 prompt 的文本。这是记忆喂回 LLM 的只读通路：

- 只读 Postgres，不走向量召回。
- 只读记忆库拼 prompt 文本，绝不改 thread_state、不改检索路径代码。
- 四道门（当前用户 + 当前项目 + 已接纳 accepted + 已确认 confirmed + 新鲜 fresh）
  写死在共享查询函数 `_query_gated_items` 体内，调用方只能收窄不能放宽：不合格的记忆
  对 Planner 是「不存在」，不是「低分」，stale / needs_rebind 直接排除而不以降权方式混入。
- 读取终态分三种返回：读到条目 success、项目内确实没有合格记忆 empty、记忆存储
  不可用 failed。empty 与 failed 不归并，调用方要能分辨该不该重试、该不该提示用户。

三档成熟度全部有资格进上下文，maturity 只决定身份标签与预算截断次序：
established 先于 leaning 先于 intuition，预算不够时先掉直觉级；
按字符预算截断，超长显式标「已截断」。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.config import get_settings
from app.models.research_memory_contracts import MemoryReadItem, MemoryReadResult
from app.repositories.research_memory_repository import get_research_memory_repository

logger = logging.getLogger(__name__)

# 注入次序：三档成熟度都有资格进上下文，位次只决定预算不够时先掉哪一档
_MATURITY_PRIORITY = {"established": 0, "leaning": 1, "intuition": 2}
# 成熟度取值超出已知三档时排到最后，不占用三档已知成熟度的位次
_UNKNOWN_MATURITY_RANK = 9
# 渲染给 LLM 的身份标签。intuition 取「假设」而不是中性的「直觉」，让模型看清这条
# 判断还没被证据坐实，不会把用户随手记下的猜测与已确立的项目认识同等使用
_MATURITY_LABEL = {"established": "已确立", "leaning": "倾向", "intuition": "假设"}
# 注入 prompt 里单条记忆的正文摘要长度：只给一句话级别的正文提示，全文回记忆页看
_BODY_SUMMARY_CHARS = 80
# 因条数上限没能进上下文的排除原因键
_EXCLUDED_OVER_MAX_ITEMS = "over_max_items"
# 允许透传到仓储的收窄维度白名单。四道资格门不在其中，所以调用方就算把
# review_state / status / stale_state 塞进收窄条件，也到不了 SQL：门只由查询函数说了算
_NARROWING_FILTER_KEYS: tuple[str, ...] = ("paper_id", "question_id", "memory_type")
# 空身份的拒绝理由：点明业务后果，让接线的人立刻看出严重性而不是当成参数格式问题
_BLANK_USER_ID_ERROR = (
    "记忆读取的 user_id 不能为空（收到 {received!r}，project_id={project_id}）："
    "空身份会让仓储跳过 user_id 过滤，于是把该项目下所有人的记忆都读进 LLM 上下文。"
    "这是调用侧的接线错误，不是记忆存储故障，请先解析出真实用户再调用。"
)


def _require_owner(user_id: str, project_id: str) -> None:
    """
    校验记忆归属人非空，空身份直接拒绝而不是落 failed。

    仓储的过滤循环只在取值非 None 时才拼 WHERE，所以空身份传下去等于 user_id 那道门
    整段消失。这属于接线错误：落 failed 会被读成「记忆存储不可用」，接线接错时显示成
    环境问题就被容忍过去了，所以这里抛错，在开发期就把问题顶出来。

    project_id 不在此校验：它走基础 WHERE（i.project_id = $1）恒定生效，不经过那个
    条件过滤循环，空值只会匹配不到行，失败方向是关闭而非放宽。

    参数:
        user_id: 待校验的记忆归属人
        project_id: 归属项目，仅用于报错时定位现场
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(_BLANK_USER_ID_ERROR.format(received=user_id, project_id=project_id))


def _narrowing_kwargs(narrowing: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    从收窄条件里挑出能透传给仓储的过滤参数。

    只认白名单里的三个维度，别的键一律丢掉：这样即便调用方把某道资格门的取值写进
    收窄条件，也拼不进 SQL，「只能收窄不能放宽」不靠调用方自觉。
    取值为 None 的键同样丢掉，让不收窄的读取落在仓储的默认行为上。

    参数:
        narrowing: 调用方给的收窄条件，None 或空字典表示不收窄
    返回:
        可直接展开给 repo.list_items 的关键字参数；不收窄时为空字典
    """
    if not narrowing:
        return {}
    return {key: narrowing[key] for key in _NARROWING_FILTER_KEYS if narrowing.get(key) is not None}


async def _query_gated_items(
    project_id: str,
    user_id: str,
    *,
    narrowing: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    带四道资格门查记忆条目，是全仓唯一一处写门的地方。

    门的取值是本函数体内的字面量，不开参数口子，所以任何调用方都无法放宽：只有
    当前 user_id + 当前 project_id + review_state=accepted + status=confirmed
    + stale_state=fresh 的条目才会回到应用层。收窄条件是在门之上再加限制，
    收窄只会让结果更少，不可能让不合格的条目通过。

    三档成熟度一次查完、不按 maturity 分次查：成熟度都是合格记忆，次序放到进程内排，
    SQL 只负责资格。

    参数:
        project_id: 归属项目，跨项目记忆不互相串
        user_id: 记忆归属人，须由调用方先校验非空
        narrowing: 可选收窄条件（paper_id / question_id / memory_type）
    返回:
        按最近更新倒序的记忆条目行；门放行零条时为空列表
    """
    repo = get_research_memory_repository()
    await repo.ensure_schema()
    rows = await repo.list_items(
        project_id,
        user_id=user_id,
        review_state="accepted",
        status="confirmed",
        stale_state="fresh",
        **_narrowing_kwargs(narrowing),
    )
    from app.services.research_memory_stale_service import checked_memory_items

    return await checked_memory_items(rows, project_id=project_id, user_id=user_id)


async def load_injectable_memory(project_id: str, user_id: str) -> MemoryReadResult:
    """
    读取当前用户在当前项目里可注入 prompt 的记忆。

    四道资格门写死在共享查询函数体内，不开参数口子，所以调用方无法放宽：只有当前
    user_id + 当前 project_id + review_state=accepted + status=confirmed
    + stale_state=fresh 的条目才会回到应用层，「合规率」是结构保证而不是提示词约定。

    参数:
        project_id: 归属项目，跨项目记忆不互相串
        user_id: 记忆归属人，别人的判断不进本人上下文；必须是非空字符串，
                 空身份会让 user_id 那道门失效，所以直接抛 ValueError
    返回:
        MemoryReadResult：读到条目为 success，项目内无合格记忆为 empty，记忆存储
        不可用为 failed 且带 error_kind；items 按成熟度次序排、按条数上限截断
    """
    # 归属人校验必须在 try 之外：try 里的 except Exception 会把它收成 failed，
    # 而接线错误一旦显示成「记忆存储不可用」就会被当成环境问题容忍过去
    _require_owner(user_id, project_id)
    max_items = get_settings().research_memory_inject_max_items
    try:
        # 1. 不给收窄条件：入口注入要的是本项目本人全部合格记忆
        rows = await _query_gated_items(project_id, user_id)
    except Exception as exc:
        # 2. 记忆存储不可用要落 failed：拿空列表冒充成功，调用方就分不清是该重试
        #    还是项目里真没记忆
        logger.error(
            "【研究记忆】【记忆读取】加载可注入记忆失败 project_id=%s user_id=%s",
            project_id,
            user_id,
            exc_info=True,
        )
        return {"status": "failed", "items": [], "excluded_counts": {}, "error_kind": type(exc).__name__}
    if not rows:
        # 3. 门放行的结果为零条：项目内确实没有符合条件的记忆，属正常空结果
        return {"status": "empty", "items": [], "excluded_counts": {}}
    # 4. repo 按 updated_at DESC 返回，Python 的 sort 是稳定排序，所以按成熟度排完，
    #    同一成熟度档内仍保持最近更新在前
    ordered = sorted(rows, key=_maturity_rank)
    items = [_project_item(row) for row in ordered[:max_items]]
    return {"status": "success", "items": items, "excluded_counts": _excluded_counts(len(ordered), len(items))}


def _maturity_rank(row: dict[str, Any]) -> int:
    """
    取一条记忆的注入次序位次。

    参数:
        row: repository 返回的记忆条目行
    返回:
        位次数字，越小越先进上下文
    """
    return _MATURITY_PRIORITY.get(row.get("maturity", ""), _UNKNOWN_MATURITY_RANK)


def _excluded_counts(passed_gates: int, kept: int) -> dict[str, int]:
    """
    统计本次读取真正测得的排除数。

    只有「条数上限截断」这一项能在应用层数出来：四道门写在 SQL WHERE 里，被门挡掉的
    行根本不会回到应用层，所以这里不放 gate_* 之类的计数——数不到的维度宁可没有键，
    也不能给出假数字。

    参数:
        passed_gates: 四道门放行的条目总数
        kept: 实际进上下文的条数
    返回:
        排除原因 → 条数；没发生截断时返回空字典
    """
    dropped = passed_gates - kept
    if dropped <= 0:
        return {}
    return {_EXCLUDED_OVER_MAX_ITEMS: dropped}


def _project_item(row: dict[str, Any]) -> MemoryReadItem:
    """
    把记忆条目行投影成注入所需的最小字段集，正文在此截断成摘要。

    question_ids 故意不填：逐条回表查归属会退化成 N+1 查询。契约声明 total=False，
    缺这个键是正常状态而不是漏填，只有按研究问题收窄的读取动作才会填上它。

    参数:
        row: repository 返回的记忆条目行
    返回:
        MemoryReadItem，身份标签已按成熟度派生
    """
    maturity = row["maturity"]
    body = str(row["body"]).strip()
    return {
        "memory_id": row["memory_id"],
        "version": row["version"],
        "title": str(row["title"]).strip(),
        "body_summary": body[:_BODY_SUMMARY_CHARS],
        "memory_type": row["memory_type"],
        "maturity": maturity,
        "source_kind": row["source_kind"],
        "provenance_label": _MATURITY_LABEL.get(maturity, ""),
        # 四道资格门的取值跟着条目一起出来：门写在 SQL WHERE 里，被挡掉的行根本不回到
        # 应用层，于是「这条凭什么进了上下文」在应用层无从复核。带上归属人与三个状态轴，
        # 合规核查（含评测的注入合规率与失效泄漏率）才有可核对的依据而不是空值
        "user_id": row["user_id"],
        "review_state": row["review_state"],
        "status": row["status"],
        "stale_state": row["stale_state"],
    }


def render_memory_prompt_block(items: list[MemoryReadItem]) -> str:
    """
    把读到的记忆渲染成一段 prompt 文本，按字符预算截断。

    参数:
        items: load_injectable_memory 产出的 items
    返回:
        「项目研究记忆」prompt 段；空列表返回空串（调用方跳过该段）
    """
    if not items:
        return ""
    budget = get_settings().research_memory_inject_max_chars
    # 段首必须交代记忆的身份：这是本项目历史判断、不是论文原文、不能当引用来源，
    # 否则模型会把用户自己的判断当成文献证据写进带引用的结论里
    lines = [
        "【项目研究记忆】（你在本项目已形成的判断，非论文原文，不得作为引用来源；"
        "可用于理解背景与延续思路，需要引用的结论必须回到论文证据）"
    ]
    used = len(lines[0])
    truncated = False
    for item in items:
        line = _render_one(item)
        if used + len(line) > budget:
            truncated = True
            break
        lines.append(line)
        used += len(line)
    if truncated:
        lines.append("（余下记忆因预算已截断）")
    return "\n".join(lines)


def _render_one(item: MemoryReadItem) -> str:
    """
    渲染单条记忆：身份标签 + 标题 + 正文摘要。

    参数:
        item: 投影后的记忆条目，正文摘要已在投影阶段截断，这里不再重复截断
    返回:
        一行「- [身份标签] 标题 — 正文摘要」文本
    """
    label = item.get("provenance_label", "")
    title = item.get("title", "")
    summary = item.get("body_summary", "")
    body_hint = f" — {summary}" if summary else ""
    return f"- [{label}] {title}{body_hint}"
