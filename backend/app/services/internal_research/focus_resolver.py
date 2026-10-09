"""焦点解析：三层焦点判定 + 软追问降级。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.models.research_contracts import FocusResolution

logger = logging.getLogger(__name__)

_CONVERSATION_SCOPE_WINDOW = 6

# 用户明确指了某几篇、但后端规则解不出来时的原因码。它与 lookup_no_focus 一类
# 「本轮压根没指」必须分开记：一个说明指代规则覆盖不到，一个说明用户没收窄，
# 排查方向完全不同（prd.md A6：记「没解析出」而不是「没收窄」）
FOCUS_REFERENCE_UNRESOLVED = "reference_unresolved"

# 序数指代的顺序基准是本轮清单顺序，最小可解的篇号
_FIRST_ORDINAL = 1

# 「第X篇」「前X篇」「这X篇」里的 X，中文数字与阿拉伯数字都收
_COUNT_TOKEN = r"([0-9]+|[一二三四五六七八九十两]+)"

# 落到具体某一篇：第一篇 / 第 2 篇
_ORDINAL_PATTERN = re.compile(rf"第\s*{_COUNT_TOKEN}\s*篇")

# 落到开头连续若干篇：前两篇 / 前 3 篇 / 这两篇。「这X篇」按当前清单开头取，
# 与「前X篇」同一口径——本轮清单就是用户眼前那几篇
_LEADING_PATTERNS = (
    re.compile(rf"前\s*{_COUNT_TOKEN}\s*篇"),
    re.compile(rf"这\s*{_COUNT_TOKEN}\s*篇"),
)

# 后端规则不解的指代。「倒数第二篇」必须在「第二篇」之前拦掉，否则会被正着数解成第 2 篇；
# 「刚才那篇」依赖会话指代，不是清单序号能定位的
_UNRESOLVABLE_PATTERNS = (
    re.compile(r"倒数"),
    re.compile(r"最后\s*[一1]?\s*篇"),
    re.compile(r"刚才\s*(提到\s*的)?\s*那?\s*篇"),
)

# 指代必带的量词，没有它就不进指代解析，避免在普通问句上误判
_PAPER_QUANTIFIER = "篇"

_CN_DIGITS: dict[str, int] = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

# 中文「十」的进位基数，用于解析十、十一、二十这类写法
_CN_TEN = 10

# compare 意图至少要两篇才谈得上比较，不足就先补会话层、再退回项目全量
_COMPARE_MIN_PAPERS = 2


@dataclass(frozen=True)
class _FocusContext:
    """按意图降级时要用到的本轮上下文，避免降级函数摊开一长串位置参数。"""

    intent: str
    project_scope: list[str]
    conversation_scope: list[str]
    previous_focus: FocusResolution | None


def resolve_focus(
    question: str,
    *,
    intent: str,
    project_paper_ids: list[str] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    previous_focus: FocusResolution | None = None,
    papers: Sequence[Mapping[str, Any]] | None = None,
) -> FocusResolution:
    """
    解析本轮内部研究的三层焦点。

    三层焦点分别是：
    1. `project_scope`：当前项目或范围内允许访问的论文集合
    2. `conversation_scope`：最近几轮对话里实际提到过的论文集合
    3. `turn_focus`：本轮问题真正想落到哪些论文上

    Args:
        question: 用户本轮问题
        intent: classify 后的意图，例如 lookup / compare / explore
        project_paper_ids: 当前项目范围内的论文集合，顺序即本轮清单顺序，
            序数指代按它数
        conversation_history: 最近对话历史
        previous_focus: 上一轮显式焦点
        papers: 本轮清单的论文记录（含 paper_id / title），顺序与
            project_paper_ids 同源；用于按标题点名收窄，缺省时只解序数指代

    Returns:
        FocusResolution: 收敛后的三层焦点结果
    """
    project_scope = list(project_paper_ids or [])
    conversation_scope = _extract_conversation_scope(conversation_history)
    turn_focus, reference_reason = _extract_turn_focus(question, project_scope, papers=papers)
    # 1. 先按意图走既有降级：指代解不出来也要经过这一步，会话层补充不能被跳过
    turn_focus, fallback_reason = _apply_intent_fallback(
        turn_focus,
        _FocusContext(
            intent=intent,
            project_scope=project_scope,
            conversation_scope=conversation_scope,
            previous_focus=previous_focus,
        ),
    )
    # 2. 用户确实指了某几篇却解不出来：原因如实记成「没解析出」，盖掉上一步那些
    #    只说明「本轮没收窄」的原因码。两者排查方向不同，不能混成一个
    if reference_reason:
        turn_focus = turn_focus or project_scope[:]
        fallback_reason = reference_reason

    return FocusResolution(
        project_scope=project_scope,
        conversation_scope=conversation_scope,
        turn_focus=turn_focus,
        fallback_reason=fallback_reason,
    )


def _apply_intent_fallback(
    turn_focus: list[str],
    context: _FocusContext,
) -> tuple[list[str], str]:
    """
    本轮焦点不足时按意图决定退回到哪一层，并记下退回原因。

    Args:
        turn_focus: 从问题文本里解析出的焦点集合，可能为空
        context: 本轮意图、项目范围、会话范围与上一轮焦点

    Returns:
        tuple[list[str], str]: 退回后的焦点集合与退回原因码；没有触发退回时原因码为空串
    """
    intent = context.intent
    project_scope = context.project_scope
    # 1. explore 没点名任何论文时就是要看全局，退回项目全量是正常路径
    if intent == "explore" and not turn_focus:
        return project_scope[:], "explore_default"
    # 2. compare 焦点不足时先从最近对话补足，补不够才退回项目全量。
    #    补充只能取授权范围内的候选：历史消息里的论文 ID 来自客户端，不是授权凭据
    if intent == "compare" and len(turn_focus) < _COMPARE_MIN_PAPERS:
        supplemented = _supplement_from_conversation(turn_focus, context.conversation_scope)
        supplemented = _narrow_to_authorized(supplemented, project_scope)
        if len(supplemented) >= _COMPARE_MIN_PAPERS:
            return supplemented, ""
        return project_scope[:], "compare_insufficient_focus"
    # 3. lookup 缺焦点时先承接上一轮显式焦点，没有就退回项目全量。
    #    上一轮焦点同样只能收窄：本轮授权已缩小后不能再把它带出来
    if intent == "lookup" and not turn_focus:
        previous_focus = context.previous_focus
        if previous_focus and previous_focus.get("turn_focus"):
            carried = _narrow_to_authorized(previous_focus["turn_focus"], project_scope)
            if carried:
                return carried, ""
        return project_scope[:], "lookup_no_focus"
    # 4. 其余路径的焦点直接来自问题文本，只在 `_match_by_title` / `_match_by_ordinal`
    #    内部按清单匹配，天然是授权范围的子集，这里不再求交一遍
    return turn_focus, ""


def _narrow_to_authorized(candidates: list[str], project_scope: list[str]) -> list[str]:
    """
    把候选焦点收窄到本轮授权范围内。

    历史消息与上一轮焦点都只是候选输入，不是授权凭据：范围外的候选一律丢弃，
    否则焦点会顺下游变成取证目标，读到本轮授权之外的论文。这里只做收窄，不放宽。

    Args:
        candidates: 候选焦点论文 ID，顺序即本轮清单顺序
        project_scope: 本轮授权论文集合，服务端按归属核对过的真相源

    Returns:
        list[str]: 候选里属于授权范围的子序列，保持原顺序；候选全在范围外时为空列表
    """
    authorized = set(project_scope)
    kept = [paper_id for paper_id in candidates if paper_id in authorized]
    dropped = len(candidates) - len(kept)
    # 1. 【焦点解析】【授权收窄】丢弃候选说明有输入试图扩大范围，如实记一条便于排查
    if dropped:
        logger.info(
            f"【焦点解析】【授权收窄】丢弃 {dropped} 个授权范围外的候选论文，"
            f"本轮授权 {len(authorized)} 篇，候选原文：{candidates}"
        )
    return kept


def _extract_conversation_scope(
    conversation_history: list[dict[str, Any]] | None,
) -> list[str]:
    """
    从最近 N 轮对话中聚合涉及的论文 ID。

    Args:
        conversation_history: 最近对话历史

    Returns:
        list[str]: 最近窗口内提到过的论文 ID 去重列表
    """
    if not conversation_history:
        return []
    recent = conversation_history[-_CONVERSATION_SCOPE_WINDOW:]
    paper_ids: list[str] = []
    for msg in recent:
        for pid in msg.get("paper_ids", []):
            pid_str = str(pid).strip()
            if pid_str and pid_str not in paper_ids:
                paper_ids.append(pid_str)
    return paper_ids


def _extract_turn_focus(
    question: str,
    project_scope: list[str],
    *,
    papers: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[str], str]:
    """
    从问题文本中解析本轮真正指向的论文。

    先按标题点名，再按序数指代（顺序即本轮清单顺序）。两者都不成立时返回空集，
    由调用方按意图决定怎么退回。

    Args:
        question: 用户本轮问题
        project_scope: 当前允许访问的论文范围，顺序即本轮清单顺序
        papers: 本轮清单的论文记录，用于标题匹配

    Returns:
        tuple[list[str], str]: 命中的论文 ID 列表，以及指代解析失败的原因码；
            没有出现指代或指代解析成功时原因码为空串
    """
    # 1. 【焦点解析】【授权收窄】标题点名与序数指代都只在授权范围内解，
    #    传全量给匹配函数、事后再求交会退化成「序号按全量数、结果再筛掉」，序号会指错人
    if not project_scope:
        return [], ""
    # 2. 标题点名优先：用户写出标题时意图最明确，不必再猜序号
    by_title = _match_by_title(question, project_scope, papers=papers)
    if by_title:
        return by_title, ""

    return _match_by_ordinal(question, project_scope)


def _match_by_title(
    question: str,
    project_scope: list[str],
    *,
    papers: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    """
    按论文标题在问题里点名匹配，保持清单顺序。

    Args:
        question: 用户本轮问题
        project_scope: 当前允许访问的论文范围，用于过滤越权 ID
        papers: 本轮清单的论文记录，取 paper_id 与 title

    Returns:
        list[str]: 标题出现在问题里的论文 ID 列表；无记录或无命中时为空列表
    """
    if not papers:
        return []
    allowed = set(project_scope)
    normalized_question = _normalize(question)
    matched: list[str] = []
    for record in papers:
        paper_id = str(record.get("paper_id") or "").strip()
        title = _normalize(str(record.get("title") or ""))
        if not paper_id or not title or paper_id not in allowed:
            continue
        if title in normalized_question and paper_id not in matched:
            matched.append(paper_id)
    return matched


def _match_by_ordinal(question: str, project_scope: list[str]) -> tuple[list[str], str]:
    """
    按序数指代匹配论文，顺序基准是传入的清单顺序。

    Args:
        question: 用户本轮问题
        project_scope: 当前允许访问的论文范围，顺序即本轮清单顺序

    Returns:
        tuple[list[str], str]: 命中的论文 ID 列表与失败原因码；
            问题里没有篇数指代时两者都为空
    """
    if _PAPER_QUANTIFIER not in question:
        return [], ""

    # 1. 先拦不解的说法：「倒数第二篇」正着数会解成第 2 篇，必须早于序数匹配
    if any(pattern.search(question) for pattern in _UNRESOLVABLE_PATTERNS):
        logger.info(f"【焦点解析】【序数指代】命中不支持的指代说法，退回项目全量：{question}")
        return [], FOCUS_REFERENCE_UNRESOLVED

    # 2. 【焦点解析】【序数指代】同一句里的「前X篇」「这X篇」「第X篇」一起收下，
    #    按各指代在问题里出现的先后拼接。谁先命中谁算数会让「对比前两篇与第三篇」
    #    这类句子只出前两篇，用户点过的第三篇被整句忽略
    references = _collect_references(question)
    if not references:
        return [], ""

    matched: list[str] = []
    for is_leading, token in references:
        if is_leading:
            resolved, reason = _take_leading(token, project_scope)
        else:
            resolved, reason = _take_positions([token], project_scope)
        # 3. 任一条指代越界或解不出，整句都算没解析出：用户点了三篇只答两篇，
        #    比如实说「没解析出」更难被发现
        if reason:
            return [], reason
        # 4. 多条指代重复点到同一篇时只保留首次出现的位置
        for paper_id in resolved:
            if paper_id not in matched:
                matched.append(paper_id)
    return matched, ""


def _collect_references(question: str) -> list[tuple[bool, str]]:
    """
    收集一句里所有已支持的篇数指代，按出现先后排列。

    Args:
        question: 用户本轮问题

    Returns:
        list[tuple[bool, str]]: 每项为（是否「前X篇」「这X篇」的开头连续形式, 篇数原文），
            按指代在问题里的起始位置升序；没有指代时为空列表
    """
    hits: list[tuple[int, bool, str]] = []
    for pattern in _LEADING_PATTERNS:
        hits.extend((match.start(), True, match.group(1)) for match in pattern.finditer(question))
    hits.extend((match.start(), False, match.group(1)) for match in _ORDINAL_PATTERN.finditer(question))
    hits.sort(key=lambda hit: hit[0])
    return [(is_leading, token) for _, is_leading, token in hits]


def _take_leading(token: str, project_scope: list[str]) -> tuple[list[str], str]:
    """
    取清单开头连续 N 篇，N 超出清单长度时视为解析失败。

    Args:
        token: 「前X篇」里的 X 原文
        project_scope: 当前允许访问的论文范围

    Returns:
        tuple[list[str], str]: 开头 N 篇的 ID 列表与失败原因码
    """
    count = _parse_count(token)
    if count is None or count < _FIRST_ORDINAL or count > len(project_scope):
        logger.info(f"【焦点解析】【序数指代】篇数 {token} 超出本轮清单 {len(project_scope)} 篇，退回项目全量")
        return [], FOCUS_REFERENCE_UNRESOLVED
    return project_scope[:count], ""


def _take_positions(tokens: list[str], project_scope: list[str]) -> tuple[list[str], str]:
    """
    取清单里点到的第 N 篇，按清单顺序返回。

    只要有一个序号解不出或越界，整句指代都算没解析出：用户点了两篇却只答其中一篇，
    比如实记成「没解析出」更难被发现。

    Args:
        tokens: 句中所有「第X篇」里的 X 原文，按出现顺序
        project_scope: 当前允许访问的论文范围，顺序即本轮清单顺序

    Returns:
        tuple[list[str], str]: 命中论文的 ID 列表与失败原因码
    """
    positions: list[int] = []
    for token in tokens:
        position = _parse_count(token)
        if position is None or position < _FIRST_ORDINAL or position > len(project_scope):
            logger.info(f"【焦点解析】【序数指代】第 {token} 篇越出本轮清单 {len(project_scope)} 篇，退回项目全量")
            return [], FOCUS_REFERENCE_UNRESOLVED
        if position not in positions:
            positions.append(position)
    return [project_scope[position - _FIRST_ORDINAL] for position in sorted(positions)], ""


def _parse_count(token: str) -> int | None:
    """
    把「2」「二」「十一」这类篇数写法解析成整数。

    Args:
        token: 指代里的篇数原文

    Returns:
        int | None: 解析出的篇数；写法不认识时返回 None
    """
    text = token.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return _parse_chinese_number(text)


def _parse_chinese_number(text: str) -> int | None:
    """
    解析中文数字，覆盖一到九、十、十几与几十几。

    Args:
        text: 中文数字原文

    Returns:
        int | None: 解析出的数值；含无法识别的字符时返回 None
    """
    if "十" in text:
        head, _, tail = text.partition("十")
        high = _CN_DIGITS.get(head, 1) if head else 1
        low = _CN_DIGITS.get(tail, 0) if tail else 0
        if (head and head not in _CN_DIGITS) or (tail and tail not in _CN_DIGITS):
            return None
        return high * _CN_TEN + low
    if len(text) == 1 and text in _CN_DIGITS:
        return _CN_DIGITS[text]
    return None


def _normalize(text: str) -> str:
    """
    归一化文本用于标题匹配：转小写并去掉空白。

    Args:
        text: 原始文本

    Returns:
        str: 归一化后的文本
    """
    return re.sub(r"\s+", "", text).lower()


def _supplement_from_conversation(
    current: list[str], conversation_scope: list[str]
) -> list[str]:
    """
    用最近对话范围补充不足的焦点集合。

    Args:
        current: 当前已识别出的焦点论文集合
        conversation_scope: 最近对话里出现过的论文集合

    Returns:
        list[str]: 补齐后的焦点集合
    """
    result = current[:]
    for pid in conversation_scope:
        if pid not in result:
            result.append(pid)
        if len(result) >= 2:
            break
    return result
