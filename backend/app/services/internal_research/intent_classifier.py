"""内研意图识别：主链上唯一一处意图判断。

把本轮问题按关键词归到 lookup / compare / inventory / explore 四档，供焦点解析聚焦降级链、
处理深度判档与回答简报共同消费，决定「这一轮计划怎么做」。
"""

from __future__ import annotations

import re

from app.models.research_contracts import Intent, IntentResult

_INVENTORY_PATTERNS = re.compile(
    r"(列出|都有哪些|清单|有多少篇|哪些论文|inventory|list\s+all|what\s+papers)",
    re.IGNORECASE,
)
_COMPARE_PATTERNS = re.compile(
    r"(对比|比较|差异|不同|区别|相似|vs\.?|versus|differ|compar|contrast)",
    re.IGNORECASE,
)
_LOOKUP_PATTERNS = re.compile(
    r"(什么是|是什么|是多少|是怎么|做什么|了什么|定义|提出了|解释下|怎么做|方法是|原文|具体|define|explain|what\s+is|how\s+does)",
    re.IGNORECASE,
)
_LOOKUP_ENDING = re.compile(
    r"(是什么|是多少|是怎么|做什么|了什么|写了什么|用了什么|哪些方法|哪些.{0,4}$)[？?。]?\s*$",
    re.IGNORECASE,
)

# 没有任何关键词命中时按 explore 处理，置信度刻意压到最低档，表达「这一轮更开放」
_DEFAULT_EXPLORE_CONFIDENCE = 0.5
# 多个类别关键词同时命中说明口径互相冲突，压低置信度但不改变首关键词判定
_CONFLICTING_SIGNAL_CONFIDENCE = 0.5
_CONFLICTING_SIGNAL_MIN_COUNT = 2


def _coarse_intent_from_rules(question: str) -> tuple[Intent, float, list[str]]:
    """按关键词顺序判定内部研究意图。

    优先级为清单 > 查询 > 对比 > 开放探索：清单类问题往往是「把库里的东西全部摆出来」，
    查询类只要带上结尾式提问就不再被对比关键词抢走，剩下的才落到对比与探索。

    Args:
        question: 本轮用户问题原文
    Returns:
        tuple[Intent, float, list[str]]: 意图、置信度与命中信号列表
    """
    signals: list[str] = []
    intent: Intent = "explore"
    confidence = 0.8
    has_lookup = bool(_LOOKUP_PATTERNS.search(question) or _LOOKUP_ENDING.search(question))
    has_compare = bool(_COMPARE_PATTERNS.search(question))
    has_inventory = bool(_INVENTORY_PATTERNS.search(question))

    # 1. 按关键词优先级定档，先判清单再判查询，最后才判对比
    if has_inventory:
        signals.append("inventory_keyword")
        intent = "inventory"
        confidence = 0.9
    elif has_lookup and (not has_compare or _LOOKUP_ENDING.search(question)):
        signals.append("lookup_keyword")
        if _LOOKUP_ENDING.search(question):
            signals.append("lookup_ending")
        intent = "lookup"
        confidence = 0.85
    elif has_compare:
        signals.append("compare_keyword")
        intent = "compare"
        confidence = 0.85
    else:
        signals.append("default_explore")
        confidence = _DEFAULT_EXPLORE_CONFIDENCE

    # 2. 两类以上关键词同时出现时压低置信度，暴露口径冲突而不是硬选一个
    conflicting = sum([
        bool(_INVENTORY_PATTERNS.search(question)),
        bool(_COMPARE_PATTERNS.search(question)),
        bool(_LOOKUP_PATTERNS.search(question)),
    ])
    if conflicting >= _CONFLICTING_SIGNAL_MIN_COUNT:
        confidence = min(confidence, _CONFLICTING_SIGNAL_CONFIDENCE)
        signals.append("conflicting_signals")
    return intent, confidence, signals


def classify_intent(question: str) -> IntentResult:
    """分类本轮问题的内部研究意图，是主链唯一一处意图判断。

    Args:
        question: 本轮用户问题原文
    Returns:
        IntentResult: 含 intent / confidence / source / signals 的意图识别结果
    """
    intent, confidence, signals = _coarse_intent_from_rules(question)
    return IntentResult(
        intent=intent,
        confidence=confidence,
        source="rule",
        signals=signals,
    )
