"""
主图入口前的安全护栏节点。

这个模块位于顶层主图的前置检查阶段，只做输入安全底线校验：
  - 拒绝空输入与超长输入
  - 拒绝明显高风险/敏感主题

需要特别区分：
  - 它不再限制"必须是学术问题"，日常提问也放行
  - 它不是 RAG 内部的质量评估层
  - 它不是文档过滤器
  - 它属于"输入安全校验"，发生在主图真正分流前
"""

from __future__ import annotations

from app.api.schemas.request_limits import MAX_QUERY_LENGTH
from app.config import get_settings

import logging

logger = logging.getLogger(__name__)

# 保留可 monkeypatch 的模块级入口，默认延迟初始化。
settings = None


def _settings():
    """按需读取配置，避免模块导入阶段触发全局初始化。"""
    return settings or get_settings()

# 明显不应处理的高风险/敏感主题关键词。
SENSITIVE_KEYWORDS = (
    "暴力袭击",
    "仇恨言论",
    "恐怖袭击",
    "制作炸弹",
)


def _normalize_query(query: str) -> str:
    """把查询统一成去首尾空白、压缩空格、转小写的比较形态。"""
    return " ".join((query or "").strip().lower().split())


def _contains_any_keyword(query: str, keywords: tuple[str, ...]) -> bool:
    """判断标准化后的查询是否命中任意关键词。"""
    normalized = _normalize_query(query)
    return any(keyword in normalized for keyword in keywords)


def _build_accept_state(reason: str) -> dict:
    """构造放行状态更新。"""
    return {"guardrails_action": "accept", "guardrails_reason": reason}


def _build_reject_state(reason: str) -> dict:
    """构造拒绝状态更新，附带统一拒绝报告文案 + source_status 全 skipped。"""
    return {
        "guardrails_action": "reject",
        "guardrails_reason": reason,
        "report": _settings().rag_guardrails_rejection_template,
        "source_status": {
            "milvus": "skipped",
            "graph": "skipped",
            "neo4j": "skipped",
            "inventory": "skipped",
        },
    }


async def guardrails_node(state: dict) -> dict:
    """
    在进入主图路由前做输入安全底线校验。

    只拦空输入、超长输入和高风险敏感主题，其余（含日常非学术提问）一律放行；
    这一步不参与 RAG 检索、答案生成或后置评估。
    """
    query = (state.get("user_query") or "").strip()
    if not query:
        return _build_reject_state("empty_query")
    if len(query) > MAX_QUERY_LENGTH:
        return _build_reject_state("query_too_long")
    if _contains_any_keyword(query, SENSITIVE_KEYWORDS):
        return _build_reject_state("blocked_sensitive_topic")
    return _build_accept_state("passed_safety_baseline")
