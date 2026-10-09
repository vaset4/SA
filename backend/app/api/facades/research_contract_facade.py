"""研究入口合同辅助。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.agents.agent import INTENT_EXTERNAL_RESEARCH, INTENT_INTERNAL_RESEARCH
from app.models.research_contracts import (
    DIALOGUE_MODE_CHAT,
    DIALOGUE_MODE_LOOKUP,
    DIALOGUE_MODE_RESEARCH,
    ENTRY_CONVERSATION,
    ENTRY_PAPER_FOCUS,
    ENTRY_PROJECT_SCOPE,
    ENTRY_WEB_SCOPE,
    VALID_DIALOGUE_MODES,
    default_evidence_layer_roles,
)
from app.services.session_mapping_service import build_session_identity

INTERNAL_RESEARCH_MODE = "internal"
EXTERNAL_RESEARCH_MODE = "external"
PROJECT_SCOPE_ENTRY = ENTRY_PROJECT_SCOPE
WEB_SCOPE_ENTRY = ENTRY_WEB_SCOPE

_RESEARCH_HINTS = (
    "贡献", "方法", "实验", "结果", "局限", "对比", "比较", "差异", "综述", "总结", "演化",
    "contribution", "method", "experiment", "result", "limitation", "compare", "difference", "overview", "survey",
)


class SessionBridgeResult(BaseModel):
    session_id: str | None = None
    conversation_id: str | None = None
    thread_id: str | None = None


def resolve_core_research_intent(project_id: str | None, paper_ids: list[str] | None = None) -> str:
    """
    根据是否绑定项目范围，决定主图应走内部研究还是外部研究意图。

    Args:
        project_id: 当前请求绑定的项目 ID
        paper_ids: 当前请求已完成归属校验的论文 ID 列表

    Returns:
        str: `internal_research` 或 `external_research`
    """
    has_project = bool(str(project_id or "").strip())
    has_papers = bool(_normalize_paper_ids(paper_ids))
    return INTENT_INTERNAL_RESEARCH if has_project or has_papers else INTENT_EXTERNAL_RESEARCH


def normalize_research_mode(project_id: str | None, paper_ids: list[str] | None = None) -> str:
    """
    归一化当前研究模式。

    Args:
        project_id: 当前请求绑定的项目 ID
        paper_ids: 当前请求已完成归属校验的论文 ID 列表

    Returns:
        str: `internal` 或 `external`
    """
    return (
        INTERNAL_RESEARCH_MODE
        if str(project_id or "").strip() or _normalize_paper_ids(paper_ids)
        else EXTERNAL_RESEARCH_MODE
    )


def _normalize_paper_ids(paper_ids: list[str] | None) -> list[str]:
    return [str(item).strip() for item in (paper_ids or []) if str(item).strip()]


def build_project_scope_contract(project_context: dict[str, Any]) -> dict[str, Any]:
    """
    构造项目级长期研究边界合同。

    Args:
        project_context: build_project_context 收敛后的项目范围信息

    Returns:
        dict[str, Any]: 可被 route、graph、frontend 共同消费的项目范围合同
    """
    project_id = str(project_context.get("project_id") or "").strip()
    paper_ids = _normalize_paper_ids(project_context.get("paper_ids"))
    resolved_scope = project_context.get("resolved_scope")
    resolved_library_ids = resolved_scope.get("library_ids", []) if isinstance(resolved_scope, dict) else []
    library_ids = _normalize_paper_ids(project_context.get("library_ids") or resolved_library_ids)
    user_id = str(project_context.get("user_id") or "").strip()
    local_retrieval_allowed = bool(
        user_id
        and paper_ids
        and project_context.get("local_retrieval_allowed")
    )
    # 1. 先归一化 project_id 和 paper_ids，明确当前请求到底有没有项目边界
    if not project_id:
        summary = "当前问题未绑定项目范围。"
    elif paper_ids:
        summary = f"当前项目范围覆盖 {len(paper_ids)} 篇论文。"
    else:
        summary = "当前问题已绑定项目，但项目范围内尚未收敛到具体论文。"
    # 2. 再把 owner、论文、过滤表达式和本地检索开关收敛成统一合同
    return {
        "user_id": user_id,
        "project_id": project_id,
        "library_ids": library_ids,
        "paper_ids": paper_ids,
        "filter_expr": str(project_context.get("filter_expr") or "").strip(),
        "local_retrieval_allowed": local_retrieval_allowed,
        **({"publication": project_context["publication"]} if "publication" in project_context else {}),
        "resolved_scope": resolved_scope,
        "has_project_scope": bool(project_id),
        "summary": summary,
    }


def build_paper_focus_contract(
    project_context: dict[str, Any],
    *,
    explicit_paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    构造当前问题的临时论文焦点合同。

    Args:
        project_context: build_project_context 收敛后的项目范围信息
        explicit_paper_ids: 用户本轮显式指定的论文 ID 列表

    Returns:
        dict[str, Any]: 表示当前问题本轮焦点的临时论文合同
    """
    explicit_ids = _normalize_paper_ids(explicit_paper_ids)
    scoped_ids = _normalize_paper_ids(project_context.get("paper_ids"))
    # 1. 优先尊重本轮显式 paper_ids，因为它代表用户这次问题的直接焦点
    if explicit_ids:
        return {
            "mode": "single_paper" if len(explicit_ids) == 1 else "multi_paper",
            "paper_ids": explicit_ids,
            "is_explicit": True,
            "source": ENTRY_PAPER_FOCUS,
            "summary": "当前问题显式聚焦单篇论文。" if len(explicit_ids) == 1 else f"当前问题显式聚焦 {len(explicit_ids)} 篇论文。",
        }
    # 2. 没有显式焦点时，退回当前项目/对话范围里的默认候选论文集
    if scoped_ids:
        return {
            "mode": "project_scope",
            "paper_ids": scoped_ids,
            "is_explicit": False,
            "source": ENTRY_PROJECT_SCOPE if str(project_context.get("project_id") or "").strip() else ENTRY_CONVERSATION,
            "summary": f"当前问题默认使用范围内的 {len(scoped_ids)} 篇论文作为候选焦点。",
        }
    # 3. 两者都没有时，明确标记为 web_scope / none，避免下游误判成局部研究
    return {
        "mode": "none",
        "paper_ids": [],
        "is_explicit": False,
        "source": ENTRY_WEB_SCOPE,
        "summary": "当前问题没有显式论文焦点。",
    }


def resolve_dialogue_mode(
    query: str,
    *,
    explicit_mode: str | None = None,
    project_id: str | None = None,
    paper_ids: list[str] | None = None,
) -> str:
    """
    根据当前问题和范围推断对话模式。

    Args:
        query: 用户问题
        explicit_mode: 前端显式指定的对话模式
        project_id: 当前请求绑定的项目 ID
        paper_ids: 当前请求显式或隐式锁定的论文范围

    Returns:
        str: `chat` / `lookup` / `research`
    """
    normalized_mode = str(explicit_mode or "").strip().lower()
    if normalized_mode in VALID_DIALOGUE_MODES:
        return normalized_mode
    normalized_query = str(query or "").strip().lower()
    scoped_ids = _normalize_paper_ids(paper_ids)
    if len(scoped_ids) > 1:
        return DIALOGUE_MODE_RESEARCH
    if len(scoped_ids) == 1:
        return DIALOGUE_MODE_LOOKUP
    if str(project_id or "").strip() and any(hint in normalized_query for hint in _RESEARCH_HINTS):
        return DIALOGUE_MODE_RESEARCH
    return DIALOGUE_MODE_CHAT


def build_dialogue_structured_query(
    *,
    query: str,
    project_context: dict[str, Any],
    project_id: str | None,
    explicit_paper_ids: list[str] | None = None,
    dialogue_mode: str | None = None,
    research_mode: str | None = None,
) -> dict[str, Any]:
    """
    统一构造内部对话入口传给主图的结构化上下文。

    Args:
        query: 用户问题
        project_context: build_project_context 收敛后的项目范围信息
        project_id: 当前请求绑定的项目 ID
        explicit_paper_ids: 用户本轮显式指定的论文 ID 列表
        dialogue_mode: 前端显式指定的对话模式
        research_mode: 前端显式指定的研究模式

    Returns:
        dict[str, Any]: 主图后续节点统一消费的结构化上下文
    """
    effective_paper_ids = explicit_paper_ids or project_context.get("paper_ids")
    # 1. 先确定当前问题最终采用的 dialogue_mode
    mode = resolve_dialogue_mode(
        query,
        explicit_mode=dialogue_mode,
        project_id=project_id,
        paper_ids=effective_paper_ids,
    )
    # 2. 再把长期 project_scope 和本轮 paper_focus 同时收敛出来
    project_scope = build_project_scope_contract(project_context)
    paper_focus = build_paper_focus_contract(project_context, explicit_paper_ids=explicit_paper_ids)
    effective_research_mode = research_mode or normalize_research_mode(
        project_id,
        paper_ids=effective_paper_ids,
    )
    # 3. 最后整理成主图可以直接消费的 structured_query
    return {
        "project_id": project_context["project_id"],
        "resolved_scope": project_context["resolved_scope"],
        "research_mode": effective_research_mode,
        "dialogue_mode": mode,
        "paper_ids": paper_focus["paper_ids"] or project_scope["paper_ids"],
        "project_scope": project_scope,
        "paper_focus": paper_focus,
        "entry_type": paper_focus["source"],
        "evidence_layers": default_evidence_layer_roles(),
    }


def build_research_structured_query(
    project_context: dict[str, Any],
    *,
    project_id: str | None,
    query: str = "",
    paper_ids: list[str] | None = None,
    dialogue_mode: str | None = None,
) -> dict[str, Any]:
    """
    统一构造研究入口传给主图的结构化上下文。

    Args:
        project_context: build_project_context 收敛后的项目范围信息
        project_id: 当前请求绑定的项目 ID
        query: 用户问题
        paper_ids: 用户显式指定的论文 ID 列表
        dialogue_mode: 前端显式指定的对话模式

    Returns:
        dict[str, Any]: 研究入口专用的 structured_query
    """
    effective_paper_ids = paper_ids or project_context.get("paper_ids")
    return build_dialogue_structured_query(
        query=query,
        project_context=project_context,
        project_id=project_id,
        explicit_paper_ids=paper_ids,
        dialogue_mode=dialogue_mode,
        research_mode=normalize_research_mode(project_id, paper_ids=effective_paper_ids),
    )


def resolve_session_bridge(
    *,
    session_id: str | None,
    conversation_id: str | None,
) -> SessionBridgeResult:
    """
    把 session / conversation / thread 的语义边界统一收敛出来。

    Args:
        session_id: LangGraph 会话 ID
        conversation_id: 前端展示层对话 ID

    Returns:
        SessionBridgeResult: 包含 session_id / conversation_id / thread_id 的统一桥接结果
    """
    if not str(session_id or "").strip():
        return SessionBridgeResult(
            session_id=None,
            conversation_id=str(conversation_id or "").strip() or None,
            thread_id=None,
        )
    payload = build_session_identity(session_id, conversation_id)
    return SessionBridgeResult(
        session_id=payload.get("session_id"),
        conversation_id=payload.get("conversation_id"),
        thread_id=payload.get("thread_id"),
    )
