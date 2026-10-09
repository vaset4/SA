"""检索策略服务。"""

from __future__ import annotations

from dataclasses import dataclass

GRAPH_EVIDENCE_HINTS = ("关系", "联系", "关联", "compare", "对比", "差异", "which papers")
LIBRARY_OVERVIEW_HINTS = ("综述", "overview", "survey", "演化", "演进", "趋势", "脉络")
DEFINITION_HINTS = ("是什么", "什么是", "定义", "指什么", "what is", "meaning of")
MILVUS_FACT_HINTS = ("贡献", "方法", "结论", "创新", "摘要", "实验", "结果", "发现")


@dataclass
class RetrievalPlan:
    primary_store: str = "milvus"
    enable_graph_evidence: bool = False
    enable_library_graph_hint: bool = False
    graph_mode: str = "hybrid"
    reason: str = ""


def _normalize_paper_ids(paper_ids: list[str] | None) -> list[str]:
    return [str(item).strip() for item in (paper_ids or []) if str(item).strip()]


def build_retrieval_plan(
    *,
    query: str,
    project_id: str | None,
    paper_ids: list[str] | None,
    dialogue_mode: str | None = None,
    paper_focus_mode: str | None = None,
) -> RetrievalPlan:
    """
    按产品语义决定检索策略。

    当前约定：
    - 项目内研究：`Milvus` 负责主证据，`Neo4j` 在答案解释阶段补关系证据
    - 库级综述：`Milvus` 负责主证据，`LightRAG` 只做辅助提示层
    """
    normalized_query = str(query or "").strip().lower()
    normalized_paper_ids = _normalize_paper_ids(paper_ids)
    normalized_dialogue_mode = str(dialogue_mode or "").strip().lower()
    normalized_paper_focus_mode = str(paper_focus_mode or "").strip().lower()
    is_research_dialogue = normalized_dialogue_mode == "research"
    is_lookup_dialogue = normalized_dialogue_mode == "lookup"
    has_project_scope = bool(str(project_id or "").strip())
    is_explicit_focus = normalized_paper_focus_mode in ("single_paper", "multi_paper")

    if has_project_scope and is_research_dialogue:
        graph_mode = "hybrid" if any(hint in normalized_query for hint in GRAPH_EVIDENCE_HINTS + LIBRARY_OVERVIEW_HINTS) else "global"
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=True,
            enable_library_graph_hint=True,
            graph_mode=graph_mode,
            reason="项目内研究使用 Milvus 主证据，并补 LightRAG 全局线索",
        )

    if has_project_scope and is_lookup_dialogue and normalized_paper_ids:
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=False,
            enable_library_graph_hint=False,
            graph_mode="local",
            reason="项目内查证问题优先使用 Milvus 原文证据",
        )

    if str(project_id or "").strip() and normalized_paper_ids:
        graph_mode = "hybrid" if any(hint in normalized_query for hint in GRAPH_EVIDENCE_HINTS) else "local"
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=True,
            enable_library_graph_hint=False,
            graph_mode=graph_mode,
            reason="项目内研究默认使用 chunk 证据，并在后续答案解释阶段补 Neo4j 关系证据",
        )
    if any(hint in normalized_query for hint in LIBRARY_OVERVIEW_HINTS):
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=False,
            enable_library_graph_hint=True,
            graph_mode="global",
            reason="库级综述问题允许引入 LightRAG 作为辅助提示层",
        )
    if any(hint in normalized_query for hint in GRAPH_EVIDENCE_HINTS):
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=False,
            enable_library_graph_hint=True,
            graph_mode="local",
            reason="关系型问题保留 Milvus 主检索，再用库级图谱提示补充线索",
        )
    if is_explicit_focus and normalized_paper_ids:
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=False,
            enable_library_graph_hint=False,
            graph_mode="local",
            reason="显式论文焦点问题优先使用 Milvus 原文证据",
        )
    if any(hint in normalized_query for hint in MILVUS_FACT_HINTS):
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=False,
            enable_library_graph_hint=False,
            graph_mode="hybrid",
            reason="文本事实型问题优先使用 Milvus 主检索",
        )
    if any(hint in normalized_query for hint in DEFINITION_HINTS):
        return RetrievalPlan(
            primary_store="milvus",
            enable_graph_evidence=False,
            enable_library_graph_hint=False,
            graph_mode="hybrid",
            reason="定义类问题优先使用文本证据，避免图谱过度检索",
        )
    return RetrievalPlan(
        primary_store="milvus",
        enable_graph_evidence=False,
        enable_library_graph_hint=False,
        graph_mode="hybrid",
        reason="默认使用 Milvus 主检索，按需再叠加其他证据层",
    )
