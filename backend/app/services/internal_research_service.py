"""
内部研究输出契约构建器。

Wave 5 不改动 `answer / citations / evidence_summary` 主链，
而是在最终答案之上补一层稳定的研究结构：

1. `research_scope`：这次内部研究覆盖了几篇论文、证据强度如何
2. `research_objects`：把方法、数据集、指标、实验、局限性、比较显式展开
3. `note_actions`：为前端提供可直接复制/保存的笔记动作
"""

from __future__ import annotations

from typing import Any

import app.services.paper_db as paper_db
from app.models.research_contracts import (
    DIALOGUE_MODE_LOOKUP,
    DIALOGUE_MODE_RESEARCH,
    default_evidence_layer_roles,
)
from app.services import paper_metadata_service
from app.services.graph.research_object_extractor import extract_research_objects


_COMPARE_HINTS = ("compare", "difference", "vs", "versus", "对比", "比较", "差异", "优劣")
_LIMITATION_HINTS = ("limitation", "weakness", "risk", "局限", "不足", "缺点", "风险")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _clip(value: str, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _normalize_structured_query(structured_query: Any) -> dict[str, Any]:
    return structured_query if isinstance(structured_query, dict) else {}


def _normalize_project_scope(structured_query: dict[str, Any]) -> dict[str, Any]:
    project_scope = structured_query.get("project_scope")
    if isinstance(project_scope, dict):
        return project_scope
    paper_ids = _dedupe_preserve_order(
        [str(item).strip() for item in structured_query.get("paper_ids", []) if str(item).strip()]
    )
    project_id = str(structured_query.get("project_id") or "").strip()
    return {
        "project_id": project_id,
        "paper_ids": paper_ids,
        "filter_expr": str(structured_query.get("filter_expr") or "").strip(),
        "resolved_scope": structured_query.get("resolved_scope") if isinstance(structured_query.get("resolved_scope"), dict) else None,
        "has_project_scope": bool(project_id),
        "summary": f"当前项目范围覆盖 {len(paper_ids)} 篇论文。" if project_id else "当前问题未绑定项目范围。",
    }


def _normalize_paper_focus(structured_query: dict[str, Any], paper_ids: list[str]) -> dict[str, Any]:
    paper_focus = structured_query.get("paper_focus")
    if isinstance(paper_focus, dict):
        return paper_focus
    if len(paper_ids) == 1:
        return {
            "mode": "single_paper",
            "paper_ids": paper_ids,
            "is_explicit": bool(structured_query.get("paper_ids")),
            "source": "paper_focus" if structured_query.get("paper_ids") else "project_scope",
            "summary": "当前问题聚焦单篇论文。",
        }
    if len(paper_ids) > 1:
        return {
            "mode": "multi_paper",
            "paper_ids": paper_ids,
            "is_explicit": bool(structured_query.get("paper_ids")),
            "source": "paper_focus" if structured_query.get("paper_ids") else "project_scope",
            "summary": f"当前问题聚焦 {len(paper_ids)} 篇论文。",
        }
    return {
        "mode": "none",
        "paper_ids": [],
        "is_explicit": False,
        "source": "web_scope",
        "summary": "当前问题没有显式论文焦点。",
    }


def _normalize_citations(citations: Any) -> list[dict[str, Any]]:
    if not isinstance(citations, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in citations:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        if not paper_id:
            continue
        normalized.append({**item, "paper_id": paper_id})
    return normalized


def _resolve_paper_ids(citations: list[dict[str, Any]], structured_query: dict[str, Any]) -> list[str]:
    resolved_scope = structured_query.get("resolved_scope")
    scope_dict = resolved_scope if isinstance(resolved_scope, dict) else {}
    scoped_ids = []
    for raw in (structured_query.get("paper_ids"), scope_dict.get("paper_ids")):
        if isinstance(raw, list):
            scoped_ids.extend(str(item).strip() for item in raw if str(item).strip())
    citation_ids = [str(item.get("paper_id") or "").strip() for item in citations if str(item.get("paper_id") or "").strip()]
    return _dedupe_preserve_order(citation_ids + scoped_ids)


def _build_scope(
    *,
    query: str,
    paper_ids: list[str],
    citations: list[dict[str, Any]],
    answer_confidence: str,
    failure_reason: str,
) -> dict[str, Any]:
    """
    构造前端展示层可直接消费的 research_scope 概览。

    Args:
        query: 用户问题
        paper_ids: 当前回答覆盖到的论文集合
        citations: 当前可用结构化引用
        answer_confidence: 当前回答置信度
        failure_reason: 本轮链路是否出现降级/失败信号

    Returns:
        dict[str, Any]: 表示单篇/比较/范围内研究模式的展示摘要
    """
    normalized_query = str(query or "").strip().lower()
    is_comparison = len(paper_ids) > 1 or any(hint in normalized_query for hint in _COMPARE_HINTS)
    if len(paper_ids) <= 1:
        mode = "single_paper"
        label = "单篇内部研究"
        summary = "当前回答主要围绕单篇论文的内部证据。"
    elif is_comparison:
        mode = "cross_paper"
        label = "多篇内部比较"
        summary = f"当前回答综合 {len(paper_ids)} 篇论文，适合继续查看方法、数据集、指标和局限性的差异。"
    else:
        mode = "scoped_internal_research"
        label = "范围内内部研究"
        summary = f"当前回答基于 {len(paper_ids)} 篇范围内论文的内部证据。"
    if not citations:
        summary += " 当前没有附带结构化引用，建议继续追问或回看原文。"
    elif failure_reason:
        summary += " 当前回答走过降级路径，结论可先作为线索使用。"
    return {
        "mode": mode,
        "label": label,
        "summary": summary,
        "paper_ids": paper_ids,
        "citation_count": len(citations),
        "confidence": str(answer_confidence or "").strip(),
    }


def _load_paper_entries(paper_ids: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        paper = paper_db.get_paper(paper_id)
        if not paper:
            continue
        analysis = paper_metadata_service.normalize_analysis_payload(paper.get("analysis"))
        extraction = extract_research_objects({**paper, "analysis": analysis})
        entries.append(
            {
                "paper_id": paper_id,
                "paper": paper,
                "analysis": analysis,
                "extraction": extraction,
            }
        )
    return entries


def _aggregate_named_section(entries: list[dict[str, Any]], field: str, *, key: str, label: str, limit: int = 6) -> dict[str, Any] | None:
    buckets: dict[str, dict[str, Any]] = {}
    for entry in entries:
        paper_id = entry["paper_id"]
        for raw in entry["extraction"].get(field, []) or []:
            name = str(raw or "").strip()
            if not name:
                continue
            bucket = buckets.setdefault(
                name.lower(),
                {
                    "label": name,
                    "paper_ids": [],
                },
            )
            if paper_id not in bucket["paper_ids"]:
                bucket["paper_ids"].append(paper_id)
    items = []
    for bucket in buckets.values():
        paper_ids = bucket["paper_ids"]
        detail = f"出现在 {len(paper_ids)} 篇论文中。" if len(paper_ids) > 1 else "见对应论文证据。"
        items.append(
            {
                "label": bucket["label"],
                "detail": detail,
                "paper_ids": paper_ids,
            }
        )
    if not items:
        return None
    return {
        "key": key,
        "label": label,
        "items": items[:limit],
    }


def _build_comparison_section(entries: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    """
    为多论文问题生成“比较”研究对象分组。

    Args:
        entries: 已加载分析结果的论文条目
        query: 用户问题

    Returns:
        dict[str, Any] | None: 比较 section；当前问题不是比较语义时返回 None
    """
    normalized_query = str(query or "").lower()
    if len(entries) < 2 and not any(hint in normalized_query for hint in _COMPARE_HINTS):
        return None
    items = []
    for entry in entries:
        paper = entry["paper"]
        analysis = entry["analysis"]
        extraction = entry["extraction"]
        detail_parts: list[str] = []
        if analysis.get("method"):
            detail_parts.append(f"方法：{_clip(analysis['method'], limit=100)}")
        if extraction.get("datasets"):
            detail_parts.append(f"数据集：{', '.join(extraction['datasets'][:3])}")
        if extraction.get("metrics"):
            detail_parts.append(f"指标：{', '.join(extraction['metrics'][:3])}")
        if analysis.get("limitation"):
            detail_parts.append(f"局限：{_clip(analysis['limitation'], limit=90)}")
        items.append(
            {
                "label": _clip(str(paper.get("title") or entry["paper_id"]), limit=90),
                "detail": "；".join(detail_parts) or "建议继续查看原文的实验和讨论部分。",
                "paper_ids": [entry["paper_id"]],
            }
        )
    return {
        "key": "comparisons",
        "label": "比较",
        "items": items[:6],
    }


def _build_experiment_section(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    从论文分析结构里提取“实验”相关研究对象。

    Args:
        entries: 已加载分析结果的论文条目

    Returns:
        dict[str, Any] | None: 实验 section；没有可用实验细节时返回 None
    """
    items = []
    for entry in entries:
        paper = entry["paper"]
        analysis = entry["analysis"]
        extraction = entry["extraction"]
        detail = str(analysis.get("methodology_detail") or "").strip()
        if not detail:
            detail_parts: list[str] = []
            if extraction.get("datasets"):
                detail_parts.append(f"数据集：{', '.join(extraction['datasets'][:3])}")
            if extraction.get("metrics"):
                detail_parts.append(f"指标：{', '.join(extraction['metrics'][:3])}")
            detail = "；".join(detail_parts)
        if not detail:
            continue
        items.append(
            {
                "label": _clip(str(paper.get("title") or entry["paper_id"]), limit=90),
                "detail": _clip(detail, limit=180),
                "paper_ids": [entry["paper_id"]],
            }
        )
    if not items:
        return None
    return {
        "key": "experiments",
        "label": "实验",
        "items": items[:6],
    }


def _build_limitation_section(entries: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    """
    构造“局限性”研究对象分组。

    Args:
        entries: 已加载分析结果的论文条目
        query: 用户问题

    Returns:
        dict[str, Any] | None: 局限性 section；当前问题无此语义且无条目时返回 None
    """
    normalized_query = str(query or "").lower()
    items = []
    for entry in entries:
        limitation = str(entry["analysis"].get("limitation") or "").strip()
        if not limitation:
            continue
        items.append(
            {
                "label": _clip(str(entry["paper"].get("title") or entry["paper_id"]), limit=90),
                "detail": _clip(limitation, limit=180),
                "paper_ids": [entry["paper_id"]],
            }
        )
    if not items and not any(hint in normalized_query for hint in _LIMITATION_HINTS):
        return None
    return {
        "key": "limitations",
        "label": "局限性",
        "items": items[:6],
    } if items else None


def _build_synthesis_state(
    *,
    query: str,
    citations: list[dict[str, Any]],
    paper_ids: list[str],
    structured_query: dict[str, Any],
    sections: list[dict[str, Any]],
    answer_confidence: str,
    failure_reason: str,
) -> dict[str, Any]:
    """
    把当前研究整理结果收敛成稳定的 synthesis_state。

    Args:
        query: 用户问题
        citations: 当前可用结构化引用
        paper_ids: 当前回答覆盖到的论文集合
        structured_query: 当前请求的结构化范围合同
        sections: 已整理出的 research_objects
        answer_confidence: 当前回答置信度
        failure_reason: 当前链路失败或降级原因

    Returns:
        dict[str, Any]: 供前端和后续回答生成复用的统一研究状态
    """
    normalized_query = str(query or "").strip().lower()
    supported_dimensions = [str(section.get("key") or "").strip() for section in sections if str(section.get("key") or "").strip()]
    unsupported_dimensions = [
        dimension for dimension in ("methods", "datasets", "metrics", "experiments", "limitations", "comparisons")
        if dimension not in supported_dimensions
    ]
    dialogue_mode = str(structured_query.get("dialogue_mode") or "").strip().lower()
    if not dialogue_mode:
        dialogue_mode = DIALOGUE_MODE_RESEARCH if len(paper_ids) > 1 or any(hint in normalized_query for hint in _COMPARE_HINTS) else DIALOGUE_MODE_LOOKUP
    project_scope = _normalize_project_scope(structured_query)
    paper_focus = _normalize_paper_focus(structured_query, paper_ids)
    open_questions: list[str] = []
    if not citations:
        open_questions.append("当前缺少结构化引用，后续需要继续补原文证据。")
    if unsupported_dimensions:
        open_questions.append(f"以下研究维度仍缺证据：{', '.join(unsupported_dimensions)}。")
    if failure_reason:
        open_questions.append("当前回答链路出现降级或失败信号，结论需谨慎使用。")
    return {
        "status": "ready" if citations else "insufficient",
        "dialogue_mode": dialogue_mode,
        "question_type": paper_focus.get("mode") or "project_scope",
        "project_scope": project_scope,
        "paper_focus": paper_focus,
        "evidence_summary": {
            "paper_count": len(paper_ids),
            "citation_count": len(citations),
            "answer_confidence": str(answer_confidence or "").strip(),
            "failure_reason": str(failure_reason or "").strip(),
            "has_direct_evidence": bool(citations),
        },
        "dimension_status": {
            dimension: ("supported" if dimension in supported_dimensions else "missing")
            for dimension in ("methods", "datasets", "metrics", "experiments", "limitations", "comparisons")
        },
        "supported_dimensions": supported_dimensions,
        "unsupported_dimensions": unsupported_dimensions,
        "open_questions": open_questions,
        "evidence_layers": default_evidence_layer_roles(),
        "notes": [item for item in (project_scope.get("summary"), paper_focus.get("summary")) if item],
        "citations_count": len(citations),
    }


def _first_primary_paper_id(entries: list[dict[str, Any]]) -> str:
    return str(entries[0]["paper_id"]) if entries else ""


def _build_note_text(query: str, answer: str, sections: list[dict[str, Any]]) -> str:
    lines = [
        f"研究问题：{_clip(query, limit=240)}",
        f"结论：{_clip(answer, limit=700)}",
    ]
    section_lookup = {item.get("key"): item for item in sections if isinstance(item, dict)}
    for key, prefix in (
        ("methods", "方法"),
        ("datasets", "数据集"),
        ("metrics", "指标"),
        ("limitations", "局限"),
    ):
        section = section_lookup.get(key)
        if not section:
            continue
        values = [entry.get("label", "") for entry in section.get("items", [])[:3] if str(entry.get("label", "")).strip()]
        if values:
            lines.append(f"{prefix}：{'；'.join(values)}")
    return "\n".join(line for line in lines if line.strip()).strip()


def _build_limitations_note(query: str, limitation_section: dict[str, Any] | None) -> str:
    lines = [f"研究空白追问：{_clip(query, limit=240)}"]
    if limitation_section:
        for item in limitation_section.get("items", [])[:3]:
            label = str(item.get("label") or "").strip()
            detail = str(item.get("detail") or "").strip()
            if label and detail:
                lines.append(f"- {label}：{detail}")
    return "\n".join(lines).strip()


def _build_note_actions(
    *,
    query: str,
    answer: str,
    sections: list[dict[str, Any]],
    is_cross_paper: bool,
    primary_paper_id: str,
    limitation_section: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    生成前端可直接使用的笔记动作清单。

    Args:
        query: 用户问题
        answer: 当前回答
        sections: 已整理出的 research_objects
        is_cross_paper: 当前是否为多论文场景
        primary_paper_id: 当前最适合挂接笔记的主论文 ID
        limitation_section: 已整理好的局限性 section

    Returns:
        list[dict[str, Any]]: 复制/保存笔记动作列表
    """
    note_text = _build_note_text(query, answer, sections)
    note_type = "comparison_note" if is_cross_paper else "reading_note"
    actions = [
        {
            "key": "copy_summary_note",
            "label": "复制结论为笔记",
            "type": "copy_note",
            "note": note_text,
        }
    ]
    if primary_paper_id:
        actions.append(
            {
                "key": "save_primary_note",
                "label": "保存到论文笔记",
                "type": "save_note",
                "note": note_text,
                "target_paper_id": primary_paper_id,
                "note_type": note_type,
                "source_type": "local_rag",
            }
        )
    if limitation_section and limitation_section.get("items"):
        actions.append(
            {
                "key": "copy_gap_note",
                "label": "复制局限与追问",
                "type": "copy_note",
                "note": _build_limitations_note(query, limitation_section),
            }
        )
    return actions


def build_internal_research_contract(
    *,
    query: str,
    answer: str,
    citations: Any,
    structured_query: Any,
    answer_confidence: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    """
    基于最终 RAG 回答构造内部研究契约。

    这里尽量使用稳定的业务数据和规则抽取，避免把前端新展示完全绑在一次 LLM 输出上。

    Args:
        query: 用户问题
        answer: 最终回答文本
        citations: 结构化引用列表
        structured_query: 当前请求的结构化范围合同
        answer_confidence: 当前回答置信度
        failure_reason: 当前链路失败或降级原因

    Returns:
        dict[str, Any]: 供主图回填给 API 层和前端的内部研究合同
    """
    normalized_citations = _normalize_citations(citations)
    normalized_structured_query = _normalize_structured_query(structured_query)
    paper_ids = _resolve_paper_ids(normalized_citations, normalized_structured_query)
    entries = _load_paper_entries(paper_ids)

    # 1. 先用稳定分析结果生成 methods/datasets/metrics 等研究对象 section
    sections: list[dict[str, Any]] = []
    for section in (
        _build_comparison_section(entries, query),
        _aggregate_named_section(entries, "methods", key="methods", label="方法"),
        _aggregate_named_section(entries, "datasets", key="datasets", label="数据集"),
        _aggregate_named_section(entries, "metrics", key="metrics", label="指标"),
        _build_experiment_section(entries),
        _build_limitation_section(entries, query),
    ):
        if section and section.get("items"):
            sections.append(section)

    limitation_section = next((item for item in sections if item.get("key") == "limitations"), None)
    # 2. 再把 research_scope、synthesis_state 和 note_actions 收敛成统一研究合同
    synthesis_state = _build_synthesis_state(
        query=query,
        citations=normalized_citations,
        paper_ids=paper_ids,
        structured_query=normalized_structured_query,
        sections=sections,
        answer_confidence=answer_confidence,
        failure_reason=failure_reason,
    )
    return {
        "research_scope": _build_scope(
            query=query,
            paper_ids=paper_ids,
            citations=normalized_citations,
            answer_confidence=answer_confidence,
            failure_reason=failure_reason,
        ),
        "research_objects": sections,
        "note_actions": _build_note_actions(
            query=query,
            answer=answer,
            sections=sections,
            is_cross_paper=len(paper_ids) > 1,
            primary_paper_id=_first_primary_paper_id(entries) if len(paper_ids) == 1 else "",
            limitation_section=limitation_section,
        ),
        "synthesis_state": synthesis_state,
        "dialogue_mode": synthesis_state["dialogue_mode"],
    }
