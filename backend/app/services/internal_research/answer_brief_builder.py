"""回答简报构造器：消费 evidence_records / missing_evidence / intent / focus 填充 AnswerBrief。"""

from __future__ import annotations

from typing import Any

from app.models import evidence_layers as layers
from app.models.research_contracts import AnswerBrief, EvidenceRecord, FocusResolution, MissingEvidence


def build(state: dict[str, Any]) -> AnswerBrief:
    """基于 RAGState 构造填充后的 AnswerBrief。"""
    evidence_records: list[EvidenceRecord] = state.get("evidence_records") or []
    missing_evidence: list[MissingEvidence] = state.get("missing_evidence") or []
    intent: str = state.get("intent") or ""
    focus: FocusResolution | dict = state.get("focus") or {}

    usable = [r for r in evidence_records if r.get("usable_as_citation")]
    graph_hints = [r for r in evidence_records if r.get("layer") == layers.EVIDENCE_LAYER_GRAPH_HINT]

    guidance_parts: list[str] = []
    if graph_hints:
        guidance_parts.append("图线索仅供参考，不可作为事实引用。")

    uncertainties: list[str] = []
    fallback_reason = focus.get("fallback_reason", "") if isinstance(focus, dict) else ""
    if fallback_reason:
        uncertainties.append(f"焦点降级：{fallback_reason}")

    forbidden_claims: list[str] = ["不要把图线索说成原文事实。"]

    if intent == "compare":
        turn_focus_papers = focus.get("turn_focus", []) if isinstance(focus, dict) else []
        covered_papers = {r.get("paper_id") for r in evidence_records if r.get("layer") == layers.EVIDENCE_LAYER_DIRECT_TEXT}
        for pid in turn_focus_papers:
            if pid and pid not in covered_papers:
                forbidden_claims.append(f"缺少 {pid} 的原文证据，不要对其做事实性断言。")

    focus_summary = ""
    if isinstance(focus, dict):
        turn = focus.get("turn_focus", [])
        if turn:
            focus_summary = f"焦点论文: {', '.join(turn)}"

    return AnswerBrief(
        intent=intent,
        focus=focus_summary,
        usable_evidence=usable,
        missing_evidence=missing_evidence,
        uncertainties=uncertainties,
        answer_guidance=" ".join(guidance_parts) if guidance_parts else "",
        forbidden_claims=forbidden_claims,
    )


def serialize_brief_to_prompt_text(brief: AnswerBrief) -> str:
    """将 AnswerBrief 序列化为提示词文本段落。"""
    lines: list[str] = []

    intent = brief.get("intent", "")
    if intent:
        lines.append(f"Intent: {intent}")

    focus = brief.get("focus", "")
    if focus:
        lines.append(f"Focus: {focus}")

    usable = brief.get("usable_evidence") or []
    lines.append(f"可用证据: {len(usable)} 条")

    missing = brief.get("missing_evidence") or []
    if missing:
        lines.append(f"缺失证据: {len(missing)} 项")
        for m in missing[:5]:
            lines.append(f"  - {m.get('paper_id', '?')}: {m.get('reason', '未知')}")

    uncertainties = brief.get("uncertainties") or []
    if uncertainties:
        lines.append("不确定性:")
        for u in uncertainties:
            lines.append(f"  - {u}")

    guidance = brief.get("answer_guidance", "")
    if guidance:
        lines.append(f"回答建议: {guidance}")

    forbidden = brief.get("forbidden_claims") or []
    if forbidden:
        lines.append("禁止表述:")
        for f in forbidden:
            lines.append(f"  FORBIDDEN: {f}")

    return "\n".join(lines)
