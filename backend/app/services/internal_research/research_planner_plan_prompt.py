"""Planner analysis prompt construction."""

from __future__ import annotations

import json
from typing import Any, Mapping

from app.services.internal_research.research_planner_loop_state import (
    PROBE_RESULTS_STATE_KEY,
)
from app.services.research_memory_injector_service import render_memory_prompt_block
from app.services.internal_research.research_planner_consumer_budget import render_read_reports
from app.services.internal_research.research_planner_final_delivery import project_delivery_findings

# 记忆读取四态里的成功态：只有真读到条目才渲染，empty / skipped / failed 不进 prompt，
# 免得给模型塞一行「本轮没有记忆」的噪音
_MEMORY_READ_SUCCESS = "success"

# 探测结果段用途与原因一行内说明白，都不带数字含义以外的业务规则。发现行的宽度与
# 截断说明不在这层做：回灌前已由编排层按消费侧总额度分配好，这层只是原样折进一行
_PROBE_PURPOSE_MAX_CHARS = 200
_PROBE_REASON_MAX_CHARS = 200

_HISTORY_MAX_MESSAGES = 6
_HISTORY_MAX_CHARS = 500

PLAN_PROMPT_VERSION = "research-plan-strategy-v5"
ACTION_PROMPT_VERSION = "research-incremental-v6"


def build_action_prompt(state: dict[str, Any]) -> str:
    """参数为本轮状态；返回当前动作及实际成果的决策上下文，不索要完整未来计划。"""
    plan = state.get("research_plan") or {}
    # 1. 【研究循环】【要求归属】与实际首次合同选择一致，后续只消费服务端已接受要求
    requirement_instruction = (
        "First identify real requested_facets. Explicit facets must be required and quote a contiguous verbatim "
        "span from Question or user history; inferred required facets need rationale. "
        "Each facet must name an independently answerable research point with its required objects, conditions "
        "and result. Do not collapse several requested results into a broad topic label. "
        "Derive necessary detail from this user's question, not a fixed paper-review checklist. "
        "Optional answer_sections establish answer organization on this first decision."
        if not state.get("coverage_review") else
        "Use the accepted requirements supplied below. Do not resubmit requested_facets or answer_sections; "
        "preserve their content, conditions and presentation obligations when choosing the next action."
    )
    # 2. 【研究循环】【拒绝反馈】保留引用校验的具体错误，避免通用履约缺口遮蔽已有材料
    observations = [_action_result(result) for result in state.get("research_unit_results") or []]
    # 3. 【研究循环】【材料目录】决策获得可定位摘要，分析动作通过既有消费预算取得真实正文
    materials = [{key: record.get(key) for key in
                  ("evidence_id", "paper_id", "section_path", "source_type", "usable_as_citation")}
                 | {"preview": str(record.get("excerpt") or record.get("summary") or "")[:_HISTORY_MAX_CHARS]}
                 for record in state.get("evidence_records") or []]
    return "\n".join([
        ACTION_PROMPT_VERSION,
        "Return one complete JSON response object matching the supplied schema. Its decision object selects "
        "execute with currently ready independent actions, clarify, or finish.",
        "Never put future work in unready executable actions.",
        requirement_instruction,
        "Read actions acquire materials. They do not have to cover every question. Only select applicable input fields.",
        "For section reads, copy the full section path from the directory; a short selector is valid only when it uniquely identifies a section.",
        "To continue a partial section, select its returned continuation_ids instead of repeating required_sections; "
        "keep the same target_paper_ids. Do not invent offsets. Materials dependencies may select specific "
        "evidence_id and a returned range_id in selections; omitted selections consumes the whole dependency.",
        "After reads, analyze returned unit_ids with their actual product (materials, claims, hypotheses). "
        "Do not invent future ids. Grounded analysis may extract paper facts or make supported inferences; "
        "it must later select actual source spans. Hypothesis and memory modes are explicit nonfactual alternatives.",
        "A materials dependency consumes text and has no required_claim_types field, even when empty. "
        "Claims dependencies may require paper_supported or evidence_grounded_inference; hypotheses dependencies "
        "may require model_prior_hypothesis or memory_continuation. Empty type requirements add no type restriction. "
        "Choose required=false only when the dependency is optional.",
        "Ordinary answer organization happens at final synthesis. Analyze only when substantive claims are needed.",
        "Material previews and result summaries are observations, not proof of final claims.",
        "Choose finish only after inspecting fulfillment. Missing requirements must prompt more work, clarification, "
        "or an explicit allow_partial finish with reason; finish never directly declares success.",
        "Repeated identical actions on unchanged inputs do not create progress. Use returned errors to correct the action.",
        "rejected_claims records explicit validation failures. A rejected citation, span or quote does not itself mean "
        "material is missing; inspect the reported support against already-read material and correct its selection "
        "in the next analysis. Retrieve more only when the needed source is actually absent.",
        "A candidate content review identifies unanswered research points or unsupported prose. Choose analysis "
        "of already acquired materials when their contents have not been analyzed; read more only for actual "
        "source gaps. If qualified claims already suffice, choose finish again to revise final prose using the "
        "review feedback; this does not require new research actions. Explicit allow_partial with a concrete "
        "reason is valid when further work is not justified. Unread or failed reads never prove absence in a paper.",
        "Question:", str(state.get("user_query") or ""), render_conversation_context(state),
        *_paper_metadata_lines(state), *_section_structure_lines(state), *_graph_context_lines(state),
        *_project_memory_lines(state), f"Focus: {state.get('focus') or state.get('paper_focus') or {}}",
        *probe_results_lines(state),
        "Authorized papers:", json.dumps((state.get("project_scope") or {}).get("paper_ids", []), ensure_ascii=False),
        "Accepted requirements:", json.dumps(_requirements_projection(state), ensure_ascii=False),
        "Accepted actions:", json.dumps([_action_identity(unit) for unit in plan.get("units", [])], ensure_ascii=False),
        "Actual results:", json.dumps(observations, ensure_ascii=False),
        "Material directory (previews may be truncated):", json.dumps(materials, ensure_ascii=False),
        "Read outcomes:", render_read_reports([item["read_report"] for item in state.get("planner_read_receipts") or []
                                               if item.get("read_report")]),
        "Current fulfillment:", json.dumps(_fulfillment_projection(state), ensure_ascii=False),
        "Candidate content review:", json.dumps(_content_review_feedback(state), ensure_ascii=False),
        "Candidate delivery findings:", json.dumps(project_delivery_findings(state.get("final_delivery_report") or {}), ensure_ascii=False),
        "Latest action feedback:", json.dumps(_action_feedback(state), ensure_ascii=False),
    ])


def _action_result(result: dict[str, Any]) -> dict[str, Any]:
    """参数为完整成果；返回决策所需正文、支持身份、资格与明确拒绝原因，原成果不修改。"""
    keys = ("claim_id", "text", "claim_type", "addressed_facet_ids", "validation_status",
            "supporting_evidence_ids", "source_claim_ids", "source_unit_ids", "validation_notes")
    metadata = result.get("metadata") or {}
    exposure = (metadata.get("provenance") or {}).get("exposure") or {}
    return {"unit_id": result["unit_id"], "status": result["status"],
            "evidence_ids": result.get("evidence_ids", []),
            "claims": [{key: claim[key] for key in keys if key in claim} for claim in result.get("claims", [])],
            "rejected_claims": metadata.get("rejected_claims", []),
            "unresolved_reason": result.get("unresolved_reason", ""),
            "unprovided_ranges": [{"range_id": item["range_id"], "evidence_id": item["evidence_id"],
                                   "unprovided_chars": item["end"] - item["start"]}
                                  for item in exposure.get("unprovided_ranges", [])]}


def _action_identity(unit: dict[str, Any]) -> dict[str, Any]:
    """参数为服务器动作账本；返回定位既有动作所需参数，不向模型转交派生许可和运行身份。"""
    policy = unit.get("evidence_policy") or {}
    return {"unit_id": unit["unit_id"], "goal": unit["goal"], "covered_facet_ids": unit.get("covered_facet_ids", []),
            "input": {key: value for key, value in policy.items() if value not in (None, "", [])},
            "dependencies": unit.get("dependencies", []), "output_role": unit.get("output_role")}


def _action_feedback(state: dict[str, Any]) -> dict[str, Any]:
    """参数为完整循环观察；只提供真实业务反馈，指纹与比较键保留在服务器状态。"""
    observation = state.get("planner_observation") or {}
    keys = ("phase", "ok", "error_code", "issue_codes", "summary", "blocked_unit_ids",
            "attempted_queries", "scoped_paper_ids", "has_answerable_evidence")
    return {key: observation[key] for key in keys if key in observation}


def _fulfillment_projection(state: dict[str, Any]) -> dict[str, Any]:
    """参数为取得报告；返回业务缺口与逐点状态，避免把账本身份当成内容核对结论。"""
    report = state.get("evidence_sufficiency_report") or {}
    projection = {key: value for key, value in report.items()
                  if key in {"missing_required_facet_ids", "blocking_unit_ids", "has_usable_results"}}
    projection["facet_reports"] = [{key: row[key] for key in ("facet_id", "required", "satisfied", "evidence_ids",
        "claim_ids", "missing_paper_ids") if key in row} | {"gaps": [gap["reason"] for gap in row.get("hard_gaps", [])]}
        for row in report.get("facet_reports", [])]
    return projection


def _content_review_feedback(state: dict[str, Any]) -> dict[str, Any]:
    """参数为上次候选交付记录；只传具体内容核对，内部摘要和尝试身份不进入行动提示。"""
    review = (state.get("final_delivery_report") or {}).get("content_review") or {}
    return {"status": review.get("status"), "assessment": review.get("assessment"), "error": review.get("error", "")} if review else {}

_PLAN_RULE_LINES = [
    "Return exactly one JSON object matching the supplied schema; no prose outside JSON.",
    "Plan an internal project-scoped paper question. Select research methods and claim intents; "
    "the server derives execution policy, question, coverage and identity metadata.",
    "Each claim_intents entry selects one claim type and whether it is required. Materials have no claim intents. "
    "Claim-producing units must declare at least one required intent. Do not repeat a type.",
    "First planning identifies requested_facets and answer_sections. Replanning changes only units and references "
    "the fixed requirements baseline; it cannot redefine or lower content, objects, conditions or presentation obligations.",
    "Initial planning with any units must include at least one real requested_facet. A pure clarification "
    "with no units may have empty requested_facets. Replanning submits units, not replacement requirements.",
    "Every explicit facet needs a nonempty user_text_span copied as a contiguous verbatim span from Question "
    "or a visible user history excerpt. Preserve spaces and line breaks. Do not quote assistant messages, "
    "resolved questions, paper text or excerpt markers as user words. Inferred facets need no fabricated quote.",
    "Every explicit facet is required. Inferred facets may also be necessary; preserve the task rationale. "
    "Required papers come from user requests and confirmed focus, not every authorized paper.",
    "Recommendation facets require final_recommendation output; ordinary facts, comparison and materials have no role obligation.",
    "Use local_direct_text for retrieved text, local_section_text for the actual text of named sections. "
    "Copy full section paths from the section list; a short selector is valid only when it uniquely identifies a section.",
    "Use local_metadata for bibliographic facts; local_structure_hint for known graph relations, "
    "local_section_structure for section lists, and local_chunk_metadata only for chunk locations. These are not paper citations.",
    "Chunk metadata requires section, block type or page filters; fields combine with AND and values with OR.",
    "Parent-chunk context is available only for local_direct_text. Direct-text verification of graph relations "
    "is available only for local_structure_hint and must be explicitly requested.",
    "Use natural-language research_action. For inference_from_units, select concrete upstream dependencies and "
    "their required products, then choose synthesis or final_recommendation. A declaration is not completed evidence.",
    "For paper comparison, jointly cover every required paper. Evidence and synthesis may span multiple units.",
    "Use evidence_grounded_inference only when intended, and retain actual evidence or eligible upstream claims. "
    "Model-prior hypotheses are separate, unverified intents and are never implicitly enabled by inference.",
    "memory_continuation must be explicitly selected and requires an available project judgment shown below; "
    "it is never paper evidence or a replacement for missing evidence.",
    "External/background needs use external_context_needed and external_needed. Include English locator phrases "
    "for the existing arXiv metadata lookup. Project-library design or synthesis is not an external-background request.",
    "Internal answer sections organize content but create no hard heading obligation. User-requested sections "
    "or exact headings need presentation_kind and an exact nonempty user_text_span from the user input. "
    "For exact_heading, title itself must occur in that span. Content facets remain independently required.",
    "If the goal cannot be confirmed, ask one plain-language clarification and return no units; "
    "do not use clarification for details that local evidence could resolve.",
    "Keep explicit conditions and metrics traceable; validation_tasks describe pending checks, not proof.",
]


def build_planner_prompt(state: dict[str, Any]) -> str:
    """参数为当前问题及累计研究状态；返回含稳定要求、真实成果与定位依据的规划提示。"""
    return "\n".join([
        f"Prompt version: {PLAN_PROMPT_VERSION}",
        *_PLAN_RULE_LINES,
        f"Question: {state.get('user_query', '')}",
        "Conversation context:",
        render_conversation_context(state),
        "Previous RAG context:",
        _previous_rag_context(state),
        f"Coarse intent: {state.get('intent', '')}",
        *_depth_assessment_lines(state),
        *_project_memory_lines(state),
        *probe_results_lines(state),
        *_paper_metadata_lines(state),
        *_section_structure_lines(state),
        *_graph_context_lines(state),
        f"Focus: {state.get('focus') or state.get('paper_focus') or {}}",
        f"Project scope: {state.get('project_scope') or {}}",
        f"Structured query: {state.get('structured_query') or {}}",
        *_replan_feedback_lines(state),
        "Stable requirements baseline (preserve required facet objects exactly):",
        json.dumps(_requirements_projection(state), ensure_ascii=False),
        "Actual requirement fulfillment:",
        json.dumps(state.get("evidence_sufficiency_report") or {}, ensure_ascii=False),
    ])


def _requirements_projection(state: dict[str, Any]) -> dict[str, Any]:
    """参数为当前状态；返回唯一要求基准，不重复发送覆盖派生表或运行状态。"""
    review = state.get("coverage_review") or (state.get("research_plan") or {}).get("coverage_review") or {}
    return {key: review[key] for key in ("explicit_facets", "inferred_facets", "answer_sections") if key in review}


def _depth_assessment_lines(state: dict[str, Any]) -> list[str]:
    """把首次分析定下的处理深度连同该档的规划指引渲进草案 prompt。

    深度决定这份草案该铺多少检索与证据要求，所以除了深度标签，还要把该档
    「计划铺到什么程度算合适」的正面原则一起交给模型；只给一个 lookup 标签，
    模型照样会铺一份完整多阶段研究计划。未评估出深度时返回空列表，
    prompt 保持原样，不编一个默认深度塞进上下文。

    Args:
        state: 当前 RAGState 字典态，取 turn_depth_assessment

    Returns:
        list[str]: 深度、原因码、Evidence 要求、自洽问题，以及该档的规划指引；
            无评估结果时为空列表；深度取值不在三档内时只渲标签、不追加指引
    """
    assessment = state.get("turn_depth_assessment")
    if not isinstance(assessment, dict) or not assessment.get("depth"):
        return []
    depth = str(assessment.get("depth"))
    # 1. 先给出本轮分流结论与判定依据，让草案对齐上一步定下的深度
    lines = [
        f"Turn depth: {depth}",
        f"Turn depth reason: {assessment.get('reason_code') or ''}",
        f"Paper evidence required this turn: {bool(assessment.get('requires_paper_evidence'))}",
    ]
    # 2. 追问型问题在分流阶段已补成自洽问题，草案与检索式都按补全后的问题铺
    resolved_query = str(assessment.get("resolved_query") or "").strip()
    if resolved_query:
        lines.append(f"Resolved self-contained question: {_compact_text(resolved_query, 500)}")
    # 3. 再给这一档的规划原则；深度取值超出三档时不造第四套指引，只保留标签
    lines.extend(_DEPTH_PLAN_GUIDANCE.get(depth, ()))
    return lines


# 三档处理深度各自「该铺多少」的正面规划原则，取自 prd.md §6.1-§6.3。
# 这里只写自然语言原则，不设检索次数或子任务数量上限：深度本来就由模型判，
# 证据不够有重规划与补证据回边兜底，证据不足的出路是升级深度或明确报缺口，不是截断。
_DEPTH_PLAN_GUIDANCE: dict[str, tuple[str, ...]] = {
    "minimal": (
        "Plan guidance for depth=minimal: this turn needs no paper fact.",
        "- One understanding-and-answer action is enough; do not create a formal Research Plan.",
        "- Do not lay out retrieval units, staged evidence work, or cross-paper comparison here.",
        "- If the answer would assert a paper fact, this turn must move to lookup or research; "
        "never answer a paper fact directly at minimal.",
    ),
    "lookup": (
        "Plan guidance for depth=lookup: one local paper question, verified lightly.",
        "- Plan only the retrieval and evidence checking that answering this one question needs.",
        "- Do not add unrelated retrieval for the sake of form, and do not lay out a full "
        "multi-stage research plan.",
        "- Every paper fact in the answer carries this-turn Evidence that locates both the paper "
        "and the source snippet.",
        "- If evidence turns out insufficient, several papers must be compared, or a prior "
        "conclusion is needed, escalate this turn to research or report the gap explicitly.",
    ),
    "research": (
        "Plan guidance for depth=research: interdependent sub-questions worked through in order.",
        "- First state executable research goals and the sub-tasks that reach them, then let the "
        "units carry those sub-tasks out.",
        "- Every important conclusion must trace back to this-turn Evidence, and evidence from "
        "different papers stays presented separately.",
        "- Revising the plan, fetching further evidence, flagging conflicts, or stopping on a "
        "clearly reported evidence gap are all allowed outcomes.",
        "- The final synthesis keeps paper-supported claims, judgments continued from project "
        "memory, and the agent's own reasoning apart.",
    ),
}


def render_project_memory_block(state: dict[str, Any]) -> str:
    """把本轮读到的项目历史判断渲染成一段带身份约束的记忆文本。

    渲染复用注入服务的记忆段：那段文本开头写明「本项目已形成的判断，非论文原文，
    不得作为引用来源」，另写一套渲染会绕掉这条身份约束，模型就会把用户自己的判断
    当文献证据写进带引用的结论里。

    只有真读到条目才渲染：读取失败或读到空结果时返回空串，让调用方整段不出现。
    补一句「暂无历史判断」等于替项目下了个没验证过的结论，模型会顺着这句说这个
    项目还没想法；而拿失败/空的读取结果照样渲染，模型就能编一段项目历史判断。

    Args:
        state: 当前 RAGState 字典态，取 memory_read_result

    Returns:
        str: 真读到记忆时返回记忆段文本；本轮没读记忆、读到空结果或读取失败时返回空串
    """
    result = state.get("memory_read_result")
    if not isinstance(result, dict) or result.get("status") != _MEMORY_READ_SUCCESS:
        return ""
    items = result.get("items")
    if not isinstance(items, list) or not items:
        return ""
    return render_memory_prompt_block(items)


def _project_memory_lines(state: dict[str, Any]) -> list[str]:
    """把本轮读到的项目历史判断渲染进草案 prompt，供 analyzer 延续既有思路。

    Args:
        state: 当前 RAGState 字典态，取 memory_read_result

    Returns:
        list[str]: 真读到记忆时返回一段带身份约束的记忆文本；本轮没读记忆、
            读到空结果或读取失败时返回空列表，prompt 保持原样
    """
    block = render_project_memory_block(state)
    return [block] if block else []


def _replan_feedback_lines(state: dict[str, Any]) -> list[str]:
    """参数为当前研究状态；返回上轮缺口与逐单元读取事实，供重规划消费而不改变研究决策。"""
    # 重规划回边时把上一轮观察（校验失败码 / 检索缺口 / 已试检索式）回灌 analyzer
    # 首轮（无 planner_observation）返回空列表，不改原始 prompt
    observation = state.get("planner_observation")
    if not isinstance(observation, dict) or not observation:
        return []
    lines = ["", "Revision feedback from previous attempt (fix these, do not repeat):"]
    error_code = str(observation.get("error_code") or "").strip()
    if error_code:
        lines.append(f"- previous validation error: {error_code}")
    issue_codes = observation.get("issue_codes")
    if isinstance(issue_codes, list) and issue_codes:
        codes = ", ".join(str(code) for code in issue_codes if str(code or "").strip())
        if codes:
            lines.append(f"- issues to resolve: {codes}")
    summary = _compact_text(str(observation.get("summary") or ""), 300)
    if summary:
        lines.append(f"- detail: {summary}")
    attempted = observation.get("attempted_queries")
    if isinstance(attempted, list) and attempted:
        queries = "; ".join(str(query) for query in attempted[:6] if str(query or "").strip())
        if queries:
            lines.append(f"- already-tried retrieval queries (wording changes alone do not justify another attempt): {queries}")
    # 1. 【科研取证】【重规划报告】正式执行与补证的未读范围仍属工具事实，不能在重规划时丢失
    for result in state.get("research_unit_results") or []:
        reports = (result.get("metadata") or {}).get("read_reports") or []
        if reports:
            lines.extend([f"- reading reports for unit {result.get('unit_id', '')}:", render_read_reports(reports)])
    # 重申 facet 自洽规则，避免重规划再次踩 invalid_facet_id
    lines.append("- ensure every unit.covered_facet_ids entry is declared in the requested facets or the stable requirements baseline.")
    return lines


def probe_results_lines(state: Mapping[str, Any]) -> list[str]:
    """
    把已经取回的探测结果渲染成提示词段，供「下一次决策」据此判断还缺什么。

    出计划前的探测决策与出计划本身是两个决策，两边都必须看到累计结果：只看得到
    「第一轮问了什么」的决策会重复申请同一份上下文，而新一轮的申请理由本来就该建立
    在上一轮拿到了什么之上。两处共用本函数，免得同一份结果在两处各写一套渲染。

    四态如实照抄（success / empty / skipped / failed 各自独立），原因与读取报告一并
    给出。empty 仅表示零返回；模型须结合报告区分未读、未匹配和调用失败，再决定后续行动。

    结果字段走 dict 原样读、不在这里做 Pydantic 校验：写方是本仓的探测轮，
    结构由写方保证；缺字段时按缺字段如实渲染，不替它编一个默认值。

    Args:
        state: 当前 RAGState 字典态，取 planner_probe_results

    Returns:
        list[str]: 探测结果文本段；本轮没探测过或结果为空时返回空列表
    """
    results = state.get(PROBE_RESULTS_STATE_KEY)
    lines: list[str] = []
    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue
        lines.append(f"- {_probe_result_line(item)}")
        lines.extend(f"    - {finding}" for finding in _probe_findings(item))
        report = item.get("read_report")
        if isinstance(report, dict):
            lines.append(render_read_reports([report]))
    if not lines:
        return []
    return [
        "Context fetched before planning (asked for by you on the previous step):",
        *lines,
        "",
    ]


def _probe_result_line(item: dict[str, Any]) -> str:
    """渲染一条探测结果的表头行：来源、四态、申请的目标与筛选条件、申请用途、取不到时的原因。

    筛选条件一并渲染：模型重问同一来源、只是换了一组筛选条件时，只看四态分不出
    「上一轮要的是哪一块」，会照原样再要一次。
    """
    source_type = str(item.get("source_type") or "")
    status = str(item.get("status") or "")
    parts = [f"source_type={source_type}", f"status={status}"]
    filters = _probe_filters_text(item.get("requested_filters"))
    if filters:
        parts.append(filters)
    purpose = str(item.get("purpose") or "").strip()
    if purpose:
        parts.append(f"purpose={_compact_text(purpose, _PROBE_PURPOSE_MAX_CHARS)}")
    reason = str(item.get("reason") or "").strip()
    if reason:
        parts.append(f"reason={_compact_text(reason, _PROBE_REASON_MAX_CHARS)}")
    return " | ".join(parts)


def _probe_filters_text(raw: Any) -> str:
    """把一条申请的筛选条件压成单行；没有给定条件时返回空串。

    四个字段与 EvidencePolicy 逐字对齐，空列表原样跳过（「没给筛选条件」不是信息，
    渲染成 `required_sections=[]` 只是噪音）。
    """
    if not isinstance(raw, Mapping):
        return ""
    items: list[str] = []
    paper_ids = _probe_value_list(raw.get("target_paper_ids"))
    if paper_ids:
        items.append(f"target_paper_ids={','.join(paper_ids)}")
    for key, label in (
        ("required_sections", "required_sections"),
        ("required_block_types", "required_block_types"),
        ("required_pages", "required_pages"),
    ):
        values = _probe_value_list(raw.get(key))
        if values:
            items.append(f"{label}={','.join(values)}")
    return " ".join(items)


def _probe_value_list(raw: Any) -> list[str]:
    """把筛选条件的取值展成非空字符串列表；非列表输入返回空列表。"""
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item or "").strip()]


def _probe_findings(item: dict[str, Any]) -> list[str]:
    """参数为探测结果；返回保留表格与公式换行的发现文本，非列表或全空时返回空列表。

    历史宽度与截断说明已经保存在探测记录中，读取时保留原投影，
    这里只清理两端空白，不再压平结构或各写一套字符切片。
    """
    findings = item.get("findings")
    if not isinstance(findings, list):
        return []
    return [str(finding).strip() for finding in findings if str(finding or "").strip()]


def _paper_metadata_lines(state: dict[str, Any]) -> list[str]:
    """
    从状态字段读取论文清单，渲染到提示词。

    Args:
        state: 当前 RAGState 字典态，取 _paper_list_lines

    Returns:
        list[str]: 论文清单文本段；无清单时返回空列表
    """
    paper_list = state.get("_paper_list_lines")
    if not isinstance(paper_list, list) or not paper_list:
        return []
    return [
        "Project papers:",
        *paper_list,
        "",
    ]


def _section_structure_lines(state: dict[str, Any]) -> list[str]:
    """
    从状态字段读取各论文的章节路径清单，渲染到提示词。

    有了这份章节名单，模型铺计划时能直接指名章节，省掉「先发一次检索问章节存不存在、
    再出计划」的一次往返。章节正文不在这里给，由工具按需取。

    Args:
        state: 当前 RAGState 字典态，取 _section_structure_lines

    Returns:
        list[str]: 章节结构文本段；无章节数据时返回空列表
    """
    sections = state.get("_section_structure_lines")
    if not isinstance(sections, list) or not sections:
        return []
    return [
        "Paper sections (available for direct locator targeting):",
        *sections,
        "",
    ]


def _graph_context_lines(state: dict[str, Any]) -> list[str]:
    """
    从状态字段读取图谱状态，渲染到提示词。

    Args:
        state: 当前 RAGState 字典态，取 _graph_status_lines

    Returns:
        list[str]: 图谱状态文本段；无状态时返回空列表
    """
    graph_status = state.get("_graph_status_lines")
    if not isinstance(graph_status, list) or not graph_status:
        return []
    return [
        *graph_status,
        "",
    ]


def render_conversation_context(state: dict[str, Any]) -> str:
    """把最近几轮对话渲染成 prompt 片段。

    Args:
        state: 当前 RAGState 字典态，取 chat_history

    Returns:
        str: 用户原文片段与助手紧凑摘要；无可用历史时为 "(none)"
    """
    history = state.get("chat_history")
    if not isinstance(history, list):
        return "(none)"
    lines = []
    for item in history[-_HISTORY_MAX_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        # 1. 【研究规划】【可信原文】用户引文保留真实字符，截断说明放在原文之外
        if role == "user" and content.strip():
            lines.extend(["user (verbatim excerpt):", content[:_HISTORY_MAX_CHARS], "[end user excerpt]"])
            if len(content) > _HISTORY_MAX_CHARS:
                lines.append("[user excerpt truncated]")
        elif role == "assistant" and content.strip():
            lines.append(f"assistant: {_compact_text(content, _HISTORY_MAX_CHARS)}")
    return "\n".join(lines) if lines else "(none)"


def _previous_rag_context(state: dict[str, Any]) -> str:
    context = state.get("previous_rag_context")
    if not isinstance(context, dict) or not context:
        return "(none)"
    lines = [
        f"previous_query: {_compact_text(str(context.get('query') or ''), 300)}",
        f"previous_answer: {_compact_text(str(context.get('answer') or ''), 700)}",
    ]
    guidance = ""
    answer_brief = context.get("answer_brief")
    if isinstance(answer_brief, dict):
        guidance = str(answer_brief.get("answer_guidance") or "")
    if guidance:
        lines.append(f"previous_answer_guidance: {_compact_text(guidance, 400)}")
    lines.extend(_previous_evidence_lines(context.get("evidence_records")))
    return "\n".join(line for line in lines if line.strip())


def _previous_evidence_lines(records: Any) -> list[str]:
    if not isinstance(records, list):
        return []
    lines = []
    for record in records[:8]:
        if not isinstance(record, dict):
            continue
        evidence_id = str(record.get("evidence_id") or "")
        text = str(record.get("expanded_text") or record.get("excerpt") or record.get("summary") or "").strip()
        if evidence_id and text:
            lines.append(f"previous_evidence {evidence_id}: {_compact_text(text, 500)}")
    return lines


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit].rstrip()}..."
