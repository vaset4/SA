"""One-way planner serialization and legacy compatibility helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.models.research_planner_contracts import (
    CURRENT_RESEARCH_CONTRACT_VERSION,
    ClaimPolicy,
    PlannerCoverageReview,
    PlannerFailure,
    ResearchPlan,
    ResearchUnit,
    ResearchUnitResult,
)
from app.models.research_planner_draft_contracts import ClaimIntentDraft, ResearchPlanStrategyDraft, ResearchUnitStrategyDraft


def project_claim_policy(intents: list[ClaimIntentDraft]) -> ClaimPolicy:
    """参数为已验证的主张意图；返回原意图的唯一执行许可投影，不添加任何主张类型。"""
    allowed = [intent.claim_type for intent in intents]
    required = {intent.claim_type for intent in intents if intent.required}
    return ClaimPolicy(allowed_claim_types=allowed, citation_required="paper_supported" in required,
                       bounded_inference_allowed=bool(set(allowed) & {
                           "evidence_grounded_inference", "model_prior_hypothesis", "memory_continuation"}))


def project_strategy_units(
    drafts: list[ResearchUnitStrategyDraft], review: PlannerCoverageReview,
) -> tuple[list[ResearchUnit], dict[str, str]]:
    """参数为唯一ID策略和要求基准；返回语义不变的编号、依赖和章节关联投影。"""
    id_map = {unit.unit_id: f"u{index}" for index, unit in enumerate(drafts, start=1)}
    units = []
    for unit in drafts:
        source = unit.evidence_policy.source_type
        role = unit.output_role if source == "inference_from_units" else (
            None if source == "external_context_needed" else "evidence")
        dependencies = [{**dep.model_dump(), "unit_id": id_map[dep.unit_id]}
                        for dep in unit.dependencies]
        units.append(ResearchUnit.model_validate({
            "unit_id": id_map[unit.unit_id], "goal": unit.goal, "research_action": unit.research_action,
            "evidence_policy": unit.evidence_policy.model_dump(),
            "claim_policy": project_claim_policy(unit.claim_intents).model_dump(),
            "expected_output": {"output_goal": unit.output_goal, "deliverable_kind": unit.deliverable_kind,
                "required_claim_types": [intent.claim_type for intent in unit.claim_intents if intent.required],
                "validation_tasks": unit.validation_tasks, "answer_section_ids": [section.section_id
                    for section in review.answer_sections if set(section.facet_ids) & set(unit.covered_facet_ids)]},
            "output_role": role, "operation_family": unit.operation_family,
            "dependencies": dependencies, "covered_facet_ids": unit.covered_facet_ids,
        }, strict=True))
    return units, id_map


def migrate_execution_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    """参数为待继续的图状态；返回旧执行合同的显式重规划增量，历史最终重试不调用此入口。"""
    plan = state.get("research_plan") or state.get("planner_draft")
    if not state.get("research_plan") and isinstance(plan, dict) and "decision" in plan:
        # 1. 【研究循环】【草案恢复】要求可先于动作被接纳，旧草稿身份由自身字段决定，不由下一步基准反推
        from app.services.internal_research.research_action_validation import parse_research_decision
        parse_research_decision(plan, initial="requested_facets" in plan)
        return {}
    if isinstance(plan, dict) and plan.get("contract_version") == CURRENT_RESEARCH_CONTRACT_VERSION:
        return {}
    # 1. 【研究执行】【草案身份】新模型策略没有执行版本副本，恢复时按严格策略合同识别
    if not state.get("research_plan") and isinstance(plan, dict) and "requested_facets" in plan:
        ResearchPlanStrategyDraft.model_validate(plan, strict=True)
        return {}
    probes = state.get("planner_probe_results") or []
    legacy_probe = any("action_basis_keys" not in item for item in probes)
    history = (state.get("planner_budget") or {}).get("history") or []
    legacy_history = any(item.get("observation") and "facts_version" not in item["observation"] for item in history)
    if not isinstance(plan, dict) and not legacy_probe and not legacy_history:
        return {}
    version = plan.get("contract_version", "research-planner-v1") if isinstance(plan, dict) else "research-planner-v1"
    if version not in {"research-planner-v1", "research-planner-v2", "research-planner-v3"}:
        raise ValueError(f"不支持继续执行的研究合同版本：{version}")
    # 1. 【研究执行】【版本迁移】旧成果只读保留；缺少新版事实时重新规划，不补造完成度或凭据
    archived = {key: state.get(key) for key in (
        "research_plan", "planner_draft", "research_unit_results", "evidence_records", "planner_read_receipts",
        "planner_budget", "planner_probe_results", "evidence_sufficiency_report",
    )}
    return {
        "planner_execution_history": [*(state.get("planner_execution_history") or []), {
            "reason": "legacy_checkpoint_requires_replanning", "contract_version": version,
            "snapshot": archived,
        }],
        "research_plan": None, "planner_draft": None, "research_unit_results": [],
        "evidence_records": [], "planner_read_receipts": [], "planner_budget": {},
        "planner_probe_results": [], "planner_observation": {}, "planner_recovery_trace": [],
        "evidence_sufficiency_report": {}, "planner_loop_decision": "replan",
        "coverage_review": {}, "final_answer_eligibility_report": {},
        "final_delivery_report": {}, "final_citation_bindings": [],
        "final_generation_attempt": {}, "final_delivery_applicability": "not_applicable",
        "error_code": "", "error_summary": "",
        "planner_trace": {"version_transition": "legacy_checkpoint_requires_replanning"},
    }


# 诚实收尾正文（范围里确实有论文）：查过了、没命中，把"可能没涉及"的可能性交回用户判断
_SCOPED_MISS_ANSWER = (
    "在你指定的 {paper_count} 篇论文里检索过了，没找到关于「{question}」的内容，"
    "可能这几篇论文没涉及这个话题，或相关内容不在当前资料范围内。"
)
_SCOPED_MISS_SUMMARY = "在当前 {paper_count} 篇论文范围内未检索到相关证据"
# trace 里「定稿为何跳过」的归因（范围里有论文）：检索真发起过，只是没命中
_SCOPED_MISS_FINAL_REASON = "scoped 论文范围内检索无命中，诚实收尾"

# 诚实收尾正文（项目里一篇论文都没有）：先说清没有可核对的材料，再给"加论文"这个下一步动作
_EMPTY_PROJECT_ANSWER = (
    "这个项目里还没有论文，关于「{question}」我手上没有可以核对的材料，"
    "现在给不出有依据的回答。你先把相关论文加进这个项目，我再照论文原文来回答。"
)
_EMPTY_PROJECT_SUMMARY = "项目内还没有论文材料，无可核对依据"
# trace 里「定稿为何跳过」的归因（项目内无论文）：不能说「范围内检索无命中」——
# 压根没有 scoped 范围也没发起过检索，那么写等于让排查的人去查检索式为什么没召回
_EMPTY_PROJECT_FINAL_REASON = "项目内没有论文，本轮无可核对材料，诚实收尾"


@dataclass(frozen=True, kw_only=True)
class PlannerFailureReadFacts:
    """【研究失败】【读取事实】由本轮回执派生的只读说明，不是新的运行状态。"""

    known: bool
    stage: str
    execution_started: bool
    actual_call_count: int | None = None
    has_successful_read: bool | None = None
    unknown_reason: str = ""


def project_planner_failure_read_facts(
    payload: dict[str, Any], facts: PlannerFailureReadFacts,
) -> dict[str, Any]:
    """参数为新失败载荷和本轮读取事实；返回正文、状态与诊断一致的投影，保留原交付字段。"""
    status = dict(payload.get("source_status") or {})
    trace = {**(payload.get("planner_trace") or {}), "read_facts": asdict(facts)}
    # 1. 【研究失败】【事实投影】未知不补造零调用；同一事实同时控制正文、状态和诊断
    if facts.known:
        local_status = "attempted" if facts.actual_call_count else "not_run"
        status["local_evidence_execution"] = local_status
        trace["local_evidence_execution"] = local_status
    else:
        status.pop("local_evidence_execution", None)
        trace.pop("local_evidence_execution", None)
    note = _failure_execution_note(facts)
    return {**payload, "answer": f"{payload.get('answer', '')}{note}",
            "source_status": status, "planner_trace": trace}


def build_planner_failure_response(failure: PlannerFailure) -> dict[str, Any]:
    """
    构建用户可见的规划失败载荷。

    增量字段 error_code / error_summary 供 API 与流式终态稳定消费；
    answer 仍承载完整用户句，兼容只读 answer 的现网前端。

    参数:
        failure: PlannerFailure 契约对象
    返回:
        失败响应字典
    """
    # 1. 稳定错误码与短摘要：优先 failure.code / message，不堆整段校验栈
    error_code = str(failure.code or "planner_failed")
    error_summary = _planner_error_summary(failure)
    return {
        "answer": f"可靠研究规划失败：{error_summary}。",
        "error_code": error_code,
        "error_summary": error_summary,
        "research_plan": None,
        "research_unit_results": [],
        "coverage_review": None,
        "planner_trace": failure.model_dump(mode="json"),
        "runtime_chain": "planner",
        "source_status": {
            "planner": "failed",
            "planner_repair": "skipped",
            "planner_final": "skipped",
            "local_evidence_execution": failure.local_evidence_execution,
        },
        "source_status_reasons": {
            "planner_repair": "planner failed before evidence repair",
            "planner_final": "planner failed before final answer generation",
        },
        "planner_final_blocks": [],
        "planner_final_validation_errors": [],
        "repair_query_plans": [],
        "planner_repair_trace": {},
        "planner_repair_merge_report": {},
        "final_answer_eligibility_report": {},
        "evidence_sufficiency_report": {},
        "missing_evidence": [],
    }


def build_evidence_not_found_response(
    *,
    question: str,
    scoped_paper_ids: list[str],
    authorized_paper_count: int,
    attempted_queries: list[str],
    missing_evidence: list[dict[str, Any]],
    scoped_paper_titles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    构建"范围内查过但没找到证据"的诚实收尾载荷。

    区别于 build_planner_failure_response：这是流程正常走完、确实检索过后的诚实结论，
    不是系统硬失败。模板确定性组装、不调 LLM（慢速 replan 后也一定跑得完、可测）。
    正文只给用户能懂的结论；attempted_queries / missing_evidence / scoped 论文列表
    作结构化字段返回供前端折叠追溯，正文与摘要都不糊检索式（本用户不懂技术）。

    参数:
        question: 用户原问题，嵌进正文结论
        scoped_paper_ids: 计划实际去检索的论文，只作结构化追溯用
        authorized_paper_count: 本轮用户授权/聚焦的论文数，决定正文说哪一档 ——
            这个数来自项目真实范围，不能用 scoped_paper_ids 的长度顶替：后者是
            模型草案点名的目标论文，项目里明明有论文而草案没点名时长度会是 0，
            照它说话就变成「这个项目里还没有论文」这句假话
        attempted_queries: 实际尝试过的检索式（复用 derive_missing_evidence 的产出）
        missing_evidence: 缺失证据记录（缺哪些维度），作结构化字段透出
        scoped_paper_titles: paper_id → 标题映射，缺失则回退到 paper_id
    返回:
        诚实收尾响应字典，error_code=evidence_not_found_in_scope
    """
    # 1. 正文说人话：只给能懂的结论，不塞检索式；项目里一篇论文都没有时换成"缺材料"的说法
    compact_question = " ".join(str(question or "").split())[:120] or "该问题"
    answer, error_summary, final_reason = _evidence_not_found_texts(
        authorized_paper_count, compact_question
    )
    # 2. scoped 论文列表带标题供前端展示与追溯
    titles = scoped_paper_titles or {}
    scoped_papers = [
        {"paper_id": paper_id, "title": titles.get(paper_id) or paper_id}
        for paper_id in scoped_paper_ids
    ]
    return {
        "answer": answer,
        "error_code": "evidence_not_found_in_scope",
        "error_summary": error_summary,
        "research_plan": None,
        "research_unit_results": [],
        "coverage_review": None,
        "planner_trace": {},
        "runtime_chain": "planner",
        "source_status": {
            "planner": "no_evidence",
            "planner_repair": "attempted",
            "planner_final": "skipped",
            "local_evidence_execution": "attempted",
        },
        "source_status_reasons": {
            "planner_final": final_reason,
        },
        "planner_final_blocks": [],
        "planner_final_validation_errors": [],
        "repair_query_plans": [],
        "planner_repair_trace": {},
        "planner_repair_merge_report": {},
        "final_answer_eligibility_report": {},
        "evidence_sufficiency_report": {},
        "evidence_records": [],
        "missing_evidence": list(missing_evidence),
        # 结构化追溯字段：检索式与 scoped 论文供前端折叠展示
        "attempted_queries": list(attempted_queries),
        "scoped_papers": scoped_papers,
    }


def build_clarification_response(*, question: str) -> dict[str, Any]:
    """
    构建"先问用户一句"的澄清载荷：那句问题本身就是本轮回答。

    区别于 build_evidence_not_found_response：这一轮压根没去检索，也不是查过没找到，
    而是目标没确认、不同理解会导致明显不同动作，所以先问再做（prd.md:100）。
    澄清是正常终止而不是失败，所以 error_code / error_summary 留空；模板确定性组装、
    不调 LLM —— 问题文本模型在深度评估那一次调用里已经给了，再调一次只会改写它。

    参数:
        question: 模型给出的那一句澄清问题，直接作为用户可见正文返回。
            调用方保证非空（信任边界上 needed=True 就必须带问题），这里不补默认问句：
            补一句"能再说清楚一点吗"会把模型没判出要问什么这件事盖掉
    返回:
        澄清响应字典，键集合与 build_evidence_not_found_response 对齐
    """
    return {
        "answer": question,
        # 1. 澄清是正常终止：既不是硬失败，也不是查过没找到的诚实收尾，两个错误位留空
        "error_code": "",
        "error_summary": "",
        "research_plan": None,
        "research_unit_results": [],
        "coverage_review": None,
        "planner_trace": {},
        "runtime_chain": "planner",
        "source_status": {
            # 本轮没铺计划也没取论文证据：skipped 会被四态换算表读成
            # retrieve_paper_evidence: skipped，而不是从能力表里整条消失
            "planner": "skipped",
            "planner_repair": "skipped",
            "planner_final": "skipped",
            "local_evidence_execution": "not_run",
        },
        "source_status_reasons": {
            "planner_final": "本轮先向用户澄清目标，未取论文证据、未走定稿引用校验",
        },
        # 2. 引用与证据相关的键全留空：这一轮一次检索都没发起，没有任何出处可挂
        "planner_final_blocks": [],
        "planner_final_validation_errors": [],
        "repair_query_plans": [],
        "planner_repair_trace": {},
        "planner_repair_merge_report": {},
        "final_answer_eligibility_report": {},
        "evidence_sufficiency_report": {},
        "evidence_records": [],
        "missing_evidence": [],
        "attempted_queries": [],
        "scoped_papers": [],
    }


def _evidence_not_found_texts(paper_count: int, question: str) -> tuple[str, str, str]:
    """
    按论文数选诚实收尾的正文、摘要与 trace 归因措辞。

    空项目（论文数 0）与"有论文但没命中"是两件事：前者根本没有可核对的材料，
    说"这几篇论文没涉及"是在指代不存在的论文，用户也不知道下一步该干什么。
    trace 归因同样要分：空项目压根没有 scoped 范围，说"范围内检索无命中"是虚报
    一次并不存在的检索，排查的人会照着去查检索式为什么没召回。

    参数:
        paper_count: 本轮 scoped 论文数
        question: 已压空白并截断的用户问题
    返回:
        (answer 正文, error_summary 短摘要, source_status_reasons.planner_final 归因)
    """
    # 1. 项目里没有论文：说明缺可核对材料，并指向"往项目里加论文"这个下一步
    if paper_count == 0:
        return (
            _EMPTY_PROJECT_ANSWER.format(question=question),
            _EMPTY_PROJECT_SUMMARY,
            _EMPTY_PROJECT_FINAL_REASON,
        )
    # 2. 范围内确实有论文：保持"查过了没找到"的原措辞
    return (
        _SCOPED_MISS_ANSWER.format(paper_count=paper_count, question=question),
        _SCOPED_MISS_SUMMARY.format(paper_count=paper_count),
        _SCOPED_MISS_FINAL_REASON,
    )


def _planner_error_summary(failure: PlannerFailure) -> str:
    """
    生成 1～2 句用户可读失败摘要。

    参数:
        failure: PlannerFailure
    返回:
        短中文摘要字符串
    """
    message = str(failure.message or "").strip()
    if message:
        # 截断过长技术文案，避免整段 ValidationError 进 answer
        compact = " ".join(message.split())
        if len(compact) > 240:
            return f"{compact[:240]}..."
        return compact
    return "研究规划失败"


def _failure_execution_note(facts: PlannerFailureReadFacts) -> str:
    """参数为可核验读取事实及失败阶段；返回不混淆规划读取、正式执行和任务完成的说明。"""
    if not facts.known:
        return "研究未完成。"
    # 1. 【研究失败】【调用事实】只有实际发出且成功的读取支持“已读到内容”
    if not facts.actual_call_count:
        reading = "本轮尚未调用本地资料读取工具。"
    elif facts.has_successful_read:
        reading = "本轮已读取部分本地资料。" if facts.execution_started else "规划阶段已读取部分本地资料。"
    else:
        reading = "本轮已尝试查询本地资料。" if facts.execution_started else "规划阶段已尝试查询本地资料。"
    if facts.execution_started:
        stage = "综合回答" if facts.stage == "synthesize" else "后续处理"
        return f"{reading}正式研究执行已开始，但{stage}失败，研究未完成。"
    if facts.stage in {"analyze", "validate"}:
        return f"{reading}研究计划未能完成。"
    return f"{reading}研究未完成。"


def to_message_payload(
    plan: ResearchPlan,
    results: list[ResearchUnitResult],
) -> dict[str, Any]:
    """Serialize planner truth payloads for additive assistant-message fields."""
    return {
        "research_plan": plan.model_dump(mode="json"),
        "research_unit_results": [result.model_dump(mode="json") for result in results],
        "coverage_review": plan.coverage_review.model_dump(mode="json"),
        "planner_trace": plan.metadata.model_dump(mode="json"),
    }


def downgrade_to_compat_payload(
    plan: ResearchPlan,
    results: list[ResearchUnitResult],
) -> dict[str, Any]:
    """Derive legacy TypedDict-shaped sinks from planner outputs."""
    statuses = {result.unit_id: result.status for result in results}
    unresolved = [r for r in results if r.status in {"blocked", "insufficient_evidence", "external_needed"}]
    return {
        "synthesis_state": _build_compat_synthesis_state(plan, results, statuses),
        "answer_brief": _build_compat_answer_brief(plan, unresolved),
    }


def reject_compat_payload_readback(payload: dict[str, Any]) -> None:
    """Make the derived-sink rule executable in tests and future wiring."""
    raise ValueError("compatibility payloads are derived sinks, not planner inputs")


def ingest_legacy_evidence_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consume trusted legacy EvidenceRecord dicts without Pydantic wrapping."""
    return [dict(record) for record in records if isinstance(record, dict)]


def _build_compat_synthesis_state(
    plan: ResearchPlan,
    results: list[ResearchUnitResult],
    statuses: dict[str, str],
) -> dict[str, Any]:
    unresolved = {"blocked", "insufficient_evidence", "external_needed"}
    return {
        "status": "ready" if any(r.status in {"evidence_found", "inferred"} for r in results) else "insufficient",
        "dialogue_mode": "research",
        "question_type": "planner",
        "project_scope": {},
        "paper_focus": {},
        "dimension_status": statuses,
        "supported_dimensions": [u.goal for u in plan.units if statuses.get(u.unit_id) not in unresolved],
        "unsupported_dimensions": [u.goal for u in plan.units if statuses.get(u.unit_id) in unresolved],
        "open_questions": [result.unresolved_reason for result in results if result.unresolved_reason],
        "evidence_layers": [],
        "notes": ["derived from planner results"],
        "citations_count": sum(len(result.evidence_ids) for result in results),
    }


def _build_compat_answer_brief(
    plan: ResearchPlan,
    unresolved: list[ResearchUnitResult],
) -> dict[str, Any]:
    return {
        "intent": plan.plan_type,
        "focus": ", ".join(_target_paper_ids(plan.units)),
        "usable_evidence": [],
        "missing_evidence": [
            {"dimension": result.unit_id, "reason": result.unresolved_reason or result.status}
            for result in unresolved
        ],
        "answer_guidance": "Use validated planner unit claims only.",
    }


def _target_paper_ids(units: list[ResearchUnit]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for unit in units:
        for paper_id in unit.evidence_policy.target_paper_ids:
            if paper_id not in seen:
                seen.add(paper_id)
                ordered.append(paper_id)
    return ordered
