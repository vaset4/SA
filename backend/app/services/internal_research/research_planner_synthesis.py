"""【研究交付】【正式综合】同次正文、引用、报告和现有消息投影。作者：xxx。"""

from __future__ import annotations

from typing import Any, Mapping
from dataclasses import replace
from time import perf_counter

from app.models.research_planner_contracts import ResearchPlan, ResearchUnitResult
from app.services.llm_service import provider_failure_message
from app.services.internal_research.research_planner_conversion import downgrade_to_compat_payload, to_message_payload
from app.services.internal_research.research_planner_final_answer import validate_planner_final_answer
from app.services.internal_research.research_planner_final_contracts import (
    PlannerFinalAnswerInput, PlannerFinalAnswerProvider, PlannerFinalProviderOutput, PlannerFinalValidationOutcome,
    REVISION_DELIVERY_POLICY_VERSION, FinalDeliveryReport,
)
from app.services.internal_research.research_planner_final_delivery import delivery_state_fields
from app.services.internal_research.research_planner_final_evidence import final_product_catalog
from app.services.internal_research.research_planner_final_provider import prepare_final_input
from app.services.internal_research.research_answer_review import review_final_answer
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewer
from app.services.internal_research.research_planner_consumer_budget import content_hash
from app.services.internal_research.research_planner_final_revision import (
    apply_final_patch, candidate_revision_request, delivery_next_action, make_candidate, research_snapshot_digest,
    FinalRevisionRejected, candidate_body_digest, candidate_normalized_draft,
)
from app.services.llm_observation import record_business_validation

_SUCCESS_STATUSES = frozenset({"evidence_found", "inferred"})
_UNRESOLVED_STATUSES = frozenset({"insufficient_evidence", "external_needed", "blocked"})
_REPAIR_FIELDS = ("evidence_sufficiency_report", "repair_query_plans", "planner_repair_trace",
                 "planner_repair_merge_report", "final_answer_eligibility_report")
_CLAIM_LABELS = {"paper_supported": "论文来源主张", "evidence_grounded_inference": "证据推断",
                 "model_prior_hypothesis": "待验证假设", "memory_continuation": "项目历史判断（非论文原文）",
                 "external_needed": "待补外部背景"}
_QUESTION_PREVIEW_MAX_CHARS = 120


def synthesize_planner_answer(plan: ResearchPlan, results: list[ResearchUnitResult], *,
    project_paper_count: int, evidence_records: list[dict[str, Any]] | None = None,
    final_answer_draft: Any = None, final_provider_error: str = "",
    repair_context: Mapping[str, Any] | None = None, final_entrypoint: str = "normal",
    historical_input_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """参数为本次计划、成果、草稿和入口；返回正式状态载荷，原始证据与交付分别保留。"""
    records = list(evidence_records or [])
    payload = prepare_final_input(PlannerFinalAnswerInput(
              question=str((repair_context or {}).get("user_query") or plan.question), plan=plan, unit_results=results,
              evidence_records=records, entrypoint=final_entrypoint,
              eligibility_report=dict((repair_context or {}).get("final_answer_eligibility_report") or {}),
              historical_input_context=dict(historical_input_context or {})))
    final = validate_planner_final_answer(payload, final_answer_draft, provider_error=final_provider_error)
    return _assemble_delivery(payload, final, dict(repair_context or {}), project_paper_count=project_paper_count)


async def synthesize_planner_answer_async(plan: ResearchPlan, results: list[ResearchUnitResult], *,
    project_paper_count: int, evidence_records: list[dict[str, Any]] | None = None,
    final_provider: PlannerFinalAnswerProvider | None = None, repair_context: Mapping[str, Any] | None = None,
    final_entrypoint: str = "normal", historical_input_context: dict[str, Any] | None = None,
    answer_reviewer: ResearchAnswerReviewer | None = None,
) -> dict[str, Any]:
    """参数为成果、注入模型和入口；一次准备输入后生成并校验，不按推荐角色或资格标签挡住部分回答。"""
    payload = prepare_final_input(PlannerFinalAnswerInput(
              question=str((repair_context or {}).get("user_query") or plan.question), plan=plan, unit_results=results,
              evidence_records=list(evidence_records or []), entrypoint=final_entrypoint,
              eligibility_report=dict((repair_context or {}).get("final_answer_eligibility_report") or {}),
              revision_context=dict((repair_context or {}).get("final_delivery_report") or {}),
              historical_input_context=dict(historical_input_context or {})))
    candidate = (dict((repair_context or {}).get("planner_final_candidate") or {})
                 if final_entrypoint in {"normal", "research_honest"} else {})
    if candidate:
        payload = replace(payload, delivery_policy_version=REVISION_DELIVERY_POLICY_VERSION)
        candidate = _refresh_candidate(payload, candidate)
    draft: Any = None
    provider_error = ""
    revision_rejected = False
    # 1. 【研究交付】【生成准入】材料、事实和明确待验证产物均按真实任务进入最终组织
    if final_provider is not None and final_product_catalog(payload).has_business_input:
        try:
            draft = await _generate_candidate(payload, final_provider, candidate)
            if draft is None:
                draft = PlannerFinalProviderOutput(payload=None)
        except Exception as exc:
            # 1. 【研究失败】【供应商说明】最终交付使用同一安全摘要，原技术异常仍由客户端记录
            provider_error = provider_failure_message(exc, include_type=True)
            revision_rejected = isinstance(exc, FinalRevisionRejected)
    # 2. 【研究交付】【同次校验】复用送模的实际投影，不用新预算重算当时可见范围
    final = validate_planner_final_answer(payload, draft, provider_error=provider_error)
    if candidate and not provider_error and candidate_body_digest([block.model_dump(mode="json") for block in final.blocks]) in (
            candidate["body_digest"], *candidate.get("previous_body_digests", [])):
        provider_error, revision_rejected = "正文生成重复已有候选，没有新进展", True
        final = validate_planner_final_answer(payload, None, provider_error=provider_error)
    if provider_error and candidate:
        final = _failed_revision(payload, candidate, final, rejected=revision_rejected)
    # 3. 【研究交付】【实质核对】仅新式研究候选正文进入内容核对，历史重开和minimal没有此调用
    if (plan.metadata.execution_mode == "incremental" and final.answer and not historical_input_context
            and not final.errors and not any(row.required and row.status == "missing"
                                             for row in final.delivery_report.section_results)):
        final = await review_final_answer(payload, final, answer_reviewer)
    delivery = _assemble_delivery(payload, final, dict(repair_context or {}), project_paper_count=project_paper_count)
    if final_entrypoint in {"normal", "research_honest"} and (
            final.delivery_report.delivery_policy_version == REVISION_DELIVERY_POLICY_VERSION or candidate):
        # 4. 【研究交付】【候选保留】补丁失败保留整份基稿；失败报告独立暴露，不能局部覆盖原候选
        source = draft.source_payload if isinstance(draft, PlannerFinalProviderOutput) and draft.source_schema else _raw_output(draft)
        current = candidate if provider_error and candidate else make_candidate(payload, final, _raw_output(draft))
        if not provider_error:
            current["raw_output"] = source
        if candidate and not provider_error:
            current["previous_body_digests"] = [*candidate.get("previous_body_digests", []), candidate["body_digest"]]
        delivery["planner_final_candidate"] = current
        delivery["planner_loop_decision"] = "terminate_fail" if provider_error else delivery_next_action(current)
    return delivery


def _raw_output(draft: Any) -> Any:
    """参数为模型边界输出；返回原始模型载体供候选历史追溯，不清洗错误字段。"""
    return draft.payload if isinstance(draft, PlannerFinalProviderOutput) else draft


def _refresh_candidate(payload: PlannerFinalAnswerInput, candidate: dict[str, Any]) -> dict[str, Any]:
    """参数为新研究成果和旧候选；研究快照变化时重验原块资格，旧通过结论不能复用。"""
    if candidate["research_snapshot_digest"] == research_snapshot_digest(payload):
        return candidate
    raw = candidate_normalized_draft(candidate)
    final = validate_planner_final_answer(replace(payload, attempt_id=candidate["attempt_id"]), raw)
    refreshed = make_candidate(payload, final, raw)
    refreshed["raw_output"] = candidate["raw_output"]
    refreshed["revision_feedback"] = candidate["report"].get("content_review") or {}
    return refreshed


def _failed_revision(payload: PlannerFinalAnswerInput, candidate: dict[str, Any],
                     failure: PlannerFinalValidationOutcome, *, rejected: bool) -> PlannerFinalValidationOutcome:
    """参数为修复失败与未改基稿；返回可读失败正文及同次重建引用，不冒充修复成功。"""
    raw = candidate_normalized_draft(candidate)
    base = validate_planner_final_answer(payload, raw)
    previous_report = FinalDeliveryReport.model_validate(candidate["report"])
    errors = [error.model_copy(update={"code": "revision_rejected"}) for error in failure.errors] if rejected else failure.errors
    reasons = [reason.model_copy(update={"source": "revision_contract", "code": "revision_rejected"})
               for reason in failure.delivery_report.reasons] if rejected else failure.delivery_report.reasons
    # 1. 【研究交付】【失败保留】正文未更新，原缺失事实仍成立；只为失败尝试重建引用身份，不复用旧通过结论
    report = base.delivery_report.model_copy(update={"status": "failed", "generation_status": "failed",
        "requirement_results": previous_report.requirement_results, "section_results": previous_report.section_results,
        "validation_errors": errors, "reasons": [*reasons, *previous_report.reasons]})
    return replace(base, status="failed", errors=errors, delivery_report=report,
                   failure_reason=failure.failure_reason)


async def _generate_candidate(payload: PlannerFinalAnswerInput, provider: PlannerFinalAnswerProvider,
                              candidate: dict[str, Any]) -> Any:
    """参数为同次输入、模型端口和可选基稿；执行一次局部或显式整体生成，记录真实改稿量。"""
    if not candidate or candidate_revision_request(candidate)["revision_scope"] == "whole":
        return await provider.generate(payload)
    response = await provider.revise(payload, candidate)
    if isinstance(response, PlannerFinalProviderOutput) and response.parsing_error:
        raise FinalRevisionRejected(f"正文补丁合同失败：{response.parsing_error}")
    raw = _raw_output(response)
    started = perf_counter()
    try:
        result = apply_final_patch(payload, candidate, raw)
    except ValueError as exc:
        record_business_validation(raw, {"validator": "planner_final_revision", "schema": "PlannerFinalPatch",
            "valid": False, "accepted": False, "base_attempt_id": candidate["attempt_id"],
            "attempt_id": payload.attempt_id, "error": str(exc)})
        raise FinalRevisionRejected(str(exc)) from exc
    old_ids = {block["block_id"] for block in candidate["blocks"]}
    retained = [block for block in result["blocks"] if block["block_id"] in old_ids]
    changed = [block for block in result["blocks"] if block["block_id"] not in old_ids]
    record_business_validation(raw, {"validator": "planner_final_revision", "schema": "PlannerFinalPatch",
        "valid": True, "accepted": True, "base_attempt_id": candidate["attempt_id"], "attempt_id": payload.attempt_id,
        "application_ms": (perf_counter() - started) * 1000,
        "output_kind": "local_patch" if retained else "whole_answer", "changed_block_count": len(changed),
        "retained_block_count": len(retained), "changed_text_chars": sum(len(block["text"]) for block in changed),
        "retained_text_chars": sum(len(block["text"]) for block in retained)})
    return PlannerFinalProviderOutput(payload=result, source_payload=raw, source_schema="PlannerFinalPatch")


def _assemble_delivery(payload: PlannerFinalAnswerInput, final: PlannerFinalValidationOutcome,
                       repair: Mapping[str, Any], *, project_paper_count: int) -> dict[str, Any]:
    """参数为同次输入、最终校验和取证修复事实；返回一致的正文、引用与交付元数据。"""
    plan, results, records = payload.plan, payload.unit_results, payload.evidence_records
    compat = downgrade_to_compat_payload(plan, results)
    message = to_message_payload(plan, results)
    reads = _executed_read_summary(repair)
    source_status = _source_status(results)
    if "planner_read_receipts" in repair:
        source_status["local_evidence_execution"] = reads["local_evidence_execution"]
    source_status.update(planner_final=final.status, planner_repair=str(repair.get("planner_repair_status") or "not_needed"))
    reasons: dict[str, str] = {}
    if final.status != "ok":
        reasons["planner_final"] = final.failure_reason
    if repair.get("planner_repair_reason"):
        reasons["planner_repair"] = str(repair["planner_repair_reason"])
    cited_ids = {binding.evidence_id for binding in final.delivery_report.citation_bindings}
    citations = [dict(record) for record in records if record.get("evidence_id") in cited_ids
                 and record.get("layer") == "direct_text" and record.get("usable_as_citation")]
    synthesis_state = dict(compat.get("synthesis_state") or {})
    synthesis_state["planner_claims"] = [item.claim.model_dump(mode="json")
        for item in final_product_catalog(payload).supports.values() if item.qualified and not item.hypothesis]
    missing = list(repair.get("missing_evidence") or [])
    answer_brief = dict(compat.get("answer_brief") or {})
    if missing:
        answer_brief["missing_evidence"] = missing
    answer = _display_answer(payload, final, repair, reads=reads, project_paper_count=project_paper_count)
    if final.delivery_report.content_review is not None:
        # 1. 【研究交付】【整稿绑定】程序追加风险说明后绑定完整正文，保存时不接受任意新增尾文
        review = final.delivery_report.content_review.model_copy(update={"delivery_answer_digest": content_hash(answer)})
        final = replace(final, delivery_report=final.delivery_report.model_copy(update={"content_review": review}))
    return {
        **compat, **message, **delivery_state_fields(final.delivery_report),
        "answer": answer,
        "answer_confidence": final.confidence or "low", "source_status": source_status,
        "source_status_reasons": reasons, "evidence_summary": "", "evidence_records": records,
        "missing_evidence": missing, "citations": citations, "chunk_citations": citations,
        "scoped_papers": reads["scoped_papers"], "attempted_queries": reads["attempted_queries"],
        "paper_sources": [{"paper_id": paper_id} for paper_id in dict.fromkeys(
                          str(record["paper_id"]) for record in records if record.get("paper_id"))],
        "graph_evidence": [], "answer_brief": answer_brief, "synthesis_state": synthesis_state,
        "planner_final_blocks": [block.model_dump(mode="json") for block in final.blocks],
        "planner_final_validation_errors": [error.model_dump(mode="json") for error in final.errors],
        **{key: repair[key] for key in _REPAIR_FIELDS if key in repair},
    }


def _executed_read_summary(context: Mapping[str, Any]) -> dict[str, Any]:
    """参数为真实读取凭据；返回已尝试请求的范围、查询和本地执行事实，不从计划补造。"""
    papers: list[str] = []
    queries: list[str] = []
    local_attempted = False
    for receipt in context.get("planner_read_receipts", []):
        # 1. 【研究交付】【执行追溯】失败或取消仍保留已发生的调用，未调用记录不授予尝试事实
        if receipt["actual_call_count"] <= 0:
            continue
        if receipt["source_type"].startswith("local_"):
            local_attempted = True
        request = receipt.get("request")
        if request is None:
            continue
        # 2. 【研究交付】【范围追溯】只消费实际提交的请求；动态请求未保存时保持范围和查询未知
        if request.get("operation") == "search_chunks":
            papers.extend(request["structured_query"].get("paper_ids", []))
            queries.extend(request["queries"])
        elif request.get("operation") == "sections":
            papers.extend(request["paper_ids"])
    return {
        "scoped_papers": [{"paper_id": paper, "title": paper} for paper in dict.fromkeys(papers)],
        "attempted_queries": list(dict.fromkeys(queries)),
        "local_evidence_execution": "attempted" if local_attempted else "not_run",
    }


def _display_answer(payload: PlannerFinalAnswerInput, final: PlannerFinalValidationOutcome,
                     repair: Mapping[str, Any], *, reads: Mapping[str, Any], project_paper_count: int) -> str:
    """参数为本次实际产物和校验；保留失败正文，追加风险时不改变已经生成的引用坐标。"""
    if final.answer.strip():
        answer = final.answer
    else:
        answer = _undelivered_description(payload, final, reads=reads, project_paper_count=project_paper_count)
    notes: list[str] = []
    if final.status in {"partial", "failed"}:
        notes.append("【注意】本次回答存在未完成事项或校验问题，请结合来源和风险提示核对。")
    eligibility = repair.get("final_answer_eligibility_report")
    if isinstance(eligibility, dict) and eligibility.get("can_call_provider") is False:
        notes.append("【注意】已有取证仍有缺口，当前正文保留供参考。")
    # 1. 【研究交付】【引用位置】风险只追加在末尾，绝不strip或重写已绑定的原始正文
    return answer + ("\n\n" + "\n".join(notes) if notes else "")


def _undelivered_description(payload: PlannerFinalAnswerInput, final: PlannerFinalValidationOutcome, *,
                              reads: Mapping[str, Any], project_paper_count: int) -> str:
    """参数为已得成果与终止事实；返回明确失败/不足说明，不把保存成果冒充本次成功生成。"""
    claims = [claim for result in payload.unit_results for claim in result.claims]
    if final.status == "failed":
        lead = "本次最终回答未能成功交付；已取得的材料仍保留，可以重新生成。"
    elif payload.evidence_records:
        lead = "已取得部分资料，但尚未取得能满足本次要求的业务成果。"
    elif reads["scoped_papers"]:
        lead = f"本次已尝试在 {len(reads['scoped_papers'])} 篇论文范围内查找，但尚未找到能支撑回答的材料。"
    elif project_paper_count <= 0:
        lead = "这个项目里还没有可核对的论文材料，请先补充相关论文。"
    else:
        lead = "本次尚未找到能支撑回答的材料，可以调整问题或补充相关论文。"
    # 1. 【研究交付】【问题上下文】不足说明保留用户原问题，内部检索式仅进入结构化追溯字段
    question = " ".join(payload.question.split())[:_QUESTION_PREVIEW_MAX_CHARS]
    if question:
        lead = f"关于“{question}”：{lead}"
    if not claims:
        return lead
    lines = [lead, "", "已保存的研究成果（不代表本次已经完成回答）："]
    lines.extend(f"- {_CLAIM_LABELS.get(claim.claim_type, '待核验判断')}：{claim.text}" for claim in claims)
    lines.append("以上保存内容仍可能含待验证假设或无效支持，请结合原始材料核对。")
    return "\n".join(lines)


def _source_status(results: list[ResearchUnitResult]) -> dict[str, str]:
    """参数为真实单元结果；只投影取证执行状态，不因最终回答失败改写取得事实。"""
    has_success = any(result.status in _SUCCESS_STATUSES for result in results)
    has_unresolved = any(result.status in _UNRESOLVED_STATUSES for result in results)
    if has_success and has_unresolved:
        return {"planner": "partial"}
    return {"planner": "ok" if has_success else "insufficient"}
