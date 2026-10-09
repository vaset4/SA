"""Final-only retry for persisted planner runs."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.models.result_delivery_contracts import DeliveryConflict
from app.services.result_delivery_service import deliver_result
from app.services.rag_run_executor_service import get_rag_run_executor_registry

from app.models.research_planner_contracts import CURRENT_RESEARCH_CONTRACT_VERSION, ResearchPlan, ResearchUnitResult
from app.services.internal_research.research_planner_validation_rules import fulfillment_contract_issues
from app.repositories.common import utc_now
from app.repositories.paper_repository import PaperRepository, get_paper_repository
from app.services.internal_research.evidence_identity import register_evidence_records
from app.services.internal_research.research_planner_final_delivery import (
    delivery_report_from_state, delivery_state_fields, project_final_delivery,
)
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.rag_run_repository import RAGRunRepository
from app.repositories.result_delivery_repository import ResultDeliveryRepository
from app.services import conversation_service, rag_run_service
from app.services.internal_research.research_planner_final_answer import PlannerFinalAnswerProvider
from app.services.internal_research.research_planner_final_provider import LLMPlannerFinalAnswerProvider
from app.services.internal_research.research_planner_observability import count_rejected_final_blocks
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer_async
from app.services.llm_service import get_llm
from app.services.internal_research.research_answer_review import LLMResearchAnswerReviewer
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReviewer

_PLANNER_TRUTH_KEYS = (
    "source_status",
    "source_status_reasons",
    "research_plan",
    "research_unit_results",
    "evidence_records",
    "coverage_review",
    "planner_trace",
    "evidence_sufficiency_report",
    "repair_query_plans",
    "planner_repair_trace",
    "planner_repair_merge_report",
    "final_answer_eligibility_report",
    "missing_evidence",
    "trace_summary",
    "turn_depth",
    "turn_outcome",
    "planner_final_retry_attempts",
)


async def retry_planner_final_answer(
    *, user_id: str, conversation_id: str, run_id: str, idempotency_key: str | None = None,
    final_provider: PlannerFinalAnswerProvider | None = None,
    answer_reviewer: ResearchAnswerReviewer | None = None,
    run_repository: RAGRunRepository | None = None,
    conversation_repository: ConversationRepository | None = None,
    paper_repository: PaperRepository | None = None,
    delivery_repository=None, executor=None,
) -> dict[str, Any]:
    """【研究交付】【最终重试】参数为原答案与依赖端口；领取轻量操作后生成并整体提交。

    只使用保存成果，不读取图 checkpoint，操作不成为新的用户提问。
    """
    run = await _load_run(user_id=user_id, conversation_id=conversation_id, run_id=run_id, repository=run_repository)
    message = await _load_message(user_id=user_id, conversation_id=conversation_id, run_id=run_id,
                                  repository=conversation_repository)
    metadata = _planner_metadata(run, message)
    timeout = float(get_settings().graph_timeout_seconds)
    operation, created = await rag_run_service.create_operation(
        user_id=user_id, conversation_id=conversation_id, session_id=run.get("session_id"), thread_id=None,
        idempotency_key=idempotency_key or str(uuid4()), query=str(run.get("query") or ""),
        deadline_at=rag_run_service.deadline_after(timeout), operation_kind="final_only", target_run_id=run_id,
        expected_delivery_revision=int(run.get("delivery_revision") or 0),
        request_payload={"target_run_id": run_id, "operation_kind": "final_only"}, repository=run_repository,
    )
    if not created:
        if operation["status"] == "completed":
            committed = await (delivery_repository or ResultDeliveryRepository()).commit(
                user_id=user_id, operation_run_id=operation["run_id"], target_run_id=run_id,
            )
            return {**committed, "retry_status": (committed["run"]["metadata"].get("source_status") or {}).get("planner_final")}
        return {"operation": operation, "run": run, "target_run_id": run_id,
                "operation_run_id": operation["run_id"], "delivery_revision": run.get("delivery_revision", 0)}
    registry = executor or get_rag_run_executor_registry()
    delivery = delivery_repository or ResultDeliveryRepository()
    return await registry.execute(
        user_id=user_id, run_id=operation["run_id"], timeout_seconds=timeout,
        runner_factory=lambda: _generate_final_result(
            user_id=user_id, run=run, operation=operation, metadata=metadata,
            provider=final_provider, paper_repository=paper_repository, delivery_repository=delivery,
            answer_reviewer=answer_reviewer,
        ),
        terminal_callback=lambda status, error: delivery.fail(
            user_id=user_id, operation_run_id=operation["run_id"], status=status, error=error,
            payload={"type": "error", "target_run_id": run_id, "operation_run_id": operation["run_id"],
                     "run_status": status, "detail": error},
        ),
    )


async def _generate_final_result(*, user_id: str, run: dict, operation: dict, metadata: dict,
                                 provider, paper_repository, delivery_repository, answer_reviewer=None) -> dict:
    """参数为已领取的目标与生成端口；返回同次提交回执，待保存产物存在时不重调模型。"""
    from app.services.research_memory_stale_service import revalidate_memory_snapshot

    if run.get("pending_artifact") is not None:
        raise DeliveryConflict("该答案仍有待保存产物，请先重试保存")
    _assert_retry_eligible(metadata)
    plan, results, records = _planner_inputs(metadata)
    await _check_saved_source_access(records, user_id=user_id, repository=paper_repository)
    await _check_saved_project_access(run=run, records=records, user_id=user_id)
    await revalidate_memory_snapshot(metadata)
    historical = _historical_input_context(plan, results, records)
    if provider is None:
        llm = get_llm()
        provider = LLMPlannerFinalAnswerProvider(llm=llm)
        answer_reviewer = LLMResearchAnswerReviewer(llm=llm)
    payload = await synthesize_planner_answer_async(
        plan, results, project_paper_count=_evidenced_paper_count(records), evidence_records=records,
        final_provider=provider, answer_reviewer=answer_reviewer, repair_context=_repair_context(metadata),
        final_entrypoint="historical_retry" if historical is not None else "final_retry",
        historical_input_context=historical,
    )
    patch = _delivery_patch(payload, metadata, str(run["run_id"]))
    result = {**metadata, **patch, "answer": payload.get("answer"), "memory_writeback": {
        "status": "skipped", "candidate_ids": [], "reason_codes": ["awaiting_result_commit"],
        "retryable": False, "error_kind": "",
    }}
    committed = await deliver_result(
        user_id=user_id, run={**operation, "session_run_id": run.get("session_run_id") or run["run_id"]},
        result=result, repository=delivery_repository, attempt_summary=_retry_attempt(payload),
    )
    return {**committed, "retry_status": patch["source_status"]["planner_final"]}


async def _load_run(
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    repository: RAGRunRepository | None,
) -> dict[str, Any]:
    """参数为用户、会话、运行及仓储；返回匹配归属的既有运行。"""
    run = await rag_run_service.get_run(user_id=user_id, run_id=run_id, repository=repository)
    if run is None or str(run.get("conversation_id") or "") != conversation_id:
        raise LookupError("RAG run not found for conversation")
    return run


async def _load_message(
    *,
    user_id: str,
    conversation_id: str,
    run_id: str,
    repository: ConversationRepository | None,
) -> dict[str, Any]:
    """参数为用户、会话、运行及仓储；返回与运行匹配的助手消息。"""
    message = await conversation_service.get_assistant_message_by_run_id(
        user_id=user_id,
        conversation_id=conversation_id,
        run_id=run_id,
        repository=repository,
    )
    if message is None:
        raise LookupError("matching assistant message with metadata.run_id not found")
    return message


def _planner_metadata(run: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """参数为同一运行与消息；从拥有保存计划的一端整体读取，禁止跨尝试逐字段拼接。"""
    metadata = run.get("metadata")
    if isinstance(metadata, dict) and metadata.get("research_plan") is not None:
        return dict(metadata)
    return {key: message[key] for key in _PLANNER_TRUTH_KEYS if key in message}


def _assert_retry_eligible(metadata: dict[str, Any]) -> None:
    """
    校验是否允许「只重试最终答案」。

    取证成功或半成功，且最终答案未达 ok 时允许重试。
    资格报告 can_call_provider 不再硬拦（与 always-produce 一致）。
    """
    raw_status = metadata.get("source_status")
    status = raw_status if isinstance(raw_status, dict) else {}
    planner = str(status.get("planner") or "").strip()
    planner_final = str(status.get("planner_final") or "").strip()
    # 1. 取证侧至少有可用结果
    if planner not in {"ok", "partial", "insufficient", "no_evidence"}:
        raise ValueError("当前取证未完成，无法只重试最终回答。请重新提问或补充材料后再试。")
    # 2. 最终侧尚未完全成功才需要重试
    if planner_final in {"ok"}:
        raise ValueError("最终回答已成功，无需重试。")
    if planner_final not in {"failed", "skipped", "partial", ""}:
        raise ValueError(f"当前最终回答状态（{planner_final or '未知'}）不支持只重试最终回答。")


def _planner_inputs(
    metadata: dict[str, Any],
) -> tuple[ResearchPlan, list[ResearchUnitResult], list[dict[str, Any]]]:
    """参数为保存输入；按明确 v1–v4 边界读取，返回原身份的计划、成果及材料。"""
    plan = ResearchPlan.model_validate(metadata.get("research_plan"))
    if plan.contract_version == CURRENT_RESEARCH_CONTRACT_VERSION:
        # 1. 【研究交付】【增量重试】累计动作沿用正常入口的校验方式，实际覆盖由最终交付核验
        issues = fulfillment_contract_issues(plan.coverage_review, plan.units,
                                             require_complete=plan.metadata.execution_mode != "incremental")
        if issues:
            raise ValueError("保存的新版研究合同无效：" + "; ".join(issue.message for issue in issues))
    elif plan.contract_version not in {"research-planner-v1", "research-planner-v2", "research-planner-v3"}:
        raise ValueError("保存的研究合同版本不受支持")
    raw_results = metadata.get("research_unit_results")
    if not isinstance(raw_results, list):
        raise ValueError("保存的研究单元结果必须是列表")
    results = [ResearchUnitResult.model_validate(item) for item in raw_results]
    raw_records = metadata.get("evidence_records")
    if not isinstance(raw_records, list) or any(not isinstance(item, dict) for item in raw_records):
        raise ValueError("保存的材料必须是来源记录列表")
    records = list(register_evidence_records({}, raw_records).values())
    return plan, results, records


async def _check_saved_source_access(
    records: list[dict[str, Any]], *, user_id: str, repository: PaperRepository | None,
) -> None:
    """参数为保存材料、用户及仓储；核验本地论文仍可读取，不读取新正文，无返回值。"""
    local = [record for record in records if record.get("source") != "arxiv"]
    if any(not str(record.get("paper_id") or "").strip() for record in local):
        raise ValueError("保存的本地材料缺少论文身份")
    paper_ids = sorted({str(record["paper_id"]) for record in local})
    if not paper_ids:
        return
    papers = await (repository or get_paper_repository()).get_papers_by_ids(paper_ids, user_id=user_id)
    readable = {str(paper["paper_id"]) for paper in papers}
    if readable != set(paper_ids):
        raise LookupError("保存来源中的论文已不存在或当前用户无权读取")


async def _check_saved_project_access(*, run: dict, records: list[dict], user_id: str) -> None:
    """【研究交付】【重试范围】参数为保存命令和材料；重验项目资格，不切换旧引用批次。"""
    from app.services.project_scope_service import _build_project_context_scope

    project_id = (run.get("metadata") or {}).get("project_id")
    if not project_id:
        return
    # 1. 【研究交付】【当前授权】只有原命令显式选择的论文可保留跨项目个人论文语义
    explicit_ids = (run.get("metadata") or {}).get("explicit_paper_ids") or []
    scope = await _build_project_context_scope(project_id, requested_paper_ids=explicit_ids, user_id=user_id)
    saved_ids = {str(record["paper_id"]) for record in records if record.get("source") != "arxiv"}
    if not saved_ids <= set(scope["paper_ids"]):
        raise LookupError("保存来源已移出当前项目，不能继续用于最终生成")


def _historical_input_context(
    plan: ResearchPlan, results: list[ResearchUnitResult], records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """参数为已核验身份的历史输入；返回可展示旧成果及具体未知项，不补造新版支持凭据。"""
    if plan.contract_version == CURRENT_RESEARCH_CONTRACT_VERSION:
        return None
    units = {unit.unit_id for unit in plan.units}
    evidence_ids = {str(record["evidence_id"]) for record in records}
    claims = [claim for result in results if result.unit_id in units and result.status in {"evidence_found", "inferred"}
              for claim in result.claims if claim.validation_status in {"validated", "needs_validation"}
              and set(claim.supporting_evidence_ids) <= evidence_ids]
    unknown = [{"source": "historical", "code": "historical_support_not_assessed", "required": True,
                "message": f"历史主张 {claim.claim_id} 未保存可核验的精确支持链和当时送模范围"} for claim in claims]
    unknown.append({"source": "historical", "code": "historical_delivery_not_assessed", "required": True,
                    "message": "历史输入未保存本次新版逐要求交付和稳定章节的全部评估事实"})
    return {"legacy_version": plan.contract_version, "unknown_items": unknown,
            "readable_claim_ids": [claim.claim_id for claim in claims],
            "readable_evidence_ids": sorted(evidence_ids)}


def _evidenced_paper_count(records: list[dict[str, Any]]) -> int:
    """参数为保存材料；返回实际已知论文数，不为缺失身份补造一篇论文。"""
    return len({str(record["paper_id"]) for record in records if record.get("paper_id")})


def _repair_context(metadata: dict[str, Any]) -> dict[str, Any]:
    """【研究交付】【重试上下文】参数为保存元数据；返回原取证修复及缺口事实，不重新取证。"""
    raw_status = metadata.get("source_status")
    status = raw_status if isinstance(raw_status, dict) else {}
    raw_reasons = metadata.get("source_status_reasons")
    reasons = raw_reasons if isinstance(raw_reasons, dict) else {}
    return {
        "final_delivery_report": metadata.get("final_delivery_report") or {},
        "planner_repair_status": status.get("planner_repair") or "not_needed",
        "planner_repair_reason": reasons.get("planner_repair") or "",
        "evidence_sufficiency_report": metadata.get("evidence_sufficiency_report") or {},
        "repair_query_plans": metadata.get("repair_query_plans") or [],
        "planner_repair_trace": metadata.get("planner_repair_trace") or {},
        "planner_repair_merge_report": metadata.get("planner_repair_merge_report") or {},
        "final_answer_eligibility_report": metadata.get("final_answer_eligibility_report") or {},
        "missing_evidence": metadata.get("missing_evidence") or [],
    }


def _delivery_patch(payload: dict[str, Any], metadata: dict[str, Any], run_id: str) -> dict[str, Any]:
    """参数为当前尝试、历史元数据及运行身份；返回不可混搭的同次正文关联字段和回合投影。"""
    report = delivery_report_from_state({**payload, "final_delivery_applicability": "research"})
    if report is None:
        raise ValueError("最终重试缺少交付报告")
    projection = project_final_delivery(report)
    outcome = {**(metadata.get("turn_outcome") or {}), "status": projection["turn_status"],
               "termination_reason": projection["termination_reason"],
               "evidence_ids": list(dict.fromkeys(binding.evidence_id for binding in report.citation_bindings))}
    keys = ("source_status", "source_status_reasons", "planner_final_blocks", "planner_final_validation_errors",
            "answer_confidence", "citations", "chunk_citations", "paper_sources", "answer_brief",
            "final_citation_bindings", "final_generation_attempt", "final_delivery_applicability",
            "error_code", "error_summary", "failure_reason")
    patch = {key: payload.get(key) for key in keys}
    # 1. 【研究交付】【同次投影】取证事实保留原运行，最终状态只取当前报告
    patch["source_status"] = {**(metadata.get("source_status") or {}),
                              "planner_final": projection["planner_final_status"]}
    patch["source_status_reasons"] = {**(metadata.get("source_status_reasons") or {}),
                                      "planner_final": _final_reason(payload, {})}
    patch.update({**delivery_state_fields(report), "turn_outcome": outcome})
    return {**patch, "run_id": run_id, "final_delivery_report": report.model_dump(mode="json"),
            "turn_outcome": outcome, "outcome": projection["chat_result_status"],
            "trace_summary": _updated_trace_summary(metadata, {**payload, **patch})}


def _retry_attempt(payload: dict[str, Any]) -> dict[str, Any]:
    """参数为当前尝试；返回时间、身份与交付状态摘要。"""
    errors = payload.get("planner_final_validation_errors") or []
    first_error = errors[0] if errors and isinstance(errors[0], dict) else {}
    return {
        "attempted_at": utc_now().isoformat(),
        "attempt_id": (payload.get("final_generation_attempt") or {}).get("attempt_id"),
        "delivery_status": (payload.get("final_delivery_report") or {}).get("status"),
        "status": _final_status(payload),
        "error_code": first_error.get("code") or "",
        "message": _final_reason(payload, first_error),
    }


def _updated_trace_summary(metadata: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """参数为旧摘要及本次完整产物；返回保留取证事实、替换最终交付的摘要。"""
    summary = dict(metadata.get("trace_summary") or {}) if isinstance(metadata.get("trace_summary"), dict) else {}
    errors = payload.get("planner_final_validation_errors") or []
    blocks = payload.get("planner_final_blocks") or []
    summary.update({
        "source_status": payload.get("source_status") or {},
        "source_status_reasons": payload.get("source_status_reasons") or {},
        "planner_final_status": _final_status(payload),
        "planner_final_accepted_block_count": len((payload.get("final_delivery_report") or {}).get("valid_business_block_ids", [])),
        "final_delivery_report": payload.get("final_delivery_report"),
        "final_citation_bindings": payload.get("final_citation_bindings"),
        "final_generation_attempt": payload.get("final_generation_attempt"),
        "turn_outcome": payload.get("turn_outcome"),
        "planner_final_rejected_block_count": count_rejected_final_blocks(errors),
        "planner_final_block_roles": [block.get("role") for block in blocks if isinstance(block, dict)],
        "planner_final_error_codes": [error.get("code") for error in errors if isinstance(error, dict)],
    })
    return summary


def _final_status(payload: dict[str, Any]) -> str:
    """参数为当前产物；返回当前报告映射的最终生成状态。"""
    report = delivery_report_from_state(payload)
    if report is not None:
        return project_final_delivery(report)["planner_final_status"]
    raw_status = payload.get("source_status")
    status = raw_status if isinstance(raw_status, dict) else {}
    return str(status.get("planner_final") or "")


def _final_reason(payload: dict[str, Any], first_error: dict[str, Any]) -> str:
    """参数为当前产物及首项错误；返回当前最终交付的可读原因。"""
    raw_reasons = payload.get("source_status_reasons")
    reasons = raw_reasons if isinstance(raw_reasons, dict) else {}
    return str(first_error.get("message") or reasons.get("planner_final") or "")
