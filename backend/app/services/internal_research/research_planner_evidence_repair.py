"""Deterministic evidence sufficiency and targeted repair for planner runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from app.middleware.trace_context import evidence_scope, record_evidence_observation
from app.models import evidence_layers as layers
from app.models.research_contracts import CapabilityStatus, MissingEvidence
from app.models.research_planner_contracts import (
    EvidenceGap,
    EvidenceRepairStatus,
    EvidenceSufficiencyReport,
    FinalAnswerEligibilityReport,
    PlannerRepairMergeReport,
    RepairQuery,
    RepairQueryPlan,
    RepairTransientUnit,
    ResearchPlan,
    ResearchUnit,
    ResearchUnitResult,
    UnitSufficiencyReport,
)
from app.services.internal_research.research_planner_executor import (
    PlannerExecutionContext,
    PlannerExecutionReport,
    ResearchPlannerExecutor,
)
from app.services.internal_research.research_planner_fulfillment import (
    build_evidence_sufficiency_report as build_evidence_sufficiency_report,
    evaluate_dependencies as validate_unit_dependencies,
    evaluate_unit_fulfillment,
    gap_repairable as _gap_repairable,
    requirements_version,
)
from app.services.internal_research.evidence_identity import (
    evidence_commitment_changes,
    register_evidence_records,
    validate_read_evidence,
)
from app.services.internal_research.research_planner_conversion import project_claim_policy

_MAX_QUERIES_PER_GAP = 3
_MAX_QUERIES_PER_UNIT = 6
_UNRESOLVED_STATUSES = frozenset({"blocked", "insufficient_evidence", "external_needed"})
_SUCCESS_STATUSES = frozenset({"evidence_found", "inferred"})


@dataclass(frozen=True, kw_only=True)
class PlannerEvidenceRepairOutcome:
    results: list[ResearchUnitResult]
    evidence_records: list[dict[str, Any]]
    first_sufficiency_report: EvidenceSufficiencyReport
    final_sufficiency_report: EvidenceSufficiencyReport
    repair_query_plans: list[RepairQueryPlan]
    repair_status: EvidenceRepairStatus
    repair_reason: str
    repair_trace: dict[str, Any]
    repair_merge_report: PlannerRepairMergeReport
    final_answer_eligibility_report: FinalAnswerEligibilityReport
    missing_evidence: list[MissingEvidence]
    # 首轮执行产出的取证四态（source_type 值 → CapabilityStatus），原样带给上层写进回合终态；
    # 补证重跑属同一能力的重试，其状态已由 repair_status 记录，不在这里覆盖
    capability_results: dict[str, CapabilityStatus] = field(default_factory=dict)
    read_receipts: list[dict[str, Any]] = field(default_factory=list)

    def synthesis_context(self) -> dict[str, Any]:
        return {
            "evidence_sufficiency_report": self.final_sufficiency_report.model_dump(mode="json"),
            "first_evidence_sufficiency_report": self.first_sufficiency_report.model_dump(mode="json"),
            "repair_query_plans": [plan.model_dump(mode="json") for plan in self.repair_query_plans],
            "planner_repair_trace": dict(self.repair_trace),
            "planner_repair_merge_report": self.repair_merge_report.model_dump(mode="json"),
            "final_answer_eligibility_report": self.final_answer_eligibility_report.model_dump(mode="json"),
            "missing_evidence": list(self.missing_evidence),
            "planner_repair_status": self.repair_status,
            "planner_repair_reason": self.repair_reason,
        }


async def execute_with_repair(
    *,
    plan: ResearchPlan,
    executor: ResearchPlannerExecutor,
    context: PlannerExecutionContext | None = None,
    mode: Literal["initial", "repair"] = "initial",
    previous_results: list[ResearchUnitResult] | None = None,
    prior_sufficiency: EvidenceSufficiencyReport | None = None,
) -> PlannerEvidenceRepairOutcome:
    """参数为已准入的执行模式与累计成果；初次只报告缺口，补证只消费上次缺口。"""
    ctx = context or PlannerExecutionContext(question=plan.question)
    if mode == "initial":
        first = await executor.execute_with_evidence(plan, ctx, previous_results=previous_results or [])
        report = build_evidence_sufficiency_report(
            plan=plan, results=first.results, evidence_records=first.evidence_records, pass_index=1,
        )
        outcome = _no_dispatch_outcome(plan, first.results, first.evidence_records, report, [])
        return replace(outcome, capability_results=dict(first.capability_results),
                       read_receipts=ctx.execution_state.snapshot().read_receipts)
    if mode != "repair" or previous_results is None or prior_sufficiency is None:
        raise ValueError("repair requires prior results and sufficiency report")
    snapshot = ctx.execution_state.snapshot()
    if prior_sufficiency.requirements_version != requirements_version(plan.coverage_review):
        raise ValueError("repair sufficiency does not match current requirements")
    first = PlannerExecutionReport(results=previous_results, evidence_records=snapshot.evidence_records)
    outcome = await _repair_after_first_pass(plan=plan, executor=executor, ctx=ctx, first=first,
                                            prior_sufficiency=prior_sufficiency)
    # 【研究规划】【补证能力】回边保留本次实际调用的能力，累计投影由 pipeline 与已有状态合并
    return replace(outcome, capability_results={**first.capability_results, **outcome.capability_results},
                   read_receipts=ctx.execution_state.snapshot().read_receipts)


async def _repair_after_first_pass(
    *,
    plan: ResearchPlan,
    executor: ResearchPlannerExecutor,
    ctx: PlannerExecutionContext,
    first: PlannerExecutionReport,
    prior_sufficiency: EvidenceSufficiencyReport,
) -> PlannerEvidenceRepairOutcome:
    """首轮执行后的补证流程：评估充分性、按缺口补检索、重跑受影响单元。

    参数:
        plan: 可执行研究计划
        executor: 已装配适配器的研究执行器
        ctx: 执行上下文
        first: 首轮执行报告，取其结果与证据池
    返回:
        补证终态；硬失败与取消保存累计快照后按原异常传播
    """
    first_report = build_evidence_sufficiency_report(
        plan=plan,
        results=first.results,
        evidence_records=first.evidence_records,
        pass_index=prior_sufficiency.pass_index,
    )
    query_plans = build_repair_query_plans(
        plan=plan,
        report=first_report,
        direct_adapter_available=layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT in executor.adapters,
    )
    if not _has_repair_queries(query_plans):
        return _no_dispatch_outcome(plan, first.results, first.evidence_records, first_report, query_plans)
    transient_units = build_transient_repair_units(plan=plan, query_plans=query_plans)
    try:
        repair = await executor.execute_transient_units([item.unit for item in transient_units], first.results, ctx)
    except BaseException as exc:
        await _preserve_failed_repair(plan, first, query_plans, ctx, exc)
        raise
    first = replace(first, capability_results={**first.capability_results, **repair.capability_results})
    try:
        outcome = await _apply_dispatched_repair(
            plan=plan,
            executor=executor,
            first=first,
            first_report=first_report,
            query_plans=query_plans,
            repair=repair,
            context=ctx,
        )
        return replace(outcome, capability_results=dict(first.capability_results))
    except BaseException as exc:
        await _preserve_failed_repair(plan, first, query_plans, ctx, exc)
        raise


async def _apply_dispatched_repair(
    *,
    plan: ResearchPlan,
    executor: ResearchPlannerExecutor,
    first: PlannerExecutionReport,
    first_report: EvidenceSufficiencyReport,
    query_plans: list[RepairQueryPlan],
    repair: PlannerExecutionReport,
    context: PlannerExecutionContext,
) -> PlannerEvidenceRepairOutcome:
    """合并补检索证据并重跑受影响单元，组装补证终态。

    参数:
        plan: 可执行研究计划
        executor: 已装配适配器的研究执行器
        first: 首轮执行报告，取其结果与证据池
        first_report: 首轮证据充分性报告
        query_plans: 补检索查询计划
        repair: 补检索阶段（transient 单元）的执行报告
    返回:
        补证终态；重跑失败由持有执行上下文的调用层保留快照并传播
    """
    repair_ids = {eid for result in repair.results for eid in result.evidence_ids}
    merge_report, merged_records = merge_repair_evidence(
        first_records=first.evidence_records,
        repair_records=[record for record in repair.evidence_records if record.get("evidence_id") in repair_ids],
        query_plans=query_plans,
    )
    results = await rerun_affected_units(
        plan=plan, first_results=first.results, evidence_records=merged_records,
        repair_results=repair.results, query_plans=query_plans, executor=executor, context=context,
    )
    merged_records = list(_record_registry([
        *merged_records, *(record for record in context.execution_state.snapshot().evidence_records
                           if record.get("evidence_id") not in merge_report.invalid_evidence_ids),
    ]).values())
    merge_report = merge_report.model_copy(update={"pass_counts": {
        **merge_report.pass_counts, "merged_records": len(merged_records),
    }})
    return _completed_outcome(plan, first_report, query_plans, merge_report, merged_records, results)


async def _preserve_failed_repair(
    plan: ResearchPlan, first: PlannerExecutionReport, query_plans: list[RepairQueryPlan],
    context: PlannerExecutionContext, exc: BaseException,
) -> None:
    """参数为补证上下文及原异常；保存一致快照并撤销受影响派生结果，不包装失败或取消。"""
    await context.execution_state.settle()
    interrupted = getattr(exc, "planner_execution_report", None)
    capabilities = {**first.capability_results, **(interrupted.capability_results if interrupted is not None else {})}
    affected = _affected_unit_ids(plan, {item.linked_unit_id for item in query_plans if item.queries})
    safe = _mark_affected_rerun_failed(first.results, affected, f"repair failed: {type(exc).__name__}: {exc}")
    reports_by_unit: dict[str, list[dict[str, Any]]] = {}
    for query_plan in query_plans:
        associations = context.execution_state.read_associations(query_plan.repair_unit_id)
        reports_by_unit.setdefault(query_plan.linked_unit_id, []).extend(item["read_report"] for item in associations)
    safe = [result.model_copy(update={"metadata": {**result.metadata, "read_reports": [
        *result.metadata.get("read_reports", []), *reports_by_unit.get(result.unit_id, []),
    ]}}) for result in safe]
    snapshot = context.execution_state.snapshot()
    report = PlannerExecutionReport(results=safe, evidence_records=snapshot.evidence_records,
                                    read_receipts=snapshot.read_receipts, capability_results=capabilities)
    setattr(exc, "planner_execution_report", report)
    record_evidence_observation("cancelled_repair_snapshot" if isinstance(exc, asyncio.CancelledError)
                                else "failed_repair_snapshot", {
        "records": snapshot.evidence_records, "read_receipts": snapshot.read_receipts,
        "results": [result.model_dump(mode="json") for result in safe],
        "repair_query_plans": [item.model_dump(mode="json") for item in query_plans],
        "error_type": type(exc).__name__,
    })




def build_repair_query_plans(
    *,
    plan: ResearchPlan,
    report: EvidenceSufficiencyReport,
    direct_adapter_available: bool,
) -> list[RepairQueryPlan]:
    units = {unit.unit_id: unit for unit in plan.units}
    plans: list[RepairQueryPlan] = []
    for unit_report in report.unit_reports:
        unit = units.get(unit_report.unit_id)
        if unit is None or not unit_report.hard_gaps or unit_report.unit_id not in report.blocking_unit_ids:
            continue
        if any(gap.gap_type == "dependency_unresolved" for gap in unit_report.hard_gaps):
            continue
        plan_item = _query_plan_for_unit(
            unit=unit,
            unit_report=unit_report,
            direct_adapter_available=direct_adapter_available,
        )
        if plan_item.queries or plan_item.skipped_gap_ids:
            plans.append(plan_item)
    return plans


def build_transient_repair_units(
    *,
    plan: ResearchPlan,
    query_plans: list[RepairQueryPlan],
) -> list[RepairTransientUnit]:
    units = {unit.unit_id: unit for unit in plan.units}
    transients: list[RepairTransientUnit] = []
    for query_plan in query_plans:
        source_unit = units.get(query_plan.linked_unit_id)
        if source_unit is None or not query_plan.queries:
            continue
        transients.append(_transient_unit(source_unit, query_plan))
    return transients


def merge_repair_evidence(
    *,
    first_records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
    query_plans: list[RepairQueryPlan],
) -> tuple[PlannerRepairMergeReport, list[dict[str, Any]]]:
    """参数为首轮、补证记录和查询计划；返回合并报告及证据池，并保存本次合并事实。"""
    registry = _record_registry(first_records)
    duplicates: list[str] = []
    invalid: list[str] = []
    conflicts: list[dict[str, Any]] = []
    target_ids = _repair_target_ids(query_plans)
    for index, record in enumerate(repair_records):
        evidence_id = str(record.get("evidence_id") or "").strip()
        invalid_reason = _invalid_repair_record_reason(record, target_ids)
        if invalid_reason:
            invalid.append(evidence_id or f"missing-evidence-id-{index}")
            conflicts.append(_conflict(record, invalid_reason))
            continue
        if evidence_id in registry:
            duplicates.append(evidence_id)
            _append_duplicate_conflict(registry[evidence_id], record, conflicts)
            continue
        validate_read_evidence(record)
        registry[evidence_id] = dict(record)
    report = PlannerRepairMergeReport(
        seen_evidence_ids=list(registry),
        duplicate_evidence_ids=duplicates,
        invalid_evidence_ids=invalid,
        conflict_records=conflicts,
        pass_counts={
            "first_records": len(first_records),
            "repair_records": len(repair_records),
            "merged_records": len(registry),
            "invalid_repair_records": len(invalid),
        },
    )
    merged = list(registry.values())
    record_evidence_observation("repair_evidence_pool", {
        "records": merged,
        "linked_unit_ids": [item.linked_unit_id for item in query_plans],
        "merge_report": report.model_dump(mode="json"),
    })
    return report, merged


async def rerun_affected_units(
    *,
    plan: ResearchPlan,
    first_results: list[ResearchUnitResult],
    evidence_records: list[dict[str, Any]],
    repair_results: list[ResearchUnitResult],
    query_plans: list[RepairQueryPlan],
    executor: ResearchPlannerExecutor,
    context: PlannerExecutionContext,
) -> list[ResearchUnitResult]:
    """参数为计划、首轮及补证结果和执行器；返回重跑结果，每个受影响单元保留独立关联。"""
    affected = _affected_unit_ids(plan, {item.linked_unit_id for item in query_plans if item.queries})
    if not affected:
        return list(first_results)
    registry = _record_registry(evidence_records)
    repair_ids = _repair_evidence_ids_by_unit(repair_results, query_plans, registry)
    repaired_units = {item.linked_unit_id for item in query_plans if item.queries}
    reports_by_unit = _read_reports_by_unit(first_results, repair_results, query_plans)
    original_by_id = {result.unit_id: result for result in first_results}
    # 1. 【研究规划】【派生失效】先撤销整个下游闭包，失败时不复活未重跑的旧主张
    results_by_id = {result.unit_id: result for result in _mark_affected_rerun_failed(
        first_results, affected, "upstream inputs changed; derived result invalidated",
    )}
    for unit in _topological_units(plan):
        if unit.unit_id not in affected:
            continue
        previous = list(results_by_id.values())
        if unit.unit_id not in repaired_units and unit.evidence_policy.source_type != "inference_from_units":
            # 2. 【研究规划】【依赖重跑】普通取证消费者也重新经过共享读取边界，首次未读与可复用材料由执行上下文区分
            execution = await executor.execute_transient_units([unit], previous, context)
            registry = _record_registry(execution.evidence_records)
            results_by_id[unit.unit_id] = execution.results[0]
            continue
        records = _records_for_unit(original_by_id.get(unit.unit_id), repair_ids.get(unit.unit_id, []), registry)
        with evidence_scope("repair_rerun", unit_id=unit.unit_id):
            record_evidence_observation("rerun_evidence_pool", {"records": records})
            results_by_id[unit.unit_id] = await _rerun_unit(
                unit=unit, records=records, previous=previous, executor=executor,
                read_reports=reports_by_unit.get(unit.unit_id, []),
                plan=plan, registry=registry, context=context,
            )
            result = results_by_id[unit.unit_id]
            result = _apply_fulfillment(unit, result, previous, registry, plan)
            results_by_id[unit.unit_id] = result.model_copy(update={
                "metadata": {**result.metadata, "read_reports": reports_by_unit.get(unit.unit_id, [])},
            })
    return [results_by_id[unit.unit_id] for unit in plan.units if unit.unit_id in results_by_id]


def _topological_units(plan: ResearchPlan) -> list[ResearchUnit]:
    """参数为已校验计划；返回依赖拓扑序，重跑不依赖单元在列表中的位置。"""
    pending = {unit.unit_id: unit for unit in plan.units}
    ordered: list[ResearchUnit] = []
    resolved: set[str] = set()
    while pending:
        ready = [unit for unit in pending.values() if all(dep.unit_id in resolved for dep in unit.dependencies)]
        if not ready:
            raise ValueError("plan dependencies cannot be ordered")
        for unit in ready:
            ordered.append(unit)
            resolved.add(unit.unit_id)
            del pending[unit.unit_id]
    return ordered


def _apply_fulfillment(
    unit: ResearchUnit, result: ResearchUnitResult, previous: list[ResearchUnitResult],
    registry: Mapping[str, dict[str, Any]], plan: ResearchPlan,
) -> ResearchUnitResult:
    """参数为重跑结果及判定上下文；返回消费同一权威报告的有效性投影。"""
    report = evaluate_unit_fulfillment(unit, result, registry=registry,
                                      results_by_id={item.unit_id: item for item in previous}, plan=plan)
    if not report.hard_gaps or result.status in {"blocked", "insufficient_evidence", "external_needed"}:
        return result
    return result.model_copy(update={
        "status": "insufficient_evidence", "unresolved_reason": "; ".join(gap.reason for gap in report.hard_gaps),
    })


def _read_reports_by_unit(
    first_results: list[ResearchUnitResult], repair_results: list[ResearchUnitResult],
    query_plans: list[RepairQueryPlan],
) -> dict[str, list[dict[str, Any]]]:
    """参数为首轮与补读结果及关联计划；返回按原单元归并的实际读取说明，不创造新读取事实。"""
    reports = {result.unit_id: list(result.metadata.get("read_reports") or []) for result in first_results}
    linked = {plan.repair_unit_id: plan.linked_unit_id for plan in query_plans}
    for result in repair_results:
        unit_id = linked.get(result.unit_id)
        if unit_id:
            reports.setdefault(unit_id, []).extend(result.metadata.get("read_reports") or [])
    return reports


def build_final_answer_eligibility_report(
    *,
    plan: ResearchPlan,
    results: list[ResearchUnitResult],
    report: EvidenceSufficiencyReport,
) -> FinalAnswerEligibilityReport:
    results_by_id = {result.unit_id: result for result in results}
    final_units = {unit.unit_id for unit in plan.units if unit.output_role == "final_recommendation"}
    dependency_closure = _required_dependency_closure(plan, final_units)
    blocking_gaps = _final_blocking_gaps(report, dependency_closure | final_units)
    unresolved_final = [unit_id for unit_id in final_units if _is_unresolved(results_by_id.get(unit_id))]
    if not blocking_gaps and not unresolved_final:
        return FinalAnswerEligibilityReport(can_call_provider=True, partial_allowed=False)
    hard_stop = unresolved_final or any(gap.gap_type in {"target_coverage_missing", "dependency_unresolved"} for gap in blocking_gaps)
    reason = _eligibility_reason(blocking_gaps, unresolved_final)
    return FinalAnswerEligibilityReport(
        can_call_provider=False,
        final_status_if_skipped="skipped" if hard_stop else "partial",
        blocking_unit_ids=sorted({gap.unit_id for gap in blocking_gaps} | set(unresolved_final)),
        blocking_gap_ids=[gap.gap_id for gap in blocking_gaps],
        partial_allowed=not hard_stop,
        reason=reason,
    )


def derive_missing_evidence(
    *,
    report: EvidenceSufficiencyReport,
    query_plans: list[RepairQueryPlan],
) -> list[MissingEvidence]:
    attempted = _attempted_queries_by_gap(query_plans)
    skipped = _skipped_reasons_by_gap(query_plans)
    missing: list[MissingEvidence] = []
    seen: set[tuple[str, str, str, str]] = set()
    for gap in _hard_gaps(report):
        key = (gap.unit_id, gap.target_paper_id, gap.evidence_dimension, gap.reason)
        if key in seen:
            continue
        seen.add(key)
        missing.append(MissingEvidence(
            paper_id=gap.target_paper_id,
            dimension=gap.evidence_dimension or gap.unit_id,
            reason=gap.reason if gap.gap_id not in skipped else skipped[gap.gap_id],
            attempted_queries=attempted.get(gap.gap_id, []),
        ))
    return missing




def _query_plan_for_unit(
    *,
    unit: ResearchUnit,
    unit_report: UnitSufficiencyReport,
    direct_adapter_available: bool,
) -> RepairQueryPlan:
    queries: list[RepairQuery] = []
    skipped: dict[str, str] = {}
    for gap in unit_report.hard_gaps:
        reason = _skip_repair_reason(unit, gap, direct_adapter_available)
        if reason:
            skipped[gap.gap_id] = reason
            continue
        for query in _queries_for_gap(unit, gap):
            if len(queries) < _MAX_QUERIES_PER_UNIT and _query_new(queries, query):
                queries.append(query)
        if not any(query.source_gap_id == gap.gap_id for query in queries):
            skipped[gap.gap_id] = "empty_repair_query"
    return RepairQueryPlan(
        source_report_pass=unit_report.pass_index,
        linked_unit_id=unit.unit_id,
        repair_unit_id=f"{unit.unit_id}::evidence_repair::p{unit_report.pass_index + 1}",
        queries=queries,
        skipped_gap_ids=list(skipped),
        skipped_reasons=skipped,
    )


def _transient_unit(source_unit: ResearchUnit, query_plan: RepairQueryPlan) -> RepairTransientUnit:
    """参数为原单元及补证查询；返回只采集材料的临时单元，不另授主张权限。"""
    target_ids = _query_target_ids(query_plan.queries) or list(source_unit.evidence_policy.target_paper_ids)
    policy = source_unit.evidence_policy.model_copy(update={
        "source_type": layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT,
        "target_paper_ids": target_ids,
        "locator_queries": [query.query for query in query_plan.queries],
    })
    claim_policy = project_claim_policy([])
    unit = source_unit.model_copy(update={
        "unit_id": query_plan.repair_unit_id,
        "goal": f"Repair direct-text evidence for {source_unit.goal}",
        "research_action": "targeted direct-text repair retrieval",
        "evidence_policy": policy,
        "claim_policy": claim_policy,
        "expected_output": source_unit.expected_output.model_copy(update={
            "deliverable_kind": "materials", "required_claim_types": [],
        }),
        "dependencies": [],
        "output_role": "evidence",
    })
    return RepairTransientUnit(
        unit=ResearchUnit.model_validate(unit.model_dump(), strict=True),
        linked_unit_id=query_plan.linked_unit_id,
        source_gap_id=query_plan.queries[0].source_gap_id,
    )


async def _rerun_unit(
    *,
    unit: ResearchUnit,
    records: list[dict[str, Any]],
    previous: list[ResearchUnitResult],
    executor: ResearchPlannerExecutor,
    read_reports: list[dict[str, Any]],
    plan: ResearchPlan,
    registry: Mapping[str, dict[str, Any]],
    context: PlannerExecutionContext,
) -> ResearchUnitResult:
    """参数为单元、累计材料、前序结果、执行器及读取说明；返回既有补证重跑结果。"""
    dependency = validate_unit_dependencies(unit, {result.unit_id: result for result in previous},
                                            evidence_records=list(registry.values()),
                                            units_by_id={item.unit_id: item for item in plan.units})
    if not dependency.satisfied:
        return _unit_result(unit, status="blocked", records=[], reason="; ".join(dependency.blocking_reasons))
    source_type = unit.evidence_policy.source_type
    if source_type == layers.PLANNER_SOURCE_INFERENCE_FROM_UNITS:
        return await _rerun_inference(unit=unit, previous=previous, executor=executor, registry=registry, context=context)
    direct_records = _direct_citable_records(records)
    if source_type == layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT:
        return await _rerun_structure_hint(
            unit=unit, records=records, previous=previous, executor=executor, read_reports=read_reports,
            registry=registry, context=context,
        )
    if unit.expected_output.deliverable_kind == "materials":
        result = _unit_result(unit, status="evidence_found", records=records)
        return _apply_fulfillment(unit, result, previous, registry, plan)
    outcome = await executor.reasoner.reason(unit, direct_records, previous, read_reports=read_reports,
                                             dependency_evidence_records=list(registry.values()), plan=context.plan,
                                             execution_identity=(context.execution_state.identity.model_dump(mode="json")
                                                                 if context.execution_state.identity else None))
    if not outcome.claims:
        reason = "; ".join(outcome.rejected_reasons) or "claim validation produced no claims after repair"
        return _unit_result(unit, status="insufficient_evidence", records=direct_records, reason=reason)
    result = _unit_result(unit, status="evidence_found", records=direct_records, claims=outcome.claims, provenance=outcome.provenance)
    return _apply_fulfillment(unit, result, previous, registry, plan)


async def _rerun_inference(
    *,
    unit: ResearchUnit,
    previous: list[ResearchUnitResult],
    executor: ResearchPlannerExecutor,
    registry: Mapping[str, dict[str, Any]],
    context: PlannerExecutionContext,
) -> ResearchUnitResult:
    """参数为推断单元和权威成果池；使用完整依赖材料重做推断并返回新结果。"""
    outcome = await executor.reasoner.reason(unit, [], previous, dependency_evidence_records=list(registry.values()), plan=context.plan,
                                             execution_identity=(context.execution_state.identity.model_dump(mode="json")
                                                                 if context.execution_state.identity else None))
    if not outcome.claims:
        reason = "; ".join(outcome.rejected_reasons) or "dependency claims did not support inference after repair"
        return _unit_result(unit, status="blocked", records=[], reason=reason)
    return _unit_result(unit, status="inferred", records=[], claims=outcome.claims, provenance=outcome.provenance)


async def _rerun_structure_hint(
    *,
    unit: ResearchUnit,
    records: list[dict[str, Any]],
    previous: list[ResearchUnitResult],
    executor: ResearchPlannerExecutor,
    read_reports: list[dict[str, Any]],
    registry: Mapping[str, dict[str, Any]],
    context: PlannerExecutionContext,
) -> ResearchUnitResult:
    """参数为结构核实单元、材料及读取事实；以正文重新核实并返回原有主张结果。"""
    direct_records = _direct_citable_records(records)
    if not direct_records:
        reason = "structure_hint still lacks direct-text verification after repair"
        return _unit_result(unit, status="insufficient_evidence", records=records, reason=reason)
    direct_unit = unit.model_copy(update={
        "evidence_policy": unit.evidence_policy.model_copy(update={"source_type": layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT}),
    })
    outcome = await executor.reasoner.reason(direct_unit, direct_records, previous, read_reports=read_reports,
                                             dependency_evidence_records=list(registry.values()), plan=context.plan,
                                             execution_identity=(context.execution_state.identity.model_dump(mode="json")
                                                                 if context.execution_state.identity else None))
    if not outcome.claims:
        reason = "; ".join(outcome.rejected_reasons) or "structure_hint direct verification produced no claims"
        return _unit_result(unit, status="insufficient_evidence", records=records, reason=reason)
    result = _unit_result(unit, status="evidence_found", records=records, claims=outcome.claims, provenance=outcome.provenance)
    return result.model_copy(update={"metadata": {**result.metadata, "structure_verification": {
        "status": "verified", "hint_evidence_ids": [record["evidence_id"] for record in records
                                                    if record.get("layer") == layers.EVIDENCE_LAYER_STRUCTURE_HINT],
        "direct_evidence_ids": [record["evidence_id"] for record in direct_records],
    }}})


def _completed_outcome(
    plan: ResearchPlan,
    first_report: EvidenceSufficiencyReport,
    query_plans: list[RepairQueryPlan],
    merge_report: PlannerRepairMergeReport,
    records: list[dict[str, Any]],
    results: list[ResearchUnitResult],
) -> PlannerEvidenceRepairOutcome:
    final_report = build_evidence_sufficiency_report(
        plan=plan,
        results=results,
        evidence_records=records,
        pass_index=first_report.pass_index + 1,
    )
    status, reason = _completed_repair_status(first_report, final_report, merge_report)
    annotated = _annotate_results(results, final_report, status)
    eligibility = build_final_answer_eligibility_report(plan=plan, results=annotated, report=final_report)
    missing = derive_missing_evidence(report=final_report, query_plans=query_plans)
    trace = _repair_trace(status, reason, first_report, final_report, query_plans, merge_report)
    return PlannerEvidenceRepairOutcome(
        results=annotated,
        evidence_records=records,
        first_sufficiency_report=first_report,
        final_sufficiency_report=final_report,
        repair_query_plans=query_plans,
        repair_status=status,
        repair_reason=reason,
        repair_trace=trace,
        repair_merge_report=merge_report,
        final_answer_eligibility_report=eligibility,
        missing_evidence=missing,
    )




def _mark_affected_rerun_failed(
    results: list[ResearchUnitResult],
    affected: set[str],
    reason: str,
) -> list[ResearchUnitResult]:
    """参数为原成果、受影响单元和失败原因；返回撤销派生主张及支持凭据后的成果列表。"""
    marked: list[ResearchUnitResult] = []
    for result in results:
        if result.unit_id not in affected:
            marked.append(result)
            continue
        metadata = {key: value for key, value in result.metadata.items()
                    if key not in {"result_input_signature", "unit_business_keys", "result_reused", "provenance",
                                   "claim_id_rebindings", "unit_id_rebindings"}}
        metadata["rerun_error"] = reason
        marked.append(result.model_copy(update={
            "status": "blocked",
            "summary": "",
            "claims": [],
            "unresolved_reason": reason,
            "metadata": metadata,
        }))
    return marked




def _no_dispatch_outcome(
    plan: ResearchPlan,
    results: list[ResearchUnitResult],
    records: list[dict[str, Any]],
    report: EvidenceSufficiencyReport,
    query_plans: list[RepairQueryPlan],
) -> PlannerEvidenceRepairOutcome:
    status = _no_dispatch_status(report)
    reason = _no_dispatch_reason(report, query_plans)
    merge = PlannerRepairMergeReport(pass_counts={"first_records": len(records), "repair_records": 0})
    annotated = _annotate_results(results, report, status)
    eligibility = build_final_answer_eligibility_report(plan=plan, results=annotated, report=report)
    missing = derive_missing_evidence(report=report, query_plans=query_plans)
    trace = _repair_trace(status, reason, report, report, query_plans, merge)
    return PlannerEvidenceRepairOutcome(
        results=annotated,
        evidence_records=records,
        first_sufficiency_report=report,
        final_sufficiency_report=report,
        repair_query_plans=query_plans,
        repair_status=status,
        repair_reason=reason,
        repair_trace=trace,
        repair_merge_report=merge,
        final_answer_eligibility_report=eligibility,
        missing_evidence=missing,
    )
















def _queries_for_gap(unit: ResearchUnit, gap: EvidenceGap) -> list[RepairQuery]:
    candidates = _query_candidates(unit, gap)
    queries: list[RepairQuery] = []
    for text in candidates:
        normalized = _normalize_query(text)
        if normalized and normalized.casefold() not in {query.query.casefold() for query in queries}:
            queries.append(_repair_query(unit, gap, normalized))
        if len(queries) >= _MAX_QUERIES_PER_GAP:
            break
    return queries


def _query_candidates(unit: ResearchUnit, gap: EvidenceGap) -> list[str]:
    dimension = gap.evidence_dimension or _gap_dimension(unit)
    candidates = [f"{query} {dimension}" for query in unit.evidence_policy.locator_queries]
    candidates.extend(f"{dimension} {section}" for section in unit.evidence_policy.required_sections)
    candidates.append(f"{_compact_goal(unit.goal)} {dimension}")
    return candidates


def _repair_query(unit: ResearchUnit, gap: EvidenceGap, query: str) -> RepairQuery:
    return RepairQuery(
        query=query,
        target_paper_id=gap.target_paper_id,
        evidence_dimension=gap.evidence_dimension,
        required_sections=list(unit.evidence_policy.required_sections),
        preferred_block_types=list(unit.evidence_policy.preferred_block_types),
        source_gap_id=gap.gap_id,
    )


def _annotate_results(
    results: list[ResearchUnitResult],
    report: EvidenceSufficiencyReport,
    repair_status: EvidenceRepairStatus,
) -> list[ResearchUnitResult]:
    reports = {item.unit_id: item for item in report.unit_reports}
    annotated: list[ResearchUnitResult] = []
    for result in results:
        unit_report = reports.get(result.unit_id)
        metadata = dict(result.metadata)
        metadata["evidence_repair_status"] = repair_status
        metadata["remaining_gap_ids"] = [gap.gap_id for gap in unit_report.hard_gaps] if unit_report else []
        metadata["coverage_by_target"] = dict(unit_report.target_coverage) if unit_report else {}
        annotated.append(result.model_copy(update={"metadata": metadata}))
    return annotated


def _completed_repair_status(
    first: EvidenceSufficiencyReport,
    final: EvidenceSufficiencyReport,
    merge: PlannerRepairMergeReport,
) -> tuple[EvidenceRepairStatus, str]:
    if not _hard_gaps(first):
        return "not_needed", "first-pass sufficiency had no hard gaps"
    if not _hard_gaps(final):
        return "resolved", "repair resolved all hard sufficiency gaps"
    if _gap_ids(final) < _gap_ids(first) or int(merge.pass_counts.get("merged_records") or 0) > int(merge.pass_counts.get("first_records") or 0):
        return "partial", "repair improved evidence coverage but hard gaps remain"
    return "failed", "repair produced no valid scoped direct-text evidence for hard gaps"


def _no_dispatch_status(report: EvidenceSufficiencyReport) -> EvidenceRepairStatus:
    if not _hard_gaps(report):
        return "not_needed"
    return "skipped"


def _no_dispatch_reason(
    report: EvidenceSufficiencyReport,
    plans: list[RepairQueryPlan],
) -> str:
    if not _hard_gaps(report):
        return "first-pass sufficiency had no repair-recommended hard gaps"
    reasons = [reason for plan in plans for reason in plan.skipped_reasons.values()]
    return "; ".join(sorted(set(reasons))) or "no executable direct-text repair query"


def _repair_trace(
    status: EvidenceRepairStatus,
    reason: str,
    first: EvidenceSufficiencyReport,
    final: EvidenceSufficiencyReport,
    query_plans: list[RepairQueryPlan],
    merge_report: PlannerRepairMergeReport,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "first_pass": _report_summary(first),
        "final_pass": _report_summary(final),
        "repair_query_count": sum(len(plan.queries) for plan in query_plans),
        "repair_unit_ids": [plan.repair_unit_id for plan in query_plans if plan.queries],
        "attempted_queries": [query.query for plan in query_plans for query in plan.queries],
        "skipped_reasons": {key: value for plan in query_plans for key, value in plan.skipped_reasons.items()},
        "merge": merge_report.model_dump(mode="json"),
    }


def _report_summary(report: EvidenceSufficiencyReport) -> dict[str, Any]:
    gaps = _hard_gaps(report)
    return {
        "pass_index": report.pass_index,
        "blocking_unit_ids": list(report.blocking_unit_ids),
        "hard_gap_count": len(gaps),
        "repair_recommended": report.repair_recommended,
        "gap_types": sorted({gap.gap_type for gap in gaps}),
    }




def _record_registry(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """参数为保存的证据记录；返回核对来源和内容承诺后的统一登记表。"""
    return register_evidence_records({}, records)




def _direct_citable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _record_citable(record)]


def _record_citable(record: dict[str, Any] | None) -> bool:
    return bool(
        record
        and record.get("evidence_id")
        and record.get("layer") == layers.EVIDENCE_LAYER_DIRECT_TEXT
        and record.get("usable_as_citation")
    )








def _skip_repair_reason(
    unit: ResearchUnit,
    gap: EvidenceGap,
    direct_adapter_available: bool,
) -> str:
    if not _gap_repairable(unit, gap):
        return "not_repair_eligible"
    if not direct_adapter_available:
        return "adapter_scope_unsupported"
    if gap.gap_type == "target_coverage_missing" and not gap.target_paper_id:
        return "missing_target_scope"
    return ""


def _query_new(existing: list[RepairQuery], query: RepairQuery) -> bool:
    return query.query.casefold() not in {item.query.casefold() for item in existing}


def _repair_target_ids(query_plans: list[RepairQueryPlan]) -> set[str]:
    return {
        query.target_paper_id
        for plan in query_plans
        for query in plan.queries
        if query.target_paper_id
    }


def _query_target_ids(queries: list[RepairQuery]) -> list[str]:
    seen: list[str] = []
    for query in queries:
        if query.target_paper_id and query.target_paper_id not in seen:
            seen.append(query.target_paper_id)
    return seen


def _invalid_repair_record_reason(record: dict[str, Any], target_ids: set[str]) -> str:
    if not _record_citable(record):
        return "repair_record_not_citable_direct_text"
    paper_id = str(record.get("paper_id") or "").strip()
    if target_ids and (not paper_id or paper_id not in target_ids):
        return "repair_record_out_of_scope"
    return ""


def _append_duplicate_conflict(
    existing: dict[str, Any],
    record: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> None:
    """参数为同 ID 记录和冲突集合；核对统一来源/内容承诺并如实记录冲突。"""
    changed = evidence_commitment_changes(existing, record)
    if changed:
        conflicts.append(_conflict(record, f"duplicate_evidence_conflict:{','.join(changed)}"))


def _conflict(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "evidence_id": str(record.get("evidence_id") or ""),
        "paper_id": str(record.get("paper_id") or ""),
        "reason": reason,
    }


def _repair_evidence_ids_by_unit(
    repair_results: list[ResearchUnitResult],
    query_plans: list[RepairQueryPlan],
    registry: Mapping[str, dict[str, Any]],
) -> dict[str, list[str]]:
    linked = {plan.repair_unit_id: plan.linked_unit_id for plan in query_plans}
    evidence_by_unit: dict[str, list[str]] = {}
    for result in repair_results:
        unit_id = linked.get(result.unit_id)
        if not unit_id:
            continue
        evidence_by_unit.setdefault(unit_id, []).extend(
            evidence_id for evidence_id in result.evidence_ids if _record_citable(registry.get(evidence_id))
        )
    return evidence_by_unit


def _records_for_unit(
    result: ResearchUnitResult | None,
    repair_ids: list[str],
    registry: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = list(result.evidence_ids if result else [])
    ids.extend(evidence_id for evidence_id in repair_ids if evidence_id not in ids)
    return [registry[evidence_id] for evidence_id in ids if evidence_id in registry]


def _affected_unit_ids(plan: ResearchPlan, seeds: set[str]) -> set[str]:
    affected = set(seeds)
    changed = True
    while changed:
        changed = False
        for unit in plan.units:
            if unit.unit_id in affected or not _depends_on_affected(unit, affected):
                continue
            affected.add(unit.unit_id)
            changed = True
    return affected


def _depends_on_affected(unit: ResearchUnit, affected: set[str]) -> bool:
    return any(dep.unit_id in affected for dep in unit.dependencies)




def _unit_result(
    unit: ResearchUnit,
    *,
    status: str,
    records: list[dict[str, Any]],
    claims: list[Any] | None = None,
    reason: str = "",
    provenance: dict[str, Any] | None = None,
) -> ResearchUnitResult:
    """参数为单元、执行状态、材料、主张、原因与来源凭据；返回本次补查的实际单元成果。"""
    return ResearchUnitResult(
        unit_id=unit.unit_id,
        status=status,
        summary=unit.goal if status in _SUCCESS_STATUSES else "",
        evidence_ids=[str(record.get("evidence_id")) for record in records if record.get("evidence_id")],
        claims=claims or [],
        unresolved_reason=reason,
        metadata={"source_type": unit.evidence_policy.source_type, "rejected_claims": [],
                  **({"provenance": provenance} if provenance else {})},
    )




def _records_in_scope(records: list[dict[str, Any]], target_paper_ids: list[str]) -> list[dict[str, Any]]:
    # target_paper_ids 是硬范围过滤器：为空表示不限定，否则只保留范围内论文的证据
    if not target_paper_ids:
        return list(records)
    scope = set(target_paper_ids)
    return [record for record in records if str(record.get("paper_id") or "") in scope]




def _is_unresolved(result: ResearchUnitResult | None) -> bool:
    return result is None or result.status in _UNRESOLVED_STATUSES


def _required_dependency_closure(plan: ResearchPlan, unit_ids: set[str]) -> set[str]:
    by_id = {unit.unit_id: unit for unit in plan.units}
    closure: set[str] = set()
    stack = list(unit_ids)
    while stack:
        unit = by_id.get(stack.pop())
        if unit is None:
            continue
        for dep in unit.dependencies:
            if dep.required and dep.unit_id not in closure:
                closure.add(dep.unit_id)
                stack.append(dep.unit_id)
    return closure


def _final_blocking_gaps(
    report: EvidenceSufficiencyReport,
    final_dependency_ids: set[str],
) -> list[EvidenceGap]:
    return [
        gap
        for gap in _hard_gaps(report)
        if gap.unit_id in final_dependency_ids or gap.gap_type == "target_coverage_missing"
    ]


def _eligibility_reason(gaps: list[EvidenceGap], unresolved_final: list[str]) -> str:
    if unresolved_final:
        return f"final recommendation unresolved: {unresolved_final}"
    if gaps:
        return gaps[0].reason
    return "final answer provider skipped by eligibility gate"


def _attempted_queries_by_gap(query_plans: list[RepairQueryPlan]) -> dict[str, list[str]]:
    attempted: dict[str, list[str]] = {}
    for plan in query_plans:
        for query in plan.queries:
            attempted.setdefault(query.source_gap_id, []).append(query.query)
    return attempted


def _skipped_reasons_by_gap(query_plans: list[RepairQueryPlan]) -> dict[str, str]:
    return {gap_id: reason for plan in query_plans for gap_id, reason in plan.skipped_reasons.items()}


def _hard_gaps(report: EvidenceSufficiencyReport) -> list[EvidenceGap]:
    """参数为统一满足报告；返回去重后的单元与必需要求缺口。"""
    gaps = [gap for unit_report in report.unit_reports if unit_report.unit_id in report.blocking_unit_ids
            for gap in unit_report.hard_gaps]
    gaps.extend(gap for facet in report.facet_reports if facet.required for gap in facet.hard_gaps)
    return list({gap.gap_id: gap for gap in gaps}.values())


def _gap_ids(report: EvidenceSufficiencyReport) -> set[str]:
    return {gap.gap_id for gap in _hard_gaps(report)}


def _has_repair_queries(query_plans: list[RepairQueryPlan]) -> bool:
    return any(plan.queries for plan in query_plans)


def _gap_dimension(unit: ResearchUnit) -> str:
    if unit.evidence_policy.evidence_dimensions:
        return unit.evidence_policy.evidence_dimensions[0]
    return unit.expected_output.output_goal or unit.goal






def _compact_goal(goal: str) -> str:
    words = str(goal or "").strip().split()
    return " ".join(words[:12])


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().split())
