"""Plan-requested direct-text verification for planner structure hints."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.models import evidence_layers as layers
from app.models.knowledge_access_contracts import EvidenceReadResult
from app.models.research_planner_contracts import ResearchUnit, ResearchUnitResult
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner

# 适配器的受控调用入口，由执行层注入（那边带重试与时间上限）。
# 本模块在执行层的下游，反向 import 会成环，所以走注入而不是自己再实现一份调用。
AdapterInvoker = Callable[[Any, ResearchUnit, Any], Awaitable[EvidenceReadResult]]


async def verify_structure_hint_unit(
    *,
    unit: ResearchUnit,
    hint_records: list[dict[str, Any]],
    direct_adapter: Any,
    invoke_adapter: AdapterInvoker,
    reasoner: ResearchUnitReasoner,
    previous_results: list[ResearchUnitResult],
    context: Any,
    read_reports: list[dict[str, Any]],
) -> tuple[ResearchUnitResult, list[dict[str, Any]]]:
    """把结构线索拿正文证据核实一遍，核实通过才升级为可引用事实。

    只有计划在单元上写了 verify_with_direct_text=true 才会走到这里：核实要额外发一次
    正文检索，索取与否由计划决定。没索取的单元不进本函数，线索只作规划已知事实。

    Args:
        unit: 计划里索取了正文核实的 local_structure_hint 单元
        hint_records: 该单元取到的结构线索记录，本身不可引用
        direct_adapter: 正文取证适配器，缺它这轮无从核实
        invoke_adapter: 执行层注入的受控调用入口，带时间上限
        reasoner: 单元推理器，把正文证据判成断言
        previous_results: 前序单元结果，供推理器参考
        context: 执行上下文
    Returns:
        tuple: (单元结果, 进池证据)；核实不成立时线索原样返回、仍不可引用
    """
    if not hint_records:
        result = _result(unit, "insufficient_evidence", [], [], "no structure hints found", [])
        return _with_reports(result, read_reports), []
    if direct_adapter is None:
        reason = "structure_hint verification requires local_direct_text adapter"
        result = _result(unit, "insufficient_evidence", _ids(hint_records), [], reason, [])
        return _with_reports(result, read_reports), hint_records
    direct_unit = _direct_verification_unit(unit)
    read_result = await _direct_records(direct_adapter, invoke_adapter, direct_unit, context)
    direct_records = [dict(record) for record in read_result.records]
    reports = [*read_reports, read_result.read_report.to_dict()]
    if not direct_records:
        reason = "structure_hint could not be verified by direct_text evidence"
        result = _result(unit, "insufficient_evidence", _ids(hint_records), [], reason, [])
        return _with_reports(result, reports), hint_records
    outcome = await reasoner.reason(
        direct_unit, direct_records, previous_results, read_reports=reports,
        dependency_evidence_records=context.execution_state.snapshot().evidence_records,
        plan=context.plan, execution_identity=(context.execution_state.identity.model_dump(mode="json")
                                               if context.execution_state.identity is not None else None),
    )
    if not outcome.claims:
        reason = "; ".join(outcome.rejected_reasons) or "structure_hint verification produced no claims"
        result = _result(unit, "insufficient_evidence", _ids(hint_records), [], reason, [])
        return _with_reports(result, reports), hint_records
    # 1. 【研究执行】【结构核实】核实属于当前派生产物，原始图线索与读取凭据保持当时事实
    records = [*hint_records, *direct_records]
    result = _result(unit, "evidence_found", _ids(records), outcome.claims, "", outcome.rejected_reasons)
    result = result.model_copy(update={"metadata": {**result.metadata, "provenance": outcome.provenance, "structure_verification": {
        "status": "verified", "hint_evidence_ids": _ids(hint_records), "direct_evidence_ids": _ids(direct_records),
    }}})
    return _with_reports(result, reports), records


async def _direct_records(
    adapter: Any,
    invoke_adapter: AdapterInvoker,
    unit: ResearchUnit,
    context: Any,
) -> EvidenceReadResult:
    """走执行层的受控入口取正文证据，只留可引用的正文记录。

    验证这一跳以前是直接 adapter(unit, context)，Milvus 不回话就在这里永久挂住；
    现在与其他取证调用共用同一个入口，重试与时间上限都由那边统一管。

    Args:
        adapter: 正文取证适配器
        invoke_adapter: 执行层注入的受控调用入口，带重试与时间上限
        unit: 已改面成 local_direct_text 的验证单元
        context: 执行上下文
    Returns:
        EvidenceReadResult: 可引用正文与该次读取说明
    """
    result = await invoke_adapter(adapter, unit, context)
    records = tuple(
        record for record in result.records
        if record.get("layer") == layers.EVIDENCE_LAYER_DIRECT_TEXT and record.get("usable_as_citation")
    )
    return EvidenceReadResult(records, result.read_report)


def _with_reports(result: ResearchUnitResult, reports: list[dict[str, Any]]) -> ResearchUnitResult:
    """参数为验证结果和两次读取说明；返回独立保存读取事实的单元结果副本。"""
    return result.model_copy(update={"metadata": {**result.metadata, "read_reports": reports}})


def _direct_verification_unit(unit: ResearchUnit) -> ResearchUnit:
    """参数为已声明正文核实的结构单元；仅切换读取来源，保留已校验的主张交付合同。"""
    policy = unit.evidence_policy.model_copy(update={"source_type": layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT})
    return unit.model_copy(update={"evidence_policy": policy})


def _result(
    unit: ResearchUnit,
    status: str,
    evidence_ids: list[str],
    claims: list[Any],
    unresolved_reason: str,
    rejected_reasons: list[str],
) -> ResearchUnitResult:
    return ResearchUnitResult(
        unit_id=unit.unit_id,
        status=status,
        summary=unit.goal if status == "evidence_found" else "",
        evidence_ids=evidence_ids,
        claims=claims,
        unresolved_reason=unresolved_reason,
        metadata={"source_type": unit.evidence_policy.source_type, "rejected_claims": rejected_reasons},
    )


def _ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record.get("evidence_id")) for record in records if record.get("evidence_id")]
