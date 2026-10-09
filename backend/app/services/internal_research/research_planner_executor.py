"""Service-level execution for validated research planner units."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Awaitable, Callable, Mapping, cast
from uuid import uuid4

from grpc import RpcError
from pymilvus.exceptions import MilvusException

from app.config import get_settings
from app.middleware.trace_context import evidence_scope, record_evidence_observation
from app.models.research_contracts import CapabilityStatus, EvidenceRecord
from app.models.knowledge_access_contracts import EvidenceReadResult, READ_CONTRACT_VERSION
from app.models.research_planner_contracts import (
    CURRENT_RESEARCH_CONTRACT_VERSION,
    EvidenceSourceType,
    ResearchPlan,
    ResearchUnit,
    ResearchUnitResult,
)
from app.models import evidence_layers as layers
from app.services.graph.neo4j_client import Neo4jDegradedError
from app.services.internal_research.evidence_identity import register_evidence_records
from app.services.internal_research.research_planner_execution_state import (
    PlannerExecutionState, result_input_signature, reuse_current_result, unit_business_keys,
)
from app.services.internal_research.research_planner_fulfillment import (
    evaluate_dependencies, evaluate_unit_fulfillment,
)
from app.services.internal_research.research_planner_lifecycle import settle_owned_work
from app.services.internal_research.research_planner_reasoner import PreparedReasoningInput, ResearchUnitReasoner
from app.services.internal_research.research_planner_structure_verifier import verify_structure_hint_unit
from app.services.llm_observation import http_error_facts, redact_error

logger = logging.getLogger(__name__)

EvidenceAdapter = Callable[
    [ResearchUnit, "PlannerExecutionContext"],
    Awaitable[EvidenceReadResult] | EvidenceReadResult,
]

# 单个取证能力的执行结果：(source_type 值, 四态之一)；单元间推断属规划推理、不产四态
CapabilityOutcome = tuple[str, CapabilityStatus]

# 不属于取证能力的 source_type：单元间推断，计划里有也不进四态表
_NON_CAPABILITY_SOURCE_TYPES = frozenset({
    layers.PLANNER_SOURCE_INFERENCE_FROM_UNITS,
})

# 章节正文能力产出的 provenance 只标识真实正文；读取说明独立传递，由 rag_node 的
# _planner_section_text_records 写入。正文记录与检索片段同 layer（direct_text），
# 按 layer 认领会把两种来源混起来，所以这里按 provenance 认领
_SECTION_TEXT_PROVENANCE_VALUES = frozenset({"section_path_read"})

# 按元数据筛块的单元三个条件全空时的阻塞原因：计划点了名却没给筛法，适配器不调用
_CHUNK_METADATA_FILTER_REQUIRED = "local_chunk_metadata requires a section, block type, or page filter"

# 可重试的基础设施故障白名单：显式可控，新增异常类必须在这里登记，
# 禁止在 except 块里按异常消息文本判可重试（照 neo4j_client._NEO4J_FAILURE_STATUS_MAP 的先例）
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,        # 本层 wait_for 到点
    OSError,             # 含连接类故障
    Neo4jDegradedError,  # 图谱侧已分类的故障
    MilvusException,     # Milvus 正常超时
    RpcError,            # Milvus 极短超时抛的 _InactiveRpcError 是它的子孙
)

# 辅助来源的失败只阻塞该单元、不终止整轮：结构线索与外部 arXiv 检索都属「取不到
# 也还能靠别的来源作答」。外部检索是单元主证据时，failed 四态 + 单元 blocked 的
# 可读原因仍可见，只是不让一个外部 API 故障把整轮答案打没（Q2 实测依据见 handoff）
_AUXILIARY_SOURCE_TYPES = frozenset({
    layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT,
    layers.PLANNER_SOURCE_EXTERNAL_CONTEXT,
})


class EvidenceCapabilityError(Exception):
    """取证适配器调用失败，携带失败面与可读原因向上冒。

    抛出即沿用既有硬失败路径：异常继续传到 execute_planner_state 的 except，由它把
    {source_type: "failed"} 与原因写进本轮状态；这里只把失败记清楚，不吞异常、
    不把硬失败变软（重试是后续切片的活，本层不重试）。

    Attributes:
        source_type: 失败的取证能力（计划里的 source_type 值）
        reason: 可读失败原因，含原始异常类型与消息
    """

    def __init__(self, source_type: str, reason: str) -> None:
        super().__init__(f"取证能力 {source_type} 执行失败：{reason}")
        self.source_type = source_type
        self.reason = reason


class EvidenceCapabilitySoftError(EvidenceCapabilityError):
    """辅助取证来源不可用的软失败信号：四态如实记 failed，但只阻塞该单元、不终止整轮。

    用于图谱关系这类「取不到也还能靠别的来源作答」的辅助来源：抛出后由执行层在
    _execute_local_policy 里消化成单元 blocked + 四态 failed，异常不冒到
    execute_planner_state 的 except；与 Milvus 侧取证能力的硬失败语义并存，别混用。

    Attributes:
        source_type: 不可用的取证能力（计划里的 source_type 值）
        reason: 可读失败原因，含原始异常类型与消息
    """


@dataclass(frozen=True, kw_only=True)
class PlannerExecutionContext:
    """取证与推理共享的本次运行上下文；序列化仅输出 execution_state 的快照。"""

    question: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_state: PlannerExecutionState = field(default_factory=PlannerExecutionState)
    request_resolver: Callable[[ResearchUnit, "PlannerExecutionContext"], Awaitable[Mapping[str, Any] | None] | Mapping[str, Any] | None] | None = None
    result_context: Mapping[str, Any] = field(default_factory=dict)
    plan: ResearchPlan | None = None
    previous_results: tuple[ResearchUnitResult, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PlannerExecutionReport:
    results: list[ResearchUnitResult]
    evidence_records: list[dict[str, Any]]
    # 本轮各取证能力的四态结果（source_type 值 → CapabilityStatus），取值严格走 CapabilityStatus 词表
    capability_results: dict[str, CapabilityStatus] = field(default_factory=dict)
    read_receipts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(kw_only=True)
class _ExecutionLedger:
    """一轮执行的产物账本：层内并发的单元按计划序回收结果与四态。

    原始材料只由执行上下文持有；这里不再维护一份独立证据池。

    Attributes:
        results: 已执行单元结果，顺序为「依赖层序 + 层内计划序」
        capabilities: 本轮各取证能力的四态表
    """

    results: list[ResearchUnitResult] = field(default_factory=list)
    capabilities: dict[str, CapabilityStatus] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ResearchPlannerExecutor:
    adapters: Mapping[EvidenceSourceType, EvidenceAdapter] = field(default_factory=dict)
    reasoner: ResearchUnitReasoner = field(default_factory=ResearchUnitReasoner)

    def with_adapter(
        self,
        source_type: EvidenceSourceType,
        adapter: EvidenceAdapter,
    ) -> "ResearchPlannerExecutor":
        adapters = {**self.adapters, source_type: adapter}
        return ResearchPlannerExecutor(adapters=adapters, reasoner=self.reasoner)

    def dispatch_source_types(self) -> set[EvidenceSourceType]:
        return set(self.adapters) | {layers.PLANNER_SOURCE_EXTERNAL_CONTEXT, layers.PLANNER_SOURCE_INFERENCE_FROM_UNITS}

    async def execute(
        self,
        plan: ResearchPlan,
        context: PlannerExecutionContext | None = None,
    ) -> list[ResearchUnitResult]:
        return (await self.execute_with_evidence(plan, context)).results

    async def execute_with_evidence(
        self,
        plan: ResearchPlan,
        context: PlannerExecutionContext | None = None,
        *,
        previous_results: list[ResearchUnitResult] | tuple[ResearchUnitResult, ...] = (),
    ) -> PlannerExecutionReport:
        """参数为当前计划、上下文及有效旧结果；接续材料并仅复用完全相同的推理输入。"""
        if plan.contract_version != CURRENT_RESEARCH_CONTRACT_VERSION:
            raise ValueError("旧研究计划仅允许历史读取；执行前必须按当前合同重新校验规划")
        base = context or PlannerExecutionContext(question=plan.question)
        reasoning_context = ({**base.result_context, "memory_grounded_allowed": self.reasoner.memory_grounded_allowed}
                             if base.result_context else {})
        ctx = replace(base, plan=plan, previous_results=tuple(previous_results), result_context=reasoning_context)
        if plan.metadata.execution_mode == "incremental":
            return await self._execute_incremental(plan, ctx)
        ledger = await self._execute_units_in_layers(_topological_unit_layers(plan.units), [], ctx)
        return _execution_report(ledger, ctx)

    async def _execute_incremental(self, plan: ResearchPlan, context: PlannerExecutionContext) -> PlannerExecutionReport:
        """参数为累计动作账本与上下文；只执行待办批次，历史产物重新验资后续接。"""
        pending = set(plan.metadata.pending_unit_ids)
        by_id = {unit.unit_id: unit for unit in plan.units}
        if not pending or not pending <= set(by_id):
            raise ValueError("incremental execution requires valid pending action identities")
        previous = {result.unit_id: result for result in context.previous_results}
        if len(previous) != len(context.previous_results) or not set(previous) <= set(by_id):
            raise ValueError("accumulated result identities conflict with accepted actions")
        carried: list[ResearchUnitResult] = []
        for layer in _topological_unit_layers(plan.units):
            for unit in layer:
                if unit.unit_id not in pending and unit.unit_id in previous:
                    carried.append(self._qualify_accumulated(unit, previous[unit.unit_id], carried, context))
        try:
            ledger = await self._execute_units_in_layers([[unit for unit in plan.units if unit.unit_id in pending]], carried, context)
        except BaseException as exc:
            report = getattr(exc, "planner_execution_report", None)
            if report is not None:
                setattr(exc, "planner_execution_report", replace(report, results=[*carried, *report.results]))
            raise
        return _execution_report(_ExecutionLedger(results=[*carried, *ledger.results], capabilities=ledger.capabilities), context)

    def _qualify_accumulated(
        self, unit: ResearchUnit, result: ResearchUnitResult, previous: list[ResearchUnitResult], context: PlannerExecutionContext,
    ) -> ResearchUnitResult:
        """参数为旧产物与当前输入；版本、支持和推理输入不符时保留原记录但取消其有效资格。"""
        qualified = _validate_result_fulfillment(unit, result, previous, context)
        if not result.claims or result.status not in {"evidence_found", "inferred"}:
            return qualified
        records = [record for record in context.execution_state.snapshot().evidence_records
                   if record["evidence_id"] in result.evidence_ids]
        prepared = self.reasoner.prepare(unit, records, previous,
            dependency_evidence_records=context.execution_state.snapshot().evidence_records, plan=context.plan)
        provenance = _reasoning_provenance(unit, records, previous, context,
                                          read_reports=result.metadata.get("read_reports", []), prepared=prepared)
        if not provenance["result_input_signature"] or provenance["result_input_signature"] != result.metadata.get("result_input_signature"):
            return qualified.model_copy(update={"status": "blocked", "unresolved_reason": "accumulated analysis inputs changed; choose a new analysis action"})
        return qualified

    async def execute_transient_units(
        self,
        units: list[ResearchUnit],
        previous_results: list[ResearchUnitResult],
        context: PlannerExecutionContext | None = None,
    ) -> PlannerExecutionReport:
        ctx = _transient_context(context or PlannerExecutionContext())
        base = list(previous_results)
        ledger = await self._execute_units_in_layers(_topological_unit_layers(units), base, ctx)
        return _execution_report(ledger, ctx)

    async def _execute_units_in_layers(
        self,
        layers: list[list[ResearchUnit]],
        previous_results: list[ResearchUnitResult],
        context: PlannerExecutionContext,
    ) -> _ExecutionLedger:
        """按依赖层执行单元：层内无依赖单元并发，层间严格按依赖序串行。

        依赖结果与四态按计划序回收；已完成读取即时形成材料快照，失败不抹掉兄弟成果。

        Args:
            layers: 依赖分层，层内单元互不依赖，由 _topological_unit_layers 产出
            previous_results: 本批执行之前的单元结果
            context: 执行上下文
        Returns:
            _ExecutionLedger: 已按层序与层内计划序登记的结果、证据与四态
        Raises:
            Exception: 层内首个失败单元的异常，按层内计划序重抛；同层兄弟已完成不取消
        """
        ledger = _ExecutionLedger()
        prior = list(previous_results)
        # 【研究执行】【并发调度】0 放开同层单元，跨层仍等待前序产物完成
        limit = get_settings().rag_planner_max_parallel_units
        try:
            for layer in layers:
                known = [*prior, *ledger.results]
                layer_outcomes = await _run_layer(self._execute_unit, layer, known, context, limit=limit)
                failures: list[BaseException] = []
                # 1. 【研究执行】【层级收尾】同层成功先全部回收，失败不能丢掉兄弟已取得的材料
                for outcome in layer_outcomes:
                    if isinstance(outcome, BaseException):
                        failures.append(outcome)
                        continue
                    result, _records, capability = outcome
                    _register_capability(ledger.capabilities, capability)
                    ledger.results.append(result)
                if failures:
                    raise failures[0]
        except BaseException as exc:
            await context.execution_state.settle()
            # 2. 【研究执行】【失败快照】附加事实不包装异常，取消仍由原取消路径收尾
            setattr(exc, "planner_execution_report", _execution_report(ledger, context))
            raise
        return ledger

    async def execute_plan(
        self,
        plan: ResearchPlan,
        context: PlannerExecutionContext | None = None,
    ) -> list[ResearchUnitResult]:
        return await self.execute(plan, context)

    async def _execute_unit(
        self,
        unit: ResearchUnit,
        previous_results: list[ResearchUnitResult],
        context: PlannerExecutionContext,
    ) -> tuple[ResearchUnitResult, list[dict[str, Any]], CapabilityOutcome | None]:
        """参数为单元、前序结果和上下文；返回执行结果、证据与该取证能力的四态，非取证能力为 None。"""
        with evidence_scope(
            "planner_unit", unit_id=unit.unit_id,
            details={"transient": bool(context.metadata.get("planner_transient_execution"))},
        ):
            started = time.perf_counter()
            snapshot = context.execution_state.snapshot()
            dependencies = evaluate_dependencies(
                unit, {result.unit_id: result for result in previous_results},
                evidence_records=snapshot.evidence_records,
                units_by_id={item.unit_id: item for item in context.plan.units} if context.plan else None,
            )
            block_reason = "; ".join(dependencies.blocking_reasons)
            outcome: tuple[ResearchUnitResult, list[dict[str, Any]], CapabilityOutcome | None]
            try:
                if block_reason:
                    # 1. 【评测观察】【单元执行】依赖阻塞保持原skipped语义，未执行不冒充空结果
                    outcome = _blocked_result(unit, block_reason), [], _skipped_capability(unit)
                elif unit.evidence_policy.source_type == layers.PLANNER_SOURCE_INFERENCE_FROM_UNITS:
                    outcome = await self._execute_inference(unit, previous_results, context), [], None
                else:
                    outcome = await self._execute_local_policy(unit, previous_results, context)
            except BaseException as exc:
                record_evidence_observation("unit_execution_result", {
                    "actor": "product", "unit_id": unit.unit_id,
                    "source_type": unit.evidence_policy.source_type, "status": "failed",
                    "error_type": type(exc).__name__, "reason": redact_error(exc),
                    "duration_seconds": time.perf_counter() - started,
                })
                raise
            result, _records, capability = outcome
            result = _validate_result_fulfillment(unit, result, previous_results, context)
            outcome = result, _records, capability
            record_evidence_observation("unit_execution_result", {
                "actor": "product", "unit_id": unit.unit_id, "status": "completed",
                "source_type": unit.evidence_policy.source_type, "unit_status": result.status,
                "capability_status": capability[1] if capability else None,
                "dependencies": [item.model_dump(mode="json") for item in unit.dependencies],
                "previous_results": [{"unit_id": item.unit_id, "status": item.status,
                                      "claims": [{"claim_type": claim.claim_type,
                                                  "validation_status": claim.validation_status} for claim in item.claims]}
                                     for item in previous_results],
                "reason": result.unresolved_reason, "evidence_ids": result.evidence_ids,
                "duration_seconds": time.perf_counter() - started,
            })
            return outcome

    async def _execute_local_policy(
        self,
        unit: ResearchUnit,
        previous_results: list[ResearchUnitResult],
        context: PlannerExecutionContext,
    ) -> tuple[ResearchUnitResult, list[dict[str, Any]], CapabilityOutcome | None]:
        """参数为单元、依赖及上下文；返回原有局部取证结果、过滤后的证据池与该能力的四态，压根没跑的能力为 None。"""
        source_type = unit.evidence_policy.source_type
        # 1. 计划点了名却没给定位词：适配器在册也不调用，如实记 skipped
        if source_type == layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT and not unit.evidence_policy.locator_queries:
            return _blocked_result(unit, "local_direct_text requires locator_queries"), [], (source_type, "skipped")
        if source_type == layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT and not (
                unit.evidence_policy.required_sections or unit.evidence_policy.continuation_ids):
            return _blocked_result(unit, "local_section_text requires a section or continuation"), [], (source_type, "skipped")
        # 三个筛块条件全空等于计划没给筛法：返回全篇块元数据没有意义，如实记 skipped
        if source_type == layers.PLANNER_SOURCE_LOCAL_CHUNK_METADATA and not _has_chunk_metadata_filters(unit):
            return _blocked_result(unit, _CHUNK_METADATA_FILTER_REQUIRED), [], (source_type, "skipped")
        adapter = self.adapters.get(source_type)
        if adapter is None:
            # 2. 枚举值没有对应适配器是装配缺陷（枚举有了、能力没接上），靠单元结果的 blocked 原因暴露；
            #    不记 skipped——那是「本轮不需要」，记成 skipped 会让缺陷在四态表里永远看不出来
            return _blocked_result(unit, f"no adapter registered for {source_type}"), [], None
        # 3. 【规划取证】【读取结果】四态按真实材料计数，完整性由读取报告独立说明
        read_result, soft_blocked = await _call_adapter_with_soft_block(adapter, unit, context)
        if soft_blocked is not None:
            # 图谱关系等辅助来源的软失败只阻塞该单元：四态如实记 failed，整轮照常往下走
            return soft_blocked, [], (source_type, "failed")
        assert read_result is not None
        reports = [read_result.read_report.to_dict()]
        capability: CapabilityOutcome = (source_type, "success" if read_result.records else "empty")
        records = [dict(record) for record in read_result.records]
        record_evidence_observation("unit_evidence_pool", {"records": records, "read_reports": reports})
        # 4. 结构线索只有计划索取正文验证时才多发一跳向量检索；不索取就走下面的通用收尾，
        #    线索作为规划已知事实进推理，终态最高到推断级（它不在可引用来源里）
        if source_type == layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT and unit.evidence_policy.verify_with_direct_text:
            return await self._verify_structure_hint(
                unit, records, previous_results, context, capability=capability, read_reports=reports,
            )
        if not records:
            reason = empty_read_reason(read_result, default="no evidence found")
            result = _insufficient_result(unit, records, reason)
        elif unit.expected_output.deliverable_kind == "materials":
            result = _unit_result(unit, "evidence_found", _evidence_ids(records), [], "", [])
        else:
            result = await self._result_from_reasoning(
                unit, records, previous_results, _success_status(unit), read_reports=reports,
                context=context,
            )
        result = result.model_copy(update={"metadata": {
            **result.metadata, "read_reports": reports,
            "read_associations": context.execution_state.read_associations(unit.unit_id),
        }})
        return result, records, capability

    async def _execute_inference(
        self,
        unit: ResearchUnit,
        previous_results: list[ResearchUnitResult],
        context: PlannerExecutionContext,
    ) -> ResearchUnitResult:
        """参数为推断单元、有效依赖及运行上下文；复用相同输入或重新生成当前主张。"""
        prepared = self.reasoner.prepare(unit, [], previous_results,
                    dependency_evidence_records=context.execution_state.snapshot().evidence_records, plan=context.plan)
        provenance = _reasoning_provenance(unit, [], previous_results, context, read_reports=[], prepared=prepared)
        reused = reuse_current_result(
            unit, signature=provenance["result_input_signature"], business_keys=provenance["unit_business_keys"],
            previous_results=context.previous_results, current_results=previous_results,
        )
        # 1. 【研究交付】【推理入口】输入与复用资格已核对，无论文读取的纯推理同样属于实际执行
        context.execution_state.mark_execution_started()
        if reused is not None:
            return reused
        outcome = await self.reasoner.reason(
            unit, [], previous_results,
            dependency_evidence_records=context.execution_state.snapshot().evidence_records,
            prepared_input=prepared, plan=context.plan,
            execution_identity=provenance["provenance"]["execution_identity"],
        )
        if not outcome.claims:
            reason = "; ".join(outcome.rejected_reasons) or "dependency claims did not support inference"
            blocked = _blocked_result(unit, reason)
            return blocked.model_copy(update={"metadata": {**blocked.metadata, **provenance,
                                                           "rejected_claims": outcome.rejected_reasons}})
        result = _unit_result(unit, "inferred", [], outcome.claims, "", outcome.rejected_reasons)
        return result.model_copy(update={"metadata": {**result.metadata, **provenance}})

    async def _result_from_reasoning(
        self,
        unit: ResearchUnit,
        records: list[dict[str, Any]],
        previous_results: list[ResearchUnitResult],
        status: str,
        *,
        read_reports: list[dict[str, Any]] | None = None,
        context: PlannerExecutionContext,
    ) -> ResearchUnitResult:
        """参数为单元、材料、前序结果、状态及读取报告；返回依据实际材料推理的单元结果。"""
        prepared = self.reasoner.prepare(unit, records, previous_results,
                    dependency_evidence_records=context.execution_state.snapshot().evidence_records, plan=context.plan)
        provenance = _reasoning_provenance(unit, records, previous_results, context,
                                          read_reports=read_reports or [], prepared=prepared)
        reused = reuse_current_result(
            unit, signature=provenance["result_input_signature"], business_keys=provenance["unit_business_keys"],
            previous_results=context.previous_results, current_results=previous_results,
        )
        if reused is not None:
            return reused
        outcome = await self.reasoner.reason(
            unit, records, previous_results, read_reports=read_reports,
            dependency_evidence_records=context.execution_state.snapshot().evidence_records,
            prepared_input=prepared, plan=context.plan,
            execution_identity=provenance["provenance"]["execution_identity"],
        )
        if not outcome.claims:
            reason = "; ".join(outcome.rejected_reasons) or "claim validation produced no claims"
            insufficient = _insufficient_result(unit, records, reason)
            return insufficient.model_copy(update={"metadata": {**insufficient.metadata, **provenance,
                                                                "rejected_claims": outcome.rejected_reasons}})
        result = _unit_result(unit, status, _evidence_ids(records), outcome.claims, "", outcome.rejected_reasons)
        return result.model_copy(update={"metadata": {**result.metadata, **provenance}})

    async def _verify_structure_hint(
        self,
        unit: ResearchUnit,
        records: list[dict[str, Any]],
        previous_results: list[ResearchUnitResult],
        context: PlannerExecutionContext,
        *,
        capability: CapabilityOutcome,
        read_reports: list[dict[str, Any]],
    ) -> tuple[ResearchUnitResult, list[dict[str, Any]], CapabilityOutcome | None]:
        """计划索取了正文核实的结构线索单元的收尾：核实通过才可引用，核实取不到就只阻塞这一个单元。

        Args:
            unit: 当前 local_structure_hint 单元，已确认索取了正文核实
            records: 已过滤的结构线索记录，本身不可引用
            previous_results: 前序单元结果，供推理器参考
            context: 执行上下文
            capability: 线索取证本身的四态，验证这一跳不改写它
        Returns:
            tuple: (单元结果, 进池证据, 四态)；验证不可用时记 failed，整轮照常往下走
        """
        try:
            result, verified = await verify_structure_hint_unit(
                unit=unit,
                hint_records=records,
                direct_adapter=self.adapters.get(layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT),
                invoke_adapter=self._invoke_verification_adapter,
                reasoner=self.reasoner,
                previous_results=previous_results,
                context=context,
                read_reports=read_reports,
            )
        except EvidenceCapabilitySoftError as exc:
            # 验证跳的失败面记的是结构线索这个辅助能力，所以走软失败：四态如实记 failed，不终止整轮
            blocked = _blocked_result(unit, exc.reason)
            blocked = blocked.model_copy(update={"metadata": {**blocked.metadata, "read_reports": read_reports}})
            return blocked, [], (unit.evidence_policy.source_type, "failed")
        return result, verified, capability

    async def _invoke_verification_adapter(
        self,
        adapter: EvidenceAdapter,
        unit: ResearchUnit,
        context: PlannerExecutionContext,
    ) -> EvidenceReadResult:
        """结构验证那一跳的正文取证入口：与其余取证共用重试与时间上限，失败面记回结构线索能力。

        验证单元的 source_type 已被改面成 local_direct_text，若照它记失败面，图谱线索的验证
        超时会变成正文能力的硬失败并终止整轮；计划里点的是辅助能力，失败就该只落在它头上。

        Args:
            adapter: 正文取证适配器
            unit: 已改面成 local_direct_text 的验证单元
            context: 执行上下文
        Returns:
            list[dict[str, Any]]: 适配器返回的证据记录
        """
        return await _call_adapter(
            adapter,
            unit,
            context,
            failure_source_type=layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT,
        )


async def _call_adapter(
    adapter: EvidenceAdapter,
    unit: ResearchUnit,
    context: PlannerExecutionContext,
    *,
    failure_source_type: str | None = None,
) -> EvidenceReadResult:
    """参数为适配器、单元和上下文；核验权限后统一复用、共享并登记真实读取事实。"""
    request = None
    if context.request_resolver is not None:
        resolved = context.request_resolver(unit, context)
        request = await resolved if inspect.isawaitable(resolved) else resolved
    try:
        result = await context.execution_state.read(
            unit_id=unit.unit_id, source_type=unit.evidence_policy.source_type, request=request,
            invoke=partial(_call_adapter_owned, adapter, unit, context),
        )
    except EvidenceCapabilityError as exc:
        if (failure_source_type is not None and failure_source_type in _AUXILIARY_SOURCE_TYPES
                and not isinstance(exc, EvidenceCapabilitySoftError)):
            raise EvidenceCapabilitySoftError(failure_source_type, exc.reason) from exc
        raise
    association = context.execution_state.read_associations(unit.unit_id)[-1]
    record_evidence_observation("evidence_read_request", {
        "actor": "product", "unit_id": unit.unit_id,
        "source_type": unit.evidence_policy.source_type, **association,
        "status": "success" if result.records else "empty", "result_count": len(result.records),
    })
    return result


# 所有消费者使用同一个调用边界；旧内部名字仍指向这个实现
read_evidence = _call_adapter


async def _call_adapter_owned(
    adapter: EvidenceAdapter,
    unit: ResearchUnit,
    context: PlannerExecutionContext,
    *,
    failure_source_type: str | None = None,
) -> EvidenceReadResult:
    """调用适配器并返回其记录；基础设施故障按配置重试，重试不成立才换成带取证面的显式异常上抛。

    Args:
        adapter: 该 source_type 对应的取证适配器
        unit: 当前研究单元，默认用它的 source_type 作为失败面
        context: 执行上下文
        failure_source_type: 显式指定失败面，供内部改过面的单元把失败记回计划里的能力
    Returns:
        EvidenceReadResult: 适配器返回的真实材料及读取说明
    Raises:
        EvidenceCapabilitySoftError: 辅助来源（图谱关系）失败，或适配器本身已是软失败
        EvidenceCapabilityError: 其余取证失败；原始异常链在 __cause__ 里保留
    """
    cfg = get_settings()
    source_type = failure_source_type or unit.evidence_policy.source_type
    max_retries = cfg.rag_planner_evidence_max_retries
    tool_call_id = str(uuid4())
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        identity = {
            "actor": "product", "tool_call_id": tool_call_id, "tool_attempt": attempt + 1,
            "unit_id": unit.unit_id, "source_type": source_type,
            "parameters": unit.evidence_policy.model_dump(mode="json"),
        }
        try:
            context.execution_state.note_adapter_attempt(unit.unit_id)
            result = await _invoke_adapter(adapter, unit, context)
            record_evidence_observation("tool_execution_result", {
                **identity, "status": "success" if result.records else "empty", "result_count": len(result.records),
                "read_report": result.read_report.to_dict(),
                "duration_seconds": time.perf_counter() - started,
            })
            records = _filter_records(
                unit, [dict(record) for record in result.records], collect_scope_conflicts=_collect_scope_conflicts(context),
            )
            return EvidenceReadResult(tuple(cast(EvidenceRecord, record) for record in records), result.read_report)
        except asyncio.CancelledError as exc:
            record_evidence_observation("tool_execution_result", {
                **identity, "status": "cancelled", "error_type": type(exc).__name__,
                "duration_seconds": time.perf_counter() - started,
            })
            raise
        except Exception as exc:
            record_evidence_observation("tool_execution_result", {
                **identity, "status": "failed", "error_type": type(exc).__name__,
                "reason": redact_error(exc), "duration_seconds": time.perf_counter() - started,
                **http_error_facts(exc, boundary="unknown"),
            })
            if _is_retryable(exc) and attempt < max_retries:
                logger.info(
                    f"【规划取证】【{source_type}】调用失败，重试 {attempt + 1}/{max_retries}：{_readable_reason(exc)}"
                )
                await asyncio.sleep(cfg.rag_planner_evidence_retry_backoff_seconds)
                continue
            # 1. 次数用尽或异常不在白名单里才是最终失败，日志带失败面与可读原因
            logger.error(f"【规划取证】【{source_type}】调用失败且重试不成立：{_readable_reason(exc)}")
            failure = _as_capability_failure(exc, source_type)
            # 2. 已是取证失败（含软子类）原样上抛：重包会把软失败洗成硬失败，也不能覆盖 __cause__ 上的原始异常链
            if failure is exc:
                raise
            raise failure from exc


async def _invoke_adapter(
    adapter: EvidenceAdapter,
    unit: ResearchUnit,
    context: PlannerExecutionContext,
) -> EvidenceReadResult:
    """调用适配器并持有其生命周期，超时或取消先发出终止信号，待清理结束再退出。

    Args:
        adapter: 该 source_type 对应的取证适配器
        unit: 当前研究单元
        context: 执行上下文
    Returns:
        EvidenceReadResult: 适配器返回的真实材料及读取说明
    Raises:
        TimeoutError: 适配器超过 rag_planner_evidence_timeout_seconds 未返回；
            清理完成后补可读原因再抛，清理等待允许超过配置时间上限
        asyncio.CancelledError: 用户取消优先于超时；已启动查询清理完成后重抛
    """
    result = adapter(unit, context)
    if not inspect.isawaitable(result):
        return _require_read_result(result, source_type=unit.evidence_policy.source_type)
    timeout = get_settings().rag_planner_evidence_timeout_seconds
    task = asyncio.ensure_future(result)
    try:
        # 1. 【规划取证】【适配器收尾】超时只终止等待包装，由本层持有并取消真实适配器
        result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        return _require_read_result(result, source_type=unit.evidence_policy.source_type)
    except (TimeoutError, asyncio.CancelledError) as exc:
        task.cancel()
        outcome, cancellation = await settle_owned_work(task)
        # 2. 【规划取证】【适配器收尾】再次取消不打断清理；超时清理时收到用户取消则按取消退出
        termination = cancellation or exc
        if isinstance(termination, TimeoutError) and not str(termination).strip():
            termination = TimeoutError(f"取证调用超过 {timeout:g} 秒未返回")
        # 3. 【研究执行】【取消后完成】不可中断工作已成功返回时，事实随终止信号交还读取所有者
        if isinstance(outcome, EvidenceReadResult):
            try:
                completed = _require_read_result(outcome, source_type=unit.evidence_policy.source_type)
                records = _filter_records(unit, [dict(record) for record in completed.records],
                                          collect_scope_conflicts=_collect_scope_conflicts(context))
                context.execution_state.record_interrupted_read(
                    unit.unit_id, EvidenceReadResult(tuple(cast(EvidenceRecord, record) for record in records), completed.read_report),
                )
            except Exception as validation_error:
                raise termination from validation_error
        # 4. 【规划取证】【适配器收尾】保留清理期间的真实故障原因，不能覆盖原超时或取消身份
        if isinstance(outcome, BaseException) and outcome is not termination:
            raise termination from outcome
        if termination is not exc:
            raise termination from exc
        raise


def empty_read_reason(result: EvidenceReadResult, *, default: str) -> str:
    """参数为空材料读取结果及常规空结果说明；返回保留快照内未发布事实的原因，不改变四态。"""
    if any(item.get("reason") == "not_in_publication_snapshot" for item in result.read_report.remaining):
        return "本轮快照中有目标论文没有可读的已发布正文；该部分尚未读取，不能据此判断内容不存在"
    return default


def _require_read_result(result: Any, *, source_type: str) -> EvidenceReadResult:
    """参数为适配器返回值及实际能力；只接受统一合同，本地正文缺少新版来源即报错。"""
    if not isinstance(result, EvidenceReadResult):
        raise TypeError("EvidenceAdapter 必须返回 EvidenceReadResult")
    if not result.records and any(item.get("reason") == "ambiguous" for item in result.read_report.remaining):
        raise ValueError("章节定位存在歧义，未执行正文读取；需要使用完整章节路径")
    # 1. 【知识取证】【实时合同】实时正文必须携带来源范围；历史记录只在保存结果回读时接受
    local_body = source_type in {layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT, layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT}
    for record in result.records:
        if local_body or record.get("layer") == layers.EVIDENCE_LAYER_DIRECT_TEXT:
            version = record.get("read_contract_version")
            if type(version) is not int or version != READ_CONTRACT_VERSION:
                raise ValueError("实时正文证据必须包含有效的 read_contract_version 和来源范围")
    register_evidence_records({}, result.records)
    return result


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否属可重试的基础设施故障。

    软失败把原始异常挂在 __cause__ 上，所以本体或 __cause__ 命中白名单都算可重试。

    Args:
        exc: 适配器或本层超时抛出的异常
    Returns:
        bool: 可重试为 True
    """
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    cause = exc.__cause__
    return cause is not None and isinstance(cause, _RETRYABLE_EXCEPTIONS)


def _as_capability_failure(exc: Exception, source_type: str) -> EvidenceCapabilityError:
    """把调用失败按 source_type 归一成带失败面的取证异常。

    已是取证失败（含软子类）原样返回——重包一次会把软失败洗成硬失败；辅助来源
    （图谱关系）包成软失败，其余包成硬失败。原始异常链由调用方的 raise ... from 保留。

    Args:
        exc: 适配器或本层超时抛出的异常
        source_type: 失败的取证能力（计划里的 source_type 值）
    Returns:
        EvidenceCapabilityError: 最终要上抛的取证失败异常
    """
    if isinstance(exc, EvidenceCapabilityError):
        return exc
    reason = _readable_reason(exc)
    if source_type in _AUXILIARY_SOURCE_TYPES:
        return EvidenceCapabilitySoftError(source_type, reason)
    return EvidenceCapabilityError(source_type, reason)


def _readable_reason(exc: BaseException) -> str:
    """把异常归一成失败消息与日志共用的可读原因。

    Args:
        exc: 待归一的异常
    Returns:
        str: 「类型名: 消息」；消息为空时只留类型名
    """
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


async def _call_adapter_with_soft_block(
    adapter: EvidenceAdapter,
    unit: ResearchUnit,
    context: PlannerExecutionContext,
) -> tuple[EvidenceReadResult | None, ResearchUnitResult | None]:
    """调用适配器并归一记录；辅助来源的软失败换成阻塞结果，不让异常继续往上冒。

    参数:
        adapter: 该 source_type 对应的取证适配器
        unit: 当前研究单元，软失败时用它组装带可读原因的阻塞结果
        context: 执行上下文
    返回:
        (read_result, soft_blocked)：正常时完整返回材料与说明；
        辅助来源失败时 read_result 为 None、阻塞结果带原因，调用方照实记 failed
    """
    try:
        raw = await _call_adapter(adapter, unit, context)
    except EvidenceCapabilitySoftError as exc:
        return None, _blocked_result(unit, exc.reason)
    return raw, None


def _skipped_capability(unit: ResearchUnit) -> CapabilityOutcome | None:
    """依赖阻塞等压根没调用的场景如实记 skipped；单元间推断属规划推理、不产四态。"""
    source_type = unit.evidence_policy.source_type
    if source_type in _NON_CAPABILITY_SOURCE_TYPES:
        return None
    return (source_type, "skipped")


def _register_capability(
    capabilities: dict[str, CapabilityStatus],
    outcome: CapabilityOutcome | None,
) -> None:
    """把单元级四态并进本轮能力表；同名能力取最后一次执行结果，不产四态的能力不入表。"""
    if outcome is None:
        return
    source_type, status = outcome
    capabilities[source_type] = status


def _topological_unit_layers(units: list[ResearchUnit]) -> list[list[ResearchUnit]]:
    """按依赖把单元分层：同层两两无依赖可并发，层间必须串行。

    分层算法沿用原来的 Kahn 逐轮取 ready，只是把每轮取出的 ready 集合原样作为一层返回，
    不再拍平——拍平会丢掉「哪些单元互不依赖」这个唯一能安全并发的信息。

    Args:
        units: 计划里的研究单元列表
    Returns:
        list[list[ResearchUnit]]: 依赖层列表，第 0 层无依赖；层内保持计划里的原始顺序
    Raises:
        ValueError: 计划里存在环，取不出 ready 集合
    """
    pending = {unit.unit_id: unit for unit in units}
    layers: list[list[ResearchUnit]] = []
    while pending:
        ready = [unit for unit in pending.values() if all(dep.unit_id not in pending for dep in unit.dependencies)]
        if not ready:
            raise ValueError("research plan contains unresolved dependency order")
        layers.append(ready)
        for unit in ready:
            pending.pop(unit.unit_id)
    return layers


async def _run_layer(
    execute: Callable[..., Awaitable[Any]],
    layer: list[ResearchUnit],
    previous_results: list[ResearchUnitResult],
    context: PlannerExecutionContext,
    *,
    limit: int,
) -> list[Any]:
    """并发执行同一层的研究单元，按入参顺序返回各单元产出（异常也在其中）。

    单元之间没有依赖，所以并发安全；层内并发数受 limit 约束，超出的单元排队，
    不是直接丢弃。异常不在这里处理，由调用方按层内顺序重抛，避免一个单元失败
    就让同层其他失败原因丢失。

    Args:
        execute: 单单元执行函数，签名与 ResearchPlannerExecutor._execute_unit 一致
        layer: 同一依赖层的单元，两两无依赖
        previous_results: 本层开始前已完成的单元结果
        context: 执行上下文
        limit: 层内最大并发单元数，0 表示该层全部单元可同时启动
    Returns:
        list[Any]: 与 layer 同序的产出；失败的单元位置上是它抛出的异常
    """
    semaphore = asyncio.Semaphore(len(layer) if limit == 0 else limit)
    tasks = [asyncio.create_task(_run_unit_guarded(semaphore, execute, unit, previous_results, context))
             for unit in layer]
    batch = asyncio.gather(*tasks, return_exceptions=True)
    try:
        return await asyncio.shield(batch)
    except asyncio.CancelledError as exc:
        # 1. 【研究执行】【整层取消】先终止全部消费者，再等待其共享工作收尾
        for task in tasks:
            task.cancel()
        await settle_owned_work(batch)
        # 2. 【研究执行】【取消归因】逐个回收原任务，避免最后结束的正常取消盖掉较早的清理故障
        failures: list[BaseException] = []
        for task in tasks:
            outcome, _cancellation = await settle_owned_work(task)
            if isinstance(outcome, BaseException):
                cause = outcome.__cause__ if isinstance(outcome, asyncio.CancelledError) else outcome
                if cause is not None:
                    failures.append(cause)
        if failures:
            raise exc from failures[0]
        raise


def _execution_report(ledger: _ExecutionLedger, context: PlannerExecutionContext) -> PlannerExecutionReport:
    """参数为当前产物账本及运行上下文；将原材料与凭据成对交回正常或失败调用方。"""
    snapshot = context.execution_state.snapshot()
    return PlannerExecutionReport(
        results=list(ledger.results), evidence_records=snapshot.evidence_records,
        read_receipts=snapshot.read_receipts, capability_results=dict(ledger.capabilities),
    )


def _validate_result_fulfillment(
    unit: ResearchUnit, result: ResearchUnitResult,
    previous_results: list[ResearchUnitResult], context: PlannerExecutionContext,
) -> ResearchUnitResult:
    """参数为实际产物及当前依赖；统一报告决定局部状态，已有局部主张仍保留。"""
    registry = {str(record["evidence_id"]): record for record in context.execution_state.snapshot().evidence_records}
    report = evaluate_unit_fulfillment(
        unit, result, registry=registry, results_by_id={item.unit_id: item for item in previous_results},
        plan=context.plan,
    )
    metadata = {**result.metadata, "fulfillment": report.model_dump(mode="json")}
    if report.hard_gaps and result.status in {"evidence_found", "inferred"}:
        return result.model_copy(update={
            "status": "insufficient_evidence", "metadata": metadata,
            "unresolved_reason": "; ".join(gap.reason for gap in report.hard_gaps),
        })
    return result.model_copy(update={"metadata": metadata})


def _reasoning_provenance(
    unit: ResearchUnit, records: list[dict[str, Any]], previous_results: list[ResearchUnitResult],
    context: PlannerExecutionContext, *, read_reports: list[dict[str, Any]], prepared: PreparedReasoningInput,
) -> dict[str, Any]:
    """参数为实际送模输入与运行配置；生成结果接续凭据，缺失配置时不开放免推理。"""
    keys = unit_business_keys(context.plan) if context.plan is not None else {}
    identity = context.execution_state.identity
    reasoning_context = ({**context.result_context, "question": context.question,
                          "execution_identity": identity.model_dump(mode="json")}
                         if context.result_context and identity is not None else {})
    if reasoning_context and unit.dependencies:
        reasoning_context["dependency_evidence_records"] = sorted(
            (prepared.records if context.plan and context.plan.metadata.execution_mode == "incremental"
             else context.execution_state.snapshot().evidence_records), key=lambda record: str(record["evidence_id"]),
        )
    signature = result_input_signature(
        unit, business_keys=keys, records=records, read_reports=read_reports,
        dependency_results=previous_results, reasoning_context=reasoning_context, exposure=prepared.consumer.exposure,
    )
    return {"result_input_signature": signature, "unit_business_keys": keys, "result_reused": False,
            "provenance": {"unit": unit.model_dump(mode="json"),
                           "exposure": prepared.consumer.exposure.model_dump(mode="json"),
                           "execution_identity": identity.model_dump(mode="json") if identity is not None else None}}


async def _run_unit_guarded(
    semaphore: asyncio.Semaphore,
    execute: Callable[..., Awaitable[Any]],
    unit: ResearchUnit,
    previous_results: list[ResearchUnitResult],
    context: PlannerExecutionContext,
) -> Any:
    """按并发上限执行单个单元；排队等待的单元在拿到名额后才开始取证调用。"""
    async with semaphore:
        return await execute(unit, previous_results, context)


def _transient_context(context: PlannerExecutionContext) -> PlannerExecutionContext:
    """参数为当前运行上下文；只标识补证角色，保留相同材料池与共享读取所有者。"""
    metadata = {**dict(context.metadata), "planner_transient_execution": True}
    return replace(context, metadata=metadata)


def _collect_scope_conflicts(context: PlannerExecutionContext) -> bool:
    return bool(context.metadata.get("planner_transient_execution"))


def _has_chunk_metadata_filters(unit: ResearchUnit) -> bool:
    """筛块至少要一个条件：章节 / 块类型 / 页码全空时适配器不被调用，如实记 skipped。"""
    policy = unit.evidence_policy
    return bool(policy.required_sections or policy.required_block_types or policy.required_pages)


def _filter_records(
    unit: ResearchUnit,
    records: list[dict[str, Any]],
    *,
    collect_scope_conflicts: bool = False,
) -> list[dict[str, Any]]:
    source_type = unit.evidence_policy.source_type
    target_ids = set(unit.evidence_policy.target_paper_ids)
    return [
        record for record in records
        if _record_matches_source(record, source_type)
        and _record_matches_scope(record, target_ids, source_type, collect_scope_conflicts)
    ]


def _record_matches_source(record: dict[str, Any], source_type: str) -> bool:
    if source_type == layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT:
        return record.get("layer") == layers.EVIDENCE_LAYER_DIRECT_TEXT and bool(record.get("usable_as_citation"))
    if source_type == layers.PLANNER_SOURCE_LOCAL_METADATA:
        return record.get("layer") == layers.EVIDENCE_LAYER_METADATA or record.get("source") == layers.EVIDENCE_SOURCE_INVENTORY
    if source_type == layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT:
        # 图谱关系记录一律带 neo4j 源，按源认领即可；原先「或层级是结构线索」那半句会把
        # 章节结构记录与正文截断概览（同为结构线索层、源是 milvus）一并认领，
        # 今天因筛选逐适配器跑、记录池不混而未发作，合并池后会立刻串味
        return record.get("source") == layers.EVIDENCE_SOURCE_NEO4J
    if source_type == layers.PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE:
        return (
            record.get("layer") == layers.EVIDENCE_LAYER_STRUCTURE_HINT
            and record.get("source") == layers.EVIDENCE_SOURCE_MILVUS
        )
    if source_type == layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT:
        return record.get("provenance") in _SECTION_TEXT_PROVENANCE_VALUES
    if source_type == layers.PLANNER_SOURCE_LOCAL_CHUNK_METADATA:
        # 筛块材料来自索引元数据；论文级详情走 inventory 源，不会被这条误收
        return (
            record.get("layer") == layers.EVIDENCE_LAYER_METADATA
            and record.get("source") == layers.EVIDENCE_SOURCE_MILVUS
        )
    if source_type == layers.PLANNER_SOURCE_EXTERNAL_CONTEXT:
        # 外部 arXiv 记录按源认领（只认 arxiv，inventory 的论文级元数据不混入）；
        # 它不可引用，且 paper_id 是 arXiv ID——范围过滤对 external 无条件放行
        return record.get("source") == layers.EVIDENCE_SOURCE_ARXIV
    return False


def _record_matches_scope(
    record: dict[str, Any],
    target_ids: set[str],
    source_type: str,
    collect_scope_conflicts: bool,
) -> bool:
    if not target_ids:
        return True
    paper_id = str(record.get("paper_id") or "").strip()
    if source_type == layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT and collect_scope_conflicts:
        return True
    if source_type == layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT:
        return paper_id in target_ids
    # 外部检索（arXiv）的 paper_id 是 arXiv ID，本就不在本轮项目范围里；
    # 修证记录的归属判断只管库内论文，外部命中如实放行，不冒充范围冲突
    if source_type == layers.PLANNER_SOURCE_EXTERNAL_CONTEXT:
        return True
    return not paper_id or paper_id in target_ids


def _success_status(unit: ResearchUnit) -> str:
    return "evidence_found"


def _blocked_result(unit: ResearchUnit, reason: str) -> ResearchUnitResult:
    return _unit_result(unit, "blocked", [], [], reason, [])


def _insufficient_result(
    unit: ResearchUnit,
    records: list[dict[str, Any]],
    reason: str,
) -> ResearchUnitResult:
    return _unit_result(unit, "insufficient_evidence", _evidence_ids(records), [], reason, [])


def _unit_result(
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
        summary=unit.goal if status in {"evidence_found", "inferred"} else "",
        evidence_ids=evidence_ids,
        claims=claims,
        unresolved_reason=unresolved_reason,
        metadata={"source_type": unit.evidence_policy.source_type, "rejected_claims": rejected_reasons},
    )


def _evidence_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record.get("evidence_id")) for record in records if record.get("evidence_id")]


PlannerUnitExecutor = ResearchPlannerExecutor


async def execute_research_plan(
    plan: ResearchPlan,
    *,
    adapters: Mapping[EvidenceSourceType, EvidenceAdapter],
    reasoner: ResearchUnitReasoner | None = None,
    context: PlannerExecutionContext | None = None,
) -> list[ResearchUnitResult]:
    executor = ResearchPlannerExecutor(adapters=adapters, reasoner=reasoner or ResearchUnitReasoner())
    return await executor.execute(plan, context)
