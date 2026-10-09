"""同一次研究的材料、读取凭据与共享工作所有权，作者：xxx。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.knowledge_access_contracts import READ_CONTRACT_VERSION, EvidenceReadResult, KnowledgeReadReport, SourceSpan
from app.models.research_contracts import EvidenceRecord
from app.models.research_planner_contracts import ConsumerExposure, ResearchPlan, ResearchUnit, ResearchUnitResult
from app.services.internal_research.evidence_identity import register_evidence_records
from app.services.internal_research.research_planner_lifecycle import settle_owned_work

EXECUTION_STATE_VERSION: Literal[1] = 1
_BODY_SOURCES = frozenset({"local_direct_text", "local_section_text"})


def content_digest(value: object) -> str:
    """参数为可序列化内容；返回不依赖字典插入顺序的内容摘要。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExecutionIdentity(BaseModel):
    """服务端运行、用户、有效范围和不可变发布快照；租约不属于内容身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    scope: Mapping[str, Any]
    publication_snapshot: Mapping[str, Any]


class ReadReportSnapshot(BaseModel):
    """读取说明的严格序列化合同，跨 checkpoint 后仍保持完整性与未读范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: str
    status: Literal["complete", "partial", "not_applicable", "unavailable"]
    requested: tuple[str, ...] = ()
    resolved: tuple[str, ...] = ()
    remaining: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    used_chars: int = Field(default=0, ge=0)
    read_spans: tuple[dict[str, Any], ...] = ()

    def to_report(self) -> KnowledgeReadReport:
        """无参数；返回独立的知识访问读取说明，不补造任何已读位置。"""
        return KnowledgeReadReport(
            operation=self.operation, status=self.status, requested=self.requested, resolved=self.resolved,
            remaining=deepcopy(self.remaining), limitations=self.limitations, used_chars=self.used_chars,
            read_spans=tuple(cast(SourceSpan, deepcopy(span)) for span in self.read_spans),
        )


class ReadReceipt(BaseModel):
    """一次真实取证的事实，不保存正文，也不据此判定研究任务完成。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal[1] = EXECUTION_STATE_VERSION
    receipt_id: str = Field(min_length=1)
    identity: ExecutionIdentity | None
    source_execution_id: str = Field(min_length=1)
    source_unit_id: str = Field(min_length=1)
    source_type: str
    request: dict[str, Any] | None
    request_key: str
    evidence_ids: tuple[str, ...] = ()
    read_report: ReadReportSnapshot | None = None
    outcome: Literal["success", "empty", "unavailable", "failed", "cancelled"]
    actual_call_count: int = Field(ge=0)
    reuse_eligible: bool = True
    error_type: str = ""
    reason: str = ""

    @model_validator(mode="after")
    def validate_completed_read(self) -> "ReadReceipt":
        """无参数；确保成功凭据有报告，失败凭据不能宣称已取得正文。"""
        if self.outcome in {"success", "empty", "unavailable"} and self.read_report is None:
            raise ValueError("已完成的读取凭据必须包含 read_report")
        if self.outcome in {"failed", "cancelled"} and self.evidence_ids:
            raise ValueError("失败读取不能登记成功材料")
        return self


@dataclass(frozen=True, kw_only=True)
class ExecutionSnapshot:
    """材料与凭据的同一可回写快照，不包含进程内任务。"""

    evidence_records: list[dict[str, Any]]
    read_receipts: list[dict[str, Any]]


@dataclass(kw_only=True)
class _SharedRead:
    """一次真实取证的所有者及其仍在等待的消费者数量。"""

    task: asyncio.Task[ReadReceipt]
    waiters: int = 0


class PlannerExecutionState:
    """持有本次研究的原始成果与在途读取；结果有效性由执行器分别判断。"""

    def __init__(
        self, identity: ExecutionIdentity | None = None, *,
        evidence_records: Sequence[Mapping[str, Any]] = (),
        read_receipts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """参数为运行身份和成对保存的材料/凭据；校验后接续，不恢复历史 Task。"""
        self.identity = identity.model_copy(deep=True) if identity else None
        self.execution_id = str(uuid4())
        self._registry = register_evidence_records({}, deepcopy(evidence_records))
        self._validate_material_binding(list(self._registry.values()))
        self._receipts: list[ReadReceipt] = []
        self._cache: dict[str, ReadReceipt] = {}
        self._pending: dict[str, _SharedRead] = {}
        self._associations: dict[str, list[dict[str, Any]]] = {}
        self._attempts: dict[str, int] = {}
        self._interrupted_attempts: dict[str, int] = {}
        self._active_read_fields: dict[str, dict[str, Any]] = {}
        self._execution_started = False
        for raw in read_receipts:
            version = raw.get("contract_version")
            if type(version) is not int or version != EXECUTION_STATE_VERSION:
                raise ValueError("读取凭据缺少当前执行合同版本，需要重新校验规划")
            receipt = ReadReceipt.model_validate(raw)
            if identity is None or receipt.identity != identity:
                raise ValueError("读取凭据的研究运行、用户、范围或发布快照不一致")
            if any(evidence_id not in self._registry for evidence_id in receipt.evidence_ids):
                raise ValueError("读取凭据引用的材料不在同一快照中")
            self._receipts.append(receipt)
            self._cache_receipt(receipt)

    @property
    def execution_started(self) -> bool:
        """无参数；返回本次上下文是否已通过输入校验并开始读取、有效复用或推理。"""
        return self._execution_started

    def mark_execution_started(self) -> None:
        """【研究交付】【执行事实】无参数；在实际工作入口登记已进入执行，无返回值。"""
        self._execution_started = True

    def snapshot(self) -> ExecutionSnapshot:
        """无参数；返回相互一致且与运行账本隔离的材料和凭据副本。"""
        return ExecutionSnapshot(
            evidence_records=deepcopy(list(self._registry.values())),
            read_receipts=[receipt.model_dump(mode="json") for receipt in self._receipts],
        )

    def read_associations(self, unit_id: str) -> list[dict[str, Any]]:
        """参数为当前消费者编号；返回逻辑申请、复用和权威读取说明的独立投影。"""
        return deepcopy(self._associations.get(unit_id, []))

    def resolve_continuations(self, continuation_ids: Sequence[str], paper_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
        """参数为模型选择的续读身份与目标论文；仅返回同次真实回执提供的未读范围，拒绝猜测和跨版本引用。"""
        catalog = {entry["continuation_id"]: entry for receipt in self._receipts if receipt.read_report is not None
                   for entry in receipt.read_report.remaining if "continuation_id" in entry}
        if len(continuation_ids) != len(set(continuation_ids)) or any(key not in catalog for key in continuation_ids):
            raise ValueError("续读身份不属于本轮已返回的未读位置")
        selected = tuple(deepcopy(catalog[key]) for key in continuation_ids)
        if {entry["paper_id"] for entry in selected} != set(paper_ids):
            raise ValueError("续读身份与声明的目标论文不一致")
        if self.identity is None or any(
                self.identity.publication_snapshot.get("paper_batches", {}).get(entry["paper_id"]) != entry["batch_id"]
                for entry in selected):
            raise ValueError("续读身份不属于当前运行的发布版本")
        return selected

    def note_adapter_attempt(self, unit_id: str) -> None:
        """参数为真实工作所属单元；仅在适配器实际调用时累计外部调用次数。"""
        self._attempts[unit_id] = self._attempts.get(unit_id, 0) + 1

    def record_interrupted_read(self, unit_id: str, result: EvidenceReadResult) -> None:
        """参数为清理期间完成的真实读取；保存成果但禁止将超时/取消请求缓存成成功。"""
        fields = {**self._active_read_fields[unit_id], "receipt_id": str(uuid4()), "reuse_eligible": False}
        self._record_completed(result, fields=fields, actual_calls=1)
        self._interrupted_attempts[unit_id] = self._interrupted_attempts.get(unit_id, 0) + 1

    async def read(
        self, *, unit_id: str, source_type: str, request: Mapping[str, Any] | None,
        invoke: Callable[[], Awaitable[EvidenceReadResult]],
    ) -> EvidenceReadResult:
        """参数为已校验请求及真实调用；共享等价正文读取，返回材料与原完整性说明。"""
        normalized = deepcopy(dict(request)) if request is not None else None
        key = self._request_key(source_type, normalized)
        # 1. 【研究交付】【读取入口】权限与请求身份已核对，真实读取或合法复用都属于研究执行
        self.mark_execution_started()
        cached = self._cache.get(key) if key else None
        if cached is not None:
            return self._consume(unit_id, cached, reused=True)
        # 1. 【研究执行】【共享读取】失败和非冻结来源不参与成功缓存或共享任务
        pending_key = key or str(uuid4())
        shared = self._pending.get(pending_key)
        reused = shared is not None
        if shared is None:
            task = asyncio.create_task(self._read_owned(
                unit_id=unit_id, source_type=source_type, request=normalized,
                request_key=key, invoke=invoke,
            ))
            shared = _SharedRead(task=task)
            self._pending[pending_key] = shared
        shared.waiters += 1
        termination: asyncio.CancelledError | None = None
        try:
            receipt = await asyncio.shield(shared.task)
            return self._consume(unit_id, receipt, reused=reused)
        except asyncio.CancelledError as exc:
            termination = exc
            raise
        finally:
            # 2. 【研究执行】【共享读取】单个消费者离开不打断别人；最后一人负责等待清理
            shared.waiters -= 1
            if shared.waiters == 0:
                if not shared.task.done():
                    shared.task.cancel()
                settled = await settle_owned_work(shared.task)
                outcome, cancellation = settled
                self._pending.pop(pending_key, None)
                signal = termination or cancellation
                if signal is not None:
                    # 3. 【研究执行】【取消归因】消费者保留原取消，真实所有者的清理失败仍留在原因链
                    cause = outcome.__cause__ or outcome if isinstance(outcome, BaseException) else None
                    if cause is not None and cause is not signal:
                        raise signal from cause
                    raise signal

    async def settle(self) -> None:
        """无参数；整次研究终止时取消所有归属读取，并等待真实清理完成。"""
        work = list(self._pending.values())
        for shared in work:
            shared.task.cancel()
        cancellation: asyncio.CancelledError | None = None
        for shared in work:
            _, signal = await settle_owned_work(shared.task)
            cancellation = cancellation or signal
        self._pending.clear()
        if cancellation is not None:
            raise cancellation

    async def _read_owned(
        self, *, unit_id: str, source_type: str, request: dict[str, Any] | None,
        request_key: str, invoke: Callable[[], Awaitable[EvidenceReadResult]],
    ) -> ReadReceipt:
        """参数为真实请求及调用所有者；材料校验与凭据登记在推理开始前同步完成。"""
        previous_attempts = self._attempts.get(unit_id, 0)
        previous_interrupted = self._interrupted_attempts.get(unit_id, 0)
        fields = dict(identity=self.identity, receipt_id=str(uuid4()),
                      source_execution_id=self.execution_id, source_unit_id=unit_id,
                      source_type=source_type, request=request, request_key=request_key)
        self._active_read_fields[unit_id] = fields
        try:
            result = await invoke()
            return self._record_completed(result, fields=fields,
                                          actual_calls=self._unrecorded_attempts(unit_id, previous_attempts, previous_interrupted))
        except BaseException as exc:
            receipt = ReadReceipt(
                **fields, outcome="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                actual_call_count=self._unrecorded_attempts(unit_id, previous_attempts, previous_interrupted),
                error_type=type(exc).__name__, reason=str(exc),
            )
            self._receipts.append(receipt)
            raise
        finally:
            self._active_read_fields.pop(unit_id, None)

    def _unrecorded_attempts(self, unit_id: str, previous: int, previous_interrupted: int) -> int:
        """参数为单元及本次调用起点；已经单独保存的中断读取不重复计入真实调用数。"""
        return self._attempts.get(unit_id, 0) - previous - self._interrupted_attempts.get(unit_id, 0) + previous_interrupted

    def _record_completed(
        self, result: EvidenceReadResult, *, fields: dict[str, Any], actual_calls: int,
    ) -> ReadReceipt:
        """参数为实际完成结果和请求身份；同步登记一致快照，取消后完成也保存真实材料。"""
        self._validate_material_binding(result.records)
        registry = register_evidence_records(self._registry, deepcopy(result.records))
        report = result.read_report
        receipt = ReadReceipt(
            **fields, evidence_ids=tuple(str(record["evidence_id"]) for record in result.records),
            read_report=ReadReportSnapshot.model_validate(report.to_dict()), actual_call_count=actual_calls,
            outcome="unavailable" if report.status == "unavailable" else ("success" if result.records else "empty"),
        )
        # 1. 【研究执行】【成果登记】此处没有 await，材料与凭据不会被取消拆成半份成功
        self._registry = registry
        self._receipts.append(receipt)
        self._cache_receipt(receipt)
        return receipt

    def _request_key(self, source_type: str, request: dict[str, Any] | None) -> str:
        """参数为来源及已核验有效请求；仅固定正文生成可以共享的请求身份。"""
        if self.identity is None or request is None or source_type not in _BODY_SOURCES:
            return ""
        return content_digest({"identity": self.identity.model_dump(mode="json"),
                               "source_type": source_type, "request": request})

    def _cache_receipt(self, receipt: ReadReceipt) -> None:
        """参数为真实凭据；只有同请求成功或空正文读取可以免除下一次 IO。"""
        if (receipt.request_key and receipt.reuse_eligible and receipt.source_type in _BODY_SOURCES
                and receipt.outcome in {"success", "empty"}
                and receipt.read_report is not None
                and receipt.read_report.status in {"complete", "partial"}):
            expected = self._request_key(receipt.source_type, receipt.request)
            if expected != receipt.request_key:
                raise ValueError("读取凭据请求内容与身份摘要不一致")
            self._cache[receipt.request_key] = receipt

    def _consume(self, unit_id: str, receipt: ReadReceipt, *, reused: bool) -> EvidenceReadResult:
        """参数为消费者、凭据与复用标志；产生任务关联，不更改原始查询或历史材料。"""
        if receipt.read_report is None:
            raise ValueError("失败读取没有可消费的成功凭据")
        self._associations.setdefault(unit_id, []).append({
            "receipt_id": receipt.receipt_id, "source_unit_id": receipt.source_unit_id,
            "source_execution_id": receipt.source_execution_id, "reused": reused,
            "actual_call_count": 0 if reused else receipt.actual_call_count,
            "read_report": receipt.read_report.to_report().to_dict(),
        })
        records = tuple(cast(EvidenceRecord, deepcopy(self._registry[key])) for key in receipt.evidence_ids)
        return EvidenceReadResult(records, receipt.read_report.to_report())

    def _validate_material_binding(self, records: Sequence[Mapping[str, Any]]) -> None:
        """参数为待接纳材料；有运行身份时，正文所有来源必须属于同一授权发布快照。"""
        if self.identity is None:
            return
        batches = self.identity.publication_snapshot.get("paper_batches", {})
        authorized = set(self.identity.scope.get("paper_ids", ()))
        for record in records:
            if record.get("layer") != "direct_text":
                continue
            # 1. 【研究执行】【快照接续】历史正文只能只读回放，缺来源承诺不能支撑当前运行的成功凭据
            version = record.get("read_contract_version")
            if type(version) is not int or version != READ_CONTRACT_VERSION:
                raise ValueError("活跃执行材料必须包含当前正文读取合同，历史材料需要重新核验")
            paper_id = record.get("paper_id")
            if paper_id not in authorized:
                raise ValueError("读取材料不属于本次运行的授权论文范围")
            spans = record.get("source_spans", ())
            if any(span.get("batch_id") != batches.get(paper_id) for span in spans):
                raise ValueError("读取材料与本次运行的固定 publication 快照不一致")


def unit_business_keys(plan: ResearchPlan) -> dict[str, str]:
    """参数为当前已校验计划；返回包含依赖合同和要求的单元业务身份，编号不参与判等。"""
    pending = {unit.unit_id: unit for unit in plan.units}
    keys: dict[str, str] = {}
    facets = {facet.facet_id: facet for facet in [*plan.coverage_review.explicit_facets,
                                               *plan.coverage_review.inferred_facets]}
    while pending:
        ready = [unit for unit in pending.values() if all(dep.unit_id in keys for dep in unit.dependencies)]
        if not ready:
            raise ValueError("结果复用无法解析依赖业务身份")
        for unit in ready:
            payload = unit.model_dump(mode="json", exclude={"unit_id", "dependencies"})
            payload["dependencies"] = [
                {**dep.model_dump(mode="json", exclude={"unit_id"}), "unit_key": keys[dep.unit_id]}
                for dep in unit.dependencies
            ]
            payload["requirements"] = [facets[key].model_dump(mode="json") for key in sorted(unit.covered_facet_ids)
                                       if key in facets]
            payload["answer_sections"] = [section.model_dump(mode="json") for section in plan.coverage_review.answer_sections
                                          if section.section_id in unit.expected_output.answer_section_ids
                                          or set(section.facet_ids) & set(unit.covered_facet_ids)]
            keys[unit.unit_id] = content_digest(payload)
            pending.pop(unit.unit_id)
    return keys


def result_input_signature(
    unit: ResearchUnit, *, business_keys: Mapping[str, str], records: list[dict[str, Any]],
    read_reports: list[dict[str, Any]], dependency_results: list[ResearchUnitResult],
    reasoning_context: Mapping[str, Any], exposure: ConsumerExposure | None = None,
) -> str:
    """参数为实际推理输入与配置；完整匹配才允许免推理，来源编号按业务身份比较。"""
    if not reasoning_context or unit.unit_id not in business_keys:
        return ""
    if exposure is None or not exposure.identity_normalized_rendered_hash:
        return ""
    claim_keys = _canonical_claim_keys(dependency_results, business_keys)
    if len(claim_keys) != sum(len(result.claims) for result in dependency_results):
        return ""
    dependencies = []
    wanted = {dep.unit_id for dep in unit.dependencies}
    for result in dependency_results:
        if result.unit_id not in wanted:
            continue
        claims = []
        for claim in result.claims:
            if any(source not in business_keys for source in claim.source_unit_ids):
                return ""
            if any(source not in claim_keys for source in claim.source_claim_ids):
                return ""
            claims.append({**claim.model_dump(mode="json", exclude={"claim_id", "source_unit_ids", "source_claim_ids"}),
                           "source_units": [business_keys[source] for source in claim.source_unit_ids],
                           "source_claims": [claim_keys[source] for source in claim.source_claim_ids]})
        dependencies.append({"unit_key": business_keys[result.unit_id], "status": result.status,
                             "summary": result.summary, "evidence_ids": result.evidence_ids, "claims": claims,
                             "read_reports": result.metadata.get("read_reports", [])})
    return content_digest({"unit_key": business_keys[unit.unit_id], "records": records,
                           "read_reports": read_reports, "dependencies": dependencies,
                           "reasoning_context": dict(reasoning_context),
                           "exposure": _normalized_exposure(exposure, claim_keys)})


def _normalized_exposure(exposure: ConsumerExposure | None, claim_keys: Mapping[str, str]) -> dict[str, Any] | None:
    """参数为实际投影及可证明主张身份；仅为判等规范编号，原渲染哈希和投影不修改。"""
    if exposure is None:
        return None
    payload = exposure.model_dump(mode="json", exclude={"claims", "rendered_hash"})
    payload["claims"] = [{"claim_key": claim_keys[item.claim_id]} for item in exposure.claims]
    return payload


def reuse_current_result(
    unit: ResearchUnit, *, signature: str, business_keys: Mapping[str, str],
    previous_results: Sequence[ResearchUnitResult], current_results: Sequence[ResearchUnitResult] = (),
) -> ResearchUnitResult | None:
    """参数为相同输入和两代有效成果；仅唯一内容身份能重绑，原曝光范围和哈希不改写。"""
    if not signature:
        return None
    candidates = [result for result in previous_results if result.status in {"evidence_found", "inferred"}
                  and result.metadata.get("result_input_signature") == signature]
    for candidate in candidates:
        bindings = candidate.metadata.get("unit_business_keys")
        if not isinstance(bindings, dict) or not candidate.metadata.get("provenance", {}).get("exposure"):
            continue
        mapping = {old: matches[0] for old, key in bindings.items()
                   if len(matches := [current for current, value in business_keys.items() if value == key]) == 1}
        mapping[candidate.unit_id] = unit.unit_id
        claim_mapping = _claim_rebindings(previous_results, current_results, mapping)
        if any(source not in claim_mapping for claim in candidate.claims for source in claim.source_claim_ids):
            continue
        if any(source not in mapping for claim in candidate.claims for source in claim.source_unit_ids):
            continue
        claims = [claim.model_copy(update={
            "claim_id": f"{unit.unit_id}.c{index}",
            "source_unit_ids": [mapping[source] for source in claim.source_unit_ids],
            "source_claim_ids": [claim_mapping[source] for source in claim.source_claim_ids],
        }) for index, claim in enumerate(candidate.claims, 1)]
        rebindings = {new: old for old, new in claim_mapping.items()}
        rebindings.update({new.claim_id: old.claim_id for new, old in zip(claims, candidate.claims)})
        provenance = {**candidate.metadata["provenance"], "unit": unit.model_dump(mode="json")}
        prior = candidate.metadata.get("claim_id_rebindings", {})
        metadata = {**candidate.metadata, "result_reused": True, "source_result_unit_id": candidate.unit_id,
                    "claim_id_rebindings": {new: prior.get(old, old) for new, old in rebindings.items()},
                    "unit_id_rebindings": mapping, "unit_business_keys": dict(business_keys), "provenance": provenance}
        return ResearchUnitResult.model_validate({**candidate.model_dump(), "unit_id": unit.unit_id,
            "claims": [claim.model_dump() for claim in claims], "metadata": metadata}, strict=True)
    return None


def _claim_rebindings(
    previous: Sequence[ResearchUnitResult], current: Sequence[ResearchUnitResult], units: Mapping[str, str],
) -> dict[str, str]:
    """参数为两代实际成果和唯一单元映射；返回内容唯一一致的主张映射，不用列表位置猜。"""
    previous_keys = _canonical_claim_keys(previous, units)
    current_keys = _canonical_claim_keys(current, {result.unit_id: result.unit_id for result in current})
    mapping: dict[str, str] = {}
    for claim_id, key in previous_keys.items():
        matches = [candidate for candidate, value in current_keys.items() if value == key]
        if len(matches) == 1:
            mapping[claim_id] = matches[0]
    return mapping


def _canonical_claim_keys(results: Sequence[ResearchUnitResult], units: Mapping[str, str]) -> dict[str, str]:
    """参数为成果及稳定单元身份；返回包含每条实际支持边的主张内容身份，循环不判等。"""
    pending = {claim.claim_id: (result.unit_id, claim) for result in results for claim in result.claims}
    keys: dict[str, str] = {}
    while pending:
        ready = [key for key, (_owner, claim) in pending.items() if all(source in keys for source in claim.source_claim_ids)]
        if not ready:
            return {}
        for key in ready:
            owner, claim = pending.pop(key)
            content = claim.model_dump(mode="json", exclude={"claim_id", "source_unit_ids", "source_claim_ids"})
            content["sources"] = [keys[source] for source in claim.source_claim_ids]
            keys[key] = content_digest({"owner": units.get(owner), "claim": content})
    return keys
