"""固定评测输入、执行槽位和真实产物的不可变合同。作者：xxx。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, SerializerFunctionWrapHandler, ValidationInfo, field_validator, model_serializer, model_validator

from evals.contracts import read_snapshot, snapshot, snapshot_hash
from evals.evidence.contracts import IndexSnapshot, SourceSnapshot

from .contracts import ChangeFactor, CheckName, ComparisonRequest, Digest, FrozenModel, RunSlot, Text
from .policy import MeasurementPolicy

JobKind = Literal["retrieval", "agent", "planner_fixture"]
EXECUTION_VERSION = "eval-execution-v1"
DEFAULT_WORKER_TIMEOUT_SECONDS = 3600.0


def _unique(values: tuple[object, ...], label: str) -> None:
    """参数为身份集合和名称；重复时拒绝准备，避免同一次运行填多个槽位。"""
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")


class PaperBinding(FrozenModel):
    """将固定候选键与已存在的业务论文和实际 PDF 内容身份绑定。"""

    paper_key: Text
    paper_id: Text
    pdf_sha256: Digest


class ExecutionJob(FrozenModel):
    """事前声明执行入口、题目、重复次数及适用检查。"""

    job_id: Text
    kind: JobKind
    dataset_path: Text
    case_ids: tuple[Text, ...] = Field(min_length=1)
    repetitions: int = Field(ge=1)
    required_checks: tuple[CheckName, ...]
    retrieval_method: Literal["dense", "hybrid"] | None = None

    @model_serializer(mode="wrap")
    def serialize_method(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """参数为序列化器；旧任务省略方法字段，返回保持原身份的内容。"""
        payload = handler(self)
        if self.retrieval_method is None:
            payload.pop("retrieval_method", None)
        return payload

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        """无额外参数；检查题目及检查名唯一，返回自身。"""
        _unique(self.case_ids, "job case")
        _unique(self.required_checks, "job check")
        if self.retrieval_method is not None and self.kind != "retrieval":
            raise ValueError("retrieval_method_requires_retrieval_job")
        return self


class ExecutionProfile(FrozenModel):
    """保存真实业务范围和固定案例选择；密钥仍由原应用环境提供。"""

    schema_version: Literal["eval-execution-v1"] = "eval-execution-v1"
    suite_id: Text
    dataset_version: Text
    purpose: Literal["capability", "regression"]
    user_id: Text
    project_id: Text | None
    uploads_root: Text
    paper_root: Text
    paper_bindings: tuple[PaperBinding, ...] = Field(min_length=1)
    retrieval_top_k: int = Field(ge=1)
    jobs: tuple[ExecutionJob, ...] = Field(min_length=1)
    worker_timeout_seconds: float = Field(default=DEFAULT_WORKER_TIMEOUT_SECONDS, gt=0)
    measurement_policy: MeasurementPolicy | None = None
    grading_timing: Literal["per-slot", "deferred"] | None = None
    repetition_order: Literal["case-first", "round-first"] | None = None

    @model_serializer(mode="wrap")
    def serialize_policy(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """参数为序列化器；历史配置不补新字段，保持原 prepared 内容校验和。"""
        payload = handler(self)
        if self.measurement_policy is None:
            payload.pop("measurement_policy", None)
        for name in ("grading_timing", "repetition_order"):
            if getattr(self, name) is None:
                payload.pop(name, None)
        return payload

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        """无额外参数；拒绝重复绑定、任务及尚未配置的身份占位符，返回自身。"""
        _unique(tuple(item.paper_key for item in self.paper_bindings), "paper key")
        _unique(tuple(item.paper_id for item in self.paper_bindings), "paper binding")
        _unique(tuple(item.pdf_sha256 for item in self.paper_bindings), "PDF binding")
        _unique(tuple(item.job_id for item in self.jobs), "execution job")
        if self.grading_timing is not None and self.measurement_policy is None:
            raise ValueError("explicit_grading_timing_requires_measurement_policy")
        if "${" in self.user_id or any("${" in item.paper_id for item in self.paper_bindings):
            raise ValueError("execution profile contains unresolved business identities")
        return self


class PreparedCase(FrozenModel):
    """保存已绑定的新案例副本，不回写候选包。"""

    case_id: Text
    case_version: Text
    record_snapshot: Text

    @field_validator("record_snapshot")
    @classmethod
    def validate_record(cls, value: str) -> str:
        """参数为案例 JSON；返回规范快照，拒绝非对象。"""
        return snapshot(read_snapshot(value))


class PreparedJob(FrozenModel):
    """执行前将每题重复数展开成固定槽位，运行器不得临时补题。"""

    job_id: Text
    kind: JobKind
    cases: tuple[PreparedCase, ...] = Field(min_length=1)
    slots: tuple[RunSlot, ...] = Field(min_length=1)
    retrieval_method: Literal["dense", "hybrid"] | None = None

    @model_serializer(mode="wrap")
    def serialize_method(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """参数为序列化器；历史准备包不补方法字段，返回可校验的原始内容。"""
        payload = handler(self)
        if self.retrieval_method is None:
            payload.pop("retrieval_method", None)
        return payload

    @model_validator(mode="after")
    def validate_slots(self) -> Self:
        """无额外参数；核对唯一案例及未执行槽位，返回自身。"""
        cases = {(case.case_id, case.case_version) for case in self.cases}
        if self.retrieval_method is not None and self.kind != "retrieval":
            raise ValueError("retrieval_method_requires_retrieval_job")
        _unique(tuple(case.case_id for case in self.cases), "prepared case")
        _unique(tuple((slot.case_id, slot.case_version, slot.repeat_id) for slot in self.slots), "prepared slot")
        if any((slot.case_id, slot.case_version) not in cases or slot.run_id is not None for slot in self.slots):
            raise ValueError("prepared slots must reference declared cases without a run")
        if cases != {(slot.case_id, slot.case_version) for slot in self.slots}:
            raise ValueError("prepared case has no execution slot")
        return self


class InputArtifact(FrozenModel):
    """记录准备时读过的实际来源文件，不靠任务目录维持运行数据。"""

    path: Text
    sha256: Digest


class PreparedSuite(FrozenModel):
    """由内容寻址绑定案例、原文、索引和固定初始状态。"""

    schema_version: Literal["eval-prepared-v1"] = "eval-prepared-v1"
    prepared_id: Digest
    profile: ExecutionProfile
    scope_snapshot: Text
    sources: tuple[SourceSnapshot, ...] = Field(min_length=1)
    index: IndexSnapshot
    initial_snapshot: Text
    mapping_issues_snapshot: Text = '{"issues":[]}'
    input_artifacts: tuple[InputArtifact, ...]
    jobs: tuple[PreparedJob, ...] = Field(min_length=1)

    @field_validator("scope_snapshot", "initial_snapshot", "mapping_issues_snapshot")
    @classmethod
    def validate_snapshot(cls, value: str) -> str:
        """参数为实际范围或固定仓储快照；返回规范对象文本。"""
        return snapshot(read_snapshot(value))

    @model_validator(mode="after")
    def validate_identity(self, info: ValidationInfo[dict[str, bool]]) -> Self:
        """参数为建模上下文；核对整个准备包内容和槽位唯一性，返回自身。"""
        _unique(tuple(job.job_id for job in self.jobs), "prepared job")
        declared_jobs = {job.job_id: job for job in self.profile.jobs}
        for job in self.jobs:
            declared = declared_jobs.get(job.job_id)
            method = declared.retrieval_method if declared else None
            if job.retrieval_method != method:
                raise ValueError("prepared_retrieval_method_differs_from_profile")
        slots = tuple((slot.case_id, slot.case_version, slot.repeat_id) for job in self.jobs for slot in job.slots)
        _unique(slots, "suite execution slot")
        body = snapshot(self.model_dump(mode="json", exclude={"prepared_id"}))
        if not (info.context or {}).get("building_identity") and snapshot_hash(body) != self.prepared_id:
            raise ValueError("prepared_suite_checksum_mismatch")
        return self


class ProducedRun(FrozenModel):
    """引用子进程实际生成的文件和状态，不携带期望分数。"""

    job_id: Text
    case_id: Text
    case_version: Text
    repeat_id: Text
    run_id: Text | None = None
    report_path: Text | None = None
    frozen_path: Text | None = None
    identity_path: Text | None = None
    status: Literal["completed", "failed", "not_run", "timed_out", "cancelled"]
    error: Text | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0)
    cleanup_snapshot: str | None = None
    grading_status: Literal["not_started", "completed", "failed"] = "not_started"
    grading_error: Text | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        """无额外参数；已完成必须具有实际产物，失败必须说明原因。"""
        if self.status == "completed" and not all((self.run_id, self.report_path, self.frozen_path, self.identity_path)):
            raise ValueError("completed execution requires actual artifacts")
        if self.status in {"failed", "timed_out", "cancelled"} and self.error is None:
            raise ValueError("failed execution requires an error")
        if self.grading_status == "failed" and self.grading_error is None:
            raise ValueError("failed grading requires an error")
        return self


class ExecutionRequest(FrozenModel):
    """本次新目录及可选基线；变化因素必须显式声明。"""

    output_dir: Path
    baseline: Path | None = None
    changed_factors: tuple[ChangeFactor, ...] = ()
    resume: bool = False

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        """无额外参数；拒绝重复变化因素及没有基线的变化声明。"""
        _unique(self.changed_factors, "changed factor")
        ComparisonRequest(changed_factors=self.changed_factors)
        if self.changed_factors and self.baseline is None:
            raise ValueError("changed_factors requires a baseline")
        return self
