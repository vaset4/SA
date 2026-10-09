"""冻结套件引用、评分切片、比较条件与报告合同。作者：xxx。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from evals.contracts import EvalCase, GradingResult, MetricResult, MetricStatus, RunRecord

SUITE_VERSION: Literal["eval-suite-v1"] = "eval-suite-v1"
SuitePurpose = Literal["capability", "regression", "online_discovery"]
ReportKind = Literal["planner", "evidence", "agent", "workflow"]
ChangeFactor = Literal["code", "model", "embedding", "prompt", "retrieval", "retrieval_method", "index"]
IdentityName = Literal[
    "dataset", "input", "truth", "source", "index", "code", "model", "embedding",
    "prompt", "retrieval", "evaluator", "environment",
]
CheckName = Literal[
    "scope", "forbidden_fact", "citation", "truth_leakage", "required_trace",
    "false_success", "stale_memory",
]
IDENTITY_NAMES: tuple[IdentityName, ...] = (
    "dataset", "input", "truth", "source", "index", "code", "model", "embedding",
    "prompt", "retrieval", "evaluator", "environment",
)
METRIC_STATUSES: tuple[MetricStatus, ...] = ("measured", "not_measured", "missing", "invalid_input")
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class FrozenModel(BaseModel):
    """所有套件对象构造后只读；额外字段和非有限数值显式拒绝。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class FrozenReportRef(FrozenModel):
    """原报告、冻结运行和可选历史身份文件的路径及内容校验和。"""

    source_id: Text
    kind: ReportKind
    report_region: Literal["root", "evidence_benchmark"] = "root"
    report_path: Text
    report_sha256: Digest
    frozen_path: Text | None = None
    frozen_sha256: Digest | None = None
    identity_path: Text | None = None
    identity_sha256: Digest | None = None
    ragas_path: Text | None = None
    ragas_sha256: Digest | None = None

    @model_validator(mode="after")
    def validate_references(self) -> FrozenReportRef:
        """无额外参数；拒绝未成对的引用及与来源不符的报告区域，返回自身。"""
        pairs = ((self.frozen_path, self.frozen_sha256), (self.identity_path, self.identity_sha256),
                 (self.ragas_path, self.ragas_sha256))
        if any((path is None) != (digest is None) for path, digest in pairs):
            raise ValueError("artifact path and checksum must be supplied together")
        if self.report_region != "root" and self.kind != "evidence":
            raise ValueError("evidence_benchmark region requires an evidence source")
        if self.kind in {"agent", "workflow"} and self.frozen_path is None:
            raise ValueError("agent report requires its frozen.jsonl artifact")
        if self.kind == "evidence" and self.report_region == "root" and self.frozen_path is None:
            raise ValueError("standalone evidence report requires its frozen.jsonl artifact")
        return self


class RunSlot(FrozenModel):
    """预先声明的案例重复槽位；空 run_id 表示尚未运行，而不是成功。"""

    case_id: Text
    case_version: Text
    repeat_id: Text
    run_id: Text | None
    required_checks: tuple[CheckName, ...]

    @model_validator(mode="after")
    def validate_checks(self) -> RunSlot:
        """无额外参数；拒绝重复检查名称，返回自身。"""
        if len(set(self.required_checks)) != len(self.required_checks):
            raise ValueError("duplicate required check")
        return self


class SuiteManifest(FrozenModel):
    """冻结套件用途、数据版本、原产物引用及每个案例的预期运行槽位。"""

    schema_version: Literal["eval-suite-v1"] = SUITE_VERSION
    suite_id: Text
    purpose: SuitePurpose
    dataset_version: Text
    sources: tuple[FrozenReportRef, ...] = Field(min_length=1)
    slots: tuple[RunSlot, ...] = Field(min_length=1)
    measurement_policy_snapshot: str | None = None
    execution_snapshot: str | None = None
    grading_timing: Literal["per-slot", "deferred"] | None = None
    grading_attempt_snapshot: str | None = None

    @model_validator(mode="after")
    def validate_membership(self) -> SuiteManifest:
        """无额外参数；拒绝重复来源、重复槽位及用同一次运行填多个槽位。"""
        source_ids = [source.source_id for source in self.sources]
        slot_ids = [(slot.case_id, slot.case_version, slot.repeat_id) for slot in self.slots]
        run_ids = [slot.run_id for slot in self.slots if slot.run_id is not None]
        duplicates = (len(source_ids) != len(set(source_ids)), len(slot_ids) != len(set(slot_ids)),
                      len(run_ids) != len(set(run_ids)))
        if any(duplicates):
            raise ValueError("duplicate source, slot or run identity")
        return self


class IdentityCondition(FrozenModel):
    """原运行的一项条件；缺失和不适用各自保留原因，不能携带替代值。"""

    name: IdentityName
    status: Literal["recorded", "missing", "not_applicable"]
    snapshot: Text | None = None
    reason: Text

    @model_validator(mode="after")
    def validate_value(self) -> IdentityCondition:
        """无额外参数；仅已记录条件可有身份值，返回自身。"""
        if (self.status == "recorded") != (self.snapshot is not None):
            raise ValueError("only recorded identity conditions carry a snapshot")
        return self


class RunConditions(FrozenModel):
    """一个原始 run_id 对应的已保存条件，不从当前进程配置补写。"""

    run_id: Text
    conditions: tuple[IdentityCondition, ...]

    @model_validator(mode="after")
    def validate_names(self) -> RunConditions:
        """无额外参数；拒绝同一条件的多个来源值，返回自身。"""
        names = [condition.name for condition in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError("duplicate identity condition")
        return self


class RecordedIdentity(FrozenModel):
    """绑定原报告及冻结记录的历史条件文件，不授权修改历史输入或评分。"""

    schema_version: Literal["eval-suite-identity-v1"] = "eval-suite-identity-v1"
    report_sha256: Digest
    frozen_sha256: Digest | None = None
    runs: tuple[RunConditions, ...]

    @model_validator(mode="after")
    def validate_runs(self) -> RecordedIdentity:
        """无额外参数；拒绝重复的运行身份记录，返回自身。"""
        if len({run.run_id for run in self.runs}) != len(self.runs):
            raise ValueError("duplicate run in recorded identity")
        return self


class ImportedRun(FrozenModel):
    """复用原案例与运行合同；JSON 快照保留来源元信息及实际观测。"""

    case: EvalCase
    run: RunRecord
    record_snapshot: str
    review_status: Literal["candidate", "reviewed", "rejected", "unlabeled"]
    topology_snapshot: str
    conditions: tuple[IdentityCondition, ...]


class ScoreSlice(FrozenModel):
    """一个原评分在报告中的阶段位置；不会产生新的主张或测量值。"""

    source_id: Text
    kind: ReportKind
    run_id: Text
    stage: Text
    call_id: str | None = None
    backend: str | None = None
    grading: GradingResult
    detail_snapshot: str


class SuiteInput(FrozenModel):
    """已校验的来源和评分，全部使用不可变合同或 JSON 字符串。"""

    manifest: SuiteManifest
    runs: tuple[ImportedRun, ...]
    scores: tuple[ScoreSlice, ...]


class StateCount(FrozenModel):
    """一个指标状态对应的真实观测数量。"""

    status: MetricStatus
    count: int


class RunMeasurement(FrozenModel):
    """同一重复运行中的原子测量；缺少槽位或调用时保留空值和原因。"""

    repeat_id: str
    run_id: str | None
    measurements: tuple[MetricResult, ...]
    value: float | None
    status: MetricStatus
    reason: str


class CaseMeasurement(FrozenModel):
    """先按运行后按题汇总，检索调用次数不会放大题目权重。"""

    case_id: str
    case_version: str
    value: float | None
    complete: bool
    runs: tuple[RunMeasurement, ...]


class MetricGroup(FrozenModel):
    """同用途、来源、审核状态和评分语义的题级汇总。"""

    key_snapshot: str
    metric: str
    mean: float | None
    complete_case_count: int
    states: tuple[StateCount, ...]
    cases: tuple[CaseMeasurement, ...]


class RepetitionResult(FrozenModel):
    """每题真实任务完成结果；至少一次成功和全部成功分别表达。"""

    key_snapshot: str
    case_id: str
    case_version: str
    expected_runs: int
    measured_runs: int
    successful_runs: int
    single_run_success_rate: float | None
    any_success: bool | None
    all_success: bool | None
    runs: tuple[RunMeasurement, ...]
    metric: str = "task_success"


class CheckOutcome(FrozenModel):
    """套件声明适用的 P0 逐项检查，缺少原始判断不能视为通过。"""

    case_id: str
    repeat_id: str
    run_id: str | None
    name: CheckName
    status: Literal["passed", "failed", "missing", "not_applicable"]
    reason: str


class SuiteSummary(FrozenModel):
    """保留原输入的派生汇总；比较阶段仍可核对每题运行身份。"""

    inputs: SuiteInput
    groups: tuple[MetricGroup, ...]
    repetitions: tuple[RepetitionResult, ...]
    checks: tuple[CheckOutcome, ...]


class ComparisonRequest(FrozenModel):
    """比较前显式声明可变化的因素；空元组表示没有允许的条件变化。"""

    changed_factors: tuple[ChangeFactor, ...]

    @model_validator(mode="after")
    def validate_factors(self) -> ComparisonRequest:
        """无额外参数；拒绝重复变化项，返回自身。"""
        if len(set(self.changed_factors)) != len(self.changed_factors):
            raise ValueError("duplicate changed factor")
        if "retrieval_method" in self.changed_factors and len(self.changed_factors) != 1:
            raise ValueError("retrieval_method_comparison_requires_single_experimental_factor")
        return self


class ComparisonDecision(FrozenModel):
    """记录运行条件的比较资格和逐项不满足原因。"""

    status: Literal["comparable", "incomparable"]
    reasons: tuple[str, ...]


class MetricDelta(FrozenModel):
    """同一指标在同题分母下的差值；不可比指标没有数值差。"""

    key_snapshot: str
    metric: str
    status: Literal["comparable", "incomparable"]
    reasons: tuple[str, ...]
    current: float | None = None
    baseline: float | None = None
    delta: float | None = None
    pairs_snapshot: str


class ComparisonResult(ComparisonDecision):
    """保留声明的变化项、指标差值和不可比原因。"""

    request: ComparisonRequest
    deltas: tuple[MetricDelta, ...]


class SuiteReport(FrozenModel):
    """可独立重放的套件报告，保留原合同及全部派生结果。"""

    schema_version: Literal["eval-suite-v1"] = SUITE_VERSION
    summary: SuiteSummary
    comparison: ComparisonResult | None = None
