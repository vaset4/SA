"""评测案例、运行观察与派生评分的版本化边界。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CONTRACT_VERSION: Literal["eval-contract-v1"] = "eval-contract-v1"
PROJECTION_VERSION: Literal["eval-projection-v1"] = "eval-projection-v1"
TRUTH_FIELDS = (
    "golden_evidence", "critical_facts", "rubric", "expected", "reference_answer",
    "evidence_benchmark",
)
AGENT_TRUTH_FIELDS = (
    "required_outcomes", "forbidden_behaviors", "allowed_terminal_states",
    "task_completion_definition", "fault_injection", "evidence_refs", "evidence_targets",
)
PRIVATE_INPUT_FIELDS = frozenset((
    *TRUTH_FIELDS, *AGENT_TRUTH_FIELDS, "reference", "fixture", "replay", "eval_case", "grading_result",
))
INPUT_FIELDS = (
    "question", "structured_query", "filter_expr", "paper_ids", "paper_titles",
    "project_id", "project_scope", "library_id", "user_id", "dialogue_mode", "chat_history",
)
OBSERVATION_FIELDS = (
    "retrieved_contexts", "response", "evidence_records", "failure_reason", "trace",
    "planner_trace", "turn_outcome", "source_status_reasons", "search_results",
    "research_plan", "research_unit_results", "planner_final_blocks", "planner_repair_trace",
    "planner_final_validation_errors", "final_delivery_report", "final_citation_bindings",
    "final_generation_attempt", "final_delivery_applicability",
    "depth_transitions", "turn_depth", "memory_read_result", "memory_writeback",
    "trace_summary", "_rag_trace_dict", "usage", "cost", "latency", "context_origin",
    "evidence_observations", "model_call_observations", "evidence_capture_errors",
)
RUNTIME_METADATA_FIELDS = frozenset((
    "adapter", "requested_mode", "requested_graph_mode", "requested_dialogue_mode",
    "resolved_mode", "resolved_graph_mode", "retrieval_mode", "graph_mode", "source_status",
    "answer_confidence", "citations", "citations_count", "chunk_citations_count", "dialogue_mode",
    "retrieval_reasoning", "paper_source_ids", "paper_sources_count", "inventory_contexts_count",
    "synthesis_state", "synthesis_status", "evidence_layers", "project_scope", "paper_focus",
    "supported_dimensions", "unsupported_dimensions", "open_questions", "synthesis_citations_count",
    "research_objects_count", "note_actions_count", "paper_ids", "paper_titles", "project_id",
    "filter_expr", "structured_query", "resolved_scope", "scope_resolution_warnings",
    "requires_citation", "requires_citations", "citation_required", "retrieval_policy",
    "evidence_records", "source_status_reasons", "failure_reason", "context_origin",
    "paper_id", "paper_title", "user_id", "library_id",
))
MetricStatus = Literal["measured", "not_measured", "missing", "invalid_input"]
RunMode = Literal[
    "replay", "live", "planner_replay", "planner_live_fixture",
    "agent_fixture", "agent_live", "oracle", "retrieval_live",
]


def snapshot(payload: Mapping[str, Any]) -> str:
    """参数为可序列化对象；返回确定性 JSON 快照，不保留可变对象引用。"""
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_snapshot(value: str) -> dict[str, Any]:
    """参数为 JSON 快照；返回独立对象，格式错误显式抛出异常。"""
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("eval snapshot must contain an object")
    return payload


def snapshot_hash(value: str) -> str:
    """参数为冻结文本；返回内容校验和，不能用此值替代原文快照。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EvalCase(BaseModel):
    """冻结问题和真值；input/truth 快照读取后返回新对象。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["eval-contract-v1"] = CONTRACT_VERSION
    case_id: str
    case_version: str
    source_type: str
    input_snapshot: str
    truth_snapshot: str

    @field_validator("input_snapshot", "truth_snapshot")
    @classmethod
    def validate_snapshot(cls, value: str) -> str:
        """参数为序列化字段；验证为对象后返回规范快照。"""
        return snapshot(read_snapshot(value))


class RunRecord(BaseModel):
    """冻结一次执行的输入、观察及声明的 fixture 环境。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["eval-contract-v1"] = CONTRACT_VERSION
    run_id: str
    case_id: str
    case_version: str
    run_mode: RunMode
    input_snapshot: str
    observation_snapshot: str
    fixture_snapshot: str | None = None
    provenance: Literal["legacy", "captured"]
    missing_fields: tuple[str, ...] = ()

    @field_validator("input_snapshot", "observation_snapshot", "fixture_snapshot")
    @classmethod
    def validate_snapshot(cls, value: str | None) -> str | None:
        """参数为输入/观察/环境快照；返回规范 JSON 或未使用的空环境。"""
        return snapshot(read_snapshot(value)) if value is not None else None


class MetricResult(BaseModel):
    """保存指标值、适用性、真实计数与缺失原因。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    status: MetricStatus
    reason: str
    value: float | None = None
    numerator: float | None = None
    denominator: float | None = None

    @model_validator(mode="after")
    def validate_measurement(self) -> MetricResult:
        """无额外参数；拒绝非有限值、假分母及未测量状态携带分数。"""
        numbers = (self.value, self.numerator, self.denominator)
        if any(number is not None and not math.isfinite(number) for number in numbers):
            raise ValueError("metric numbers must be finite")
        if self.status == "measured" and self.value is None:
            raise ValueError("measured metric requires a value")
        if self.status != "measured" and any(number is not None for number in numbers):
            raise ValueError("unmeasured metric cannot carry a score")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("metric counts must be supplied together")
        if self.denominator is not None and self.denominator <= 0:
            raise ValueError("measured denominator must be positive")
        return self


class GradingResult(BaseModel):
    """保存派生评分身份、投影版本及可审计的逐指标结果。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["eval-contract-v1"] = CONTRACT_VERSION
    case_id: str
    run_id: str
    grader: str
    grader_version: str
    projection_version: Literal["eval-projection-v1"] = PROJECTION_VERSION
    projection_hash: str
    measurements: tuple[MetricResult, ...]


def validate_execution_input(payload: Any, *, path: str = "input") -> None:
    """参数为被测入口输入及定位路径；嵌套真值字段存在时显式拒绝。"""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in PRIVATE_INPUT_FIELDS:
                raise ValueError(f"evaluation truth is not allowed in execution input: {path}.{key}")
            validate_execution_input(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            validate_execution_input(value, path=f"{path}[{index}]")


def runtime_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """参数为旧/新元信息；只复制已知运行字段，真值和标注不进入观察。"""
    return read_snapshot(snapshot({key: value for key, value in metadata.items() if key in RUNTIME_METADATA_FIELDS}))


def create_case(record: dict[str, Any]) -> EvalCase:
    """参数为旧样本；分离问题与真值，缺少版本的案例显式标记 legacy。"""
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    truth = {key: record.get(key, metadata.get(key)) for key in TRUTH_FIELDS if key in record or key in metadata}
    inputs = {key: record.get(key, metadata.get(key)) for key in INPUT_FIELDS if key in record or key in metadata}
    validate_execution_input(inputs)
    identity = snapshot({"input": inputs, "truth": truth})
    return EvalCase(
        case_id=str(record.get("id") or record.get("case_id") or f"legacy-{snapshot_hash(identity)}"),
        case_version=str(record.get("case_version") or record.get("dataset_version") or "legacy"),
        source_type=str(record.get("source_type") or metadata.get("source_type") or "legacy_unknown"),
        input_snapshot=snapshot(inputs), truth_snapshot=snapshot(truth),
    )


def required_observation_fields(mode: str) -> tuple[str, ...]:
    """参数为执行模式；返回该入口确实应该产生的原始观察字段。"""
    if mode == "retrieval_live":
        return ("evidence_observations", "model_call_observations")
    if mode.startswith("planner"):
        return ("research_unit_results", "planner_trace")
    if mode == "live":
        return ("retrieved_contexts", "response", "planner_trace", "turn_outcome", "evidence_records")
    return ("retrieved_contexts", "response")


def freeze_record(
    record: dict[str, Any], *,
    mode: RunMode = "replay",
    execution_input: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None, fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """参数为案例和实际执行载体；返回带冻结合同的新记录，已有合同只验证不重建。"""
    if "eval_case" in record or "run_record" in record:
        case, run = record_contracts(record)
        return {**record, "eval_case": case.model_dump(), "run_record": run.model_dump()}
    case = create_case(record)
    inputs = execution_input if execution_input is not None else read_snapshot(case.input_snapshot)
    validate_execution_input(inputs)
    observed = observation if observation is not None else _legacy_observation(record, mode=mode)
    required = required_observation_fields(mode)
    missing = tuple(key for key in required if key not in observed)
    if execution_input is None:
        missing += ("captured_execution_input",)
    run = RunRecord(
        run_id=str(uuid4()), case_id=case.case_id, case_version=case.case_version,
        run_mode=mode, input_snapshot=snapshot(inputs), observation_snapshot=snapshot(observed),
        fixture_snapshot=snapshot(fixture) if fixture is not None else None,
        provenance="captured" if execution_input is not None else "legacy", missing_fields=missing,
    )
    return {**record, "eval_case": case.model_dump(), "run_record": run.model_dump()}


def _legacy_observation(record: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """参数为旧记录及模式；提取声明的观察载体，不用 truth 填补结果。"""
    if mode.startswith("planner"):
        replay = record.get("replay")
        if not isinstance(replay, dict):
            raise ValueError("planner observation must be an object")
        return read_snapshot(snapshot(replay))
    observation = {key: record[key] for key in OBSERVATION_FIELDS if key in record}
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    observation["metadata"] = runtime_metadata(metadata)
    return observation


def record_contracts(record: dict[str, Any]) -> tuple[EvalCase, RunRecord]:
    """参数为携带冻结合同的记录；校验案例/运行身份后返回模型。"""
    case = EvalCase.model_validate(record["eval_case"])
    run = RunRecord.model_validate(record["run_record"])
    if (case.case_id, case.case_version) != (run.case_id, run.case_version):
        raise ValueError("evaluation case/run identity mismatch")
    return case, run


def grading_projection(case: EvalCase, run: RunRecord) -> dict[str, Any]:
    """参数为冻结案例和运行；重建独立的真值/观察投影，调用方修改不影响快照。"""
    if (case.case_id, case.case_version) != (run.case_id, run.case_version):
        raise ValueError("evaluation case/run identity mismatch")
    return {
        "projection_version": PROJECTION_VERSION,
        "case_id": case.case_id, "case_version": case.case_version,
        "input": read_snapshot(run.input_snapshot), "truth": read_snapshot(case.truth_snapshot),
        "observation": read_snapshot(run.observation_snapshot),
    }


def project_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    """参数为冻结记录；为现有评分器生成一次性旧格式投影，保留独立评分边界。"""
    case, run = record_contracts(record)
    projection = grading_projection(case, run)
    inputs = read_snapshot(case.input_snapshot)
    if run.run_mode.startswith("planner"):
        return {**inputs, **projection["truth"], "replay": projection["observation"]}
    return {**inputs, **projection["truth"], **projection["observation"]}


def build_grading(
    record: dict[str, Any], measurements: list[MetricResult], *,
    grader: str, grader_version: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    """参数为冻结记录、测量及评分器身份/版本；返回含原投影校验和的独立判断。"""
    case, run = record_contracts(record)
    return GradingResult(
        case_id=case.case_id, run_id=run.run_id, grader=grader, grader_version=grader_version,
        projection_hash=snapshot_hash(snapshot(grading_projection(case, run))), measurements=tuple(measurements),
    ).model_dump(mode="json")


def measurement_coverage(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """参数为含 grading_result 的逐题记录；按指标统计真实测量与各类缺口。"""
    counters: dict[str, Counter[str]] = {}
    for record in records:
        for metric in record["grading_result"]["measurements"]:
            counts = counters.setdefault(metric["name"], Counter())
            counts[f"{metric['status']}_count"] += 1
    statuses = ("measured", "not_measured", "missing", "invalid_input")
    return {name: {f"{status}_count": counts[f"{status}_count"] for status in statuses} for name, counts in counters.items()}


def measurement_lines(coverage: dict[str, dict[str, int]]) -> list[str]:
    """参数为测量覆盖；返回 Markdown 行，明确显示未测量的原因类别。"""
    return ["", "## Measurement Coverage", "", *[
        f"- {name}: " + ", ".join(f"{key}={value}" for key, value in counts.items())
        for name, counts in sorted(coverage.items())
    ]]


def merge_measurement_coverage(coverages: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    """参数为各数据集的指标覆盖；返回按真实测量状态累加的新统计。"""
    totals: dict[str, Counter[str]] = {}
    for coverage in coverages:
        for name, counts in coverage.items():
            totals.setdefault(name, Counter()).update(counts)
    return {name: dict(counts) for name, counts in totals.items()}


def rebuild_grading_input(record: dict[str, Any]) -> dict[str, Any]:
    """参数为保存的三层报告记录；重建并校验原始评分投影，篡改时抛错。"""
    case, run = record_contracts(record)
    grading = GradingResult.model_validate(record["grading_result"])
    if grading.case_id != case.case_id or grading.run_id != run.run_id:
        raise ValueError("grading case/run identity mismatch")
    projection = grading_projection(case, run)
    if snapshot_hash(snapshot(projection)) != grading.projection_hash:
        raise ValueError("grading projection checksum mismatch")
    return projection
