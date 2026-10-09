"""Claim/Citation 校准合同。作者：xxx。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evals.contracts import MetricResult, PROJECTION_VERSION, snapshot, read_snapshot

CLAIM_CALIBRATION_VERSION: Literal["claim-calibration-v1"] = "claim-calibration-v1"
RUBRIC_VERSION = "answer-rubric-v1"
JUDGE_VERSION = "claim-judge-v1"
HumanDimension = Literal[
    "task_completion",
    "factual_correctness",
    "evidence_attribution",
    "coverage_synthesis",
    "epistemic_honesty",
    "research_usefulness",
]
DIMENSIONS: tuple[HumanDimension, ...] = (
    "task_completion", "factual_correctness", "evidence_attribution",
    "coverage_synthesis", "epistemic_honesty", "research_usefulness",
)
HumanScore = Literal[0, 1, 2]
MeasurementStatus = Literal["measured", "not_measured", "missing", "invalid_input"]
SemanticErrorLabel = Literal[
    "false_resolution", "incorrect_causal_relation", "unsupported_specific_assertion",
    "unit_condition_negation_mismatch", "invalid_attribution", "factual_error",
]


class CalibrationModel(BaseModel):
    """校准数据的不可变基础模型。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class AnswerPosition(CalibrationModel):
    """记录主张在最终回答块中的位置，无法定位时保留空值。"""

    block_index: int = Field(ge=0)
    block_role: str = Field(min_length=1)
    text_span: str | None = None


class ClaimReviewItem(CalibrationModel):
    """从冻结运行提取的一条实际回答主张及其结构化引用关系。"""

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_type: str = Field(min_length=1)
    validation_status: str = ""
    requires_citation: bool
    cited_evidence_ids: tuple[str, ...] = ()
    citable_evidence_ids: tuple[str, ...] = ()
    observed_evidence_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    related_evidence_ids: tuple[str, ...] = ()
    evidence_span_ids: tuple[str, ...] = ()
    evidence_span_links: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    answer_positions: tuple[AnswerPosition, ...] = ()
    missing_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_identity(self) -> "ClaimReviewItem":
        """无额外参数；拒绝重复主张或引用身份。"""
        if len(set(self.cited_evidence_ids)) != len(self.cited_evidence_ids):
            raise ValueError("duplicate cited evidence id")
        if len(set(self.evidence_span_ids)) != len(self.evidence_span_ids):
            raise ValueError("duplicate evidence span id")
        return self


class ClaimExtraction(CalibrationModel):
    """一次运行的主张提取快照；缺失时不从回答文本猜测。"""

    schema_version: Literal["claim-calibration-v1"] = CLAIM_CALIBRATION_VERSION
    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_mode: str = Field(min_length=1)
    review_status: str = Field(min_length=1)
    human_quality_eligible: bool
    status: MeasurementStatus
    reason: str
    items: tuple[ClaimReviewItem, ...] = ()
    answer: str = ""
    extraction_version: str = "claim-extraction-v1"
    projection_version: str = PROJECTION_VERSION
    projection_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_human_eligibility(self) -> "ClaimExtraction":
        """无额外参数；人工分母身份必须由运行模式和审核状态唯一决定。"""
        expected = self.run_mode == "agent_live" and self.review_status == "reviewed"
        if self.human_quality_eligible != expected:
            raise ValueError("human quality eligibility does not match run identity")
        return self


class ClaimMetricReport(CalibrationModel):
    """Claim/Citation 确定性指标及其逐项输入。"""

    status: MeasurementStatus
    reason: str
    metrics: tuple[MetricResult, ...]
    claim_ids: tuple[str, ...] = ()


class ClaimAnnotation(CalibrationModel):
    """单条主张的人工支持、事实和错误类型判断。"""

    claim_id: str = Field(min_length=1)
    support: Literal["supported", "partially_supported", "unsupported", "not_applicable"]
    factual_correctness: HumanScore
    error_labels: tuple[SemanticErrorLabel, ...] = ()
    reason: str = Field(min_length=1)


class RubricAnchor(CalibrationModel):
    """人工维度的一个 0-2 分可判定锚点。"""

    score: HumanScore
    criteria: str = Field(min_length=1)


class RubricDimensionDefinition(CalibrationModel):
    """人工评分维度及其完整锚点。"""

    dimension: HumanDimension
    title_zh: str = Field(min_length=1)
    anchors: tuple[RubricAnchor, RubricAnchor, RubricAnchor]

    @model_validator(mode="after")
    def validate_anchor_scores(self) -> "RubricDimensionDefinition":
        """无额外参数；每个维度必须按 0、1、2 顺序声明锚点。"""
        if tuple(item.score for item in self.anchors) != (0, 1, 2):
            raise ValueError("rubric anchors must contain scores 0, 1, 2")
        return self


class RubricDefinition(CalibrationModel):
    """版本化六维人工评分规则。"""

    rubric_version: str = Field(default=RUBRIC_VERSION, min_length=1)
    dimensions: tuple[RubricDimensionDefinition, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "RubricDefinition":
        """无额外参数；每个标准维度必须且只能出现一次。"""
        if tuple(item.dimension for item in self.dimensions) != DIMENSIONS:
            raise ValueError("rubric dimensions must use the canonical order")
        return self


class AnnotationRecord(CalibrationModel):
    """一名人工标注者的完整回答评分，不合成或覆盖原始分数。"""

    annotation_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    annotator_id: str = Field(min_length=1)
    rubric_version: str = Field(default=RUBRIC_VERSION, min_length=1)
    dimensions: dict[HumanDimension, HumanScore]
    claim_annotations: tuple[ClaimAnnotation, ...] = ()
    reasons: dict[HumanDimension, str]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    independent: bool = True

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: dict[HumanDimension, HumanScore]) -> dict[HumanDimension, HumanScore]:
        """参数为六维评分；要求每个维度显式给出 0-2 分。"""
        if set(value) != set(DIMENSIONS):
            raise ValueError("all six rubric dimensions are required")
        return dict(value)

    @model_validator(mode="after")
    def validate_annotation_identity(self) -> "AnnotationRecord":
        """无额外参数；独立评分不得伪装成同一标注者的重复提交。"""
        ids = [item.claim_id for item in self.claim_annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate claim annotation")
        if not self.independent:
            raise ValueError("human calibration annotations must be independent")
        if set(self.reasons) != set(DIMENSIONS) or any(not value.strip() for value in self.reasons.values()):
            raise ValueError("all six rubric dimension reasons are required")
        return self


class ArbitrationRecord(CalibrationModel):
    """保存双人分歧及仲裁后的单独结果。"""

    arbitration_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    annotation_ids: tuple[str, ...] = Field(min_length=2)
    arbiter_id: str = Field(min_length=1)
    rubric_version: str = Field(default=RUBRIC_VERSION, min_length=1)
    disagreements: tuple[str, ...] = ()
    resolved_dimensions: dict[HumanDimension, HumanScore] = Field(default_factory=dict)
    resolved_claims: tuple[ClaimAnnotation, ...] = ()
    reason: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def validate_resolution(self) -> "ArbitrationRecord":
        """无额外参数；仲裁只解决显式分歧并保留两份不同原始记录。"""
        if len(set(self.annotation_ids)) != 2:
            raise ValueError("arbitration requires two distinct annotations")
        if set(self.resolved_dimensions) != set(self.disagreements):
            raise ValueError("arbitration resolutions must match disagreements")
        return self


class DiagnosticJudgeResult(CalibrationModel):
    """自动 judge 的可追溯诊断结果；永远不表示硬门禁资格。"""

    judge_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    judge_version: str = Field(default=JUDGE_VERSION, min_length=1)
    status: Literal["diagnostic"] = "diagnostic"
    calibrated: bool = False
    dimensions: dict[HumanDimension, HumanScore] = Field(default_factory=dict)
    claim_labels: dict[str, str] = Field(default_factory=dict)
    request_snapshot: str
    response_snapshot: str
    error_labels: tuple[SemanticErrorLabel, ...] = ()

    @field_validator("request_snapshot", "response_snapshot")
    @classmethod
    def canonical_snapshot(cls, value: str) -> str:
        """参数为 judge 请求/响应 JSON；返回规范快照，禁止保存不可解析文本。"""
        return snapshot(read_snapshot(value))

    @property
    def hard_gate_eligible(self) -> bool:
        """无参数；自动 judge 未完成校准时始终不能进入硬门禁。"""
        return False


class AgreementReport(CalibrationModel):
    """两名独立标注者的逐维一致性与分歧统计。"""

    case_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    annotation_ids: tuple[str, str]
    annotator_ids: tuple[str, str]
    comparable_dimensions: int = Field(ge=0)
    agreeing_dimensions: int = Field(ge=0)
    disagreement_dimensions: int = Field(ge=0)
    agreement_rate: float | None = None
    disagreement_rate: float | None = None
    arbitration_required: bool = False

    @model_validator(mode="after")
    def validate_counts(self) -> "AgreementReport":
        """无额外参数；确保一致和分歧计数可重建。"""
        if self.agreeing_dimensions + self.disagreement_dimensions != self.comparable_dimensions:
            raise ValueError("agreement counts do not sum to comparable dimensions")
        if self.annotator_ids[0] == self.annotator_ids[1]:
            raise ValueError("agreement requires two annotators")
        return self


class CalibrationRecord(CalibrationModel):
    """报告行使用的统一评分容器。"""

    extraction: ClaimExtraction
    claim_metrics: ClaimMetricReport
    annotations: tuple[AnnotationRecord, ...] = ()
    arbitration: ArbitrationRecord | None = None
    judge: DiagnosticJudgeResult | None = None

    @model_validator(mode="after")
    def validate_record_identities(self) -> "CalibrationRecord":
        """无额外参数；拒绝将其他运行的标注、仲裁和 judge 混入本记录。"""
        identity = (self.extraction.case_id, self.extraction.run_id)
        if any((item.case_id, item.run_id) != identity for item in self.annotations):
            raise ValueError("annotation identity mismatch")
        if len(self.annotations) > 2:
            raise ValueError("calibration record supports at most two independent annotations")
        versions = {item.rubric_version for item in self.annotations}
        if len(versions) > 1:
            raise ValueError("calibration record cannot mix rubric versions")
        if self.arbitration and (self.arbitration.case_id, self.arbitration.run_id) != identity:
            raise ValueError("arbitration identity mismatch")
        if self.arbitration and set(self.arbitration.annotation_ids) != {item.annotation_id for item in self.annotations}:
            raise ValueError("arbitration annotation references mismatch")
        if self.arbitration and versions != {self.arbitration.rubric_version}:
            raise ValueError("arbitration rubric version mismatch")
        if self.judge and (self.judge.case_id, self.judge.run_id) != identity:
            raise ValueError("judge identity mismatch")
        return self


def annotation_is_eligible(extraction: ClaimExtraction) -> bool:
    """参数为主张提取结果；仅真实 agent_live 且已审核回答进入人工质量分母。"""
    complete_items = all(not item.missing_fields for item in extraction.items)
    return extraction.human_quality_eligible and extraction.status == "measured" and complete_items


def dump_annotation(annotation: AnnotationRecord) -> dict[str, Any]:
    """参数为人工记录；返回可持久化字典，保留版本和原分。"""
    return annotation.model_dump(mode="json")
