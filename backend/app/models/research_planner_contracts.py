"""Strict contracts for the project-scoped research planner."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.models.evidence_layers import (
    PLANNER_SOURCE_LOCAL_DIRECT_TEXT,
    PLANNER_SOURCE_LOCAL_SECTION_TEXT,
)
from app.models.research_contracts import TurnDepth
from app.models.research_memory_contracts import MemoryType


PlanType = Literal["single_unit", "multi_unit"]
OperationFamily = Literal[
    "evidence_lookup",
    "comparison",
    "synthesis",
    "creative_analysis",
    "external_context_request",
]
EvidenceSourceType = Literal[
    "local_direct_text",
    "local_metadata",
    "local_structure_hint",
    "external_context_needed",
    "inference_from_units",
    "local_section_structure",
    "local_section_text",
    "local_chunk_metadata",
]
ClaimType = Literal[
    "paper_supported",
    "evidence_grounded_inference",
    "model_prior_hypothesis",
    "external_needed",
    "memory_continuation",
]
# 这两类主张都不是从本轮论文证据来的：假设出自模型先验，延续判断出自项目历史记忆，
# 带上 supporting_evidence_ids 就会在最终回答里冒充带引用的论文结论
_CLAIM_TYPES_WITHOUT_EVIDENCE = frozenset({"model_prior_hypothesis", "memory_continuation"})
# 可产出 paper_supported 引用的本地来源：向量检索片段，以及按章节路径取回的真实正文。
# 其余本地来源只能支撑 evidence_grounded_inference，不能拿自己的记录当论文结论的引用
_CITABLE_LOCAL_SOURCE_TYPES = frozenset({
    PLANNER_SOURCE_LOCAL_DIRECT_TEXT,
    PLANNER_SOURCE_LOCAL_SECTION_TEXT,
})

UnitStatus = Literal[
    "planned",
    "evidence_found",
    "inferred",
    "insufficient_evidence",
    "external_needed",
    "blocked",
]
FacetSource = Literal["explicit", "inferred"]
RepairStatus = Literal["not_needed", "needed", "repaired", "failed"]
ClaimValidationStatus = Literal["validated", "unsupported", "needs_validation", "rejected"]
PreferredBlockType = Literal["paragraph", "table", "figure_caption", "formula", "code"]
# 单元级取证粒度：文本取证单元要多少正文。子块只取向量命中的片段，父块额外回读该块所属父块全文。
# 「只要位置不要正文」不在这里表达，那是 source_type=local_chunk_metadata 这一独立来源的职责
EvidenceGranularity = Literal["child_chunk", "parent_chunk"]
UnitOutputRole = Literal["evidence", "synthesis", "final_recommendation"]
EvidenceRepairStatus = Literal["not_needed", "attempted", "resolved", "partial", "failed", "skipped"]
EvidenceGapSeverity = Literal["hard", "warning"]
CURRENT_RESEARCH_CONTRACT_VERSION = "research-planner-v4"
ResearchContractVersion = Literal["research-planner-v1", "research-planner-v2", "research-planner-v3", "research-planner-v4"]
PresentationKind = Literal["internal", "user_section", "exact_heading"]
DeliverableKind = Literal["materials", "claims"]
DependencyProduct = Literal["materials", "claims", "hypotheses"]


class PlannerContractModel(BaseModel):
    """Base class for planner payloads crossing the LLM trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


class RequestedFacet(PlannerContractModel):
    facet_id: str = Field(min_length=1, pattern=r"\S")
    label: str = Field(min_length=1, pattern=r"\S")
    source: FacetSource
    user_text_span: str = ""
    rationale: str = ""
    # 【研究规划】【要求基准】None 保留旧记录的未知身份，实时计划必须明确交付责任
    required: bool | None = None
    required_paper_ids: list[str] = Field(default_factory=list)
    deliverable_kind: DeliverableKind | None = None
    required_claim_types: list[ClaimType] = Field(default_factory=list)
    min_material_count: int = Field(default=0, ge=0)
    required_paper_counts: dict[str, Annotated[int, Field(ge=1)]] = Field(default_factory=dict)
    required_output_role: UnitOutputRole | None = None


class AnswerSectionRequirement(PlannerContractModel):
    """【研究规划】【章节要求】保存稳定章节、关联要求与必需性的任务依据。"""

    section_id: str = Field(min_length=1, pattern=r"\S")
    title: str = Field(min_length=1, pattern=r"\S")
    facet_ids: list[str] = Field(default_factory=list)
    required: bool
    source: FacetSource
    user_text_span: str = ""
    rationale: str = ""
    # 1. 【研究规划】【展示语义】旧记录缺失时保持未知，新执行由实时合同明确声明
    presentation_kind: PresentationKind | None = None


class PlannerCoverageReview(PlannerContractModel):
    answer_sections: list[AnswerSectionRequirement] = Field(default_factory=list)
    explicit_facets: list[RequestedFacet] = Field(default_factory=list)
    inferred_facets: list[RequestedFacet] = Field(default_factory=list)
    covered_facet_ids: list[str] = Field(default_factory=list)
    missing_explicit_facet_ids: list[str] = Field(default_factory=list)
    missing_inferred_facet_ids: list[str] = Field(default_factory=list)
    unit_facet_map: dict[str, list[str]] = Field(default_factory=dict)
    repair_status: RepairStatus = "not_needed"


class EvidencePolicy(PlannerContractModel):
    source_type: EvidenceSourceType
    # 决定证据文本取子块片段还是父块全文：child_chunk 只取向量命中片段（默认，与未声明粒度
    # 时逐字段一致），parent_chunk 额外回读父块全文填进 expanded_text（条数受
    # rag_auto_expand_parent_limit 约束）。只在 local_direct_text 上真正生效
    evidence_granularity: EvidenceGranularity = "child_chunk"
    # 结构线索（local_structure_hint）要不要先用正文证据验证过才升级为 evidence_found：
    # 不索取时线索只作规划已知事实，本单元一次向量检索都不发；索取时才多走一跳正文验证。
    # 「有哪些章节」这类结构性问题因此默认不为验证付一次检索
    verify_with_direct_text: bool = False
    target_paper_ids: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    continuation_ids: list[str] = Field(default_factory=list)
    evidence_dimensions: list[str] = Field(default_factory=list)
    preferred_block_types: list[PreferredBlockType] = Field(default_factory=list)
    # 按元数据筛块（local_chunk_metadata）的硬条件字段：块类型与页码，与 required_sections 一起
    # 构成筛块条件，三者至少给一个才会调用适配器
    required_block_types: list[PreferredBlockType] = Field(default_factory=list)
    # 页码精确匹配、不做范围解析；元素下限 1，0 和负数在信任边界即被拒
    required_pages: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)
    locator_queries: list[str] = Field(default_factory=list)
    min_citations_per_target: int = Field(default=1, ge=0)

    @property
    def can_produce_citations(self) -> bool:
        """按章节路径取回的正文与检索片段同等待遇，都能产出带引用的 paper_supported。"""
        return self.source_type in _CITABLE_LOCAL_SOURCE_TYPES or (
            self.source_type == "local_structure_hint" and self.verify_with_direct_text
        )


class ClaimPolicy(PlannerContractModel):
    allowed_claim_types: list[ClaimType] = Field(default_factory=list)
    citation_required: bool = False
    bounded_inference_allowed: bool = False


class MaterialSelection(PlannerContractModel):
    """【研究分析】【材料选择】只选择已返回材料及系统提供的范围身份，不接受模型字符偏移。"""

    evidence_id: str = Field(min_length=1, pattern=r"\S")
    range_id: str | None = Field(default=None, min_length=1, pattern=r"\S")


class UnitDependency(PlannerContractModel):
    unit_id: str = Field(min_length=1, pattern=r"\S")
    required: bool = True
    required_claim_types: list[ClaimType] = Field(default_factory=list)
    reason: str = ""
    required_product: DependencyProduct | None = None
    selections: list[MaterialSelection] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_material_selection_owner(self) -> "UnitDependency":
        """参数为依赖合同；具体证据选择仅属于材料依赖，不能冒充结论选择。"""
        if self.selections is not None and self.required_product != "materials":
            raise ValueError("material selections require a materials dependency")
        return self


class ExpectedOutput(PlannerContractModel):
    output_goal: str = Field(min_length=1, pattern=r"\S")
    required_claim_types: list[ClaimType] = Field(default_factory=list)
    required_answer_sections: list[str] = Field(default_factory=list)
    validation_tasks: list[str] = Field(default_factory=list)
    deliverable_kind: DeliverableKind | None = None
    answer_section_ids: list[str] = Field(default_factory=list)


class ResearchUnit(PlannerContractModel):
    unit_id: str = Field(min_length=1, pattern=r"\S")
    goal: str = Field(min_length=1, pattern=r"\S")
    research_action: str = Field(min_length=1, pattern=r"\S")
    evidence_policy: EvidencePolicy
    claim_policy: ClaimPolicy
    expected_output: ExpectedOutput
    output_role: UnitOutputRole | None = None
    operation_family: OperationFamily | None = None
    dependencies: list[UnitDependency] = Field(default_factory=list)
    covered_facet_ids: list[str] = Field(default_factory=list)


class ClaimSupportSpan(PlannerContractModel):
    """【研究主张】【直接支持】保留材料版本及实际可见字段中的原文摘录。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)
    evidence_id: str = Field(min_length=1, pattern=r"\S")
    evidence_version: str = Field(min_length=1, pattern=r"\S")
    field: str = Field(min_length=1, pattern=r"\S")
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    quote: str = Field(min_length=1)


class EvidenceExposure(PlannerContractModel):
    """【研究输入】【材料可见性】记录实际送模区间及哈希，不复制原文。"""

    evidence_id: str
    evidence_version: str
    field: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    content_hash: str


class ClaimExposure(PlannerContractModel):
    """【研究输入】【主张可见性】绑定进入依赖上下文的具体主张内容。"""

    claim_id: str
    content_hash: str


class UnprovidedEvidenceRange(EvidenceExposure):
    """【研究输入】【未提供范围】实际材料中未送入本次模型的范围，身份由系统生成。"""

    range_id: str


class ConsumerExposure(PlannerContractModel):
    """【研究输入】【实际投影】不可变地保存一次准备的范围、预算与省略事实。"""

    schema_version: Literal["research-exposure-v1", "research-exposure-v2"] = "research-exposure-v1"
    evidence: tuple[EvidenceExposure, ...] = ()
    claims: tuple[ClaimExposure, ...] = ()
    acquired_evidence_ids: tuple[str, ...] = ()
    omitted_evidence_ids: tuple[str, ...] = ()
    omitted_claim_ids: tuple[str, ...] = ()
    evidence_max_chars: int | None = Field(ge=0)
    unprovided_ranges: tuple[UnprovidedEvidenceRange, ...] = ()
    prompt_version: str
    rendering_version: str = "research-consumer-v3"
    rendered_hash: str
    # 1. 【研究输入】【缓存判等】历史投影缺少身份规范渲染摘要时保留原记录，但不开放免推理
    identity_normalized_rendered_hash: str = ""


class ResearchClaim(PlannerContractModel):
    claim_id: str = Field(min_length=1, pattern=r"\S")
    text: str = Field(min_length=1, pattern=r"\S")
    claim_type: ClaimType
    validation_status: ClaimValidationStatus = "needs_validation"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    source_unit_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    supporting_spans: list[ClaimSupportSpan] = Field(default_factory=list)
    reasoning_summary: str = ""
    validation_tasks: list[str] = Field(default_factory=list)
    addressed_facet_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_claim_sources(self) -> "ResearchClaim":
        """参数为当前主张；返回来源声明合法的主张，缺少必要支持或推断说明时显式报错。"""
        if self.claim_type == "paper_supported" and not self.supporting_evidence_ids:
            raise ValueError("paper_supported claims require supporting_evidence_ids")
        if self.claim_type == "evidence_grounded_inference":
            has_source = self.supporting_evidence_ids or self.source_claim_ids or self.source_unit_ids
            if not has_source or not self.reasoning_summary:
                raise ValueError("inference claims require source links and reasoning_summary")
        if self.claim_type in _CLAIM_TYPES_WITHOUT_EVIDENCE and self.supporting_evidence_ids:
            raise ValueError(f"{self.claim_type} cannot carry supporting_evidence_ids")
        return self


class ResearchClaimDraft(PlannerContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=False)

    text: str = Field(min_length=1, pattern=r"\S")
    claim_type: ClaimType
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    supporting_spans: list[ClaimSupportSpan] = Field(default_factory=list)
    reasoning_summary: str = ""
    validation_tasks: list[str] = Field(default_factory=list)
    addressed_facet_ids: list[str] = Field(default_factory=list)


class ResearchClaimDraftBatch(PlannerContractModel):
    claims: list[ResearchClaimDraft] = Field(default_factory=list)


class ResearchClaimDraftRejection(PlannerContractModel):
    """【研究主张】【单条拒绝】保留从1开始的原位置和模型原稿，不给错误来源补版本。"""

    claim_position: int = Field(ge=1)
    code: Literal["support_selection_not_exposed"]
    message: str
    evidence_ids: list[str]
    raw_claim: dict[str, Any]


class BoundResearchClaimDraftBatch(ResearchClaimDraftBatch):
    """【研究主张】【绑定结果】部分可绑定不等于整批合格，合法稿继续接受完整支持校验。"""

    claim_positions: list[int]
    rejected_claims: list[ResearchClaimDraftRejection]

    @model_validator(mode="after")
    def _validate_original_positions(self) -> "BoundResearchClaimDraftBatch":
        """参数为当前绑定批次；返回位置完整且互不重叠的结果，程序丢位置时直接报错。"""
        positions = [*self.claim_positions, *(item.claim_position for item in self.rejected_claims)]
        if (len(self.claim_positions) != len(self.claims) or self.claim_positions != sorted(self.claim_positions)
                or sorted(positions) != list(range(1, len(positions) + 1))):
            raise ValueError("bound claim positions must preserve every original item exactly once")
        return self


class ResearchUnitResult(PlannerContractModel):
    unit_id: str = Field(min_length=1, pattern=r"\S")
    status: UnitStatus
    summary: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    claims: list[ResearchClaim] = Field(default_factory=list)
    unresolved_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceGap(PlannerContractModel):
    gap_id: str = Field(min_length=1, pattern=r"\S")
    unit_id: str = Field(min_length=1, pattern=r"\S")
    gap_type: str = Field(min_length=1, pattern=r"\S")
    severity: EvidenceGapSeverity = "hard"
    target_paper_id: str = ""
    evidence_dimension: str = ""
    required_count: int = Field(default=0, ge=0)
    found_count: int = Field(default=0, ge=0)
    reason: str = Field(min_length=1, pattern=r"\S")
    facet_id: str = ""
    requirement_key: str = ""


class UnitSufficiencyReport(PlannerContractModel):
    unit_id: str = Field(min_length=1, pattern=r"\S")
    pass_index: int = Field(ge=1)
    hard_gaps: list[EvidenceGap] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    target_coverage: dict[str, int] = Field(default_factory=dict)
    claim_coverage: dict[str, list[str]] = Field(default_factory=dict)
    blocking_for_final: bool = False
    repair_recommended: bool = False
    fulfilled: bool = False
    usable_evidence_ids: list[str] = Field(default_factory=list)
    usable_claim_ids: list[str] = Field(default_factory=list)


class FacetSufficiencyReport(PlannerContractModel):
    """【研究规划】【实际满足】要求点聚合的实际关联，不宣称自然语言语义已经证明。"""

    facet_id: str = Field(min_length=1, pattern=r"\S")
    required: bool
    satisfied: bool
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    target_coverage: dict[str, int] = Field(default_factory=dict)
    missing_paper_ids: list[str] = Field(default_factory=list)
    hard_gaps: list[EvidenceGap] = Field(default_factory=list)
    fulfilled_requirement_keys: list[str] = Field(default_factory=list)


class EvidenceSufficiencyReport(PlannerContractModel):
    pass_index: int = Field(ge=1)
    unit_reports: list[UnitSufficiencyReport] = Field(default_factory=list)
    blocking_unit_ids: list[str] = Field(default_factory=list)
    repair_recommended: bool = False
    contract_version: ResearchContractVersion = "research-planner-v1"
    requirements_version: str = ""
    facet_reports: list[FacetSufficiencyReport] = Field(default_factory=list)
    fulfilled_required_facet_ids: list[str] = Field(default_factory=list)
    missing_required_facet_ids: list[str] = Field(default_factory=list)
    has_usable_results: bool = False
    fulfilled_requirement_keys: list[str] = Field(default_factory=list)


class RepairQuery(PlannerContractModel):
    query: str = Field(min_length=1, pattern=r"\S")
    target_paper_id: str = ""
    evidence_dimension: str = ""
    required_sections: list[str] = Field(default_factory=list)
    preferred_block_types: list[PreferredBlockType] = Field(default_factory=list)
    source_gap_id: str = Field(min_length=1, pattern=r"\S")


class RepairQueryPlan(PlannerContractModel):
    source_report_pass: int = Field(ge=1)
    linked_unit_id: str = Field(min_length=1, pattern=r"\S")
    repair_unit_id: str = Field(min_length=1, pattern=r"\S")
    queries: list[RepairQuery] = Field(default_factory=list)
    skipped_gap_ids: list[str] = Field(default_factory=list)
    skipped_reasons: dict[str, str] = Field(default_factory=dict)


class RepairTransientUnit(PlannerContractModel):
    unit: ResearchUnit
    linked_unit_id: str = Field(min_length=1, pattern=r"\S")
    source_gap_id: str = Field(min_length=1, pattern=r"\S")
    retrieval_pass: int = Field(default=2, ge=2)


class PlannerRepairMergeReport(PlannerContractModel):
    seen_evidence_ids: list[str] = Field(default_factory=list)
    duplicate_evidence_ids: list[str] = Field(default_factory=list)
    invalid_evidence_ids: list[str] = Field(default_factory=list)
    conflict_records: list[dict[str, Any]] = Field(default_factory=list)
    pass_counts: dict[str, int] = Field(default_factory=dict)


class FinalAnswerEligibilityReport(PlannerContractModel):
    can_call_provider: bool
    final_status_if_skipped: Literal["partial", "failed", "skipped"] = "skipped"
    blocking_unit_ids: list[str] = Field(default_factory=list)
    blocking_gap_ids: list[str] = Field(default_factory=list)
    partial_allowed: bool = False
    reason: str = ""


class PlannerRepairRecord(PlannerContractModel):
    code: str = Field(min_length=1, pattern=r"\S")
    message: str = Field(min_length=1, pattern=r"\S")
    unit_id: str | None = None
    original: Any = None
    repaired: Any = None


class PlannerValidationIssue(PlannerContractModel):
    code: str = Field(min_length=1, pattern=r"\S")
    message: str = Field(min_length=1, pattern=r"\S")
    unit_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PlannerMetadata(PlannerContractModel):
    execution_mode: Literal["whole_plan", "incremental"] = "whole_plan"
    pending_unit_ids: list[str] = Field(default_factory=list)
    analyzer_plan_type: PlanType | None = None
    analyzer_reasoning: str = ""
    normalized_unit_id_map: dict[str, str] = Field(default_factory=dict)
    validation_errors: list[PlannerValidationIssue] = Field(default_factory=list)
    repairs: list[PlannerRepairRecord] = Field(default_factory=list)
    repair_attempted: bool = False
    repaired: bool = False


class ResearchPlanDraft(PlannerContractModel):
    """历史计划草案读取合同；实时模型使用独立的 ResearchPlanStrategyDraft。

    澄清是 Planner 的一个动作，但出计划这一步此前没有任何字段能表达它：判出「目标没确认」
    时只能交一份空计划（契约拒绝）或硬塞一个无关单元（等于先按某种理解把活干完）。
    `clarification` 给这条判开一个正式出口，与首次分析里那个澄清判断共用同一份契约。
    两个出口至少成立一个：`clarification` 说不问，`units` 就不许为空。
    """

    question: str = Field(min_length=1, pattern=r"\S")
    plan_type: PlanType
    units: list[ResearchUnit] = Field(default_factory=list)
    coverage_review: PlannerCoverageReview = Field(default_factory=PlannerCoverageReview)
    metadata: PlannerMetadata = Field(default_factory=PlannerMetadata)
    # 模型没提这件事时为 None，按不澄清处理
    clarification: TurnClarificationRequest | None = None
    contract_version: ResearchContractVersion = "research-planner-v1"

    @model_validator(mode="after")
    def _validate_clarification_or_units(self) -> "ResearchPlanDraft":
        # 判要澄清：这句问题就是本轮回答，不需要任何单元；缺问题或原因码的澄清
        # 经 TurnClarificationRequest 自己的校验拒掉，这里不重复
        if self.clarification is not None and self.clarification.needed:
            return self
        # 不澄清就必须有单元：空计划没有任何可执行动作，这一轮等于白烧
        if not self.units:
            raise ValueError("draft requires at least one unit unless clarification asks the user")
        return self


class ResearchPlan(PlannerContractModel):
    question: str = Field(min_length=1, pattern=r"\S")
    plan_type: PlanType
    units: list[ResearchUnit] = Field(min_length=1)
    coverage_review: PlannerCoverageReview
    metadata: PlannerMetadata = Field(default_factory=PlannerMetadata)
    contract_version: ResearchContractVersion = "research-planner-v1"

    @model_validator(mode="after")
    def _validate_current_presentation(self) -> "ResearchPlan":
        """参数为正式计划；返回保留历史未知且新版展示义务完整的计划。"""
        if self.contract_version != CURRENT_RESEARCH_CONTRACT_VERSION:
            return self
        for section in self.coverage_review.answer_sections:
            if section.presentation_kind is None:
                raise ValueError("current plan requires presentation_kind on every answer section")
            if section.presentation_kind != "internal" and not section.user_text_span.strip():
                raise ValueError("user presentation requirement requires user_text_span")
        return self

    @model_validator(mode="after")
    def _validate_current_execution(self) -> "ResearchPlan":
        """参数为正式计划；新版执行必须保留依赖和产物声明，历史读取不补造这些事实。"""
        if self.contract_version != CURRENT_RESEARCH_CONTRACT_VERSION:
            return self
        for unit in self.units:
            output = unit.expected_output
            if output.deliverable_kind is None:
                raise ValueError("current unit requires deliverable_kind")
            if not set(output.required_claim_types) <= set(unit.claim_policy.allowed_claim_types):
                raise ValueError("required claim types exceed the execution policy")
            if unit.evidence_policy.source_type == "inference_from_units" and (
                not unit.dependencies or unit.output_role not in {"synthesis", "final_recommendation"}
            ):
                raise ValueError("current inference requires dependencies and an inference output role")
            if any(dep.required_product is None for dep in unit.dependencies):
                raise ValueError("current dependency requires required_product")
            analysis = self.metadata.execution_mode == "incremental" and unit.evidence_policy.source_type == "inference_from_units"
            if "paper_supported" in unit.claim_policy.allowed_claim_types and not (unit.evidence_policy.can_produce_citations or analysis):
                raise ValueError("paper_supported requires a citable source")
        return self


class ProjectMemoryReadRequest(PlannerContractModel):
    """模型对本回合要不要翻项目研究记忆给出的判断与收窄条件。"""

    # 本轮是否需要读项目里已经形成的判断；False 是「模型判断本轮用不上」的正常结论
    needed: bool
    # 以下四个键是模型能给的收窄条件全集，只能把读取范围缩小、不能放宽。
    # 这里没有 review_state / status / stale_state / maturity / user_id / project_id：
    # 前四个是「哪些判断算数」的资格门、后两个是数据归属，都由服务层按登录用户与
    # 当前项目强制取值；放进来等于让模型有机会读到未被用户接纳或不属于本项目的判断。
    # 只看这篇论文相关的判断，取本轮聚焦论文；空串表示不按论文收窄
    paper_id: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    # 只看挂在这个研究问题下的判断；空串表示不按研究问题收窄
    question_id: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    # 只在明确只要某一类判断时给（例如只回忆过去的决定就给 decision_note）。
    # 用受控枚举而不是裸 str：模型编出不存在的记忆类型时在信任边界即被拒，
    # 不会带着无效类型流到记忆层查出空结果、再被误读成「项目里没这类判断」
    memory_type: MemoryType | None = None
    # 需要顺着「基于 / 冲突 / 修正」关系多看一跳时给
    follow_links: bool = False
    # 判断要不要翻记忆的稳定 snake_case 原因码，落在深度评估结果里供排查读取时机
    reason_code: str = ""


class TurnClarificationRequest(PlannerContractModel):
    """模型判断「本轮目标没法确认，得先问用户一句」时给出的澄清请求。

    澄清是 Planner 的一个可调用动作，不是深度评估失败：目标无法确认且不同理解会导致
    明显不同动作时先问用户，而不是擅自挑一条重研究路径去猜（prd.md:100）。
    """

    # 本轮是否必须先向用户澄清；False 是「模型判断目标已经足够明确」的正常结论
    needed: bool
    # 要问用户的那一句话，直接作为本轮回答返回给用户，所以必须是人话、不含内部组件名
    question: str = ""
    # 为什么非问不可的稳定 snake_case 原因码，随深度评估结果进 trace
    reason_code: str = ""

    @model_validator(mode="after")
    def _validate_clarification(self) -> "TurnClarificationRequest":
        # 判要问却没给问题，这一轮就没有任何可返回给用户的内容；没给原因码则 trace 上
        # 看不出为什么问，澄清就退化成一个无从追查的偷懒默认值。两者都在信任边界拒掉，
        # 不在下游补一句默认问句或默认原因码把模型的越界输出兜圆
        if self.needed and not (self.question.strip() and self.reason_code.strip()):
            raise ValueError("clarification requires both question and reason_code")
        return self


class DirectAnswerDraft(PlannerContractModel):
    """minimal 直答的结构化输出：正文 + 这一轮要不要附系统来源说明。

    正文归模型、来源归系统。模型最清楚自己刚写的内容是不是知识性的；而「这一轮到底
    跑了哪些能力」只有系统知道，那句来源说明由系统按真实四态拼，不经过模型生成。
    所以这里只要一个判断，不要让模型复述系统状态——实测那样做每轮都出现做不到。
    """

    answer: str = Field(min_length=1, pattern=r"\S")
    # 正文有没有给用户知识性内容（概念、事实、方法等）；纯寒暄、承接对话为 false。
    # true 才由系统补来源说明，闲聊底下挂一句「本轮未检索论文」是噪声
    knowledge_bearing: bool


class TurnDepthProposal(PlannerContractModel):
    """模型对本回合处理深度的偏好，是 Planner 首次分析的结构化输出。

    这只是「偏好」：确定性硬约束优先，越界由 turn_depth_assessor 判为决策失败，
    不在此处放宽字段类型来兼容越界输出。
    """

    depth: TurnDepth
    # 选择该深度的稳定原因码，snake_case，进 trace 供深度判断准确率评测取数
    reason_code: str = Field(min_length=1, pattern=r"\S")
    # 本轮回答里的论文事实是否必须由已验证的论文 Evidence 支撑
    requires_paper_evidence: bool
    # 把追问补全成自包含问句后的本轮检索问题
    resolved_query: str = Field(min_length=1, pattern=r"\S")
    # 本轮要不要翻项目研究记忆及怎么收窄；深度判断与记忆读取判断共用这一次结构化调用，
    # 不为「要不要翻记忆」再多调一次模型。None 表示模型没提这件事，按不翻处理
    project_memory: ProjectMemoryReadRequest | None = None
    # 本轮要不要先问用户一句再往下走；同样搭在这次调用上，不为澄清判断多调一次模型。
    # None 表示模型没提这件事，按不澄清处理
    clarification: TurnClarificationRequest | None = None


class PlannerFailure(PlannerContractModel):
    code: str = Field(min_length=1, pattern=r"\S")
    message: str = Field(min_length=1, pattern=r"\S")
    local_evidence_execution: Literal["not_run", "attempted"] = "not_run"
    trace: dict[str, Any] = Field(default_factory=dict)


class PlannerValidationResult(PlannerContractModel):
    plan: ResearchPlan | None = None
    failure: PlannerFailure | None = None

    @property
    def is_valid(self) -> bool:
        return self.plan is not None and self.failure is None
