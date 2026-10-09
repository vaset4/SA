"""【研究规划】【实时策略】模型只提交研究选择，执行许可由服务端派生。作者：xxx。"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from app.models.research_planner_contracts import (
    AnswerSectionRequirement, ClaimType, DeliverableKind, DependencyProduct, EvidencePolicy,
    FacetSource, OperationFamily, PlannerContractModel, PresentationKind, RequestedFacet, TurnClarificationRequest,
)


class StrategyContractModel(PlannerContractModel):
    """【研究规划】【信任边界】严格接收模型原值，不清洗类型和原文。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=False)


class ClaimIntentDraft(StrategyContractModel):
    """【研究规划】【产出意图】每类主张只声明一次，明确是否必须交付。"""

    claim_type: ClaimType
    required: bool


class ClaimSupportSelectionDraft(StrategyContractModel):
    """【研究主张】【原文选择】模型选取原文区间，材料版本由同次可见性记录绑定。"""

    evidence_id: str = Field(min_length=1, pattern=r"\S")
    field: str = Field(min_length=1, pattern=r"\S")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class ResearchClaimStrategyDraft(StrategyContractModel):
    """【研究主张】【模型输出】只保留模型判断的内容和支持选择，身份由程序提供。"""

    text: str = Field(min_length=1, pattern=r"\S", description=(
        "State only what the selected support establishes, preserving objects, protocol, conditions, metrics, "
        "quantifiers and author attribution. Do not strengthen an aggregate or unspecified finding into all/every/always."))
    claim_type: ClaimType
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    supporting_spans: list[ClaimSupportSelectionDraft] = Field(default_factory=list)
    reasoning_summary: str = ""
    validation_tasks: list[str] = Field(default_factory=list)
    addressed_facet_ids: list[str] = Field(default_factory=list, description=(
        "Assigned facets this exact claim addresses for the same requested object, experiment protocol, conditions "
        "and metrics. General background or a different/proxy protocol is not the requested result merely because "
        "the topic matches; use [] for such background. A source-backed limitation may address its affected facet "
        "only by explicitly identifying unavailable requested items, without claiming those results were supplied."))


class ResearchClaimStrategyDraftBatch(StrategyContractModel):
    """【研究主张】【批量草拟】实时结构化模型输出的唯一schema。"""

    claims: list[ResearchClaimStrategyDraft]


class UnitDependencyDraft(StrategyContractModel):
    """【研究规划】【上游选择】产物类型与必需性都必须显式选择。"""

    unit_id: str = Field(min_length=1, pattern=r"\S")
    required_product: DependencyProduct
    required: bool
    required_claim_types: list[ClaimType] = Field(default_factory=list)
    reason: str = ""


class RequestedFacetDraft(RequestedFacet):
    """【研究规划】【任务要求】实时要求没有历史未知默认值。"""

    model_config = StrategyContractModel.model_config
    source: FacetSource = Field(description=(
        "explicit means a user-requested requirement and must remain required with verbatim user_text_span; "
        "inferred means a task inference, with rationale when required; do not fabricate user words."
    ))
    user_text_span: str = Field(default="", description=(
        "For explicit facets, copy a nonempty contiguous span from Question or a visible user history excerpt, "
        "preserving whitespace and line breaks exactly. Assistant text, resolved questions, paper text and "
        "excerpt markers are not user words. Inferred facets may leave this empty and explain their rationale."
    ))
    required: bool
    deliverable_kind: DeliverableKind


class AnswerSectionRequirementDraft(AnswerSectionRequirement):
    """【研究规划】【展示要求】区分内部组织与用户要求，保留用户原话。"""

    model_config = StrategyContractModel.model_config
    presentation_kind: PresentationKind
    user_text_span: str

    @model_validator(mode="after")
    def _validate_user_presentation(self) -> "AnswerSectionRequirementDraft":
        """参数为展示要求；返回具备原文依据的用户要求或内部组织说明。"""
        if self.presentation_kind != "internal" and not self.user_text_span.strip():
            raise ValueError("user presentation requires nonempty user_text_span")
        return self


class ResearchUnitStrategyDraft(StrategyContractModel):
    """【研究规划】【行动策略】首次与重规划共享单元，禁止模型填写执行政策。"""

    unit_id: str = Field(min_length=1, pattern=r"\S")
    goal: str = Field(min_length=1, pattern=r"\S")
    research_action: str = Field(min_length=1, pattern=r"\S")
    evidence_policy: EvidencePolicy
    output_goal: str = Field(min_length=1, pattern=r"\S")
    deliverable_kind: DeliverableKind
    claim_intents: list[ClaimIntentDraft]
    output_role: Literal["synthesis", "final_recommendation"] | None = None
    operation_family: OperationFamily | None = None
    dependencies: list[UnitDependencyDraft] = Field(default_factory=list)
    covered_facet_ids: list[str] = Field(default_factory=list)
    validation_tasks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_output_intent(self) -> "ResearchUnitStrategyDraft":
        """参数为行动单元；返回产物意图、来源角色和依赖声明一致的单元。"""
        kinds = [intent.claim_type for intent in self.claim_intents]
        if len(set(kinds)) != len(kinds):
            raise ValueError("claim_intents cannot repeat a claim_type")
        if self.deliverable_kind == "materials" and kinds:
            raise ValueError("materials cannot declare claim_intents")
        if self.deliverable_kind == "claims" and not any(item.required for item in self.claim_intents):
            raise ValueError("claims requires an explicitly required claim intent")
        inference = self.evidence_policy.source_type == "inference_from_units"
        if inference and (not self.dependencies or self.output_role is None):
            raise ValueError("inference_from_units requires dependencies and output_role")
        if not inference and self.output_role is not None:
            raise ValueError("output_role is only selected for inference_from_units")
        return self


class ResearchReplanStrategyDraft(StrategyContractModel):
    """【研究规划】【重规划】只改变行动，已接受的内容与展示要求由程序保留。"""

    units: list[ResearchUnitStrategyDraft]
    clarification: TurnClarificationRequest | None = None

    @model_validator(mode="after")
    def _validate_action_or_clarification(self) -> "ResearchReplanStrategyDraft":
        """参数为草案；返回有效行动或明确澄清，拒绝没有动作的空成功。"""
        if not self.units and not (self.clarification and self.clarification.needed):
            raise ValueError("strategy requires units unless clarification asks the user")
        return self


class ResearchPlanStrategyDraft(ResearchReplanStrategyDraft):
    """【研究规划】【首次规划】同一次模型调用识别任务要求并选择行动。"""

    requested_facets: list[RequestedFacetDraft] = Field(description=(
        "Initial planning with any units requires at least one real task requirement. "
        "A pure clarification with no units may leave this empty. "
        "Replanning uses the separate action schema and preserves the accepted requirements baseline."
    ))
    answer_sections: list[AnswerSectionRequirementDraft]

    @model_validator(mode="after")
    def _validate_initial_requirements(self) -> "ResearchPlanStrategyDraft":
        """参数为首次策略；返回具备内容要求的行动计划，澄清允许暂不建立要求。"""
        if self.units and not self.requested_facets:
            raise ValueError("initial action plan requires requested_facets")
        return self
