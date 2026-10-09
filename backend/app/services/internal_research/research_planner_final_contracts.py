"""【研究交付】【合同】最终正文、引用和交付事实。作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.models.research_planner_contracts import ResearchPlan, ResearchUnitResult
from app.services.internal_research.research_answer_review_contracts import ResearchAnswerReview

FINAL_CONTRACT_VERSION = "research-planner-final-v3"
DELIVERY_POLICY_VERSION = "research-delivery-v4"
FINAL_DRAFT_VERSION = "research-planner-final-draft-v4"
REVISION_DELIVERY_POLICY_VERSION: Literal["research-delivery-v5"] = "research-delivery-v5"
FINAL_PROMPT_VERSION = "research-planner-final-prompt-v12"
FinalSectionPresentation = dict[str, dict[str, frozenset[int]]]
PlannerFinalStatus = Literal["ok", "partial", "failed", "skipped"]
FinalDeliveryStatus = Literal["complete", "partial", "insufficient", "failed"]
FinalContentKind = Literal["fact", "inference", "materials", "hypothesis", "history", "structure", "gap"]
PlannerFinalErrorCode = Literal[
    "malformed_provider_output", "provider_failed", "unsupported_contract_version", "invalid_block",
    "invalid_block_field", "duplicate_block_id", "unknown_claim_id", "unvalidated_claim_id",
    "unknown_evidence_id", "evidence_id_not_allowed_for_block", "missing_block_claim_ids",
    "missing_block_evidence_ids", "missing_support_citations", "unknown_facet_id", "unknown_section_id",
    "invalid_content_identity", "unassociated_product", "model_written_citation", "invalid_auxiliary_field",
    "material_not_visible", "claim_not_visible", "historical_support_not_assessed",
    "revision_rejected",
]


class PlannerFinalModel(BaseModel):
    """【研究交付】【合同】保留 Markdown 原始字符，不做字符串空白归一化。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FinalAnswerBlockDraft(PlannerFinalModel):
    """【研究交付】【正文块】模型只提供正文与选择关系，不提供第二份答案。"""

    block_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    content_kind: FinalContentKind
    facet_ids: list[str]
    section_ids: list[str]
    claim_ids: list[str]
    evidence_ids: list[str]


class FinalUnresolvedDraft(PlannerFinalModel):
    """【研究交付】【未解决声明】必需性由当前要求决定，不由模型自行降低。"""

    facet_id: str = ""
    section_id: str = ""
    reason: str = Field(min_length=1)


class PlannerFinalDraftBody(PlannerFinalModel):
    """【研究交付】【原始载体】正文块延后逐块验证，保留局部可读正文。"""

    blocks: list[Any]
    status: Literal["answered", "incomplete", "unable"] = "answered"
    confidence: str = ""
    failure_reason: str = ""
    unresolved_items: list[FinalUnresolvedDraft] = Field(default_factory=list)


class PlannerFinalAnswerDraft(PlannerFinalDraftBody):
    """【研究交付】【历史草稿】保留显式旧版输入的原始验证语义。"""

    contract_version: Literal["research-planner-final-v3"]


class FinalProductBlockBase(PlannerFinalModel):
    """【研究交付】【草稿正文】模型选择实际回应范围，身份与引用由服务端管理。"""

    text: str = Field(min_length=1)
    facet_ids: list[str]
    section_ids: list[str]


class FinalClaimBlockDraft(FinalProductBlockBase):
    """【研究交付】【主张草稿】仅选择主张，不重复填写主张类型和来源闭包。"""

    product_kind: Literal["claims"]
    claim_ids: list[str] = Field(min_length=1)


class FinalMaterialBlockDraft(FinalProductBlockBase):
    """【研究交付】【材料草稿】材料来源必须属于当前可见的合法产物。"""

    product_kind: Literal["materials"]
    evidence_ids: list[str] = Field(min_length=1)


class FinalOrganizationBlockDraft(FinalProductBlockBase):
    """【研究交付】【展示草稿】标题与不足说明不声明支持关系。"""

    product_kind: Literal["structure", "gap"]


FinalProductBlockDraft = Annotated[
    FinalClaimBlockDraft | FinalMaterialBlockDraft | FinalOrganizationBlockDraft, Field(discriminator="product_kind")]


class PlannerFinalProductDraft(PlannerFinalDraftBody):
    """【研究交付】【新版草稿】显式新线协议，逐块校验仍保留可读错误。"""

    contract_version: Literal["research-planner-final-draft-v4"]


class PlannerFinalValidationError(PlannerFinalModel):
    """【研究交付】【校验问题】按原列表位置定位，不修复或重编号错误块。"""

    code: PlannerFinalErrorCode
    block_index: int | None = None
    block_id: str = ""
    block_role: str = ""
    field: str = ""
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    message: str = ""


class FinalAnswerBlock(PlannerFinalModel):
    """【研究交付】【正文投影】可读性与关系有效性分别保存。"""

    block_index: int
    block_id: str = ""
    text: str
    content_kind: str = ""
    role: str = ""
    facet_ids: list[str] = Field(default_factory=list)
    section_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    valid: bool = False
    business_delivery: bool = False
    validation_errors: list[PlannerFinalValidationError] = Field(default_factory=list)


class FinalCitationBinding(PlannerFinalModel):
    """【研究交付】【引用位置】同次最终 Markdown 中 Unicode 码点半开区间。"""

    attempt_id: str
    block_id: str
    alias: str
    evidence_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=1)


class FinalDeliveryReason(PlannerFinalModel):
    """【研究交付】【原因】保留取得、最终生成、模型声明和历史未知的出处。"""

    source: str
    code: str
    message: str
    requirement_id: str = ""
    section_id: str = ""
    required: bool = True


class FinalRequirementDelivery(PlannerFinalModel):
    """【研究交付】【要求验收】从有效正文选择的真实产物派生交付事实。"""

    requirement_id: str
    required: bool
    status: Literal["delivered", "missing", "not_assessed"]
    block_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    reasons: list[FinalDeliveryReason] = Field(default_factory=list)


class FinalDeliveryReport(PlannerFinalModel):
    """【研究交付】【交付报告】各消费者共用交付结果，不改写取证与操作状态。"""

    contract_version: Literal["research-planner-final-v3"] = "research-planner-final-v3"
    delivery_policy_version: Literal["research-delivery-v4", "research-delivery-v5"] | None = None
    attempt_id: str
    entrypoint: str
    status: FinalDeliveryStatus
    generation_status: Literal["not_called", "completed", "failed"]
    requirement_results: list[FinalRequirementDelivery] = Field(default_factory=list)
    section_results: list[FinalRequirementDelivery] = Field(default_factory=list)
    validation_errors: list[PlannerFinalValidationError] = Field(default_factory=list)
    unresolved_items: list[FinalDeliveryReason] = Field(default_factory=list)
    reasons: list[FinalDeliveryReason] = Field(default_factory=list)
    valid_business_block_ids: list[str] = Field(default_factory=list)
    citation_bindings: list[FinalCitationBinding] = Field(default_factory=list)
    exposure: dict[str, Any] = Field(default_factory=dict)
    semantic_validation: Literal["not_assessed", "passed", "issues_found", "failed"] = "not_assessed"
    content_review: ResearchAnswerReview | None = None
    evidence_not_found: bool = False


@dataclass(frozen=True, kw_only=True)
class PlannerFinalAnswerInput:
    """【研究交付】【生成输入】同次提示、选择校验和报告使用相同可见投影。"""

    question: str
    plan: ResearchPlan
    unit_results: list[ResearchUnitResult]
    evidence_records: list[dict[str, Any]]
    attempt_id: str = field(default_factory=lambda: uuid4().hex)
    entrypoint: str = "normal"
    exposure: dict[str, Any] = field(default_factory=dict)
    context_text: str = ""
    historical_input_context: dict[str, Any] = field(default_factory=dict)
    eligibility_report: dict[str, Any] = field(default_factory=dict)
    revision_context: dict[str, Any] = field(default_factory=dict)
    delivery_policy_version: Literal["research-delivery-v4", "research-delivery-v5"] = "research-delivery-v4"


@dataclass(frozen=True, kw_only=True)
class PlannerFinalProviderOutput:
    """【研究交付】【模型边界】原始结构与解析错误一并交付，未提前过滤正文。"""

    payload: Any = None
    raw_text: str = ""
    parsing_error: str = ""
    expected_contract_version: str = ""
    source_payload: Any = None
    source_schema: str = ""


@dataclass(frozen=True, kw_only=True)
class PlannerFinalValidationOutcome:
    """【研究交付】【校验结果】正文、块、引用和交付报告属于同一尝试。"""

    status: PlannerFinalStatus
    answer: str
    confidence: str
    failure_reason: str
    blocks: list[FinalAnswerBlock]
    errors: list[PlannerFinalValidationError]
    delivery_report: FinalDeliveryReport


class PlannerFinalAnswerProvider(Protocol):
    async def generate(self, payload: PlannerFinalAnswerInput) -> PlannerFinalProviderOutput | dict[str, Any]:
        """参数为本次实际可见输入；返回原始 JSON 载体或明确解析失败。"""

    async def revise(self, payload: PlannerFinalAnswerInput, candidate: dict[str, Any]) -> PlannerFinalProviderOutput | dict[str, Any]:
        """参数为当前实际输入与唯一基稿；返回显式补丁，不重发无关正文。"""
