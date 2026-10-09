"""【研究循环】【当前动作】模型只选择当前具备输入的动作。作者：xxx。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.models.research_planner_contracts import EvidenceGranularity, MaterialSelection, PreferredBlockType, TurnClarificationRequest
from app.models.research_planner_draft_contracts import (
    AnswerSectionRequirementDraft, RequestedFacetDraft, StrategyContractModel,
)


NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]


class ReadSection(StrategyContractModel):
    """【研究动作】【章节正文】只提供章节定位参数，不暴露图线索开关。"""

    source_type: Literal["local_section_text"]
    target_paper_ids: list[NonemptyText] = Field(min_length=1)
    required_sections: list[NonemptyText] = Field(default_factory=list)
    continuation_ids: list[NonemptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_one_locator(self) -> "ReadSection":
        """参数为章节动作；首次章节与已有续读身份二选一，避免两份定位互相覆盖。"""
        if bool(self.required_sections) == bool(self.continuation_ids):
            raise ValueError("section read requires either required_sections or continuation_ids")
        return self


class SearchText(StrategyContractModel):
    """【研究动作】【正文检索】查询与上下文粒度由模型选择。"""

    source_type: Literal["local_direct_text"]
    target_paper_ids: list[NonemptyText] = Field(default_factory=list)
    locator_queries: list[NonemptyText] = Field(min_length=1)
    evidence_granularity: EvidenceGranularity = "child_chunk"
    preferred_block_types: list[PreferredBlockType] = Field(default_factory=list)


class StructureHint(StrategyContractModel):
    """【研究动作】【图线索】只有此动作可以要求回原文核实。"""

    source_type: Literal["local_structure_hint"]
    target_paper_ids: list[NonemptyText] = Field(default_factory=list)
    locator_queries: list[NonemptyText] = Field(min_length=1)
    verify_with_direct_text: bool = False


class LocateMetadata(StrategyContractModel):
    """【研究动作】【元数据定位】筛选条件只用于元数据读取。"""

    source_type: Literal["local_chunk_metadata"]
    target_paper_ids: list[NonemptyText] = Field(default_factory=list)
    required_sections: list[NonemptyText] = Field(default_factory=list)
    required_block_types: list[PreferredBlockType] = Field(default_factory=list)
    required_pages: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_filter(self) -> "LocateMetadata":
        """参数为定位动作；返回具有真实筛选条件的请求，否则明确报错。"""
        if not (self.required_sections or self.required_block_types or self.required_pages):
            raise ValueError("metadata action requires section, block type or page filters")
        return self


class ReadStructure(StrategyContractModel):
    """【研究动作】【章节目录】保留现有目录能力，不升级为正文。"""

    source_type: Literal["local_section_structure"]
    target_paper_ids: list[NonemptyText] = Field(min_length=1)


class ExternalGap(StrategyContractModel):
    """【研究动作】【外部缺口】登记现有能力无法取得的材料，不假装联网。"""

    source_type: Literal["external_context_needed"]
    target_paper_ids: list[NonemptyText] = Field(default_factory=list)
    locator_queries: list[NonemptyText] = Field(default_factory=list)


class ReadMetadata(StrategyContractModel):
    """【研究动作】【书目信息】沿用本地论文元数据能力。"""

    source_type: Literal["local_metadata"]
    target_paper_ids: list[NonemptyText] = Field(default_factory=list)
    locator_queries: list[NonemptyText] = Field(default_factory=list)


class AnalysisDependencyBase(StrategyContractModel):
    """【研究动作】【上游选择】共同字段只表达已返回单元、必需性与分析用途。"""

    unit_id: NonemptyText = Field(description="Identity of an already returned unit result; never a future action.")
    required: bool = Field(description="Whether this product must be available before analysis can proceed.")
    reason: str = ""


class MaterialsDependency(AnalysisDependencyBase):
    """【研究动作】【材料依赖】读取真实材料，不预先承诺材料会产出何种结论。"""

    required_product: Literal["materials"]
    selections: list[MaterialSelection] | None = Field(default=None, min_length=1)


class ClaimsDependency(AnalysisDependencyBase):
    """【研究动作】【结论依赖】仅选择已具备支持与验证资格的事实或证据推断。"""

    required_product: Literal["claims"]
    required_claim_types: list[Literal["paper_supported", "evidence_grounded_inference"]] = Field(
        default_factory=list, description="Required fact types; empty means no additional type requirement.")


class HypothesesDependency(AnalysisDependencyBase):
    """【研究动作】【假设依赖】先验假设与记忆延续保留待验证性质，不升级为论文事实。"""

    required_product: Literal["hypotheses"]
    required_claim_types: list[Literal["model_prior_hypothesis", "memory_continuation"]] = Field(
        default_factory=list, description="Required hypothesis types; empty means no additional type requirement.")


ActionDependency = Annotated[
    MaterialsDependency | ClaimsDependency | HypothesesDependency,
    Field(discriminator="required_product"),
]


class AnalyzeMaterials(StrategyContractModel):
    """【研究动作】【分析成果】依赖只能引用已返回的成果，事实资格由支持链决定。"""

    source_type: Literal["inference_from_units"]
    dependencies: list[ActionDependency] = Field(min_length=1)
    output_role: Literal["synthesis", "final_recommendation"] = "synthesis"
    analysis_mode: Literal["grounded", "hypothesis", "memory"] = "grounded"


ActionInput = Annotated[
    ReadSection | SearchText | StructureHint | LocateMetadata | ReadStructure | ReadMetadata | ExternalGap | AnalyzeMaterials,
    Field(discriminator="source_type"),
]


class ResearchAction(StrategyContractModel):
    """【研究动作】【业务目的】身份由系统分配，模型只描述目的、输入与对应问题。"""

    goal: str = Field(min_length=1, pattern=r"\S")
    input: ActionInput
    covered_facet_ids: list[NonemptyText] = Field(default_factory=list)


class ExecuteActions(StrategyContractModel):
    """【研究决策】【执行】本批次动作互不依赖，依赖输入须已取得。"""

    kind: Literal["execute"]
    actions: list[ResearchAction] = Field(min_length=1)


class FinishResearch(StrategyContractModel):
    """【研究决策】【交付提议】完整交付仍检查缺口，部分交付必须明确选择。"""

    kind: Literal["finish"]
    allow_partial: bool = False
    reason: str = Field(min_length=1, pattern=r"\S")


class ClarifyResearch(StrategyContractModel):
    """【研究决策】【澄清】沿用用户问题合同，不编造默认澄清句。"""

    kind: Literal["clarify"]
    clarification: TurnClarificationRequest

    @model_validator(mode="after")
    def require_question(self) -> "ClarifyResearch":
        """参数为澄清提议；仅接受明确需要用户回答的请求。"""
        if not self.clarification.needed:
            raise ValueError("clarify requires needed=true")
        return self


class ResearchDecision(StrategyContractModel):
    """【研究循环】【后续决策】已接受要求由系统回填，模型不能在此改写。"""

    decision: Annotated[ExecuteActions | FinishResearch | ClarifyResearch, Field(discriminator="kind")]
    research_intent: str = Field(default="", description=(
        "Optional future research outline at the root of the complete response, outside decision. "
        "It is explanatory, not an action or progress, and need not be repeated on every decision."
    ))


class InitialResearchDecision(ResearchDecision):
    """【研究循环】【首次决策】识别问题要求与当前动作，不承诺全部未来步骤。"""

    requested_facets: list[RequestedFacetDraft] = Field(description=(
        "Establish task requirements on the first decision only; later decisions use the server-accepted requirements."
    ))
    answer_sections: list[AnswerSectionRequirementDraft] = Field(default_factory=list, description=(
        "Optional answer organization on the first decision only; do not resubmit on later decisions."
    ))

    @model_validator(mode="after")
    def require_task(self) -> "InitialResearchDecision":
        """参数为首次决策；执行或交付必须具有真实任务要求，澄清可暂不确定。"""
        if self.decision.kind != "clarify" and not self.requested_facets:
            raise ValueError("research requires requested_facets")
        return self


# 【研究决策】【持久边界】以上类型保留原草稿身份；实时模型只使用下方不含额外思路的输出合同
class ResearchDecisionOutput(StrategyContractModel):
    """【研究决策】【当前输出】只选择动作、交付或澄清，稳定要求由程序保留。"""

    decision: Annotated[ExecuteActions | FinishResearch | ClarifyResearch, Field(discriminator="kind")]


class InitialResearchDecisionOutput(ResearchDecisionOutput):
    """【研究决策】【首次输出】建立真实要求并选择当前动作，不另写未来研究思路。"""

    requested_facets: list[RequestedFacetDraft] = Field(description=(
        "Establish task requirements on the first decision only; later decisions use the server-accepted requirements."
    ))
    answer_sections: list[AnswerSectionRequirementDraft] = Field(default_factory=list, description=(
        "Optional answer organization on the first decision only; do not resubmit on later decisions."
    ))

    @model_validator(mode="after")
    def require_task(self) -> "InitialResearchDecisionOutput":
        """参数为首次输出；执行或交付须有要求，澄清可暂不确定，返回原对象。"""
        if self.decision.kind != "clarify" and not self.requested_facets:
            raise ValueError("research requires requested_facets")
        return self
