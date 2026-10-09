"""【研究交付】【内容核对合同】同次正文的研究要点与来源核对。作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

REVIEW_PROMPT_VERSION = "research-answer-review-v4"


class ReviewModel(BaseModel):
    """【研究交付】【核对边界】严格保存原始文本与模型判断，不清洗错误字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReviewSource(ReviewModel):
    """【研究交付】【核对依据】判断所依据的实际可见原文。"""

    evidence_id: str = Field(description="Copy the exact evidence_id of the source text being quoted from facts.sources.")
    quote: str = Field(min_length=1, description=(
        "A short, sufficient, contiguous verbatim substring of facts.sources[evidence_id].text. Preserve whitespace, "
        "Unicode, punctuation and LaTeX exactly. Never paraphrase, normalize or insert an ellipsis. "
        "For separate passages, use separate source entries with the same evidence_id; do not copy entire sections."))


class ResearchPointReview(ReviewModel):
    """【研究交付】【逐点核对】分别判断实质回答、未回答与无支持断言。"""

    facet_id: str
    verdict: Literal["answered", "missing", "unsupported"] = Field(description=(
        "answered means the requested content under the requested conditions is actually supplied; "
        "missing also applies when its absence is honestly explained or only a different protocol has results; "
        "unsupported means the answer makes a claim not supported by its sources."))
    explanation: str = Field(min_length=1)
    block_ids: list[str] = Field(description=(
        "Copy existing block IDs. For answered, every selected block must have business_delivery=true and "
        "include this facet_id in its facet_ids. Do not select gap or structure blocks as answered support; "
        "inspect them as context or reference them for missing/unsupported judgments."))
    claim_ids: list[str] = Field(description=(
        "Copy qualified claim IDs actually selected by the listed blocks. An answered claims requirement "
        "must select its supporting claims; do not borrow claims from unrelated blocks."))
    sources: list[ReviewSource] = Field(description=(
        "Exact source excerpts for this judgment. For answered factual/material points, provide sources from "
        "the selected blocks' evidence_ids or the selected claims' evidence_ids; at least one is required."))


class ResearchAnswerIssue(ReviewModel):
    """【研究交付】【正文问题】覆盖宽要点遗漏及无来源的新增断言。"""

    kind: Literal["missing_content", "unsupported_statement", "misrepresented_source"]
    facet_ids: list[str]
    block_ids: list[str]
    answer_quote: str = Field(description=(
        "Exact contiguous substring of the actual answer within a listed block's start/end offsets. "
        "Required for unsupported_statement or misrepresented_source; do not paraphrase or insert ellipses."))
    explanation: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    sources: list[ReviewSource]


class ResearchPartialBasis(ReviewModel):
    """【研究交付】【部分结束依据】区分论文明确限制与本轮读取、范围不足。"""

    kind: Literal["source_statement", "read_outcome", "current_scope"]
    explanation: str = Field(min_length=1)
    sources: list[ReviewSource]
    read_unit_ids: list[str]


class ResearchAnswerReviewDraft(ReviewModel):
    """【研究交付】【核对结论】模型给具体依据，程序决定报告和回边。"""

    points: list[ResearchPointReview]
    issues: list[ResearchAnswerIssue]
    recommendation: Literal["deliver", "revise_answer", "continue_research", "partial"]
    reason: str = Field(min_length=1)
    partial_basis: ResearchPartialBasis | None = None
    revision_scope: Literal["local", "whole"] | None = Field(default=None, description=(
        "Required when recommendation=revise_answer: local for located prose or omissions; whole only when "
        "the argument or overall structure needs rebuilding. Explain the scope in reason; never use block count."))


class ResearchAnswerReview(ReviewModel):
    """【研究交付】【核对记录】由程序绑定实际输入、正文及最终生成身份。"""

    prompt_version: str = REVIEW_PROMPT_VERSION
    attempt_id: str
    input_digest: str
    research_snapshot_digest: str
    answer_digest: str
    answer_length: int = Field(ge=0)
    delivery_answer_digest: str
    status: Literal["passed", "issues_found", "failed"]
    assessment: ResearchAnswerReviewDraft | None = None
    error: str = ""


@dataclass(frozen=True, kw_only=True)
class ResearchAnswerReviewInput:
    """【研究交付】【核对输入】仅包含本次实际成果与将交付的正文。"""

    attempt_id: str
    answer: str
    facts: dict[str, Any]
    exposure: dict[str, Any] = field(default_factory=dict)


class ResearchAnswerReviewer(Protocol):
    async def review(self, payload: ResearchAnswerReviewInput) -> ResearchAnswerReviewDraft:
        """参数为同次正文及真实材料；返回逐要点语义判断，禁止自行发起补查。"""
