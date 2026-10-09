"""证据真值、版本映射与阶段观察的不可变合同。作者：xxx。"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, StrictFloat, StrictInt,
    StrictStr, ValidationInfo, field_validator, model_validator,
)

from evals.contracts import MetricResult, read_snapshot, snapshot, snapshot_hash


def _nonblank(value: str) -> str:
    """参数为标识或名称；拒绝纯空白并返回未经改写的原值。"""
    if not value.strip():
        raise ValueError("identity must not be blank")
    return value


Identity = Annotated[StrictStr, AfterValidator(_nonblank)]
Checksum = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
Position = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
FiniteNumber = Annotated[StrictFloat, Field(allow_inf_nan=False)]
RankingStage = Literal["query_raw", "query_filtered", "ranked_hits"]
Stage = Literal["query_raw", "query_filtered", "merged_hits", "ranked_hits", "evidence_pool", "claim_input", "final_input"]
PayloadKind = Literal["child", "parent", "model_excerpt"]
Importance = Literal["critical", "important", "supporting"]
Normalization = Literal["exact-text-v1", "whitespace-v1"]
EMPTY_IDENTITY = "0" * 64


def require_unique(values: tuple[Any, ...], *, label: str) -> None:
    """参数为一组身份和错误名称；重复时抛错，无重复时返回 None。"""
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def ranking_unit_key(paper_id: str, chunk_id: str) -> str:
    """参数为论文和 child 身份；返回无分隔符碰撞的排序单位键。"""
    return json.dumps([_nonblank(paper_id), _nonblank(chunk_id)], ensure_ascii=False, separators=(",", ":"))


class FrozenContract(BaseModel):
    """字段不可变且拒绝额外字段；集合使用 tuple 保持深层不可变。"""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class SourceBlock(FrozenContract):
    """保存原文块身份、从 1 开始的页码、章节及实际正文。"""

    block_id: Identity
    page_num: PositiveInt
    section_path: StrictStr
    text: Annotated[StrictStr, Field(min_length=1)]


class SourceSnapshot(FrozenContract):
    """绑定原文产物和可重建内容；历史引擎版本未知时保留 None。"""

    schema_version: Literal["evidence-source-v1"] = "evidence-source-v1"
    snapshot_id: Checksum
    paper_id: Identity
    pdf_sha256: Checksum
    artifact_manifest_snapshot: StrictStr
    locator_version: Literal["exact-text-v1"] = "exact-text-v1"
    parser_engine_version: Identity | None = None
    blocks: tuple[SourceBlock, ...] = Field(min_length=1)

    @field_validator("artifact_manifest_snapshot")
    @classmethod
    def canonical_manifest(cls, value: str) -> str:
        """参数为产物 JSON；返回规范快照，非法对象或非有限数字抛错。"""
        return snapshot(read_snapshot(value))

    @model_validator(mode="after")
    def validate_identity(self, info: ValidationInfo[dict[str, bool]]) -> Self:
        """参数为校验上下文；验证块身份唯一和快照内容后返回自身。"""
        require_unique(tuple(block.block_id for block in self.blocks), label="source block")
        if not (info.context or {}).get("building_identity"):
            body = self.model_dump(mode="json", exclude={"snapshot_id"})
            if snapshot_hash(snapshot(body)) != self.snapshot_id:
                raise ValueError("source_snapshot_checksum_mismatch")
        return self


def make_source_snapshot(
    *, paper_id: str, pdf_sha256: str, artifact_manifest_snapshot: str,
    blocks: tuple[SourceBlock, ...], parser_engine_version: str | None = None,
) -> SourceSnapshot:
    """参数为真实产物和原文块；返回由完整内容计算身份的原文快照。"""
    draft = SourceSnapshot.model_validate({
        "snapshot_id": EMPTY_IDENTITY, "paper_id": paper_id, "pdf_sha256": pdf_sha256,
        "artifact_manifest_snapshot": artifact_manifest_snapshot, "blocks": blocks,
        "parser_engine_version": parser_engine_version,
    }, context={"building_identity": True})
    body = draft.model_dump(mode="json", exclude={"snapshot_id"})
    return SourceSnapshot.model_validate({**body, "snapshot_id": snapshot_hash(snapshot(body))})


class EvidenceSpan(FrozenContract):
    """在一个确定版本的原文块内定位证据摘录。"""

    span_id: Identity
    source_snapshot_id: Checksum
    block_id: Identity
    start: Position
    end: PositiveInt
    quote: Annotated[StrictStr, Field(min_length=1), AfterValidator(_nonblank)]
    quote_sha256: Checksum

    @model_validator(mode="after")
    def validate_quote(self) -> Self:
        """无额外参数；验证摘录校验和及有序字符区间，返回自身。"""
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("span_range_invalid")
        if snapshot_hash(self.quote) != self.quote_sha256:
            raise ValueError("span_checksum_mismatch")
        return self


class EvidenceProof(FrozenContract):
    """一份证明要求所有引用片段同时满足，不允许空证明或重复片段。"""

    span_ids: tuple[Identity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """无额外参数；验证片段引用唯一后返回自身。"""
        require_unique(self.span_ids, label="proof span")
        return self


class EvidenceGroup(FrozenContract):
    """独立信息需求可由任意一份完整证明满足，重要程度与排序等级分离。"""

    group_id: Identity
    importance: Importance
    proofs: tuple[EvidenceProof, ...] = Field(min_length=1)


class MappedRange(FrozenContract):
    """记录实际正文区间到原文区间；空白归一必须保存逐字符原文范围。"""

    source_snapshot_id: Checksum
    block_id: Identity
    source_start: Position
    source_end: PositiveInt
    payload_start: Position
    payload_end: PositiveInt
    normalization: Normalization = "exact-text-v1"
    character_ranges: tuple[tuple[Position, PositiveInt], ...] = ()

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """无额外参数；检查区间和字符映射结构，返回自身。"""
        if self.source_end <= self.source_start or self.payload_end <= self.payload_start:
            raise ValueError("mapping_range_invalid")
        if self.normalization == "exact-text-v1" and self.character_ranges:
            raise ValueError("exact_mapping_has_character_ranges")
        if self.normalization == "whitespace-v1" and len(self.character_ranges) != self.payload_end - self.payload_start:
            raise ValueError("normalization_character_count_mismatch")
        if any(left >= right for left, right in self.character_ranges):
            raise ValueError("normalization_character_range_invalid")
        return self


class IndexMapping(FrozenContract):
    """把某索引版本的实际 child、parent 或模型摘录映射到原文。"""

    index_snapshot_id: Identity
    paper_id: Identity
    chunk_id: Identity
    parent_id: Identity | None = None
    payload_kind: PayloadKind
    text_snapshot: StrictStr
    text_sha256: Checksum
    ranges: tuple[MappedRange, ...]

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        """无额外参数；验证真实文本校验和及父块身份后返回自身。"""
        if snapshot_hash(self.text_snapshot) != self.text_sha256:
            raise ValueError("index_payload_checksum_mismatch")
        if self.payload_kind == "parent" and self.parent_id is None:
            raise ValueError("parent_mapping_identity_missing")
        return self


class IndexUnit(FrozenContract):
    """冻结实际索引的一个 child 单位，用于验证排序标注全集。"""

    paper_id: Identity
    chunk_id: Identity
    parent_id: Identity | None = None
    text_snapshot: StrictStr
    text_sha256: Checksum

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        """无额外参数；验证索引正文校验和后返回自身。"""
        if snapshot_hash(self.text_snapshot) != self.text_sha256:
            raise ValueError("index_unit_checksum_mismatch")
        return self


def _index_identity_body(value: IndexSnapshot) -> dict[str, Any]:
    """参数为索引快照；返回排除自身身份引用的确定性内容。"""
    body = value.model_dump(mode="json", exclude={"index_snapshot_id"})
    body["mappings"] = [mapping.model_dump(mode="json", exclude={"index_snapshot_id"}) for mapping in value.mappings]
    return body


class IndexSnapshot(FrozenContract):
    """保存版本化 child 全集与定位映射；离线和线上来源独立命名。"""

    schema_version: Literal["evidence-index-v1"] = "evidence-index-v1"
    index_snapshot_id: Checksum
    source_snapshot_ids: tuple[Checksum, ...] = Field(min_length=1)
    retrieval_backend: Identity
    units: tuple[IndexUnit, ...]
    mappings: tuple[IndexMapping, ...] = ()

    @model_validator(mode="after")
    def validate_identity(self, info: ValidationInfo[dict[str, bool]]) -> Self:
        """参数为校验上下文；检查索引单位、映射身份及完整内容版本。"""
        require_unique(self.source_snapshot_ids, label="index source")
        keys = tuple(ranking_unit_key(unit.paper_id, unit.chunk_id) for unit in self.units)
        require_unique(keys, label="index unit")
        mapping_keys = tuple((item.paper_id, item.chunk_id, item.parent_id, item.payload_kind, item.text_sha256) for item in self.mappings)
        require_unique(mapping_keys, label="index mapping")
        if (info.context or {}).get("building_identity"):
            return self
        if any(item.index_snapshot_id != self.index_snapshot_id for item in self.mappings):
            raise ValueError("mapping_index_version_mismatch")
        if snapshot_hash(snapshot(_index_identity_body(self))) != self.index_snapshot_id:
            raise ValueError("index_snapshot_checksum_mismatch")
        return self


def make_index_snapshot(
    *, source_snapshot_ids: tuple[str, ...], retrieval_backend: str,
    units: tuple[IndexUnit, ...], mappings: tuple[IndexMapping, ...] = (),
) -> IndexSnapshot:
    """参数为实际索引全集及映射；生成新内容身份，不改写传入映射。"""
    draft = IndexSnapshot.model_validate({
        "index_snapshot_id": EMPTY_IDENTITY, "source_snapshot_ids": source_snapshot_ids,
        "retrieval_backend": retrieval_backend, "units": units, "mappings": mappings,
    }, context={"building_identity": True})
    identity = snapshot_hash(snapshot(_index_identity_body(draft)))
    body = draft.model_dump(mode="json")
    body["index_snapshot_id"] = identity
    body["mappings"] = [{**item.model_dump(mode="json"), "index_snapshot_id": identity} for item in mappings]
    return IndexSnapshot.model_validate(body)


class RankingPolicy(FrozenContract):
    """在运行前固定阶段、child 排序单位、k 和有限递增等级增益。"""

    version: Literal["evidence-ranking-v1"] = "evidence-ranking-v1"
    stage: RankingStage
    ranking_unit: Literal["child_chunk"] = "child_chunk"
    top_k: PositiveInt
    gains: tuple[Annotated[FiniteNumber, Field(ge=0)], ...] = (0.0, 1.0, 3.0, 7.0)

    @model_validator(mode="after")
    def validate_gains(self) -> Self:
        """无额外参数；拒绝空增益、非零基线及非递增等级，返回自身。"""
        if len(self.gains) < 2 or self.gains[0] != 0.0:
            raise ValueError("ranking_gains_require_zero_and_positive_levels")
        if any(right <= left for left, right in zip(self.gains, self.gains[1:])):
            raise ValueError("ranking_gains_must_increase")
        return self


class RelevanceJudgment(FrozenContract):
    """单个实际排序单位的独立相关性等级。"""

    unit_key: Identity
    relevance: Position


class RankingJudgments(FrozenContract):
    """将完整排序标注绑定具体查询、阶段及索引版本。"""

    judgment_id: Identity
    index_snapshot_id: Checksum
    stage: RankingStage
    query: StrictStr
    ranking_unit: Literal["child_chunk"] = "child_chunk"
    labels: tuple[RelevanceJudgment, ...]

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        """无额外参数；拒绝同一排序单位被重复标注，返回自身。"""
        require_unique(tuple(item.unit_key for item in self.labels), label="ranking judgment unit")
        return self


class PaperRequirement(FrozenContract):
    """显式声明一篇论文必须满足的证据组，各引用只允许属于该论文。"""

    paper_id: Identity
    groups: tuple[EvidenceGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        """无额外参数；验证该论文的组要求不重复，返回自身。"""
        require_unique(tuple(group.group_id for group in self.groups), label="paper evidence group")
        return self


class SectionTarget(FrozenContract):
    """显式声明章节诊断的论文和章节名称。"""

    paper_id: Identity
    section_path: Identity


ExistingEvidenceMetric = Literal[
    "evidence_group_recall_at_k", "evidence_group_pool_coverage",
    "evidence_group_context_coverage", "critical_evidence_recall",
    "section_recall", "cross_paper_coverage", "mrr", "ndcg_at_k", "scope_adherence",
]
EvidenceGradingPolicy = Literal["reviewed", "candidate_diagnostic"]
CORE_EVIDENCE_METRICS: tuple[ExistingEvidenceMetric, ...] = (
    "evidence_group_recall_at_k", "evidence_group_pool_coverage",
    "evidence_group_context_coverage", "critical_evidence_recall",
    "cross_paper_coverage", "scope_adherence",
)


class EvidenceMeasurementPolicy(FrozenContract):
    """记录事前启用的已有指标；缺失观察不能通过事后删指标消失。"""

    version: Literal["evidence-measurement-v1"] = "evidence-measurement-v1"
    enabled_metrics: tuple[ExistingEvidenceMetric, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """无额外参数；拒绝重复启用项及关闭实际范围检查，返回自身。"""
        require_unique(self.enabled_metrics, label="enabled metric")
        if "scope_adherence" not in self.enabled_metrics:
            raise ValueError("scope_adherence_must_remain_enabled")
        return self


class EvidenceBenchmark(FrozenContract):
    """独立 Evidence 真值区，候选复核状态与来源版本均显式保留。"""

    schema_version: Literal["evidence-benchmark-v1"] = "evidence-benchmark-v1"
    dataset_version: Identity
    task_family: Identity | None = None
    review_status: Literal["candidate", "reviewed", "rejected"]
    sources: tuple[SourceSnapshot, ...]
    spans: tuple[EvidenceSpan, ...]
    groups: tuple[EvidenceGroup, ...]
    target_paper_ids: tuple[Identity, ...] = ()
    paper_requirements: tuple[PaperRequirement, ...] | None = None
    target_sections: tuple[SectionTarget, ...] | None = None
    index_snapshots: tuple[IndexSnapshot, ...] = ()
    ranking_policy: RankingPolicy | None = None
    ranking_judgments: tuple[RankingJudgments, ...] = ()
    measurement_policy: EvidenceMeasurementPolicy | None = None

    @model_validator(mode="after")
    def validate_identities(self) -> Self:
        """无额外参数；验证同类标识唯一，引用和文本校验由 alignment 执行。"""
        identities = {
            "source": tuple(item.snapshot_id for item in self.sources),
            "span": tuple(item.span_id for item in self.spans),
            "group": tuple(item.group_id for item in self.groups),
            "index": tuple(item.index_snapshot_id for item in self.index_snapshots),
            "judgment": tuple(item.judgment_id for item in self.ranking_judgments),
            "target paper": self.target_paper_ids,
            "paper requirement": tuple(item.paper_id for item in self.paper_requirements or ()),
            "section target": tuple((item.paper_id, item.section_path) for item in self.target_sections or ()),
        }
        for label, values in identities.items():
            require_unique(values, label=label)
        return self


class EffectiveScope(FrozenContract):
    """冻结实际生效范围；缺失和无范围约束不能解释为同一种状态。"""

    status: Literal["constrained", "unconstrained", "missing"]
    allowed_paper_ids: tuple[Identity, ...] = ()

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        """无额外参数；只有 constrained 可携带允许论文集合，返回自身。"""
        require_unique(self.allowed_paper_ids, label="scope paper")
        if self.status != "constrained" and self.allowed_paper_ids:
            raise ValueError("unconstrained_or_missing_scope_has_allowed_ids")
        return self


class ObservedItem(FrozenContract):
    """只保存该阶段实际出现的正文和身份；缺字段保持缺失。"""

    paper_id: Identity | None = None
    chunk_id: Identity | None = None
    parent_id: Identity | None = None
    payload_kind: PayloadKind
    text_snapshot: StrictStr | None = None
    rank: PositiveInt | None = None
    score: FiniteNumber | None = None
    evidence_id: Identity | None = None
    index_snapshot_id: Identity | None = None


class StageObservation(FrozenContract):
    """一次真实阶段或调用的独立观察，成功空结果与失败分别表达。"""

    stage: Stage
    call_id: Identity
    scope_id: Identity | None = None
    parent_scope_id: Identity | None = None
    unit_id: Identity | None = None
    query: StrictStr | None = None
    retrieval_backend: Identity
    status: Literal["complete", "missing", "failed"]
    reason: Identity | None = None
    items: tuple[ObservedItem, ...] | None = None
    scope: EffectiveScope = EffectiveScope(status="missing")
    index_snapshot_id: Identity | None = None


class RankedUnit(FrozenContract):
    """真实原始位置和单位键；重复单位仍占用原位置。"""

    rank: PositiveInt
    unit_key: Identity


class StageGrade(FrozenContract):
    """独立派生的阶段结果；保留真实覆盖身份及不能平均掉的越界事件。"""

    measurements: tuple[MetricResult, ...]
    covered_span_ids: tuple[Identity, ...] = ()
    hit_group_ids: tuple[Identity, ...] = ()
    scope_violations: tuple[Identity, ...] = ()
