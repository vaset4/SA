"""【研究交付】【正文关系】分别保留可读正文和校验后的引用关系。作者：xxx。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from markdown_it import MarkdownIt
from pydantic import TypeAdapter, ValidationError

from app.models.research_planner_contracts import ConsumerExposure, EvidenceExposure
from app.services.internal_research.research_planner_consumer_budget import content_hash, evidence_version
from app.services.internal_research.research_planner_final_contracts import (
    FinalAnswerBlock, FinalAnswerBlockDraft, FinalCitationBinding, FinalSectionPresentation, PlannerFinalAnswerInput,
    FinalProductBlockDraft, PlannerFinalErrorCode, PlannerFinalValidationError,
)
from app.services.internal_research.research_planner_final_evidence import FinalProductCatalog

_CODE = re.compile(r"(?ms)^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?P=fence)[ \t]*(?=\n|$)|(?P<tick>`+).*?(?P=tick)")
_MODEL_CITATION = re.compile(r"\[E\d+(?:\s*[,，]\s*E?\d+)*\]|#evidence-[^\s)]*")
_NON_BUSINESS_KINDS = frozenset({"structure", "gap"})
_PRODUCT_ADAPTER = TypeAdapter(FinalProductBlockDraft)
_CLAIM_CONTENT_KINDS = {"paper_supported": "fact", "evidence_grounded_inference": "inference",
                        "model_prior_hypothesis": "hypothesis", "memory_continuation": "history"}


def normalize_product_blocks(raw_blocks: list[Any], catalog: FinalProductCatalog, *, attempt_id: str
                             ) -> tuple[list[Any], list[PlannerFinalValidationError]]:
    """参数为新版原块、合法目录及候选身份；返回规范化块和错误，保留原文字与原位置。"""
    normalized: list[Any] = []
    errors: list[PlannerFinalValidationError] = []
    for index, raw in enumerate(raw_blocks):
        block_id = f"{attempt_id}:{index}"
        try:
            draft = _PRODUCT_ADAPTER.validate_python(raw)
        except ValidationError as exc:
            normalized.append({**raw, "block_id": block_id} if isinstance(raw, dict) else raw)
            errors.append(final_error("invalid_block_field", str(exc), index=index, block_id=block_id))
            continue
        block = {"block_id": block_id, "text": draft.text, "facet_ids": draft.facet_ids,
                 "section_ids": draft.section_ids, "claim_ids": [], "evidence_ids": [],
                 "content_kind": draft.product_kind}
        if draft.product_kind == "claims":
            kinds = set()
            evidence = set()
            for claim_id in draft.claim_ids:
                support = catalog.supports.get(claim_id)
                if claim_id in catalog.historical_claims:
                    kinds.add("history")
                elif support is not None:
                    kinds.add(_CLAIM_CONTENT_KINDS.get(support.claim.claim_type, "history"))
                    evidence.update(support.citable_evidence_ids)
            if len(kinds) > 1:
                errors.append(final_error("invalid_content_identity", "不同类型的主张必须拆为各自有支持的正文块",
                                          index=index, block_id=block_id))
            block.update(claim_ids=draft.claim_ids, evidence_ids=sorted(evidence),
                         content_kind=next(iter(kinds)) if len(kinds) == 1 else "fact")
        elif draft.product_kind == "materials":
            block["evidence_ids"] = draft.evidence_ids
        normalized.append(block)
    return normalized, errors


@dataclass(frozen=True, kw_only=True)
class _BlockContext:
    """【研究交付】【关系上下文】同次要求、可见性和来源只准备一次。"""

    payload: PlannerFinalAnswerInput
    catalog: FinalProductCatalog
    visible_evidence: dict[str, EvidenceExposure]
    visible_claims: frozenset[str]
    facets: frozenset[str]
    sections: frozenset[str]


def final_error(code: PlannerFinalErrorCode, message: str, *, index: int | None = None,
                block_id: str = "", field: str = "") -> PlannerFinalValidationError:
    """参数为错误类型、原因及原块位置；返回可序列化的问题，不修改模型正文。"""
    return PlannerFinalValidationError(code=code, message=message, block_index=index,
                                       block_id=block_id, field=field)


def validate_final_blocks(payload: PlannerFinalAnswerInput, raw_blocks: list[Any],
                          catalog: FinalProductCatalog) -> tuple[list[FinalAnswerBlock], list[PlannerFinalValidationError]]:
    """参数为同次输入、原块列表和合法成果；返回全部可读块及逐位置错误。"""
    exposure = ConsumerExposure.model_validate(payload.exposure)
    facets = [*payload.plan.coverage_review.explicit_facets, *payload.plan.coverage_review.inferred_facets]
    context = _BlockContext(payload=payload, catalog=catalog,
                            visible_evidence={item.evidence_id: item for item in exposure.evidence},
                            visible_claims=frozenset(item.claim_id for item in exposure.claims),
                            facets=frozenset(item.facet_id for item in facets),
                            sections=frozenset(item.section_id for item in payload.plan.coverage_review.answer_sections))
    counts = Counter[str](item["block_id"] for item in raw_blocks
                     if isinstance(item, dict) and isinstance(item.get("block_id"), str))
    blocks: list[FinalAnswerBlock] = []
    errors: list[PlannerFinalValidationError] = []
    for index, raw in enumerate(raw_blocks):
        block, problems = _read_block(raw, index, context, counts)
        errors.extend(problems)
        if block is not None:
            blocks.append(block)
    return blocks, errors


def _read_block(raw: Any, index: int, context: _BlockContext,
                counts: Counter[str]) -> tuple[FinalAnswerBlock | None, list[PlannerFinalValidationError]]:
    """参数为原块、位置和关系上下文；先保存可读文字，再验证关联字段。"""
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str) or not raw["text"].strip():
        return None, [final_error("invalid_block", "正文块缺少非空字符串正文", index=index, field="text")]
    raw_id = raw.get("block_id")
    block_id = raw_id if isinstance(raw_id, str) else ""
    try:
        draft = FinalAnswerBlockDraft.model_validate(raw)
    except ValidationError as exc:
        errors = [final_error("invalid_block_field", str(exc), index=index, block_id=block_id)]
        return FinalAnswerBlock(block_index=index, block_id=block_id, text=raw["text"],
                                validation_errors=errors), errors
    errors = _relation_errors(draft, context)
    if not draft.block_id.strip():
        errors.append(final_error("invalid_block_field", "正文块身份不能是空白", field="block_id"))
    if counts[draft.block_id] > 1:
        errors.append(final_error("duplicate_block_id", "正文块身份重复，不能唯一绑定引用", block_id=draft.block_id))
    errors = [error.model_copy(update={"block_index": index, "block_id": draft.block_id}) for error in errors]
    valid = not errors
    return FinalAnswerBlock(
        block_index=index, **draft.model_dump(), role=draft.content_kind, valid=valid,
        business_delivery=valid and _has_business_body(draft), validation_errors=errors,
    ), errors


def _relation_errors(block: FinalAnswerBlockDraft, context: _BlockContext) -> list[PlannerFinalValidationError]:
    """参数为可解析块和同次上下文；返回要求、内容身份、来源与自写标记错误。"""
    errors: list[PlannerFinalValidationError] = []
    if set(block.facet_ids) - context.facets:
        errors.append(final_error("unknown_facet_id", "正文关联了不存在的任务要求"))
    if set(block.section_ids) - context.sections:
        errors.append(final_error("unknown_section_id", "正文关联了不存在的最终章节"))
    if _MODEL_CITATION.search(_CODE.sub("", block.text)):
        errors.append(final_error("model_written_citation", "正文含模型自写引用标记，未获得来源绑定"))
    if block.content_kind in _NON_BUSINESS_KINDS:
        if block.claim_ids or block.evidence_ids:
            errors.append(final_error("invalid_content_identity", "结构或缺口说明不能声明业务支持关系"))
        return errors
    if not block.facet_ids:
        errors.append(final_error("unassociated_product", "业务正文未关联实际任务要求"))
    if block.content_kind == "materials":
        errors.extend(_material_errors(block, context))
    else:
        errors.extend(_claim_errors(block, context))
    return errors


def _material_errors(block: FinalAnswerBlockDraft, context: _BlockContext) -> list[PlannerFinalValidationError]:
    """参数为材料块和当前目录；材料必须真实取得且实际送模，不凭存在升级为事实。"""
    errors: list[PlannerFinalValidationError] = []
    if block.claim_ids or not block.evidence_ids:
        errors.append(final_error("invalid_content_identity", "材料正文应选择材料，不能代替主张正文"))
    for evidence_id in block.evidence_ids:
        if evidence_id not in context.catalog.materials:
            errors.append(final_error("unknown_evidence_id", f"材料不属于当前任务产物：{evidence_id}"))
        elif evidence_id not in context.visible_evidence:
            errors.append(final_error("material_not_visible", f"本次最终输入未展示材料：{evidence_id}"))
        elif not set(block.facet_ids) & context.catalog.material_facets[evidence_id]:
            errors.append(final_error("unassociated_product", f"材料未关联该正文要求：{evidence_id}"))
        else:
            visible = context.visible_evidence[evidence_id]
            record = context.catalog.materials[evidence_id]
            body = record.get(visible.field)
            if (visible.evidence_version != evidence_version(record) or not isinstance(body, str)
                    or visible.end > len(body) or content_hash(body[visible.start:visible.end]) != visible.content_hash):
                errors.append(final_error("material_not_visible", f"本次材料内容与送模凭据不符：{evidence_id}"))
    return errors


def _claim_errors(block: FinalAnswerBlockDraft, context: _BlockContext) -> list[PlannerFinalValidationError]:
    """参数为主张正文及共享闭包；只接受明确选择且符合内容身份的主张来源。"""
    errors: list[PlannerFinalValidationError] = []
    allowed: set[str] = set()
    required_citations: set[str] = set()
    if not block.claim_ids:
        errors.append(final_error("missing_block_claim_ids", "主张正文没有选择其依据"))
    for claim_id in block.claim_ids:
        problems, sources, required = _claim_selection(claim_id, block, context)
        errors.extend(problems)
        allowed.update(sources)
        required_citations.update(required)
    if set(block.evidence_ids) - allowed:
        errors.append(final_error("evidence_id_not_allowed_for_block", "正文选择的证据不属于这些主张的支持链"))
    if required_citations - set(block.evidence_ids):
        errors.append(final_error("missing_support_citations", "论文正文缺少所选主张支持链的引用"))
    return errors


def _claim_selection(claim_id: str, block: FinalAnswerBlockDraft, context: _BlockContext
                     ) -> tuple[list[PlannerFinalValidationError], set[str], set[str]]:
    """参数为单条选择、正文和上下文；返回关系错误、允许来源及必须呈现的论文来源。"""
    legacy = context.catalog.historical_claims.get(claim_id)
    if legacy is not None and block.content_kind == "history":
        if claim_id not in context.visible_claims:
            return [final_error("claim_not_visible", f"本次未展示历史主张：{claim_id}")], set(), set()
        return [], set(legacy.supporting_evidence_ids), set()
    support = context.catalog.supports.get(claim_id)
    if support is None:
        return [final_error("unknown_claim_id", f"主张不存在：{claim_id}")], set(), set()
    errors: list[PlannerFinalValidationError] = []
    if not support.qualified:
        errors.append(final_error("unvalidated_claim_id", f"主张当前不合格：{claim_id}；{','.join(support.reasons)}"))
    if claim_id not in context.visible_claims:
        errors.append(final_error("claim_not_visible", f"本次最终输入未展示主张：{claim_id}"))
    expected = {"paper_supported": "fact", "evidence_grounded_inference": "inference",
                "model_prior_hypothesis": "hypothesis", "memory_continuation": "history"}
    if expected.get(support.claim.claim_type) != block.content_kind:
        errors.append(final_error("invalid_content_identity", f"正文与主张产物身份不一致：{claim_id}"))
    if not set(block.facet_ids) & set(support.claim.addressed_facet_ids):
        errors.append(final_error("unassociated_product", f"主张未回应该正文关联的要求：{claim_id}"))
    return errors, set(support.evidence_ids), set(support.citable_evidence_ids)


def _has_business_body(block: FinalAnswerBlockDraft) -> bool:
    """参数为已通过关系校验的块；返回是否含实际业务正文，空标题和缺口不能完成任务。"""
    if block.content_kind in _NON_BUSINESS_KINDS:
        return False
    tokens = MarkdownIt("commonmark").parse(block.text)
    for index, token in enumerate(tokens):
        if token.type not in {"inline", "fence", "code_block"} or not token.content.strip():
            continue
        if index > 0 and tokens[index - 1].type == "heading_open":
            continue
        return True
    return False


def render_final_blocks(blocks: list[FinalAnswerBlock], attempt_id: str, *, citable_evidence_ids: frozenset[str]
                        ) -> tuple[str, list[FinalCitationBinding], FinalSectionPresentation]:
    """参数为全部可读块和尝试身份；返回同次Markdown、引用坐标及实际分节的有效正文块索引。"""
    answer = ""
    aliases: dict[str, str] = {}
    bindings: list[FinalCitationBinding] = []
    spans: dict[int, tuple[int, int]] = {}
    for block in blocks:
        if answer:
            answer += "\n\n"
        start = len(answer)
        answer += block.text
        spans[block.block_index] = (start, len(answer))
        cited_ids = [eid for eid in dict.fromkeys(block.evidence_ids) if eid in citable_evidence_ids]
        if not block.valid or not cited_ids:
            continue
        answer += "\n\n来源："
        for index, evidence_id in enumerate(cited_ids):
            if index:
                answer += " "
            alias = aliases.setdefault(evidence_id, f"E{len(aliases) + 1}")
            marker = f"[{alias}]"
            start = len(answer)
            answer += marker
            bindings.append(FinalCitationBinding(attempt_id=attempt_id, block_id=block.block_id,
                            alias=alias, evidence_id=evidence_id, start=start, end=len(answer)))
    return answer, bindings, _rendered_sections(answer, blocks, spans)


def _rendered_sections(answer: str, blocks: list[FinalAnswerBlock], spans: dict[int, tuple[int, int]]
                       ) -> FinalSectionPresentation:
    """参数为最终正文、原块及其坐标；解析一次标准Markdown，将真实标题下正文映射回章节身份。"""
    # 1. 【研究交付】【分节定位】只消费标准解析器的顶层标题，代码和引文中的标题不承担用户展示义务
    tokens = MarkdownIt("commonmark").parse(answer)
    line_offsets = [0, *(match.end() for match in re.finditer(r"\r\n|\r|\n", answer)), len(answer)]
    active: list[tuple[int, str, frozenset[str]]] = []
    presented: dict[str, dict[str, set[int]]] = {}
    for index, token in enumerate(tokens):
        if token.map is None:
            continue
        start, end = (line_offsets[line] for line in token.map)
        owners = [block for block in blocks if block.valid
                  and spans[block.block_index][0] < end and spans[block.block_index][1] > start]
        if token.type == "heading_open" and token.level == 0:
            level = int(token.tag[1:])
            title = "".join(child.content for child in tokens[index + 1].children or []
                            if child.type in {"text", "code_inline"}).strip()
            section_ids = frozenset(sid for block in owners for sid in block.section_ids)
            active = [heading for heading in active if heading[0] < level]
            active.append((level, title, section_ids))
            continue
        # 2. 【研究交付】【展示正文】标题文字和服务端来源行不计正文，有效结构/缺口说明仅获得展示位置
        if token.type not in {"inline", "fence", "code_block"} or not token.content.strip():
            continue
        if index > 0 and tokens[index - 1].type == "heading_open":
            continue
        for _, title, section_ids in active:
            for section_id in section_ids:
                indices = {block.block_index for block in owners
                           if section_id in block.section_ids}
                presented.setdefault(section_id, {}).setdefault(title, set()).update(indices)
    return {sid: {title: frozenset(indices) for title, indices in headings.items()}
            for sid, headings in presented.items()}
