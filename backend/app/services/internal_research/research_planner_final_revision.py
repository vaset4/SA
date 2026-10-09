"""【研究交付】【正文修复】同基稿的局部补丁与候选版本管理。作者：xxx。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from app.services.internal_research.research_planner_consumer_budget import content_hash
from app.services.internal_research.research_planner_final_blocks import normalize_product_blocks, validate_final_blocks
from app.services.internal_research.research_planner_final_contracts import (
    FINAL_CONTRACT_VERSION, FINAL_DRAFT_VERSION, FinalAnswerBlock, FinalUnresolvedDraft, PlannerFinalAnswerInput, PlannerFinalModel,
    PlannerFinalValidationOutcome,
)
from app.services.internal_research.research_planner_final_evidence import final_product_catalog

PATCH_VERSION = "research-planner-final-patch-v1"
_BLOCK_FIELDS = ("block_id", "text", "content_kind", "facet_ids", "section_ids", "claim_ids", "evidence_ids")


class FinalRevisionRejected(ValueError):
    """【研究交付】【补丁拒绝】收到的修复不符合当前基稿合同，区别于模型调用失败。"""


class ReplaceBlocks(PlannerFinalModel):
    """【研究交付】【替换操作】目标为原稿中按原顺序连续排列的块，可拆分或删除。"""

    operation: Literal["replace_blocks"]
    block_ids: list[str] = Field(min_length=1)
    blocks: list[Any]


class InsertAfter(PlannerFinalModel):
    """【研究交付】【插入操作】空锚点只表示稿首，所有锚点均来自同一原稿。"""

    operation: Literal["insert_after"]
    after_block_id: str | None
    blocks: list[Any] = Field(min_length=1)


class RevisionExpansion(PlannerFinalModel):
    """【研究交付】【扩大范围】额外目标必须关联指定问题，不凭说明任意重写。"""

    block_id: str
    issue_index: int = Field(ge=0)
    reason: str = Field(min_length=1)


class PlannerFinalPatch(PlannerFinalModel):
    """【研究交付】【补丁合同】所有操作在同一不可变基稿上一次校验、一次应用。"""

    contract_version: Literal["research-planner-final-patch-v1"]
    base_attempt_id: str
    base_digest: str
    research_snapshot_digest: str
    operations: list[Annotated[ReplaceBlocks | InsertAfter, Field(discriminator="operation")]] = Field(min_length=1)
    expansions: list[RevisionExpansion] = Field(default_factory=list)
    unresolved_items: list[FinalUnresolvedDraft]
    status: Literal["answered", "incomplete", "unable"]


def research_snapshot_digest(payload: PlannerFinalAnswerInput) -> str:
    """参数为当前研究输入；返回包含完整要求、主张和证据版本的摘要。"""
    return content_hash({"research_plan": payload.plan.model_dump(mode="json"),
                         "research_unit_results": [item.model_dump(mode="json") for item in payload.unit_results],
                         "evidence_records": payload.evidence_records})


def make_candidate(payload: PlannerFinalAnswerInput, final: PlannerFinalValidationOutcome, raw: Any) -> dict[str, Any]:
    """参数为同次输入、预检/核对结果及原输出；返回唯一候选记录，未提交为用户答案。"""
    blocks = [block.model_dump(mode="json") for block in final.blocks]
    if isinstance(raw, dict) and isinstance(raw.get("blocks"), list):
        present = {block["block_index"] for block in blocks}
        for index, original in enumerate(raw["blocks"]):
            if index in present:
                continue
            block_id = f"{payload.attempt_id}:{index}"
            if raw["contract_version"] != FINAL_DRAFT_VERSION and isinstance(original, dict):
                block_id = original["block_id"]
            blocks.append(FinalAnswerBlock(block_index=index, block_id=block_id, text="",
                validation_errors=[error for error in final.errors if error.block_index == index]).model_dump(mode="json"))
        blocks.sort(key=lambda block: block["block_index"])
    return {"attempt_id": final.delivery_report.attempt_id, "blocks": blocks, "answer": final.answer,
            "section_facets": {section.section_id: list(section.facet_ids) for section in payload.plan.coverage_review.answer_sections},
            "body_digest": candidate_body_digest(blocks),
            "previous_body_digests": [],
            "base_digest": content_hash({"blocks": blocks, "answer": final.answer}),
            "research_snapshot_digest": research_snapshot_digest(payload),
            "report": final.delivery_report.model_dump(mode="json"), "raw_output": raw}


def candidate_body_digest(blocks: list[dict[str, Any]]) -> str:
    """参数为规范化正文块；返回独立于新编号的实际正文和支持关系摘要，用于已有无进展判断。"""
    return content_hash([{key: block[key] for key in _BLOCK_FIELDS if key != "block_id"} for block in blocks])


def candidate_normalized_draft(candidate: dict[str, Any]) -> dict[str, Any]:
    """参数为同次候选；返回规范化正文和原始完成/缺失声明，刷新或失败投影均不丢声明。"""
    raw = candidate["raw_output"]
    if not isinstance(raw, dict):
        raise TypeError("正文候选缺少原始结构化输出")
    return {"contract_version": FINAL_CONTRACT_VERSION,
            "blocks": [{key: block[key] for key in _BLOCK_FIELDS} for block in candidate["blocks"]],
            **{key: raw[key] for key in ("status", "confidence", "failure_reason", "unresolved_items") if key in raw}}


def candidate_revision_request(candidate: dict[str, Any]) -> dict[str, Any]:
    """参数为当前候选；返回真实问题、允许目标和关联上下文，不按错误数量扩大范围。"""
    report = candidate["report"]
    # 1. 【研究交付】【错误定位】不可读块的原始错误可能没有身份，按原位置关联候选占位块，不改写错误
    block_ids = {block["block_index"]: block["block_id"] for block in candidate["blocks"]}
    issues = [{"block_ids": [block_ids[error["block_index"]]],
               "facet_ids": [], "section_ids": [], "reason": error["message"]}
              for error in report["validation_errors"] if error.get("block_index") is not None]
    # 2. 【研究交付】【展示修复】漏填章节关联时仍可定位同要点正文，范围只来自同次稳定章节要求
    issues.extend({"block_ids": row["block_ids"], "facet_ids": candidate["section_facets"][row["requirement_id"]],
                   "section_ids": [row["requirement_id"]],
                   "reason": "; ".join(reason["message"] for reason in row["reasons"])}
                  for row in report["section_results"] if row["required"] and row["status"] == "missing")
    review = report.get("content_review") or candidate.get("revision_feedback") or {}
    assessment = review.get("assessment") or {}
    issues.extend(assessment.get("issues", []))
    issues.extend({"block_ids": row["block_ids"], "facet_ids": [row["requirement_id"]], "kind": "missing_content",
                   "reason": "; ".join(reason["message"] for reason in row["reasons"])}
                  for row in report["requirement_results"] if row["required"] and row["status"] == "missing"
                  and any(reason["source"] == "final_delivery" for reason in row["reasons"]))
    issues.extend({"block_ids": point["block_ids"], "facet_ids": [point["facet_id"]],
                   "reason": point["explanation"]} for point in assessment.get("points", [])
                  if point["verdict"] != "answered")
    targets = {key for issue in issues for key in issue["block_ids"]}
    related = [_related_blocks(candidate["blocks"], issue) for issue in issues]
    return {"base_attempt_id": candidate["attempt_id"], "base_digest": candidate["base_digest"],
            "research_snapshot_digest": candidate["research_snapshot_digest"], "issues": issues,
            "allowed_block_ids": sorted(targets), "related_block_ids": [sorted(keys) for keys in related],
            "allow_insert": any(not issue["block_ids"] or issue.get("kind") == "missing_content"
                                or issue.get("section_ids") for issue in issues),
            "revision_scope": assessment.get("revision_scope") or "local",
            "reason": assessment.get("reason") or "先修正确定性结构与展示问题"}


def _related_blocks(blocks: list[dict[str, Any]], issue: dict[str, Any]) -> set[str]:
    """参数为原稿和单个问题；返回共享真实主张、要点或用户章节的检查上下文。"""
    selected = [block for block in blocks if block["block_id"] in issue["block_ids"]]
    associations = {field: set(issue.get(field, [])) | {key for block in selected for key in block[field]}
                    for field in ("claim_ids", "facet_ids", "section_ids")}
    return {block["block_id"] for block in blocks
            if any(set(block[field]) & keys for field, keys in associations.items())}


def _authorized_targets(patch: PlannerFinalPatch, request: dict[str, Any]) -> set[str]:
    """参数为补丁及本次授权范围；返回可改目标，拒绝无真实关联的扩大范围。"""
    allowed = set(request["allowed_block_ids"])
    seen = set()
    for expansion in patch.expansions:
        if expansion.block_id in seen or not expansion.reason.strip():
            raise ValueError("重复或无理由的正文修复扩大范围")
        if (expansion.issue_index >= len(request["issues"])
                or expansion.block_id not in request["related_block_ids"][expansion.issue_index]):
            raise ValueError("额外修改目标与原问题没有已知关联")
        allowed.add(expansion.block_id)
        seen.add(expansion.block_id)
    return allowed


def _patch_layout(patch: PlannerFinalPatch, candidate: dict[str, Any], request: dict[str, Any]
                  ) -> tuple[dict[int, ReplaceBlocks], dict[int, InsertAfter], set[str]]:
    """参数为同基稿补丁和范围；返回已核验的区间与锚点，冲突时不修改任何原块。"""
    ids = [block["block_id"] for block in candidate["blocks"]]
    allowed = _authorized_targets(patch, request)
    replaced: dict[int, ReplaceBlocks] = {}
    inserted: dict[int, InsertAfter] = {}
    targets: set[str] = set()
    for operation in patch.operations:
        if isinstance(operation, ReplaceBlocks):
            if not set(operation.block_ids) <= allowed or len(set(operation.block_ids)) != len(operation.block_ids):
                raise ValueError("补丁含未授权或重复目标")
            positions = [ids.index(key) for key in operation.block_ids]
            if positions != list(range(positions[0], positions[0] + len(positions))) or targets & set(operation.block_ids):
                raise ValueError("替换目标必须为不重叠的连续原稿区间")
            targets.update(operation.block_ids)
            replaced[positions[0]] = operation
        else:
            if not request["allow_insert"]:
                raise ValueError("当前问题没有新增正文的插入依据")
            position = -1 if operation.after_block_id is None else ids.index(operation.after_block_id)
            if position in inserted:
                raise ValueError("同一原稿锚点不能重复插入")
            inserted[position] = operation
    if any(operation.after_block_id in targets for operation in inserted.values()):
        raise ValueError("插入锚点不能同时被替换或删除")
    return replaced, inserted, targets


def _validate_patch_base(payload: PlannerFinalAnswerInput, candidate: dict[str, Any], patch: PlannerFinalPatch) -> None:
    """参数为补丁及当前基稿；拒绝正文或研究快照冲突，无返回值、不修改候选。"""
    if payload.attempt_id == candidate["attempt_id"]:
        raise ValueError("正文修复必须分配新的候选身份")
    request = candidate_revision_request(candidate)
    for key in ("base_attempt_id", "base_digest", "research_snapshot_digest"):
        if getattr(patch, key) != request[key]:
            raise ValueError(f"正文补丁基稿版本冲突：{key}")
    if research_snapshot_digest(payload) != patch.research_snapshot_digest:
        raise ValueError("正文补丁研究快照已变化")
    section_facets = {section.section_id: list(section.facet_ids) for section in payload.plan.coverage_review.answer_sections}
    if candidate["section_facets"] != section_facets:
        raise ValueError("正文补丁章节关联与当前稳定要求不符")
    if content_hash({"blocks": candidate["blocks"], "answer": candidate["answer"]}) != patch.base_digest:
        raise ValueError("正文补丁基稿摘要不符")


def apply_final_patch(payload: PlannerFinalAnswerInput, candidate: dict[str, Any], raw: Any) -> dict[str, Any]:
    """参数为当前输入、原稿及原始补丁；返回原子组装的新稿，版本/操作/新块失败均不改原稿。"""
    patch = PlannerFinalPatch.model_validate(raw)
    _validate_patch_base(payload, candidate, patch)
    request = candidate_revision_request(candidate)
    replacements, insertions, targets = _patch_layout(patch, candidate, request)
    raw_new = [block for operation in patch.operations for block in operation.blocks]
    catalog = final_product_catalog(payload)
    normalized, errors = normalize_product_blocks(raw_new, catalog, attempt_id=payload.attempt_id)
    _, relation_errors = validate_final_blocks(payload, normalized, catalog)
    if errors or relation_errors:
        raise ValueError("补丁新块未通过结构预检：" + "; ".join(error.message for error in [*errors, *relation_errors]))
    # 1. 【研究交付】【原子应用】全部输入通过后才分配新稿，未改正文的身份、文字和业务关系逐字保留
    cursor = 0
    projected = {}
    for operation in patch.operations:
        projected[id(operation)] = normalized[cursor:cursor + len(operation.blocks)]
        cursor += len(operation.blocks)
    blocks = list(projected[id(insertions[-1])]) if -1 in insertions else []
    for index, block in enumerate(candidate["blocks"]):
        if index in replacements:
            blocks.extend(projected[id(replacements[index])])
        elif block["block_id"] not in targets:
            blocks.append({key: block[key] for key in _BLOCK_FIELDS})
        if index in insertions:
            blocks.extend(projected[id(insertions[index])])
    original = [{key: block[key] for key in _BLOCK_FIELDS if key != "block_id"} for block in candidate["blocks"]]
    changed = [{key: block[key] for key in _BLOCK_FIELDS if key != "block_id"} for block in blocks]
    if original == changed:
        raise ValueError("正文修复没有带来变化，停止重复执行")
    if content_hash(changed) in candidate.get("previous_body_digests", []):
        raise ValueError("正文修复重复了已经核对的候选，没有新进展")
    return {"contract_version": FINAL_CONTRACT_VERSION, "blocks": blocks, "status": patch.status,
            "unresolved_items": [item.model_dump(mode="json") for item in patch.unresolved_items]}


def delivery_next_action(candidate: dict[str, Any]) -> str:
    """参数为同次候选；返回修文、研究或终止决定，不把未核对记为通过。"""
    report = candidate["report"]
    review = report.get("content_review") or {}
    if report["generation_status"] != "completed" or review.get("status") == "failed":
        return "terminate_fail"
    if not any(block["text"].strip() for block in candidate["blocks"]):
        return "terminate_fail"
    if report["validation_errors"]:
        if any(error["block_index"] is None and error["code"] != "invalid_block"
               for error in report["validation_errors"]):
            return "terminate_fail"
        if not any(error["block_index"] is not None for error in report["validation_errors"]):
            return "terminate_fail"
        return "revise_answer"
    if any(row["required"] and row["status"] == "missing" for row in report["section_results"]):
        return "revise_answer"
    assessment = review.get("assessment") or {}
    recommendation = assessment.get("recommendation")
    if recommendation == "revise_answer":
        return "revise_answer"
    if recommendation == "continue_research":
        return "replan"
    if recommendation == "partial" and (any(issue["kind"] != "missing_content" for issue in assessment["issues"])
            or any(point["verdict"] == "unsupported" for point in assessment["points"])):
        return "revise_answer"
    if recommendation == "deliver" and report["status"] == "partial":
        return "revise_answer"
    return "terminate_fail" if report["status"] == "failed" else "continue"
