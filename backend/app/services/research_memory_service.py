"""
项目研究记忆层服务。

这一层是研究记忆的业务入口，把仓储层的原子读写编排成符合业务语义的用例，
并集中承载业务规则：

- 受控枚举校验（非法值显式抛 ValueError，不静默纠正）
- 研究目的 is_purpose 每项目 singleton（应用层保证，不下推 DB 约束）
- evidence gate：事实型记忆确认时必须有证据，否则显式拒绝
- 四轴状态流转（status / stale_state / review_state / maturity），每次变更留版本快照
- 证据 quote 规范化与稳定 hash
- 记忆关系语义（supersedes 保留旧条目 / conflicts_with 单行存双向查）

失败一律显式（ValueError / LookupError），不静默降级、不 mock 成功。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.models.research_memory_contracts import (
    DEFAULT_SOURCE_KIND,
    USER_AUTHORED_SOURCE_KINDS,
    VALID_EVIDENCE_MODES,
    VALID_LINK_TYPES,
    VALID_MATURITIES,
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
    VALID_REF_ROLES,
    VALID_REVIEW_STATES,
    VALID_SOURCE_KINDS,
    VALID_STALE_STATES,
)
from app.repositories.research_memory_repository import get_research_memory_repository

_schema_ready = False

# quote 规范化：折叠连续空白，配合 strip + 小写，得到稳定 hash 输入
_WHITESPACE_PATTERN = re.compile(r"\s+")
# 证据 quote hash 取 sha1 前 16 位，与 evidence_id 风格统一
_QUOTE_DIGEST_CHARS = 16


async def _ensure_schema() -> None:
    global _schema_ready
    if not _schema_ready:
        await get_research_memory_repository().ensure_schema()
        _schema_ready = True


def normalize_and_hash_quote(text: str) -> tuple[str, str]:
    """
    规范化证据原句并计算稳定 hash。

    参数:
        text: 证据原句
    返回:
        (规范化文本, 16 位十六进制 hash)；规范化 = strip + 小写 + 折叠空白
    """
    normalized = _WHITESPACE_PATTERN.sub(" ", str(text or "").strip().lower())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:_QUOTE_DIGEST_CHARS]
    return normalized, digest


def _require_enum(value: Any, valid: tuple[str, ...], field: str) -> None:
    # 受控枚举校验：非法值显式抛错，不静默纠正为默认值
    if value is not None and value not in valid:
        raise ValueError(f"{field} 取值非法: {value!r}，合法值为 {valid}")


def _derive_initial_status(payload: dict[str, Any]) -> str:
    """
    按来源派生新建记忆的生命周期状态。

    用户亲手写下的判断即刻生效，不再要求二次点「确认」。draft 那个人审位是给 AI 草稿留的，
    对用户手写的判断没有语义。读取侧即将按 status=confirmed 收口（四道资格门），
    届时仍停在 draft 的手写记忆会一条都进不了主链——用户以为记下了，Agent 却读不到。
    所以派生必须先于收口落地。

    参数:
        payload: 新建记忆的字段，读取其中的 status / evidence_mode / source_kind
    返回:
        生命周期状态；派生结果为 'draft' 或 'confirmed'，
        调用方显式指定时原样返回该值（可为任一合法 status，由枚举校验兜住）
    """
    # 1. 调用方显式指定 status 时以显式值为准（curator 走仓储直连，此处兜住其它显式传参场景）
    explicit = payload.get("status")
    if explicit:
        return str(explicit)
    # 2. 例外：声称有文献背书的条目仍落 draft。证据要事后经 add_evidence 单独挂，
    #    直接 confirmed 会绕过 confirm_item 里「事实型记忆确认前必须至少绑一条证据」这道检查
    if payload.get("evidence_mode") in {"evidence_backed", "mixed"}:
        return "draft"
    # 3. 用户来源即刻生效；source_kind 缺键时按仓储默认的 manual 算
    #    （前端 CreateMemoryRequest 不带这个字段，缺键就是最常见的手写场景）
    source_kind = payload.get("source_kind") or DEFAULT_SOURCE_KIND
    if source_kind in USER_AUTHORED_SOURCE_KINDS:
        return "confirmed"
    return "draft"


def _validate_item_enums(payload: dict[str, Any]) -> None:
    # 记忆条目落库前统一校验其上所有受控枚举字段
    _require_enum(payload.get("memory_type"), VALID_MEMORY_TYPES, "memory_type")
    _require_enum(payload.get("status"), VALID_MEMORY_STATUSES, "status")
    _require_enum(payload.get("stale_state"), VALID_STALE_STATES, "stale_state")
    _require_enum(payload.get("review_state"), VALID_REVIEW_STATES, "review_state")
    _require_enum(payload.get("maturity"), VALID_MATURITIES, "maturity")
    _require_enum(payload.get("evidence_mode"), VALID_EVIDENCE_MODES, "evidence_mode")
    _require_enum(payload.get("source_kind"), VALID_SOURCE_KINDS, "source_kind")


async def create_question(project_id: str, payload: dict, *, user_id: str) -> dict:
    """参数为可信项目、owner 与问题；创建有明确归属的研究问题，返回版本。"""
    await _ensure_schema()
    if not str(payload.get("title", "")).strip():
        raise ValueError("研究问题标题不能为空")
    return await get_research_memory_repository().create_question(project_id, {**payload, "user_id": user_id})


async def list_questions(project_id: str, *, user_id: str, review_state: str | None = "accepted") -> list[dict]:
    """参数为可信归属和审核过滤；返回该 owner/project 的研究问题。"""
    await _ensure_schema()
    return await get_research_memory_repository().list_questions(project_id, user_id=user_id, review_state=review_state)


async def update_question(project_id: str, question_id: str, patch: dict, *, user_id: str,
                          expected_version: int) -> dict:
    """参数为归属、问题与版本；事务内更新，迟到版本不能覆盖新编辑。"""
    await _ensure_schema()
    return await get_research_memory_repository().change_question(question_id, project_id=project_id,
        user_id=user_id, expected_version=expected_version, patch=patch)


async def delete_question(project_id: str, question_id: str, *, user_id: str, expected_version: int) -> dict:
    """参数为归属、问题与版本；删除具体问题及关联，返回真实删除事实。"""
    await _ensure_schema()
    return await get_research_memory_repository().change_question(question_id, project_id=project_id,
        user_id=user_id, expected_version=expected_version, patch=None)


async def create_item(project_id: str, payload: dict, *, user_id: str) -> dict:
    """参数为可信归属和手工判断；校验字段后原子创建条目、问题关联与初始快照。"""
    await _ensure_schema()
    if not str(payload.get("title", "")).strip() or not payload.get("memory_type"):
        raise ValueError("记忆标题和类型不能为空")
    _validate_item_enums(payload)
    return await get_research_memory_repository().create_item(project_id,
        {**payload, "user_id": user_id, "status": _derive_initial_status(payload)})


async def get_detail(memory_id: str, *, project_id: str, user_id: str) -> dict:
    """参数为记忆与可信 owner/project；只返回同归属的具体资源，越权等同不存在。"""
    await _ensure_schema()
    detail = await get_research_memory_repository().get_detail(memory_id, project_id=project_id, user_id=user_id)
    if detail is None:
        raise LookupError("记忆不存在")
    return detail


async def list_items(project_id: str, *, user_id: str, **filters: Any) -> list[dict]:
    """参数为可信范围与过滤；all 是管理列表，自动读取仍使用四门条件。"""
    await _ensure_schema()
    if filters.get("status") == "all":
        filters = {**filters, "status": None, "review_state": None}
    for field, valid in (("status", VALID_MEMORY_STATUSES), ("stale_state", VALID_STALE_STATES),
                         ("review_state", VALID_REVIEW_STATES), ("maturity", VALID_MATURITIES),
                         ("memory_type", VALID_MEMORY_TYPES)):
        _require_enum(filters.get(field), valid, field)
    return await get_research_memory_repository().list_items(project_id, user_id=user_id, **filters)


async def list_review_queue(project_id: str, *, user_id: str) -> list[dict]:
    """参数为可信归属；列出当前用户的待审条目。"""
    return await list_items(project_id, user_id=user_id, review_state="pending_review")


async def search_memories_literal(project_id: str, keyword: str, *, user_id: str,
                                  stale_state: str | None = None) -> list[dict]:
    """参数为归属与字面词；只搜索当前 owner/project 的已接纳条目。"""
    await _ensure_schema()
    if not keyword.strip():
        raise ValueError("搜索关键词不能为空")
    return await get_research_memory_repository().search_memories_literal(
        project_id, keyword.strip(), user_id=user_id, stale_state=stale_state)


async def count_by_maturity(project_id: str, *, user_id: str) -> dict[str, int]:
    """参数为可信归属；返回当前用户已接纳条目的成熟度统计。"""
    await _ensure_schema()
    return await get_research_memory_repository().count_by_maturity(project_id, user_id=user_id)


async def _validated_support(item: dict, refs: list[dict], *, project_id: str, user_id: str) -> list[dict]:
    """参数为修改后的判断和引用；统一确认/接纳资格，返回实际核验后的引用集合。"""
    from app.services.memory_evidence_service import validate_memory_evidence

    if item.get("evidence_mode") == "user_note_only" and all(ref.get("ref_role") == "context_only" for ref in refs):
        return []
    if not any(ref.get("ref_role") in {"primary_support", "derived_basis"} for ref in refs):
        raise ValueError("文献型记忆确认前必须绑定可核对的主支持或推断依据")
    # 1. 【研究记忆】【共同确认】主支持、约束和反证都必须仍适用，背景引用不替代主支持
    return [await validate_memory_evidence(ref, user_id=user_id, project_id=project_id)
            for ref in refs if ref.get("ref_role") != "context_only"]


async def update_item(memory_id: str, patch: dict, *, project_id: str, user_id: str,
                      expected_version: int, confirm: bool = False, accept: bool = False) -> dict:
    """参数为归属、版本和编辑/确认意图；正文改动回草稿，显式确认同事务恢复 fresh。"""
    detail = await get_detail(memory_id, project_id=project_id, user_id=user_id)
    _validate_item_enums(patch)
    if "title" in patch and not str(patch["title"]).strip():
        raise ValueError("记忆标题不能为空")
    current = detail["item"]
    edit = dict(patch)
    if "body" in edit and edit["body"] != current["body"]:
        edit["status"] = "draft"
    if "evidence_mode" in edit and edit["evidence_mode"] != current["evidence_mode"]:
        edit.update(status="draft", stale_state="needs_rebind", stale_reasons=["证据声明已改变，需重新确认"])
    if confirm and patch.get("status") == "draft":
        raise ValueError("回到草稿与确认不能同时执行")
    refs = []
    if confirm or accept:
        refs = await _validated_support({**current, **edit}, detail["evidence_refs"], project_id=project_id, user_id=user_id)
        edit.update(status="confirmed", stale_state="fresh", stale_reasons=[])
    if accept:
        edit["review_state"] = "accepted"
    return await get_research_memory_repository().change_item(memory_id, project_id=project_id, user_id=user_id,
        expected_version=expected_version, change={"kind": "accept_draft" if accept else "confirm" if confirm else "edit_body",
        "patch": edit, "validated_refs": refs})


async def accept_draft(memory_id: str, patch: dict | None = None, *, project_id: str, user_id: str,
                       expected_version: int) -> dict:
    """参数为待审条目和修改后的正文；与确认共用支持核验，返回已接纳版本。"""
    return await update_item(memory_id, patch or {}, project_id=project_id, user_id=user_id,
                             expected_version=expected_version, accept=True)


async def confirm_item(memory_id: str, *, project_id: str, user_id: str, expected_version: int) -> dict:
    """参数为具体资源和版本；确认当前支持并原子恢复 fresh。"""
    return await update_item(memory_id, {}, project_id=project_id, user_id=user_id,
                             expected_version=expected_version, confirm=True)


async def dismiss_draft(memory_id: str, *, project_id: str, user_id: str, expected_version: int) -> dict:
    """参数为具体资源和版本；拒绝草稿并保存审核历史。"""
    return await update_item(memory_id, {"review_state": "dismissed"}, project_id=project_id,
                             user_id=user_id, expected_version=expected_version)


async def set_maturity(memory_id: str, maturity: str, *, project_id: str, user_id: str, expected_version: int) -> dict:
    """参数为资源、版本与成熟度；成熟度变化不改变现有确认事实。"""
    return await update_item(memory_id, {"maturity": maturity}, project_id=project_id,
                             user_id=user_id, expected_version=expected_version)


async def mark_stale(memory_id: str, reasons: list[str], *, project_id: str, user_id: str, expected_version: int) -> dict:
    """参数为具体资源和原因；保留正文与确认历史，暂停依赖记忆复用。"""
    return await update_item(memory_id, {"stale_state": "needs_rebind", "stale_reasons": reasons},
                             project_id=project_id, user_id=user_id, expected_version=expected_version)


async def delete_item(memory_id: str, *, project_id: str, user_id: str, expected_version: int) -> dict:
    """参数为资源与版本；只删除仍与用户看到版本一致的记忆。"""
    await get_detail(memory_id, project_id=project_id, user_id=user_id)
    return await get_research_memory_repository().change_item(memory_id, project_id=project_id, user_id=user_id,
        expected_version=expected_version, change={"kind": "delete"})


async def add_evidence(memory_id: str, payload: dict, *, project_id: str, user_id: str,
                       expected_version: int, confirm: bool = False) -> dict:
    """参数为原文选择、版本和确认意图；核验精确定位后原子绑定，可同时重绑并确认。"""
    from app.services.memory_evidence_service import validate_memory_evidence

    detail = await get_detail(memory_id, project_id=project_id, user_id=user_id)
    _require_enum(payload.get("ref_role"), VALID_REF_ROLES, "ref_role")
    ref = await validate_memory_evidence(payload, user_id=user_id, project_id=project_id)
    _, quote_hash = normalize_and_hash_quote(ref["quote_text"])
    ref = {**ref, "quote_hash": quote_hash}
    patch = {}
    if ref.get("ref_role") != "context_only":
        patch = {"stale_state": "needs_rebind", "stale_reasons": ["已声明的支持依据改变，需确认当前判断"]}
    validated = [ref]
    if confirm:
        validated = await _validated_support(detail["item"], [*detail["evidence_refs"], ref],
                                               project_id=project_id, user_id=user_id)
        if ref.get("ref_role") == "context_only":
            validated.append(ref)
        patch = {"status": "confirmed", "stale_state": "fresh", "stale_reasons": []}
    return await get_research_memory_repository().change_item(memory_id, project_id=project_id, user_id=user_id,
        expected_version=expected_version, change={"kind": "attach_evidence", "patch": patch,
                                                  "add_ref": ref, "validated_refs": validated})


async def remove_evidence(memory_id: str, ref_id: str, *, project_id: str, user_id: str,
                          expected_version: int) -> dict:
    """参数为资源与版本；移除主支持/反证或未知角色后待重绑，明确背景引用可保持 fresh。"""
    detail = await get_detail(memory_id, project_id=project_id, user_id=user_id)
    ref = next((ref for ref in detail["evidence_refs"] if ref["ref_id"] == ref_id), None)
    if ref is None:
        raise LookupError("证据引用不存在")
    patch = {} if ref.get("ref_role") == "context_only" else {
        "stale_state": "needs_rebind", "stale_reasons": ["已声明的支持或约束依据被移除，需重新绑定并确认"]}
    return await get_research_memory_repository().change_item(memory_id, project_id=project_id, user_id=user_id,
        expected_version=expected_version, change={"kind": "detach_evidence", "patch": patch, "remove_ref_id": ref_id})


async def add_link(project_id: str, src_memory_id: str, dst_memory_id: str, link_type: str, *, user_id: str,
                   expected_version: int, dst_expected_version: int, note: str = "") -> dict:
    """参数为关系两端及各自版本；核对两端归属并原子建立关系和双端版本快照。"""
    await _ensure_schema()
    _require_enum(link_type, VALID_LINK_TYPES, "link_type")
    if src_memory_id == dst_memory_id:
        raise ValueError("关系两端不能相同")
    return await get_research_memory_repository().change_link(project_id=project_id, user_id=user_id,
        expected_version=expected_version, dst_expected_version=dst_expected_version,
        payload={"src_memory_id": src_memory_id, "dst_memory_id": dst_memory_id, "link_type": link_type, "note": note})


async def remove_link(link_id: str, *, project_id: str, user_id: str,
                      expected_version: int, dst_expected_version: int) -> dict:
    """参数为关系及两端版本；删除时仍核对两端 owner/project 并各留版本。"""
    await _ensure_schema()
    return await get_research_memory_repository().change_link(project_id=project_id, user_id=user_id,
        expected_version=expected_version, dst_expected_version=dst_expected_version, payload={"link_id": link_id})


async def list_links(project_id: str, *, user_id: str) -> list[dict]:
    """参数为可信归属；只返回两端均属于当前 owner/project 的关系。"""
    await _ensure_schema()
    return await get_research_memory_repository().list_links(project_id, user_id=user_id)


async def attach_question(memory_id: str, question_id: str, *, project_id: str, user_id: str,
                           expected_version: int) -> dict:
    """参数为两端资源和记忆版本；归属校验、关联与快照共享事务。"""
    await _ensure_schema()
    return await get_research_memory_repository().change_item(memory_id, project_id=project_id, user_id=user_id,
        expected_version=expected_version, change={"kind": "attach_question", "question_id": question_id})


async def detach_question(memory_id: str, question_id: str, *, project_id: str, user_id: str,
                           expected_version: int) -> dict:
    """参数为两端资源和记忆版本；取消归属时仍核对研究问题属于当前项目。"""
    await _ensure_schema()
    return await get_research_memory_repository().change_item(memory_id, project_id=project_id, user_id=user_id,
        expected_version=expected_version, change={"kind": "detach_question", "question_id": question_id})


async def list_versions(memory_id: str, *, project_id: str, user_id: str) -> list[dict]:
    """参数为具体归属；版本历史与详情使用同一权限边界。"""
    return (await get_detail(memory_id, project_id=project_id, user_id=user_id))["versions"]
