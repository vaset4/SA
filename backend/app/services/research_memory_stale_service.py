"""【研究记忆】【来源复核】按不可变原文维护复用资格，作者：xxx。"""
from __future__ import annotations

from typing import Any

from app.repositories.memory_source_repository import mark_checked_stale, record_source_checks
from app.repositories.research_memory_repository import get_research_memory_repository
from app.services.memory_evidence_service import MemoryEvidenceError, validate_memory_evidence
from app.models.research_memory_contracts import MemoryVersionConflict


class MemoryCheckedModel:
    """仅在持有记忆快照的消费边界，确保每次模型调用前重新核对来源。"""

    def __init__(self, model: Any, state: dict[str, Any]) -> None:
        """参数为现有模型和本次状态；保留模型接口，不改变其请求参数。"""
        self._model, self._state = model, state

    def __getattr__(self, name: str) -> Any:
        """参数为现有模型属性；返回原模型配置与能力。"""
        return getattr(self._model, name)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """参数为现有结构化输出约束；返回仍保留来源复核的结构化模型。"""
        return MemoryCheckedModel(self._model.with_structured_output(*args, **kwargs), self._state)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """参数为模型消息及选项；先核对当前记忆，再执行原模型请求。"""
        await revalidate_memory_snapshot(self._state)
        return await self._model.ainvoke(*args, **kwargs)


async def revalidate_memory_snapshot(state: dict[str, Any]) -> None:
    """参数为将被模型消费的图状态；复核记忆快照身份、版本和来源，失效立即停止后续消费。"""
    snapshot = (state.get("memory_read_result") or {}).get("items") or []
    if not snapshot:
        return
    scope = state.get("project_scope") or {}
    user_id, project_id = scope.get("user_id"), scope.get("project_id")
    if not user_id or not project_id:
        raise ValueError("记忆快照缺少可信项目归属")
    repo = get_research_memory_repository()
    current = []
    for saved in snapshot:
        item = await repo.get_item(saved["memory_id"], user_id=user_id, project_id=project_id)
        if not item or item["version"] != saved["version"] or (
                item.get("review_state"), item.get("status"), item.get("stale_state")) != ("accepted", "confirmed", "fresh"):
            raise MemoryEvidenceError("本轮记忆已更新或失效，请重新读取后继续研究")
        current.append(item)
    verified = await checked_memory_items(current, project_id=project_id, user_id=user_id)
    if len(verified) != len(current):
        raise MemoryEvidenceError("本轮记忆来源已失效，不能继续用于模型判断")


async def checked_memory_items(items: list[dict[str, Any]], *, project_id: str, user_id: str) -> list[dict[str, Any]]:
    """参数为当前门内条目和可信归属；核对真实支持后返回仍可复用条目，存储失败原样抛出。"""
    result = []
    repo = get_research_memory_repository()
    for item in items:
        if item.get("project_id") != project_id or item.get("user_id") != user_id:
            raise LookupError("记忆不属于当前项目或用户")
        if item.get("evidence_mode") not in {"evidence_backed", "mixed"}:
            result.append(item)
            continue
        refs = await repo.list_evidence_refs(item["memory_id"])
        checked, reasons = [], []
        if not any(ref.get("ref_role") in {"primary_support", "derived_basis"} for ref in refs):
            reasons.append("文献判断缺少明确主支持，需要重新核对")
        for ref in refs:
            if ref.get("ref_role") == "context_only":
                continue
            try:
                checked.append(await validate_memory_evidence(ref, user_id=user_id, project_id=project_id))
            except (MemoryEvidenceError, LookupError) as error:
                reasons.append(str(error))
        if reasons:
            await mark_checked_stale(item, list(dict.fromkeys(reasons)))
        else:
            result.append(await record_source_checks(item, checked))
    return result


async def check_evidence_staleness(project_id: str, memory_id: str | None = None, *, user_id: str,
                                   expected_version: int | None = None) -> dict[str, Any]:
    """参数为可信项目/用户及可选记忆；执行同一来源复核，返回实际检查及失效数量。"""
    repo = get_research_memory_repository()
    await repo.ensure_schema()
    if memory_id:
        item = await repo.get_item(memory_id, user_id=user_id, project_id=project_id)
        if item is None:
            raise LookupError("记忆不存在或无权访问")
        if expected_version is not None and item["version"] != expected_version:
            raise MemoryVersionConflict("记忆版本已变化，请刷新后重新核对")
        items = [item]
    else:
        items = [item for item in await repo.list_items(project_id, user_id=user_id)
                 if item.get("evidence_mode") in {"evidence_backed", "mixed"}]
    valid = await checked_memory_items(items, project_id=project_id, user_id=user_id)
    valid_ids = {item["memory_id"] for item in valid}
    details = [{"memory_id": item["memory_id"], "stale_state": "needs_rebind"}
               for item in items if item["memory_id"] not in valid_ids]
    return {"checked": len(items), "marked_stale": len(details), "details": details}
