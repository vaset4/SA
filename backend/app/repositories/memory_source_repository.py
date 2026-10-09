"""【研究记忆】【来源维护】来源授权和同事务失效维护，作者：xxx。"""
from __future__ import annotations

from typing import Any

from app.models.source_access_models import SourceVersionChange
from app.repositories.common import dump_json, load_json
from app.services.database import get_pool


async def source_state(connection: Any, *, user_id: str, project_id: str, paper_id: str) -> dict[str, Any]:
    """参数为连接及可信归属；锁定当前论文和项目范围，返回可比较的现役来源事实。"""
    paper = await connection.fetchrow("SELECT * FROM papers WHERE paper_id=$1 AND user_id=$2 AND deletion_status='active' FOR SHARE",
                                      paper_id, user_id)
    project = await connection.fetchrow("SELECT * FROM projects WHERE project_id=$1 AND user_id=$2 FOR SHARE", project_id, user_id)
    if paper is None or project is None:
        raise LookupError("记忆来源或项目不存在或无权访问")
    scope = await connection.fetchrow("SELECT * FROM project_scopes WHERE project_id=$1 FOR SHARE", project_id)
    scope = dict(scope) if scope else {}
    explicit = load_json(scope.get("paper_ids"), [])
    libraries = load_json(scope.get("library_ids"), []) or [project["default_library_id"]]
    excluded = load_json(scope.get("excluded_paper_ids"), [])
    if paper_id in excluded or (paper_id not in explicit if explicit else paper["library_id"] not in libraries):
        raise LookupError("记忆来源已不在当前项目范围")
    batch = await connection.fetchrow("SELECT * FROM upload_batches WHERE batch_id=$1 AND paper_id=$2 AND user_id=$3 AND state='published' AND published_at IS NOT NULL FOR SHARE",
                                      paper["index_batch_id"], paper_id, user_id)
    if batch is None:
        raise LookupError("记忆来源没有可核验的已发布原文")
    return {"current_batch_id": batch["batch_id"], "manifest_ref": load_json(batch["manifest_ref"], {})}


async def read_source_state(*, user_id: str, project_id: str, paper_id: str) -> dict[str, Any]:
    """参数为用户、项目和论文；在短事务核对当前来源，返回版本事实，不读取正文。"""
    pool = await get_pool()
    async with pool.acquire() as connection, connection.transaction():
        return await source_state(connection, user_id=user_id, project_id=project_id, paper_id=paper_id)


async def revalidate_memory_sources(connection: Any, *, user_id: str, project_id: str,
                                    evidence_refs: list[dict[str, Any]]) -> None:
    """参数为变更事务和已核验引用；先锁来源再做记忆CAS，拒绝核验后的撤权与新发布。"""
    states: dict[str, dict[str, Any]] = {}
    for paper_id in sorted({ref["paper_id"] for ref in evidence_refs}):
        states[paper_id] = await source_state(connection, user_id=user_id, project_id=project_id, paper_id=paper_id)
    for ref in evidence_refs:
        verified = ref.get("verification") or {}
        if verified.get("current_batch_id") != states[ref["paper_id"]]["current_batch_id"]:
            raise ValueError("来源在核验后已更新，请重新核对后提交")


async def invalidate_source(connection: Any, *, user_id: str, paper_id: str | None,
                            reason: str, project_id: str | None = None) -> int:
    """参数为来源变更连接及影响范围；保留旧正文/引用并为受影响文献型记忆记失效版本。"""
    from app.repositories.research_memory_repository import get_research_memory_repository

    rows = await connection.fetch("""SELECT item.* FROM research_memory_items item
        WHERE item.user_id=$1 AND item.evidence_mode IN ('evidence_backed','mixed')
        AND ($2::text IS NULL OR item.project_id=$2)
        AND ($3::text IS NULL OR EXISTS(SELECT 1 FROM memory_evidence_refs ref
            WHERE ref.memory_id=item.memory_id AND ref.paper_id=$3 AND ref.ref_role<>'context_only'))
        ORDER BY item.memory_id FOR UPDATE""", user_id, project_id, paper_id)
    changed = 0
    for row in rows:
        reasons = load_json(row["stale_reasons"], [])
        if reason in reasons and row["stale_state"] == "needs_rebind":
            continue
        await connection.execute("""UPDATE research_memory_items SET stale_state='needs_rebind',
            stale_reasons=$2::jsonb,version=version+1,updated_at=NOW() WHERE memory_id=$1""",
            row["memory_id"], dump_json([*reasons, reason]))
        await get_research_memory_repository()._snapshot(connection, row["memory_id"], "source_changed", reason, user_id)
        changed += 1
    return changed


async def apply_source_change(connection: Any, change: SourceVersionChange) -> None:
    """参数为发布连接和绑定旧指针的内容分类；只有实际变化或未知才使原支持待重绑。"""
    if change.classification in {"initial", "unchanged"}:
        return
    reason = "来源正文已变化" if change.classification == "changed" else "来源版本缺少可比较的历史正文身份"
    await invalidate_source(connection, user_id=change.user_id, paper_id=change.paper_id,
                            reason=f"{reason}: {change.paper_id}")


async def invalidate_project_scope(connection: Any, *, project_id: str, user_id: str) -> None:
    """参数为范围变更连接和项目归属；仅使已移出有效范围的来源依赖失效。"""
    rows = await connection.fetch("""SELECT DISTINCT ref.paper_id FROM memory_evidence_refs ref
        JOIN research_memory_items item ON item.memory_id=ref.memory_id
        WHERE item.project_id=$1 AND item.user_id=$2 ORDER BY ref.paper_id""", project_id, user_id)
    for row in rows:
        try:
            await source_state(connection, user_id=user_id, project_id=project_id, paper_id=row["paper_id"])
        except LookupError:
            await invalidate_source(connection, user_id=user_id, project_id=project_id, paper_id=row["paper_id"],
                                    reason=f"支持资料已不在项目范围: {row['paper_id']}")


async def mark_checked_stale(item: dict[str, Any], reasons: list[str]) -> None:
    """参数为实际读到的条目版本和失效原因；CAS写回失效，竞争更新时明确拒绝。"""
    from app.repositories.research_memory_repository import get_research_memory_repository

    pool = await get_pool()
    async with pool.acquire() as connection, connection.transaction():
        updated = await connection.fetchval("""UPDATE research_memory_items SET stale_state='needs_rebind',
            stale_reasons=$5::jsonb,version=version+1,updated_at=NOW()
            WHERE memory_id=$1 AND user_id=$2 AND project_id=$3 AND version=$4 RETURNING memory_id""",
            item["memory_id"], item["user_id"], item["project_id"], item["version"], dump_json(reasons))
        if updated is None:
            raise ValueError("记忆在核对期间已更新，请重新读取")
        await get_research_memory_repository()._snapshot(connection, updated, "source_checked", "；".join(reasons), item["user_id"])


async def record_source_checks(item: dict[str, Any], evidence_refs: list[dict[str, Any]]) -> dict[str, Any]:
    """参数为读取版本和本次真实核验引用；记录历史首次检查事实，不回填为当年的精确支持。"""
    pool = await get_pool()
    async with pool.acquire() as connection, connection.transaction():
        await revalidate_memory_sources(connection, user_id=item["user_id"], project_id=item["project_id"], evidence_refs=evidence_refs)
        version = await connection.fetchval("SELECT version FROM research_memory_items WHERE memory_id=$1 AND user_id=$2 AND project_id=$3 FOR UPDATE",
                                            item["memory_id"], item["user_id"], item["project_id"])
        if version != item["version"]:
            raise ValueError("记忆在来源核验期间已更新")
        changed = False
        for ref in evidence_refs:
            updated = await connection.fetchval("""UPDATE memory_evidence_refs SET verification=$3::jsonb,
                source_ref=$4::jsonb,document_ref=$5::jsonb WHERE ref_id=$1 AND memory_id=$2
                AND (verification IS DISTINCT FROM $3::jsonb OR source_ref IS DISTINCT FROM $4::jsonb
                    OR document_ref IS DISTINCT FROM $5::jsonb) RETURNING ref_id""",
                ref["ref_id"], item["memory_id"], dump_json(ref["verification"]), dump_json(ref["source_ref"]), dump_json(ref["document_ref"]))
            changed = changed or updated is not None
        if not changed:
            return item
        from app.repositories.research_memory_repository import get_research_memory_repository

        await connection.execute("UPDATE research_memory_items SET version=version+1,updated_at=NOW() WHERE memory_id=$1", item["memory_id"])
        await get_research_memory_repository()._snapshot(connection, item["memory_id"], "source_checked", "记录本次实际原文核验", item["user_id"])
        return {**item, "version": item["version"] + 1}
