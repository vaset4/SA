"""
项目研究记忆层仓储。

这一层把研究记忆相关的数据库读写收口到一起，覆盖 6 张表：
- research_questions        研究问题（数据归属 + 研究目的置顶）
- research_memory_items     记忆条目本体（一条判断 + 四轴状态）
- memory_question_links     记忆 ↔ 研究问题 多对多
- memory_evidence_refs      记忆的证据引用（回指论文原句）
- memory_links              记忆 ↔ 记忆 有向关系
- memory_versions           每次变更的全量版本快照

建表沿用项目既有约定：无 ORM / Alembic，SCHEMA_SQL 内嵌 + 启动时
ensure_schema() 幂等执行（模仿 project_repository）。

写操作的原子性约定：任何改动记忆条目的方法，都在同一数据库事务内完成
「改主表 + version+1 + 插 memory_versions 全量快照」，避免主表与版本历史脱节。
change_kind 由 service 层决定并传入，本层只负责保证原子落库。
"""

from __future__ import annotations

import uuid
from typing import Any

from app.repositories.common import dump_json, load_json, normalize_record
from app.services.database import execute, fetch, fetchrow, get_pool
from app.models.research_memory_contracts import MemoryVersionConflict

_repository: "ResearchMemoryRepository | None" = None
_PAPER_DELETED_REASON = "证据论文已从知识库删除: {paper_id}"
_LOCK_PAPER_MEMORY_SQL = """
SELECT item.memory_id
FROM research_memory_items AS item
WHERE item.user_id=$1
  AND EXISTS (
      SELECT 1 FROM memory_evidence_refs AS ref
      WHERE ref.memory_id=item.memory_id AND ref.paper_id=$2
  )
FOR UPDATE
"""

SCHEMA_SQL = """
-- research_questions: 研究问题，作记忆的数据归属与研究目的置顶项
CREATE TABLE IF NOT EXISTS research_questions (
    question_id  TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL DEFAULT 'anonymous',
    title        TEXT NOT NULL,
    body         TEXT NOT NULL DEFAULT '',
    is_purpose   BOOLEAN NOT NULL DEFAULT FALSE,
    position     INT NOT NULL DEFAULT 0,
    source_kind  TEXT NOT NULL DEFAULT 'manual',
    review_state TEXT NOT NULL DEFAULT 'accepted',
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rq_project ON research_questions(project_id);

-- research_memory_items: 记忆条目本体，四轴状态正交
--   status(生命周期) / stale_state(新鲜度) / review_state(是否接纳) / maturity(认知成熟度)
CREATE TABLE IF NOT EXISTS research_memory_items (
    memory_id          TEXT PRIMARY KEY,
    project_id         TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    user_id            TEXT NOT NULL DEFAULT 'anonymous',
    memory_type        TEXT NOT NULL,
    title              TEXT NOT NULL,
    body               TEXT NOT NULL DEFAULT '',
    status             TEXT NOT NULL DEFAULT 'draft',
    stale_state        TEXT NOT NULL DEFAULT 'fresh',
    review_state       TEXT NOT NULL DEFAULT 'accepted',
    maturity           TEXT NOT NULL DEFAULT 'intuition',
    stale_reasons      JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_mode      TEXT NOT NULL DEFAULT 'user_note_only',
    related_paper_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
    tags               JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_kind        TEXT NOT NULL DEFAULT 'manual',
    source_session_id  TEXT,
    source_conversation_id TEXT,
    source_answer_id   TEXT,
    confidence_score   DOUBLE PRECISION,
    confidence_signals JSONB NOT NULL DEFAULT '{}'::jsonb,
    possible_duplicate BOOLEAN NOT NULL DEFAULT FALSE,
    version            INT  NOT NULL DEFAULT 1,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at TIMESTAMPTZ,
    archived_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_rmem_project  ON research_memory_items(project_id);
CREATE INDEX IF NOT EXISTS idx_rmem_type     ON research_memory_items(memory_type);
CREATE INDEX IF NOT EXISTS idx_rmem_stale    ON research_memory_items(stale_state);
CREATE INDEX IF NOT EXISTS idx_rmem_review   ON research_memory_items(review_state);
CREATE INDEX IF NOT EXISTS idx_rmem_maturity ON research_memory_items(maturity);

-- memory_question_links: 记忆与研究问题的多对多归属
CREATE TABLE IF NOT EXISTS memory_question_links (
    memory_id   TEXT NOT NULL REFERENCES research_memory_items(memory_id) ON DELETE CASCADE,
    question_id TEXT NOT NULL REFERENCES research_questions(question_id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (memory_id, question_id)
);

-- memory_evidence_refs: 记忆的证据引用，回指论文的具体 chunk 与原句
CREATE TABLE IF NOT EXISTS memory_evidence_refs (
    ref_id       TEXT PRIMARY KEY,
    memory_id    TEXT NOT NULL REFERENCES research_memory_items(memory_id) ON DELETE CASCADE,
    paper_id     TEXT NOT NULL,
    chunk_id     TEXT,
    parent_id    TEXT,
    evidence_id  TEXT,
    section      TEXT,
    section_path TEXT,
    page_num     INT,
    quote_text   TEXT NOT NULL,
    quote_hash   TEXT NOT NULL,
    excerpt      TEXT,
    claim_key    TEXT,
    ref_role     TEXT NOT NULL DEFAULT 'primary_support',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mref_memory ON memory_evidence_refs(memory_id);
CREATE INDEX IF NOT EXISTS idx_mref_paper  ON memory_evidence_refs(paper_id);

-- memory_links: 记忆间有向关系
--   based_on=src基于dst / conflicts_with=双向(存一行查两向) / supersedes=src修正dst(dst保留)
CREATE TABLE IF NOT EXISTS memory_links (
    link_id       TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    src_memory_id TEXT NOT NULL REFERENCES research_memory_items(memory_id) ON DELETE CASCADE,
    dst_memory_id TEXT NOT NULL REFERENCES research_memory_items(memory_id) ON DELETE CASCADE,
    link_type     TEXT NOT NULL,
    note          TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (src_memory_id, dst_memory_id, link_type)
);
CREATE INDEX IF NOT EXISTS idx_mlnk_src ON memory_links(src_memory_id);
CREATE INDEX IF NOT EXISTS idx_mlnk_dst ON memory_links(dst_memory_id);

-- memory_versions: 每次变更的全量快照，支撑版本回看与演进
CREATE TABLE IF NOT EXISTS memory_versions (
    version_id     TEXT PRIMARY KEY,
    memory_id      TEXT NOT NULL REFERENCES research_memory_items(memory_id) ON DELETE CASCADE,
    project_id     TEXT NOT NULL,
    version        INT  NOT NULL,
    snapshot       JSONB NOT NULL,
    change_kind    TEXT NOT NULL,
    change_summary TEXT,
    created_by     TEXT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mver_memory ON memory_versions(memory_id);
ALTER TABLE research_questions ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;
ALTER TABLE research_memory_items ADD COLUMN IF NOT EXISTS candidate_key TEXT;
ALTER TABLE research_memory_items ADD COLUMN IF NOT EXISTS source_attempt_id TEXT;
ALTER TABLE research_memory_items ADD COLUMN IF NOT EXISTS source_claim_id TEXT;
ALTER TABLE memory_evidence_refs ADD COLUMN IF NOT EXISTS source_ref JSONB;
ALTER TABLE memory_evidence_refs ADD COLUMN IF NOT EXISTS document_ref JSONB;
ALTER TABLE memory_evidence_refs ADD COLUMN IF NOT EXISTS support JSONB;
ALTER TABLE memory_evidence_refs ADD COLUMN IF NOT EXISTS verification JSONB;
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_candidate
    ON research_memory_items(project_id, user_id, candidate_key) WHERE candidate_key IS NOT NULL;
"""

# 记忆条目上会被解码的 JSONB 字段清单
_ITEM_JSON_FIELDS = ("stale_reasons", "related_paper_ids", "tags", "confidence_signals")


def _gen_id(prefix: str) -> str:
    # 各实体统一用「短前缀 + uuid 前 8 位」，便于调试与前端展示
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _normalize_item(row: Any) -> dict[str, Any] | None:
    # 把记忆条目行转普通字典，并解码其上所有 JSONB 字段
    item = normalize_record(row)
    if item is None:
        return None
    item["stale_reasons"] = load_json(item.get("stale_reasons"), [])
    item["related_paper_ids"] = load_json(item.get("related_paper_ids"), [])
    item["tags"] = load_json(item.get("tags"), [])
    item["confidence_signals"] = load_json(item.get("confidence_signals"), {})
    return item


def _normalize_version(row: Any) -> dict[str, Any] | None:
    # 版本行的 snapshot 是 JSONB，需要解码
    version = normalize_record(row)
    if version is None:
        return None
    version["snapshot"] = load_json(version.get("snapshot"), {})
    return version


def _normalize_evidence(row: Any) -> dict:
    """参数为引用行；解码精确来源与验证字段，历史缺失保持未知而不伪造核验。"""
    result = normalize_record(row) or {}
    for key in ("source_ref", "document_ref", "support", "verification"):
        result[key] = load_json(result.get(key), None)
    return result


class ResearchMemoryRepository:
    async def lock_owned_item(self, conn: Any, memory_id: str, *, user_id: str,
                              project_id: str, expected_version: int) -> dict:
        """参数为事务、资源与期望版本；锁定具体归属，迟到操作抛版本冲突。"""
        row = _normalize_item(await conn.fetchrow(
            "SELECT * FROM research_memory_items WHERE memory_id=$1 AND user_id=$2 AND project_id=$3 FOR UPDATE",
            memory_id, user_id, project_id,
        ))
        if row is None:
            raise LookupError("记忆不存在")
        if row["version"] != expected_version:
            raise MemoryVersionConflict("记忆已有新版本，请刷新后重试")
        return row

    async def change_item(self, memory_id: str, *, user_id: str, project_id: str,
                          expected_version: int, change: dict) -> dict:
        """参数为已核验的治理命令；同事务修改引用/条目/版本，返回此次提交的完整详情。"""
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            # 1. 【研究记忆】【治理提交】版本锁保护引用集合，来源锁保护验证后至提交间的授权变化
            if change.get("validated_refs"):
                from app.repositories.memory_source_repository import revalidate_memory_sources

                await revalidate_memory_sources(conn, user_id=user_id, project_id=project_id,
                                                evidence_refs=change["validated_refs"])
            if change.get("question_id"):
                question = await conn.fetchrow("SELECT question_id FROM research_questions WHERE question_id=$1 "
                    "AND user_id=$2 AND project_id=$3 FOR UPDATE", change["question_id"], user_id, project_id)
                if question is None:
                    raise LookupError("研究问题不存在")
            if change["kind"] == "delete":
                return await self._delete_memory_with_links(conn, memory_id, user_id=user_id,
                    project_id=project_id, expected_version=expected_version)
            current = await self.lock_owned_item(conn, memory_id, user_id=user_id,
                project_id=project_id, expected_version=expected_version)
            await self._apply_reference_change(conn, memory_id, change, scope={"user_id": user_id, "project_id": project_id})
            patch = change.get("patch", {})
            assignments, params = self._build_item_update(patch, current)
            await conn.execute(f"UPDATE research_memory_items SET {assignments}, version=version+1, updated_at=NOW() "
                               "WHERE memory_id=$1", memory_id, *params)
            await self._snapshot(conn, memory_id, change["kind"], change.get("summary", ""), user_id)
            return await self._transaction_detail(conn, memory_id)

    async def _delete_memory_with_links(self, conn: Any, memory_id: str, *, user_id: str,
                                        project_id: str, expected_version: int) -> dict:
        """参数为删除事务和具体版本；稳定锁定关系两端，删除后为存活端保留关系变更快照。"""
        rows = await conn.fetch("""SELECT i.* FROM research_memory_items i
            WHERE i.user_id=$2 AND i.project_id=$3 AND (i.memory_id=$1 OR EXISTS(
                SELECT 1 FROM memory_links l WHERE
                (l.src_memory_id=$1 AND l.dst_memory_id=i.memory_id) OR
                (l.dst_memory_id=$1 AND l.src_memory_id=i.memory_id)))
            ORDER BY i.memory_id FOR UPDATE""", memory_id, user_id, project_id)
        target = next((row for row in rows if row["memory_id"] == memory_id), None)
        if target is None:
            raise LookupError("记忆不存在")
        if target["version"] != expected_version:
            raise MemoryVersionConflict("记忆已有新版本，请刷新后重试")
        await conn.execute("DELETE FROM research_memory_items WHERE memory_id=$1 AND user_id=$2 AND project_id=$3",
                           memory_id, user_id, project_id)
        for row in rows:
            if row["memory_id"] != memory_id:
                await conn.execute("UPDATE research_memory_items SET version=version+1,updated_at=NOW() WHERE memory_id=$1",
                                   row["memory_id"])
                await self._snapshot(conn, row["memory_id"], "remove_link", "关联记忆已删除", user_id)
        return {"deleted": True, "memory_id": memory_id}

    async def _apply_reference_change(self, conn: Any, memory_id: str, change: dict, *, scope: dict) -> None:
        """参数为当前记忆锁与治理命令；引用和问题关联仅在该事务内变更。"""
        if change.get("add_ref") is not None:
            await self._insert_evidence(conn, memory_id, change["add_ref"])
        if change.get("remove_ref_id"):
            removed = await conn.fetchrow(
                "DELETE FROM memory_evidence_refs WHERE ref_id=$1 AND memory_id=$2 RETURNING ref_id",
                change["remove_ref_id"], memory_id,
            )
            if removed is None:
                raise LookupError("证据引用不存在")
        for ref in change.get("validated_refs", []):
            if ref.get("ref_id"):
                await conn.execute("UPDATE memory_evidence_refs SET source_ref=$3::jsonb, document_ref=$4::jsonb, "
                    "support=$5::jsonb, verification=$6::jsonb WHERE ref_id=$1 AND memory_id=$2",
                    ref["ref_id"], memory_id, *[dump_json(ref.get(key)) for key in
                                              ("source_ref", "document_ref", "support", "verification")])
        question_id = change.get("question_id")
        if question_id:
            question = await conn.fetchrow("SELECT question_id FROM research_questions WHERE question_id=$1 "
                "AND user_id=$2 AND project_id=$3 FOR UPDATE", question_id, scope["user_id"], scope["project_id"])
            if question is None:
                raise LookupError("研究问题不存在")
            if change["kind"] == "attach_question":
                await conn.execute("INSERT INTO memory_question_links(memory_id,question_id) VALUES($1,$2) "
                                   "ON CONFLICT DO NOTHING", memory_id, question_id)
            else:
                await conn.execute("DELETE FROM memory_question_links WHERE memory_id=$1 AND question_id=$2",
                                   memory_id, question_id)

    async def _insert_evidence(self, conn: Any, memory_id: str, payload: dict) -> dict:
        """参数为事务与已核对引用；保留原文身份和支持范围，返回新增引用。"""
        row = await conn.fetchrow(
            """INSERT INTO memory_evidence_refs(ref_id,memory_id,paper_id,chunk_id,parent_id,evidence_id,
                section,section_path,page_num,quote_text,quote_hash,excerpt,claim_key,ref_role,
                source_ref,document_ref,support,verification)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15::jsonb,$16::jsonb,$17::jsonb,$18::jsonb)
                RETURNING *""", _gen_id("mref"), memory_id, payload["paper_id"], payload.get("chunk_id"),
            payload.get("parent_id"), payload.get("evidence_id"), payload.get("section"), payload.get("section_path"),
            payload.get("page_num"), payload["quote_text"], payload["quote_hash"], payload.get("excerpt"),
            payload.get("claim_key"), payload.get("ref_role", "primary_support"),
            *[dump_json(payload.get(key)) for key in ("source_ref", "document_ref", "support", "verification")],
        )
        return normalize_record(row) or {}

    async def _transaction_detail(self, conn: Any, memory_id: str) -> dict:
        """参数为已持有记忆锁的事务；返回同一提交版本的条目、引用、关系和历史。"""
        item = _normalize_item(await conn.fetchrow("SELECT * FROM research_memory_items WHERE memory_id=$1", memory_id))
        return {"item": item,
                "evidence_refs": [_normalize_evidence(row) for row in await conn.fetch(
                    "SELECT * FROM memory_evidence_refs WHERE memory_id=$1 ORDER BY created_at", memory_id)],
                "links": [normalize_record(row) for row in await conn.fetch(
                    "SELECT * FROM memory_links WHERE src_memory_id=$1 OR dst_memory_id=$1", memory_id)],
                "question_ids": [row["question_id"] for row in await conn.fetch(
                    "SELECT question_id FROM memory_question_links WHERE memory_id=$1", memory_id)],
                "versions": [_normalize_version(row) for row in await conn.fetch(
                    "SELECT * FROM memory_versions WHERE memory_id=$1 ORDER BY version", memory_id)]}

    async def change_question(self, question_id: str, *, project_id: str, user_id: str,
                               expected_version: int, patch: dict | None) -> dict:
        """参数为问题、归属与版本；锁定后更新/删除，拒绝跨项目对象和迟到写入。"""
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.fetchrow("SELECT project_id FROM projects WHERE project_id=$1 AND user_id=$2 FOR UPDATE",
                                project_id, user_id)
            row = await conn.fetchrow("SELECT * FROM research_questions WHERE question_id=$1 "
                "AND project_id=$2 AND user_id=$3 FOR UPDATE", question_id, project_id, user_id)
            if row is None:
                raise LookupError("研究问题不存在")
            if row["version"] != expected_version:
                raise MemoryVersionConflict("研究问题已有新版本，请刷新后重试")
            if patch is None:
                memories = await conn.fetch("SELECT i.* FROM research_memory_items i JOIN memory_question_links q "
                    "ON q.memory_id=i.memory_id WHERE q.question_id=$1 AND i.project_id=$2 AND i.user_id=$3 "
                    "ORDER BY i.memory_id FOR UPDATE OF i", question_id, project_id, user_id)
                await conn.execute("DELETE FROM research_questions WHERE question_id=$1", question_id)
                for memory in memories:
                    await conn.execute("UPDATE research_memory_items SET version=version+1,updated_at=NOW() WHERE memory_id=$1",
                                       memory["memory_id"])
                    await self._snapshot(conn, memory["memory_id"], "detach_question", "研究问题已删除", user_id)
                return {"status": "deleted"}
            if patch.get("is_purpose"):
                other = await conn.fetchrow("SELECT question_id FROM research_questions WHERE project_id=$1 "
                    "AND is_purpose=TRUE AND question_id<>$2", project_id, question_id)
                if other is not None:
                    raise ValueError("项目已存在研究目的")
            fields = ("title", "body", "is_purpose", "position")
            values = [patch.get(key, row[key]) for key in fields]
            updated = await conn.fetchrow("UPDATE research_questions SET title=$2,body=$3,is_purpose=$4,position=$5, "
                "version=version+1,updated_at=NOW() WHERE question_id=$1 RETURNING *", question_id, *values)
            return normalize_record(updated) or {}

    async def change_link(self, *, project_id: str, user_id: str, expected_version: int,
                           dst_expected_version: int, payload: dict) -> dict:
        """参数为关系与两端版本；稳定顺序锁两端，关联变更和双端快照原子提交。"""
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            link = payload
            if payload.get("link_id"):
                link = await conn.fetchrow("SELECT * FROM memory_links WHERE link_id=$1 AND project_id=$2",
                                          payload["link_id"], project_id)
                if link is None:
                    raise LookupError("记忆关系不存在")
            versions = {link["src_memory_id"]: expected_version, link["dst_memory_id"]: dst_expected_version}
            for memory_id in sorted(versions):
                await self.lock_owned_item(conn, memory_id, user_id=user_id, project_id=project_id,
                                           expected_version=versions[memory_id])
            if payload.get("link_id"):
                await conn.execute("DELETE FROM memory_links WHERE link_id=$1", payload["link_id"])
                result = {"status": "deleted"}
            else:
                result = normalize_record(await conn.fetchrow("INSERT INTO memory_links "
                    "(link_id,project_id,src_memory_id,dst_memory_id,link_type,note) VALUES($1,$2,$3,$4,$5,$6) "
                    "ON CONFLICT(src_memory_id,dst_memory_id,link_type) DO UPDATE SET note=EXCLUDED.note RETURNING *",
                    _gen_id("mlnk"), project_id, link["src_memory_id"], link["dst_memory_id"], link["link_type"], link.get("note", ""))) or {}
            for memory_id in sorted(versions):
                await conn.execute("UPDATE research_memory_items SET version=version+1,updated_at=NOW() WHERE memory_id=$1",
                                   memory_id)
                await self._snapshot(conn, memory_id, "remove_link" if payload.get("link_id") else "add_link", "", user_id)
            return {**result, "version": expected_version + 1, "dst_version": dst_expected_version + 1}

    async def ensure_schema(self) -> None:
        # 仓储层负责保证研究记忆功能运行所需的 6 张表结构存在，幂等执行
        await execute(SCHEMA_SQL)

    # ─── 研究问题 CRUD ───

    async def create_question(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """参数为可信项目与问题；项目行锁内保证研究目的唯一，并保存明确 owner。"""
        user_id = payload.get("user_id")
        if not user_id:
            raise ValueError("研究问题必须具有用户归属")
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            project = await conn.fetchrow("SELECT project_id FROM projects WHERE project_id=$1 AND user_id=$2 FOR UPDATE",
                                          project_id, user_id)
            if project is None:
                raise LookupError("研究项目不存在")
            if payload.get("is_purpose"):
                existing = await conn.fetchrow("SELECT question_id FROM research_questions WHERE project_id=$1 AND is_purpose=TRUE",
                                              project_id)
                if existing is not None:
                    raise ValueError("该项目已存在研究目的")
            row = await conn.fetchrow("INSERT INTO research_questions "
                "(question_id,project_id,user_id,title,body,is_purpose,position,source_kind,review_state) "
                "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *", _gen_id("rq"), project_id, user_id,
                payload["title"], payload.get("body", ""), bool(payload.get("is_purpose")), payload.get("position", 0),
                payload.get("source_kind", "manual"), payload.get("review_state", "accepted"))
            return normalize_record(row) or {}

    async def get_question(self, question_id: str, *, user_id: str | None = None,
                           project_id: str | None = None) -> dict[str, Any] | None:
        """参数为问题与可信范围；内部审计可不限定，用户整理同时传owner/project。"""
        return normalize_record(await fetchrow("SELECT * FROM research_questions WHERE question_id=$1 "
            "AND ($2::text IS NULL OR user_id=$2) AND ($3::text IS NULL OR project_id=$3)",
            question_id, user_id, project_id))

    async def list_questions(self, project_id: str, *, user_id: str | None = None, review_state: str | None = "accepted") -> list[dict[str, Any]]:
        # 研究目的置顶，其余按 position 再按创建时间排序
        if review_state:
            rows = await fetch(
                "SELECT * FROM research_questions WHERE project_id=$1 AND review_state=$2 AND ($3::text IS NULL OR user_id=$3) "
                "ORDER BY is_purpose DESC, position ASC, created_at ASC",
                project_id,
                review_state, user_id,
            )
        else:
            rows = await fetch(
                "SELECT * FROM research_questions WHERE project_id=$1 AND ($2::text IS NULL OR user_id=$2) "
                "ORDER BY is_purpose DESC, position ASC, created_at ASC",
                project_id, user_id,
            )
        return [normalize_record(row) or {} for row in rows]

    async def find_purpose_question(self, project_id: str) -> dict[str, Any] | None:
        # 研究目的 singleton 查询，用于 service 层保证每项目至多一条
        return normalize_record(
            await fetchrow(
                "SELECT * FROM research_questions WHERE project_id=$1 AND is_purpose=TRUE LIMIT 1",
                project_id,
            )
        )



    # ─── 记忆条目：创建与读取 ───

    async def create_item(self, project_id: str, payload: dict[str, Any], *,
                          evidence_refs: list[dict] | None = None, candidate_key: str | None = None) -> dict[str, Any]:
        """
        新建记忆条目并落首个版本快照（version=1, change_kind=create），整体在一个事务内。

        参数:
            project_id: 归属项目
            payload: 记忆各字段（title/memory_type 必填，其余走默认）
        返回:
            含证据/关系/归属/版本摘要的详情字典
        """
        memory_id = _gen_id("mem")
        if not payload.get("user_id"):
            raise ValueError("研究记忆必须具有可信用户归属")
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                if payload.get("source_delivery_revision") is not None:
                    await self._lock_candidate_delivery(conn, payload)
                if evidence_refs:
                    from app.repositories.memory_source_repository import revalidate_memory_sources

                    await revalidate_memory_sources(conn, user_id=payload["user_id"],
                                                    project_id=project_id, evidence_refs=evidence_refs)
                project = await conn.fetchrow("SELECT project_id FROM projects WHERE project_id=$1 AND user_id=$2 FOR UPDATE",
                                              project_id, payload["user_id"])
                if project is None:
                    raise LookupError("研究项目不存在")
                if candidate_key:
                    existing = await conn.fetchrow("SELECT memory_id FROM research_memory_items "
                        "WHERE project_id=$1 AND user_id=$2 AND candidate_key=$3", project_id, payload["user_id"], candidate_key)
                    if existing is not None:
                        return await self._transaction_detail(conn, existing["memory_id"])
                await conn.execute(
                    """
                    INSERT INTO research_memory_items (
                        memory_id, project_id, user_id, memory_type, title, body,
                        status, stale_state, review_state, maturity,
                        evidence_mode, related_paper_ids, tags,
                        source_kind, source_session_id, source_conversation_id, source_answer_id,
                        confidence_score, confidence_signals, possible_duplicate, version,
                        candidate_key,source_attempt_id,source_claim_id,stale_reasons
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13::jsonb,
                            $14,$15,$16,$17,$18,$19::jsonb,$20,1,$21,$22,$23,$24::jsonb)
                    """,
                    memory_id, project_id,
                    payload["user_id"],
                    payload["memory_type"], payload["title"], payload.get("body") or "",
                    payload.get("status") or "draft",
                    payload.get("stale_state") or "fresh",
                    payload.get("review_state") or "accepted",
                    payload.get("maturity") or "intuition",
                    payload.get("evidence_mode") or "user_note_only",
                    dump_json(payload.get("related_paper_ids") or []),
                    dump_json(payload.get("tags") or []),
                    payload.get("source_kind") or "manual",
                    payload.get("source_session_id"),
                    payload.get("source_conversation_id"),
                    payload.get("source_answer_id"),
                    payload.get("confidence_score"),
                    dump_json(payload.get("confidence_signals") or {}),
                    bool(payload.get("possible_duplicate", False)),
                    candidate_key, payload.get("source_attempt_id"), payload.get("source_claim_id"),
                    dump_json(payload.get("stale_reasons") or []),
                )
                for ref in evidence_refs or []:
                    await self._insert_evidence(conn, memory_id, ref)
                if payload.get("question_id"):
                    await self._apply_reference_change(conn, memory_id,
                        {"kind": "attach_question", "question_id": payload["question_id"]},
                        scope={"user_id": payload["user_id"], "project_id": project_id})
                await self._snapshot(conn, memory_id, "create", payload.get("change_summary") or "", payload.get("user_id"))
                return await self._transaction_detail(conn, memory_id)

    async def create_candidate(self, project_id: str, payload: dict, evidence_refs: list[dict], *, candidate_key: str) -> dict:
        """参数为已提交主张、精确引用和幂等身份；同事务保存候选、引用及初始完整快照。"""
        if not candidate_key or not payload.get("source_answer_id"):
            raise ValueError("候选缺少来源操作或幂等身份")
        return await self.create_item(project_id, {**payload, "status": "draft", "review_state": "pending_review"},
                                      evidence_refs=evidence_refs, candidate_key=candidate_key)

    async def _lock_candidate_delivery(self, conn: Any, payload: dict) -> None:
        """参数为候选事务与生产版本；最终重试已提交新答案时，旧模型提议不得再建立候选。"""
        source = await conn.fetchrow("SELECT * FROM rag_runs WHERE run_id=$1 AND user_id=$2 FOR SHARE",
                                    payload["source_answer_id"], payload["user_id"])
        metadata = load_json(source["metadata"], {}) if source is not None else {}
        if source is None or source["delivery_revision"] != payload["source_delivery_revision"] or (
                metadata.get("final_generation_attempt") or {}).get("attempt_id") != payload.get("source_attempt_id"):
            raise MemoryVersionConflict("来源答案已有新版本，旧候选不能写入")

    async def get_item(self, memory_id: str, *, user_id: str | None = None,
                       project_id: str | None = None) -> dict[str, Any] | None:
        """参数为记忆与可选内部审计范围；用户服务同时传入 owner/project，在查询条件隔离。"""
        return _normalize_item(await fetchrow(
            "SELECT * FROM research_memory_items WHERE memory_id=$1 "
            "AND ($2::text IS NULL OR user_id=$2) AND ($3::text IS NULL OR project_id=$3)",
            memory_id, user_id, project_id,
        ))

    async def get_detail(self, memory_id: str, *, user_id: str | None = None,
                         project_id: str | None = None) -> dict[str, Any] | None:
        """聚合记忆详情：条目本体 + 证据 + 关系 + 归属问题 + 版本摘要。"""
        item = await self.get_item(memory_id, user_id=user_id, project_id=project_id)
        if item is None:
            return None
        return {
            "item": item,
            "evidence_refs": await self.list_evidence_refs(memory_id),
            "links": await self.list_links_of_memory(memory_id, user_id=item["user_id"], project_id=item["project_id"]),
            "question_ids": await self.list_question_ids(memory_id, user_id=item["user_id"], project_id=item["project_id"]),
            "versions": await self.list_versions(memory_id),
        }

    async def list_items(
        self,
        project_id: str,
        *,
        user_id: str | None = None,
        review_state: str | None = "accepted",
        question_id: str | None = None,
        memory_type: str | None = None,
        evidence_mode: str | None = None,
        paper_id: str | None = None,
        tag: str | None = None,
        status: str | None = None,
        stale_state: str | None = None,
        maturity: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        按多维 filter 列出记忆条目；默认只返回已接纳（accepted）的正式记忆。

        参数:
            project_id: 归属项目，跨项目记忆不互相串
            user_id: 记忆归属人；注入记忆前要收窄到当前用户，不传则不限归属人
            review_state/memory_type/status/stale_state/maturity: 记忆条目本体上的等值过滤
            question_id: 按归属研究问题过滤，走 memory_question_links 关联
            paper_id/tag: JSONB 数组包含过滤
        返回:
            按最近更新倒序排列的记忆条目列表
        """
        where = ["i.project_id = $1"]
        params: list[Any] = [project_id]
        filters: list[tuple[str, Any]] = [
            ("i.user_id", user_id),
            ("i.review_state", review_state),
            ("i.memory_type", memory_type),
            ("i.evidence_mode", evidence_mode),
            ("i.status", status),
            ("i.stale_state", stale_state),
            ("i.maturity", maturity),
        ]
        for column, value in filters:
            if value is not None:
                params.append(value)
                where.append(f"{column} = ${len(params)}")
        params, where = self._apply_jsonb_filters(params, where, paper_id, tag)
        join = ""
        if question_id is not None:
            params.append(question_id)
            join = "JOIN memory_question_links q ON q.memory_id = i.memory_id AND q.question_id = $" + str(len(params))
        rows = await fetch(
            f"SELECT i.* FROM research_memory_items i {join} "
            f"WHERE {' AND '.join(where)} ORDER BY i.updated_at DESC",
            *params,
        )
        return [_normalize_item(row) or {} for row in rows]

    async def list_items_by_source_answer(
        self, project_id: str, *, user_id: str, source_answer_id: str
    ) -> list[dict[str, Any]]:
        """
        取某一次回答已落的记忆条目，供候选写回做幂等去重。

        与 list_items 分开是因为这里查的是「本轮来源身份」而不是资格过滤，
        且不受 review_state 默认收窄影响：写回要看到的正是尚未被人接纳的草稿。

        参数:
            project_id: 归属项目，跨项目不互串
            user_id: 记忆归属人，跨用户不互串
            source_answer_id: 来源回答身份（写回用 request_id）；本方法不对空值特判，
                              是否值得查由调用方判断
        返回:
            该次回答已落的记忆条目列表，按创建时间正序
        """
        rows = await fetch(
            "SELECT * FROM research_memory_items "
            "WHERE project_id=$1 AND user_id=$2 AND source_answer_id=$3 "
            "ORDER BY created_at ASC",
            project_id,
            user_id,
            source_answer_id,
        )
        return [_normalize_item(row) or {} for row in rows]

    async def list_review_queue(self, project_id: str, *, order_by_confidence: bool = True) -> list[dict[str, Any]]:
        """
        列出待审草稿（review_state=pending_review），默认按置信度降序排。

        参数:
            project_id: 归属项目
            order_by_confidence: True 时按 confidence_score 降序（低置信度沉底待重点核对），
                                 否则按创建时间倒序
        返回:
            待审草稿条目列表
        """
        # confidence_score 可能为 NULL，用 NULLS LAST 保证有分的排前面
        order = "i.confidence_score DESC NULLS LAST, i.created_at DESC" if order_by_confidence else "i.created_at DESC"
        rows = await fetch(
            f"SELECT i.* FROM research_memory_items i "
            f"WHERE i.project_id=$1 AND i.review_state='pending_review' ORDER BY {order}",
            project_id,
        )
        return [_normalize_item(row) or {} for row in rows]

    @staticmethod
    def _apply_jsonb_filters(
        params: list[Any], where: list[str], paper_id: str | None, tag: str | None
    ) -> tuple[list[Any], list[str]]:
        # related_paper_ids / tags 是 JSONB 数组，用 @> 包含判断做过滤
        if paper_id is not None:
            params.append(dump_json([paper_id]))
            where.append(f"i.related_paper_ids @> ${len(params)}::jsonb")
        if tag is not None:
            params.append(dump_json([tag]))
            where.append(f"i.tags @> ${len(params)}::jsonb")
        return params, where

    async def search_memories_literal(
        self, project_id: str, keyword: str, *, user_id: str | None = None, review_state: str = "accepted", stale_state: str | None = None
    ) -> list[dict[str, Any]]:
        """
        B3 第一期字面过滤：按 body/title 的 ILIKE 命中召回记忆。

        参数:
            project_id: 归属项目（跨项目不串）
            keyword: 搜索词，做 ILIKE 模糊匹配
            review_state: 默认只搜已接纳记忆
            stale_state: 可选按新鲜度过滤
        返回:
            命中的记忆条目列表
        """
        pattern = f"%{keyword}%"
        where = ["project_id = $1", "review_state = $2", "(title ILIKE $3 OR body ILIKE $3)"]
        params: list[Any] = [project_id, review_state, pattern]
        if user_id is not None:
            params.append(user_id)
            where.append("user_id=$4")
        if stale_state is not None:
            params.append(stale_state)
            where.append(f"stale_state = ${len(params)}")
        rows = await fetch(
            f"SELECT * FROM research_memory_items WHERE {' AND '.join(where)} ORDER BY updated_at DESC",
            *params,
        )
        return [_normalize_item(row) or {} for row in rows]

    async def count_by_maturity(self, project_id: str, *, user_id: str | None = None, review_state: str = "accepted") -> dict[str, int]:
        """按 maturity 聚合计数，供论证台账顶部状态条使用。"""
        rows = await fetch(
            "SELECT maturity, COUNT(*) AS n FROM research_memory_items "
            "WHERE project_id=$1 AND review_state=$2 AND ($3::text IS NULL OR user_id=$3) GROUP BY maturity",
            project_id,
            review_state, user_id,
        )
        return {str(row["maturity"]): int(row["n"]) for row in rows}

    async def list_evidence_backed_items(self, project_id: str) -> list[dict[str, Any]]:
        """列出项目下所有 evidence_backed 记忆，供 B4 证据失效检测遍历。"""
        rows = await fetch(
            "SELECT * FROM research_memory_items WHERE project_id=$1 AND evidence_mode='evidence_backed'",
            project_id,
        )
        return [_normalize_item(row) or {} for row in rows]

    # ─── 记忆条目：变更（均在事务内 version+1 + 快照） ───


    @staticmethod
    def _build_item_update(patch: dict[str, Any], current: dict[str, Any]) -> tuple[str, list[Any]]:
        # 把 patch 里出现的字段拼成 SET 子句（$2 起，$1 留给 memory_id），JSONB 字段带 ::jsonb
        assignments: list[str] = []
        params: list[Any] = []
        editable = (
            "memory_type", "title", "body", "status", "stale_state", "review_state",
            "maturity", "evidence_mode", "source_kind", "confidence_score", "possible_duplicate",
        )
        for field in editable:
            if field in patch:
                params.append(patch[field])
                assignments.append(f"{field}=${len(params) + 1}")
        for field in ("stale_reasons", "related_paper_ids", "tags", "confidence_signals"):
            if field in patch:
                params.append(dump_json(patch[field]))
                assignments.append(f"{field}=${len(params) + 1}::jsonb")
        # status 流转到 confirmed/archived 时自动打对应时间戳，避免上层漏设
        if patch.get("status") == "confirmed":
            assignments.append("confirmed_at=NOW()")
        elif patch.get("status") == "archived":
            assignments.append("archived_at=COALESCE(archived_at, NOW())")
        if not assignments:
            # 无字段变化时也要推进版本（如仅记录一次动作），用恒等赋值占位
            assignments.append("memory_id=memory_id")
        return ", ".join(assignments), params



    # ─── 证据引用 CRUD ───


    async def list_evidence_refs(self, memory_id: str) -> list[dict[str, Any]]:
        rows = await fetch(
            "SELECT * FROM memory_evidence_refs WHERE memory_id=$1 ORDER BY created_at ASC", memory_id
        )
        return [_normalize_evidence(row) for row in rows]





    async def list_orphan_evidence_refs(self, valid_paper_ids: set[str]) -> list[dict[str, Any]]:
        """
        列出引用了业务库不存在论文的研究记忆证据。

        Args:
            valid_paper_ids: 业务库全量论文主键
        Returns:
            包含证据、记忆、论文和记忆 owner 的孤儿引用
        """
        rows = await fetch(
            """
            SELECT ref.ref_id, ref.memory_id, ref.paper_id, item.user_id
            FROM memory_evidence_refs AS ref
            JOIN research_memory_items AS item ON item.memory_id=ref.memory_id
            WHERE NOT (ref.paper_id = ANY($1::text[]))
            ORDER BY ref.ref_id
            """,
            sorted(valid_paper_ids),
        )
        return [normalize_record(row) or {} for row in rows]

    async def remove_orphan_evidence_refs(self, refs: list[dict[str, Any]]) -> dict[str, Any]:
        """
        在一个事务中按孤儿论文和 owner 清理证据，并复用记忆失效合同。

        Args:
            refs: list_orphan_evidence_refs 返回的待清理引用
        Returns:
            删除引用数、更新记忆和失去全部证据的记忆 ID
        """
        targets = sorted(
            {
                (str(ref.get("paper_id") or "").strip(), str(ref.get("user_id") or "").strip())
                for ref in refs
                if str(ref.get("paper_id") or "").strip() and str(ref.get("user_id") or "").strip()
            }
        )
        if not targets:
            return {"deleted_evidence_refs": 0, "updated_memory_ids": [], "stale_memory_ids": []}
        pool = await get_pool()
        deleted_count = 0
        updated_memory_ids: set[str] = set()
        stale_memory_ids: set[str] = set()
        async with pool.acquire() as connection:
            async with connection.transaction():
                for paper_id, user_id in targets:
                    result = await self.remove_paper_references(
                        paper_id,
                        user_id=user_id,
                        connection=connection,
                    )
                    deleted_count += int(result["deleted_evidence_refs"])
                    updated_memory_ids.update(result["updated_memory_ids"])
                    stale_memory_ids.update(result["stale_memory_ids"])
        return {
            "deleted_evidence_refs": deleted_count,
            "updated_memory_ids": sorted(updated_memory_ids),
            "stale_memory_ids": sorted(stale_memory_ids),
        }

    async def remove_paper_references(self, paper_id: str, *, user_id: str, connection: Any) -> dict[str, Any]:
        """参数为来源失效事务；保留历史引用，按声明作用维护失效及版本，不按剩余引用数量猜测。"""
        from app.repositories.memory_source_repository import invalidate_source

        rows = await connection.fetch(_LOCK_PAPER_MEMORY_SQL, user_id, paper_id)
        reason = _PAPER_DELETED_REASON.format(paper_id=paper_id)
        await invalidate_source(connection, user_id=user_id, paper_id=paper_id, reason=reason)
        memory_ids = [str(row["memory_id"]) for row in rows]
        stale = await connection.fetch("SELECT memory_id FROM research_memory_items "
            "WHERE user_id=$1 AND memory_id=ANY($2::text[]) AND stale_state='needs_rebind'", user_id, memory_ids)
        return {"deleted_evidence_refs": 0, "updated_memory_ids": memory_ids,
                "stale_memory_ids": [str(row["memory_id"]) for row in stale]}

    # ─── 记忆关系 CRUD ───


    async def list_links(self, project_id: str, *, user_id: str | None = None) -> list[dict[str, Any]]:
        # 关系图谱数据源：项目下全部记忆关系
        rows = await fetch("SELECT l.* FROM memory_links l JOIN research_memory_items src ON src.memory_id=l.src_memory_id "
            "JOIN research_memory_items dst ON dst.memory_id=l.dst_memory_id WHERE l.project_id=$1 "
            "AND src.project_id=$1 AND dst.project_id=$1 AND ($2::text IS NULL OR (src.user_id=$2 AND dst.user_id=$2)) "
            "ORDER BY l.created_at ASC", project_id, user_id)
        return [normalize_record(row) or {} for row in rows]

    async def list_links_of_memory(self, memory_id: str, *, user_id: str | None = None,
                                   project_id: str | None = None) -> list[dict[str, Any]]:
        # 某条记忆的关系：src 或 dst 命中都算（conflicts_with 单行存双向查）
        rows = await fetch(
            "SELECT l.* FROM memory_links l JOIN research_memory_items src ON src.memory_id=l.src_memory_id "
            "JOIN research_memory_items dst ON dst.memory_id=l.dst_memory_id "
            "WHERE (l.src_memory_id=$1 OR l.dst_memory_id=$1) AND ($2::text IS NULL OR (src.user_id=$2 AND dst.user_id=$2)) "
            "AND ($3::text IS NULL OR (src.project_id=$3 AND dst.project_id=$3)) ORDER BY l.created_at ASC",
            memory_id, user_id, project_id,
        )
        return [normalize_record(row) or {} for row in rows]


    # ─── 记忆 ↔ 研究问题 归属 ───



    async def list_question_ids(self, memory_id: str, *, user_id: str | None = None,
                                project_id: str | None = None) -> list[str]:
        """参数为记忆与归属；历史错误关联不能让外项目问题进入当前详情。"""
        rows = await fetch("SELECT q.question_id FROM memory_question_links l JOIN research_questions q "
            "ON q.question_id=l.question_id WHERE l.memory_id=$1 AND ($2::text IS NULL OR q.user_id=$2) "
            "AND ($3::text IS NULL OR q.project_id=$3)", memory_id, user_id, project_id)
        return [str(row["question_id"]) for row in rows]

    # ─── 版本 ───

    async def list_versions(self, memory_id: str) -> list[dict[str, Any]]:
        rows = await fetch(
            "SELECT * FROM memory_versions WHERE memory_id=$1 ORDER BY version ASC", memory_id
        )
        return [_normalize_version(row) or {} for row in rows]

    async def _snapshot(
        self, conn: Any, memory_id: str, change_kind: str, change_summary: str, created_by: str | None
    ) -> None:
        """
        在当前事务内读取记忆完整状态，写一条 memory_versions 全量快照。

        参数:
            conn: 当前事务连接（必须复用，脱离事务会破坏原子性）
            memory_id: 目标记忆
            change_kind: 变更类型
            change_summary: 变更说明
            created_by: 发起人
        """
        item_row = await conn.fetchrow("SELECT * FROM research_memory_items WHERE memory_id=$1", memory_id)
        item = _normalize_item(item_row)
        if item is None:
            raise LookupError("记忆版本快照缺少对应条目")
        # 快照含条目本体 + 证据 + 归属问题 + 关系，保证版本可完整回看
        evidence_rows = await conn.fetch("SELECT * FROM memory_evidence_refs WHERE memory_id=$1", memory_id)
        question_rows = await conn.fetch("SELECT question_id FROM memory_question_links WHERE memory_id=$1", memory_id)
        link_rows = await conn.fetch(
            "SELECT * FROM memory_links WHERE src_memory_id=$1 OR dst_memory_id=$1", memory_id
        )
        snapshot = {
            "item": item,
            "evidence_refs": [_normalize_evidence(r) for r in evidence_rows],
            "question_ids": [str(r["question_id"]) for r in question_rows],
            "links": [normalize_record(r) for r in link_rows],
        }
        await conn.execute(
            """
            INSERT INTO memory_versions (version_id, memory_id, project_id, version, snapshot, change_kind, change_summary, created_by)
            VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8)
            """,
            _gen_id("mver"), memory_id, item["project_id"], int(item["version"]),
            dump_json(snapshot), change_kind, change_summary, created_by,
        )


def get_research_memory_repository() -> ResearchMemoryRepository:
    """返回研究记忆仓储单例。"""
    global _repository
    if _repository is None:
        _repository = ResearchMemoryRepository()
    return _repository
