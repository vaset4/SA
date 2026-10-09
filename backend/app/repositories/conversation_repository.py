"""
前端会话与消息仓储。

这一层直接对 `conversations` / `messages` 两张表负责，
是前端展示库最贴近数据库的一层。

阅读时建议重点关注三件事：
1. 用户隔离是如何在 SQL 层落实的
2. 会话元数据和消息明细是如何拆表的
3. 为什么 `set_messages` 使用覆盖式写法
"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.repositories.common import dump_json, load_json, normalize_record, parse_datetime, utc_now
from app.services.database import execute, fetch, fetchrow
from app.models.result_delivery_contracts import DeliveryConflict

_SORT_FIELDS = {"updated_at", "created_at", "title", "message_count"}
_repository: "ConversationRepository | None" = None
SERVER_DELIVERY_FIELDS = frozenset({
    "persistence_status", "delivery_revision", "producer_run_id", "final_generation_attempt",
    "final_delivery_report", "final_citation_bindings", "final_delivery_applicability",
})


def _conversation_data(row: Any) -> dict[str, Any]:
    # repository 层负责把 JSONB / 布尔等数据库表示转回应用层友好结构。
    data = normalize_record(row) or {}
    data["is_pinned"] = bool(data.get("is_pinned"))
    data["tags"] = load_json(data.get("tags"), [])
    data["associated_papers"] = load_json(data.get("associated_papers"), [])
    return data


def _message_data(row: Any) -> dict[str, Any]:
    data = normalize_record(row) or {}
    metadata = load_json(data.pop("metadata", None), {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {**metadata, **data}


class ConversationRepository:
    """封装会话展示层持久化。"""

    async def commit_delivery(self, *, connection, user_id: str, target: dict,
                              answer: str, metadata: dict) -> dict | None:
        """【研究交付】【消息提交】参数为同事务连接、目标及整份答案；返回对应助手消息。

        用户问题按原运行身份幂等写入，重试更新明确关联消息，不按最后一条消息猜测目标。
        """
        conversation_id = target.get("conversation_id")
        if not conversation_id:
            return None
        conversation = await connection.fetchrow(
            "SELECT id FROM conversations WHERE user_id=$1 AND id=$2 FOR UPDATE", user_id, conversation_id,
        )
        if conversation is None:
            raise LookupError("结果关联的对话不存在")
        run_id = str(target["run_id"])
        now = utc_now()
        # 1. 【研究交付】【问题幂等】固定用户问题身份，补存和最终重试不新增提问
        if target.get("operation_kind") is not None:
            await connection.execute(
                """INSERT INTO messages (id,conversation_id,role,content,metadata,created_at)
                    VALUES ($1,$2,'user',$3,$4::jsonb,$5) ON CONFLICT (id) DO NOTHING""",
                str(uuid5(NAMESPACE_URL, f"rag-delivery:{run_id}:user")), conversation_id,
                target["query"], dump_json({"run_id": run_id, "delivery_command_id": run_id}), parse_datetime(target["created_at"]),
            )
        existing = await connection.fetchrow(
            """SELECT id FROM messages WHERE conversation_id=$1 AND role='assistant'
                AND metadata->>'run_id'=$2 ORDER BY created_at ASC LIMIT 1 FOR UPDATE""",
            conversation_id, run_id,
        )
        message_id = existing["id"] if existing else str(uuid5(NAMESPACE_URL, f"rag-delivery:{run_id}:assistant"))
        # 2. 【研究交付】【整体消息】整份替换交付元数据，空正文也属于本次真实结果
        message = await connection.fetchrow(
            """INSERT INTO messages (id,conversation_id,role,content,render_mode,metadata,created_at)
                VALUES ($1,$2,'assistant',$3,'markdown',$4::jsonb,$5)
                ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content,
                    render_mode=EXCLUDED.render_mode,metadata=EXCLUDED.metadata RETURNING *""",
            message_id, conversation_id, answer, dump_json(metadata), now,
        )
        await connection.execute(
            """UPDATE conversations SET session_id=COALESCE(session_id,$3), updated_at=$4,
                message_count=(SELECT COUNT(*) FROM messages WHERE conversation_id=$2)
                WHERE user_id=$1 AND id=$2""", user_id, conversation_id, target.get("session_id"), now,
        )
        return _message_data(message)

    async def get_previous_delivery(self, *, user_id: str, conversation_id: str) -> dict | None:
        """【研究交付】【上一轮】参数为用户及对话；按提问持久顺序读取已保存结果，不受重试时间影响。"""
        row = await fetchrow(
            """SELECT r.* FROM rag_runs r JOIN conversations c ON c.id=r.conversation_id
                WHERE c.user_id=$1 AND r.user_id=$1 AND c.id=$2
                  AND r.target_run_id IS NULL AND r.answer IS NOT NULL
                  AND (r.metadata->>'persistence_status'='saved'
                       OR (r.operation_kind IS NULL AND r.status='completed'))
                ORDER BY r.created_at DESC,r.run_id DESC LIMIT 1""", user_id, conversation_id,
        )
        data = normalize_record(row)
        if data is not None:
            data["metadata"] = load_json(data.get("metadata"), {})
        return data

    async def ensure_schema(self) -> None:
        # 这里统一做建表 + 轻量迁移，保证旧环境也能平滑升级。
        await execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                session_id TEXT,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                is_pinned BOOLEAN DEFAULT FALSE,
                tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                associated_papers JSONB NOT NULL DEFAULT '[]'::jsonb,
                message_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                title TEXT,
                tone TEXT,
                render_mode TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL
            );
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'anonymous';
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS session_id TEXT;
            UPDATE conversations SET user_id = 'anonymous' WHERE user_id IS NULL OR user_id = '';
            ALTER TABLE messages ADD COLUMN IF NOT EXISTS render_mode TEXT;
            ALTER TABLE messages ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_conversations_pinned ON conversations(is_pinned DESC, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
            """
        )

    async def list_conversations(self, user_id: str, limit: int, offset: int, sort_by: str) -> list[dict[str, Any]]:
        sort_field = sort_by if sort_by in _SORT_FIELDS else "updated_at"
        rows = await fetch(
            f"SELECT * FROM conversations WHERE user_id=$1 ORDER BY is_pinned DESC, {sort_field} DESC LIMIT $2 OFFSET $3",
            user_id,
            limit,
            offset,
        )
        return [_conversation_data(row) for row in rows]

    async def get_conversation(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        # 会话详情天然包含消息列表，因此这里顺手把子表一起查回。
        row = await fetchrow("SELECT * FROM conversations WHERE user_id=$1 AND id=$2", user_id, conversation_id)
        if row is None:
            return None
        data = _conversation_data(row)
        messages = await fetch(
            "SELECT * FROM messages WHERE conversation_id=$1 ORDER BY created_at ASC",
            conversation_id,
        )
        data["messages"] = [_message_data(message) for message in messages]
        return data

    async def find_conversation_by_session_id(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        row = await fetchrow(
            """
            SELECT *
            FROM conversations
            WHERE user_id=$1 AND session_id=$2
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            user_id,
            session_id,
        )
        if row is None:
            return None
        return _conversation_data(row)

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str,
        associated_papers: list[str] | None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        papers = associated_papers or []
        await execute(
            """
            INSERT INTO conversations (
                id, user_id, session_id, title, created_at, updated_at, is_pinned, tags, associated_papers, message_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, FALSE, '[]'::jsonb, $7::jsonb, 0)
            """,
            conversation_id,
            user_id,
            session_id,
            title,
            now,
            now,
            dump_json(papers),
        )
        return {
            "id": conversation_id,
            "user_id": user_id,
            "session_id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "is_pinned": False,
            "tags": [],
            "associated_papers": papers,
            "message_count": 0,
        }

    async def update_conversation(
        self,
        user_id: str,
        conversation_id: str,
        *,
        title: str | None,
        is_pinned: bool | None,
        tags: list[str] | None,
        associated_papers: list[str] | None,
        session_id: str | None,
    ) -> bool:
        # 动态拼 SQL 的原因是：只更新显式传入字段，避免把未传值误覆盖成 NULL。
        updates: list[str] = []
        params: list[Any] = []
        index = 1
        for column, value in (
            ("title", title),
            ("is_pinned", is_pinned),
            ("tags", dump_json(tags) if tags is not None else None),
            ("associated_papers", dump_json(associated_papers) if associated_papers is not None else None),
            ("session_id", session_id),
        ):
            if value is None:
                continue
            cast = "::jsonb" if column in {"tags", "associated_papers"} else ""
            updates.append(f"{column}=${index}{cast}")
            params.append(value)
            index += 1
        if not updates:
            return False
        updates.append(f"updated_at=${index}")
        params.append(utc_now())
        params.append(user_id)
        params.append(conversation_id)
        row = await fetchrow(
            f"UPDATE conversations SET {', '.join(updates)} WHERE user_id=${index + 1} AND id=${index + 2} RETURNING id",
            *params,
        )
        return row is not None

    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        await execute(
            "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id=$1 AND id=$2)",
            user_id,
            conversation_id,
        )
        row = await fetchrow("DELETE FROM conversations WHERE user_id=$1 AND id=$2 RETURNING id", user_id, conversation_id)
        return row is not None

    async def add_message(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        role: str,
        content: str,
        title: str | None,
        tone: str | None,
        render_mode: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 1. 【研究交付】【可信来源】单条追加与批量回写都不能声明服务端支持身份
        if SERVER_DELIVERY_FIELDS.intersection(metadata or {}):
            raise DeliveryConflict("最终答案及支持身份只能由服务端结果提交产生")
        # 2. 先验证 conversation 归属，避免把消息写进别的用户的会话
        exists = await fetchrow("SELECT id FROM conversations WHERE user_id=$1 AND id=$2", user_id, conversation_id)
        if exists is None:
            raise LookupError("对话不存在")
        now = utc_now()
        await execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, title, tone, render_mode, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
            """,
            message_id,
            conversation_id,
            role,
            content,
            title,
            tone,
            render_mode,
            dump_json(metadata or {}),
            now,
        )
        await execute(
            "UPDATE conversations SET message_count=message_count + 1, updated_at=$1 WHERE user_id=$2 AND id=$3",
            now,
            user_id,
            conversation_id,
        )
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "title": title,
            "tone": tone,
            "render_mode": render_mode,
            **(metadata or {}),
            "created_at": now,
        }

    async def get_assistant_message_by_run_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
        run_id: str,
    ) -> dict[str, Any] | None:
        row = await fetchrow(
            """
            SELECT m.*
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.user_id=$1 AND c.id=$2 AND m.role='assistant' AND m.metadata->>'run_id'=$3
            ORDER BY m.created_at DESC
            LIMIT 1
            """,
            user_id,
            conversation_id,
            run_id,
        )
        return _message_data(row) if row is not None else None

    async def update_assistant_message_by_run_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
        run_id: str,
        content: str | None = None,
        render_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        row = await fetchrow(
            """
            UPDATE messages AS m
            SET content=COALESCE($4, m.content),
                render_mode=COALESCE($5, m.render_mode),
                metadata=m.metadata || $6::jsonb
            WHERE m.id = (
                SELECT selected.id
                FROM messages AS selected
                JOIN conversations AS c ON c.id = selected.conversation_id
                WHERE c.user_id=$1
                  AND c.id=$2
                  AND selected.role='assistant'
                  AND selected.metadata->>'run_id'=$3
                ORDER BY selected.created_at DESC
                LIMIT 1
            )
            RETURNING m.*
            """,
            user_id,
            conversation_id,
            run_id,
            content,
            render_mode,
            dump_json(metadata or {}),
        )
        return _message_data(row) if row is not None else None

    async def set_messages(
        self, user_id: str, conversation_id: str, messages: list[dict[str, Any]], *, session_id: str | None = None,
    ) -> bool:
        """【研究交付】【历史批量保存】参数为旧客户端消息；返回保存事实，新版交付禁止浏览器覆盖。"""
        from app.services import database

        pool = await database.get_pool()
        async with pool.acquire() as connection, connection.transaction():
            # 1. 【研究交付】【消息互斥】和结果提交锁定同一对话，避免检查后被并发提交越过
            now = utc_now()
            await connection.execute(
                """INSERT INTO conversations (id,user_id,session_id,title,created_at,updated_at)
                    VALUES ($1,$2,$3,'新对话',$4,$4) ON CONFLICT (id) DO NOTHING""",
                conversation_id, user_id, session_id, now,
            )
            existing = await connection.fetchrow(
                "SELECT id FROM conversations WHERE user_id=$1 AND id=$2 FOR UPDATE", user_id, conversation_id,
            )
            if existing is None:
                raise LookupError("对话不存在")
            committed = await connection.fetchrow(
                """SELECT id FROM messages WHERE conversation_id=$1 AND role='assistant'
                    AND metadata->>'persistence_status'='saved' LIMIT 1""", conversation_id,
            )
            if committed is not None:
                raise DeliveryConflict("对话已有服务端保存的答案，请刷新消息后继续；不能用本地历史覆盖")
            # 2. 【研究交付】【可信来源】浏览器不能凭上传字段授予已提交版本或精确支持身份
            for message in messages:
                metadata = message.get("metadata") or {}
                if SERVER_DELIVERY_FIELDS.intersection(metadata):
                    raise DeliveryConflict("最终答案及支持身份只能由服务端结果提交产生")
            # 3. 【研究交付】【历史原子保存】合法旧消息列表整体替换，失败保留原列表
            await connection.execute("DELETE FROM messages WHERE conversation_id=$1", conversation_id)
            for message in messages:
                await connection.execute(
                    """INSERT INTO messages (id,conversation_id,role,content,title,tone,render_mode,metadata,created_at)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9)""",
                    message.get("id") or str(uuid4()), conversation_id, message["role"], message["content"],
                    message.get("title"), message.get("tone"), message.get("render_mode"),
                    dump_json(message.get("metadata") or {}), parse_datetime(message.get("created_at")) or now,
                )
            await connection.execute(
                """UPDATE conversations SET message_count=$1,updated_at=$2,session_id=COALESCE($3,session_id)
                    WHERE user_id=$4 AND id=$5""", len(messages), now, session_id, user_id, conversation_id,
            )
        return True

    async def add_paper_to_conversation(self, user_id: str, conversation_id: str, paper_id: str) -> bool:
        row = await fetchrow(
            "SELECT associated_papers FROM conversations WHERE user_id=$1 AND id=$2",
            user_id,
            conversation_id,
        )
        if row is None:
            return False
        papers = load_json(row["associated_papers"], [])
        if paper_id not in papers:
            papers.append(paper_id)
            await execute(
                "UPDATE conversations SET associated_papers=$1::jsonb, updated_at=$2 WHERE user_id=$3 AND id=$4",
                dump_json(papers),
                utc_now(),
                user_id,
                conversation_id,
            )
        return True

    async def remove_paper_from_conversation(self, user_id: str, conversation_id: str, paper_id: str) -> bool:
        row = await fetchrow(
            "SELECT associated_papers FROM conversations WHERE user_id=$1 AND id=$2",
            user_id,
            conversation_id,
        )
        if row is None:
            return False
        papers = load_json(row["associated_papers"], [])
        if paper_id in papers:
            papers.remove(paper_id)
            await execute(
                "UPDATE conversations SET associated_papers=$1::jsonb, updated_at=$2 WHERE user_id=$3 AND id=$4",
                dump_json(papers),
                utc_now(),
                user_id,
                conversation_id,
            )
        return True

    async def remove_paper_references(self, paper_id: str, *, user_id: str | None = None) -> int:
        """从所有 conversation 的 `associated_papers` 中移除指定论文。"""
        if user_id:
            rows = await fetch(
                "SELECT id, associated_papers FROM conversations WHERE user_id=$1",
                user_id,
            )
        else:
            rows = await fetch("SELECT id, user_id, associated_papers FROM conversations")
        affected = 0
        for row in rows:
            conversation_id = str(row["id"])
            owner_id = user_id or str(row["user_id"])
            papers = [item for item in load_json(row["associated_papers"], []) if item]
            if paper_id not in papers:
                continue
            next_papers = [item for item in papers if item != paper_id]
            await execute(
                "UPDATE conversations SET associated_papers=$1::jsonb, updated_at=$2 WHERE user_id=$3 AND id=$4",
                dump_json(next_papers),
                utc_now(),
                owner_id,
                conversation_id,
            )
            affected += 1
        return affected


def get_conversation_repository() -> ConversationRepository:
    """返回会话仓储单例。"""
    global _repository
    if _repository is None:
        _repository = ConversationRepository()
    return _repository
