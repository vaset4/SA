"""阶段 1 验证：研究记忆 6 张表建表幂等 + maturity 列存在。

连真实 Postgres 开发库执行 ensure_schema()，验证：
1. 重复执行幂等，不报错；
2. 6 张表全部建好；
3. research_memory_items 含 maturity 列（B2 第四轴）。
"""

from __future__ import annotations

import asyncio

from app.repositories.research_memory_repository import get_research_memory_repository
from app.services.database import close_pool, fetch, fetchval


_EXPECTED_TABLES = (
    "research_questions",
    "research_memory_items",
    "memory_question_links",
    "memory_evidence_refs",
    "memory_links",
    "memory_versions",
)


def test_ensure_schema_idempotent_and_maturity_column() -> None:
    async def exercise() -> None:
        repository = get_research_memory_repository()
        # 1. 连续执行两次，验证 ensure_schema 幂等不报错
        await repository.ensure_schema()
        await repository.ensure_schema()

        # 2. 6 张表全部存在
        rows = await fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name = ANY($1::text[])",
            list(_EXPECTED_TABLES),
        )
        existing = {row["table_name"] for row in rows}
        assert existing == set(_EXPECTED_TABLES)

        # 3. research_memory_items 含 maturity 列
        maturity_type = await fetchval(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name='research_memory_items' AND column_name='maturity'",
        )
        assert maturity_type == "text"

        await close_pool()

    asyncio.run(exercise())
