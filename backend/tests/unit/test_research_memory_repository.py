"""
研究记忆仓储层真库测试（阶段 2）。

覆盖：6 表 CRUD 往返、外键级联删除、maturity filter、B3 字面过滤、
版本随变更自增并落全量快照。连不上库时整体 skip。
"""

from __future__ import annotations

import asyncio

import pytest

from app.repositories.research_memory_repository import get_research_memory_repository
from research_memory_db_fixture import can_connect_db, temp_project

pytestmark = pytest.mark.skipif(not can_connect_db(), reason="本地 Postgres 不可连，跳过研究记忆真库测试")


def test_question_and_item_crud_roundtrip() -> None:
    """建问题 + 建记忆并归属问题 + 详情聚合往返正确。"""
    async def scenario() -> None:
        repo = get_research_memory_repository()
        async with temp_project() as project_id:
            question = await repo.create_question(project_id, {"title": "RAG 评估怎么做", "is_purpose": False})
            assert question["question_id"].startswith("rq_")

            detail = await repo.create_item(
                project_id,
                {"memory_type": "decision_note", "title": "RAGAS 可行", "body": "无需参考答案", "maturity": "established"},
            )
            memory_id = detail["item"]["memory_id"]
            assert memory_id.startswith("mem_")
            assert detail["item"]["version"] == 1
            assert detail["item"]["maturity"] == "established"

            await repo.attach_question(memory_id, question["question_id"])
            got = await repo.get_detail(memory_id)
            assert got["question_ids"] == [question["question_id"]]

    asyncio.run(scenario())


def test_evidence_ref_and_count() -> None:
    """证据引用增删 + 计数（evidence gate 判定依据）。"""
    async def scenario() -> None:
        repo = get_research_memory_repository()
        async with temp_project() as project_id:
            detail = await repo.create_item(
                project_id, {"memory_type": "paper_note", "title": "证据测试", "evidence_mode": "evidence_backed"}
            )
            memory_id = detail["item"]["memory_id"]
            assert await repo.count_evidence_refs(memory_id) == 0

            ref = await repo.add_evidence_ref(
                memory_id,
                {"paper_id": "paper_1", "chunk_id": "c1", "quote_text": "some quote", "quote_hash": "h1"},
            )
            assert await repo.count_evidence_refs(memory_id) == 1
            await repo.delete_evidence_ref(ref["ref_id"])
            assert await repo.count_evidence_refs(memory_id) == 0

    asyncio.run(scenario())


def test_version_increments_on_update() -> None:
    """更新记忆时版本自增并落全量快照。"""
    async def scenario() -> None:
        repo = get_research_memory_repository()
        async with temp_project() as project_id:
            detail = await repo.create_item(project_id, {"memory_type": "method_note", "title": "初始"})
            memory_id = detail["item"]["memory_id"]

            updated = await repo.update_item(memory_id, {"body": "改了正文"}, change_kind="edit_body")
            assert updated["item"]["version"] == 2
            versions = await repo.list_versions(memory_id)
            assert [v["version"] for v in versions] == [1, 2]
            assert versions[-1]["change_kind"] == "edit_body"
            assert versions[-1]["snapshot"]["item"]["body"] == "改了正文"

    asyncio.run(scenario())


def test_links_and_maturity_filter() -> None:
    """记忆关系建/查 + maturity filter + 计数聚合。"""
    async def scenario() -> None:
        repo = get_research_memory_repository()
        async with temp_project() as project_id:
            a = (await repo.create_item(project_id, {"memory_type": "decision_note", "title": "A", "maturity": "established"}))["item"]["memory_id"]
            b = (await repo.create_item(project_id, {"memory_type": "paper_note", "title": "B", "maturity": "intuition"}))["item"]["memory_id"]
            await repo.add_link(project_id, a, b, "supersedes")

            links = await repo.list_links(project_id)
            assert len(links) == 1 and links[0]["link_type"] == "supersedes"

            established = await repo.list_items(project_id, maturity="established")
            assert [i["memory_id"] for i in established] == [a]

            counts = await repo.count_by_maturity(project_id)
            assert counts.get("established") == 1 and counts.get("intuition") == 1

    asyncio.run(scenario())


def test_literal_search_scoped_by_project() -> None:
    """B3 字面过滤命中关键词，且跨项目不串。"""
    async def scenario() -> None:
        repo = get_research_memory_repository()
        async with temp_project() as project_a, temp_project() as project_b:
            await repo.create_item(project_a, {"memory_type": "method_note", "title": "反向传播推导", "body": "链式法则"})
            await repo.create_item(project_b, {"memory_type": "method_note", "title": "无关记忆", "body": "别的内容"})

            hits_a = await repo.search_memories_literal(project_a, "反向传播")
            assert len(hits_a) == 1 and "反向传播" in hits_a[0]["title"]

            hits_b = await repo.search_memories_literal(project_b, "反向传播")
            assert hits_b == []

    asyncio.run(scenario())


def test_cascade_delete_clears_all_tables() -> None:
    """删项目级联清空记忆/证据/关系/版本/问题。"""
    async def scenario() -> None:
        repo = get_research_memory_repository()
        async with temp_project() as project_id:
            detail = await repo.create_item(
                project_id, {"memory_type": "paper_note", "title": "待删", "evidence_mode": "evidence_backed"}
            )
            memory_id = detail["item"]["memory_id"]
            await repo.add_evidence_ref(memory_id, {"paper_id": "p", "quote_text": "q", "quote_hash": "h"})
            # 退出 temp_project 会删 project，触发级联
        # 级联后该记忆应查不到
        assert await repo.get_item(memory_id) is None

    asyncio.run(scenario())


def test_delete_item_cascades_children() -> None:
    """删单条记忆级联清其证据/关系/版本，不影响其他记忆。"""
    async def scenario() -> None:
        repo = get_research_memory_repository()
        async with temp_project() as project_id:
            keep = (await repo.create_item(project_id, {"memory_type": "paper_note", "title": "保留"}))["item"]["memory_id"]
            drop = (await repo.create_item(project_id, {"memory_type": "paper_note", "title": "删除"}))["item"]["memory_id"]
            await repo.add_evidence_ref(drop, {"paper_id": "p", "quote_text": "q", "quote_hash": "h"})

            assert await repo.delete_item(drop) is True
            assert await repo.get_item(drop) is None
            assert await repo.list_versions(drop) == []
            assert await repo.get_item(keep) is not None

    asyncio.run(scenario())
