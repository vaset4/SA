"""
阶段 7 待审区测试：AI 草稿的审阅工作流（含 B6 置信度排序）。

覆盖（对齐 prd B6 / implement 阶段 7 验证）：
- 待审区按 confidence_score 降序返回，低置信度沉底
- accept 后草稿进正式（accepted）列表、退出待审区，接纳时可编辑（含改 maturity）
- dismiss 后草稿在正式库与待审区都不显示（软删，可审计保留）
- 高置信度草稿不会自动进正式库（无自动 approve），curate 后仍全 pending_review

LLM 与 Milvus 全部 mock；草稿落真库，测试结束级联清理。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import research_memory_db_fixture as fixture
from app.services import research_memory_curator_service as curator
from app.services import research_memory_service
from app.services.research_memory_curator_service import (
    CuratorCandidateBatch,
    CuratorMemoryCandidate,
)

pytestmark = pytest.mark.skipif(
    not fixture.can_connect_db(), reason="本地 Postgres 不可用，跳过真库待审区测试"
)


def _batch(*candidates: CuratorMemoryCandidate) -> CuratorCandidateBatch:
    return CuratorCandidateBatch(candidates=list(candidates))


def _mock_llm(batch: CuratorCandidateBatch) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=batch)
    return llm


class _FakeStore:
    """伪 Milvus store：无任何 chunk 命中（待审区测试不关心证据校验细节）。"""

    def get_chunk_row(self, chunk_id: str):
        return None

    def list_chunk_rows_by_paper(self, paper_id: str, batch_size: int = 1000):
        return []


async def _curate(project_id: str, batch: CuratorCandidateBatch) -> list[dict]:
    with patch.object(curator, "get_llm", return_value=_mock_llm(batch)), \
         patch.object(curator, "get_milvus_store", return_value=_FakeStore()):
        return await curator.curate(project_id, {"conversation_id": "conv-x"}, user_id="reviewer")


def test_review_queue_sorted_by_confidence_desc() -> None:
    """待审区按置信度降序：多证据推断分高在前，纯直觉分低在后。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            # is_synthesis 且无证据会被扣分，普通候选基线分更高
            batch = _batch(
                CuratorMemoryCandidate(title="纯推断低置信", body="", memory_type="synthesis_note", is_synthesis=True),
                CuratorMemoryCandidate(title="普通候选较高", body="", memory_type="decision_note", is_synthesis=False),
            )
            await _curate(project_id, batch)
            queue = await research_memory_service.list_review_queue(project_id)
            return [item["confidence_score"] for item in queue]

    scores = asyncio.run(scenario())
    assert len(scores) == 2
    # 降序：前一条置信度不低于后一条
    assert scores[0] >= scores[1]


def test_accept_moves_draft_to_accepted_with_edit() -> None:
    """accept 带编辑：草稿进正式库、退出待审区，maturity 被改到 established。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            batch = _batch(CuratorMemoryCandidate(title="待接纳草稿", body="原正文", memory_type="decision_note"))
            drafts = await _curate(project_id, batch)
            memory_id = drafts[0]["item"]["memory_id"]
            await research_memory_service.accept_draft(
                memory_id, {"maturity": "established", "body": "接纳时改写"}, created_by="reviewer"
            )
            accepted = await research_memory_service.list_items(project_id, review_state="accepted")
            pending = await research_memory_service.list_items(project_id, review_state="pending_review")
            detail = await research_memory_service.get_detail(memory_id)
            return accepted, pending, detail

    accepted, pending, detail = asyncio.run(scenario())
    assert len(accepted) == 1
    assert pending == []
    assert detail["item"]["maturity"] == "established"
    assert detail["item"]["body"] == "接纳时改写"


def test_dismiss_hides_from_both_lists() -> None:
    """dismiss 后草稿在正式库与待审区都不出现。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            batch = _batch(CuratorMemoryCandidate(title="待否决草稿", body="", memory_type="decision_note"))
            drafts = await _curate(project_id, batch)
            memory_id = drafts[0]["item"]["memory_id"]
            await research_memory_service.dismiss_draft(memory_id, created_by="reviewer")
            accepted = await research_memory_service.list_items(project_id, review_state="accepted")
            pending = await research_memory_service.list_review_queue(project_id)
            dismissed = await research_memory_service.get_detail(memory_id)
            return accepted, pending, dismissed

    accepted, pending, dismissed = asyncio.run(scenario())
    assert accepted == []
    assert pending == []
    # 软删：记录本身仍可查（可审计）
    assert dismissed["item"]["review_state"] == "dismissed"


def test_no_auto_approve_all_drafts_pending() -> None:
    """无自动 approve：curate 后所有草稿一律 pending_review，不因高置信度自动进正式库。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            batch = _batch(
                CuratorMemoryCandidate(title="候选甲", body="", memory_type="decision_note"),
                CuratorMemoryCandidate(title="候选乙", body="", memory_type="metric_note"),
            )
            drafts = await _curate(project_id, batch)
            accepted = await research_memory_service.list_items(project_id, review_state="accepted")
            return drafts, accepted

    drafts, accepted = asyncio.run(scenario())
    assert len(drafts) == 2
    assert all(d["item"]["review_state"] == "pending_review" for d in drafts)
    assert accepted == []


def test_dismiss_missing_memory_raises() -> None:
    """否决不存在的草稿显式抛 LookupError。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            _ = project_id
            with pytest.raises(LookupError):
                await research_memory_service.dismiss_draft("mem_ghost", created_by="reviewer")

    asyncio.run(scenario())
