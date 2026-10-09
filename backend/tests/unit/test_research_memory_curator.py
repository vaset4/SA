"""
阶段 6 curator 单测：AI 辅助整理助手的校验护栏与初判逻辑。

覆盖点（对齐 prd B6 / implement 阶段 6 验证）：
- 假 chunk_id 的证据被丢弃、无合法证据的候选降级 user_note_only
- 真 chunk + quote 命中的候选标 evidence_backed、maturity=leaning
- 草稿一律 review_state=pending_review / source_kind=agent_draft，不进正式库
- 每条草稿带 confidence_score + confidence_signals
- curator 不触碰已确认记忆的 body
- scope 为空显式拒绝（不做全库扫）

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
    CuratorEvidenceCandidate,
    CuratorMemoryCandidate,
)

pytestmark = pytest.mark.skipif(not fixture.can_connect_db(), reason="本地 Postgres 不可用，跳过真库 curator 测试")


class _FakeStore:
    """伪 Milvus store：只认预置的 chunk 行，其余返回 None（模拟不存在）。"""

    def __init__(self, rows: dict[str, dict]) -> None:
        self._rows = rows

    def get_chunk_row(self, chunk_id: str):
        return self._rows.get(chunk_id)

    def list_chunk_rows_by_paper(self, paper_id: str, batch_size: int = 1000):
        return [row for row in self._rows.values() if row.get("paper_id") == paper_id]


def _batch(*candidates: CuratorMemoryCandidate) -> CuratorCandidateBatch:
    return CuratorCandidateBatch(candidates=list(candidates))


def _mock_llm(batch: CuratorCandidateBatch) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=batch)
    return llm


def _curate_in_project(scope, batch, rows):
    """在一个临时 project 内跑 curate，返回 (drafts, project_id, 事后查询函数结果)。"""
    holder = {}

    async def scenario():
        async with fixture.temp_project() as project_id:
            holder["project_id"] = project_id
            store = _FakeStore(rows)
            with patch.object(curator, "get_llm", return_value=_mock_llm(batch)), \
                 patch.object(curator, "get_milvus_store", return_value=store):
                drafts = await curator.curate(project_id, scope, user_id="curator_tester")
            accepted = await research_memory_service.list_items(project_id, review_state="accepted")
            pending = await research_memory_service.list_items(project_id, review_state="pending_review")
            holder["accepted"] = accepted
            holder["pending"] = pending
            return drafts

    drafts = asyncio.run(scenario())
    return drafts, holder


def test_fake_chunk_dropped() -> None:
    """假 chunk 证据丢弃 → 草稿 user_note_only + intuition，无证据。"""
    batch = _batch(CuratorMemoryCandidate(
        title="GraphRAG 构图成本被低估",
        body="来自对话推断",
        memory_type="synthesis_note",
        is_synthesis=True,
        evidence=[CuratorEvidenceCandidate(paper_id="paper-x", chunk_id="ghost", quote="幻觉引用")],
    ))
    drafts, holder = _curate_in_project({"paper_ids": ["paper-x"]}, batch, rows={})

    assert len(drafts) == 1
    item = drafts[0]["item"]
    assert item["evidence_mode"] == "user_note_only"
    assert item["maturity"] == "intuition"
    assert drafts[0]["evidence_refs"] == []


def test_real_chunk_quote_hit_marks_leaning() -> None:
    """真 chunk + quote 命中 full_text → evidence_backed + leaning + 证据落库。"""
    rows = {
        "chunk-1": {
            "chunk_id": "chunk-1",
            "paper_id": "paper-a",
            "parent_id": "parent-1",
            "section": "3.2",
            "section_path": "Method>Eval",
            "page_num": 4,
            "full_text": "RAGAS 提供无需参考答案的评估能力，适合项目内评测。",
        }
    }
    batch = _batch(CuratorMemoryCandidate(
        title="RAGAS 可行",
        body="有文献支撑",
        memory_type="metric_note",
        is_synthesis=False,
        evidence=[CuratorEvidenceCandidate(paper_id="paper-a", chunk_id="chunk-1", quote="RAGAS 提供无需参考答案的评估能力")],
    ))
    drafts, _ = _curate_in_project({"paper_ids": ["paper-a"]}, batch, rows=rows)

    item = drafts[0]["item"]
    assert item["evidence_mode"] == "evidence_backed"
    assert item["maturity"] == "leaning"
    assert len(drafts[0]["evidence_refs"]) == 1
    assert drafts[0]["evidence_refs"][0]["chunk_id"] == "chunk-1"


def test_drafts_are_pending_review_not_in_accepted() -> None:
    """草稿一律 pending_review / agent_draft，不出现在正式（accepted）列表。"""
    batch = _batch(CuratorMemoryCandidate(
        title="一条候选判断",
        body="",
        memory_type="decision_note",
        is_synthesis=False,
        evidence=[],
    ))
    drafts, holder = _curate_in_project({"conversation_id": "conv-x"}, batch, rows={})

    assert drafts[0]["item"]["review_state"] == "pending_review"
    assert drafts[0]["item"]["source_kind"] == "agent_draft"
    assert holder["accepted"] == []
    assert len(holder["pending"]) == 1


def test_every_draft_has_confidence_fields() -> None:
    """每条草稿带 confidence_score(0-1) + confidence_signals。"""
    batch = _batch(CuratorMemoryCandidate(
        title="候选带置信度",
        body="",
        memory_type="decision_note",
        is_synthesis=False,
        evidence=[],
    ))
    drafts, _ = _curate_in_project({"conversation_id": "conv-x"}, batch, rows={})

    item = drafts[0]["item"]
    assert 0.0 <= item["confidence_score"] <= 1.0
    signals = item["confidence_signals"]
    assert "evidence_count" in signals
    assert "quote_match_strength" in signals
    assert "is_synthesis" in signals


def test_curate_does_not_touch_confirmed_body() -> None:
    """curate 不改动已确认记忆的 body。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            confirmed = await research_memory_service.create_item(project_id, {
                "memory_type": "decision_note",
                "title": "已确认判断",
                "body": "原始正文不可被 AI 改动",
                "evidence_mode": "user_note_only",
            })
            memory_id = confirmed["item"]["memory_id"]
            await research_memory_service.confirm_item(memory_id)
            batch = _batch(CuratorMemoryCandidate(
                title="AI 新候选",
                body="新内容",
                memory_type="decision_note",
                is_synthesis=False,
                evidence=[],
            ))
            store = _FakeStore({})
            with patch.object(curator, "get_llm", return_value=_mock_llm(batch)), \
                 patch.object(curator, "get_milvus_store", return_value=store):
                await curator.curate(project_id, {"conversation_id": "conv-x"}, user_id="curator_tester")
            after = await research_memory_service.get_detail(memory_id)
            return after["item"]["body"]

    body = asyncio.run(scenario())
    assert body == "原始正文不可被 AI 改动"


def test_empty_scope_rejected() -> None:
    """scope 为空显式拒绝，不做全库扫。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            with pytest.raises(ValueError):
                await curator.curate(project_id, {}, user_id="curator_tester")

    asyncio.run(scenario())


def test_illegal_memory_type_skipped_not_written() -> None:
    """LLM 返回非法 memory_type 的候选被跳过、不写脏枚举值，同批合法候选照常落库。"""
    batch = _batch(
        CuratorMemoryCandidate(
            title="非法类型候选",
            body="LLM 吐了库里不认的 memory_type",
            memory_type="not_a_real_type",
            is_synthesis=False,
            evidence=[],
        ),
        CuratorMemoryCandidate(
            title="合法类型候选",
            body="同批的正当候选不应被牵连",
            memory_type="decision_note",
            is_synthesis=False,
            evidence=[],
        ),
    )
    drafts, holder = _curate_in_project({"conversation_id": "conv-x"}, batch, rows={})

    # 只落了合法那条，非法候选被跳过
    assert len(drafts) == 1
    assert drafts[0]["item"]["title"] == "合法类型候选"
    # 待审库里不含非法枚举值，也没有非法标题
    pending_types = {item["memory_type"] for item in holder["pending"]}
    assert "not_a_real_type" not in pending_types
    assert all(item["title"] != "非法类型候选" for item in holder["pending"])
