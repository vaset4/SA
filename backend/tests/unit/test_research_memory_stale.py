"""
阶段 8 B4 证据失效检测单测。

覆盖点（对齐 prd B4 / implement 阶段 8 验证）：
- chunk 不存在 → needs_rebind + stale_reasons 可读 + body 不变
- quote 漂移（chunk 存在但原句不再是 full_text 子串）→ needs_rebind
- 健康证据（chunk 存在且 quote 仍命中）→ 保持 fresh，不误标
- 校验只改 stale_state + stale_reasons，body 全程不变
- 误杀边界：Milvus 临时不可用（get_chunk_row 抛异常）→ 校验整体中止、健康记忆不被误标 stale

Milvus store mock；记忆落真库，结束级联清理。
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import research_memory_db_fixture as fixture
from app.services import research_memory_service
from app.services import research_memory_stale_service as stale

pytestmark = pytest.mark.skipif(
    not fixture.can_connect_db(), reason="本地 Postgres 不可用，跳过真库失效检测测试"
)


class _FakeStore:
    """伪 Milvus store：只认预置 chunk 行，其余返回 None（模拟已删）。"""

    def __init__(self, rows: dict[str, dict]) -> None:
        self._rows = rows

    def get_chunk_row(self, chunk_id: str):
        return self._rows.get(chunk_id)


async def _make_backed_item(project_id: str, chunk_id: str, quote_text: str) -> str:
    """建一条 evidence_backed 记忆并挂一条指定 chunk/quote 的证据，返回 memory_id。"""
    detail = await research_memory_service.create_item(project_id, {
        "memory_type": "metric_note",
        "title": "带证据的判断",
        "body": "原始正文，校验不应改动它",
        "evidence_mode": "evidence_backed",
    })
    memory_id = detail["item"]["memory_id"]
    await research_memory_service.add_evidence(memory_id, {
        "paper_id": "paper-a",
        "chunk_id": chunk_id,
        "quote_text": quote_text,
    })
    return memory_id


def test_missing_chunk_marks_needs_rebind_body_unchanged() -> None:
    """chunk 已删 → needs_rebind + 原因可读 + body 不变。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            memory_id = await _make_backed_item(project_id, "chunk-gone", "任意原句")
            before = await research_memory_service.get_detail(memory_id)
            body_before = before["item"]["body"]
            with patch.object(stale, "get_milvus_store", return_value=_FakeStore({})):
                result = await stale.check_evidence_staleness(project_id)
            after = await research_memory_service.get_detail(memory_id)
            return result, body_before, after["item"]

    result, body_before, item = asyncio.run(scenario())
    assert result["marked_stale"] == 1
    assert item["stale_state"] == "needs_rebind"
    assert item["stale_reasons"]
    assert "chunk" in item["stale_reasons"][0]
    # body 全程不变
    assert item["body"] == body_before == "原始正文，校验不应改动它"


def test_quote_drift_marks_stale() -> None:
    """chunk 存在但 quote 不再是 full_text 子串 → needs_rebind。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            memory_id = await _make_backed_item(project_id, "chunk-1", "这句话原文里没有")
            rows = {"chunk-1": {"chunk_id": "chunk-1", "full_text": "chunk 全文与原句已不一致"}}
            with patch.object(stale, "get_milvus_store", return_value=_FakeStore(rows)):
                await stale.check_evidence_staleness(project_id)
            after = await research_memory_service.get_detail(memory_id)
            return after["item"]

    item = asyncio.run(scenario())
    assert item["stale_state"] == "needs_rebind"
    assert item["stale_reasons"]


def test_healthy_evidence_not_marked() -> None:
    """chunk 存在且 quote 仍命中 → 保持 fresh，不误标。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            memory_id = await _make_backed_item(project_id, "chunk-1", "RAGAS 无需参考答案")
            rows = {"chunk-1": {"chunk_id": "chunk-1", "full_text": "结论：RAGAS 无需参考答案即可评估。"}}
            with patch.object(stale, "get_milvus_store", return_value=_FakeStore(rows)):
                result = await stale.check_evidence_staleness(project_id)
            after = await research_memory_service.get_detail(memory_id)
            return result, after["item"]

    result, item = asyncio.run(scenario())
    assert result["marked_stale"] == 0
    assert item["stale_state"] == "fresh"


def test_single_memory_check() -> None:
    """单条模式只校验指定记忆。"""
    async def scenario():
        async with fixture.temp_project() as project_id:
            memory_id = await _make_backed_item(project_id, "chunk-gone", "任意原句")
            with patch.object(stale, "get_milvus_store", return_value=_FakeStore({})):
                result = await stale.check_evidence_staleness(project_id, memory_id)
            after = await research_memory_service.get_detail(memory_id)
            return result, after["item"]

    result, item = asyncio.run(scenario())
    assert result["checked"] == 1
    assert item["stale_state"] == "needs_rebind"


class _UnavailableStore:
    """伪 Milvus store：get_chunk_row 抛异常，模拟 Milvus 临时不可用（连接故障/查询报错）。"""

    def get_chunk_row(self, chunk_id: str):
        raise RuntimeError("Milvus 暂时不可用")


def test_milvus_unavailable_aborts_without_false_marking() -> None:
    """
    误杀边界锁定：Milvus 临时不可用（get_chunk_row 抛异常）时，校验整体中止（异常上抛），
    健康记忆绝不被误标 stale——「读不到」不等于「chunk 不存在」。

    这是 B4 最关键的正确性属性：区分「chunk 真删了（get_chunk_row 返回 None）→ 标 stale」
    与「Milvus 临时故障（抛异常）→ 中止不标」。若有人误把 stale_service 改成 catch 异常
    兜底当作缺失，该断言会失败，从而挡住这种静默退化。
    """
    async def scenario():
        async with fixture.temp_project() as project_id:
            memory_id = await _make_backed_item(project_id, "chunk-1", "健康原句")
            aborted = False
            with patch.object(stale, "get_milvus_store", return_value=_UnavailableStore()):
                try:
                    await stale.check_evidence_staleness(project_id)
                except RuntimeError:
                    # 期望：异常向上传播、显式失败，而不是被吞掉并把记忆误标 stale
                    aborted = True
            after = await research_memory_service.get_detail(memory_id)
            return aborted, after["item"]

    aborted, item = asyncio.run(scenario())
    # 1. 临时不可用必须显式失败（异常传播），不静默兜底
    assert aborted
    # 2. 健康记忆未被误标 stale，保持 fresh
    assert item["stale_state"] == "fresh"
