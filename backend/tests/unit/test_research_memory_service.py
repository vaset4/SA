"""
阶段 3 服务层测试：业务规则在真库上的编排行为。

覆盖：evidence gate（事实型无证据确认被拒 / 有证据放行）、版本自增、
quote_hash 稳定性、研究目的 singleton、supersedes 保留旧条目、
conflicts_with 单行双向查、maturity 流转、mark_stale 不改 body、枚举校验。

真库不可连时整体跳过（见 research_memory_db_fixture）。
"""

from __future__ import annotations

import asyncio

import pytest

import research_memory_db_fixture as db_fixture
from app.services import research_memory_service as svc

pytestmark = pytest.mark.skipif(
    not db_fixture.can_connect_db(), reason="业务库不可连，跳过研究记忆真库测试"
)


def _run(coro):
    return asyncio.run(coro)


def test_normalize_and_hash_quote_stable() -> None:
    # 相同语义、不同空白/大小写的 quote 应产出相同 hash
    norm_a, hash_a = svc.normalize_and_hash_quote("  RAGAS  Needs   No Reference ")
    norm_b, hash_b = svc.normalize_and_hash_quote("ragas needs no reference")
    assert norm_a == norm_b == "ragas needs no reference"
    assert hash_a == hash_b
    assert len(hash_a) == 16


def test_invalid_enum_rejected() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            with pytest.raises(ValueError):
                await svc.create_item(
                    project_id, {"title": "x", "memory_type": "not_a_type"}
                )

    _run(scenario())


def test_purpose_question_singleton() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            await svc.create_question(project_id, {"title": "研究目的", "is_purpose": True})
            # 第二条研究目的应被拒绝
            with pytest.raises(ValueError):
                await svc.create_question(project_id, {"title": "另一个目的", "is_purpose": True})

    _run(scenario())


def test_evidence_gate_blocks_confirm_without_evidence() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            detail = await svc.create_item(
                project_id,
                {"title": "事实判断", "memory_type": "metric_note", "evidence_mode": "evidence_backed"},
            )
            memory_id = detail["item"]["memory_id"]
            # 无证据确认：显式 ValueError，不静默通过
            with pytest.raises(ValueError):
                await svc.confirm_item(memory_id)

    _run(scenario())


def test_evidence_gate_passes_with_evidence_and_sets_timestamp() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            detail = await svc.create_item(
                project_id,
                {"title": "有据判断", "memory_type": "metric_note", "evidence_mode": "evidence_backed"},
            )
            memory_id = detail["item"]["memory_id"]
            await svc.add_evidence(
                memory_id, {"paper_id": "paper_1", "quote_text": "some supporting quote"}
            )
            confirmed = await svc.confirm_item(memory_id)
            assert confirmed["item"]["status"] == "confirmed"
            assert confirmed["item"]["confirmed_at"] is not None

    _run(scenario())


def test_version_increments_on_each_change() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            detail = await svc.create_item(
                project_id, {"title": "判断", "memory_type": "decision_note"}
            )
            memory_id = detail["item"]["memory_id"]
            assert detail["item"]["version"] == 1
            await svc.update_item(memory_id, {"body": "改了正文"})
            after = await svc.get_detail(memory_id)
            assert after["item"]["version"] == 2
            # 每次变更都留了一条版本快照
            versions = await svc.list_versions(memory_id)
            assert len(versions) == 2
            assert versions[0]["change_kind"] == "create"
            assert versions[1]["change_kind"] == "edit_body"

    _run(scenario())


def test_mark_stale_keeps_body_unchanged() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            detail = await svc.create_item(
                project_id, {"title": "判断", "memory_type": "method_note", "body": "原始正文"}
            )
            memory_id = detail["item"]["memory_id"]
            updated = await svc.mark_stale(memory_id, ["chunk 已删除"])
            assert updated["item"]["stale_state"] == "needs_rebind"
            assert updated["item"]["stale_reasons"] == ["chunk 已删除"]
            # 标 stale 绝不动 body
            assert updated["item"]["body"] == "原始正文"

    _run(scenario())


def test_maturity_flow() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            detail = await svc.create_item(
                project_id, {"title": "判断", "memory_type": "decision_note", "maturity": "intuition"}
            )
            memory_id = detail["item"]["memory_id"]
            upgraded = await svc.set_maturity(memory_id, "established")
            assert upgraded["item"]["maturity"] == "established"
            # 成熟度变更留了独立版本轨迹
            kinds = [v["change_kind"] for v in await svc.list_versions(memory_id)]
            assert "change_maturity" in kinds

    _run(scenario())


def test_supersedes_keeps_old_item_readable() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            old = await svc.create_item(project_id, {"title": "旧判断", "memory_type": "decision_note"})
            new = await svc.create_item(project_id, {"title": "新判断", "memory_type": "decision_note"})
            old_id = old["item"]["memory_id"]
            new_id = new["item"]["memory_id"]
            await svc.add_link(project_id, new_id, old_id, "supersedes")
            # supersedes 后旧记忆仍可读、版本自增（关系轨迹）
            old_after = await svc.get_detail(old_id)
            assert old_after["item"]["memory_id"] == old_id
            assert old_after["item"]["version"] == 2

    _run(scenario())


def test_conflicts_with_single_row_bidirectional_read() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            a = await svc.create_item(project_id, {"title": "判断A", "memory_type": "decision_note"})
            b = await svc.create_item(project_id, {"title": "判断B", "memory_type": "decision_note"})
            a_id = a["item"]["memory_id"]
            b_id = b["item"]["memory_id"]
            await svc.add_link(project_id, a_id, b_id, "conflicts_with")
            # 单行存储，但两端各自查关系都能看到这条冲突
            a_links = (await svc.get_detail(a_id))["links"]
            b_links = (await svc.get_detail(b_id))["links"]
            assert len(a_links) == 1
            assert len(b_links) == 1
            assert a_links[0]["link_id"] == b_links[0]["link_id"]

    _run(scenario())


def test_self_link_rejected() -> None:
    async def scenario() -> None:
        async with db_fixture.temp_project() as project_id:
            item = await svc.create_item(project_id, {"title": "判断", "memory_type": "decision_note"})
            memory_id = item["item"]["memory_id"]
            with pytest.raises(ValueError):
                await svc.add_link(project_id, memory_id, memory_id, "based_on")

    _run(scenario())
