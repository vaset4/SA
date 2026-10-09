"""
项目研究记忆层 HTTP 路由测试。

直接调用 route 函数（对齐现有 API 测试风格），走真库验证端到端：
- 建问题 → 建记忆并归属 → 加证据 → confirm 成功
- evidence_backed 记忆无证据 confirm → 422（evidence gate 显式拒绝）
- 建关系 → 标 stale → 列版本 → 字面搜索
- 删项目触发级联，各记忆实体清空

真库不可连时整体 skip，无库环境不报错。
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import app.api.routes.research_memory_api as api
from research_memory_db_fixture import can_connect_db, temp_project

pytestmark = pytest.mark.skipif(not can_connect_db(), reason="业务库不可连，跳过研究记忆真库测试")

_USER = {"user_id": "rmem_tester"}


async def _pass_guard(project_id: str, user_id=None):
    # 归属校验放行：测试项目由 temp_project 建好，直接视为属于当前用户
    return {"project_id": project_id}


def _patch_guard(monkeypatch) -> None:
    monkeypatch.setattr(api.project_service, "require_project", _pass_guard)


def test_memory_crud_flow_end_to_end(monkeypatch) -> None:
    _patch_guard(monkeypatch)

    async def scenario() -> None:
        async with temp_project() as project_id:
            # 1. 建研究问题
            question = await api.create_question(
                project_id,
                api.CreateQuestionRequest(title="RAGAS 是否可行"),
                user=_USER,
            )
            question_id = question["question_id"]

            # 2. 建记忆并归属问题
            detail = await api.create_memory(
                project_id,
                api.CreateMemoryRequest(
                    memory_type="method_note",
                    title="RAGAS 无需参考答案即可评估",
                    body="RAGAS 支持无参考评估",
                    maturity="leaning",
                    evidence_mode="evidence_backed",
                    question_id=question_id,
                ),
                user=_USER,
            )
            memory_id = detail["item"]["memory_id"]
            assert question_id in detail["question_ids"]

            # 3. 无证据 confirm 应被 evidence gate 拒绝（422）
            with pytest.raises(HTTPException) as gate_err:
                await api.confirm_memory(project_id, memory_id, user=_USER)
            assert gate_err.value.status_code == 422

            # 4. 加证据后 confirm 成功
            await api.add_evidence(
                project_id,
                memory_id,
                api.AddEvidenceRequest(paper_id="paper-a", quote_text="RAGAS does not require references"),
                user=_USER,
            )
            confirmed = await api.confirm_memory(project_id, memory_id, user=_USER)
            assert confirmed["item"]["status"] == "confirmed"
            assert confirmed["item"]["confirmed_at"]

            # 5. 字面搜索命中
            hits = await api.search_memory(project_id, q="RAGAS", stale_state=None, user=_USER)
            assert any(h["memory_id"] == memory_id for h in hits)

            # 6. 版本历史随变更累积（create + attach_question + attach_evidence + confirm）
            versions = await api.list_versions(project_id, memory_id, user=_USER)
            assert len(versions) >= 4

    asyncio.run(scenario())


def test_memory_link_and_stale_flow(monkeypatch) -> None:
    _patch_guard(monkeypatch)

    async def scenario() -> None:
        async with temp_project() as project_id:
            first = await api.create_memory(
                project_id,
                api.CreateMemoryRequest(memory_type="decision_note", title="旧结论"),
                user=_USER,
            )
            second = await api.create_memory(
                project_id,
                api.CreateMemoryRequest(memory_type="decision_note", title="新结论修正旧结论"),
                user=_USER,
            )
            old_id = first["item"]["memory_id"]
            new_id = second["item"]["memory_id"]

            # 建 supersedes 关系：新结论修正旧结论，旧结论保留
            await api.add_link(
                project_id,
                new_id,
                api.AddLinkRequest(dst_memory_id=old_id, link_type="supersedes"),
                user=_USER,
            )
            links = await api.list_links(project_id, user=_USER)
            assert any(l["src_memory_id"] == new_id and l["dst_memory_id"] == old_id for l in links)
            # 旧结论仍可读
            old_detail = await api.get_memory(project_id, old_id, user=_USER)
            assert old_detail["item"]["memory_id"] == old_id

            # 标 stale 只改状态，body 不变
            body_before = old_detail["item"]["body"]
            staled = await api.mark_stale(
                project_id,
                old_id,
                api.MarkStaleRequest(reasons=["原 chunk 已失效"]),
                user=_USER,
            )
            assert staled["item"]["stale_state"] == "needs_rebind"
            assert staled["item"]["body"] == body_before
            assert "原 chunk 已失效" in staled["item"]["stale_reasons"]

    asyncio.run(scenario())


def test_cascade_delete_clears_memory(monkeypatch) -> None:
    _patch_guard(monkeypatch)

    async def scenario() -> None:
        async with temp_project() as project_id:
            detail = await api.create_memory(
                project_id,
                api.CreateMemoryRequest(memory_type="paper_note", title="待级联删的记忆"),
                user=_USER,
            )
            memory_id = detail["item"]["memory_id"]
            # temp_project 退出时删 project，级联应清空记忆
        # 出了 async with，project 已删；此处新开事务确认记忆不存在
        from app.repositories.research_memory_repository import get_research_memory_repository

        assert await get_research_memory_repository().get_item(memory_id) is None

    asyncio.run(scenario())
