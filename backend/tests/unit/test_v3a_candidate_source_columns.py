"""候选来源列（session / conversation / answer）落库真库测试。

三条产出路径的来源身份口径各不相同，落库结果必须能区分是谁产的：
回合写回三列齐全、AI 整理只有会话、手动保存三列全空。

作者：xxx
2026-09-01 10:30:00
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.repositories.research_memory_repository import get_research_memory_repository
from app.services import research_memory_curator_service as curator
from app.services.internal_research import research_memory_writeback as writeback
from research_memory_db_fixture import can_connect_db, temp_project

pytestmark = pytest.mark.skipif(not can_connect_db(), reason="本地 Postgres 不可连，跳过候选来源列真库测试")

# 写回路径的来源身份，三列各自的期望值都从这里取
_CONVERSATION_ID = "conv_v3a_1"
_SESSION_ID = "sess_v3a_1"
_REQUEST_ID = "req_v3a_1"


class _FakeProposalLLM:
    """替身模型：写回评估直接给一条够格提议，不打真实模型。"""

    def with_structured_output(self, schema: Any) -> "_FakeProposalLLM":
        return self

    async def ainvoke(self, messages: list[Any]) -> writeback.TurnMemoryProposalBatch:
        return writeback.TurnMemoryProposalBatch(
            proposals=[
                writeback.TurnMemoryProposal(
                    source_claim_id="c-1",
                    title="STGCN 在本项目数据上更稳",
                    body="两篇对照实验指向同一结论",
                    memory_type="comparison_note",
                    durable_project_value=True,
                    reusable_judgment=True,
                    transient_or_lookup_detail=False,
                )
            ]
        )


def _research_state(project_id: str) -> dict[str, Any]:
    """一条正常收敛的 research 回合状态，带上会话与 session 身份供写回取来源。"""
    return {
        "user_query": "这两篇的实验设置差别在哪",
        "answer": "差别在采样率与骨架图构造。",
        "turn_depth": "research",
        "conversation_id": _CONVERSATION_ID,
        "session_id": _SESSION_ID,
        "project_scope": {"project_id": project_id, "user_id": "rmem_tester", "paper_ids": ["paper-a"]},
        "source_status": {"planner": "ok"},
        "planner_loop_decision": "",
        "evidence_records": [{"evidence_id": "ev-1", "paper_id": "paper-a", "chunk_id": "chunk-1"}],
        "research_unit_results": [
            {
                "unit_id": "u-1",
                "status": "evidence_found",
                "evidence_ids": ["ev-1"],
                "claims": [
                    {
                        "claim_id": "c-1",
                        "text": "论文 A 的采样率是 5 分钟",
                        "claim_type": "paper_supported",
                        "validation_status": "validated",
                        "supporting_evidence_ids": ["ev-1"],
                    }
                ],
            }
        ],
    }


def _candidate(title: str) -> curator.CuratorMemoryCandidate:
    """一条不带证据的候选，落库走 user_note_only 分支。"""
    return curator.CuratorMemoryCandidate(
        title=title,
        body="正文",
        memory_type="comparison_note",
        is_synthesis=False,
        evidence=[],
    )


async def _item_of(memory_id: str) -> dict[str, Any]:
    item = await get_research_memory_repository().get_item(memory_id)
    assert item is not None
    return item


def test_turn_writeback_fills_all_three_source_columns() -> None:
    """回合写回：候选要能回连到本轮会话、session 和那一次回答，三列都不能是 NULL。"""
    async def scenario() -> None:
        async with temp_project() as project_id:
            result = await writeback.write_back_turn_memory(
                _research_state(project_id), request_id=_REQUEST_ID, llm=_FakeProposalLLM()
            )
            assert result["status"] == "success", result
            item = await _item_of(result["candidate_ids"][0])
            assert item["source_conversation_id"] == _CONVERSATION_ID
            assert item["source_session_id"] == _SESSION_ID
            assert item["source_answer_id"] == _REQUEST_ID

    asyncio.run(scenario())


def test_turn_writeback_without_conversation_keeps_null() -> None:
    """state 里没带会话与 session 时（两者在 AgentState 里都是可选）落 NULL，不留空串冒充身份。"""
    async def scenario() -> None:
        async with temp_project() as project_id:
            state = _research_state(project_id)
            state.pop("conversation_id")
            state.pop("session_id")
            result = await writeback.write_back_turn_memory(
                state, request_id=_REQUEST_ID, llm=_FakeProposalLLM()
            )
            assert result["status"] == "success", result
            item = await _item_of(result["candidate_ids"][0])
            assert item["source_conversation_id"] is None
            assert item["source_session_id"] is None
            # 来源回答身份始终有，缺的只是会话两列
            assert item["source_answer_id"] == _REQUEST_ID

    asyncio.run(scenario())


def test_curate_fills_conversation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """AI 整理：圈定的会话要落进来源列，但它不挂在某次回答上，另两列保持 NULL。"""
    async def scenario() -> None:
        async def fake_material(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"conversation_text": "", "existing_titles": [], "papers": []}

        async def fake_llm(material: dict[str, Any]) -> curator.CuratorCandidateBatch:
            return curator.CuratorCandidateBatch(candidates=[_candidate("整理产出的判断")])

        monkeypatch.setattr(curator, "_gather_material", fake_material)
        monkeypatch.setattr(curator, "_invoke_curator_llm", fake_llm)
        async with temp_project() as project_id:
            drafts = await curator.curate(
                project_id, {"conversation_id": _CONVERSATION_ID}, user_id="rmem_tester"
            )
            item = await _item_of(drafts[0]["item"]["memory_id"])
            assert item["source_conversation_id"] == _CONVERSATION_ID
            # 整理入口没有 session，也不挂某次回答：这两列留 NULL 而不是空串
            assert item["source_session_id"] is None
            assert item["source_answer_id"] is None

    asyncio.run(scenario())


def test_manual_persist_leaves_source_columns_null() -> None:
    """手动保存不带任何来源身份：三列全 NULL，不给空串冒充来源。"""
    async def scenario() -> None:
        async with temp_project() as project_id:
            drafts = await curator.persist_candidates(
                project_id, [_candidate("手动保存的判断")], user_id="rmem_tester", source_answer_id=""
            )
            item = await _item_of(drafts[0]["item"]["memory_id"])
            assert item["source_conversation_id"] is None
            assert item["source_session_id"] is None
            assert item["source_answer_id"] is None

    asyncio.run(scenario())
