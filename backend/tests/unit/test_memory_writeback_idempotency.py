"""候选写回的幂等落库测试（单元 3bc · A 组）。

作者: xxx
创建时间: 2026-08-30 15:10:00

被验的产品约束：
- 同一次回答重复写回只留一条候选，第二次拿回已有那条（重试不制造重复草稿）
- 内容相同但来源回答不同必须各留一条：不同回答里各自形成的判断不按内容相似归并
- 落库形状仍被写死为 draft / pending_review / agent_draft，未经人审进不了正式库
- 没有来源回答身份（手动 /curate，source_answer_id=""）时不查不去重，空串也不落库冒充身份

分两组：
- 内存组（TestIdempotencyWithFakeRepo）：用替身仓储验去重决策，不连库，任何环境都真跑
- 真库组（TestIdempotencyOnRealDb）：替身证不了 source_answer_id 那一列真的写进去、
  也证不了按该列的 SQL 真的筛对，这两件事只能在真实 Postgres 上验，库不可连时整组跳过
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import research_memory_db_fixture as db_fixture
from app.repositories import research_memory_repository as repo_mod
from app.services import research_memory_curator_service as curator
from app.services.research_memory_curator_service import CuratorMemoryCandidate

# 库不可连只跳真库组，内存组照常跑
requires_db = pytest.mark.skipif(
    not db_fixture.can_connect_db(), reason="业务库不可连，跳过写回幂等真库测试"
)

# 被验的草稿三轴取值，集中在此避免断言里散落裸字符串
_STATUS_DRAFT = "draft"
_REVIEW_PENDING = "pending_review"
_SOURCE_AGENT_DRAFT = "agent_draft"

_PROJECT_ID = "proj-writeback"
_USER_ID = "writeback_owner"
_MEMORY_TYPE = "decision_note"
# 来源回答身份即写回用的 request_id，两次不同回答用两个值
_ANSWER_ID = "req-answer-1"
_OTHER_ANSWER_ID = "req-answer-2"

_TITLE = "RAGAS 在本项目可用"
_BODY = "评测不必等 Human Label"
# 同一条判断的书写变体：标题内部多一个空格、正文换大小写，规范化后应折成同一指纹
_TITLE_EXTRA_SPACE = "RAGAS  在本项目可用"
_BODY_UPPER = "评测不必等 HUMAN LABEL"


class _FakeMemoryRepo:
    """替身仓储：只实现候选落库路径用到的方法，并记录本轮去重查了几次。"""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.create_calls = 0
        self.source_answer_queries: list[str] = []

    async def ensure_schema(self) -> None:
        return None

    async def list_items(self, project_id: str, /, **gates: Any) -> list[dict[str, Any]]:
        # possible_duplicate 提示的比对基准；本文件不造正式记忆，恒空
        return []

    async def list_items_by_source_answer(
        self, project_id: str, *, user_id: str, source_answer_id: str
    ) -> list[dict[str, Any]]:
        # 与真仓储同口径：项目 + 归属人 + 来源回答三条件等值过滤
        self.source_answer_queries.append(source_answer_id)
        return [
            item
            for item in self.items.values()
            if item["project_id"] == project_id
            and item["user_id"] == user_id
            and item["source_answer_id"] == source_answer_id
        ]

    async def create_item(self, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        # 直接按 payload 存，落库形状原样可断言；不写 source_answer_id 键会 KeyError 暴露
        self.create_calls += 1
        memory_id = f"mem_fake_{self.create_calls}"
        self.items[memory_id] = {"memory_id": memory_id, "project_id": project_id, **payload}
        return await self.get_detail(memory_id)

    async def create_candidate(self, project_id: str, payload: dict[str, Any], evidence_refs: list, *, candidate_key: str) -> dict:
        """参数为草稿及幂等键；模拟真实仓储唯一键与原子创建，不使用先查后写去重。"""
        for item in self.items.values():
            if item.get("candidate_key") == candidate_key:
                return await self.get_detail(item["memory_id"])
        return await self.create_item(project_id, {**payload, "candidate_key": candidate_key})

    async def get_detail(self, memory_id: str) -> dict[str, Any] | None:
        item = self.items.get(memory_id)
        return None if item is None else {"item": item}


def _candidate(title: str = _TITLE, body: str = _BODY) -> CuratorMemoryCandidate:
    """
    造一条无证据的记忆候选。

    参数:
        title: 候选标题
        body: 候选正文
    返回:
        记忆候选（证据真实性校验不在本文件的验证范围，故不带 evidence）
    """
    return CuratorMemoryCandidate(
        title=title, body=body, memory_type=_MEMORY_TYPE, is_synthesis=False, evidence=[]
    )


def _persist(
    repo: _FakeMemoryRepo, candidates: list[CuratorMemoryCandidate], *, source_answer_id: str
) -> list[dict[str, Any]]:
    """
    用替身仓储跑一次候选落库。

    参数:
        repo: 替身仓储
        candidates: 待落库候选
        source_answer_id: 来源回答身份，空串表示调用方没有来源回答
    返回:
        草稿详情列表
    """
    with patch.object(curator, "get_research_memory_repository", return_value=repo):
        return asyncio.run(
            curator.persist_candidates(
                _PROJECT_ID, candidates, user_id=_USER_ID, source_answer_id=source_answer_id
            )
        )


# ─── A 组：去重决策（不连库） ───


class TestIdempotencyWithFakeRepo:
    """验「同一次回答不重复落库、不同回答不按内容归并」在逻辑上成立。"""

    def test_same_answer_same_content_persists_once(self) -> None:
        """同一次回答重复写回：只落一条，第二次拿回已有那条的 memory_id。"""
        repo = _FakeMemoryRepo()
        first = _persist(repo, [_candidate()], source_answer_id=_ANSWER_ID)
        second = _persist(repo, [_candidate()], source_answer_id=_ANSWER_ID)

        assert repo.create_calls == 1
        assert len(repo.items) == 1
        assert second[0]["item"]["memory_id"] == first[0]["item"]["memory_id"]

    def test_normalized_content_matches_despite_case_and_spacing(self) -> None:
        """指纹按规范化后的标题+正文比对：大小写与多余空白不算新判断。

        这条锁住「复用 normalize_and_hash_quote 的规范化口径」——改成裸字符串相等即变红。
        """
        repo = _FakeMemoryRepo()
        _persist(repo, [_candidate()], source_answer_id=_ANSWER_ID)
        _persist(
            repo,
            [_candidate(title=_TITLE_EXTRA_SPACE, body=_BODY_UPPER)],
            source_answer_id=_ANSWER_ID,
        )

        assert repo.create_calls == 1

    def test_different_content_under_same_answer_keeps_both(self) -> None:
        """去重只对同一条判断生效：同一次回答里两条不同判断照常各落一条。"""
        repo = _FakeMemoryRepo()
        drafts = _persist(
            repo, [_candidate(), _candidate(title="构图成本被低估")], source_answer_id=_ANSWER_ID
        )

        assert repo.create_calls == 2
        assert len(drafts) == 2

    def test_different_answer_ids_never_merge_on_content(self) -> None:
        """内容相同但来源回答不同：各留一条，不按内容相似归并（产品硬规则）。"""
        repo = _FakeMemoryRepo()
        first = _persist(repo, [_candidate()], source_answer_id=_ANSWER_ID)
        second = _persist(repo, [_candidate()], source_answer_id=_OTHER_ANSWER_ID)

        assert repo.create_calls == 2
        assert first[0]["item"]["memory_id"] != second[0]["item"]["memory_id"]
        assert {item["source_answer_id"] for item in repo.items.values()} == {
            _ANSWER_ID,
            _OTHER_ANSWER_ID,
        }

    def test_forced_draft_shape_survives_the_refactor(self) -> None:
        """三轴仍写死为 draft / pending_review / agent_draft，来源回答身份进条目。"""
        repo = _FakeMemoryRepo()
        drafts = _persist(repo, [_candidate()], source_answer_id=_ANSWER_ID)
        item = drafts[0]["item"]

        assert item["status"] == _STATUS_DRAFT
        assert item["review_state"] == _REVIEW_PENDING
        assert item["source_kind"] == _SOURCE_AGENT_DRAFT
        assert item["source_answer_id"] == _ANSWER_ID

    def test_blank_source_answer_id_rejected_before_write(self) -> None:
        """保存必须绑定回答或整理操作，空身份不能通过原子候选接口。"""
        repo = _FakeMemoryRepo()
        with pytest.raises(ValueError, match="操作身份"):
            _persist(repo, [_candidate()], source_answer_id="")
        assert repo.create_calls == 0


# ─── B 组：真库落列与查询（库不可连时整组跳过） ───


@requires_db
class TestIdempotencyOnRealDb:
    """在真实 Postgres 上验 source_answer_id 真的落了列、按该列的查询真的筛对。"""

    def test_source_answer_id_persisted_and_dedups(self) -> None:
        """同一次回答写回两遍：库里只留一行，第二遍返回已有那条。"""
        async def scenario() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
            async with db_fixture.temp_project() as project_id:
                with patch.object(curator, "get_milvus_store", return_value=MagicMock()):
                    first = await curator.persist_candidates(
                        project_id, [_candidate()], user_id=_USER_ID, source_answer_id=_ANSWER_ID
                    )
                    second = await curator.persist_candidates(
                        project_id, [_candidate()], user_id=_USER_ID, source_answer_id=_ANSWER_ID
                    )
                rows = await repo_mod.get_research_memory_repository().list_items_by_source_answer(
                    project_id, user_id=_USER_ID, source_answer_id=_ANSWER_ID
                )
                return first, second, rows

        first, second, rows = asyncio.run(scenario())

        assert len(rows) == 1
        assert rows[0]["source_answer_id"] == _ANSWER_ID
        assert rows[0]["review_state"] == _REVIEW_PENDING
        assert second[0]["item"]["memory_id"] == first[0]["item"]["memory_id"]
