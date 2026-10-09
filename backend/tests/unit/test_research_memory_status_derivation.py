"""
研究记忆新建状态派生测试。

作者: xxx
创建时间: 2026-08-29 15:20:00

本文件分两组：
- 纯函数组（TestDeriveInitialStatus）：只调 service 的 _derive_initial_status，
  不连库，任何环境都真跑。
- 真库组（TestStatusDerivationOnRealDb）：在真实 Postgres 上验证落库结果，
  本机库不可连时整个 class 跳过（见 research_memory_db_fixture.can_connect_db）。

被验的业务语义：用户亲手写下的判断新建即生效（status=confirmed），不必再点一次确认；
唯一例外是声称有文献背书（evidence_mode=evidence_backed）的条目仍落 draft，等证据事后
挂上再走 confirm_item 的 evidence gate；AI 草稿仍落 draft 占人审位，accept 时才推 confirmed。
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

import pytest

import research_memory_db_fixture as db_fixture
from app.models import research_memory_contracts as contracts
from app.repositories import research_memory_repository as repo_mod
from app.services import research_memory_service as svc

# 真库组的跳过条件：库不可连只跳真库用例，纯函数用例照常跑
requires_db = pytest.mark.skipif(
    not db_fixture.can_connect_db(), reason="业务库不可连，跳过研究记忆真库状态派生测试"
)

# 被验的枚举字面量集中在此，避免测试里散落裸字符串
_STATUS_DRAFT = "draft"
_STATUS_CONFIRMED = "confirmed"
_REVIEW_PENDING = "pending_review"
_REVIEW_ACCEPTED = "accepted"
_EVIDENCE_BACKED = "evidence_backed"
_MANUAL_SOURCE = contracts.DEFAULT_SOURCE_KIND
_AGENT_DRAFT_SOURCE = "agent_draft"
_USER_MEMORY_TYPE = "decision_note"
_FACT_MEMORY_TYPE = "metric_note"

# 非 evidence_backed 的证据模式：这两种新建即生效，不受 evidence gate 例外影响
_NON_BACKED_EVIDENCE_MODES = tuple(
    mode for mode in contracts.VALID_EVIDENCE_MODES if mode != _EVIDENCE_BACKED
)
# AI/系统来源：不在用户亲笔白名单里，新建仍落 draft 占人审位
_AI_SOURCE_KINDS = tuple(
    kind for kind in contracts.VALID_SOURCE_KINDS if kind not in contracts.USER_AUTHORED_SOURCE_KINDS
)

# 新建落库即 version=1，accept_draft 后推进到 2（只多一条 accept_draft 快照）
_VERSION_ON_CREATE = 1
_VERSION_AFTER_ACCEPT = 2


def _run(coro):
    return asyncio.run(coro)


# ─── A 组：纯函数（不连库） ───


class TestDeriveInitialStatus:
    """_derive_initial_status 的全部分支：谁新建即生效、谁仍占人审位。"""

    def test_missing_source_kind_treated_as_manual(self) -> None:
        """
        source_kind 缺键时按手写算，落 confirmed。

        前端建记忆的请求体里根本没有 source_kind 这个字段，缺键就是最常见的用户手写场景；
        若这里退回 draft，用户以为记下了、实际停在人审位。
        """
        payload = {"title": "缺 source_kind 的用户记忆", "memory_type": _USER_MEMORY_TYPE}
        assert svc._derive_initial_status(payload) == _STATUS_CONFIRMED
        # 仓储默认值必须仍在用户亲笔白名单里，缺键兜底才成立
        assert contracts.DEFAULT_SOURCE_KIND in contracts.USER_AUTHORED_SOURCE_KINDS

    @pytest.mark.parametrize("source_kind", contracts.USER_AUTHORED_SOURCE_KINDS)
    def test_user_authored_sources_confirmed(self, source_kind: str) -> None:
        """手写 / 从回答保存 / 从 chunk 查看器保存，三种用户亲笔来源新建即生效。"""
        payload = {"title": "用户判断", "memory_type": _USER_MEMORY_TYPE, "source_kind": source_kind}
        assert svc._derive_initial_status(payload) == _STATUS_CONFIRMED

    @pytest.mark.parametrize("source_kind", _AI_SOURCE_KINDS)
    def test_ai_sources_stay_draft(self, source_kind: str) -> None:
        """AI / 系统草稿仍落 draft：draft 那个人审位就是给它们留的，不能被派生规则放行。"""
        payload = {"title": "AI 草稿", "memory_type": _USER_MEMORY_TYPE, "source_kind": source_kind}
        assert svc._derive_initial_status(payload) == _STATUS_DRAFT

    @pytest.mark.parametrize("source_kind", [None, *contracts.USER_AUTHORED_SOURCE_KINDS])
    def test_evidence_backed_stays_draft_even_for_user(self, source_kind: str | None) -> None:
        """
        唯一例外：声称有文献背书的条目即便是用户亲笔也落 draft。

        证据要事后经 add_evidence 单独挂，直接 confirmed 会绕过 confirm_item 里
        「事实型记忆确认前必须至少绑一条证据」这道既有检查。
        """
        payload: dict[str, Any] = {
            "title": "事实判断",
            "memory_type": _FACT_MEMORY_TYPE,
            "evidence_mode": _EVIDENCE_BACKED,
        }
        if source_kind is not None:
            payload["source_kind"] = source_kind
        assert svc._derive_initial_status(payload) == _STATUS_DRAFT

    @pytest.mark.parametrize("evidence_mode", _NON_BACKED_EVIDENCE_MODES)
    def test_non_backed_modes_confirmed(self, evidence_mode: str) -> None:
        """例外只针对 evidence_backed：user_note_only / mixed 的用户记忆照常新建即生效。"""
        payload = {
            "title": "用户笔记",
            "memory_type": _USER_MEMORY_TYPE,
            "source_kind": _MANUAL_SOURCE,
            "evidence_mode": evidence_mode,
        }
        assert svc._derive_initial_status(payload) == _STATUS_CONFIRMED

    @pytest.mark.parametrize(
        "payload, expected",
        [
            # curator 走仓储直连并显式传 draft：即便 source_kind 缺键（派生本会给 confirmed），
            # 显式值也必须原样保留，这条边界是 AI 草稿人审位的最后一道保障
            ({"status": _STATUS_DRAFT}, _STATUS_DRAFT),
            # 显式 draft 叠加用户亲笔来源：显式值仍优先，不被「用户即生效」覆盖
            ({"status": _STATUS_DRAFT, "source_kind": _MANUAL_SOURCE}, _STATUS_DRAFT),
            # 显式 confirmed 叠加 AI 来源：调用方已明确要它生效，派生的 draft 不回压
            ({"status": _STATUS_CONFIRMED, "source_kind": _AGENT_DRAFT_SOURCE}, _STATUS_CONFIRMED),
            # 显式 confirmed 叠加声称有背书：显式值优先于 evidence_backed 例外
            ({"status": _STATUS_CONFIRMED, "evidence_mode": _EVIDENCE_BACKED}, _STATUS_CONFIRMED),
        ],
        ids=["curator_direct_draft", "explicit_draft_user_source", "explicit_confirmed_ai", "explicit_confirmed_backed"],
    )
    def test_explicit_status_wins_over_derivation(self, payload: dict[str, Any], expected: str) -> None:
        """payload 里显式带了 status 时一律以显式值为准，派生规则不覆盖调用方的明确意图。"""
        full_payload = {"title": "显式指定状态", "memory_type": _USER_MEMORY_TYPE, **payload}
        assert svc._derive_initial_status(full_payload) == expected

    def test_curator_full_payload_keeps_draft(self) -> None:
        """
        curator 造 AI 草稿的完整 payload 形状过一遍派生，结果必须仍是 draft。

        curator 现在走仓储直连，但显式 draft + agent_draft 来源两道都指向 draft，
        将来若它改走 service 也不会被误提成 confirmed。
        """
        curator_payload = {
            "title": "AI 抽出的候选判断",
            "memory_type": _USER_MEMORY_TYPE,
            "status": _STATUS_DRAFT,
            "review_state": _REVIEW_PENDING,
            "source_kind": _AGENT_DRAFT_SOURCE,
            "evidence_mode": _EVIDENCE_BACKED,
        }
        assert svc._derive_initial_status(curator_payload) == _STATUS_DRAFT

    @pytest.mark.parametrize("falsy_status", [None, ""], ids=["none", "empty_string"])
    def test_falsy_status_falls_back_to_derivation(self, falsy_status: Any) -> None:
        """status 带了空值不算「显式指定」，退回按来源派生，用户手写照样 confirmed。"""
        payload = {
            "title": "空 status 的用户记忆",
            "memory_type": _USER_MEMORY_TYPE,
            "source_kind": _MANUAL_SOURCE,
            "status": falsy_status,
        }
        assert svc._derive_initial_status(payload) == _STATUS_CONFIRMED

    def test_derived_status_never_leaves_enum(self) -> None:
        """穷举来源 × 证据模式，派生结果只落在 draft/confirmed 两值内，不会造出非法 status。"""
        for source_kind, evidence_mode in itertools.product(
            contracts.VALID_SOURCE_KINDS, contracts.VALID_EVIDENCE_MODES
        ):
            payload = {
                "title": "枚举遍历",
                "memory_type": _USER_MEMORY_TYPE,
                "source_kind": source_kind,
                "evidence_mode": evidence_mode,
            }
            derived = svc._derive_initial_status(payload)
            assert derived in (_STATUS_DRAFT, _STATUS_CONFIRMED), f"{source_kind}/{evidence_mode} 派生出 {derived}"
            assert derived in contracts.VALID_MEMORY_STATUSES


# ─── B 组：真库（库不可连时整个 class 跳过） ───


async def _create_pending_ai_draft(project_id: str) -> str:
    """造一条待审 AI 草稿（curator 的落库形状：draft + pending_review + agent_draft）。"""
    repo = repo_mod.get_research_memory_repository()
    detail = await repo.create_item(
        project_id,
        {
            "title": "AI 抽出的候选判断",
            "memory_type": _USER_MEMORY_TYPE,
            "status": _STATUS_DRAFT,
            "review_state": _REVIEW_PENDING,
            "source_kind": _AGENT_DRAFT_SOURCE,
        },
    )
    return detail["item"]["memory_id"]


@requires_db
class TestStatusDerivationOnRealDb:
    """在真实 Postgres 上验证派生结果真的落到了库里，而不只是函数返回值对。"""

    def test_manual_memory_lands_confirmed(self) -> None:
        """
        用户手写一条记忆、不做任何二次动作，库里就是 confirmed。

        请求体不带 source_kind（前端真实形状），派生按 manual 算，用户记下就是记下了。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                detail = await svc.create_item(
                    project_id, {"title": "我认为这个指标不可靠", "memory_type": _USER_MEMORY_TYPE}
                )
                # 重新从库里读，确认是落库结果而非入参回显
                return await svc.get_detail(detail["item"]["memory_id"])

        item = _run(scenario())["item"]
        assert item["status"] == _STATUS_CONFIRMED
        assert item["source_kind"] == _MANUAL_SOURCE
        assert item["review_state"] == _REVIEW_ACCEPTED

    def test_evidence_backed_creation_lands_draft(self) -> None:
        """
        声称有文献背书的新建仍落 draft，等证据挂上再走 confirm_item 的 evidence gate。

        直接 confirmed 会让「事实型记忆确认前必须至少绑一条证据」这道检查形同虚设。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                detail = await svc.create_item(
                    project_id,
                    {
                        "title": "该方法在长文档上更优",
                        "memory_type": _FACT_MEMORY_TYPE,
                        "evidence_mode": _EVIDENCE_BACKED,
                    },
                )
                return await svc.get_detail(detail["item"]["memory_id"])

        item = _run(scenario())["item"]
        assert item["status"] == _STATUS_DRAFT
        assert item["evidence_mode"] == _EVIDENCE_BACKED

    def test_curator_direct_repo_draft_untouched(self) -> None:
        """
        curator 走仓储直连显式传 draft 时不受 service 派生影响，三轴仍是草稿形状。

        这条通路是 AI 草稿的人审位入口，被派生规则碰到就等于草稿自动进正式库。
        """
        async def scenario() -> dict[str, Any]:
            async with db_fixture.temp_project() as project_id:
                memory_id = await _create_pending_ai_draft(project_id)
                return await svc.get_detail(memory_id)

        item = _run(scenario())["item"]
        assert item["status"] == _STATUS_DRAFT
        assert item["review_state"] == _REVIEW_PENDING
        assert item["source_kind"] == _AGENT_DRAFT_SOURCE

    def test_accept_draft_confirms_in_single_snapshot(self) -> None:
        """
        接纳 AI 草稿时 status 与 review_state 在同一次变更里一起推进，只多一条版本快照。

        只置 review_state 会让接纳变成「点了没反应」；分两次写则会给用户一个动作留两条历史。
        """
        async def scenario() -> tuple[dict[str, Any], list[dict[str, Any]]]:
            async with db_fixture.temp_project() as project_id:
                memory_id = await _create_pending_ai_draft(project_id)
                before = await svc.list_versions(memory_id)
                assert len(before) == _VERSION_ON_CREATE
                await svc.accept_draft(memory_id, created_by="reviewer")
                detail = await svc.get_detail(memory_id)
                return detail["item"], await svc.list_versions(memory_id)

        item, versions = _run(scenario())
        assert item["status"] == _STATUS_CONFIRMED
        assert item["review_state"] == _REVIEW_ACCEPTED
        assert item["version"] == _VERSION_AFTER_ACCEPT
        # 接纳只追加一条 accept_draft 快照，create 之后不再有别的动作
        assert len(versions) == _VERSION_AFTER_ACCEPT
        assert [v["change_kind"] for v in versions] == ["create", "accept_draft"]
