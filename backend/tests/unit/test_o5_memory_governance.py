"""【研究记忆】【治理回归】正式路由/服务/事务的离线验证，作者：xxx。"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes import research_memory_api as api
from app.repositories import research_memory_repository as repository_module
from app.repositories import memory_source_repository
from app.services import memory_evidence_service, research_memory_service as service


class MemoryStore:
    """仅替换数据库原语；正式仓储的锁、事务顺序、版本与快照逻辑照常执行。"""

    def __init__(self):
        """无参数；建立一个确认后的文献判断及两个不同作用的来源。"""
        self.item = {"memory_id": "memory", "project_id": "project", "user_id": "owner",
                     "title": "判断", "body": "原判断", "memory_type": "decision_note", "version": 1,
                     "status": "confirmed", "review_state": "accepted", "stale_state": "fresh",
                     "evidence_mode": "evidence_backed", "stale_reasons": [], "tags": [],
                     "related_paper_ids": [], "confidence_signals": {}}
        self.refs = [{"ref_id": role, "memory_id": "memory", "paper_id": "paper", "quote_text": "原文",
                      "quote_hash": "hash", "ref_role": role} for role in ("primary_support", "context_only")]
        self.versions = []
        self.lock = asyncio.Lock()
        self.fail_snapshot = False

    @asynccontextmanager
    async def acquire(self):
        """无参数；返回唯一隔离连接。"""
        yield self

    @asynccontextmanager
    async def transaction(self):
        """无参数；事务串行并可回滚，验证引用与版本不能部分提交。"""
        async with self.lock:
            previous = deepcopy((self.item, self.refs, self.versions))
            try:
                yield self
            except BaseException:
                self.item, self.refs, self.versions = previous
                raise

    async def fetchrow(self, sql, *args):
        """参数为实际SQL；只实现本组场景需要的原语，未知请求立即失败。"""
        if sql.startswith("SELECT * FROM research_memory_items"):
            if args[0] != self.item.get("memory_id"):
                return None
            if len(args) > 1 and (args[1] != self.item["user_id"] or args[2] != self.item["project_id"]):
                return None
            return deepcopy(self.item)
        if sql.startswith("DELETE FROM memory_evidence_refs"):
            ref = next((ref for ref in self.refs if ref["ref_id"] == args[0] and ref["memory_id"] == args[1]), None)
            if ref is not None:
                self.refs.remove(ref)
            return ref
        if sql.startswith("INSERT INTO memory_evidence_refs"):
            fields = ("ref_id", "memory_id", "paper_id", "chunk_id", "parent_id", "evidence_id", "section",
                      "section_path", "page_num", "quote_text", "quote_hash", "excerpt", "claim_key", "ref_role",
                      "source_ref", "document_ref", "support", "verification")
            ref = dict(zip(fields, args))
            self.refs.append(ref)
            return deepcopy(ref)
        if sql.startswith("SELECT question_id FROM research_questions"):
            return None
        if sql.startswith("SELECT project_id FROM projects"):
            return {"project_id": "project"} if args == ("project", "owner") else None
        if sql.startswith("SELECT memory_id FROM research_memory_items"):
            return {"memory_id": self.item["memory_id"]} if self.item.get("candidate_key") == args[2] else None
        if sql.startswith("SELECT * FROM rag_runs"):
            return {"delivery_revision": 2, "metadata": {"final_generation_attempt": {"attempt_id": "new-attempt"}}}
        raise AssertionError(sql)

    async def fetch(self, sql, *args):
        """参数为关系读取SQL；返回本事务内的真实当前集合。"""
        if sql.startswith("SELECT i.* FROM research_memory_items"):
            return [deepcopy(self.item)]
        if "FROM memory_evidence_refs" in sql:
            return deepcopy(self.refs)
        if "FROM memory_versions" in sql:
            return deepcopy(self.versions)
        if "FROM memory_links" in sql or "FROM memory_question_links" in sql:
            return []
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        """参数为更新SQL；按绑定位置应用字段并记录快照，可注入快照写入失败。"""
        if sql.startswith("UPDATE research_memory_items"):
            for field, index in re.findall(r"(\w+)=\$(\d+)", sql.split(" WHERE ")[0]):
                value = args[int(index) - 1]
                self.item[field] = json.loads(value) if field in {"stale_reasons", "tags", "related_paper_ids"} else value
            self.item["version"] += 1
            return
        if "INSERT INTO memory_versions" in sql:
            if self.fail_snapshot:
                raise OSError("snapshot failed")
            self.versions.append({"version": args[3], "snapshot": json.loads(args[4])})
            return
        if sql.startswith("UPDATE memory_evidence_refs"):
            ref = next(ref for ref in self.refs if ref["ref_id"] == args[0])
            ref.update(dict(zip(("source_ref", "document_ref", "support", "verification"), args[2:])))
            return
        if "INSERT INTO research_memory_items" in sql:
            keys = ("memory_id", "project_id", "user_id", "memory_type", "title", "body", "status", "stale_state",
                    "review_state", "maturity", "evidence_mode", "related_paper_ids", "tags", "source_kind",
                    "source_session_id", "source_conversation_id", "source_answer_id", "confidence_score",
                    "confidence_signals", "possible_duplicate", "candidate_key", "source_attempt_id", "source_claim_id", "stale_reasons")
            self.item = {**dict(zip(keys, args)), "version": 1}
            return
        if sql.startswith("DELETE FROM research_memory_items"):
            self.item, self.refs, self.versions = {}, [], []
            return
        raise AssertionError(sql)


@pytest.fixture
def store(monkeypatch):
    """参数为补丁工具；使用正式仓储和服务，只隔离连接、项目校验及精确原文读取。"""
    store = MemoryStore()
    repository = repository_module.ResearchMemoryRepository()
    monkeypatch.setattr(repository_module, "get_pool", AsyncMock(return_value=store))
    monkeypatch.setattr(repository_module, "fetchrow", store.fetchrow)
    monkeypatch.setattr(repository_module, "fetch", store.fetch)
    monkeypatch.setattr(service, "get_research_memory_repository", lambda: repository)
    monkeypatch.setattr(service, "_schema_ready", True)
    monkeypatch.setattr(api.project_service, "require_project", AsyncMock(return_value={"project_id": "project"}))
    monkeypatch.setattr(memory_source_repository, "revalidate_memory_sources", AsyncMock())

    async def validate(evidence, **scope):
        """参数为选择的引用与可信范围；只模拟来源端口，保留治理端支持判断。"""
        assert scope == {"user_id": "owner", "project_id": "project"}
        return {**evidence, "verification": {"current_batch_id": "batch"}}

    monkeypatch.setattr(memory_evidence_service, "validate_memory_evidence", validate)
    return store


@pytest.mark.parametrize("model,payload", [
    (api.UpdateMemoryRequest, {"expected_version": 1, "status": "confirmed"}),
    (api.UpdateMemoryRequest, {"expected_version": 1, "bogus": "ignored"}),
    (api.AddEvidenceRequest, {"paper_id": "paper", "quote_text": "原文"}),
])
def test_unknown_or_unversioned_commands_are_rejected(model, payload):
    """参数为外部命令；不能忽略未知字段、暗中确认或遗漏期望版本。"""
    with pytest.raises(ValidationError):
        model(**payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["detail", "versions", "confirm", "edit", "delete", "accept", "remove_ref"])
@pytest.mark.parametrize("foreign_field", ["user_id", "project_id"])
async def test_owned_project_url_cannot_access_foreign_memory(store, action, foreign_field):
    """参数为正式操作；自有项目URL不能授权另一个owner的具体记忆。"""
    store.item[foreign_field] = "other"
    user = {"user_id": "owner"}
    operations = {
        "detail": lambda: api.get_memory("project", "memory", user=user),
        "versions": lambda: api.list_versions("project", "memory", user=user),
        "confirm": lambda: api.confirm_memory("project", "memory", api.VersionRequest(expected_version=1), user=user),
        "edit": lambda: api.update_memory("project", "memory", api.UpdateMemoryRequest(expected_version=1, body="篡改"), user=user),
        "delete": lambda: api.delete_memory("project", "memory", expected_version=1, user=user),
        "accept": lambda: api.accept_draft("project", "memory", api.AcceptDraftRequest(expected_version=1), user=user),
        "remove_ref": lambda: api.remove_evidence("project", "memory", "primary_support", expected_version=1, user=user),
    }
    with pytest.raises(HTTPException) as error:
        await operations[action]()
    assert error.value.status_code == 404
    assert store.item["version"] == 1


@pytest.mark.asyncio
async def test_body_edit_requires_explicit_confirmation(store):
    """参数为隔离事务；正文改动回草稿，保存并确认可一次提交 fresh 新版本。"""
    first = await api.update_memory("project", "memory", api.UpdateMemoryRequest(expected_version=1, body="新判断"),
                                    user={"user_id": "owner"})
    assert first["item"]["status"] == "draft"
    second = await api.update_memory("project", "memory", api.UpdateMemoryRequest(expected_version=2, body="再判断", confirm=True),
                                     user={"user_id": "owner"})
    assert second["item"]["status"] == "confirmed"
    assert second["item"]["stale_state"] == "fresh"
    assert [row["version"] for row in store.versions] == [2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("role,expected", [("primary_support", "needs_rebind"), ("context_only", "fresh")])
async def test_remove_support_uses_role_not_remaining_reference_count(store, role, expected):
    """参数为来源角色；剩余背景引用不能保住被删主支持，删背景不误伤确认。"""
    result = await api.remove_evidence("project", "memory", role, expected_version=1, user={"user_id": "owner"})
    assert result["item"]["stale_state"] == expected
    assert result["item"]["version"] == 2
    assert len(result["evidence_refs"]) == 1


@pytest.mark.asyncio
async def test_accept_and_confirm_share_evidence_gate(store):
    """参数为无支持AI草稿；接纳不能绕过确认使用的支持资格。"""
    store.refs = []
    store.item.update(status="draft", review_state="pending_review")
    with pytest.raises(HTTPException) as error:
        await api.accept_draft("project", "memory", api.AcceptDraftRequest(expected_version=1), user={"user_id": "owner"})
    assert error.value.status_code == 422
    assert store.item["review_state"] == "pending_review"
    store.item["evidence_mode"] = "user_note_only"
    result = await api.accept_draft("project", "memory", api.AcceptDraftRequest(expected_version=1), user={"user_id": "owner"})
    assert result["item"]["status"] == "confirmed"


@pytest.mark.asyncio
async def test_bind_and_confirm_is_single_version_transaction(store):
    """参数为失效记忆；新原文绑定并确认一次提交恢复fresh且清除已解决原因。"""
    store.item.update(stale_state="needs_rebind", stale_reasons=["旧依据移除"])
    store.refs = []
    result = await api.add_evidence("project", "memory", api.AddEvidenceRequest(expected_version=1,
        paper_id="paper", quote_text="原文", ref_role="primary_support", confirm=True), user={"user_id": "owner"})
    assert result["item"]["version"] == 2
    assert result["item"]["stale_reasons"] == []
    assert result["item"]["stale_state"] == "fresh"
    assert len(store.versions) == 1
    assert len(store.versions[0]["snapshot"]["evidence_refs"]) == 1


@pytest.mark.asyncio
async def test_competing_version_edits_have_one_winner(store):
    """参数为隔离事务；两个相同版本编辑只能有一次成功，迟到者返回409。"""
    results = await asyncio.gather(*[api.update_memory("project", "memory",
        api.UpdateMemoryRequest(expected_version=1, title=title), user={"user_id": "owner"})
        for title in ("A", "B")], return_exceptions=True)
    assert len([result for result in results if isinstance(result, dict)]) == 1
    errors = [result for result in results if isinstance(result, HTTPException)]
    assert len(errors) == 1 and errors[0].status_code == 409
    assert store.item["version"] == 2


@pytest.mark.asyncio
async def test_snapshot_failure_rolls_back_reference_and_version(store):
    """参数为失败事务；快照写入失败时引用删除和version更新必须一起回滚。"""
    store.fail_snapshot = True
    with pytest.raises(OSError, match="snapshot failed"):
        await api.remove_evidence("project", "memory", "primary_support", expected_version=1, user={"user_id": "owner"})
    assert len(store.refs) == 2
    assert store.item["version"] == 1


@pytest.mark.asyncio
async def test_question_relation_rejects_foreign_target_in_transaction(store):
    """参数为异项目问题；源记忆可见也不能授权问题端，事务不改变记忆版本。"""
    with pytest.raises(HTTPException) as error:
        await api.attach_question("project", "memory", "foreign-question", api.VersionRequest(expected_version=1),
                                  user={"user_id": "owner"})
    assert error.value.status_code == 404
    assert store.item["version"] == 1


@pytest.mark.asyncio
async def test_link_checks_destination_owner_before_writing(store):
    """参数为隔离事务；可见源记忆不能授权另一项目或用户的关系目标。"""
    with pytest.raises(HTTPException) as error:
        await api.add_link("project", "memory", api.AddLinkRequest(expected_version=1,
            dst_expected_version=1, dst_memory_id="foreign", link_type="based_on"), user={"user_id": "owner"})
    assert error.value.status_code == 404
    assert store.versions == []


@pytest.mark.asyncio
async def test_confirm_cannot_overwrite_concurrent_evidence_removal(store, monkeypatch):
    """参数为隔离事务；原文校验期间支持被删，旧确认CAS必须409，不能恢复fresh。"""
    validating, release = asyncio.Event(), asyncio.Event()

    async def validate(ref, **scope):
        """参数为引用；模拟文件核验等待，让解绑先提交。"""
        validating.set()
        await release.wait()
        return ref

    monkeypatch.setattr(memory_evidence_service, "validate_memory_evidence", validate)
    confirmation = asyncio.create_task(api.confirm_memory("project", "memory", api.VersionRequest(expected_version=1),
                                                           user={"user_id": "owner"}))
    await validating.wait()
    await api.remove_evidence("project", "memory", "primary_support", expected_version=1, user={"user_id": "owner"})
    release.set()
    with pytest.raises(HTTPException) as error:
        await confirmation
    assert error.value.status_code == 409
    assert store.item["stale_state"] == "needs_rebind"


@pytest.mark.asyncio
async def test_title_edit_keeps_confirmation_and_draft_command_is_real(store):
    """参数为隔离事务；整理标题保持确认，主动回草稿必须改变真实状态。"""
    renamed = await api.update_memory("project", "memory", api.UpdateMemoryRequest(expected_version=1, title="标题"),
                                      user={"user_id": "owner"})
    assert renamed["item"]["status"] == "confirmed"
    draft = await api.update_memory("project", "memory", api.UpdateMemoryRequest(expected_version=2, status="draft"),
                                    user={"user_id": "owner"})
    assert draft["item"]["status"] == "draft"


@pytest.mark.asyncio
async def test_candidate_replay_has_one_initial_snapshot_with_refs(store):
    """参数为隔离事务；同attempt/claim重放不重复候选，首个版本必须已有精确引用。"""
    store.item, store.refs = {}, []
    repo = service.get_research_memory_repository()
    payload = {"user_id": "owner", "title": "候选", "memory_type": "decision_note", "source_answer_id": "run",
               "source_attempt_id": "attempt", "source_claim_id": "claim", "evidence_mode": "evidence_backed"}
    refs = [{"paper_id": "paper", "quote_text": "原文", "quote_hash": "hash", "ref_role": "primary_support"}]
    first = await repo.create_candidate("project", payload, refs, candidate_key="attempt/claim")
    second = await repo.create_candidate("project", payload, refs, candidate_key="attempt/claim")
    assert first["item"]["memory_id"] == second["item"]["memory_id"]
    assert len(store.versions) == 1
    assert len(store.versions[0]["snapshot"]["evidence_refs"]) == 1


@pytest.mark.asyncio
async def test_candidate_snapshot_failure_rolls_back_whole_creation(store):
    """参数为隔离事务；初始快照写入失败，候选与已插入引用也全部回滚。"""
    store.item, store.refs, store.fail_snapshot = {}, [], True
    with pytest.raises(OSError, match="snapshot failed"):
        await service.get_research_memory_repository().create_candidate("project",
            {"user_id": "owner", "title": "候选", "memory_type": "decision_note", "source_answer_id": "run"},
            [{"paper_id": "paper", "quote_text": "原文", "quote_hash": "hash"}], candidate_key="attempt/claim")
    assert store.item == {} and store.refs == [] and store.versions == []


@pytest.mark.asyncio
async def test_management_all_list_keeps_owner_and_includes_pending(store, monkeypatch):
    """参数为管理列表；all仅取消审核/状态筛选，不取消可信用户和项目范围。"""
    listing = AsyncMock(return_value=[{**store.item, "review_state": "pending_review"}])
    monkeypatch.setattr(service.get_research_memory_repository(), "list_items", listing)
    result = await api.list_memory("project", question_id=None, memory_type=None, paper_id=None, tag=None,
        status="all", stale_state=None, maturity=None, user={"user_id": "owner"})
    assert result[0]["review_state"] == "pending_review"
    listing.assert_awaited_once_with("project", user_id="owner", status=None, review_state=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,status", [("user_note_only", "confirmed"), ("evidence_backed", "draft"), ("mixed", "draft")])
async def test_manual_create_keeps_personal_judgment_and_document_gate(store, mode, status):
    """参数为用户声明的证据模式；手写个人判断保留用途，声明文献支持先进入待确认草稿。"""
    store.item, store.refs = {}, []
    detail = await api.create_memory("project", api.CreateMemoryRequest(memory_type="decision_note", title="手写",
        evidence_mode=mode), user={"user_id": "owner"})
    assert detail["item"]["status"] == status
    assert detail["item"]["user_id"] == "owner"


@pytest.mark.asyncio
async def test_delete_memory_requires_current_version_and_commits_real_delete(store):
    """参数为隔离事务；删除先检查用户所见版本，通过后才执行级联删除。"""
    with pytest.raises(HTTPException) as error:
        await api.delete_memory("project", "memory", expected_version=2, user={"user_id": "owner"})
    assert error.value.status_code == 409
    assert store.item["version"] == 1
    result = await api.delete_memory("project", "memory", expected_version=1, user={"user_id": "owner"})
    assert result["status"] == "deleted"
    assert store.item == {} and store.refs == []


@pytest.mark.asyncio
async def test_old_candidate_proposal_cannot_write_after_final_result_changed(store):
    """参数为候选事务；模型提议期间目标答案已更新，旧attempt候选必须在写入前失败。"""
    from app.models.research_memory_contracts import MemoryVersionConflict

    store.item, store.refs = {}, []
    with pytest.raises(MemoryVersionConflict, match="来源答案已有新版本"):
        await service.get_research_memory_repository().create_candidate("project",
            {"user_id": "owner", "title": "旧候选", "memory_type": "decision_note", "source_answer_id": "run",
             "source_delivery_revision": 1, "source_attempt_id": "old-attempt", "source_claim_id": "claim"},
            [], candidate_key="old-attempt/claim")
    assert store.item == {} and store.versions == []
