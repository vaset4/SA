"""【研究记忆】【来源回归】正式核验、候选和失效入口的隔离验证，作者：xxx。"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.source_access_models import SourceVersionChange
from app.repositories import memory_source_repository as sources
from app.services import memory_evidence_service as evidence
from app.services import research_memory_curator_service as curator
from app.services import research_memory_stale_service as stale
from app.services.internal_research import research_memory_writeback as writeback
from app.services.ingest.upload_artifacts import write_json_artifact
from test_source_lifecycle_o5 import source as _source_fixture
from test_source_lifecycle_o5 import BATCH_A, BATCH_B

source_fixture = _source_fixture


@pytest.fixture
def memory_source(source_fixture, monkeypatch):
    """参数为隔离文件与替身；实际核验函数仍读取SHA绑定的临时原文清单。"""
    monkeypatch.setattr(evidence, "read_source_state", AsyncMock(return_value={"current_batch_id": BATCH_A}))
    monkeypatch.setattr(evidence, "get_source_access_repository", lambda: source_fixture.gate)
    monkeypatch.setattr(evidence, "get_upload_publication_repository", lambda: source_fixture.repository)
    return source_fixture


@pytest.mark.asyncio
async def test_quote_verified_from_original_and_wrong_paper_rejected(memory_source):
    """原始引用读取成功，伪造论文与chunk关联必须拒绝，向量无需存在。"""
    payload = {"paper_id": "paper", "chunk_id": f"v3:{BATCH_A}:0", "quote_text": "原版本"}
    ref = await evidence.validate_memory_evidence(payload, user_id="owner", project_id="project")
    assert ref["verification"]["source_start"] == 0
    assert ref["verification"]["source_end"] == 3
    with pytest.raises(evidence.MemoryEvidenceError, match="不属于"):
        await evidence.validate_memory_evidence({**payload, "paper_id": "other"}, user_id="owner", project_id="project")


@pytest.mark.asyncio
async def test_quote_ambiguity_requires_concrete_occurrence(memory_source):
    """同一块原句重复时返回两处真实坐标，选择具体位置后方可核验。"""
    chunk = {"chunk_id": f"v3:{BATCH_A}:0", "paper_id": "paper", "batch_id": BATCH_A,
             "parent_id": "parent", "full_text": "依据 与 依据"}
    manifest = {"format": 3, "paper_id": "paper", "user_id": "owner", "batch_id": BATCH_A, "chunks": [chunk]}
    memory_source.database.manifest_ref = write_json_artifact(memory_source.root, manifest)
    payload = {"paper_id": "paper", "quote_text": "依据"}
    with pytest.raises(evidence.MemoryEvidenceError) as caught:
        await evidence.validate_memory_evidence(payload, user_id="owner", project_id="project")
    assert [item["source_start"] for item in caught.value.candidates] == [0, 5]
    ref = await evidence.validate_memory_evidence({**payload, **caught.value.candidates[1]}, user_id="owner", project_id="project")
    assert ref["verification"]["source_start"] == 5


@pytest.mark.asyncio
async def test_current_scope_denial_happens_before_original_read(memory_source, monkeypatch):
    """当前项目已撤回论文时，原文读取和文件占用都不开始。"""
    monkeypatch.setattr(evidence, "read_source_state", AsyncMock(side_effect=LookupError("移出项目")))
    loader = AsyncMock()
    monkeypatch.setattr(evidence, "_manifest", loader)
    with pytest.raises(LookupError):
        await evidence.validate_memory_evidence({"paper_id": "paper", "quote": "原文"}, user_id="owner", project_id="project")
    loader.assert_not_called()


@pytest.mark.parametrize("classification,invalidated", [("initial", False), ("unchanged", False), ("changed", True), ("unknown", True)])
@pytest.mark.asyncio
async def test_publication_classification_changes_memory_in_same_connection(monkeypatch, classification, invalidated):
    """索引重建保持fresh，实际正文变化和未知来源使支持失效，且使用发布连接。"""
    change = SourceVersionChange(paper_id="paper", user_id="owner", previous_batch_id=BATCH_A,
        batch_id=BATCH_B, previous_manifest_sha256="old", manifest_sha256="new", classification=classification,
        source_sha256="source", document_sha256="document", normalized_sha256="normalized")
    invalidate = AsyncMock()
    monkeypatch.setattr(sources, "invalidate_source", invalidate)
    connection = object()
    await sources.apply_source_change(connection, change)
    assert invalidate.await_count == int(invalidated)
    if invalidated:
        assert invalidate.await_args.args[0] is connection


@pytest.mark.asyncio
async def test_transaction_rejects_source_changed_after_file_validation(monkeypatch):
    """文件核验后发布新批次，确认事务必须拒绝原检查事实。"""
    monkeypatch.setattr(sources, "source_state", AsyncMock(return_value={"current_batch_id": BATCH_B}))
    with pytest.raises(ValueError, match="更新"):
        await sources.revalidate_memory_sources(object(), user_id="owner", project_id="project",
            evidence_refs=[{"paper_id": "paper", "verification": {"current_batch_id": BATCH_A}}])


@pytest.mark.asyncio
async def test_invalid_curator_quote_stays_evidence_draft_not_personal_note(monkeypatch):
    """无效文献提议保留错误与原提议待审，不伪装成无文献个人判断。"""
    repository = SimpleNamespace(ensure_schema=AsyncMock(), create_candidate=AsyncMock(return_value={"item": {"memory_id": "m"}}))
    monkeypatch.setattr(curator, "get_research_memory_repository", lambda: repository)
    monkeypatch.setattr(curator, "validate_memory_evidence", AsyncMock(side_effect=evidence.MemoryEvidenceError("引文不存在")))
    candidate = curator.CuratorMemoryCandidate(title="判断", body="正文", memory_type="paper_note",
        evidence=[curator.CuratorEvidenceCandidate(paper_id="paper", quote="编造引文")])
    await curator.persist_candidates("project", [candidate], user_id="owner", source_answer_id="answer")
    payload = repository.create_candidate.await_args.args[1]
    assert payload["evidence_mode"] == "evidence_backed" and payload["stale_state"] == "needs_rebind"
    assert payload["stale_reasons"] == ["引文不存在"] and payload["confidence_signals"]["proposed_evidence"]


@pytest.mark.asyncio
async def test_candidate_replay_uses_stable_claim_attempt_key(monkeypatch):
    """相同已提交claim/attempt重放复用同一键，内容整理变化也不重复创建该来源候选。"""
    repository = SimpleNamespace(ensure_schema=AsyncMock(), create_candidate=AsyncMock(return_value={"item": {"memory_id": "m"}}))
    monkeypatch.setattr(curator, "get_research_memory_repository", lambda: repository)
    candidate = curator.CuratorMemoryCandidate(title="判断", memory_type="paper_note", source_claim_id="claim", source_attempt_id="attempt")
    for title in ("判断", "编辑后的判断"):
        await curator.persist_candidates("project", [candidate.model_copy(update={"title": title})], user_id="owner", source_answer_id="run")
    keys = [call.kwargs["candidate_key"] for call in repository.create_candidate.await_args_list]
    assert keys[0] == keys[1]


def test_inherited_candidate_maps_only_selected_support_range():
    """继承主张沿支持闭包保留精确片段，不把整材料或整单元作为引用。"""
    selected = SimpleNamespace(evidence_id="e", field="expanded_text", start=2, end=4,
                               model_dump=lambda **kwargs: {"start": 2, "end": 4})
    upstream = SimpleNamespace(claim=SimpleNamespace(supporting_spans=[selected]))
    support = SimpleNamespace(claim=SimpleNamespace(claim_id="derived", supporting_spans=[]), claim_ids=("upstream",), citable_evidence_ids=("e",))
    record = {"expanded_text": "背景依据结尾", "source_spans": [{"paper_id": "paper", "chunk_id": f"v3:{BATCH_A}:0",
        "field": "full_text", "start": 10, "end": 16, "text_start": 0, "text_end": 6}]}
    refs = writeback._candidate_support_evidence(support, supports={"derived": support, "upstream": upstream}, registry={"e": record})
    assert len(refs) == 1 and refs[0].quote == "依据"
    assert refs[0].source_start == 12 and refs[0].ref_role == "derived_basis"
    assert refs[0].support["supporting_claim_id"] == "upstream"


@pytest.mark.asyncio
async def test_legacy_memory_missing_support_is_marked_and_storage_failure_not_empty(monkeypatch):
    """旧文献型无主支持进入待重绑；存储故障仍是错误，不伪装为空记忆池。"""
    item = {"memory_id": "m", "project_id": "project", "user_id": "owner", "version": 1, "evidence_mode": "evidence_backed"}
    repository = SimpleNamespace(list_evidence_refs=AsyncMock(return_value=[]))
    monkeypatch.setattr(stale, "get_research_memory_repository", lambda: repository)
    mark = AsyncMock()
    monkeypatch.setattr(stale, "mark_checked_stale", mark)
    assert await stale.checked_memory_items([item], project_id="project", user_id="owner") == []
    mark.assert_awaited_once()
    repository.list_evidence_refs.side_effect = RuntimeError("storage unavailable")
    with pytest.raises(RuntimeError):
        await stale.checked_memory_items([item], project_id="project", user_id="owner")


@pytest.mark.asyncio
async def test_model_rechecks_cached_memory_each_consumption(monkeypatch):
    """首次模型消费后记忆版本变更，后续同轮模型调用在发送前停止。"""
    item = {"memory_id": "m", "project_id": "project", "user_id": "owner", "version": 1,
        "review_state": "accepted", "status": "confirmed", "stale_state": "fresh", "evidence_mode": "user_note_only"}
    repository = SimpleNamespace(get_item=AsyncMock(side_effect=lambda *args, **kwargs: dict(item)))
    monkeypatch.setattr(stale, "get_research_memory_repository", lambda: repository)
    model = SimpleNamespace(ainvoke=AsyncMock(return_value="ok"))
    state = {"project_scope": {"user_id": "owner", "project_id": "project"}, "memory_read_result": {"items": [{"memory_id": "m", "version": 1}]}}
    consumer = stale.MemoryCheckedModel(model, state)
    assert await consumer.ainvoke("first") == "ok"
    item["version"] = 2
    with pytest.raises(evidence.MemoryEvidenceError):
        await consumer.ainvoke("second")
    assert model.ainvoke.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("processing_source,owner", [("delivery_run", "operation"), ("http_retry", None)])
async def test_writeback_uses_committed_projection_and_persists_independent_failure(monkeypatch, processing_source, owner):
    """提交后写回使用数据库同revision正文，候选失败单独落状态，不重新生成或撤销答案。"""
    from app.repositories import rag_run_repository, result_delivery_repository

    run = {"run_id": "run", "user_id": "owner", "answer": "已保存正文", "metadata": {"delivery_revision": 2,
        "memory_writeback": {"status": "skipped", "reason_codes": ["awaiting_result_commit"]}}}
    monkeypatch.setattr(rag_run_repository, "get_rag_run_repository", lambda: SimpleNamespace(get_run=AsyncMock(return_value=deepcopy(run))))
    save = AsyncMock()
    monkeypatch.setattr(result_delivery_repository, "ResultDeliveryRepository", lambda: SimpleNamespace(record_memory_writeback=save))
    result = {"status": "failed", "candidate_ids": [], "retryable": True, "error_kind": "ModelError", "reason_codes": []}
    writer = AsyncMock(return_value=result)
    monkeypatch.setattr(writeback, "write_back_turn_memory", writer)
    assert await writeback.after_result_committed({"run": run, "delivery_revision": 2, "operation_run_id": "operation"},
                                                processing_source=processing_source) == result
    assert writer.await_args.args[0]["answer"] == "已保存正文"
    assert [call.kwargs["outcome"]["status"] for call in save.await_args_list] == ["pending", "failed"]
    assert all(call.kwargs["delivery_revision"] == 2 for call in save.await_args_list)
    pending = save.await_args_list[0].kwargs["outcome"]
    assert pending["processing_source"] == processing_source and pending["processing_run_id"] == owner


@pytest.mark.parametrize("case,expected", [("index", True), ("body", False), ("source", False), ("unknown", False)])
def test_historical_original_freshness_uses_content_identity(case, expected):
    """相同正文重建保持支持，正文改变/原文件改变/旧身份缺失不能继续授信。"""
    old = {"batch_id": BATCH_A, "source_ref": {"sha256": "original"}, "document_ref": {"sha256": "parsed-a"},
        "content_identity": {"version": 1, "normalized_sha256": "same-text"}}
    new = {**old, "batch_id": BATCH_B, "document_ref": {"sha256": "parsed-b"}}
    if case == "body":
        new["content_identity"] = {"version": 1, "normalized_sha256": "new-text"}
    if case == "source":
        new["source_ref"] = {"sha256": "replaced-file"}
    if case == "unknown":
        old.pop("content_identity")
    assert evidence._same_content(old, new) is expected


@pytest.mark.asyncio
async def test_curate_rejects_foreign_scope_before_model(monkeypatch):
    """拥有项目URL不代表拥有所选论文；未经授权全文不能送入整理模型。"""
    from app.services import project_service

    monkeypatch.setattr(project_service, "require_project", AsyncMock(return_value={"project_id": "project"}))
    repository = SimpleNamespace(list_items=AsyncMock(return_value=[]))
    monkeypatch.setattr(curator, "get_research_memory_repository", lambda: repository)
    monkeypatch.setattr(curator, "read_source_state", AsyncMock(side_effect=LookupError("论文不在项目")))
    model = AsyncMock()
    monkeypatch.setattr(curator, "_invoke_curator_llm", model)
    with pytest.raises(LookupError):
        await curator.curate("project", {"paper_ids": ["foreign-paper"]}, user_id="owner")
    model.assert_not_called()


@pytest.mark.asyncio
async def test_invalidation_keeps_original_body_and_records_version(monkeypatch):
    """来源失效同连接修改资格并记录版本，不删除历史原文引用或覆盖判断正文。"""
    from app.repositories import research_memory_repository

    connection = SimpleNamespace(fetch=AsyncMock(return_value=[{"memory_id": "m", "stale_state": "fresh", "stale_reasons": []}]),
                                 execute=AsyncMock())
    snapshot = AsyncMock()
    monkeypatch.setattr(research_memory_repository, "get_research_memory_repository", lambda: SimpleNamespace(_snapshot=snapshot))
    assert await sources.invalidate_source(connection, user_id="owner", paper_id="paper", reason="来源已变化") == 1
    sql = connection.execute.await_args.args[0]
    assert "version=version+1" in sql and "DELETE" not in sql and "body=" not in sql
    assert snapshot.await_args.args[0] is connection


@pytest.mark.asyncio
async def test_source_state_checks_scope_membership_before_batch():
    """事务级复核按项目默认知识库或显式论文范围授权，不能只核对论文owner。"""
    connection = SimpleNamespace(fetchrow=AsyncMock(side_effect=[
        {"paper_id": "paper", "library_id": "foreign", "index_batch_id": BATCH_A},
        {"user_id": "owner", "default_library_id": "project-library"},
        {"paper_ids": [], "library_ids": [], "excluded_paper_ids": []}]))
    with pytest.raises(LookupError, match="项目范围"):
        await sources.source_state(connection, user_id="owner", project_id="project", paper_id="paper")
    assert connection.fetchrow.await_count == 3
