"""隔离 PostgreSQL schema 中验证真实上传事务，作者：xxx。"""
from __future__ import annotations

from dataclasses import replace
from typing import Any
import uuid

import asyncpg
import pytest
import pytest_asyncio

from app.config import get_settings
from app.models.upload_models import UploadConflictError, UploadFailure, UploadPausedError, UploadRequest
from app.repositories.library_repository import SCHEMA_SQL as LIBRARY_SCHEMA
from app.repositories.paper_repository import SCHEMA_SQL as PAPER_SCHEMA
from app.repositories.project_repository import SCHEMA_SQL as PROJECT_SCHEMA
from app.repositories.upload_repository import UploadRepository
from app.services.ingest.upload_artifacts import fingerprint, write_artifact, write_json_artifact
from app.services.ingest.upload_recovery import UploadStageRunner
from app.services.ingest.mineru_client import MinerUQuotaError, MinerUTemporaryError


@pytest_asyncio.fixture
async def upload_database(tmp_path):
    """创建本测试专属 schema 和真实连接池，结束后只清理该 schema。"""
    schema = "upload_test_" + uuid.uuid4().hex
    dsn = get_settings().database_url
    bootstrap = await asyncpg.connect(dsn, timeout=5)
    await bootstrap.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, command_timeout=5, server_settings={"search_path": schema})
    try:
        async with pool.acquire() as connection:
            await connection.execute(LIBRARY_SCHEMA + PAPER_SCHEMA + PROJECT_SCHEMA)
            await connection.execute("INSERT INTO libraries(library_id,name,owner) VALUES('default','Default','system')")
            await connection.execute("INSERT INTO projects(project_id,user_id,name) VALUES('project-a','a','A'),('project-b','a','B')")
        async def pool_provider():
            """返回本测试的真实隔离连接池。"""
            return pool
        yield {"repository": UploadRepository(pool_provider), "pool": pool,
               "source": write_artifact(tmp_path, b"%PDF-1.4\nunit source", suffix=".pdf")}
    finally:
        await pool.close()
        if not schema.startswith("upload_test_") or len(schema) != len("upload_test_") + 32:
            raise RuntimeError("拒绝清理未经核对的测试 schema")
        await bootstrap.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await bootstrap.close()


def _request(database: dict[str, Any], *, user_id: str = "a", project_id: str | None = None) -> UploadRequest:
    """建立不同上传意图但相同原文件的请求，返回真实仓储输入合同。"""
    source = database["source"]
    return UploadRequest(user_id, uuid.uuid4().hex, fingerprint([source["sha256"], "default", project_id]),
        "upload_" + uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex, "default", project_id, "paper.pdf", source)


async def _claim(database: dict[str, Any], task_id: str, worker_id: str | None = None) -> dict[str, Any]:
    """为测试任务授予真实数据库租约，返回 worker 将消费的完整记录。"""
    from app.repositories.task_repository import _task_run_data

    async with database["pool"].acquire() as connection:
        row = await connection.fetchrow("UPDATE task_runs SET status='running',worker_id=$2,lease_until=NOW()+INTERVAL '1 minute' WHERE task_id=$1 RETURNING *", task_id, worker_id or uuid.uuid4().hex)
    return _task_run_data(row)


@pytest.mark.asyncio
async def test_concurrent_new_intents_reuse_one_paper_and_task(upload_database) -> None:
    """多个并发新意图上传同一文件时，业务记录与任务都只能创建一次。"""
    import asyncio

    repository = upload_database["repository"]
    requests = [_request(upload_database) for _ in range(6)]
    results = await asyncio.gather(*(repository.accept(request) for request in requests))
    assert len({result["task_id"] for result in results}) == 1
    assert len({result["payload"]["paper_id"] for result in results}) == 1
    async with upload_database["pool"].acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM task_runs") == 1
        assert await connection.fetchval("SELECT count(*) FROM upload_documents") == 1
        assert await connection.fetchval("SELECT count(*) FROM upload_requests") == 6


@pytest.mark.asyncio
async def test_same_request_replays_and_changed_input_conflicts(upload_database) -> None:
    """原请求可查回，改变原文或范围不能借相同标识被误复用。"""
    repository = upload_database["repository"]
    request = _request(upload_database)
    first = await repository.accept(request)
    replay = await repository.accept(request)
    assert replay["task_id"] == first["task_id"]
    assert replay["reuse_kind"] == "same_request"
    assert (await repository.get_request(request.request_id, user_id="a"))["task_id"] == first["task_id"]
    assert await repository.get_request(request.request_id, user_id="b") is None
    with pytest.raises(UploadConflictError, match="发生变化"):
        await repository.accept(replace(request, request_fingerprint="different"))


@pytest.mark.asyncio
async def test_same_file_different_users_never_share_logical_paper(upload_database) -> None:
    """相同文件跨账号分别受理，无法读取对方论文受理记录。"""
    repository = upload_database["repository"]
    first = await repository.accept(_request(upload_database, user_id="a"))
    second = await repository.accept(_request(upload_database, user_id="b"))
    assert first["payload"]["paper_id"] != second["payload"]["paper_id"]
    assert await repository.get_document(first["payload"]["paper_id"], user_id="b") is None
    with pytest.raises(LookupError, match="项目"):
        await repository.accept(_request(upload_database, user_id="b", project_id="project-a"))


@pytest.mark.asyncio
async def test_paused_duplicate_stays_paused_and_resume_is_idempotent(upload_database) -> None:
    """重复选择文件不解除暂停，连续点击继续只排队一次且历史错误仍存在。"""
    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    upload_database["paper_id"] = accepted["payload"]["paper_id"]
    execution = await repository.begin_attempt(await _claim(upload_database, accepted["task_id"]))
    failure = UploadFailure("parse", "temporary", "temporary provider failure", retryable=True)
    await repository.reserve_attempt(execution, stage="parse", limit=3)
    assert await repository.record_retry(execution, failure) == 1
    await repository.pause(execution, failure)
    duplicate = await repository.accept(_request(upload_database))
    assert duplicate["status"] == "paused"
    assert duplicate["task_id"] == accepted["task_id"]
    with pytest.raises(LookupError):
        await repository.resume(accepted["task_id"], user_id="b")
    resumed = await repository.resume(accepted["task_id"], user_id="a")
    again = await repository.resume(accepted["task_id"], user_id="a")
    assert resumed["status"] == again["status"] == "queued"
    assert resumed["error"] is None
    document = await repository.get_document(upload_database["paper_id"], user_id="a")
    assert document["resume_sequence"] == 1
    assert document["last_failure"]["detail"] == failure.detail
    async with upload_database["pool"].acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM task_events WHERE event_type='resumed'") == 1


@pytest.mark.asyncio
async def test_unknown_creation_is_not_repeated_and_late_receipt_is_recoverable(upload_database) -> None:
    """原创建结果未知时拒绝重复创建；匹配操作的晚到结果能成为恢复事实。"""
    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    upload_database["paper_id"] = accepted["payload"]["paper_id"]
    execution = await repository.begin_attempt(await _claim(upload_database, accepted["task_id"]))
    checkpoint = await repository.start_checkpoint(execution, stage="mineru_slot", input_hash="source-config", repeatable=False)
    with pytest.raises(UploadPausedError):
        await repository.start_checkpoint(execution, stage="mineru_slot", input_hash="source-config", repeatable=False)
    async with upload_database["pool"].acquire() as connection:
        await connection.execute("UPDATE task_runs SET worker_id='worker-b' WHERE task_id=$1", accepted["task_id"])
    await repository.complete_checkpoint(execution, checkpoint, {"batch_id": "remote-original"})
    saved = await repository.checkpoint(execution, "mineru_slot")
    assert saved["payload"]["batch_id"] == "remote-original"
    with pytest.raises(Exception, match="租约"):
        await repository.check_execution(execution)


@pytest.mark.asyncio
async def test_project_link_reuse_preserves_existing_note_and_role(upload_database) -> None:
    """跨项目复用只增加关系，旧项目中的备注与角色不能被默认值覆盖。"""
    repository = upload_database["repository"]
    first = await repository.accept(_request(upload_database, project_id="project-a"))
    paper_id = first["payload"]["paper_id"]
    async with upload_database["pool"].acquire() as connection:
        await connection.execute("INSERT INTO papers(paper_id,user_id,title,abstract) VALUES($1,'a','Paper','')", paper_id)
        await connection.execute("INSERT INTO project_paper_links(project_id,paper_id,role,note) VALUES('project-a',$1,'reference','keep note')", paper_id)
    await repository.accept(_request(upload_database, project_id="project-a"))
    second = await repository.accept(_request(upload_database, project_id="project-b"))
    assert second["payload"]["paper_id"] == paper_id
    async with upload_database["pool"].acquire() as connection:
        row = await connection.fetchrow("SELECT role,note FROM project_paper_links WHERE project_id='project-a'")
        assert dict(row) == {"role": "reference", "note": "keep note"}
        assert await connection.fetchval("SELECT count(*) FROM project_paper_links WHERE project_id='project-b'") == 1


@pytest.mark.asyncio
async def test_transient_stage_budget_survives_worker_replacement(upload_database) -> None:
    """外部调用前保存预算，换 worker 后不能从零开始重试同一失败阶段。"""
    from unittest.mock import AsyncMock

    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    context = {**upload_database, "paper_id": accepted["payload"]["paper_id"]}
    execution = await repository.begin_attempt(await _claim(context, accepted["task_id"]))
    action = AsyncMock(side_effect=MinerUTemporaryError("temporarily unavailable"))
    runner = UploadStageRunner(repository, execution, sleeper=AsyncMock())
    with pytest.raises(UploadPausedError):
        await runner.run("download", action, inputs={"source": "same"})
    assert action.await_count == 3
    replacement = await repository.begin_attempt(await _claim(context, accepted["task_id"], "worker-b"))
    after_restart = AsyncMock(return_value={"download": "done"})
    with pytest.raises(UploadPausedError, match="尝试已用完"):
        await UploadStageRunner(repository, replacement).run("download", after_restart, inputs={"source": "same"})
    after_restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_checkpoint_prevents_repeating_upstream_call(upload_database) -> None:
    """恢复时直接复用同输入的完成检查点，避免重复调用已成功上游。"""
    from unittest.mock import AsyncMock

    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    context = {**upload_database, "paper_id": accepted["payload"]["paper_id"]}
    execution = await repository.begin_attempt(await _claim(context, accepted["task_id"]))
    action = AsyncMock(return_value={"batch_id": "original"})
    runner = UploadStageRunner(repository, execution)
    assert await runner.run("slot", action, inputs="same", repeatable=False) == {"batch_id": "original"}
    assert await runner.run("slot", action, inputs="same", repeatable=False) == {"batch_id": "original"}
    assert action.await_count == 1
    with pytest.raises(UploadConflictError):
        await runner.run("slot", action, inputs="changed", repeatable=False)


@pytest.mark.asyncio
async def test_quota_failure_pauses_without_short_retry(upload_database) -> None:
    """日配额错误只调用一次，记录明确拒绝而非自动等待到下一天。"""
    from unittest.mock import AsyncMock

    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    context = {**upload_database, "paper_id": accepted["payload"]["paper_id"]}
    execution = await repository.begin_attempt(await _claim(context, accepted["task_id"]))
    action = AsyncMock(side_effect=MinerUQuotaError("daily quota exhausted", code="-60018"))
    runner = UploadStageRunner(repository, execution)
    with pytest.raises(UploadPausedError) as error:
        await runner.run("slot", action, inputs="same", repeatable=False)
    assert error.value.failure.kind == "quota"
    assert action.await_count == 1
    assert (await repository.checkpoint(execution, "slot"))["state"] == "rejected"


@pytest.mark.asyncio
async def test_uncertain_creation_is_not_automatically_repeated(upload_database) -> None:
    """创建请求网络超时后仍保留 pending，不能假称未创建并重新请求。"""
    from unittest.mock import AsyncMock
    import requests

    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    context = {**upload_database, "paper_id": accepted["payload"]["paper_id"]}
    execution = await repository.begin_attempt(await _claim(context, accepted["task_id"]))
    action = AsyncMock(side_effect=requests.Timeout("unknown server result"))
    runner = UploadStageRunner(repository, execution)
    with pytest.raises(UploadPausedError) as error:
        await runner.run("slot", action, inputs="same", repeatable=False)
    assert error.value.failure.kind == "remote_result_unknown"
    assert action.await_count == 1
    assert (await repository.checkpoint(execution, "slot"))["state"] == "pending"


def _pdf_and_zip() -> tuple[bytes, bytes]:
    """创建能通过页级核验的原文与解析 ZIP，返回两份真实字节。"""
    import io
    import json
    import zipfile
    from pdf_fixtures import pdf_bytes

    source = pdf_bytes(("A verifiable document",))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("full.md", "# A verifiable document\n\nOriginal content.")
        archive.writestr("content_list.json", json.dumps([{"type": "text", "text": "Original content.", "page_idx": 0}]))
        archive.writestr("model.json", json.dumps([{"page_info": {"page_no": 0}}]))
    return source, output.getvalue()


@pytest.mark.asyncio
async def test_completed_parse_download_failure_resumes_original_remote_task(upload_database, tmp_path) -> None:
    """解析完成后下载故障，换执行者继续只下载原批次，不再申请或上传。"""
    from unittest.mock import AsyncMock, Mock
    import requests
    from app.services.ingest.mineru_client import MinerUBatchFile, MinerUExtractResult
    from app.services.ingest.upload_parse_service import UploadDocumentParser

    source, archive = _pdf_and_zip()
    database = {**upload_database, "source": write_artifact(tmp_path, source, suffix=".pdf")}
    repository = database["repository"]
    accepted = await repository.accept(_request(database))
    database["paper_id"] = accepted["payload"]["paper_id"]
    execution = await repository.begin_attempt(await _claim(database, accepted["task_id"]))
    client = Mock(timeout_seconds=5, poll_interval_seconds=0)
    client.request_upload_slot.return_value = MinerUBatchFile("remote-original", "source.pdf", "url", "url", execution.parse_id)
    client.get_result.return_value = MinerUExtractResult("remote-original", "source.pdf", "done", "result-url", {})
    client.download_result_zip.side_effect = [requests.Timeout(), requests.Timeout(), requests.Timeout(), archive]
    parser = UploadDocumentParser(runner=UploadStageRunner(repository, execution, sleeper=AsyncMock()),
        client=client, source=database["source"], filename="source.pdf", root=tmp_path)
    with pytest.raises(UploadPausedError) as failure:
        await parser._mineru_bundle({"source": database["source"]["sha256"]})
    assert failure.value.failure.stage == "mineru_artifacts"
    await repository.pause(execution, failure.value.failure)
    await repository.resume(accepted["task_id"], user_id="a")
    resumed = await repository.begin_attempt(await _claim(database, accepted["task_id"], "worker-b"))
    recovered = replace(parser, runner=UploadStageRunner(repository, resumed))
    bundle = await recovered._mineru_bundle({"source": database["source"]["sha256"]})
    assert bundle["normalized_page_count"] == 1
    assert client.request_upload_slot.call_count == 1
    assert client.upload_bytes.call_count == 1
    assert client.download_result_zip.call_count == 4
    assert all(call.args[0] == "remote-original" for call in client.get_result.call_args_list)


from app.repositories.upload_publication_repository import UploadPublicationRepository
from app.models.upload_models import UploadLeaseLostError

async def _prepared(database, tmp_path, *, existing_task=None):
    """在真实仓储中建立已核验批次，返回执行身份与完整发布清单。"""
    repository = database["repository"]
    task = existing_task or await repository.accept(_request(database, project_id="project-a"))
    execution = await repository.begin_attempt(await _claim(database, task["task_id"]))
    manifest = {"paper": {"title": "A paper", "authors": ["First Author"], "abstract": "An abstract",
        "published_date": "2026-01-01", "library_id": "default", "pdf_path": database["source"]["path"]},
        "metadata_missing": [], "batch_id": execution.batch_id, "paper_id": execution.paper_id,
        "user_id": execution.user_id, "payload_sha256": "verified-payload"}
    reference = write_json_artifact(tmp_path, manifest)
    publication = UploadPublicationRepository(repository._pool_provider)
    await publication.verify_batch(execution, {"manifest_ref": reference, "embedding_fingerprint": "current-model",
        "payload_sha256": "verified-payload", "chunk_count": 2, "collection_name": "test_upload_v3"})
    return publication, execution, manifest


@pytest.mark.asyncio
async def test_publish_commits_paper_task_and_project_link_together(upload_database, tmp_path):
    """基础发布同时让论文可用、任务完成并补项目关系，关闭图谱无需调用增强。"""
    publication, execution, manifest = await _prepared(upload_database, tmp_path)
    result = await publication.publish(execution, manifest)
    assert result["status"] == "completed"
    assert result["result"]["searchable"] is True
    async with upload_database["pool"].acquire() as connection:
        paper = await connection.fetchrow("SELECT * FROM papers WHERE paper_id=$1", execution.paper_id)
        assert paper["index_batch_id"] == execution.batch_id
        assert paper["is_vectorized"] is True
        assert await connection.fetchval("SELECT count(*) FROM project_paper_links WHERE paper_id=$1", execution.paper_id) == 1
        assert await connection.fetchval("SELECT count(*) FROM task_runs") == 1


@pytest.mark.asyncio
async def test_stale_worker_cannot_publish_verified_private_batch(upload_database, tmp_path):
    """索引核验后失租，旧执行也不能发布或把任务标为完成。"""
    publication, execution, manifest = await _prepared(upload_database, tmp_path)
    async with upload_database["pool"].acquire() as connection:
        await connection.execute("UPDATE task_runs SET worker_id='replacement' WHERE task_id=$1", execution.task_id)
    with pytest.raises(UploadLeaseLostError):
        await publication.publish(execution, manifest)
    async with upload_database["pool"].acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM papers") == 0
        assert await connection.fetchval("SELECT status FROM task_runs WHERE task_id=$1", execution.task_id) == "running"


@pytest.mark.asyncio
async def test_rebuild_pause_keeps_old_result_and_old_read_lease(upload_database, tmp_path):
    """重建暂停仍保留旧发布；成功切换后旧快照读完才允许清理旧向量。"""
    repository = upload_database["repository"]
    publication, first, manifest = await _prepared(upload_database, tmp_path)
    await publication.publish(first, manifest)
    old_snapshot = await publication.snapshot([first.paper_id], user_id="a")
    rebuild = await repository.request_operation(first.paper_id, user_id="a", operation="rebuild")
    _, rebuilding, _ = await _prepared(upload_database, tmp_path, existing_task=rebuild)
    await repository.pause(rebuilding, UploadFailure("index", "temporary", "索引服务不可用"))
    assert (await publication.snapshot([first.paper_id], user_id="a"))["batch_ids"] == [first.batch_id]
    async with upload_database["pool"].acquire() as connection:
        await connection.execute("DELETE FROM upload_read_leases WHERE lease_id<>$1", old_snapshot["lease_id"])
    await repository.resume(rebuild["task_id"], user_id="a")
    _, replacement, manifest = await _prepared(upload_database, tmp_path, existing_task=rebuild)
    await publication.publish(replacement, manifest)
    new_snapshot = await publication.snapshot([first.paper_id], user_id="a")
    assert new_snapshot["batch_ids"] == [replacement.batch_id]
    await publication.renew_snapshot(old_snapshot)
    retired = await publication.retire_candidates(first.paper_id, user_id="a")
    assert first.batch_id not in {item["batch_id"] for item in retired}
    await publication.release_snapshot(old_snapshot)
    retired = await publication.retire_candidates(first.paper_id, user_id="a")
    assert first.batch_id in {item["batch_id"] for item in retired}
    assert (await publication.content_reference(first.batch_id, user_id="a"))["manifest_ref"]
    await publication.record_cleanup(first.batch_id)
    assert first.batch_id not in {item["batch_id"] for item in await publication.retire_candidates(first.paper_id, user_id="a")}
    assert (await publication.content_reference(first.batch_id, user_id="a"))["manifest_ref"]
    with pytest.raises(LookupError):
        await publication.content_reference(first.batch_id, user_id="b")


@pytest.mark.asyncio
async def test_unverified_or_changed_pointer_cannot_publish(upload_database, tmp_path):
    """同数量或准备好文件均不能替代索引核验；指针变化也必须拒绝。"""
    publication, execution, manifest = await _prepared(upload_database, tmp_path)
    async with upload_database["pool"].acquire() as connection:
        await connection.execute("UPDATE upload_batches SET state='building' WHERE batch_id=$1", execution.batch_id)
    with pytest.raises(UploadConflictError, match="核验"):
        await publication.publish(execution, manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["metadata", "graph"])
async def test_metadata_failure_preserves_basic_ready_state(upload_database, tmp_path, *, operation):
    """元信息或图谱故障独立暂停，基础论文保持原批次可检索。"""
    publication, execution, manifest = await _prepared(upload_database, tmp_path)
    await publication.publish(execution, manifest)
    repository = upload_database["repository"]
    task = await repository.request_operation(execution.paper_id, user_id="a", operation=operation)
    metadata_execution = await repository.begin_attempt(await _claim(upload_database, task["task_id"]))
    await repository.pause(metadata_execution, UploadFailure(operation, "rate_limit", "模型限流"))
    document = await repository.get_document(execution.paper_id, user_id="a")
    assert document["state"] == "ready"
    assert document[f"{operation}_status"] == "paused"
    assert document["index_batch_id"] == execution.batch_id
    assert (await publication.snapshot([execution.paper_id], user_id="a"))["batch_ids"] == [execution.batch_id]


class _IndexPort:
    """模拟向量 IO 的确定故障端口，恢复编排及全部数据库操作仍执行真实代码。"""

    collection_name = "upload_integration"

    def __init__(self):
        """首轮三次网络失败，并保留部分写入用于验证恢复清理。"""
        self.failures_left = 3
        self.rows = {}

    def write_upload_batch(self, batch_id, rows):
        """在私有批次保存载荷，前三次报告明确的网络超时。"""
        import requests
        self.rows[batch_id] = rows
        if self.failures_left:
            self.failures_left -= 1
            raise requests.Timeout("index write timed out")
        return rows

    def delete_upload_batch(self, batch_id):
        """删除已退役私有批次，不清除其他已发布结果。"""
        self.rows.pop(batch_id, None)


@pytest.mark.asyncio
async def test_real_pipeline_resumes_without_reparse_or_reembedding(upload_database, tmp_path, monkeypatch):
    """真实 parser、仓储、handler 和发布主链在索引故障后恢复，源、解析及向量均复用。"""
    from unittest.mock import Mock
    from app.services.ingest import upload_pipeline, document_parse_service
    from app.services import milvus_store
    from app.tasks.upload_task import run_upload_ingest_task

    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "embedding_dimension", 2)
    repository = upload_database["repository"]
    publication = UploadPublicationRepository(repository._pool_provider)
    index = _IndexPort()
    encoder = Mock()
    encoder.embed_texts.side_effect = lambda texts: [[0.25, 0.5] for _ in texts]
    parser = Mock(wraps=document_parse_service.parse_local_document)
    monkeypatch.setattr(document_parse_service, "parse_local_document", parser)
    monkeypatch.setattr(upload_pipeline, "get_upload_repository", lambda: repository)
    monkeypatch.setattr(upload_pipeline, "get_upload_publication_repository", lambda: publication)
    monkeypatch.setattr(milvus_store, "get_milvus_store", lambda: index)
    monkeypatch.setattr(upload_pipeline, "get_upload_embedding_client", lambda: encoder)
    source = write_artifact(tmp_path, ("# Reliability study\n\n## 1 Method\n\n" + "This is the original document. " * 80
        + "\n\n## 2 Results\n\n" + "The experiment confirms the result. " * 60).encode(), suffix=".md")
    database = {**upload_database, "source": source}
    accepted = await repository.accept(replace(_request(database), filename="source.md"))
    first_task = await _claim(database, accepted["task_id"])
    with pytest.raises(UploadPausedError) as failure:
        await run_upload_ingest_task(first_task)
    assert failure.value.failure.kind == "network_timeout", failure.value.failure
    await repository.pause(failure.value.execution, failure.value.failure)
    async with database["pool"].acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM papers") == 0
    await repository.resume(accepted["task_id"], user_id="a")
    resumed_task = await _claim(database, accepted["task_id"])
    prepared = await run_upload_ingest_task(resumed_task)
    completed = await upload_pipeline.commit_upload_result(prepared, task_id=resumed_task["task_id"], worker_id=resumed_task["worker_id"])
    assert completed["status"] == "completed"
    assert completed["result"]["searchable"] is True
    assert parser.call_count == 1
    assert encoder.embed_texts.call_count == 1
    assert set(index.rows) == {completed["result"]["batch_id"]}
    assert source["path"] == (await repository.get_document(accepted["payload"]["paper_id"], user_id="a"))["source_ref"]["path"]


@pytest.mark.asyncio
async def test_index_budget_is_shared_across_private_batches(upload_database):
    """同任务换独立写入批次后仍沿用索引预算，不能自动取得第四次调用。"""
    from unittest.mock import AsyncMock

    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    first = await repository.begin_attempt(await _claim(upload_database, accepted["task_id"]))
    action = AsyncMock(side_effect=MinerUTemporaryError("index unavailable"))
    with pytest.raises(UploadPausedError):
        await UploadStageRunner(repository, first, sleeper=AsyncMock()).run(
            "index_" + first.batch_id, action, inputs="first batch", budget_stage="index")
    assert action.await_count == 3
    replacement = await repository.begin_attempt(await _claim(upload_database, accepted["task_id"]))
    after_restart = AsyncMock(return_value={"indexed": True})
    with pytest.raises(UploadPausedError, match="尝试已用完"):
        await UploadStageRunner(repository, replacement).run(
            "index_" + replacement.batch_id, after_restart, inputs="new batch", budget_stage="index")
    after_restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_exhausted_budget_does_not_change_rejected_remote_creation(upload_database):
    """预算预留失败必须回滚整个开始事务，明确拒绝不能被改成结果未知。"""
    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    execution = await repository.begin_attempt(await _claim(upload_database, accepted["task_id"]))
    checkpoint = await repository.start_checkpoint(execution, stage="mineru_slot",
        input_hash="source-config", repeatable=False, attempt_limit=1)
    await repository.reject_checkpoint(execution, checkpoint, UploadFailure("mineru_slot", "credentials", "token rejected"))
    with pytest.raises(UploadPausedError, match="尝试已用完"):
        await repository.start_checkpoint(execution, stage="mineru_slot",
            input_hash="source-config", repeatable=False, attempt_limit=1)
    saved = await repository.checkpoint(execution, "mineru_slot")
    assert saved["state"] == "rejected"
    assert saved["operation_id"] == checkpoint["operation_id"]


@pytest.mark.asyncio
async def test_cached_completed_stage_does_not_spend_another_attempt(upload_database):
    """调用预算用满之后仍可读取已完成检查点，不重发已成功的请求。"""
    repository = upload_database["repository"]
    accepted = await repository.accept(_request(upload_database))
    execution = await repository.begin_attempt(await _claim(upload_database, accepted["task_id"]))
    checkpoint = await repository.start_checkpoint(execution, stage="parse",
        input_hash="source-config", repeatable=True, attempt_limit=1)
    await repository.complete_checkpoint(execution, checkpoint, {"original": "document"})
    replay = await repository.start_checkpoint(execution, stage="parse",
        input_hash="source-config", repeatable=True, attempt_limit=1)
    assert replay["state"] == "completed"
    assert replay["payload"] == {"original": "document"}
    with pytest.raises(UploadPausedError, match="尝试已用完"):
        await repository.reserve_attempt(execution, stage="parse", limit=1)


@pytest.mark.asyncio
async def test_metadata_resume_preserves_sibling_graph_budget(upload_database, tmp_path):
    """元信息单独继续时不重置同时存在的图谱预算，两个独立任务不会互相续命。"""
    repository = upload_database["repository"]
    accepted = await repository.accept(replace(_request(upload_database), graph_requested=True))
    publication, original, manifest = await _prepared(upload_database, tmp_path, existing_task=accepted)
    await publication.publish(original, {**manifest, "metadata_missing": ["abstract"]})
    document = await repository.get_document(original.paper_id, user_id="a")
    graph = await repository.begin_attempt(await _claim(upload_database, document["graph_task_id"]))
    metadata = await repository.begin_attempt(await _claim(upload_database, document["metadata_task_id"]))
    await repository.reserve_attempt(graph, stage="graph", limit=1)
    await repository.pause(metadata, UploadFailure("metadata", "credentials", "token rejected"))
    await repository.resume(metadata.task_id, user_id="a")
    with pytest.raises(UploadPausedError, match="尝试已用完"):
        await repository.reserve_attempt(graph, stage="graph", limit=1)


@pytest.mark.asyncio
async def test_new_metadata_task_preserves_other_paused_task_budget(upload_database, tmp_path):
    """新建元信息任务不会清掉暂停图谱的既有尝试记录。"""
    repository = upload_database["repository"]
    publication, original, manifest = await _prepared(upload_database, tmp_path)
    await publication.publish(original, manifest)
    task = await repository.request_operation(original.paper_id, user_id="a", operation="graph")
    graph = await repository.begin_attempt(await _claim(upload_database, task["task_id"]))
    await repository.reserve_attempt(graph, stage="graph", limit=1)
    await repository.pause(graph, UploadFailure("graph", "temporary", "graph unavailable"))
    before = await repository.get_document(original.paper_id, user_id="a")
    await repository.request_operation(original.paper_id, user_id="a", operation="metadata")
    after = await repository.get_document(original.paper_id, user_id="a")
    assert after["retry_counts"] == before["retry_counts"]


@pytest.mark.asyncio
async def test_rebuild_invalidates_previous_enrichment_task_heads(upload_database, tmp_path):
    """重建建立新原文代次后，旧暂停增强任务不能借新代次恢复。"""
    repository = upload_database["repository"]
    publication, original, manifest = await _prepared(upload_database, tmp_path)
    await publication.publish(original, manifest)
    task = await repository.request_operation(original.paper_id, user_id="a", operation="metadata")
    metadata = await repository.begin_attempt(await _claim(upload_database, task["task_id"]))
    await repository.pause(metadata, UploadFailure("metadata", "temporary", "model unavailable"))
    await repository.request_operation(original.paper_id, user_id="a", operation="rebuild")
    with pytest.raises(UploadConflictError, match="后续任务"):
        await repository.resume(metadata.task_id, user_id="a")


@pytest.mark.asyncio
async def test_deletion_revokes_snapshots_and_tombstones_upload(upload_database, tmp_path, monkeypatch):
    """删除核对 owner，清除来源和检查点后旧引用、快照及继续操作均被拒绝。"""
    from app.repositories import paper_deletion_repository
    from app.models.research_contracts import RetrievalScopeError

    repository = upload_database["repository"]
    publication, execution, manifest = await _prepared(upload_database, tmp_path)
    checkpoint = await repository.start_checkpoint(execution, stage="parse", input_hash="source", repeatable=True)
    await repository.complete_checkpoint(execution, checkpoint, {"document": "saved"})
    await publication.publish(execution, manifest)
    snapshot = await publication.snapshot([execution.paper_id], user_id="a")
    monkeypatch.setattr(paper_deletion_repository, "get_pool", repository._pool_provider)
    deletion = paper_deletion_repository.PaperDeletionRepository()
    async with upload_database["pool"].acquire() as connection:
        await connection.execute("UPDATE papers SET deletion_status='deleting',deletion_task_id='delete-task' WHERE paper_id=$1", execution.paper_id)
    assert not await deletion.hard_delete_paper(execution.paper_id, user_id="b", task_id="delete-task")
    assert await deletion.hard_delete_paper(execution.paper_id, user_id="a", task_id="delete-task")
    document = await repository.get_document(execution.paper_id, user_id="a")
    assert document["state"] == "deleted"
    assert document["source_ref"] == {}
    task = await repository._tasks.get_task_run(execution.task_id, user_id="a", connection=upload_database["pool"])
    assert not (await repository.describe_task(task))["upload"]["searchable"]
    async with upload_database["pool"].acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM upload_checkpoints") == 0
        assert await connection.fetchval("SELECT manifest_ref FROM upload_batches WHERE batch_id=$1", execution.batch_id) is None
    with pytest.raises(LookupError):
        await publication.content_reference(execution.batch_id, user_id="a")
    with pytest.raises(RetrievalScopeError):
        await publication.renew_snapshot(snapshot)


@pytest.mark.asyncio
async def test_deletion_waits_for_active_upload_and_cancels_paused_sibling(upload_database, tmp_path, monkeypatch):
    """删除先请求取消活跃上传并等待，已暂停的增强任务立即取消且不重新排队。"""
    from app.repositories import paper_deletion_repository

    repository = upload_database["repository"]
    accepted = await repository.accept(replace(_request(upload_database), graph_requested=True))
    publication, original, manifest = await _prepared(upload_database, tmp_path, existing_task=accepted)
    await publication.publish(original, {**manifest, "metadata_missing": ["abstract"]})
    document = await repository.get_document(original.paper_id, user_id="a")
    metadata = await repository.begin_attempt(await _claim(upload_database, document["metadata_task_id"]))
    await repository.pause(metadata, UploadFailure("metadata", "credentials", "model rejected"))
    graph = await repository.begin_attempt(await _claim(upload_database, document["graph_task_id"]))
    monkeypatch.setattr(paper_deletion_repository, "get_pool", repository._pool_provider)
    deletion = paper_deletion_repository.PaperDeletionRepository()
    assert not await deletion.stop_uploads(original.paper_id, user_id="a")
    async with upload_database["pool"].acquire() as connection:
        assert await connection.fetchval("SELECT status FROM task_runs WHERE task_id=$1", metadata.task_id) == "cancelled"
        assert await connection.fetchval("SELECT cancel_requested FROM task_runs WHERE task_id=$1", graph.task_id)
        await connection.execute("UPDATE task_runs SET status='cancelled',lease_until=NULL WHERE task_id=$1", graph.task_id)
    assert await deletion.stop_uploads(original.paper_id, user_id="a")


@pytest.mark.asyncio
async def test_terminal_write_failure_rolls_back_entire_publication(upload_database, tmp_path, monkeypatch):
    """任务终态提交失败时，论文和批次切换必须一起回滚；修复后沿原执行可再次发布。"""
    from unittest.mock import AsyncMock

    publication, execution, manifest = await _prepared(upload_database, tmp_path)
    finish = publication._finish_task
    monkeypatch.setattr(publication, "_finish_task", AsyncMock(side_effect=RuntimeError("terminal write failed")))
    with pytest.raises(RuntimeError, match="terminal write failed"):
        await publication.publish(execution, manifest)
    async with upload_database["pool"].acquire() as connection:
        assert await connection.fetchval("SELECT count(*) FROM papers") == 0
        assert await connection.fetchval("SELECT state FROM upload_batches WHERE batch_id=$1", execution.batch_id) == "verified"
        assert await connection.fetchval("SELECT status FROM task_runs WHERE task_id=$1", execution.task_id) == "running"
    monkeypatch.setattr(publication, "_finish_task", finish)
    assert (await publication.publish(execution, manifest))["status"] == "completed"
