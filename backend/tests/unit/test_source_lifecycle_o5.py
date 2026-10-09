"""O5 来源正式入口与读删占用的离线回归，作者：xxx。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.research_contracts import RetrievalScopeError
from app.models.upload_models import UploadConflictError, UploadExecution, UploadLeaseLostError
from app.repositories.source_access_repository import SourceAccessRepository, source_lock_key
from app.repositories.upload_publication_repository import UploadPublicationRepository
from app.services import source_io_service, upload_read_service as reads
from app.services.ingest import upload_pipeline
from app.services.ingest.upload_artifacts import write_artifact, write_json_artifact
from app.services.ingest.upload_content_identity import compare_source_versions, content_identity

BATCH_A = "a" * 32
BATCH_B = "b" * 32


class ReadDatabase:
    """实现被测授权与租约 SQL 谓词的内存端口，不代表真实 PostgreSQL 并发验收。"""

    def __init__(self, manifest_ref):
        """参数为临时清单；初始化固定 A、当前 B 和仍有效的原读取租约。"""
        self.owner, self.active, self.leased = "owner", True, True
        self.renewals, self.now, self.expires = 0, 0, 900
        self.state = "superseded"
        self.manifest_ref = manifest_ref

    async def pool(self):
        """无参数；返回内存连接池。"""
        return self

    def acquire(self):
        """无参数；返回连接上下文。"""
        return self

    def transaction(self):
        """无参数；返回隔离事务端口。"""
        return self

    async def __aenter__(self):
        """无参数；进入内存上下文。"""
        return self

    async def __aexit__(self, *_args):
        """参数为退出信息；没有真实数据库资源。"""

    async def fetch(self, sql, *args):
        """参数为生产 SQL；只实现续期和退役条件。"""
        if "FOR SHARE OF p,b" in sql:
            allowed = self.owner == args[1] and self.active and self.state != "retired"
            return [{"batch_id": BATCH_A, "paper_id": "paper"}] if allowed else []
        if "cleanup_completed_at IS NULL" in sql:
            return [] if self.leased and self.expires > self.now else [self.batch()]
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        """参数为生产续租语句；仅原租约可以继续使用。"""
        assert "UPDATE upload_read_leases" in sql
        self.renewals += 1
        if not self.leased:
            return None
        self.expires = self.now + args[3]
        return args[0]

    async def fetchrow(self, sql, *args):
        """参数为历史正文查验；删除或换 owner 后拒绝原引用。"""
        assert "published_at IS NOT NULL" in sql
        return self.batch() if self.owner == args[1] and self.active else None

    async def execute(self, sql, *args):
        """参数为生产释放和退役语句；维护虚拟租约与退役状态。"""
        if "SET state='retired'" in sql and args[0]:
            self.state = "retired"
        if "DELETE FROM upload_read_leases" in sql and ("lease_id=" in sql or self.expires <= self.now):
            self.leased = False

    def batch(self):
        """无参数；返回原版本 A 的真实临时清单引用。"""
        return {"batch_id": BATCH_A, "paper_id": "paper", "user_id": "owner", "manifest_ref": self.manifest_ref,
            "published_at": 1, "state": self.state, "collection_name": "isolated"}


class FileGate:
    """模拟共享/排他端口的时序，只用于验证生产调用持有到实际工作退出。"""

    def __init__(self, connection):
        """参数为授权连接；初始化可观察读者计数。"""
        self.connection, self.readers, self.exclusive = connection, 0, False
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def file_access(self, paper_ids, *, exclusive=False):
        """参数为目标及占用方式；返回连接并记录真实范围持有时序。"""
        async with self.condition:
            await self.condition.wait_for(lambda: not self.exclusive and (not exclusive or self.readers == 0))
            self.exclusive = exclusive
            self.readers += int(not exclusive)
        try:
            yield self.connection
        finally:
            async with self.condition:
                self.exclusive = False
                self.readers -= int(not exclusive)
                self.condition.notify_all()


@pytest.fixture
def source(monkeypatch, tmp_path):
    """参数为替身和临时目录；安装真实读取服务使用的独占文件与隔离仓储。"""
    chunk = {"chunk_id": f"v3:{BATCH_A}:0", "parent_id": "parent", "paper_id": "paper", "batch_id": BATCH_A,
        "text": "原版本 A", "full_text": "原版本 A"}
    manifest = {"format": 3, "paper_id": "paper", "user_id": "owner", "batch_id": BATCH_A, "chunks": [chunk]}
    database = ReadDatabase(write_json_artifact(tmp_path, manifest))
    repository = UploadPublicationRepository(database.pool)
    gate = FileGate(database)
    snapshot = {"user_id": "owner", "lease_id": "lease", "batch_ids": [BATCH_A], "paper_batches": {"paper": BATCH_A},
        "collection_names": ["isolated"], "embedding_fingerprints": ["model"]}
    monkeypatch.setattr(reads, "get_upload_publication_repository", lambda: repository)
    monkeypatch.setattr(reads, "get_source_access_repository", lambda: gate)
    monkeypatch.setattr(reads, "get_settings", lambda: SimpleNamespace(data_dir=str(tmp_path)))
    monkeypatch.setattr(reads, "embedding_identity", lambda: "model")
    monkeypatch.setattr(reads.paper_db, "_run", asyncio.run)
    monkeypatch.setattr(upload_pipeline, "cleanup_retired_batches", AsyncMock())
    return SimpleNamespace(database=database, repository=repository, gate=gate, snapshot=snapshot,
        scope={"user_id": "owner", "paper_ids": ["paper"], "publication": snapshot, "filter_expr": 'paper_id == "paper"'}, root=tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("revocation", ["owner", "deletion", "lease", "retirement"])
async def test_locator_adapters_reject_current_revocation_before_query(source, monkeypatch, revocation):
    """目录和筛块正式适配器对当前撤权/删除/失租显式失败，不能返回假空或旧定位。"""
    from app.nodes import rag_node
    from app.services.milvus_store import MilvusStore

    if revocation == "owner":
        source.database.owner = "another"
    elif revocation == "deletion":
        source.database.active = False
    elif revocation == "lease":
        source.database.leased = False
    else:
        source.database.state = "retired"
    store = MilvusStore(collection_name="isolated")
    client = Mock()
    monkeypatch.setattr(store, "ensure_collection", Mock())
    monkeypatch.setattr(store, "_get_client", lambda: client)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    unit = SimpleNamespace(evidence_policy=SimpleNamespace(target_paper_ids=["paper"], required_sections=[],
        required_block_types=[], required_pages=[]))
    for adapter in (rag_node._planner_section_structure_records, rag_node._planner_chunk_metadata_records):
        with pytest.raises(RetrievalScopeError):
            await adapter({"project_scope": source.scope}, unit, None)
    client.query.assert_not_called()
    client.query_iterator.assert_not_called()


@pytest.mark.asyncio
async def test_current_project_revoke_cannot_be_hidden_by_old_snapshot(source, monkeypatch):
    """项目默认范围移除论文后，固定旧版本不继续授予访问权。"""
    from app.services import project_scope_service

    snapshot = {**source.snapshot, "access_scope": {"project_id": "project", "requested_paper_ids": [], "paper_ids": ["paper"]}}
    resolve = AsyncMock(return_value={"paper_ids": []})
    monkeypatch.setattr(project_scope_service, "_build_project_context_scope", resolve)
    with pytest.raises(RetrievalScopeError, match="项目范围"):
        await reads.authorize_publication_read(snapshot, user_id="owner")
    resolve.assert_awaited_once_with("project", user_id="owner", requested_paper_ids=[])
    assert source.database.renewals == 0


@pytest.mark.asyncio
async def test_section_cache_identity_rechecks_current_project_before_reuse(source, monkeypatch):
    """参数为隔离来源与替身；章节正文复用前项目撤权必须失败，不能只凭旧租约认可缓存。"""
    from app.nodes import rag_node
    from app.repositories import upload_publication_repository
    from app.services import project_scope_service

    monkeypatch.setattr(upload_publication_repository, "get_upload_publication_repository", lambda: source.repository)
    source.snapshot["access_scope"] = {"project_id": "project", "requested_paper_ids": [], "paper_ids": ["paper"]}
    resolve = AsyncMock(return_value={"paper_ids": ["paper"]})
    monkeypatch.setattr(project_scope_service, "_build_project_context_scope", resolve)
    unit = SimpleNamespace(evidence_policy=SimpleNamespace(
        source_type=rag_node.layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT,
        target_paper_ids=["paper"], required_sections=["方法"], continuation_ids=[]))
    state = {"project_scope": source.scope}
    identity = await rag_node._planner_read_request(state, unit, None)
    assert identity["section_paths"] == ["方法"]
    resolve.return_value = {"paper_ids": []}
    with pytest.raises(RetrievalScopeError, match="项目范围"):
        await rag_node._planner_read_request(state, unit, None)


@pytest.mark.asyncio
async def test_batch_read_rechecks_project_after_acquiring_file_access(source, monkeypatch):
    """参数为隔离来源与替身；等待文件占用期间移出项目，实际读文件前必须拒绝。"""
    from app.services import project_scope_service

    source.snapshot["access_scope"] = {"project_id": "project", "requested_paper_ids": [], "paper_ids": ["paper"]}
    resolve = AsyncMock(return_value={"paper_ids": ["paper"]})
    monkeypatch.setattr(project_scope_service, "_build_project_context_scope", resolve)
    original_access = source.gate.file_access

    @asynccontextmanager
    async def revoke_while_waiting(paper_ids):
        """参数为待读论文；模拟占用等待后项目权限已撤回，返回原隔离连接。"""
        async with original_access(paper_ids) as connection:
            resolve.return_value = {"paper_ids": []}
            yield connection

    monkeypatch.setattr(source.gate, "file_access", revoke_while_waiting)
    loader = Mock(side_effect=AssertionError("撤权后不得读取文件"))
    monkeypatch.setattr(reads, "_load_manifest", loader)
    with pytest.raises(RetrievalScopeError, match="项目范围"):
        await reads._read_published_batches((BATCH_A,), user_id="owner", publication=source.snapshot)
    loader.assert_not_called()


@pytest.mark.asyncio
async def test_request_heartbeat_keeps_A_until_release(source, monkeypatch):
    """模型等待期间持续维护原 A 租约，超过原 TTL 后真实退役入口仍拒绝清理。"""
    monkeypatch.setattr(reads, "READ_HEARTBEAT_SECONDS", 0.001)
    async with reads.publication_scope_lifetime(source.scope):
        source.database.now = 899
        while source.database.renewals < 2:
            await asyncio.sleep(0.001)
        source.database.now = 901
        assert await source.repository.retire_candidates("paper", user_id="owner") == []
        original = await reads.read_published_chunk_async(f"v3:{BATCH_A}:0", user_id="owner", publication=source.snapshot)
        assert original["text"] == "原版本 A"
    assert not source.database.leased
    assert source.database.state == "superseded"


@pytest.mark.asyncio
async def test_heartbeat_failure_keeps_cause_and_drains_started_file(source, monkeypatch):
    """续期失败取消执行，但已启动线程退出前不释放租约，最终保留来源错误。"""
    monkeypatch.setattr(reads, "READ_HEARTBEAT_SECONDS", 0.001)
    entered, finish = threading.Event(), threading.Event()
    original_loader = reads._load_manifest

    def slow_read(record):
        """参数为清单；等待测试准许真实文件读取结束。"""
        entered.set()
        finish.wait()
        return original_loader(record)

    async def operation():
        """无参数；通过真实占用和读取路径运行一次研究文件工作。"""
        async with reads.publication_scope_lifetime(source.scope):
            return await reads.read_published_chunk_async(f"v3:{BATCH_A}:0", user_id="owner")

    monkeypatch.setattr(reads, "_load_manifest", slow_read)
    task = asyncio.create_task(operation())
    await asyncio.to_thread(entered.wait)
    source.database.owner = "revoked"
    await asyncio.sleep(0.01)
    assert not task.done() and source.database.leased and source.gate.readers == 1
    task.cancel()
    finish.set()
    with pytest.raises(RetrievalScopeError, match="快照已失效"):
        await task
    assert source.gate.readers == 0 and not source.database.leased


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("body failed"), asyncio.CancelledError()])
async def test_lifetime_all_exits_release(source, failure):
    """正常、异常、取消三条退出路径都释放同一个原快照。"""
    async def operation():
        """无参数；在真实生命周期内触发指定出口。"""
        async with reads.publication_scope_lifetime(source.scope):
            if failure is not None:
                raise failure

    if failure is None:
        await operation()
    else:
        with pytest.raises(type(failure)):
            await operation()
    assert not source.database.leased


@pytest.mark.asyncio
async def test_download_body_consumption_blocks_delete_files(source, monkeypatch):
    """正式下载响应消费期间，独立删除入口等待排他占用后才删除临时文件。"""
    from app.api.routes import task_api
    from app.tasks import paper_delete_task
    from app.services.paper_file_cleanup_service import delete_managed_paper_files

    reference = write_artifact(source.root / "uploads" / "paper", b"original", suffix=".pdf")
    source.gate.source_reference = AsyncMock(return_value={"filename": "原文.pdf", "source_ref": reference})
    monkeypatch.setattr(paper_delete_task, "get_source_access_repository", lambda: source.gate)
    def delete_files(paper_id, *, pdf_path):
        """参数为删除目标；确认排他占用后只删除本测试的受管临时文件。"""
        assert source.gate.exclusive
        return delete_managed_paper_files(paper_id, pdf_path=pdf_path, data_dir=source.root)

    delete = Mock(side_effect=delete_files)
    monkeypatch.setattr(paper_delete_task, "delete_managed_paper_files", delete)
    started, consumed = asyncio.Event(), asyncio.Event()

    async def send(message):
        """参数为真实 ASGI 消息；阻塞消费正文以观察下载占用。"""
        if message["type"] == "http.response.body":
            assert message["body"] == b"original"
            started.set()
            await consumed.wait()

    response = await task_api.download_upload_source("paper", {"user_id": "owner"})
    download = asyncio.create_task(response({}, AsyncMock(), send))
    await started.wait()
    context = paper_delete_task.PaperDeletionContext("task", "paper", "owner", "library", reference["path"])
    deletion = asyncio.create_task(paper_delete_task._delete_files(context))
    await asyncio.sleep(0)
    assert source.gate.readers == 1 and not deletion.done()
    paper_delete_task.delete_managed_paper_files.assert_not_called()
    consumed.set()
    await download
    await deletion
    paper_delete_task.delete_managed_paper_files.assert_called_once()
    assert source.gate.readers == 0 and not source.gate.exclusive
    assert not Path(reference["path"]).exists()


@pytest.mark.asyncio
async def test_retired_original_stays_readable_but_deleted_and_wrong_paper_fail(source):
    """向量退役保留历史原文，删除/错误论文关联仍在读取前拒绝。"""
    source.database.state = "retired"
    row = await reads.read_published_chunk_async(f"v3:{BATCH_A}:0", user_id="owner", paper_id="paper")
    assert row["text"] == "原版本 A"
    with pytest.raises(LookupError, match="不属于"):
        await reads.read_published_chunk_async(f"v3:{BATCH_A}:0", user_id="owner", paper_id="other")
    source.database.active = False
    with pytest.raises(LookupError):
        await reads.read_published_chunk_async(f"v3:{BATCH_A}:0", user_id="owner")


@pytest.mark.asyncio
async def test_shared_and_exclusive_use_same_stable_sql_lock_order():
    """生产文件占用端口按固定论文顺序、相同锁键申请共享和排他锁。"""
    connection = ReadDatabase({})
    connection.execute = AsyncMock()
    repository = SourceAccessRepository(connection.pool)
    for exclusive in (False, True):
        async with repository.file_access(("z", "a", "a"), exclusive=exclusive) as acquired:
            assert acquired is connection
    calls = connection.execute.await_args_list
    assert [call.args[1] for call in calls] == [source_lock_key("a"), source_lock_key("z")] * 2
    assert "lock_shared" in calls[0].args[0] and "lock_shared" not in calls[2].args[0]


@pytest.mark.asyncio
async def test_upload_file_work_rechecks_writer_under_lock(source, monkeypatch):
    """已取消或删除的 writer 在文件锁内拒绝，不能重新生成已删除目录。"""
    from app.repositories import source_access_repository, upload_repository

    monkeypatch.setattr(source_access_repository, "get_source_access_repository", lambda: source.gate)
    check = AsyncMock(side_effect=UploadLeaseLostError("writer stopped"))
    monkeypatch.setattr(upload_repository, "require_upload_execution", check)
    function = Mock()
    execution = UploadExecution("task", "worker", "paper", "owner", "parse", BATCH_B, BATCH_A)
    with pytest.raises(UploadLeaseLostError):
        await source_io_service.run_upload_file_io(execution, function)
    check.assert_awaited_once_with(source.database, execution)
    function.assert_not_called()
    assert source.gate.readers == 0


@pytest.mark.parametrize("case,expected", [("embedding", "unchanged"), ("metadata", "unchanged"),
    ("body", "changed"), ("source", "changed"), ("legacy", "unknown")])
def test_version_comparison_uses_source_and_actual_body(case, expected):
    """批次或 embedding 变化不冒充正文变化，metadata 不污染正文指纹，旧未知保持未知。"""
    execution = UploadExecution("task", "worker", "paper", "owner", "parse", BATCH_B, BATCH_A)
    source, document = {"sha256": "source"}, {"sha256": "document"}
    body = {"full_text": "原正文", "metadata": {"title": "旧标题"}}
    old = {"source_ref": source, "document_ref": document,
        "content_identity": content_identity(body, source_ref=source, document_ref=document)}
    new = deepcopy(old)
    if case in ("metadata", "legacy", "body"):
        new["document_ref"] = {"sha256": "new-document"}
        current = {**body, "metadata": {"title": "新标题"}, "full_text": "变化正文" if case == "body" else "原正文"}
        new["content_identity"] = content_identity(current, source_ref=source, document_ref=new["document_ref"])
    if case == "source":
        new["source_ref"] = {"sha256": "new-source"}
    if case == "legacy":
        old.pop("content_identity")
    change = compare_source_versions(execution, manifest=new, manifest_sha256="new-manifest",
        previous=old, previous_manifest_sha256="old-manifest")
    assert change.classification == expected and change.previous_batch_id == BATCH_A


@pytest.mark.asyncio
async def test_publication_rejects_stale_comparison_and_uses_same_connection():
    """版本比较绑定两侧清单和旧指针，记忆状态变更接收同一发布连接。"""
    execution = UploadExecution("task", "worker", "paper", "owner", "parse", BATCH_B, BATCH_A)
    manifest = {"source_ref": {"sha256": "source"}, "document_ref": {"sha256": "document"}}
    change = compare_source_versions(execution, manifest=manifest, manifest_sha256="new",
        previous=manifest, previous_manifest_sha256="old")
    connection = SimpleNamespace(fetchrow=AsyncMock(return_value={"manifest_ref": {"sha256": "old"}}))
    handler = AsyncMock()
    repository = UploadPublicationRepository(source_change_handler=handler)
    batch = {"manifest_ref": {"sha256": "new"}}
    await repository._apply_source_change(connection, execution, batch, change)
    handler.assert_awaited_once_with(connection, change)
    with pytest.raises(UploadConflictError):
        await repository._apply_source_change(connection, execution, batch, replace(change, previous_batch_id="stale"))
    with pytest.raises(UploadConflictError):
        await repository._apply_source_change(connection, execution, batch, replace(change, previous_manifest_sha256="stale"))
    assert handler.await_count == 1


class PublicationDatabase(ReadDatabase):
    """正式发布入口的连接端口，记录指针与事务顺序而不连接数据库。"""

    def __init__(self):
        """无参数；初始化实际旧指针和新批次核验证据。"""
        super().__init__({})
        self.previous, self.calls = BATCH_A, []

    async def fetchrow(self, sql, *args):
        """参数为发布SQL；返回本测试的上传、论文及前后批次。"""
        if "FROM upload_documents" in sql:
            return {"paper_id": "paper", "graph_requested": False}
        if "FROM papers" in sql:
            return {"index_batch_id": self.previous, "user_id": "owner", "deletion_status": "active"}
        if "FROM upload_batches" in sql:
            return {"state": "verified", "worker_id": "worker", "payload_sha256": "payload", "chunk_count": 1,
                "manifest_ref": {"sha256": "new" if args[0] == BATCH_B else "old"}}
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        """参数为论文提交；记录实际进入当前指针更新的时点。"""
        assert "INSERT INTO papers" in sql
        self.calls.append("pointer")
        return "paper"

    async def execute(self, sql, *args):
        """参数为批次和状态更新；记录执行顺序，不操作真实数据。"""
        self.calls.append(sql.split()[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
async def test_publish_checks_previous_pointer_before_memory_and_switch(conflict, monkeypatch):
    """正式发布事务先核对旧指针，再在同连接维护记忆和切指针；过期比较不能写入。"""
    from app.repositories import upload_publication_repository as publications

    database = PublicationDatabase()
    database.previous = "newer" if conflict else BATCH_A
    execution = UploadExecution("task", "worker", "paper", "owner", "parse", BATCH_B, BATCH_A)
    manifest = {"batch_id": BATCH_B, "user_id": "owner", "paper_id": "paper", "payload_sha256": "payload",
        "metadata_missing": [], "source_ref": {"sha256": "source"}, "document_ref": {"sha256": "document"},
        "paper": {"title": "title", "library_id": "library", "authors": [], "abstract": "abstract", "pdf_path": "", "published_date": ""}}
    change = compare_source_versions(execution, manifest=manifest, manifest_sha256="new",
        previous=manifest, previous_manifest_sha256="old")

    async def memory_change(connection, received):
        """参数为同连接和可靠分类；记录在指针更新前完成失效维护。"""
        assert connection is database and received == change
        database.calls.append("memory")

    repository = UploadPublicationRepository(database.pool, source_change_handler=memory_change)
    monkeypatch.setattr(publications, "require_upload_execution", AsyncMock())
    monkeypatch.setattr(repository, "_finish_task", AsyncMock(return_value={"status": "completed"}))
    if conflict:
        with pytest.raises(UploadConflictError):
            await repository.publish(execution, manifest, source_change=change)
        assert database.calls == []
    else:
        result = await repository.publish(execution, manifest, source_change=change)
        assert result["status"] == "completed"
        assert database.calls[:2] == ["memory", "pointer"]


@pytest.mark.asyncio
async def test_commit_verifies_artifacts_before_transaction_and_binds_comparison(source, monkeypatch):
    """正式完成提交在事务前核验临时原文和正文，用明确旧新清单身份调用发布仓储。"""
    from dataclasses import asdict

    execution = UploadExecution("task", "worker", "paper", "owner", "parse", BATCH_B, BATCH_A)
    original = write_artifact(source.root, b"original", suffix=".md")
    document = {"full_text": "正文", "metadata": {}}
    document_ref = write_json_artifact(source.root, document)
    manifest = {"source_ref": original, "document_ref": document_ref,
        "content_identity": content_identity(document, source_ref=original, document_ref=document_ref)}
    reference = write_json_artifact(source.root, manifest)
    publication = SimpleNamespace(content_reference=AsyncMock(return_value={"manifest_ref": reference}), publish=AsyncMock(return_value={"status": "completed"}))
    monkeypatch.setattr(upload_pipeline, "get_upload_publication_repository", lambda: publication)
    monkeypatch.setattr(upload_pipeline, "get_settings", lambda: SimpleNamespace(data_dir=str(source.root)))

    async def local_files(_execution, function, *args, **kwargs):
        """参数为执行及文件工作；仅隔离writer授权，真实文件读取与SHA核验仍执行。"""
        return await source_io_service.run_source_io(function, *args, **kwargs)

    monkeypatch.setattr(upload_pipeline, "run_upload_file_io", local_files)
    prepared = {"_upload_publication": {"execution": asdict(execution), "manifest_ref": reference}}
    assert (await upload_pipeline.commit_upload_result(prepared, task_id="task", worker_id="worker"))["status"] == "completed"
    change = publication.publish.await_args.kwargs["source_change"]
    assert change.classification == "unchanged" and change.previous_batch_id == BATCH_A
    assert change.manifest_sha256 == reference["sha256"] == change.previous_manifest_sha256



@pytest.mark.asyncio
@pytest.mark.parametrize("source_aware", [False, True], ids=["legacy", "source-schema-1"])
async def test_parser_processing_upgrade_keeps_old_published_reads(source, monkeypatch, source_aware):
    """参数为旧发布端口、替身和来源格式；验证两种旧清单仍按原字节读取，处理版本不参与，无返回值。"""
    from app.services.ingest import upload_parse_service
    from app.services.ingest.upload_artifacts import read_json_artifact

    assert upload_parse_service.PARSER_FORMAT_VERSION == 4
    reference = source.database.manifest_ref
    manifest = read_json_artifact(reference, root=source.root)
    if source_aware:
        # 1. 【来源测试】【旧来源格式】独立合成 schema 1 的已发布清单，不通过新版 parser 生成
        chunk = manifest["chunks"][0]
        body = chunk["text"]
        original = write_artifact(source.root / "original", body.encode("utf-8"), suffix=".md")
        item = {"source_item": "src-v1:markdown:0", "text": body, "source_kind": "markdown",
            "source_index": 0, "block_type": "paragraph", "page_num": None, "artifact_ref": original}
        fragment = {"source_item": item["source_item"], "source_start": 0, "source_end": len(body),
            "text_start": 0, "text_end": len(body), "page_num": None}
        manifest = {**manifest, "content_schema_version": 1, "source_items": {item["source_item"]: item},
            "chunks": [{**chunk, "display_text": body,
                "source_fragments": [fragment], "parent_source_fragments": [fragment]}]}
        reference = write_json_artifact(source.root / "source-aware", manifest)
        source.database.manifest_ref = reference
    before_files = {path: path.read_bytes() for path in source.root.rglob("*") if path.is_file()}
    parser_entry = AsyncMock(side_effect=AssertionError("读取旧发布不得重新解析"))
    parameters = Mock(side_effect=AssertionError("读取旧发布不得计算处理版本"))
    monkeypatch.setattr(upload_parse_service.UploadDocumentParser, "parse", parser_entry)
    monkeypatch.setattr(upload_parse_service.UploadDocumentParser, "_parameters", parameters)

    # 2. 【来源测试】【旧批次读取】沿真实鉴权、文件读取和清单核验返回既有内容
    chunk = await reads.read_published_chunk_async(f"v3:{BATCH_A}:0", user_id="owner", publication=source.snapshot)
    batches = await reads._read_published_batches((BATCH_A,), user_id="owner", publication=source.snapshot)
    assert chunk == manifest["chunks"][0]
    assert batches[BATCH_A] == manifest and batches[BATCH_A]["format"] == 3
    assert source.database.manifest_ref == reference
    assert source.snapshot["paper_batches"] == {"paper": BATCH_A}
    assert source.database.state == "superseded"
    parser_entry.assert_not_awaited()
    parameters.assert_not_called()
    assert {path: path.read_bytes() for path in source.root.rglob("*") if path.is_file()} == before_files
