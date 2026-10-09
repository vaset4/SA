"""上传批次发布、授权读取占用和退役记录，作者：xxx。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
import uuid

from app.models.research_contracts import RetrievalScopeError, ScopeOwnershipError
from app.models.source_access_models import SourceVersionChange
from app.models.upload_models import UploadConflictError, UploadExecution, UploadLeaseLostError
from app.repositories.common import dump_json
from app.repositories.task_repository import TaskRepository, _task_run_data
from app.repositories.upload_repository import UploadRepository, require_upload_execution, upload_row
from app.services.database import get_pool

READ_LEASE_SECONDS = 900


class UploadPublicationRepository:
    """由同一事务决定论文当前版本和基础任务完成状态。"""

    def __init__(self, pool_provider: Callable[[], Awaitable[Any]] = get_pool, *,
                 source_change_handler: Callable[[Any, SourceVersionChange], Awaitable[None]] | None = None) -> None:
        """接收数据库连接池端口，隔离测试与运行环境。"""
        self._pool_provider = pool_provider
        self._tasks = TaskRepository()
        self._uploads = UploadRepository(pool_provider, tasks=self._tasks)
        self._source_change_handler = source_change_handler

    async def verify_batch(self, execution: UploadExecution, evidence: dict[str, Any]) -> None:
        """【上传】【索引核验】只有持租执行可以登记完整载荷核验的证据。"""
        await self._save_batch_evidence(execution, evidence, verified=True)

    async def stage_batch(self, execution: UploadExecution, evidence: dict[str, Any]) -> None:
        """写索引之前保存批次集合及清单，即使部分写入失败也能定位并清理。"""
        await self._save_batch_evidence(execution, evidence, verified=False)

    async def _save_batch_evidence(self, execution: UploadExecution, evidence: dict[str, Any], *, verified: bool) -> None:
        """以当前租约登记预期清单或实际核验证据，未核验批次始终保持私有。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            batch_id = await connection.fetchval("""UPDATE upload_batches SET state=$9,manifest_ref=$2::jsonb,
                embedding_fingerprint=$3,payload_sha256=$4,chunk_count=$5,collection_name=$6,
                verified_at=CASE WHEN $9='verified' THEN NOW() ELSE NULL END
                WHERE batch_id=$1 AND worker_id=$7 AND task_id=$8 AND state='building' RETURNING batch_id""",
                execution.batch_id, dump_json(evidence["manifest_ref"]), evidence["embedding_fingerprint"],
                evidence["payload_sha256"], evidence["chunk_count"], evidence["collection_name"], execution.worker_id, execution.task_id,
                "verified" if verified else "building")
            if batch_id is None:
                raise UploadLeaseLostError("索引批次已变化，当前执行不能登记核验通过")

    async def publish(self, execution: UploadExecution, manifest: dict[str, Any], *,
                      source_change: SourceVersionChange) -> dict[str, Any]:
        """【上传】【原子发布】校验 owner、生命周期、租约和旧指针后同时切换论文及完成任务。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            document = await connection.fetchrow("SELECT * FROM upload_documents WHERE paper_id=$1 FOR UPDATE", execution.paper_id)
            paper = await connection.fetchrow("SELECT * FROM papers WHERE paper_id=$1 FOR UPDATE", execution.paper_id)
            previous = paper["index_batch_id"] if paper is not None else None
            if previous != execution.previous_batch_id:
                raise UploadConflictError("论文已发布其他结果，本次不能覆盖新的发布记录")
            if paper is not None and (paper["user_id"] != execution.user_id or paper["deletion_status"] != "active"):
                raise UploadConflictError("论文归属或删除状态已变化，不能发布")
            batch = await connection.fetchrow("SELECT * FROM upload_batches WHERE batch_id=$1 FOR UPDATE", execution.batch_id)
            if batch is None or batch["state"] != "verified" or batch["worker_id"] != execution.worker_id:
                raise UploadConflictError("当前索引尚未完整核验，不能显示可检索")
            if (manifest.get("batch_id") != execution.batch_id or manifest.get("user_id") != execution.user_id
                    or manifest.get("paper_id") != execution.paper_id or manifest.get("payload_sha256") != batch["payload_sha256"]):
                raise UploadConflictError("发布清单身份或载荷指纹不一致")
            await self._apply_source_change(connection, execution, batch, source_change)
            await self._publish_paper(connection, execution, manifest)
            await connection.execute("UPDATE upload_batches SET state='superseded' WHERE batch_id=$1 AND state='published'", previous)
            await connection.execute("UPDATE upload_batches SET state='published',published_at=NOW() WHERE batch_id=$1", execution.batch_id)
            if execution.operation != "metadata":
                await connection.execute("UPDATE upload_documents SET state='ready',last_failure=NULL,updated_at=NOW() WHERE paper_id=$1", execution.paper_id)
            await self._finish_metadata_state(connection, execution, manifest)
            await self._publish_project_links(connection, execution)
            result = {"paper_id": execution.paper_id, "batch_id": execution.batch_id, "searchable": True,
                "title": manifest["paper"]["title"], "chunk_count": batch["chunk_count"], "metadata_missing": manifest["metadata_missing"]}
            task = await self._finish_task(connection, execution, result)
            if execution.operation != "metadata":
                await self._queue_enrichments(connection, document, manifest)
            return task

    async def _apply_source_change(self, connection: Any, execution: UploadExecution, batch: Any,
                                   change: SourceVersionChange) -> None:
        """参数为发布连接、执行、批次及已核验事实；同事务核对比较对象并衔接记忆失效。"""
        identity = (change.paper_id, change.user_id, change.previous_batch_id, change.batch_id)
        expected = (execution.paper_id, execution.user_id, execution.previous_batch_id, execution.batch_id)
        if identity != expected or change.manifest_sha256 != upload_row(batch)["manifest_ref"]["sha256"]:
            raise UploadConflictError("来源版本比较与本次发布对象不一致")
        previous = None
        if execution.previous_batch_id:
            row = await connection.fetchrow("SELECT * FROM upload_batches WHERE batch_id=$1 FOR SHARE", execution.previous_batch_id)
            previous = upload_row(row)
        previous_sha = previous["manifest_ref"]["sha256"] if previous else None
        if previous_sha != change.previous_manifest_sha256:
            raise UploadConflictError("来源比较的旧清单已变化，不能提交过期结论")
        if self._source_change_handler is None:
            raise RuntimeError("来源发布未绑定同事务记忆失效端口")
        await self._source_change_handler(connection, change)

    async def _publish_paper(self, connection: Any, execution: UploadExecution, manifest: dict[str, Any]) -> None:
        """完整正文独立保存，业务主数据只在新索引核验后更新。"""
        paper = manifest["paper"]
        status = "incomplete" if manifest["metadata_missing"] else "completed"
        saved = await connection.fetchval("""INSERT INTO papers(paper_id,user_id,library_id,title,authors,abstract,pdf_path,
            published_date,source,is_vectorized,index_batch_id,metadata_status,metadata_missing)
            VALUES($1,$2,$3,$4,$5::jsonb,$6,$7,$8,'upload',TRUE,$9,$10,$11::jsonb)
            ON CONFLICT(paper_id) DO UPDATE SET title=EXCLUDED.title,authors=EXCLUDED.authors,abstract=EXCLUDED.abstract,
            pdf_path=EXCLUDED.pdf_path,published_date=EXCLUDED.published_date,is_vectorized=TRUE,index_batch_id=EXCLUDED.index_batch_id,
            metadata_status=EXCLUDED.metadata_status,metadata_missing=EXCLUDED.metadata_missing,metadata_error=''
            WHERE papers.user_id=EXCLUDED.user_id AND papers.deletion_status='active' RETURNING paper_id""",
            execution.paper_id, execution.user_id, paper["library_id"], paper["title"], dump_json(paper["authors"]), paper["abstract"],
            paper["pdf_path"], paper["published_date"], execution.batch_id, status, dump_json(manifest["metadata_missing"]))
        if saved is None:
            raise UploadConflictError("论文归属不一致，拒绝写入发布结果")

    async def _finish_metadata_state(self, connection: Any, execution: UploadExecution, manifest: dict[str, Any]) -> None:
        """缺失项保留为空并明确标记，不以文件名和正文片段伪造识别成功。"""
        status = "incomplete" if manifest["metadata_missing"] else "completed"
        await connection.execute("""UPDATE upload_documents SET metadata_status=$2,metadata_error='',metadata_failure=NULL
            WHERE paper_id=$1""", execution.paper_id, status)

    async def _publish_project_links(self, connection: Any, execution: UploadExecution) -> None:
        """受理期间登记的项目意图在发布时补齐，已有角色与备注保持原值。"""
        await connection.execute("""INSERT INTO project_paper_links(project_id,paper_id)
            SELECT DISTINCT p.project_id,$1 FROM upload_requests r JOIN projects p ON p.project_id=r.project_id
            WHERE r.paper_id=$1 AND r.user_id=$2 AND p.user_id=$2 ON CONFLICT(project_id,paper_id) DO NOTHING""",
            execution.paper_id, execution.user_id)

    async def _finish_task(self, connection: Any, execution: UploadExecution, result: dict[str, Any]) -> dict[str, Any]:
        """在发布事务内提交完成状态，避免先清原文件或提前显示成功。"""
        row = await connection.fetchrow("""UPDATE task_runs SET status='completed',progress=100,message='处理完成',
            result=$2::jsonb,error=NULL,lease_until=NULL,finished_at=NOW(),updated_at=NOW() WHERE task_id=$1 RETURNING *""",
            execution.task_id, dump_json(result))
        await connection.execute("UPDATE upload_attempts SET state='completed',stage='publish',finished_at=NOW() WHERE attempt_id=$1", execution.batch_id)
        await self._tasks.add_task_event(execution.task_id, "completed", "已核验结果并完成发布", result, connection=connection)
        task = _task_run_data(row)
        if task is None:
            raise RuntimeError("上传任务完成状态未持久保存")
        return task

    async def _queue_enrichments(self, connection: Any, document: Any, manifest: dict[str, Any]) -> None:
        """基础任务完成后，缺失元信息和用户启用的图谱分别成为独立任务。"""
        if manifest["metadata_missing"]:
            await self._uploads.queue_operation(connection, document, operation="metadata")
            await connection.execute("UPDATE papers SET metadata_status='queued' WHERE paper_id=$1", document["paper_id"])
        if document["graph_requested"]:
            await self._uploads.queue_operation(connection, document, operation="graph")

    async def finish_graph(self, execution: UploadExecution, result: dict[str, Any]) -> dict[str, Any]:
        """图谱单独提交完成，不改变基础论文发布指针。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            await connection.execute("UPDATE upload_documents SET graph_status='completed',graph_error='',graph_failure=NULL WHERE paper_id=$1", execution.paper_id)
            await connection.execute("UPDATE upload_batches SET state='abandoned' WHERE batch_id=$1", execution.batch_id)
            return await self._finish_task(connection, execution, result)

    async def snapshot(self, paper_ids: list[str], *, user_id: str, allow_empty: bool = False,
                       acquire_lease: bool = True, expected_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        """【检索】【发布快照】参数为论文、用户及占用选项；返回稳定批次，可在同一事务核对后领取租约。"""
        if not paper_ids or not user_id:
            raise RetrievalScopeError("检索需要非空用户和论文范围")
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            papers = await connection.fetch("SELECT paper_id,index_batch_id FROM papers WHERE paper_id=ANY($1::text[]) AND user_id=$2 AND deletion_status='active' FOR SHARE",
                paper_ids, user_id)
            if {paper["paper_id"] for paper in papers} != set(paper_ids):
                raise ScopeOwnershipError("论文范围不存在或无权访问")
            batch_ids = sorted(paper["index_batch_id"] for paper in papers if paper["index_batch_id"])
            batches = await connection.fetch("SELECT * FROM upload_batches WHERE batch_id=ANY($1::text[]) AND user_id=$2 AND state='published' FOR SHARE", batch_ids, user_id)
            if len(batches) != len(batch_ids):
                raise RetrievalScopeError("论文的已发布索引记录不完整，需要重新处理")
            if not batches and not allow_empty:
                raise RetrievalScopeError("论文尚未完成核验，暂不可检索")
            result = {"user_id": user_id, "batch_ids": batch_ids,
                "paper_batches": {batch["paper_id"]: batch["batch_id"] for batch in batches},
                "embedding_fingerprints": sorted({batch["embedding_fingerprint"] for batch in batches}),
                "collection_names": sorted({batch["collection_name"] for batch in batches})}
            if expected_snapshot is not None and result != expected_snapshot:
                raise RetrievalScopeError("冻结的发布批次已变化，不能用当前版本替代本次评测输入")
            if acquire_lease:
                lease_id = uuid.uuid4().hex
                await connection.execute("INSERT INTO upload_read_leases(lease_id,user_id,batch_ids,expires_at) VALUES($1,$2,$3,NOW()+$4*INTERVAL '1 second')",
                    lease_id, user_id, batch_ids, READ_LEASE_SECONDS)
                result = {**result, "lease_id": lease_id}
        return result

    async def current_batch(self, paper_id: str, *, user_id: str | None = None) -> str | None:
        """返回业务库当前核验发布的批次；内部审计可不限定 owner，用户入口必须提供。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            return await connection.fetchval("""SELECT b.batch_id FROM papers p JOIN upload_batches b ON b.batch_id=p.index_batch_id
                WHERE p.paper_id=$1 AND ($2::text IS NULL OR p.user_id=$2) AND p.deletion_status='active'
                AND b.state='published' AND b.user_id=p.user_id AND b.paper_id=p.paper_id""", paper_id, user_id)

    async def renew_snapshot(self, snapshot: dict[str, Any]) -> None:
        """仍可读的同一快照续租；旧批次已清理或论文删除时显式拒绝，不能暗换版本。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            batches = await connection.fetch("""SELECT b.batch_id,b.paper_id FROM upload_batches b JOIN papers p ON p.paper_id=b.paper_id
                WHERE b.batch_id=ANY($1::text[]) AND b.user_id=$2 AND p.user_id=$2 AND p.deletion_status='active'
                AND b.state IN ('published','superseded') FOR SHARE OF p,b""", snapshot["batch_ids"], snapshot["user_id"])
            if {row["batch_id"] for row in batches} != set(snapshot["batch_ids"]):
                raise RetrievalScopeError("当前检索快照已失效，请重新发起查询")
            if {row["paper_id"]: row["batch_id"] for row in batches} != snapshot["paper_batches"]:
                raise RetrievalScopeError("发布快照的论文与批次对应关系已失效")
            renewed = await connection.fetchval("""UPDATE upload_read_leases SET expires_at=NOW()+$4*INTERVAL '1 second'
                WHERE lease_id=$1 AND user_id=$2 AND batch_ids=$3 RETURNING lease_id""",
                snapshot["lease_id"], snapshot["user_id"], snapshot["batch_ids"], READ_LEASE_SECONDS)
            if renewed is None:
                raise RetrievalScopeError("检索快照读取占用已结束，请重新发起查询")

    async def release_snapshot(self, snapshot: dict[str, Any]) -> None:
        """结束检索占用；进程崩溃时由持久期限回收，不无限保留旧向量。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            await connection.execute("DELETE FROM upload_read_leases WHERE lease_id=$1 AND user_id=$2", snapshot["lease_id"], snapshot["user_id"])

    async def content_reference(self, batch_id: str, *, user_id: str, connection: Any = None) -> dict[str, Any]:
        """参数为批次、账号与可选持锁连接；返回历史引用清单，删除立即撤销后续授权。"""
        if connection is None:
            pool = await self._pool_provider()
            async with pool.acquire() as acquired:
                return await self.content_reference(batch_id, user_id=user_id, connection=acquired)
        row = await connection.fetchrow("""SELECT b.* FROM upload_batches b JOIN papers p ON p.paper_id=b.paper_id
            WHERE b.batch_id=$1 AND b.user_id=$2 AND p.user_id=$2 AND p.deletion_status='active'
            AND b.published_at IS NOT NULL""", batch_id, user_id)
        result = upload_row(row)
        if result is None:
            raise LookupError("引用正文不存在或无权读取")
        return result

    async def retire_candidates(self, paper_id: str, *, user_id: str) -> list[dict[str, Any]]:
        """【上传】【旧索引退役】只退役已无活动读占用且不再发布的批次，保留引用正文。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch("""SELECT b.* FROM upload_batches b
                WHERE b.paper_id=$1 AND b.user_id=$2 AND b.state IN ('superseded','abandoned','retired')
                AND b.cleanup_completed_at IS NULL
                AND NOT EXISTS(SELECT 1 FROM papers p WHERE p.index_batch_id=b.batch_id)
                AND NOT EXISTS(SELECT 1 FROM upload_read_leases r WHERE b.batch_id=ANY(r.batch_ids) AND r.expires_at>NOW())
                AND NOT EXISTS(SELECT 1 FROM task_runs t WHERE t.task_id=b.task_id AND t.worker_id=b.worker_id
                    AND t.status='running' AND t.lease_until>NOW()) FOR UPDATE OF b""", paper_id, user_id)
            ids = [row["batch_id"] for row in rows]
            await connection.execute("UPDATE upload_batches SET state='retired',retired_at=NOW() WHERE batch_id=ANY($1::text[])", ids)
            await connection.execute("DELETE FROM upload_read_leases WHERE expires_at<=NOW()")
        return [upload_row(row) or {} for row in rows]

    async def record_cleanup(self, batch_id: str, error: str = "") -> None:
        """保留清理失败事实，已提交的论文不会因为清理失败回退。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            await connection.execute("""UPDATE upload_batches SET cleanup_error=$2,
                cleanup_completed_at=CASE WHEN $2='' THEN NOW() ELSE NULL END WHERE batch_id=$1""", batch_id, error)


async def _apply_memory_source_change(connection: Any, change: SourceVersionChange) -> None:
    """参数为发布事务连接及已核验来源事实；调用记忆仓储同事务维护复用资格。"""
    from app.repositories.memory_source_repository import apply_source_change

    await apply_source_change(connection, change)


def get_upload_publication_repository() -> UploadPublicationRepository:
    """返回复用项目连接池的批次发布仓储。"""
    return UploadPublicationRepository(source_change_handler=_apply_memory_source_change)
