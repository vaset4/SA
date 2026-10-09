"""上传受理去重、检查点与尝试记录仓储，作者：xxx。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any, cast
import uuid

from app.models.library_models import DEFAULT_LIBRARY_OWNER
from app.models.upload_models import (
    UploadCancelledError, UploadConflictError, UploadExecution, UploadFailure,
    UploadLeaseLostError, UploadOperation, UploadPausedError, UploadRequest,
)
from app.repositories.common import dump_json, load_json, normalize_record
from app.repositories.task_repository import TaskRepository, _task_run_data, get_task_repository
from app.services.database import get_pool


def upload_row(row: Any) -> dict[str, Any] | None:
    """接收数据库记录，恢复上传专属 JSON 字段并保留真实空值。"""
    result = normalize_record(row)
    if result is None:
        return None
    for key in ("source_ref", "retry_counts", "last_failure", "metadata_failure", "graph_failure", "payload", "manifest_ref", "errors"):
        if key in result and result[key] is not None:
            result[key] = load_json(result[key], None)
    return result


async def _lock_upload_task(connection: Any, execution: UploadExecution) -> dict[str, Any]:
    """【上传】【任务租约】先锁定任务行，维持所有上传事务一致的加锁顺序。"""
    row = await connection.fetchrow(
        """SELECT * FROM task_runs WHERE task_id=$1 AND user_id=$2 AND worker_id=$3
           AND payload->>'paper_id'=$4 AND task_type='upload_ingest' AND status='running' AND lease_until > NOW() FOR UPDATE""",
        execution.task_id, execution.user_id, execution.worker_id, execution.paper_id,
    )
    if row is None:
        raise UploadLeaseLostError("上传任务执行租约已变化，当前执行不能继续写入")
    if row["cancel_requested"]:
        raise UploadCancelledError("上传任务已取消")
    return dict(row)


async def require_upload_execution(connection: Any, execution: UploadExecution) -> dict[str, Any]:
    """【上传】【执行归属】当前租约、原文件代次和最新任务必须同时一致。"""
    task = await _lock_upload_task(connection, execution)
    current = await connection.fetchval(
        """SELECT u.paper_id FROM upload_documents u LEFT JOIN papers p ON p.paper_id=u.paper_id
           WHERE u.paper_id=$1 AND u.user_id=$2 AND u.parse_id=$3 AND u.state <> 'deleted'
           AND (CASE WHEN $5='metadata' THEN u.metadata_task_id WHEN $5='graph' THEN u.graph_task_id ELSE u.task_id END)=$4
           AND (p.paper_id IS NULL OR p.deletion_status='active')""",
        execution.paper_id, execution.user_id, execution.parse_id, execution.task_id, execution.operation,
    )
    if current is None:
        raise UploadLeaseLostError("上传来源、解析代次或论文生命周期已变化，当前执行不能继续")
    return task


class UploadRepository:
    """把受理原子性和阶段恢复事实集中在现有 PostgreSQL 中。"""

    def __init__(self, pool_provider: Callable[[], Awaitable[Any]] = get_pool, *, tasks: TaskRepository | None = None) -> None:
        """接收连接池端口；默认复用项目数据库连接生命周期。"""
        self._pool_provider = pool_provider
        self._tasks = tasks or get_task_repository()

    async def get_document(self, paper_id: str, *, user_id: str) -> dict[str, Any] | None:
        """按服务端用户和论文 ID 读取受理记录及当前发布事实。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT u.*, p.index_batch_id, p.deletion_status FROM upload_documents u
                   LEFT JOIN papers p ON p.paper_id=u.paper_id AND p.user_id=u.user_id
                   WHERE u.paper_id=$1 AND u.user_id=$2""", paper_id, user_id,
            )
        return upload_row(row)

    async def find_source(self, user_id: str, *, library_id: str, source_sha256: str) -> dict[str, Any] | None:
        """按同用户同文献库的原文件指纹读取活动逻辑论文。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT * FROM upload_documents WHERE user_id=$1 AND library_id=$2
                   AND source_sha256=$3 AND state <> 'deleted'""", user_id, library_id, source_sha256,
            )
        return upload_row(row)

    async def get_request(self, request_id: str, *, user_id: str) -> dict[str, Any] | None:
        """响应丢失后按原受理标识查回任务，不生成替代任务。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT t.* FROM upload_requests r JOIN task_runs t ON t.task_id=r.task_id
                   WHERE r.user_id=$1 AND r.request_id=$2 AND t.user_id=$1""", user_id, request_id,
            )
        return _task_run_data(row)

    async def describe_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """返回任务的基础就绪、独立补全状态和当前允许操作，不向浏览器暴露原文件路径。"""
        if task["task_type"] != "upload_ingest":
            return task
        document = await self.get_document(task["payload"].get("paper_id", ""), user_id=task["user_id"])
        if document is None:
            return {**task, "allowed_actions": [], "upload": {"searchable": False, "legacy": True}}
        operation = task["payload"].get("operation", "ingest")
        supplementary = operation in {"metadata", "graph"}
        current_id = document[f"{operation}_task_id"] if supplementary else document["task_id"]
        failure = document[f"{operation}_failure"] if supplementary else document["last_failure"]
        alive = document["state"] != "deleted" and document["deletion_status"] in (None, "active")
        ready = alive and bool(document["index_batch_id"])
        actions = []
        if alive and task["task_id"] == current_id and task["status"] in {"paused", "cancelled", "failed"}:
            if not failure or failure.get("can_resume", True):
                actions.append("resume")
        if alive and task["status"] not in {"queued", "running"}:
            actions.append("rebuild")
        return {**task, "allowed_actions": actions, "upload": {"paper_id": document["paper_id"],
            "filename": document["filename"], "state": document["state"], "searchable": ready,
            "published_batch_id": document["index_batch_id"],
            "using_previous_result": ready and document["state"] != "ready", "current_task_id": document["task_id"],
            "metadata_task_id": document["metadata_task_id"], "metadata_status": document["metadata_status"],
            "metadata_error": document["metadata_error"], "graph_task_id": document["graph_task_id"],
            "graph_status": document["graph_status"], "graph_error": document["graph_error"],
            "last_failure": failure, "resume_sequence": document["resume_sequence"]}}

    async def _validate_target(self, connection: Any, request: UploadRequest) -> None:
        """【上传】【目标归属】受理事务内核对文献库和项目，系统库不代表共享论文。"""
        library = await connection.fetchval(
            """SELECT library_id FROM libraries WHERE library_id=$1 AND status='active'
               AND (owner=$2 OR owner=$3) FOR KEY SHARE""",
            request.library_id, request.user_id, DEFAULT_LIBRARY_OWNER,
        )
        if library is None:
            raise LookupError("目标文献库不存在或无权使用")
        if request.project_id is not None:
            project = await connection.fetchval(
                "SELECT project_id FROM projects WHERE project_id=$1 AND user_id=$2 FOR KEY SHARE",
                request.project_id, request.user_id,
            )
            if project is None:
                raise LookupError("目标项目不存在或无权使用")

    async def accept(self, request: UploadRequest) -> dict[str, Any]:
        """【上传】【受理去重】同请求和同文件在一笔事务内复用唯一任务及论文。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            locks = [f"request:{request.user_id}:{request.request_id}",
                     f"source:{request.user_id}:{request.library_id}:{request.source_ref['sha256']}"]
            for key in sorted(locks):
                await connection.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", key)
            await self._validate_target(connection, request)
            receipt = await connection.fetchrow(
                "SELECT * FROM upload_requests WHERE user_id=$1 AND request_id=$2", request.user_id, request.request_id,
            )
            if receipt is not None:
                if receipt["request_fingerprint"] != request.request_fingerprint:
                    raise UploadConflictError("同一上传标识对应的文件或目标范围发生变化")
                state = await connection.fetchval("SELECT state FROM upload_documents WHERE paper_id=$1", receipt["paper_id"])
                if state == "deleted":
                    raise UploadConflictError("原上传对应的论文已经删除；再次上传应使用新的上传意图")
                return await self._accepted_result(connection, receipt["task_id"], request, kind="same_request")
            existing = await connection.fetchrow(
                """SELECT u.*, p.deletion_status FROM upload_documents u LEFT JOIN papers p ON p.paper_id=u.paper_id
                   WHERE u.user_id=$1 AND u.library_id=$2 AND u.source_sha256=$3 AND u.state <> 'deleted' FOR UPDATE OF u""",
                request.user_id, request.library_id, request.source_ref["sha256"],
            )
            if existing is not None and existing["deletion_status"] not in (None, "active"):
                raise UploadConflictError("相同文件对应的论文正在删除，不能复用或恢复")
            document = dict(existing) if existing is not None else await self._create_document(connection, request)
            kind = "same_file" if existing is not None else "new_file"
            if request.graph_requested:
                await connection.execute("UPDATE upload_documents SET graph_requested=TRUE WHERE paper_id=$1", document["paper_id"])
            await connection.execute(
                """INSERT INTO upload_requests(user_id, request_id, request_fingerprint, paper_id, task_id, project_id, request_kind)
                   VALUES($1,$2,$3,$4,$5,$6,$7)""", request.user_id, request.request_id, request.request_fingerprint,
                document["paper_id"], document["task_id"], request.project_id, kind,
            )
            await self._link_existing_paper(connection, request, document["paper_id"])
            return await self._accepted_result(connection, document["task_id"], request, kind=kind)

    async def _create_document(self, connection: Any, request: UploadRequest) -> dict[str, Any]:
        """在同一受理事务中建立来源记录、稳定论文身份及排队任务。"""
        row = await connection.fetchrow(
            """INSERT INTO upload_documents(paper_id,user_id,library_id,filename,source_ref,source_sha256,
                   source_size,parse_id,task_id,graph_requested) VALUES($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10) RETURNING *""",
            request.paper_id, request.user_id, request.library_id, request.filename, dump_json(request.source_ref),
            request.source_ref["sha256"], request.source_ref["size"], request.parse_id, request.task_id, request.graph_requested,
        )
        await self._tasks.create_task_run("upload_ingest", {
            "paper_id": request.paper_id, "filename": request.filename, "library_id": request.library_id,
            "project_id": request.project_id, "request_id": request.request_id, "operation": "ingest",
        }, "文件已保存，等待解析", task_id=request.task_id, user_id=request.user_id, connection=connection)
        await self._tasks.add_task_event(request.task_id, "received", "原文件已保存并完成指纹核验", {
            "paper_id": request.paper_id, "source_sha256": request.source_ref["sha256"],
            "source_size": request.source_ref["size"], "request_id": request.request_id,
        }, connection=connection)
        if row is None:
            raise RuntimeError("上传来源记录未写入")
        return dict(row)

    async def _link_existing_paper(self, connection: Any, request: UploadRequest, paper_id: str) -> None:
        """新上传意图指向另一项目时只补关联，保留已存在关系的角色与备注。"""
        if request.project_id is None:
            return
        await connection.execute(
            """INSERT INTO project_paper_links(project_id,paper_id)
               SELECT $1,paper_id FROM papers WHERE paper_id=$2 AND user_id=$3 AND deletion_status='active'
               ON CONFLICT(project_id,paper_id) DO NOTHING""", request.project_id, paper_id, request.user_id,
        )

    async def _accepted_result(self, connection: Any, task_id: str, request: UploadRequest, *, kind: str) -> dict[str, Any]:
        """返回原任务及受理类型，不把重复上传转为自动继续或重新解析。"""
        task = await self._tasks.get_task_run(task_id, user_id=request.user_id, connection=connection)
        if task is None:
            raise UploadConflictError("上传受理记录缺少原任务，不能生成替代身份")
        return {**task, "request_id": request.request_id, "reuse_kind": kind}

    async def begin_attempt(self, task: dict[str, Any]) -> UploadExecution:
        """【上传】【执行尝试】每次独立持租执行建立批次，保留前一次发布身份。"""
        operation = str(task["payload"].get("operation", "ingest"))
        if operation not in {"ingest", "rebuild", "metadata", "graph"}:
            raise ValueError("未知上传操作")
        batch_id = uuid.uuid4().hex
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            initial = UploadExecution(task["task_id"], task["worker_id"], task["payload"]["paper_id"], task["user_id"], "", batch_id, None)
            await _lock_upload_task(connection, initial)
            document = await connection.fetchrow(
                """SELECT u.*, p.index_batch_id, p.deletion_status FROM upload_documents u
                   LEFT JOIN papers p ON p.paper_id=u.paper_id WHERE u.paper_id=$1 AND u.user_id=$2 FOR UPDATE OF u""",
                task["payload"]["paper_id"], task["user_id"],
            )
            if document is None or document["state"] == "deleted" or document["deletion_status"] not in (None, "active"):
                raise UploadConflictError("上传来源不存在、归属不符或论文正在删除")
            current_task = document[f"{operation}_task_id"] if operation in {"metadata", "graph"} else document["task_id"]
            if current_task != task["task_id"]:
                raise UploadConflictError("此上传已有后续任务，不能执行旧任务")
            execution = UploadExecution(task["task_id"], task["worker_id"], document["paper_id"], task["user_id"],
                document["parse_id"], batch_id, document["index_batch_id"], cast(UploadOperation, operation))
            await connection.execute("""UPDATE upload_batches b SET state='abandoned' WHERE paper_id=$1
                AND state IN ('building','verified') AND NOT EXISTS(SELECT 1 FROM task_runs t
                    WHERE t.task_id=b.task_id AND t.worker_id=b.worker_id AND t.status='running' AND t.lease_until>NOW())""", execution.paper_id)
            await connection.execute(
                """INSERT INTO upload_attempts(attempt_id,task_id,worker_id,paper_id,user_id,parse_id,batch_id,
                   previous_batch_id,operation,resume_sequence) VALUES($1,$2,$3,$4,$5,$6,$1,$7,$8,$9)""",
                batch_id, execution.task_id, execution.worker_id, execution.paper_id, execution.user_id,
                execution.parse_id, execution.previous_batch_id, operation, document["resume_sequence"],
            )
            await connection.execute(
                """INSERT INTO upload_batches(batch_id,paper_id,user_id,task_id,worker_id,previous_batch_id)
                   VALUES($1,$2,$3,$4,$5,$6)""", batch_id, execution.paper_id, execution.user_id,
                execution.task_id, execution.worker_id, execution.previous_batch_id,
            )
            if operation in {"metadata", "graph"}:
                await connection.execute(f"UPDATE upload_documents SET {operation}_status='processing',updated_at=NOW() WHERE paper_id=$1", execution.paper_id)
            else:
                await connection.execute("UPDATE upload_documents SET state='processing',updated_at=NOW() WHERE paper_id=$1", execution.paper_id)
        return execution

    async def check_execution(self, execution: UploadExecution) -> None:
        """在新的外部步骤前核对取消和租约；输入是当前执行身份，无返回值。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)

    async def checkpoint(self, execution: UploadExecution, stage: str) -> dict[str, Any] | None:
        """读取同一解析代次的阶段事实，旧代次不会作为当前缓存。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM upload_checkpoints WHERE parse_id=$1 AND stage=$2 AND user_id=$3 AND paper_id=$4",
                execution.parse_id, stage, execution.user_id, execution.paper_id,
            )
        return upload_row(row)

    async def start_checkpoint(self, execution: UploadExecution, *, stage: str, input_hash: str, repeatable: bool,
                               attempt_limit: int | None = None, budget_stage: str | None = None) -> dict[str, Any]:
        """【上传】【阶段开始】先保存操作身份；未知的非幂等操作不能重新提交。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            existing = await connection.fetchrow(
                "SELECT * FROM upload_checkpoints WHERE parse_id=$1 AND stage=$2 AND user_id=$3 AND paper_id=$4 FOR UPDATE",
                execution.parse_id, stage, execution.user_id, execution.paper_id,
            )
            if existing is not None and existing["input_hash"] != input_hash:
                raise UploadConflictError("上传阶段的原文件或处理配置发生变化，需要明确重新处理")
            if existing is not None and existing["state"] == "completed":
                return cast(dict[str, Any], upload_row(existing))
            if existing is not None and existing["state"] == "pending" and not repeatable:
                raise UploadPausedError(UploadFailure(stage, "remote_result_unknown", "上次解析创建结果尚未核实，已暂停以避免重复创建", can_resume=stage != "mineru_slot"))
            attempt = await self._reserve_attempt(connection, execution, stage=budget_stage or stage, limit=attempt_limit) if attempt_limit else None
            operation_id = uuid.uuid4().hex
            row = await connection.fetchrow(
                """INSERT INTO upload_checkpoints(parse_id,stage,paper_id,user_id,operation_id,input_hash,state)
                   VALUES($1,$2,$3,$4,$5,$6,'pending') ON CONFLICT(parse_id,stage) DO UPDATE
                   SET operation_id=EXCLUDED.operation_id,state='pending',updated_at=NOW()
                   WHERE upload_checkpoints.user_id=EXCLUDED.user_id AND upload_checkpoints.paper_id=EXCLUDED.paper_id RETURNING *""",
                execution.parse_id, stage, execution.paper_id, execution.user_id, operation_id, input_hash,
            )
        result = upload_row(row)
        if result is None:
            raise UploadConflictError("检查点身份冲突，不能复用其他上传的状态")
        return {**result, "attempt": attempt}

    async def complete_checkpoint(self, execution: UploadExecution, checkpoint: dict[str, Any], payload: dict[str, Any]) -> None:
        """【上传】【阶段事实】保存匹配操作的真实结果；迟到结果只补历史事实，不获得发布权。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """UPDATE upload_checkpoints SET state='completed',payload=$1::jsonb,updated_at=NOW()
                   WHERE parse_id=$2 AND stage=$3 AND operation_id=$4 AND user_id=$5 AND paper_id=$6 AND state='pending'
                   RETURNING operation_id""", dump_json(payload), execution.parse_id, checkpoint["stage"],
                checkpoint["operation_id"], execution.user_id, execution.paper_id,
            )
        if row is None:
            raise UploadLeaseLostError("阶段操作已被后续执行替换，不能覆盖其检查点")

    async def record_retry(self, execution: UploadExecution, failure: UploadFailure) -> int:
        """记录本次失败及已持久预留的尝试序号，保留真实错误历史。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            count = await connection.fetchval(
                "SELECT COALESCE((retry_counts->>$2)::int,0) FROM upload_documents WHERE paper_id=$1",
                execution.paper_id, f"{execution.task_id}:{failure.stage}",
            )
            await connection.execute(
                "UPDATE upload_attempts SET errors=errors || $2::jsonb,stage=$3 WHERE attempt_id=$1",
                execution.batch_id, dump_json([asdict(failure)]), failure.stage,
            )
            await self._tasks.add_task_event(execution.task_id, "attempt_failed", failure.detail,
                {**asdict(failure), "attempt": count}, connection=connection)
        return int(count)

    async def reserve_attempt(self, execution: UploadExecution, *, stage: str, limit: int) -> int:
        """【上传】【尝试预算】外部调用前持久计数，调用后崩溃也不能重获完整重试预算。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            return await self._reserve_attempt(connection, execution, stage=stage, limit=limit)

    async def _reserve_attempt(self, connection: Any, execution: UploadExecution, *, stage: str, limit: int) -> int:
        """把调用预算和操作检查点放在同一事务，预算不足不能把明确拒绝改成结果未知。"""
        # 1. 【上传】【尝试预算】任务身份固定预算归属，元信息与图谱不能重置或消耗彼此的次数
        budget_key = f"{execution.task_id}:{stage}"
        count = await connection.fetchval(
                """UPDATE upload_documents SET retry_counts=jsonb_set(retry_counts,ARRAY[$2::text],
                   to_jsonb(COALESCE((retry_counts->>$2)::int,0)+1)),updated_at=NOW()
                   WHERE paper_id=$1 AND COALESCE((retry_counts->>$2)::int,0) < $3
                   RETURNING (retry_counts->>$2)::int""", execution.paper_id, budget_key, limit,
        )
        if count is None:
            raise UploadPausedError(UploadFailure(stage, "retry_budget_exhausted", "本轮短暂尝试已用完，原文件与进度已保存，请手动继续"))
        await self._tasks.add_task_event(execution.task_id, "attempt_started", "开始当前阶段尝试",
            {"stage": stage, "attempt": count, "batch_id": execution.batch_id}, connection=connection)
        return int(count)

    async def reject_checkpoint(self, execution: UploadExecution, checkpoint: dict[str, Any], failure: UploadFailure) -> None:
        """仅记录供应商明确拒绝的创建请求；网络结果未知时不能使用此入口。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """UPDATE upload_checkpoints SET state='rejected',payload=$1::jsonb,updated_at=NOW()
                   WHERE parse_id=$2 AND stage=$3 AND operation_id=$4 AND user_id=$5 AND state='pending' RETURNING operation_id""",
                dump_json(asdict(failure)), execution.parse_id, checkpoint["stage"], checkpoint["operation_id"], execution.user_id,
            )
            if row is None:
                raise UploadLeaseLostError("阶段操作已被替换，拒绝结果只能属于原操作")
            await self._tasks.add_task_event(execution.task_id, "provider_rejected", failure.detail,
                {**asdict(failure), "operation_id": checkpoint["operation_id"]}, connection=connection)

    async def progress(self, execution: UploadExecution, *, stage: str, percent: int, message: str) -> None:
        """更新当前合法执行的阶段，旧 worker 不能回写新任务进度。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            await connection.execute("UPDATE task_runs SET progress=$2,message=$3,updated_at=NOW() WHERE task_id=$1",
                execution.task_id, percent, message)
            await connection.execute("UPDATE upload_attempts SET stage=$2 WHERE attempt_id=$1", execution.batch_id, stage)
            await self._tasks.add_task_event(execution.task_id, "stage", message, {"stage": stage, "batch_id": execution.batch_id}, connection=connection)

    async def pause(self, execution: UploadExecution, failure: UploadFailure) -> dict[str, Any]:
        """【上传】【持久暂停】失败与历史尝试一起提交，保留原文件、检查点和有效旧发布。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await require_upload_execution(connection, execution)
            row = await connection.fetchrow(
                """UPDATE task_runs SET status='paused',error=$2,message=$2,lease_until=NULL,finished_at=NOW(),updated_at=NOW()
                   WHERE task_id=$1 RETURNING *""", execution.task_id, failure.detail,
            )
            if execution.operation in {"metadata", "graph"}:
                operation = execution.operation
                await connection.execute(f"""UPDATE upload_documents SET {operation}_status='paused',
                    {operation}_error=$2,{operation}_failure=$3::jsonb,updated_at=NOW() WHERE paper_id=$1""",
                    execution.paper_id, failure.detail, dump_json(asdict(failure)))
                if operation == "metadata":
                    await connection.execute("UPDATE papers SET metadata_status='paused',metadata_error=$2 WHERE paper_id=$1", execution.paper_id, failure.detail)
            else:
                await connection.execute(
                    """UPDATE upload_documents SET state='paused',last_failure=$2::jsonb,updated_at=NOW()
                       WHERE paper_id=$1 AND user_id=$3""", execution.paper_id, dump_json(asdict(failure)), execution.user_id,
                )
            await connection.execute(
                """UPDATE upload_attempts SET state='paused',stage=$2,errors=errors || $3::jsonb,finished_at=NOW()
                   WHERE attempt_id=$1""", execution.batch_id, failure.stage, dump_json([asdict(failure)]),
            )
            await connection.execute("UPDATE upload_batches SET state='abandoned' WHERE batch_id=$1 AND published_at IS NULL", execution.batch_id)
            await self._tasks.add_task_event(execution.task_id, "paused", failure.detail, asdict(failure), connection=connection)
        task = _task_run_data(row)
        if task is None:
            raise RuntimeError("上传暂停状态未保存")
        return task

    async def resume(self, task_id: str, *, user_id: str) -> dict[str, Any]:
        """【上传】【手动继续】并发点击只将原任务排队一次，保留历史及原 paper_id。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow("SELECT * FROM task_runs WHERE task_id=$1 AND user_id=$2 FOR UPDATE", task_id, user_id)
            task = _task_run_data(row)
            if task is None or task["task_type"] != "upload_ingest":
                raise LookupError("上传任务不存在")
            if task["status"] in {"queued", "running", "completed"}:
                return task
            document = await connection.fetchrow(
                """SELECT u.*, p.deletion_status FROM upload_documents u LEFT JOIN papers p ON p.paper_id=u.paper_id
                   WHERE u.paper_id=$1 AND u.user_id=$2 FOR UPDATE OF u""", task["payload"]["paper_id"], user_id,
            )
            if document is None or document["state"] == "deleted" or document["deletion_status"] not in (None, "active"):
                raise UploadConflictError("原文记录不存在或论文正在删除，不能继续")
            operation = task["payload"].get("operation", "ingest")
            supplementary = operation in {"metadata", "graph"}
            current_task = document[f"{operation}_task_id"] if supplementary else document["task_id"]
            if current_task != task_id:
                raise UploadConflictError("此上传已有后续任务，请查看最新任务")
            failure = load_json(document[f"{operation}_failure"] if supplementary else document["last_failure"], None)
            if failure and not failure.get("can_resume", True):
                raise UploadConflictError(failure["detail"])
            await self._tasks.add_task_event(task_id, "resumed", "用户继续处理，复用原论文和有效检查点", {
                "previous_error": task.get("error"), "previous_retry_counts": load_json(document["retry_counts"], {}),
            }, connection=connection)
            status_assignment = f"{operation}_status='queued'" if supplementary else "state='queued'"
            # 1. 【上传】【手动继续】只开启当前任务的新预算，其他任务的暂停事实和计数继续保留
            await connection.execute(f"""UPDATE upload_documents SET {status_assignment},resume_sequence=resume_sequence+1,
                retry_counts=COALESCE((SELECT jsonb_object_agg(key,value) FROM jsonb_each(retry_counts)
                    WHERE left(key,length($2))<>$2),'{{}}'::jsonb),updated_at=NOW() WHERE paper_id=$1""",
                document["paper_id"], task_id + ":")
            resumed = await connection.fetchrow(
                """UPDATE task_runs SET status='queued',cancel_requested=FALSE,worker_id=NULL,lease_until=NULL,
                   error=NULL,finished_at=NULL,updated_at=NOW(),message='已继续，等待处理' WHERE task_id=$1 RETURNING *""", task_id,
            )
        result = _task_run_data(resumed)
        if result is None:
            raise RuntimeError("继续上传状态未保存")
        return result

    async def request_operation(self, paper_id: str, *, user_id: str, operation: UploadOperation) -> dict[str, Any]:
        """【上传】【后续处理】重建建立新解析代次；补全和建图独立排队，重复点击复用当前任务。"""
        if operation not in {"rebuild", "metadata", "graph"}:
            raise ValueError("不支持的后续处理操作")
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", f"operation:{user_id}:{paper_id}")
            row = await connection.fetchrow("SELECT * FROM upload_documents WHERE paper_id=$1 AND user_id=$2", paper_id, user_id)
            if row is None or row["state"] == "deleted":
                raise LookupError("上传论文不存在")
            task_ids = [row[key] for key in ("task_id", "metadata_task_id", "graph_task_id") if row[key]]
            tasks = await connection.fetch("SELECT * FROM task_runs WHERE task_id=ANY($1::text[]) ORDER BY task_id FOR UPDATE", task_ids)
            current = await connection.fetchrow("SELECT * FROM upload_documents WHERE paper_id=$1 FOR UPDATE", paper_id)
            if any(current[key] != row[key] for key in ("task_id", "metadata_task_id", "graph_task_id")):
                raise UploadConflictError("上传状态刚刚变化，请刷新后继续")
            for task in tasks:
                if task["status"] not in {"queued", "running"}:
                    continue
                if load_json(task["payload"], {}).get("operation") == operation:
                    return cast(dict[str, Any], _task_run_data(task))
                raise UploadConflictError("论文仍有处理任务，请先等待或取消后再操作")
            status = await connection.fetchrow("SELECT deletion_status,index_batch_id FROM papers WHERE paper_id=$1 AND user_id=$2 FOR UPDATE", paper_id, user_id)
            if status is not None and status["deletion_status"] != "active":
                raise UploadConflictError("论文正在删除，不能继续处理")
            if operation != "rebuild" and (status is None or not status["index_batch_id"]):
                raise UploadConflictError("请先完成基础上传，再补全信息或构建图谱")
            task = await self.queue_operation(connection, current, operation=operation)
            return task

    async def queue_operation(self, connection: Any, document: Any, *, operation: UploadOperation) -> dict[str, Any]:
        """在已锁定的上传事务内排队后续操作，返回持久任务。"""
        task_id = str(uuid.uuid4())
        paper_id = document["paper_id"]
        task = await self._tasks.create_task_run("upload_ingest", {"paper_id": paper_id, "operation": operation,
            "filename": document["filename"], "library_id": document["library_id"]},
            {"rebuild": "重新处理已排队，已有可用结果继续保留", "metadata": "论文信息补全已排队", "graph": "可选图谱构建已排队"}[operation],
            task_id=task_id, user_id=document["user_id"], connection=connection)
        if operation == "rebuild":
            # 1. 【上传】【重新处理】新解析代次解除旧增强任务入口，旧任务和错误仍保留在任务历史
            await connection.execute("""UPDATE upload_documents SET task_id=$2,parse_id=$3,state='queued',
                metadata_task_id=NULL,graph_task_id=NULL,metadata_status='not_started',graph_status='not_requested',
                metadata_error='',graph_error='',metadata_failure=NULL,graph_failure=NULL,
                retry_counts='{}'::jsonb,resume_sequence=0,last_failure=NULL,updated_at=NOW() WHERE paper_id=$1""",
                paper_id, task_id, uuid.uuid4().hex)
        else:
            await connection.execute(f"""UPDATE upload_documents SET {operation}_task_id=$2,{operation}_status='queued',
                {operation}_error='',{operation}_failure=NULL,updated_at=NOW() WHERE paper_id=$1""", paper_id, task_id)
        await self._tasks.add_task_event(task_id, "queued", task["message"], {"paper_id": paper_id, "operation": operation}, connection=connection)
        return task

    async def record_cancelled(self, task: dict[str, Any]) -> None:
        """同步已取消上传任务的可见状态，后续任务和已发布结果不受旧任务影响。"""
        operation = task["payload"].get("operation", "ingest")
        if operation not in {"ingest", "rebuild", "metadata", "graph"}:
            raise ValueError("未知上传操作")
        supplementary = operation in {"metadata", "graph"}
        column = f"{operation}_status" if supplementary else "state"
        task_column = f"{operation}_task_id" if supplementary else "task_id"
        pool = await self._pool_provider()
        async with pool.acquire() as connection:
            await connection.execute(f"""UPDATE upload_documents SET {column}='cancelled',updated_at=NOW()
                WHERE paper_id=$1 AND user_id=$2 AND {task_column}=$3 AND state <> 'deleted'""",
                task["payload"]["paper_id"], task["user_id"], task["task_id"])


def get_upload_repository() -> UploadRepository:
    """返回使用当前数据库端口的上传仓储；连接池由基础设施复用。"""
    return UploadRepository()
