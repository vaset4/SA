"""【研究交付】【事务保存】同连接提交所有用户投影，作者：xxx，2026-10-04 13:00:00。"""

from __future__ import annotations

from typing import Any

from app.models.result_delivery_contracts import DeliveryConflict, PendingDelivery
from app.repositories.common import dump_json, load_json, normalize_record, utc_now
from app.repositories.conversation_repository import ConversationRepository, _message_data
from app.repositories.rag_run_repository import RAGRunRepository, _run_data
from app.services import database

RECENT_ATTEMPT_LIMIT = 5
ACTIVE_STATUSES = {"queued", "running"}


class ResultDeliveryRepository:
    """维护待提交产物和当前结果；所有投影仅由一次数据库事务更新。"""

    def __init__(self, *, pool_provider=None, run_repository=None, conversation_repository=None):
        """参数为连接池与同连接仓储端口；初始化结果保存依赖，无返回值。"""
        self._pool_provider = pool_provider or database.get_pool
        self._runs = run_repository or RAGRunRepository()
        self._conversations = conversation_repository or ConversationRepository()

    async def stage(self, *, user_id: str, pending: PendingDelivery) -> None:
        """参数为已生成产物及生产身份；原子暂存恢复材料，冲突时保留旧产物。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            target, operation = await self._lock_runs(connection, user_id, pending.target_run_id, pending.producer_run_id)
            self._assert_operation(operation, target)
            if int(target.get("delivery_revision") or 0) != pending.expected_delivery_revision:
                raise DeliveryConflict("答案版本已变化，旧产物不能覆盖当前答案")
            existing = load_json(target.get("pending_artifact"), None)
            payload = pending.model_dump(mode="json")
            if existing is not None and existing != payload:
                raise DeliveryConflict("该答案仍有待保存产物，请先重试保存")
            await self._check_session(connection, user_id, pending, required=operation.get("target_run_id") is None)
            await connection.execute(
                "UPDATE rag_runs SET pending_artifact=$3::jsonb WHERE user_id=$1 AND run_id=$2",
                user_id, pending.target_run_id, dump_json(payload),
            )

    async def commit(self, *, user_id: str, operation_run_id: str, target_run_id: str) -> dict[str, Any]:
        """参数为当前轻量/生成操作与目标；返回同一事务已提交的运行、消息和响应。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            target, operation = await self._lock_runs(connection, user_id, target_run_id, operation_run_id)
            raw_pending = load_json(target.get("pending_artifact"), None)
            if raw_pending is None:
                return await self._already_committed(connection, target, operation)
            pending = PendingDelivery.model_validate(raw_pending)
            self._assert_operation(operation, target)
            if operation_run_id != pending.producer_run_id and operation.get("operation_kind") != "persist_only":
                raise DeliveryConflict("只有原生产操作或补存操作可以提交该产物")
            revision = int(target.get("delivery_revision") or 0)
            if revision != pending.expected_delivery_revision:
                raise DeliveryConflict("答案版本已变化，不能提交旧产物")
            if operation_run_id != pending.producer_run_id:
                producer = await connection.fetchrow(
                    "SELECT * FROM rag_runs WHERE user_id=$1 AND run_id=$2 FOR UPDATE", user_id, pending.producer_run_id,
                )
                if producer is None or producer.get("cancel_requested") or producer["status"] == "cancelled":
                    raise DeliveryConflict("取消操作的产物不能补存成新答案")
            session = await self._check_session(connection, user_id, pending,
                                                 required=operation.get("target_run_id") is None)
            metadata, response = self._projection(target, operation, pending)
            # 1. 【研究交付】【同次提交】运行、消息、会话与终态共享连接，任一步失败整体回滚
            updated = await connection.fetchrow(
                """UPDATE rag_runs SET answer=$3, metadata=$4::jsonb,
                    delivery_revision=$5, pending_artifact=NULL, updated_at=$6
                    WHERE user_id=$1 AND run_id=$2 RETURNING *""",
                user_id, target_run_id, pending.artifact.answer, dump_json(metadata), revision + 1, utc_now(),
            )
            message = await self._conversations.commit_delivery(
                connection=connection, user_id=user_id, target=target,
                answer=pending.artifact.answer, metadata=metadata,
            )
            await self._save_session(connection, user_id, session, pending=pending, response=response)
            terminal = response if operation_run_id == target_run_id else self._operation_summary(response)
            finished = await self._runs.finish_run(
                connection=connection, user_id=user_id, run_id=operation_run_id,
                expected_status=operation["status"], status="completed", payload=terminal,
                metadata=metadata if operation_run_id == target_run_id else self._operation_summary(response),
            )
            if finished is None:
                raise DeliveryConflict("提交时操作已失去执行权")
            run = finished if operation_run_id == target_run_id else _run_data(updated)
            return {**self._operation_summary(response), "run": run, "response": response, "message_patch": message}

    async def fail(self, *, user_id: str, operation_run_id: str, error: str, payload: dict,
                   status: str = "failed") -> dict | None:
        """参数为真实保存错误与停止回执；事务保存失败终态及轻量尝试摘要，不覆盖答案。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            operation = _run_data(await connection.fetchrow(
                "SELECT * FROM rag_runs WHERE user_id=$1 AND run_id=$2", user_id, operation_run_id,
            ))
            if operation is None:
                return operation
            target, operation = await self._lock_runs(
                connection, user_id, operation.get("target_run_id") or operation_run_id, operation_run_id,
            )
            if operation["status"] not in ACTIVE_STATUSES:
                return operation
            target_id = operation.get("target_run_id")
            if target_id:
                if target is not None:
                    summaries = list(target["metadata"].get("planner_final_retry_attempts") or [])
                    summaries.append({"operation_run_id": operation_run_id, "status": status, "error": error})
                    await connection.execute(
                        "UPDATE rag_runs SET metadata=metadata || $3::jsonb WHERE user_id=$1 AND run_id=$2",
                        user_id, target_id, dump_json({"planner_final_retry_attempts": summaries[-RECENT_ATTEMPT_LIMIT:]}),
                    )
            await connection.execute(
                """UPDATE sessions SET status=$5,last_error=$3,updated_at=$4
                    WHERE user_id=$1 AND current_run_id=$2""", user_id, operation_run_id, error, utc_now(), status,
            )
            return await self._runs.finish_run(
                connection=connection, user_id=user_id, run_id=operation_run_id,
                expected_status=operation["status"], status=status, payload=payload,
                error=error, metadata=self._operation_summary(payload),
            )

    async def _lock_runs(self, connection, user_id: str, target_id: str, operation_id: str) -> tuple[dict, dict]:
        """参数为连接和两个身份；按固定顺序锁行并返回目标、操作，避免交错提交。"""
        rows = await connection.fetch(
            "SELECT * FROM rag_runs WHERE user_id=$1 AND run_id=ANY($2::text[]) ORDER BY run_id FOR UPDATE",
            user_id, sorted({target_id, operation_id}),
        )
        runs = {str(row["run_id"]): _run_data(row) for row in rows}
        if target_id not in runs or operation_id not in runs:
            raise LookupError("结果或操作不存在")
        return runs[target_id], runs[operation_id]

    async def record_memory_writeback(self, *, user_id: str, run_id: str, delivery_revision: int, outcome: dict) -> bool:
        """【研究交付】【候选状态】参数为已提交目标、版本和候选事实；返回是否更新同事务三个投影。"""
        pool = await self._pool_provider()
        async with pool.acquire() as connection, connection.transaction():
            target = await connection.fetchrow(
                "SELECT * FROM rag_runs WHERE user_id=$1 AND run_id=$2 FOR UPDATE", user_id, run_id,
            )
            if target is None or int(target.get("delivery_revision") or 0) != delivery_revision:
                raise DeliveryConflict("答案版本已变化，旧候选状态不能覆盖当前结果")
            metadata = load_json(target.get("metadata"), {})
            if metadata.get("persistence_status") != "saved":
                raise DeliveryConflict("候选处理必须关联已保存的答案")
            previous = metadata.get("memory_writeback") or {}
            if outcome.get("status") == "pending":
                if previous.get("status") == "pending" and "awaiting_candidate_processing" not in previous.get("reason_codes", []):
                    raise DeliveryConflict("候选正在处理，不能重复启动")
                if previous.get("status") in {"success", "empty"}:
                    return False
            # 1. 【研究交付】【版本归属】候选状态不重算正文或终态，仅修改匹配版本的独立事实
            patch = dump_json({"memory_writeback": outcome})
            await connection.execute(
                "UPDATE rag_runs SET metadata=metadata || $3::jsonb WHERE user_id=$1 AND run_id=$2",
                user_id, run_id, patch,
            )
            await connection.execute(
                """UPDATE messages m SET metadata=m.metadata || $4::jsonb FROM conversations c
                    WHERE m.conversation_id=c.id AND c.user_id=$1 AND m.role='assistant'
                    AND m.metadata->>'run_id'=$2 AND (m.metadata->>'delivery_revision')::int=$3""",
                user_id, run_id, delivery_revision, patch,
            )
            await connection.execute(
                """UPDATE sessions SET current_result=current_result || $4::jsonb
                    WHERE user_id=$1 AND result_run_id=$2 AND delivery_revision=$3""",
                user_id, run_id, delivery_revision, patch,
            )
            return True

    @staticmethod
    def _assert_operation(operation: dict, target: dict) -> None:
        """参数为已锁操作和目标；校验活动身份、取消及目标归属，无返回值。"""
        if operation["status"] not in ACTIVE_STATUSES or operation.get("cancel_requested"):
            raise DeliveryConflict("操作已结束或正在停止，不能再提交结果")
        if operation["run_id"] != target["run_id"] and operation.get("target_run_id") != target["run_id"]:
            raise DeliveryConflict("操作与目标答案不匹配")
        expected = operation.get("expected_delivery_revision")
        if expected is not None and int(expected) != int(target.get("delivery_revision") or 0):
            raise DeliveryConflict("操作领取时的结果版本已变化")

    async def _check_session(self, connection, user_id: str, pending: PendingDelivery, *, required: bool) -> dict | None:
        """参数为连接、产物与严格归属标记；返回仍归本次研究的会话或历史补存空关联。"""
        if not pending.session_id or not pending.session_run_id:
            return None
        row = await connection.fetchrow(
            "SELECT * FROM sessions WHERE user_id=$1 AND session_id=$2 FOR UPDATE", user_id, pending.session_id,
        )
        if row is not None and row.get("current_run_id") == pending.session_run_id:
            return normalize_record(row)
        if required:
            raise DeliveryConflict("研究会话已进入其他尝试，旧执行不能暂存当前结果")
        return None

    @staticmethod
    def _projection(target: dict, operation: dict, pending: PendingDelivery) -> tuple[dict, dict]:
        """参数为已锁事实及产物；返回同版本元数据和响应，摘要基于事务内最新数组。"""
        revision = int(target.get("delivery_revision") or 0) + 1
        saved = {"target_run_id": target["run_id"], "operation_run_id": operation["run_id"],
                 "operation_status": "completed",
                 "delivery_revision": revision, "persistence_status": "saved", "persistence_error": None,
                 "can_retry_save": False, "producer_run_id": pending.producer_run_id}
        response_fields = {key: value for key, value in pending.artifact.response.items()
                           if key not in {"answer", "report", "type"}}
        scope = {key: target["metadata"][key] for key in ("project_id", "paper_ids", "explicit_paper_ids")
                 if key in target["metadata"]}
        metadata = {**scope, **response_fields, **pending.artifact.metadata, **saved, "run_id": target["run_id"],
                    "response_has_report": "report" in pending.artifact.response,
                    "run_status": "completed" if operation["run_id"] == target["run_id"] else target["status"]}
        # 1. 【研究交付】【候选归属】待处理候选属于实际提交操作，补存不得沿用旧产出run的停止时间
        metadata["memory_writeback"] = {**metadata["memory_writeback"], "processing_source": "delivery_run",
                                        "processing_run_id": operation["run_id"]}
        metadata.pop("report", None)
        summaries = list(target["metadata"].get("planner_final_retry_attempts") or [])
        if pending.artifact.attempt_summary is not None:
            summaries.append({**pending.artifact.attempt_summary, "operation_run_id": operation["run_id"],
                              "producer_run_id": pending.producer_run_id})
        if summaries:
            metadata["planner_final_retry_attempts"] = summaries[-RECENT_ATTEMPT_LIMIT:]
        response = {**pending.artifact.response, **saved, "type": "done", "run_id": target["run_id"],
                    "run_status": metadata["run_status"], "answer": pending.artifact.answer,
                    "memory_writeback": metadata["memory_writeback"]}
        return metadata, response

    async def _save_session(self, connection, user_id: str, session: dict | None, *, pending: PendingDelivery, response: dict) -> None:
        """参数为仍有效会话及已提交投影；只更新本研究尝试的结果与状态，不触碰新尝试。"""
        if session is None:
            return
        await connection.execute(
            """UPDATE sessions SET current_result=$3::jsonb, result_run_id=$4, delivery_revision=$5,
                status=COALESCE($6,status), last_error=NULL, updated_at=$7
                WHERE user_id=$1 AND session_id=$2 AND current_run_id=$8""",
            user_id, pending.session_id, dump_json(response), pending.target_run_id,
            response["delivery_revision"], pending.artifact.session_status, utc_now(), pending.session_run_id,
        )

    @staticmethod
    def _operation_summary(response: dict) -> dict:
        """参数为结果响应；返回无正文的轻量操作回执，避免复制历次完整答案。"""
        keys = ("target_run_id", "operation_run_id", "delivery_revision", "persistence_status",
                "persistence_error", "can_retry_save", "producer_run_id", "final_generation_attempt", "operation_status")
        return {key: response[key] for key in keys if key in response}

    async def _already_committed(self, connection, target: dict, operation: dict) -> dict:
        """参数为当前结果与操作；仅同操作已提交时幂等回读，其余情况明确拒绝。"""
        metadata = target["metadata"]
        if metadata.get("operation_run_id") != operation["run_id"] or metadata.get("persistence_status") != "saved":
            raise DeliveryConflict("没有可补存的服务端产物")
        response = {**metadata, "answer": target.get("answer"), "run_id": target["run_id"]}
        if metadata.get("response_has_report"):
            response["report"] = target.get("answer")
        message = None
        if target.get("conversation_id"):
            row = await connection.fetchrow(
                """SELECT m.* FROM messages m WHERE m.conversation_id=$1 AND m.role='assistant'
                    AND m.metadata->>'run_id'=$2 ORDER BY m.created_at LIMIT 1""",
                target["conversation_id"], target["run_id"],
            )
            message = _message_data(row) if row else None
        return {**ResultDeliveryRepository._operation_summary(response), "run": target,
                "response": response, "message_patch": message}
