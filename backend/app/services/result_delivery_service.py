"""【研究交付】【保存服务】暂存、原子提交和显式补存，作者：xxx，2026-10-04 13:00:00。"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from app.models.result_delivery_contracts import DELIVERY_METADATA_FIELDS, DeliveryArtifact, DeliveryConflict, PendingDelivery
from app.repositories.result_delivery_repository import ResultDeliveryRepository

logger = logging.getLogger(__name__)



def read_delivery_response(run: dict) -> dict:
    """【研究交付】【结果回读】参数为 owner 已校验运行；返回整份提交投影，历史缺失事实保持缺失。"""
    metadata = run.get("metadata") or {}
    response = {**metadata, "answer": run.get("answer"), "run_id": run["run_id"], "run_status": run["status"]}
    if metadata.get("response_has_report"):
        response["report"] = run.get("answer")
    response.pop("response_has_report", None)
    response["execution_stopped_at"] = run.get("execution_stopped_at")
    response["execution_started_at"] = run.get("execution_started_at")
    response["conversation_id"] = run.get("conversation_id")
    if metadata.get("persistence_status") == "saved":
        response["delivery_revision"] = run["delivery_revision"]
    return response


async def read_current_delivery_response(run: dict) -> dict:
    """【研究交付】【执行事实】参数为已授权目标；返回当前整份结果及实际产出操作的停止事实。"""
    from app.services import rag_run_service

    response = read_delivery_response(run)
    producer_id = response.get("operation_run_id") or response.get("producer_run_id")
    if producer_id and producer_id != run["run_id"]:
        producer = await rag_run_service.get_run(user_id=run["user_id"], run_id=producer_id)
        if producer is None:
            raise LookupError("已提交结果的执行操作不存在")
        # 1. 【研究交付】【停止快照】先确认停止再读目标，避免候选刚完成时拼出旧pending与新停止事实
        if producer.get("execution_stopped_at"):
            latest = await rag_run_service.get_run(user_id=run["user_id"], run_id=run["run_id"])
            if latest is None:
                raise LookupError("已提交结果不存在")
            response = read_delivery_response(latest)
            current_operation = response.get("operation_run_id") or response.get("producer_run_id")
            if current_operation != producer_id:
                response["execution_started_at"] = None
                response["execution_stopped_at"] = None
                return response
        response["execution_started_at"] = producer.get("execution_started_at")
        response["execution_stopped_at"] = producer.get("execution_stopped_at")
    return response


def assemble_delivery(*, run: dict, result: dict, response: dict | None = None,
                      session_status: str | None = None, attempt_summary: dict | None = None) -> PendingDelivery:
    """参数为一次服务器生成结果和生产身份；返回不可混版产物，保持 Markdown 原字节内容。"""
    answer = result.get("answer")
    if answer is None and not result.get("final_delivery_report"):
        answer = result.get("report", "")
    if not isinstance(answer, str):
        raise ValueError("最终产物缺少同次正文")
    report = result.get("final_delivery_report")
    attempt = result.get("final_generation_attempt")
    if report is not None and (not isinstance(attempt, dict) or report.get("attempt_id") != attempt.get("attempt_id")):
        raise ValueError("最终正文的交付报告与生成尝试不匹配")
    if report is not None and report.get("content_review") is not None:
        from app.services.internal_research.research_answer_review import validate_review_delivery
        validate_review_delivery(answer, report, research_state=result)
    from app.services.internal_research.research_memory_writeback import prepare_committed_memory_writeback

    metadata = {key: result[key] for key in DELIVERY_METADATA_FIELDS if key in result}
    metadata["memory_writeback"] = prepare_committed_memory_writeback(result)
    projection = {**(response or metadata), "answer": answer, "memory_writeback": metadata["memory_writeback"]}
    # 1. 【研究交付】【同次事实】支持、报告和引用只能取整份产物，响应旧字段不能覆盖它们
    for key in ("final_delivery_report", "final_citation_bindings", "final_generation_attempt", "final_delivery_applicability"):
        projection[key] = result.get(key)
    return PendingDelivery(
        target_run_id=run.get("target_run_id") or run["run_id"], producer_run_id=run["run_id"],
        expected_delivery_revision=int(run["expected_delivery_revision"] if run.get("expected_delivery_revision") is not None
                                       else run.get("delivery_revision") or 0),
        session_id=run.get("session_id"),
        session_run_id=run.get("session_run_id") or (run["run_id"] if session_status is not None else None),
        artifact=DeliveryArtifact(answer=answer, metadata=metadata, response=projection,
                                  session_status=session_status, attempt_summary=attempt_summary),
    )


async def deliver_result(*, user_id: str, run: dict, result: dict, response: dict | None = None,
                         session_status: str | None = None, repository=None, post_commit=None,
                         attempt_summary: dict | None = None) -> dict[str, Any]:
    """参数为实际产物、执行身份与保存端口；返回已保存或明确未保存的整体响应。

    暂存失败不承诺关闭页面后仍可恢复；候选记忆回调只在成功提交后执行。
    """
    repo = repository or ResultDeliveryRepository()
    pending = assemble_delivery(run=run, result=result, response=response,
                                session_status=session_status, attempt_summary=attempt_summary)
    staged = False
    try:
        await repo.stage(user_id=user_id, pending=pending)
        staged = True
        committed = await repo.commit(user_id=user_id, operation_run_id=run["run_id"], target_run_id=pending.target_run_id)
    except DeliveryConflict:
        raise
    except Exception as exc:
        return await _failed_delivery(repo, user_id=user_id, run=run, pending=pending, staged=staged, error=exc)
    return await _after_commit(committed, post_commit)


async def persist_result(*, user_id: str, run: dict, repository=None, post_commit=None) -> dict:
    """参数为新补存操作；提交服务端暂存原产物，无模型、检索或浏览器正文输入。"""
    if run.get("operation_kind") != "persist_only" or not run.get("target_run_id"):
        raise DeliveryConflict("重试保存需要指向原答案的补存操作")
    repo = repository or ResultDeliveryRepository()
    committed = await repo.commit(user_id=user_id, operation_run_id=run["run_id"], target_run_id=run["target_run_id"])
    return await _after_commit(committed, post_commit)


async def retry_saved_result(*, user_id: str, target_run_id: str, idempotency_key: str | None = None,
                             conversation_id: str | None = None, repository=None, run_repository=None,
                             executor=None, post_commit=None) -> dict:
    """【研究交付】【补存命令】参数为原目标和动作标识；领取轻量操作后补存，不执行模型。"""
    from app.config import get_settings
    from app.services import rag_run_service
    from app.services.rag_run_executor_service import get_rag_run_executor_registry

    target = await rag_run_service.get_run(user_id=user_id, run_id=target_run_id, repository=run_repository)
    if target is None or (conversation_id is not None and target.get("conversation_id") != conversation_id):
        raise LookupError("待保存结果不存在")
    timeout = float(get_settings().graph_timeout_seconds)
    operation, created = await rag_run_service.create_operation(
        user_id=user_id, conversation_id=target.get("conversation_id"), session_id=target.get("session_id"),
        thread_id=None, idempotency_key=idempotency_key or str(uuid4()), query=str(target.get("query") or ""),
        deadline_at=rag_run_service.deadline_after(timeout), operation_kind="persist_only", target_run_id=target_run_id,
        expected_delivery_revision=int(target.get("delivery_revision") or 0),
        request_payload={"target_run_id": target_run_id, "operation_kind": "persist_only"}, repository=run_repository,
    )
    if not created:
        if operation["status"] == "completed":
            return await persist_result(user_id=user_id, run=operation, repository=repository, post_commit=post_commit)
        return {"operation": operation, "run": target, "target_run_id": target_run_id,
                "operation_run_id": operation["run_id"], "delivery_revision": target.get("delivery_revision", 0)}
    return await (executor or get_rag_run_executor_registry()).execute(
        user_id=user_id, run_id=operation["run_id"], timeout_seconds=timeout,
        runner_factory=lambda: persist_result(user_id=user_id, run=operation, repository=repository, post_commit=post_commit),
    )


async def _failed_delivery(repo, *, user_id: str, run: dict, pending: PendingDelivery, staged: bool, error: Exception) -> dict:
    """参数为失败阶段和原产物；返回真实保存错误并保留正文，终态写入失败另行暴露。"""
    logger.error("【研究交付】【保存失败】run=%s staged=%s", run["run_id"], staged, exc_info=error)
    failure = {"target_run_id": pending.target_run_id, "operation_run_id": run["run_id"],
               "operation_status": "failed",
               "delivery_revision": pending.expected_delivery_revision,
               "persistence_status": "pending" if staged else "not_staged", "persistence_error": str(error),
               "can_retry_save": staged, "run_status": "failed"}
    response = {**pending.artifact.response, **failure, "answer": pending.artifact.answer, "run_id": pending.target_run_id}
    failed_run = run
    try:
        terminal = failure if run.get("target_run_id") else response
        failed_run = await repo.fail(user_id=user_id, operation_run_id=run["run_id"], error=str(error),
                                     payload={"type": "error", **terminal})
    except Exception as terminal_error:
        logger.error("【研究交付】【终态未保存】run=%s", run["run_id"], exc_info=terminal_error)
        response["terminal_persist_error"] = str(terminal_error)
        response["run_status"] = run["status"]
    return {**failure, "run": failed_run if not run.get("target_run_id") else None,
            "operation": failed_run, "response": response, "message_patch": None}


async def _after_commit(committed: dict, post_commit) -> dict:
    """参数为已提交结果和可选候选写回端口；返回保存回执，候选错误不修改答案状态。"""
    if post_commit is None:
        from app.services.internal_research.research_memory_writeback import after_result_committed

        post_commit = after_result_committed
    try:
        outcome = await post_commit(committed)
    except Exception as exc:
        logger.error("【研究交付】【提交后候选失败】run=%s", committed.get("target_run_id"), exc_info=True)
        outcome = {"status": "failed", "candidate_ids": [], "reason_codes": ["post_commit_processing_failed"],
                   "retryable": not isinstance(exc, ValueError), "error_kind": type(exc).__name__, "error": str(exc)}
    # 1. 【研究交付】【独立候选状态】只补充本提交版本的候选事实，不改变正文、引用或保存状态
    response = {**committed["response"], "memory_writeback": outcome}
    run = committed.get("run")
    message = committed.get("message_patch")
    return {**committed, "response": response, "post_commit": outcome,
            "run": {**run, "metadata": {**run["metadata"], "memory_writeback": outcome}} if run else None,
            "message_patch": {**message, "memory_writeback": outcome} if message else None}
