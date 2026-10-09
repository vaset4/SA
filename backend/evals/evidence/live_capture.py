"""实时评测的请求观察采集与失败快照。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.middleware.trace_context import (
    TraceContext,
    export_evidence_observations,
    record_evidence_observation,
    reset_trace_ctx,
    set_trace_ctx,
)
from evals.contracts import snapshot, validate_execution_input


@dataclass(frozen=True, kw_only=True)
class FailedLiveCapture:
    """保存失败请求的真实输入、已采集观察和原始错误身份，供独立报告冻结。"""

    request_id: str
    input_snapshot: str
    observation_snapshot: str
    error_type: str
    error_message: str


class CapturedRunFailure(RuntimeError):
    """携带失败快照的评测异常，原始业务异常通过 cause 链保留。"""

    def __init__(self, message: str, capture: FailedLiveCapture) -> None:
        """参数为展示消息与失败快照；构造仍可追溯原始运行的异常，不生成回答。"""
        super().__init__(message)
        self.capture = capture


async def capture_live_run(
    graph: Any, live_input: dict[str, Any], *, index_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """参数为实际图、隔离输入和运行前索引声明；返回未裁剪观察，失败保留快照与 cause。"""
    validate_execution_input(live_input)
    frozen_input = snapshot(live_input)
    trace = TraceContext(request_id=str(uuid4()), user_id=str(live_input.get("user_id") or ""))
    token = set_trace_ctx(trace)
    try:
        # 1. 【评测观察】【索引声明】声明只用于后续映射核对，不作为业务输入或匹配结论
        if index_snapshot_id is not None:
            record_evidence_observation("index_snapshot_declared", {
                "index_snapshot_id": index_snapshot_id,
                "source": "eval_declared_before_execution",
            })
        result = await graph.ainvoke(deepcopy(live_input))
        if not isinstance(result, dict):
            raise TypeError("live graph result must be an object")
        return {**result, **export_evidence_observations(trace)}
    except Exception as exc:
        # 2. 【评测观察】【失败冻结】失败时仍保留已经发生的观察，错误不能替代回答
        failed = FailedLiveCapture(
            request_id=trace.request_id, input_snapshot=frozen_input,
            observation_snapshot=snapshot(export_evidence_observations(trace)),
            error_type=type(exc).__name__, error_message=str(exc),
        )
        raise CapturedRunFailure(str(exc), failed) from exc
    finally:
        reset_trace_ctx(token)
