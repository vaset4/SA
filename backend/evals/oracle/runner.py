"""Oracle Actual/Oracle 配对执行与独立进程隔离。作者：xxx。"""

from __future__ import annotations

import importlib
import json
import multiprocessing
import queue
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast
from uuid import uuid4

from evals.contracts import snapshot

from .contracts import OraclePairRecord, OraclePairSpec, OracleSideRun
from .source import build_oracle_evidence
from evals.evidence.contracts import EvidenceBenchmark
from .attribution import attribute_pair

SideExecutor = Callable[..., Mapping[str, Any]]


def run_oracle_pair(
    spec: OraclePairSpec,
    *,
    benchmark: EvidenceBenchmark,
    actual_executor: SideExecutor,
    oracle_executor: SideExecutor,
    timeout_seconds: float = 60.0,
) -> OraclePairRecord:
    """参数为配对合同、Evidence 真值、两侧执行器和超时；返回冻结诊断记录。"""
    oracle_evidence = build_oracle_evidence(benchmark, index_snapshot_id=spec.index_snapshot_id)
    actual = _run_side(
        spec, side="actual", executor=actual_executor, oracle_evidence=(),
        timeout_seconds=timeout_seconds,
    )
    oracle = _run_side(
        spec, side="oracle", executor=oracle_executor, oracle_evidence=oracle_evidence,
        timeout_seconds=timeout_seconds,
    )
    attribution = attribute_pair(actual, oracle, spec=spec)
    return OraclePairRecord(
        pair=spec,
        actual=actual,
        oracle=oracle,
        attribution=attribution,
        control_hash=_control_hash(spec),
    )


def run_oracle_pair_isolated(
    spec: OraclePairSpec,
    *,
    benchmark: EvidenceBenchmark,
    actual_executor_path: str,
    oracle_executor_path: str,
    timeout_seconds: float = 60.0,
) -> OraclePairRecord:
    """参数为配对合同、真值和两个可导入执行器；分别在独立进程运行两侧。"""
    oracle_evidence = build_oracle_evidence(benchmark, index_snapshot_id=spec.index_snapshot_id)
    actual = _run_isolated_side(
        spec, side="actual", executor_path=actual_executor_path,
        oracle_evidence=(), timeout_seconds=timeout_seconds,
    )
    oracle = _run_isolated_side(
        spec, side="oracle", executor_path=oracle_executor_path,
        oracle_evidence=oracle_evidence, timeout_seconds=timeout_seconds,
    )
    return OraclePairRecord(
        pair=spec, actual=actual, oracle=oracle,
        attribution=attribute_pair(actual, oracle, spec=spec), control_hash=_control_hash(spec),
    )


def _run_side(
    spec: OraclePairSpec,
    *,
    side: str,
    executor: SideExecutor,
    oracle_evidence: tuple[dict[str, Any], ...],
    timeout_seconds: float,
) -> OracleSideRun:
    """参数为一侧合同和执行器；返回规范化运行记录，执行异常保留为失败。"""
    try:
        result = executor(
            json.loads(spec.input_snapshot), retrieval_mode=side,
            oracle_evidence=oracle_evidence,
            cache_namespace=_namespace(spec, side),
        )
        return _side_from_result(spec, side, result)
    except Exception as exc:
        return _failed_side(spec, side, type(exc).__name__, str(exc))


def _run_isolated_side(
    spec: OraclePairSpec,
    *,
    side: str,
    executor_path: str,
    oracle_evidence: tuple[dict[str, Any], ...],
    timeout_seconds: float,
) -> OracleSideRun:
    """参数为一侧合同、执行器路径和超时；返回独立进程的实际结果。"""
    if timeout_seconds <= 0:
        raise ValueError("oracle side timeout must be positive")
    context = multiprocessing.get_context("spawn")
    channel: Any = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            json.loads(spec.input_snapshot), side, executor_path,
            list(oracle_evidence), _namespace(spec, side), channel,
        ),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return _failed_side(spec, side, "TimeoutError", "oracle side timed out")
    try:
        result = channel.get(timeout=1)
    except queue.Empty:
        return _failed_side(spec, side, "ChildProcessError", f"oracle child exited with code {process.exitcode}")
    if result.get("status") != "completed":
        return _failed_side(spec, side, str(result.get("error_type") or "ChildProcessError"), str(result.get("error_message") or "oracle child failed"))
    return _side_from_result(spec, side, result.get("result"))


def _worker(
    input_payload: dict[str, Any], side: str, executor_path: str,
    oracle_evidence: list[dict[str, Any]], cache_namespace: str, channel: Any,
) -> None:
    """参数为可序列化一侧输入和执行器路径；返回真实执行结果或错误。"""
    try:
        executor = _load_executor(executor_path)
        result = executor(
            input_payload, retrieval_mode=side,
            oracle_evidence=tuple(oracle_evidence), cache_namespace=cache_namespace,
        )
        channel.put({"status": "completed", "result": dict(result)})
    except Exception as exc:
        channel.put({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})


def _load_executor(path: str) -> SideExecutor:
    """参数为 module:attribute 路径；返回可调用执行器。"""
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("oracle executor path must use module:attribute format")
    executor = getattr(importlib.import_module(module_name), attribute)
    if not callable(executor):
        raise TypeError("oracle executor must be callable")
    return cast(SideExecutor, executor)


def _side_from_result(spec: OraclePairSpec, side: str, result: Mapping[str, Any] | None) -> OracleSideRun:
    """参数为执行结果；补齐一侧合同身份并校验观察快照。"""
    payload = dict(result or {})
    observation = payload.get("observation")
    if not isinstance(observation, Mapping):
        observation = payload.get("observation_snapshot")
    if isinstance(observation, str):
        observation_snapshot = observation
    elif isinstance(observation, Mapping):
        observation_snapshot = snapshot(dict(observation))
    else:
        observation_snapshot = snapshot({})
    status = str(payload.get("status") or "completed")
    if status not in {"completed", "failed", "invalid_input"}:
        status = "invalid_input"
    return OracleSideRun(
        side=cast(Literal["actual", "oracle"], side),
        run_id=str(payload.get("run_id") or uuid4()),
        run_mode=cast(Literal["live", "oracle"], "oracle" if side == "oracle" else "live"),
        retrieval_mode=cast(Literal["actual", "oracle"], "oracle" if side == "oracle" else "actual"),
        input_snapshot=spec.input_snapshot, observation_snapshot=observation_snapshot,
        cache_namespace=_namespace(spec, side),
        status=cast(Literal["completed", "failed", "invalid_input"], status),
        missing_fields=tuple(str(item) for item in payload.get("missing_fields", ()) if str(item)),
        error_type=str(payload["error_type"]) if payload.get("error_type") else None,
        error_message=str(payload["error_message"]) if payload.get("error_message") else None,
    )


def _failed_side(spec: OraclePairSpec, side: str, error_type: str, error_message: str) -> OracleSideRun:
    """参数为一侧错误身份；返回显式失败观察，不生成答案。"""
    return _side_from_result(spec, side, {
        "status": "failed", "error_type": error_type, "error_message": error_message,
        "observation": {"error": {"type": error_type, "message": error_message}},
    })


def _namespace(spec: OraclePairSpec, side: str) -> str:
    """参数为配对合同和一侧名称；返回预先固定的缓存命名空间。"""
    return spec.actual_cache_namespace if side == "actual" else spec.oracle_cache_namespace


def _control_hash(spec: OraclePairSpec) -> str:
    """参数为配对合同；返回控制条件快照校验和。"""
    from evals.contracts import snapshot_hash

    return snapshot_hash(spec.control_snapshot)
