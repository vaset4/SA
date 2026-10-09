"""Agent episode 的独立进程调度边界。作者：xxx。"""

from __future__ import annotations

import importlib
import multiprocessing
import queue
from dataclasses import dataclass
from typing import Any, Literal, cast

from .contracts import AgentCase
from .environment import AgentEnvironment
from .runner import run_episode_sync


@dataclass(frozen=True, kw_only=True)
class IsolatedProcessResult:
    """保存独立进程的退出状态、运行记录或明确错误。"""

    status: str
    record: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


def run_isolated_episode(
    case: AgentCase,
    *,
    graph_factory_path: str,
    timeout_seconds: float = 60.0,
    run_mode: str = "agent_fixture",
) -> IsolatedProcessResult:
    """参数为冻结案例、可导入图工厂路径、超时和模式；在独立进程执行一次 episode。"""
    if timeout_seconds <= 0:
        raise ValueError("episode timeout must be positive")
    context = multiprocessing.get_context("spawn")
    channel: Any = context.Queue()
    process = context.Process(
        target=_worker,
        args=(case.model_dump(mode="json"), graph_factory_path, run_mode, channel),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return IsolatedProcessResult(status="timed_out", error_type="TimeoutError", error_message="agent episode timed out")
    try:
        result = channel.get(timeout=1)
    except queue.Empty:
        return IsolatedProcessResult(status="failed", error_type="ChildProcessError", error_message=f"agent episode exited with code {process.exitcode}")
    if result.get("status") == "completed":
        return IsolatedProcessResult(status="completed", record=result["record"])
    return IsolatedProcessResult(status="failed", error_type=result.get("error_type"), error_message=result.get("error_message"))


def _worker(case_payload: dict[str, Any], factory_path: str, run_mode: str, channel: Any) -> None:
    """参数为可序列化案例、工厂路径、运行模式和队列；执行并返回真实结果。"""
    try:
        case = AgentCase.model_validate(case_payload)
        factory = _load_factory(factory_path)
        environment = AgentEnvironment(graph_factory=factory)
        mode = cast(Literal["agent_fixture", "agent_live"], run_mode)
        record = run_episode_sync(case, environment, run_mode=mode)
        channel.put({"status": "completed", "record": record})
    except Exception as exc:
        channel.put({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})


def _load_factory(path: str) -> Any:
    """参数为 module:attribute 工厂路径；返回可调用对象，格式错误显式失败。"""
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("graph factory path must use module:attribute format")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("graph factory must be callable")
    return factory
