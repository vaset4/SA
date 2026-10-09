"""Agent episode 的真实多轮流采集与冻结入口。作者：xxx。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, cast

from app.middleware.trace_context import export_evidence_observations, reset_trace_ctx, set_trace_ctx
from evals.contracts import snapshot

from .contracts import AgentCase, agent_case_payload, freeze_agent_run
from .environment import AgentEnvironment


@dataclass(frozen=True, kw_only=True)
class TurnCapture:
    """保存一轮实际输入、流事件、最终状态和终态判断。"""

    turn: int
    input_snapshot: str
    events: tuple[dict[str, Any], ...]
    values_snapshot: str
    assistant_output: str
    terminal_state: str | None
    status: str
    evidence_observation_snapshot: str
    input_captured: bool = True
    error: dict[str, str] | None = None


async def collect_graph_turn(graph: Any, turn_input: dict[str, Any], *, turn: int, env: AgentEnvironment) -> dict[str, Any]:
    """参数为真实子图、当前轮输入、轮次和环境；返回可重放的实际流观察。"""
    events: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    actual_error: Exception | None = None
    trace = env.create_turn_trace(turn_input)
    token = set_trace_ctx(trace)
    env.record_access("research_graph", "stream", status="started")
    try:
        env.record_index_declaration()
        stream = getattr(graph, "astream", None)
        if not callable(stream):
            raise TypeError("research graph must expose astream")
        async for item in stream(deepcopy(turn_input), stream_mode=["updates", "values"]):
            mode, chunk = _stream_item(item)
            if mode == "updates":
                events.append({"mode": mode, "payload": _serializable(chunk)})
            elif mode == "values":
                values = _serializable(chunk)
        env.record_access("research_graph", "stream", status="succeeded")
    except Exception as exc:
        actual_error = exc
        env.record_access("research_graph", "stream", status="failed", error=exc)
    finally:
        evidence = export_evidence_observations(trace)
        reset_trace_ctx(token)
    return _capture_payload(
        turn, turn_input, events=events, values=values,
        evidence_observation=evidence, error=actual_error,
    )


def _stream_item(item: Any) -> tuple[str, Any]:
    """参数为 LangGraph 流事件；返回严格的模式和载荷。"""
    if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
        raise TypeError("research graph stream item must be a (mode, payload) tuple")
    if item[0] not in {"updates", "values"}:
        raise ValueError(f"unsupported research graph stream mode: {item[0]}")
    return item[0], item[1]


def _serializable(value: Any) -> dict[str, Any]:
    """参数为流载荷；返回 JSON 可序列化对象，不把结构化对象降级成字符串。"""
    value = _to_jsonable(value)
    if not isinstance(value, dict):
        raise TypeError("research graph stream payload must be an object")
    snapshot_value = snapshot(value)
    import json

    payload = json.loads(snapshot_value)
    if not isinstance(payload, dict):
        raise TypeError("research graph stream payload must be an object")
    return cast(dict[str, Any], payload)


def _to_jsonable(value: Any) -> Any:
    """参数为任意流载荷；递归转换 Pydantic/映射/序列，拒绝未知对象。"""
    from app.middleware.observability import RAGTrace

    if isinstance(value, RAGTrace):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"research graph stream payload contains unsupported value: {type(value).__name__}")


def _capture_payload(
    turn: int,
    turn_input: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    values: dict[str, Any],
    evidence_observation: dict[str, Any],
    input_captured: bool = True,
    error: Exception | None = None,
) -> dict[str, Any]:
    """参数为一轮采集数据；返回冻结前的结构化轮次记录。"""
    terminal_state = _terminal_state(values)
    assistant_output = values.get("answer")
    if not isinstance(assistant_output, str):
        assistant_output = ""
    return TurnCapture(
        turn=turn,
        input_snapshot=snapshot(turn_input),
        events=tuple(events),
        values_snapshot=snapshot(values),
        assistant_output=assistant_output,
        terminal_state=terminal_state,
        status="failed" if error is not None else "completed",
        evidence_observation_snapshot=snapshot(evidence_observation),
        input_captured=input_captured,
        error={"type": type(error).__name__, "message": str(error)} if error else None,
    ).__dict__.copy()


def _terminal_state(values: dict[str, Any]) -> str | None:
    """参数为一轮最终 values；按产品结构化终态字段返回状态。"""
    outcome = values.get("turn_outcome")
    if isinstance(outcome, dict):
        for key in ("state", "status", "outcome"):
            if isinstance(outcome.get(key), str) and outcome[key].strip():
                return str(outcome[key])
    for key in ("error_code", "planner_loop_decision"):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return str(value)
    return None


def _history_entry(turn: dict[str, Any]) -> dict[str, str]:
    """参数为已冻结轮次；返回下一轮允许使用的真实消息。"""
    values = _read_snapshot(turn["values_snapshot"])
    return {"role": "assistant", "content": turn["assistant_output"] or str(values.get("answer") or "")}


def _read_snapshot(value: str) -> dict[str, Any]:
    """参数为 JSON 快照；返回独立对象。"""
    import json

    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("turn snapshot must contain an object")
    return payload


async def run_episode(
    case: AgentCase,
    env: AgentEnvironment,
    *,
    run_mode: Literal["agent_fixture", "agent_live"] = "agent_fixture",
) -> dict[str, Any]:
    """参数为冻结案例、集中环境和执行模式；运行完整多轮 episode。"""
    if run_mode not in {"agent_fixture", "agent_live"}:
        raise ValueError(f"unsupported agent execution mode: {run_mode}")
    payload = agent_case_payload(case)
    history = payload["input"].get("chat_history")
    if not isinstance(history, list):
        raise ValueError("agent case chat_history must be a list")
    observations: list[dict[str, Any]] = []
    status: Literal["completed", "failed", "cancelled", "timed_out"] = "completed"
    for turn in payload["turns"]:
        user_message = turn["user_message"]
        observation = await _run_turn(
            case, env, user_message=user_message, history=history, turn=turn["turn"],
        )
        observations.append(observation)
        history = [*history, {"role": "user", "content": user_message}, _history_entry(observation)]
        if observation["status"] == "failed":
            status = "failed"
            break
    missing = env.validate_after_run()
    return freeze_agent_run(
        case, observations, run_mode=run_mode, environment=env.snapshot(),
        status=status, missing_fields=missing,
    )


async def _run_turn(
    case: AgentCase, env: AgentEnvironment, *, user_message: str,
    history: list[dict[str, Any]], turn: int,
) -> dict[str, Any]:
    """参数为案例、环境及当前消息；构造失败也留痕，未调用的输入不冒充采集成功。"""
    turn_input: dict[str, Any] = {}
    try:
        turn_input = env.build_turn_input(case, user_message=user_message, history=history)
        graph = env.create_graph()
    except Exception as exc:
        env.record_access("research_graph", "prepare_turn", status="failed", error=exc)
        return _capture_payload(
            turn, turn_input, events=[], values={}, evidence_observation={},
            input_captured=False, error=exc,
        )
    return await collect_graph_turn(graph, turn_input, turn=turn, env=env)


def run_episode_sync(
    case: AgentCase,
    env: AgentEnvironment,
    *,
    run_mode: Literal["agent_fixture", "agent_live"] = "agent_fixture",
) -> dict[str, Any]:
    """参数为冻结案例、环境和执行模式；在同步 CLI 中运行一个 Agent episode。"""
    return asyncio.run(run_episode(case, env, run_mode=run_mode))


EpisodeRunner = Callable[[AgentCase, AgentEnvironment], Awaitable[dict[str, Any]]]
