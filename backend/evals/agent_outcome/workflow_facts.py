"""【流程评测】【事实边界】读取冻结观察并保留条件指标覆盖。作者：xxx。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from evals.contracts import MetricResult, read_snapshot


@dataclass(frozen=True, kw_only=True)
class WorkflowTurn:
    """单轮实际输入与观察；字段缺失保留为缺口，不从标准答案补齐。"""

    number: int
    raw: dict[str, Any]
    inputs: dict[str, Any]
    values: dict[str, Any]
    capture: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    missing: tuple[str, ...]


def read_turn(raw: dict[str, Any]) -> WorkflowTurn:
    """参数为冻结轮次；返回独立快照副本及缺失字段，畸形快照明确报错。"""
    snapshots: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in ("input_snapshot", "values_snapshot", "evidence_observation_snapshot"):
        value = raw.get(name)
        if not isinstance(value, str):
            missing.append(name)
            snapshots[name] = {}
        else:
            snapshots[name] = read_snapshot(value)
    capture = snapshots["evidence_observation_snapshot"]
    events = capture.get("evidence_observations")
    if events is None:
        missing.append("evidence_observations")
        events = []
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("workflow evidence_observations must be a list of objects")
    if raw.get("input_captured") is False:
        missing.append("captured_execution_input")
    return WorkflowTurn(number=raw["turn"], raw=raw, inputs=snapshots["input_snapshot"],
                        values=snapshots["values_snapshot"], capture=capture,
                        events=tuple(events), missing=tuple(missing))


def event_rows(turns: tuple[WorkflowTurn, ...], stage: str) -> list[dict[str, Any]]:
    """参数为实际轮次及事件类型；返回带轮次、时间和事件引用的载荷副本。"""
    return [
        {**event["payload"], "turn": turn.number, "event_id": event.get("event_id"),
         "observed_at_ns": event.get("observed_at_ns"), "scope": event.get("scope")}
        for turn in turns for event in turn.events if event.get("stage") == stage
    ]


def ratio(name: str, numerator: int, denominator: int, *, reason: str) -> MetricResult:
    """参数为指标和真实计数；返回条件比例，无可判分母时保持缺失。"""
    if not denominator:
        return MetricResult(name=name, status="missing", reason=reason)
    return MetricResult(name=name, status="measured", value=numerator / denominator,
                        numerator=float(numerator), denominator=float(denominator), reason=reason)


def boolean_metric(name: str, passed: bool | None, *, reason: str) -> MetricResult:
    """参数为已证实判断或未知；返回单案例指标，未知不填成功或零分。"""
    if passed is None:
        return MetricResult(name=name, status="missing", reason=reason)
    return ratio(name, int(passed), 1, reason=reason)


def coverage(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    """参数为明细和状态键；返回含缺失状态的真实分布。"""
    return dict(Counter(str(row.get(key, "missing")) for row in rows))
