"""Agent Outcome 评测的案例、运行和输入隔离合同。作者：xxx。"""

from __future__ import annotations

from typing import Any, Literal, Mapping, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, field_validator

from evals.contracts import (
    EvalCase,
    RunRecord,
    snapshot,
    validate_execution_input,
)

AGENT_SCHEMA_VERSION: Literal["agent-outcome-v1"] = "agent-outcome-v1"
AgentRunMode = Literal["agent_fixture", "agent_live", "replay"]
AgentReviewStatus = Literal["candidate", "reviewed"]
AgentExecutionStatus = Literal["not_run", "running", "completed", "failed"]
_AGENT_TRUTH_KEYS = frozenset({
    "required_outcomes", "forbidden_behaviors", "allowed_terminal_states",
    "task_completion_definition", "fault_injection", "evidence_refs",
    "evidence_benchmark", "evidence_targets",
    "reference_answer", "answerability", "negative_reason", "absence_review",
    "metric_applicability",
})


class AgentTurn(BaseModel):
    """冻结单轮用户消息；预期行为只属于真值快照。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    turn: int
    user_message: str


class AgentCase(BaseModel):
    """冻结 Agent 场景的输入、真值和声明环境。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["agent-outcome-v1"] = AGENT_SCHEMA_VERSION
    case_id: str
    case_version: str
    title_zh: str
    task_family: str
    review_status: AgentReviewStatus
    execution_status: AgentExecutionStatus
    input_snapshot: str
    truth_snapshot: str
    environment_snapshot: str
    turn_count: int

    @field_validator("input_snapshot", "truth_snapshot", "environment_snapshot")
    @classmethod
    def validate_snapshot(cls, value: str) -> str:
        """参数为 JSON 对象快照；返回排序后的不可变文本。"""
        payload = _read_object(value)
        return snapshot(payload)


class AgentRun(BaseModel):
    """冻结一次 Agent episode 的实际输入、观察、环境和错误。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["agent-outcome-v1"] = AGENT_SCHEMA_VERSION
    run_id: str
    case_id: str
    case_version: str
    run_mode: AgentRunMode
    status: Literal["completed", "failed", "cancelled", "timed_out"]
    input_snapshot: str
    observation_snapshot: str
    environment_snapshot: str
    missing_fields: tuple[str, ...] = ()

    @field_validator("input_snapshot", "observation_snapshot", "environment_snapshot")
    @classmethod
    def validate_run_snapshot(cls, value: str) -> str:
        """参数为运行快照；验证对象形状并返回规范文本。"""
        return snapshot(_read_object(value))


def _read_object(value: str) -> dict[str, Any]:
    """参数为 JSON 文本；返回对象，拒绝数组和标量快照。"""
    import json

    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("agent snapshot must contain an object")
    return payload


def validate_agent_execution_input(payload: Any, *, path: str = "input") -> None:
    """参数为实际 Agent 输入；递归拒绝真值、预期和故障字段。"""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in _AGENT_TRUTH_KEYS:
                raise ValueError(f"agent truth is not allowed in execution input: {path}.{key}")
            validate_execution_input({key: value}, path=f"{path}.{key}")
            validate_agent_execution_input(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            validate_agent_execution_input(value, path=f"{path}[{index}]")


def load_agent_case(record: Mapping[str, Any]) -> AgentCase:
    """参数为原始 Agent 场景；分离可执行输入、评分真值和声明环境。"""
    case_id = _required_text(record, "id")
    raw_input = _required_object(record, "input")
    raw_initial = _required_object(record, "initial_conditions")
    raw_turns = record.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"agent case {case_id} requires turns")
    turns = _load_turns(case_id, raw_turns)
    execution_input = {
        "question": _required_text(raw_input, "question_zh"),
        "paper_keys": _required_string_list(raw_input, "paper_keys"),
        "chat_history": raw_initial.get("history", []),
        "focused_paper_key": raw_initial.get("focused_paper_key"),
        **{key: raw_input[key] for key in (
            "project_scope", "structured_query", "user_id", "project_id", "paper_ids",
        ) if key in raw_input},
    }
    validate_agent_execution_input(execution_input)
    truth = {key: record.get(key) for key in _AGENT_TRUTH_KEYS if key in record}
    environment = {
        "initial_conditions": raw_initial,
        "fault_injection": record.get("fault_injection"),
        "turns": [turn.model_dump(mode="json") for turn in turns],
    }
    return AgentCase(
        case_id=case_id,
        case_version=str(record.get("case_version") or record.get("dataset_version") or AGENT_SCHEMA_VERSION),
        title_zh=_required_text(record, "title_zh"),
        task_family=_required_text(record, "task_family"),
        review_status=cast(AgentReviewStatus, str(record.get("review_status") or "candidate")),
        execution_status=cast(AgentExecutionStatus, str(record.get("execution_status") or "not_run")),
        input_snapshot=snapshot(execution_input),
        truth_snapshot=snapshot(truth),
        environment_snapshot=snapshot(environment),
        turn_count=len(turns),
    )


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """参数为对象和字段名；返回非空文本字段。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"agent case field {key} must be non-empty text")
    return value.strip()


def _required_object(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    """参数为对象和字段名；返回独立字典副本。"""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"agent case field {key} must be an object")
    return dict(value)


def _required_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    """参数为对象和列表字段名；返回去空白后的字符串列表。"""
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"agent case field {key} must be a non-empty string list")
    return [item.strip() for item in value]


def _load_turns(case_id: str, raw_turns: list[Any]) -> tuple[AgentTurn, ...]:
    """参数为案例编号和原始轮次；返回连续且不重复的用户轮次。"""
    turns: list[AgentTurn] = []
    expected_number = 1
    for raw in raw_turns:
        if not isinstance(raw, Mapping) or raw.get("turn") != expected_number:
            raise ValueError(f"agent case {case_id} turn sequence is invalid")
        turns.append(AgentTurn(turn=expected_number, user_message=_required_text(raw, "user_message")))
        expected_number += 1
    return tuple(turns)


def freeze_agent_run(
    case: AgentCase,
    observations: list[dict[str, Any]],
    *,
    run_mode: Literal["agent_fixture", "agent_live"],
    environment: dict[str, Any],
    status: Literal["completed", "failed", "cancelled", "timed_out"] = "completed",
    missing_fields: tuple[str, ...] = (),
    run_id: str | None = None,
) -> dict[str, Any]:
    """参数为冻结案例、实际轮次和环境；返回通用合同可复用的运行记录。"""
    observation = {"turns": observations, "status": status}
    first = observations[0] if observations else {}
    captured = isinstance(first.get("input_snapshot"), str) and first.get("input_captured") is not False
    actual_input = str(first["input_snapshot"]) if captured else case.input_snapshot
    missing = tuple(dict.fromkeys((*missing_fields, *(("captured_execution_input",) if not captured else ()))))
    agent_run = AgentRun(
        run_id=run_id or str(uuid4()), case_id=case.case_id, case_version=case.case_version,
        run_mode=run_mode, status=status, input_snapshot=actual_input,
        observation_snapshot=snapshot(observation), environment_snapshot=snapshot(environment),
        missing_fields=missing,
    )
    generic_case = EvalCase(
        case_id=case.case_id, case_version=case.case_version, source_type="agent_scenario",
        input_snapshot=case.input_snapshot, truth_snapshot=case.truth_snapshot,
    )
    generic_run = RunRecord(
        run_id=agent_run.run_id, case_id=case.case_id, case_version=case.case_version,
        run_mode=run_mode, input_snapshot=actual_input,
        observation_snapshot=agent_run.observation_snapshot,
        fixture_snapshot=agent_run.environment_snapshot if run_mode == "agent_fixture" else None,
        provenance="captured" if captured else "legacy", missing_fields=missing,
    )
    return {
        "agent_case": case.model_dump(mode="json"),
        "agent_run": agent_run.model_dump(mode="json"),
        "eval_case": generic_case.model_dump(mode="json"),
        "run_record": generic_run.model_dump(mode="json"),
    }


def agent_case_payload(case: AgentCase) -> dict[str, Any]:
    """参数为冻结案例；返回运行器使用的独立输入、轮次、真值和环境投影。"""
    environment = _read_object(case.environment_snapshot)
    truth = _read_object(case.truth_snapshot)
    turns = environment.get("turns")
    if not isinstance(turns, list):
        raise ValueError("agent case turn snapshot is missing")
    return {
        "input": _read_object(case.input_snapshot),
        "turns": turns,
        "truth": truth,
        "environment": environment,
    }


def agent_record_contracts(record: Mapping[str, Any]) -> tuple[AgentCase, AgentRun]:
    """参数为 Agent 三层冻结记录；校验案例和运行身份后返回模型。"""
    case = AgentCase.model_validate(record["agent_case"])
    run = AgentRun.model_validate(record["agent_run"])
    if (case.case_id, case.case_version) != (run.case_id, run.case_version):
        raise ValueError("agent case/run identity mismatch")
    return case, run
