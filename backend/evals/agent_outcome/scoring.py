"""基于真实原文及执行事实判断任务完成、失败处理与恢复。作者：xxx。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evals.contracts import MetricResult, build_grading, read_snapshot

from .contracts import AgentCase, AgentRun, agent_record_contracts
from .evidence import (
    EpisodeFacts, collect_episode_facts, completed_terminal, evaluate_predicates, honest_failure,
)

AGENT_GRADER_VERSION = "agent-outcome-v3"


def grade_agent_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """参数为三层冻结记录；按原文和执行事实返回五项既有测量，不改写历史评分。"""
    case, run = agent_record_contracts(record)
    facts = collect_episode_facts(case, run)
    predicates = evaluate_predicates(case, facts)
    task = measure_task_success(case, run, facts=facts, predicates=predicates)
    metrics = [
        task, measure_failure_handling(run, facts), measure_recovery(run, facts, task=task),
        facts.scope_adherence, facts.delivered.metric,
    ]
    return build_grading(dict(record), metrics, grader="agent_outcome", grader_version=AGENT_GRADER_VERSION)


def measure_task_success(
    case: AgentCase, run: AgentRun, *, facts: EpisodeFacts, predicates: tuple[MetricResult, ...],
) -> MetricResult:
    """参数为案例、运行及逐项事实；必要目标、来源、范围和允许终态共同决定完成。"""
    name = "task_success"
    if read_snapshot(case.truth_snapshot).get("answerability") == "insufficient_evidence":
        return MetricResult(name=name, status="not_measured", reason="declared_insufficient_evidence_no_positive_task_denominator")
    if not facts.turns or run.missing_fields:
        return MetricResult(name=name, status="missing", reason="actual_turn_or_environment_missing")
    if run.status != "completed":
        return _measured(name, False, "actual_execution_failed")
    if facts.turns[-1].terminal is None:
        return MetricResult(name=name, status="missing", reason="terminal_state_missing")
    gap = next((item for item in predicates if item.status in {"invalid_input", "missing"}), None)
    if gap is not None:
        return gap.model_copy(update={"name": name})
    passed = all(item.status == "measured" and item.value == 1.0 for item in predicates)
    passed = passed and completed_terminal(case, run, facts)
    passed = passed and facts.scope_adherence.value != 0.0 and not facts.delivered.issues
    return _measured(name, passed, "all_required_outcomes_and_terminal_state" if passed else "required_outcome_or_terminal_failed")


def measure_failure_handling(run: AgentRun, facts: EpisodeFacts) -> MetricResult:
    """参数为真实运行和故障事实；仅实际故障进入分母，诚实失败必须对用户可见。"""
    name = "failure_handling_correctness"
    if not facts.fault_observed:
        return MetricResult(name=name, status="not_measured", reason="no_actual_fault_observed")
    if run.missing_fields:
        return MetricResult(name=name, status="missing", reason="runtime_environment_observation_missing")
    return _measured(name, honest_failure(facts), "visible_failure_and_uncompleted_goal_preserved")


def measure_recovery(run: AgentRun, facts: EpisodeFacts, *, task: MetricResult) -> MetricResult:
    """参数为实际运行、故障关系和任务结果；要求关联补查、成功重执行及任务完成。"""
    name = "recovery_success"
    if not facts.recoverable_fault:
        return MetricResult(name=name, status="not_measured", reason="no_actual_recoverable_fault")
    if run.missing_fields or task.status in {"missing", "invalid_input"}:
        return MetricResult(name=name, status="missing", reason="recovery_completion_observation_missing")
    return _measured(name, facts.recovery_observed and task.value == 1.0, "related_action_reexecuted_and_task_completed")


def _measured(name: str, passed: bool, reason: str) -> MetricResult:
    """参数为指标名、判断和原因；返回一项保留分母的已测结果。"""
    value = float(passed)
    return MetricResult(name=name, status="measured", value=value, numerator=value, denominator=1.0, reason=reason)
