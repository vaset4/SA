"""【流程评测】【独立判定】工作流与结构正确性侧车，不修改旧任务质量评分。作者：xxx。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from evals.contracts import MetricResult, build_grading, read_snapshot, record_contracts

from .contracts import agent_record_contracts
from .workflow_delivery import DeliveryAssessment, assess_delivery, grade_false_success
from .workflow_execution import grade_tools, plan_checks, route_checks, unit_checks
from .workflow_facts import WorkflowTurn, boolean_metric, read_turn
from .workflow_faults import collect_faults, efficiency_facts
from .workflow_structure import grade_structure
from .workflow_scope import grade_scope

WORKFLOW_GRADER_VERSION = "agent-workflow-v2"
WORKFLOW_GRADER_VERSIONS = ("agent-workflow-v1", WORKFLOW_GRADER_VERSION)
_EARLY_FAILURES = frozenset({
    "planner_depth_assessment_failed", "planner_context_load_failed",
    "memory_read_failed", "planner_structured_output_failed",
})
_FAILURE_TERMINALS = frozenset({"failed", "incomplete", "terminate_fail", "terminate_honest", *_EARLY_FAILURES})
_KNOWN_TERMINALS = frozenset({"completed", "converged", *_FAILURE_TERMINALS})


def _terminal_check(turn: WorkflowTurn) -> tuple[list[dict[str, Any]], list[str]]:
    """参数为真实终态；验证决策、失败可见性和用户终态一致，不评价回答内容。"""
    terminal = turn.raw.get("terminal_state")
    if terminal is None:
        return [], ["terminal_state_missing"]
    violations: list[dict[str, Any]] = []
    if terminal not in _KNOWN_TERMINALS:
        violations.append({"turn": turn.number, "code": "unknown_terminal_state", "terminal": terminal})
    outcome = turn.values.get("turn_outcome", {})
    if outcome.get("status") and outcome["status"] != terminal:
        violations.append({"turn": turn.number, "code": "captured_terminal_mismatch"})
    decision = str(turn.values.get("planner_loop_decision") or "")
    expected = {"terminate_fail": "failed", "terminate_honest": "incomplete"}.get(decision)
    if expected and terminal not in {expected, decision, *_EARLY_FAILURES}:
        violations.append({"turn": turn.number, "code": "decision_terminal_mismatch"})
    if terminal in {"completed", "converged"}:
        if turn.raw.get("error") or turn.values.get("error_code"):
            violations.append({"turn": turn.number, "code": "error_reported_as_completed"})
        if turn.values.get("source_status", {}).get("planner") == "partial":
            violations.append({"turn": turn.number, "code": "partial_evidence_reported_as_completed"})
    return violations, []


def _turn_rules(turn: WorkflowTurn) -> dict[str, Any]:
    """参数为一轮完整观察；汇总独立检查及覆盖，已证实违规优先于缺失。"""
    violations: list[dict[str, Any]] = []
    missing = list(turn.missing)
    for checker in (route_checks, _terminal_check, plan_checks, unit_checks):
        issues, gaps = checker(turn)
        violations.extend(issues)
        missing.extend(gaps)
    missing.extend(f"capture_error:{row.get('stage')}" for row in turn.capture.get("evidence_capture_errors", []))
    status = "failed" if violations else "missing" if missing else "passed"
    return {"turn": turn.number, "terminal_state": turn.raw.get("terminal_state"),
            "execution_status": turn.raw.get("status"), "rule_status": status,
            "violations": violations, "missing": sorted(set(missing))}


def _cross_boundary_violations(turns: tuple[WorkflowTurn, ...], structure: dict[str, Any]) -> list[dict[str, Any]]:
    """参数为真实终态及结构操作；识别外部失败后伪成功和非法业务输出放行。"""
    by_turn = {turn.number: turn for turn in turns}
    violations: list[dict[str, Any]] = []
    for row in structure["operations"]:
        turn = by_turn[row["turn"]]
        if row["final_status"] in {"api_blocked", "failed"} and turn.raw.get("terminal_state") in {"completed", "converged"} and not _nonblocking_failure_disclosed(turn, row):
            violations.append({"turn": row["turn"], "code": "failed_operation_reported_as_completed",
                               "operation_id": row["operation_id"]})
    for row in structure["business_validations"]:
        if row.get("valid") is False and row.get("accepted") is True:
            violations.append({"turn": row["turn"], "code": "invalid_business_output_accepted",
                               "event_id": row.get("event_id")})
    return violations


def _nonblocking_failure_disclosed(turn: WorkflowTurn, operation: dict[str, Any]) -> bool:
    """参数为本轮与失败操作；只接受回答后记忆提议失败的现役显式四态合同。"""
    if operation.get("schema") != "TurnMemoryProposalBatch":
        return False
    writeback = turn.values.get("memory_writeback", {})
    capabilities = turn.values.get("turn_outcome", {}).get("capability_results", {})
    return (writeback.get("status") == "failed" and bool(writeback.get("error_kind"))
            and writeback.get("candidate_ids") == [] and capabilities.get("write_back_project_memory") == "failed")


def _completion(turns: tuple[WorkflowTurn, ...], *, declared: int, status: str) -> tuple[MetricResult, str]:
    """参数为实际轮次、事前声明数及执行状态；任何剩余轮次均不折算完整案例。"""
    if status in {"cancelled", "timed_out", "failed"}:
        return boolean_metric("workflow_completion", False, reason=status), status
    if len(turns) != declared:
        return boolean_metric("workflow_completion", False, reason="declared_turns_not_completed"), "partial"
    if any(turn.raw.get("terminal_state") is None for turn in turns):
        return boolean_metric("workflow_completion", None, reason="terminal_observation_missing"), "missing"
    terminal = turns[-1].raw["terminal_state"]
    assessment = turns[-1].values.get("turn_depth_assessment", {})
    if terminal in {"incomplete", "terminate_honest"} and assessment.get("needs_clarification") is True:
        return boolean_metric("workflow_completion", False, reason="waiting_clarification"), "waiting_clarification"
    completed = terminal in {"completed", "converged"}
    earlier_failed = any(turn.raw.get("terminal_state") in {"failed", "terminate_fail", *_EARLY_FAILURES} for turn in turns)
    completed = completed and not earlier_failed
    return boolean_metric("workflow_completion", completed, reason="all_declared_turns_and_actual_terminal"), "completed" if completed else "incomplete"


def _error_handling(
    turns: tuple[WorkflowTurn, ...], faults: list[dict[str, Any]], *, structure: dict[str, Any], rules: list[dict[str, Any]],
    deliveries: tuple[DeliveryAssessment, ...] = (),
) -> MetricResult:
    """参数为实际故障链、结构结局与流程规则；已恢复与诚实失败分别接受，伪成功拒绝。"""
    product = [row for row in faults if row.get("actor") == "product"]
    if not product:
        return MetricResult(name="error_handling_correctness", status="not_measured", reason="no_observed_product_error")
    if any(row["violations"] for row in rules):
        return boolean_metric("error_handling_correctness", False, reason="workflow_violation_after_observed_fault")
    checks: list[bool | None] = []
    for turn in turns:
        actual = [row for row in product if row.get("turn") == turn.number]
        if not actual:
            continue
        delivery = next((item for item in deliveries if item.turn == turn.number), None)
        if delivery is not None and delivery.missing:
            checks.append(None)
            continue
        operations = [row for row in structure["operations"] if row["turn"] == turn.number]
        resolved = {row["operation_id"] for row in operations if row["final_status"] in {"direct", "repaired", "cache_valid"}}
        resolved.update(row["operation_id"] for row in operations if _nonblocking_failure_disclosed(turn, row))
        unresolved = [row for row in actual if not _fault_recovered(row, resolved, structure)]
        if not unresolved:
            checks.append(True)
            continue
        terminal = turn.raw.get("terminal_state")
        if terminal is None:
            checks.append(None)
            continue
        if deliveries:
            checks.append(next(item.honest_failure for item in deliveries if item.turn == turn.number))
            continue
        summary = turn.values.get("error_summary")
        answer = turn.raw.get("assistant_output") or turn.values.get("answer")
        visible = isinstance(summary, str) and bool(summary.strip()) and isinstance(answer, str) and summary in answer
        checks.append(terminal in _FAILURE_TERMINALS and visible)
    result = False if False in checks else None if None in checks or not checks else True
    return boolean_metric("error_handling_correctness", result, reason="recovered_or_visible_failure_with_honest_terminal")


def _fault_recovered(fault: dict[str, Any], resolved: set[str], structure: dict[str, Any]) -> bool:
    """参数为实际错误和后续操作；只用同责任边界的成功事实确认恢复。"""
    if fault.get("recovered"):
        return True
    if fault.get("source") in {"model_response_usage", "structured_response_validation"}:
        return fault.get("operation_id") in resolved
    if fault.get("source") != "business_validation":
        return False
    found = False
    for row in structure["business_validations"]:
        if row.get("event_id") == fault.get("event_id"):
            found = True
            continue
        if found and row.get("turn") == fault.get("turn") and row.get("validator") == fault.get("validator"):
            if row.get("valid") is True and row.get("accepted") is True:
                return True
    return False


def grade_workflow_record(record: Mapping[str, Any], *, grader_version: str = WORKFLOW_GRADER_VERSION) -> dict[str, Any]:
    """参数为冻结Agent记录和明确评分版本；返回同案例/运行/投影评分，不升级旧策略。"""
    if grader_version not in WORKFLOW_GRADER_VERSIONS:
        raise ValueError(f"unsupported workflow grader version: {grader_version}")
    case, run = agent_record_contracts(record)
    generic_case, generic_run = record_contracts(dict(record))
    if (case.case_id, run.run_id, run.observation_snapshot) != (generic_case.case_id, generic_run.run_id, generic_run.observation_snapshot):
        raise ValueError("workflow generic and agent snapshot identity mismatch")
    if (case.input_snapshot, case.truth_snapshot, run.input_snapshot) != (generic_case.input_snapshot, generic_case.truth_snapshot, generic_run.input_snapshot):
        raise ValueError("workflow generic and agent input/truth snapshot mismatch")
    raw_turns = read_snapshot(run.observation_snapshot).get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("workflow turn observations missing")
    turns = tuple(read_turn(raw) for raw in raw_turns)
    rules = [_turn_rules(turn) for turn in turns]
    deliveries = tuple(assess_delivery(turn) for turn in turns) if grader_version == WORKFLOW_GRADER_VERSION else ()
    for item in deliveries:
        if item.violations or item.missing:
            rules.append({"turn": item.turn, "rule_status": "failed" if item.violations else "missing",
                          "missing": list(item.missing), "violations": [{"turn": item.turn, "code": code} for code in item.violations]})
    numbers = [turn.number for turn in turns]
    sequence_valid = numbers == list(range(1, len(numbers) + 1)) and len(numbers) <= case.turn_count
    if not sequence_valid:
        rules.append({"turn": None, "rule_status": "failed", "missing": [],
                      "violations": [{"code": "invalid_turn_sequence", "observed": numbers}]})
    structure_metrics, structure = grade_structure(turns)
    tool_metrics, tools = grade_tools(turns)
    identity_metrics, identity = grade_scope(turns)
    for violation in _cross_boundary_violations(turns, structure):
        rules.append({"turn": violation["turn"], "rule_status": "failed", "missing": [], "violations": [violation]})
    for row in identity["scope"]:
        if row["passed"] is False:
            rules.append({"turn": row["turn"], "rule_status": "failed", "missing": [],
                          "violations": [{"turn": row["turn"], "code": "returned_evidence_outside_scope",
                                          "paper_ids": row["outside_paper_ids"]}]})
    violations = [issue for row in rules for issue in row["violations"]]
    faults = collect_faults(turns, violations=violations)
    completion, outcome = _completion(turns, declared=case.turn_count, status=run.status)
    # 1. 【流程评测】【覆盖】可判定的违规保留失败；缺少剩余轮次或观察时不宣称全流程通过
    passed = False if violations else None if not turns or any(row["missing"] for row in rules) or len(turns) != case.turn_count else True
    metrics = [completion, boolean_metric("workflow_rule_compliance", passed, reason="all_declared_turns_and_observed_invariants"),
               _error_handling(turns, faults, structure=structure, rules=rules, deliveries=deliveries), *structure_metrics, *tool_metrics, *identity_metrics]
    if grader_version == WORKFLOW_GRADER_VERSION:
        metrics.append(grade_false_success(deliveries, violations=violations,
                                          all_turns_observed=sequence_valid and len(turns) == case.turn_count))
    grading = build_grading(dict(record), metrics, grader="agent_workflow", grader_version=grader_version)
    return {"case_id": case.case_id, "run_id": run.run_id, "grader_version": grader_version,
            **({"delivery": [item.to_dict() for item in deliveries]} if grader_version == WORKFLOW_GRADER_VERSION else {}),
            "grading_result": grading, "measurements": grading["measurements"],
            "workflow": {"declared_turn_count": case.turn_count, "observed_turn_count": len(turns),
                         "execution_status": run.status, "outcome": outcome, "turns": rules,
                         "rule_coverage": dict(Counter(row["rule_status"] for row in rules)),
                         "missing_turns": sorted(set(range(1, case.turn_count + 1)) - set(numbers)),
                         "environment_missing_fields": list(run.missing_fields)},
            "structure": structure, "tools": tools, "identity_checks": identity, "faults": faults,
            "fault_distribution": dict(Counter(row["category"] for row in faults)), "efficiency": efficiency_facts(turns)}
