"""【流程评测】【执行规则】核对现役图路径、已接受计划、依赖与工具四态。作者：xxx。"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.models.research_planner_contracts import ResearchPlan, ResearchUnitResult
from app.services.internal_research.research_planner_validation_rules import dependency_issues, scope_issues, unit_shape_issues
from evals.contracts import MetricResult, snapshot

from .workflow_facts import WorkflowTurn, coverage, event_rows, ratio

_SUCCESS_UNITS = frozenset({"evidence_found", "inferred"})
_TERMINAL_NODES = frozenset({"write_back"})


def _next_node(node: str, state: dict[str, Any]) -> str | None:
    """参数为已执行节点及其实际状态；返回现役研究图合同中的唯一后继。"""
    decision = str(state.get("planner_loop_decision") or "")
    if node in {"rag_init", "planner_synthesize"}:
        return {"rag_init": "planner_analyze", "planner_synthesize": "write_back"}[node]
    if node == "planner_analyze":
        if decision in {"terminate_fail", "terminate_honest"}:
            return "write_back"
        if decision == "probe_context":
            return "planner_analyze"
        return "planner_synthesize" if state.get("turn_depth") == "minimal" and not state.get("planner_draft") else "planner_validate"
    if node == "planner_validate":
        return {"terminate_fail": "write_back", "replan": "planner_analyze"}.get(decision, "planner_execute")
    if node == "planner_execute":
        return {"terminate_fail": "write_back", "replan": "planner_analyze", "repair_evidence": "planner_execute"}.get(decision, "planner_synthesize")
    return None


def route_checks(turn: WorkflowTurn) -> tuple[list[dict[str, Any]], list[str]]:
    """参数为实际一轮流；验证节点顺序和结束后执行，流缺口不当通过。"""
    raw = turn.raw.get("events")
    if not isinstance(raw, (list, tuple)) or not raw:
        return [], ["graph_updates_missing"]
    state = dict(turn.inputs)
    expected: str | None = "rag_init"
    violations: list[dict[str, Any]] = []
    visited: list[str] = []
    for event in raw:
        if event.get("mode") != "updates":
            continue
        for node, delta in event.get("payload", {}).items():
            if node != expected:
                violations.append({"turn": turn.number, "code": "illegal_node_transition",
                                   "expected": expected, "actual": node, "previous": visited[-1:]})
            if not isinstance(delta, dict):
                return violations, ["graph_update_payload_missing"]
            state = {**state, **delta}
            visited.append(node)
            expected = _next_node(node, state)
    if not visited:
        return violations, ["graph_updates_missing"]
    if visited[-1] not in _TERMINAL_NODES and turn.raw.get("status") == "completed":
        return violations, ["graph_terminal_update_missing"]
    return violations, []


def plan_checks(turn: WorkflowTurn) -> tuple[list[dict[str, Any]], list[str]]:
    """参数为实际已接受计划；复用纯合同验证并独立核对依赖、引用，不读取证据真值。"""
    candidates = [turn.values.get("research_plan")]
    candidates.extend(row.get("accepted_plan") for row in event_rows((turn,), "structured_business_validation"))
    violations: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        identity = snapshot(raw)
        if identity in seen:
            continue
        seen.add(identity)
        issues, gaps = _accepted_plan_checks(raw, turn=turn)
        violations.extend(issues)
        missing.extend(gaps)
    references, gaps = reference_checks(turn)
    return [*violations, *references], [*missing, *gaps]


def _accepted_plan_checks(raw: dict[str, Any], *, turn: WorkflowTurn) -> tuple[list[dict[str, Any]], list[str]]:
    """参数为实际被接受的一份计划与本轮范围；拒绝中途非法计划被最终正常计划覆盖。"""
    try:
        plan = ResearchPlan.model_validate(raw)
    except ValidationError as exc:
        return [{"turn": turn.number, "code": "accepted_plan_schema_invalid", "errors": exc.errors(include_url=False)}], []
    scope = turn.inputs.get("project_scope")
    missing = [] if isinstance(scope, dict) and isinstance(scope.get("paper_ids"), list) else ["plan_scope_missing"]
    allowed = frozenset(scope["paper_ids"]) if isinstance(scope, dict) and not missing else None
    issues = [*dependency_issues(plan.units), *scope_issues(plan.units, allowed)]
    seen: set[tuple[str, str, str]] = set()
    for unit in plan.units:
        issues.extend(unit_shape_issues(unit, seen))
    violations = [{"turn": turn.number, "code": item.code, "unit_id": item.unit_id} for item in issues]
    identities = [unit.unit_id for unit in plan.units]
    if len(identities) != len(set(identities)):
        violations.append({"turn": turn.number, "code": "duplicate_unit_id"})
    return violations, missing


def reference_checks(turn: WorkflowTurn) -> tuple[list[dict[str, Any]], list[str]]:
    """参数为实际最终状态；验证引用存在于原始记录及论文身份，不判断结论语义。"""
    results = turn.values.get("research_unit_results", [])
    records = turn.values.get("evidence_records")
    references = [*turn.values.get("planner_final_blocks", []), *turn.values.get("citations", []), turn.values.get("turn_outcome", {})]
    violations: list[dict[str, Any]] = []
    for raw in results:
        try:
            result = ResearchUnitResult.model_validate(raw)
        except ValidationError as exc:
            violations.append({"turn": turn.number, "code": "unit_result_schema_invalid", "errors": exc.errors(include_url=False)})
            continue
        references.append({"evidence_ids": result.evidence_ids})
        references.extend({"evidence_ids": claim.supporting_evidence_ids} for claim in result.claims)
    requested = [identity for row in references for identity in row.get("evidence_ids", [row.get("evidence_id")]) if identity]
    if requested and not isinstance(records, list):
        return violations, ["evidence_reference_registry_missing"]
    registry = {row.get("evidence_id"): row for row in records or []}
    for identity in requested:
        if identity not in registry:
            violations.append({"turn": turn.number, "code": "unknown_evidence_reference", "evidence_id": identity})
    for reference in turn.values.get("citations", []):
        # 1. 【流程评测】【引用身份】已返回的引用必须可定位到Evidence，缺ID不是合法的无引用路径
        if not isinstance(reference.get("evidence_id"), str) or not reference["evidence_id"].strip():
            violations.append({"turn": turn.number, "code": "citation_identity_missing"})
            continue
        bound = registry.get(reference.get("evidence_id"))
        if bound and any(reference.get(key) not in {None, bound.get(key)} for key in ("paper_id", "chunk_id", "parent_id")):
            violations.append({"turn": turn.number, "code": "citation_identity_mismatch"})
    return violations, []


def unit_checks(turn: WorkflowTurn) -> tuple[list[dict[str, Any]], list[str]]:
    """参数为逐单元真实结果；核对必需依赖先完成，工具失败不能伪造成功单元。"""
    rows = event_rows((turn,), "unit_execution_result")
    violations: list[dict[str, Any]] = []
    missing: list[str] = []
    if turn.values.get("research_unit_results") and not rows:
        missing.append("unit_execution_observations_missing")
    for row in rows:
        unit = row.get("unit_id")
        previous = row.get("previous_results")
        if not isinstance(previous, list) or not isinstance(row.get("dependencies"), list):
            missing.append(f"unit_dependency_observation_missing:{unit}")
            continue
        by_id = {item["unit_id"]: item for item in previous}
        for dependency in row["dependencies"]:
            prior = by_id.get(dependency["unit_id"])
            needed = dependency.get("required", True)
            ready = _dependency_ready(prior, dependency)
            if needed and not ready and row.get("unit_status") != "blocked":
                violations.append({"turn": turn.number, "unit_id": unit, "code": "required_dependency_not_completed"})
        if row.get("capability_status") in {"failed", "skipped"} and row.get("unit_status") in _SUCCESS_UNITS:
            violations.append({"turn": turn.number, "unit_id": unit, "code": "tool_failure_reported_as_unit_success"})
    return violations, missing


def _dependency_ready(prior: dict[str, Any] | None, dependency: dict[str, Any]) -> bool:
    """参数为实际前序结果和依赖；按当前合同检查可用主张及指定主张类型。"""
    if prior is None or prior.get("status") in {"blocked", "insufficient_evidence", "external_needed"}:
        return False
    claims = [row for row in prior.get("claims", []) if row.get("validation_status") in {"validated", "needs_validation"}]
    kinds = dependency.get("required_claim_types", [])
    if not kinds:
        return bool(claims)
    return any(row.get("claim_type") in kinds and row.get("validation_status") == "validated" for row in claims)


def grade_tools(turns: tuple[WorkflowTurn, ...]) -> tuple[list[MetricResult], dict[str, Any]]:
    """参数为全部轮次；按逻辑工具归并尝试，跳过调用与失败保留独立分布。"""
    attempts = event_rows(turns, "tool_execution_result")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    unlinked = []
    for row in attempts:
        if not row.get("tool_call_id"):
            unlinked.append(row)
            continue
        grouped.setdefault((row["turn"], row["tool_call_id"]), []).append(row)
    operations = [{**rows[-1], "attempts": rows} for rows in grouped.values()]
    skipped = [row for row in event_rows(turns, "unit_execution_result") if row.get("capability_status") == "skipped"]
    distribution = coverage(operations, "status")
    distribution["skipped"] = len(skipped)
    denominator = sum(distribution.get(key, 0) for key in ("success", "empty", "failed"))
    metrics = [
        ratio("tool_completion", distribution.get("success", 0) + distribution.get("empty", 0), denominator,
              reason="called_tools_with_known_outcomes"),
        ratio("tool_effective_return", distribution.get("success", 0), denominator,
              reason="nonempty_called_tools_with_known_outcomes"),
    ]
    if not operations and not unlinked and _uncalled_tools_observed(turns):
        metrics = [MetricResult(name=metric.name, status="not_measured", reason="observed_route_did_not_call_tools") for metric in metrics]
    return metrics, {"operations": operations, "attempts": attempts, "distribution": distribution,
                     "skipped": skipped, "skipped_count": len(skipped), "unlinked_attempts": unlinked,
                     "operation_count": len(operations), "repeated_requests": _repeated_requests(operations, turns)}


def _uncalled_tools_observed(turns: tuple[WorkflowTurn, ...]) -> bool:
    """参数为真实图路径；只在路径完整且确未执行取证节点时确认无工具调用。"""
    if not turns:
        return False
    for turn in turns:
        violations, missing = route_checks(turn)
        if violations or missing:
            return False
        nodes = [node for event in turn.raw["events"] for node in event.get("payload", {})]
        if "planner_execute" in nodes:
            units = event_rows((turn,), "unit_execution_result")
            if not units or any(not _unit_did_not_call(row) for row in units):
                return False
    return True


def _unit_did_not_call(row: dict[str, Any]) -> bool:
    """参数为真实单元结果；只有明确跳过或非工具来源能证明没有工具调用。"""
    if row.get("capability_status") == "skipped":
        return True
    return row.get("capability_status") is None and row.get("source_type") in {"inference_from_units", "external_context_needed"}


def _repeated_requests(operations: list[dict[str, Any]], turns: tuple[WorkflowTurn, ...]) -> list[dict[str, Any]]:
    """参数为逻辑请求和实际范围；只列完全相同请求，不将合理复用判断为无效执行。"""
    scopes = {turn.number: turn.inputs.get("project_scope") for turn in turns}
    seen: dict[str, str] = {}
    repeated: list[dict[str, Any]] = []
    for row in operations:
        if not isinstance(row.get("parameters"), dict) or scopes.get(row["turn"]) is None:
            continue
        key = snapshot({"scope": scopes[row["turn"]], "source_type": row.get("source_type"), "parameters": row["parameters"]})
        if key in seen:
            repeated.append({"turn": row["turn"], "tool_call_id": row["tool_call_id"],
                             "prior_tool_call_id": seen[key], "classification": "identical_request_only"})
        seen[key] = row["tool_call_id"]
    return repeated
