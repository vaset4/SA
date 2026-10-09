"""【流程评测】【交付判定】只按冻结版本核对实际披露，不推断自由正文语义。作者：xxx。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from evals.contracts import MetricResult

from .workflow_facts import WorkflowTurn, boolean_metric

CURRENT_DELIVERY_VERSION = "research-planner-final-v3"
INCOMPLETE_NOTICE = "【注意】本次回答存在未完成事项或校验问题，请结合来源和风险提示核对。"
EARLY_FAILURES = frozenset({"planner_depth_assessment_failed", "planner_context_load_failed",
                          "memory_read_failed", "planner_structured_output_failed"})
_SUCCESS_TERMINALS = frozenset({"completed", "converged"})
_PROJECTIONS = {"complete": ("completed", "answered", "ok"),
                "partial": ("incomplete", "partial_answer", "partial"),
                "insufficient": ("incomplete", "insufficient_answer", "skipped"),
                "failed": ("failed", "planner_failed", "failed")}
_REPORT_FIELDS = ("attempt_id", "entrypoint", "status", "generation_status", "reasons",
                  "requirement_results", "section_results", "validation_errors", "unresolved_items")
_FALSE_SUCCESS_CODES = frozenset({"error_reported_as_completed", "partial_evidence_reported_as_completed",
                                 "failed_operation_reported_as_completed"})
_INSUFFICIENT_LEADS = frozenset({
    "已取得部分资料，但尚未取得能满足本次要求的业务成果。",
    "这个项目里还没有可核对的论文材料，请先补充相关论文。",
    "本次尚未找到能支撑回答的材料，可以调整问题或补充相关论文。",
})


@dataclass(frozen=True, kw_only=True)
class DeliveryAssessment:
    """同轮交付检查；假成功未知与披露错误分开，缺字段不填通过。"""

    turn: int
    contract: str
    honest_failure: bool | None = None
    false_success: bool | None = None
    violations: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """无额外参数；返回独立可序列化的判定及证据缺口。"""
        return asdict(self)


def assess_delivery(turn: WorkflowTurn) -> DeliveryAssessment:
    """参数为同次冻结轮次；先识别合同，当前缺报不能套用旧披露规则。"""
    report = turn.values.get("final_delivery_report")
    applicability = turn.values.get("final_delivery_applicability")
    plan = turn.values.get("research_plan")
    plan_version = plan.get("contract_version") if isinstance(plan, dict) else None
    if report is not None or applicability == "research":
        if not isinstance(report, dict):
            return DeliveryAssessment(turn=turn.number, contract=CURRENT_DELIVERY_VERSION, missing=("final_delivery_report",))
        version = report.get("contract_version")
        if version != CURRENT_DELIVERY_VERSION:
            return DeliveryAssessment(turn=turn.number, contract=str(version or "unknown"), missing=("supported_delivery_contract_version",))
        return _current_delivery(turn, report)
    if applicability in {"minimal", "clarification"}:
        return _nonresearch_delivery(turn, contract=applicability)
    if applicability == "pre_execution_failure":
        return _legacy_delivery(turn, contract=applicability)
    if plan_version in {"research-planner-v1", "research-planner-v2"}:
        return _legacy_delivery(turn, contract=plan_version)
    if turn.values.get("error_code") in EARLY_FAILURES and turn.values.get("planner_loop_decision") == "terminate_fail":
        return _legacy_delivery(turn, contract="pre_execution_failure")
    if turn.values.get("turn_depth") == "minimal" and plan_version not in {"research-planner-v3", "research-planner-v4"}:
        return _nonresearch_delivery(turn, contract="minimal")
    if turn.values.get("turn_depth_assessment", {}).get("needs_clarification") is True:
        return _nonresearch_delivery(turn, contract="clarification")
    return DeliveryAssessment(turn=turn.number, contract="unknown", missing=("delivery_contract_or_nonapplicability",))


def _visible_body(turn: WorkflowTurn) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    """参数为冻结轮次；严格核对实际输出和保存正文，缺失时不从另一字段补齐。"""
    actual, saved = turn.raw.get("assistant_output"), turn.values.get("answer")
    missing = tuple(name for name, value in (("assistant_output", actual), ("answer", saved)) if not isinstance(value, str))
    if missing:
        return None, missing, ()
    if actual != saved:
        return None, (), ("displayed_and_saved_answer_mismatch",)
    return actual, (), ()


def _current_delivery(turn: WorkflowTurn, report: dict[str, Any]) -> DeliveryAssessment:
    """参数为V3报告及原轮次；核对必要字段与attempt后评价披露和完成声明。"""
    body, body_missing, body_issues = _visible_body(turn)
    missing = [*body_missing, *_report_missing(report)]
    outcome, attempt = turn.values.get("turn_outcome"), turn.values.get("final_generation_attempt")
    if not isinstance(outcome, dict) or not outcome.get("status"):
        missing.append("turn_outcome.status")
    if turn.raw.get("terminal_state") is None:
        missing.append("terminal_state")
    for name in ("attempt_id", "entrypoint", "generation_status"):
        if not isinstance(attempt, dict) or not isinstance(attempt.get(name), str) or not attempt[name].strip():
            missing.append(f"final_generation_attempt.{name}")
    issues = list(body_issues)
    if isinstance(attempt, dict):
        issues.extend(f"delivery_{name}_mismatch" for name in ("attempt_id", "entrypoint", "generation_status")
                      if attempt.get(name) is not None and report.get(name) is not None and attempt[name] != report[name])
    if missing or issues:
        return DeliveryAssessment(turn=turn.number, contract=CURRENT_DELIVERY_VERSION,
                                  honest_failure=False if issues else None, missing=tuple(missing), violations=tuple(issues))
    return _current_disclosure(turn, report, body=body or "", outcome=outcome)


def _report_missing(report: dict[str, Any]) -> tuple[str, ...]:
    """参数为原始V3报告；显式缺值或畸形观察保持missing，不让默认值补成可评分数据。"""
    missing = [name for name in _REPORT_FIELDS if name not in report]
    for name in ("attempt_id", "entrypoint"):
        if not isinstance(report.get(name), str) or not report[name].strip():
            missing.append(name)
    if not isinstance(report.get("status"), str) or report["status"] not in _PROJECTIONS:
        missing.append("status")
    if not isinstance(report.get("generation_status"), str) or report["generation_status"] not in {"completed", "failed", "not_called"}:
        missing.append("generation_status")
    for name in ("reasons", "requirement_results", "section_results", "validation_errors", "unresolved_items"):
        rows = report.get(name)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            missing.append(name)
            continue
        if name in {"requirement_results", "section_results"} and any(
            not isinstance(row.get("required"), bool) or row.get("status") not in {"delivered", "missing", "not_assessed"} for row in rows
        ):
            missing.append(name)
        if name in {"reasons", "unresolved_items"} and any(
            not isinstance(row.get("required"), bool) or not all(isinstance(row.get(key), str) and row[key].strip() for key in ("source", "code", "message")) for row in rows
        ):
            missing.append(name)
    return tuple(f"final_delivery_report.{name}" for name in dict.fromkeys(missing))


def _current_disclosure(turn: WorkflowTurn, report: dict[str, Any], *, body: str, outcome: dict[str, Any]) -> DeliveryAssessment:
    """参数为完整V3观察；固定尾注及报告原因是披露证据，不进行关键词语义评分。"""
    status = report["status"]
    expected = _PROJECTIONS[status]
    actual = (turn.raw["terminal_state"], outcome["status"], turn.values.get("source_status", {}).get("planner_final"))
    missing = ("source_status.planner_final",) if actual[2] is None else ()
    issues = [] if actual == (expected[0], expected[0], expected[2]) else ["delivery_terminal_projection_mismatch"]
    if missing:
        issues = []
    chat = outcome.get("result_status")
    if chat is not None and chat not in {expected[1], "evidence_not_found" if status == "insufficient" and report.get("evidence_not_found") is True else expected[1]}:
        issues.append("delivery_chat_projection_mismatch")
    reasons = report["reasons"]
    valid_reasons = isinstance(reasons, list) and bool(reasons) and all(
        isinstance(row, dict) and all(isinstance(row.get(key), str) and row[key].strip() for key in ("source", "code", "message")) for row in reasons)
    notice = INCOMPLETE_NOTICE in body.splitlines()[-2:]
    visible_reason = valid_reasons and any(row["message"] in body for row in reasons)
    disclosed = notice or visible_reason or (status == "insufficient" and _insufficiency_disclosed(body))
    # 1. 【流程评测】【假成功】只有明确完成声明与本轮未完成事实冲突才记录1；无披露不自动算说谎
    claims_complete = status == "complete" or actual[0] in _SUCCESS_TERMINALS or actual[1] in _SUCCESS_TERMINALS
    incomplete_fact = status != "complete" or notice or _report_has_required_gaps(report) or report["generation_status"] != "completed"
    false_success = claims_complete and incomplete_fact
    if status != "complete" and not valid_reasons:
        missing = (*missing, "delivery_failure_reasons")
    if status == "complete" and incomplete_fact:
        issues.append("incomplete_delivery_reported_as_complete")
    honest = not issues and disclosed and status != "complete"
    if missing:
        honest = None
    known = not missing and (status == "complete" or disclosed) and not issues
    return DeliveryAssessment(turn=turn.number, contract=CURRENT_DELIVERY_VERSION, honest_failure=honest,
                              false_success=True if false_success else False if known else None,
                              violations=tuple(issues), missing=missing)


def _insufficiency_disclosed(body: str) -> bool:
    """参数为实际正文；仅识别V3未交付描述的完整首句，不按任意不足关键词猜测。"""
    if not body:
        return False
    lead = body.splitlines()[0]
    if lead.startswith("关于“"):
        _question, separator, lead = lead.rpartition("”：")
        if not separator:
            return False
    return lead in _INSUFFICIENT_LEADS or re.fullmatch(
        r"本次已尝试在 [0-9]+ 篇论文范围内查找，但尚未找到能支撑回答的材料。", lead) is not None


def _report_has_required_gaps(report: dict[str, Any]) -> bool:
    """参数为同次报告；只用必需项、校验失败及未解决声明判断结构性未完成。"""
    rows = [*report["requirement_results"], *report["section_results"]]
    return (any(row.get("required") is True and row.get("status") != "delivered" for row in rows)
            or bool(report["validation_errors"])
            or any(row.get("required") is True for row in [*report["reasons"], *report["unresolved_items"]]))


def _legacy_delivery(turn: WorkflowTurn, *, contract: str) -> DeliveryAssessment:
    """参数为明确历史或前置失败合同；沿用旧summary披露，仅在该合同边界生效。"""
    body, missing, issues = _visible_body(turn)
    terminal = turn.raw.get("terminal_state")
    if terminal is None:
        missing = (*missing, "terminal_state")
    summary = turn.values.get("error_summary")
    visible = isinstance(summary, str) and bool(summary.strip()) and body is not None and summary in body
    failed = terminal in {"failed", "incomplete", "terminate_fail", "terminate_honest", *EARLY_FAILURES}
    claims_complete = terminal in _SUCCESS_TERMINALS
    failure_fact = bool(turn.values.get("error_code") or turn.raw.get("error"))
    honest = failed and visible and not issues
    return DeliveryAssessment(turn=turn.number, contract=contract, honest_failure=None if missing else honest,
                              false_success=True if claims_complete and failure_fact else
                              False if not missing and not issues and (honest or claims_complete) else None,
                              missing=missing, violations=issues)


def _nonresearch_delivery(turn: WorkflowTurn, *, contract: str) -> DeliveryAssessment:
    """参数为明确不适用研究交付的路径；仅验证已有正文与状态，不伪造研究报告。"""
    body, missing, issues = _visible_body(turn)
    terminal = turn.raw.get("terminal_state")
    outcome = turn.values.get("turn_outcome", {}).get("status")
    if terminal is None or outcome is None:
        missing = (*missing, "terminal_or_outcome")
    if terminal is not None and outcome is not None and terminal != outcome:
        issues = (*issues, "captured_terminal_mismatch")
    allowed = terminal in _SUCCESS_TERMINALS if contract == "minimal" else terminal in {"incomplete", "terminate_honest"}
    return DeliveryAssessment(turn=turn.number, contract=contract,
                              false_success=False if not missing and not issues and allowed and body else None,
                              missing=missing, violations=issues)


def grade_false_success(assessments: tuple[DeliveryAssessment, ...], *, violations: list[dict[str, Any]], all_turns_observed: bool) -> MetricResult:
    """参数为各轮明确交付和规则违规；1是假成功，0是未观察到假成功，未知保持missing。"""
    proven = any(item.false_success is True for item in assessments) or any(row.get("code") in _FALSE_SUCCESS_CODES for row in violations)
    known = all_turns_observed and bool(assessments) and all(item.false_success is False for item in assessments)
    return boolean_metric("false_success", True if proven else False if known else None,
                          reason="observed_completion_contradiction" if proven else "versioned_delivery_disclosure" if known else "delivery_claim_observation_missing")
