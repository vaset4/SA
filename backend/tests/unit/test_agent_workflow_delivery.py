"""【流程评测】【交付合同】当前交付披露与假成功的独立反例。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from evals.agent_outcome.workflow import grade_workflow_record
from evals.contracts import read_snapshot, snapshot
from test_agent_workflow import _event, _metrics, _record, _structure, _turn

NOTICE = "【注意】本次回答存在未完成事项或校验问题，请结合来源和风险提示核对。"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为测试替换器；阻止外联，保留Windows事件循环socketpair的本机通信。"""
    import socket

    original = socket.socket.connect

    def reject(connection: Any, address: Any) -> None:
        """参数为连接及地址；只允许回环socketpair，外部调用明确失败。"""
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original(connection, address)
        raise AssertionError("offline workflow grading attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject)


def current_turn(*, status: str = "partial", faults: bool = True) -> dict[str, Any]:
    """参数为交付状态及是否观察到拒绝；返回同次报告、正文和终态一致的冻结轮次。"""
    projection = {"partial": ("incomplete", "partial_answer", "partial"),
                  "complete": ("completed", "answered", "ok"),
                  "failed": ("failed", "planner_failed", "failed")}[status]
    reasons = [] if status == "complete" else [{"source": "final_delivery", "code": "section_not_delivered",
                                                "message": "指定章节尚未交付完整正文：方法", "required": True}]
    report = {"contract_version": "research-planner-final-v3", "attempt_id": "attempt-1",
              "entrypoint": "research", "status": status, "generation_status": "completed",
              "reasons": reasons, "requirement_results": [], "section_results": [],
              "validation_errors": [], "unresolved_items": []}
    answer = "已找到比例与依据" + ("\n\n" + NOTICE if status != "complete" else "")
    events = [_event("structured_business_validation", validator="final_delivery", valid=False,
                     accepted=False, errors=["section_not_delivered"])] if faults else []
    return _turn(events, answer=answer, final_delivery_applicability="research", final_delivery_report=report,
                 final_generation_attempt={"attempt_id": "attempt-1", "entrypoint": "research", "generation_status": "completed"},
                 turn_outcome={"status": projection[0], "result_status": projection[1],
                               "termination_reason": "section_not_delivered" if reasons else "delivery_complete"},
                 source_status={"planner": "ok", "planner_final": projection[2]})


def update_values(turn: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """参数为原轮次与指定字段；返回新快照，不修改调用方保存的观察。"""
    values = {**read_snapshot(turn["values_snapshot"]), **changes}
    return {**turn, "values_snapshot": snapshot(values)}


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_current_partial_is_honest_but_not_complete(status: str) -> None:
    """参数为部分或失败交付；如实披露无需旧summary，工作流完成率仍为零。"""
    record = _record([current_turn(status=status)])
    original = deepcopy(record)
    result = grade_workflow_record(record)
    metrics = _metrics(result)
    assert metrics["workflow_completion"]["value"] == 0
    assert metrics["error_handling_correctness"]["value"] == 1
    assert metrics["false_success"]["value"] == 0
    assert result["grader_version"] == "agent-workflow-v2"
    assert record == original


def test_complete_without_fault_has_independent_false_success_measurement() -> None:
    """没有故障时错误处理不适用，但完整观察仍可判定没有假成功。"""
    metrics = _metrics(grade_workflow_record(_record([current_turn(status="complete", faults=False)])))
    assert metrics["error_handling_correctness"]["status"] == "not_measured"
    assert metrics["false_success"]["value"] == 0


@pytest.mark.parametrize("contradiction", ["terminal", "outcome", "body", "attempt"])
def test_current_delivery_contradictions_cannot_pass(contradiction: str) -> None:
    """参数为矛盾来源；报告、终态、正文和同次身份不一致必须保留失败事实。"""
    turn = current_turn()
    if contradiction == "terminal":
        turn = {**turn, "terminal_state": "completed"}
    elif contradiction == "outcome":
        turn = update_values(turn, turn_outcome={"status": "completed", "result_status": "answered"})
    elif contradiction == "body":
        turn = current_turn(status="complete")
        answer = "已找到比例与依据\n\n" + NOTICE
        turn = {**update_values(turn, answer=answer), "assistant_output": answer}
    else:
        turn = update_values(turn, final_generation_attempt={"attempt_id": "another-attempt", "entrypoint": "research", "generation_status": "completed"})
    metrics = _metrics(grade_workflow_record(_record([turn])))
    assert metrics["error_handling_correctness"]["value"] == 0
    if contradiction != "attempt":
        assert metrics["false_success"]["value"] == 1
    else:
        assert metrics["false_success"]["status"] == "missing"


@pytest.mark.parametrize("field", ["contract_version", "report", "assistant_output", "answer", "turn_outcome", "reasons"])
def test_missing_current_delivery_observation_never_defaults_to_pass(field: str) -> None:
    """参数为缺失字段；当前研究合同不能降级到旧error_summary或空默认值而通过。"""
    turn = current_turn()
    values = read_snapshot(turn["values_snapshot"])
    if field in {"contract_version", "reasons"}:
        values["final_delivery_report"].pop(field)
    elif field == "report":
        values.pop("final_delivery_report")
    elif field == "assistant_output":
        turn.pop(field)
    else:
        values.pop(field)
    values["error_summary"] = NOTICE
    turn["values_snapshot"] = snapshot(values)
    metrics = _metrics(grade_workflow_record(_record([turn])))
    assert metrics["false_success"]["status"] == "missing"
    assert metrics["error_handling_correctness"]["status"] == "missing"


def test_hidden_partial_is_bad_error_handling_without_invented_completion_claim() -> None:
    """正文隐藏未完成说明但没有完成声明时，披露失败与假成功证据不足分别报告。"""
    turn = current_turn()
    turn = {**update_values(turn, answer="已找到比例与依据"), "assistant_output": "已找到比例与依据"}
    metrics = _metrics(grade_workflow_record(_record([turn])))
    assert metrics["error_handling_correctness"]["value"] == 0
    assert metrics["false_success"]["status"] == "missing"


def test_schema_recovery_remains_recovered_with_current_complete_delivery() -> None:
    """当前完整交付的同operation已修复错误不成为未恢复故障。"""
    turn = current_turn(status="complete", faults=False)
    capture = read_snapshot(turn["evidence_observation_snapshot"])
    capture["evidence_observations"] = _structure(good=False, repaired=True)
    turn["evidence_observation_snapshot"] = snapshot(capture)
    metrics = _metrics(grade_workflow_record(_record([turn])))
    assert metrics["error_handling_correctness"]["value"] == 1
    assert metrics["false_success"]["value"] == 0


def test_explicit_v1_regrading_preserves_old_result_semantics() -> None:
    """显式旧grader保留旧披露规则，新grader结果另存，不能静默覆盖旧评分。"""
    record = _record([current_turn()])
    metrics = _metrics(grade_workflow_record(record, grader_version="agent-workflow-v1"))
    assert metrics["error_handling_correctness"]["value"] == 0
    assert "false_success" not in metrics


@pytest.mark.parametrize("field,value", [("generation_status", None), ("generation_status", "unknown"),
                                       ("attempt_id", ""), ("reasons", None), ("section_results", [None]),
                                       ("requirement_results", "missing"), ("unresolved_items", [{}]),
                                       ("validation_errors", False)])
def test_malformed_complete_report_does_not_earn_passing_measurement(field: str, value: Any) -> None:
    """参数为缺值或错型字段；完整状态标签不能使不完整报告通过，也不能导致评分器崩溃。"""
    turn = current_turn(status="complete", faults=False)
    values = read_snapshot(turn["values_snapshot"])
    values["final_delivery_report"][field] = value
    turn["values_snapshot"] = snapshot(values)
    metrics = _metrics(grade_workflow_record(_record([turn])))
    assert metrics["false_success"]["status"] == "missing"


@pytest.mark.parametrize("paper_count,materials,reads", [(0, False, False), (2, False, False),
                                                       (2, False, True), (2, True, False)])
def test_real_insufficient_delivery_disclosure_is_recognized(paper_count: int, materials: bool, reads: bool) -> None:
    """参数为真实不足分支条件；正式同步综合器的实际首句可评分，不要求它伪造partial尾注。"""
    from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer
    from test_research_planner_final_delivery import _material_plan

    receipts = [{"actual_call_count": 1, "source_type": "local_section_text", "request": {
        "operation": "sections", "paper_ids": ["p1"]}}] if reads else []
    evidence = [{"evidence_id": "e1", "paper_id": "p1", "layer": "direct_text", "excerpt": "没有目标数据"}] if materials else []
    result = synthesize_planner_answer(_material_plan(), [], project_paper_count=paper_count,
                                      evidence_records=evidence, repair_context={"planner_read_receipts": receipts,
                                                                               "user_query": "比较“方法”：结果"})
    assert result["final_delivery_report"]["status"] == "insufficient"
    assert NOTICE not in result["answer"]
    turn = _turn([], **{**result, "turn_outcome": {"status": "incomplete"}})
    metrics = _metrics(grade_workflow_record(_record([turn])))
    assert metrics["workflow_completion"]["value"] == 0
    assert metrics["false_success"]["value"] == 0


def test_current_clarification_then_complete_preserves_both_turns() -> None:
    """明确澄清不制造研究缺报；随后完整回合可完成，多轮剩余观察缺失时不授予假成功通过。"""
    first = _turn([], answer="请明确要比较的论文", turn_outcome={"status": "incomplete"},
                  final_delivery_applicability="clarification", turn_depth_assessment={"needs_clarification": True})
    second = {**current_turn(status="complete", faults=False), "turn": 2}
    metrics = _metrics(grade_workflow_record(_record([first, second], declared=2)))
    assert metrics["workflow_completion"]["value"] == 1
    assert metrics["false_success"]["value"] == 0
    missing = _metrics(grade_workflow_record(_record([first], declared=2)))
    assert missing["false_success"]["status"] == "missing"


def test_unknown_research_without_report_is_not_assumed_legacy() -> None:
    """没有版本和明确不适用标记的研究观察不能因正文非空当旧合同通过。"""
    turn = _turn([], turn_depth="research")
    metrics = _metrics(grade_workflow_record(_record([turn])))
    assert metrics["false_success"]["status"] == "missing"


def test_explicit_current_early_failure_uses_actual_summary() -> None:
    """前置失败尚无交付报告，明确失败出口和可见summary仍可诚实失败而不完成任务。"""
    turn = _turn([], failure=True, final_delivery_applicability="pre_execution_failure")
    result = grade_workflow_record(_record([turn]))
    assert _metrics(result)["error_handling_correctness"]["value"] == 1
    assert _metrics(result)["false_success"]["value"] == 0
    assert _metrics(result)["workflow_completion"]["value"] == 0
