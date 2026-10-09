"""【流程评测】【回归】独立预期验证接口故障、结构化与多轮覆盖。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from evals.agent_outcome.contracts import freeze_agent_run, load_agent_case
from evals.agent_outcome.workflow import grade_workflow_record
from evals.contracts import snapshot


def _event(stage: str, **payload: Any) -> dict[str, Any]:
    """参数为事件名和实际载荷；构造不会执行网络的受控观察。"""
    return {"stage": stage, "event_id": stage, "payload": payload}


def _structure(*, good: bool = True, cached: bool = False, repaired: bool = False) -> list[dict[str, Any]]:
    """参数为完整响应和缓存/回修事实；返回同一逻辑操作的有序事件。"""
    rows = [_event("structured_operation_start", operation_id="op", schema="Plan")]
    rows.append(_event("structured_response_validation", operation_id="op", schema="Plan", phase="initial",
                       schema_valid=good, raw_json_valid=good, cache_hit=cached, normalization_applied=False))
    if repaired:
        rows.append(_event("structured_response_validation", operation_id="op", schema="Plan", phase="repair",
                           schema_valid=True, raw_json_valid=True, cache_hit=False))
    rows.append(_event("structured_operation_end", operation_id="op", status="success" if good or repaired else "failed",
                       repair_count=int(repaired), repair_succeeded=repaired))
    return rows


def _http(status: int, *, attempt: int = 1, **extra: Any) -> dict[str, Any]:
    """参数为实际HTTP状态与尝试序号；保留API事实而非模拟评分结果。"""
    return _event("model_response_usage", actor="product", operation_id="op", call_id="call", http_attempt=attempt,
                  http_status=status, status="completed" if status == 200 else "failed",
                  error_boundary="http_status" if status != 200 else None, **extra)


def _turn(events: list[dict[str, Any]], *, failure: bool = False, number: int = 1, **patch: Any) -> dict[str, Any]:
    """参数为观察和终态变体；生成符合现役minimal/早失败路径的真实形状轮次。"""
    inputs = {"question": "查看论文", "project_scope": {"paper_ids": ["p1"], "user_id": "u1"}}
    values: dict[str, Any] = {"turn_depth": "minimal", "turn_outcome": {"status": "completed"},
                             "answer": "结果", "evidence_records": [], "citations": []}
    analyze = {"turn_depth": "minimal"}
    if failure:
        values.update(turn_outcome={"status": "failed"}, error_summary="接口拒绝请求", error_code="planner_structured_output_failed",
                      planner_loop_decision="terminate_fail", answer="接口拒绝请求")
        analyze["planner_loop_decision"] = "terminate_fail"
    updates = [{"mode": "updates", "payload": {"rag_init": {}}},
               {"mode": "updates", "payload": {"planner_analyze": analyze}}]
    if not failure:
        updates.append({"mode": "updates", "payload": {"planner_synthesize": {}}})
    updates.append({"mode": "updates", "payload": {"write_back": values}})
    values.update(patch)
    return {"turn": number, "status": "completed", "input_snapshot": snapshot(inputs), "values_snapshot": snapshot(values),
            "terminal_state": values["turn_outcome"]["status"], "assistant_output": values["answer"], "events": updates,
            "evidence_observation_snapshot": snapshot({"evidence_observations": events,
                                                        "model_call_observations": [], "evidence_capture_errors": []})}


def _record(turns: list[dict[str, Any]], *, declared: int = 1) -> dict[str, Any]:
    """参数为实际轮次与事前声明数；冻结无证据映射的案例以验证过程指标独立。"""
    case = load_agent_case({
        "id": "workflow-fixture", "title_zh": "流程回归", "task_family": "workflow",
        "input": {"question_zh": "查看论文", "paper_keys": ["p1"]}, "initial_conditions": {"history": []},
        "turns": [{"turn": index, "user_message": "查看论文"} for index in range(1, declared + 1)],
    })
    return freeze_agent_run(case, turns, run_mode="agent_fixture", environment={})


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    """参数为独立侧车；返回按名称检索的测量，不改写记录。"""
    return {row["name"]: row for row in result["measurements"]}


def test_403_without_content_is_missing_and_honest_failure_passes() -> None:
    """无参数；没有模型响应不能编造结构零分，显式失败处理与正常完成分开。"""
    events = [_event("structured_operation_start", operation_id="op"), _http(403),
              _event("structured_operation_end", operation_id="op", status="failed")]
    result = grade_workflow_record(_record([_turn(events, failure=True)]))
    metrics = _metrics(result)
    assert metrics["structured_first_response_compliance"]["status"] == "missing"
    assert metrics["structured_final_compliance"]["status"] == "missing"
    assert metrics["workflow_completion"]["value"] == 0
    assert metrics["error_handling_correctness"]["value"] == 1
    assert result["structure"]["final_distribution"] == {"api_blocked": 1}
    assert result["fault_distribution"]["product_api"] == 1


def test_503_recovered_keeps_attempt_and_first_complete_response_passes() -> None:
    """无参数；重试成功仍保留503，第一次完整响应合法不等于首次HTTP成功。"""
    events = [_http(503), _http(200, attempt=2), *_structure()]
    result = grade_workflow_record(_record([_turn(events)]))
    metrics = _metrics(result)
    assert metrics["workflow_completion"]["value"] == 1
    assert metrics["structured_first_response_compliance"]["value"] == 1
    assert metrics["error_handling_correctness"]["value"] == 1
    assert result["efficiency"]["http_retry_count"] == 1
    assert result["fault_distribution"]["product_api"] == 1


def test_http_200_invalid_json_is_model_error_and_repair_keeps_first_failure() -> None:
    """无参数；正常HTTP中的非法内容属于模型输出，回修不能覆盖首次失败。"""
    result = grade_workflow_record(_record([_turn([_http(200), *_structure(good=False, repaired=True)])]))
    metrics = _metrics(result)
    assert metrics["structured_first_response_compliance"]["value"] == 0
    assert metrics["structured_final_compliance"]["value"] == 1
    assert "product_api" not in result["fault_distribution"]
    assert result["fault_distribution"]["model_output"] == 1


def test_first_invalid_content_survives_repair_api_failure() -> None:
    """无参数；先非法JSON、后回修503同时保留，不让API错误抹掉内容失败。"""
    events = [*_structure(good=False)[:-1], _http(503), _structure(good=False)[-1]]
    result = grade_workflow_record(_record([_turn(events, failure=True)]))
    assert _metrics(result)["structured_first_response_compliance"]["value"] == 0
    assert result["structure"]["final_distribution"] == {"api_blocked": 1}
    assert {"model_output", "product_api"} <= result["fault_distribution"].keys()


def test_api_failure_then_completed_keeps_both_external_and_system_defect() -> None:
    """无参数；接口故障后伪成功必须另计系统缺陷，不能被外部错误豁免。"""
    events = [_event("structured_operation_start", operation_id="op"), _http(403),
              _event("structured_operation_end", operation_id="op", status="failed")]
    result = grade_workflow_record(_record([_turn(events)]))
    assert {"product_api", "system"} <= result["fault_distribution"].keys()
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 0
    assert _metrics(result)["error_handling_correctness"]["value"] == 0


def test_cache_is_valid_object_but_not_new_generation() -> None:
    """无参数；缓存单列，不计入新模型生成的首次或最终合规分母。"""
    result = grade_workflow_record(_record([_turn(_structure(cached=True))]))
    assert result["structure"]["final_distribution"] == {"cache_valid": 1}
    assert _metrics(result)["structured_first_response_compliance"]["status"] == "missing"
    assert _metrics(result)["structured_final_compliance"]["denominator"] is None


def test_missing_operation_cannot_be_hidden_by_other_success() -> None:
    """无参数；完整成功与另一个中断操作同时保留，条件分数显式展示覆盖。"""
    result = grade_workflow_record(_record([_turn([*_structure(), _event("structured_operation_start", operation_id="lost")])]))
    assert result["structure"]["final_distribution"] == {"direct": 1, "missing": 1}
    assert _metrics(result)["structured_observation_coverage"]["value"] == 0.5


def test_all_declared_turns_required_and_missing_observation_not_pass() -> None:
    """无参数；第一轮完成不代表两轮案例完成，无结构观测不推断首次成功。"""
    result = grade_workflow_record(_record([_turn([])], declared=2))
    assert _metrics(result)["workflow_completion"]["value"] == 0
    assert _metrics(result)["workflow_rule_compliance"]["status"] == "missing"
    assert result["workflow"]["missing_turns"] == [2]
    assert _metrics(result)["structured_first_response_compliance"]["status"] == "missing"


def test_two_complete_turns_and_mapping_absence_do_not_block_workflow() -> None:
    """无参数；两轮已完整观察即能判流程，不依赖答案质量或证据映射。"""
    result = grade_workflow_record(_record([_turn([]), _turn([], number=2)], declared=2))
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 1
    assert _metrics(result)["workflow_completion"]["value"] == 1


def test_missing_stream_cannot_be_replaced_by_terminal_or_final_plan() -> None:
    """无参数；最终状态存在不能弥补未采集图事件的路径缺口。"""
    turn = {**_turn([]), "events": []}
    result = grade_workflow_record(_record([turn]))
    assert _metrics(result)["workflow_rule_compliance"]["status"] == "missing"


def test_execution_after_writeback_is_illegal() -> None:
    """无参数；结束节点后再次执行即违规，独立于生产路由返回值。"""
    turn = _turn([])
    turn["events"].append({"mode": "updates", "payload": {"planner_execute": {}}})
    result = grade_workflow_record(_record([turn]))
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 0


def test_business_invalid_accepted_is_model_error_and_system_defect() -> None:
    """无参数；非法依赖被接受同时计模型输出与系统校验缺陷。"""
    event = _event("structured_business_validation", operation_id="op", valid=False, accepted=True,
                   errors=[{"code": "bad_dependency_id"}])
    result = grade_workflow_record(_record([_turn([*_structure(), event])]))
    assert _metrics(result)["structured_business_compliance"]["value"] == 0
    assert {"model_output", "system"} <= result["fault_distribution"].keys()


@pytest.mark.parametrize("dependency", [{"unit_id": "missing", "required": True},
                                        {"unit_id": "prior", "required": True, "required_claim_types": ["paper_supported"]}])
def test_illegal_dependency_cannot_execute_successfully(dependency: dict[str, Any]) -> None:
    """参数为缺失依赖或未验证主张；独立检查前序事实而非仅看成功标签。"""
    event = _event("unit_execution_result", unit_id="next", unit_status="inferred", capability_status=None,
                   dependencies=[dependency], previous_results=[{"unit_id": "prior", "status": "evidence_found", "claims": []}])
    result = grade_workflow_record(_record([_turn([event])]))
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 0


def test_tool_four_states_and_attempts_have_distinct_denominators() -> None:
    """无参数；空返回算完成不算有效，未调用跳过不进调用分母，重试不加逻辑调用数。"""
    events = [_event("tool_execution_result", tool_call_id="a", tool_attempt=1, status="failed"),
              _event("tool_execution_result", tool_call_id="a", tool_attempt=2, status="success"),
              _event("tool_execution_result", tool_call_id="b", tool_attempt=1, status="empty"),
              _event("tool_execution_result", tool_call_id="c", tool_attempt=1, status="failed"),
              _event("unit_execution_result", unit_id="skip", capability_status="skipped", unit_status="blocked",
                     dependencies=[], previous_results=[])]
    result = grade_workflow_record(_record([_turn(events)]))
    assert _metrics(result)["tool_completion"]["value"] == pytest.approx(2 / 3)
    assert _metrics(result)["tool_effective_return"]["value"] == pytest.approx(1 / 3)
    assert result["tools"]["operation_count"] == 3
    assert result["tools"]["skipped_count"] == 1


def test_empty_api_envelope_does_not_become_json_zero() -> None:
    """无参数；HTTP成功但信封缺内容仍归API，内容指标保持缺失。"""
    http = _http(200, protocol_status="failed", error_type="EmptyAssistantContent")
    http["payload"]["error_boundary"] = "api_envelope"
    events = [_event("structured_operation_start", operation_id="op"), http,
              _event("structured_operation_end", operation_id="op", status="failed")]
    result = grade_workflow_record(_record([_turn(events, failure=True)]))
    assert result["fault_distribution"]["product_api"] == 1
    assert _metrics(result)["structured_first_response_compliance"]["status"] == "missing"


def test_grading_is_sidecar_and_preserves_frozen_record() -> None:
    """无参数；新评分保留旧记录且身份可复查，不重定义旧task_success。"""
    record = _record([_turn([])])
    original = deepcopy(record)
    result = grade_workflow_record(record)
    assert record == original
    assert result["grader_version"] == "agent-workflow-v2"
    assert result["grading_result"]["run_id"] == record["run_record"]["run_id"]
    assert "task_success" not in _metrics(result)


def _evidence_turn(*, paper_id: str = "p1", dependency: str | None = None) -> dict[str, Any]:
    """参数为目标与依赖变体；构造合法章节直读且无需最终模型生成的执行事实。"""
    deps = [{"unit_id": dependency, "required": True}] if dependency else []
    unit = {"unit_id": "u1", "goal": "读取章节", "research_action": "读取正文", "output_role": "evidence",
            "evidence_policy": {"source_type": "local_section_text", "target_paper_ids": [paper_id], "required_sections": ["方法"]},
            "claim_policy": {"allowed_claim_types": ["paper_supported"], "citation_required": True},
            "expected_output": {"output_goal": "取得正文"}, "dependencies": deps}
    plan = {"question": "取得方法章节", "plan_type": "single_unit", "units": [unit], "coverage_review": {}}
    events = [_event("unit_execution_result", unit_id="u1", unit_status="evidence_found", capability_status="success",
                     dependencies=deps, previous_results=[]),
              _event("tool_execution_result", tool_call_id="read", tool_attempt=1, source_type="local_section_text",
                     parameters=unit["evidence_policy"], status="success", result_count=1)]
    turn = _turn(events, research_plan=plan, research_unit_results=[{"unit_id": "u1", "status": "evidence_found", "evidence_ids": ["e1"]}],
                 evidence_records=[{"evidence_id": "e1", "paper_id": "p1"}], source_status={"planner": "ok", "planner_final": "skipped"})
    turn["events"] = [{"mode": "updates", "payload": {node: delta}} for node, delta in [
        ("rag_init", {}), ("planner_analyze", {"planner_draft": plan, "turn_depth": "deep"}),
        ("planner_validate", {"research_plan": plan}), ("planner_execute", {}),
        ("planner_synthesize", {}), ("write_back", {}),
    ]]
    return turn


def test_legal_section_read_evidence_only_does_not_require_final_generation() -> None:
    """无参数；章节直读及证据输出路线无需补造检索或最终回答模型调用。"""
    result = grade_workflow_record(_record([_evidence_turn()]))
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 1
    assert _metrics(result)["tool_effective_return"]["value"] == 1
    assert _metrics(result)["structured_first_response_compliance"]["status"] == "missing"


@pytest.mark.parametrize("patch", [{"paper_id": "outside"}, {"dependency": "missing"}, {"dependency": "u1"}])
def test_accepted_illegal_paper_or_dependency_is_independently_rejected(patch: dict[str, str]) -> None:
    """参数为越权paper、缺失依赖或环；独立合同检查拒绝已被系统接受的非法计划。"""
    result = grade_workflow_record(_record([_evidence_turn(**patch)]))
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 0
    assert result["fault_distribution"]["system"] >= 1


def test_no_tool_route_is_not_applicable_but_missing_execution_is_missing() -> None:
    """无参数；已观察直答路径与取证节点缺少调用观察不能使用相同状态。"""
    direct = grade_workflow_record(_record([_turn([])]))
    missing = _evidence_turn()
    missing["evidence_observation_snapshot"] = snapshot({"evidence_observations": [], "model_call_observations": []})
    assert _metrics(direct)["tool_completion"]["status"] == "not_measured"
    assert _metrics(grade_workflow_record(_record([missing])))["tool_completion"]["status"] == "missing"


def test_probe_and_repair_loops_are_legal_when_actual_decisions_match() -> None:
    """无参数；合法探测和取证修复自环保留，不按重复节点或人为轮数判违规。"""
    turn = _evidence_turn()
    turn["events"].insert(1, {"mode": "updates", "payload": {"planner_analyze": {"planner_loop_decision": "probe_context"}}})
    turn["events"][2]["payload"]["planner_analyze"]["planner_loop_decision"] = "continue"
    turn["events"].insert(4, {"mode": "updates", "payload": {"planner_execute": {"planner_loop_decision": "repair_evidence"}}})
    turn["events"][5]["payload"]["planner_execute"]["planner_loop_decision"] = "continue"
    result = grade_workflow_record(_record([turn]))
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 1


@pytest.mark.parametrize("disclosed", [True, False])
def test_memory_proposal_failure_requires_actual_nonblocking_failure_state(disclosed: bool) -> None:
    """参数为是否如实保留写回失败；生产终态转换允许回答完成，但不允许伪造写回成功。"""
    from app.services.internal_research.research_memory_writeback import _failed, build_turn_outcome

    memory = _failed("HTTPError")
    outcome = build_turn_outcome({"turn_depth": "research", "planner_loop_decision": "continue"}, memory_writeback=memory)
    events = [_event("structured_operation_start", operation_id="op", schema="TurnMemoryProposalBatch"), _http(503),
              _event("structured_operation_end", operation_id="op", status="failed")]
    turn = _turn(events, memory_writeback=memory if disclosed else {"status": "success"}, turn_outcome=outcome)
    result = grade_workflow_record(_record([turn]))
    assert _metrics(result)["workflow_rule_compliance"]["value"] == float(disclosed)
    assert _metrics(result)["error_handling_correctness"]["value"] == float(disclosed)
    assert result["fault_distribution"]["product_api"] == 1
    assert result["structure"]["final_distribution"]["api_blocked"] == 1


@pytest.mark.parametrize("exception_name", ["InvalidHeader", "InvalidURL", "MissingSchema"])
def test_real_local_request_exception_facts_are_system_defects(exception_name: str) -> None:
    """参数为requests本地构造异常；使用生产事实采集器，不能靠transport标签归咎供应商。"""
    from requests import exceptions

    from app.services.llm_observation import http_error_facts

    error = getattr(exceptions, exception_name)("invalid local request")
    facts = http_error_facts(error, boundary="transport")
    event = _event("model_response_usage", actor="product", operation_id="op", status="failed", **facts)
    result = grade_workflow_record(_record([_turn([event], failure=True)]))
    assert result["fault_distribution"]["system"] == 1
    assert "product_api" not in result["fault_distribution"]


def test_returned_scope_and_citation_checks_do_not_require_original_text_mapping() -> None:
    """无参数；实际返回越界论文和未知引用独立失败，纯映射缺失不能掩盖身份问题。"""
    turn = _turn([], evidence_records=[{"evidence_id": "e1", "paper_id": "outside"}], citations=[{"evidence_id": "invented"}])
    result = grade_workflow_record(_record([turn]))
    assert _metrics(result)["scope_adherence"]["value"] == 0
    assert _metrics(result)["citation_reference_validity"]["value"] == 0
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 0


@pytest.mark.parametrize("citation", [{"paper_id": "p1"}, {"evidence_id": "", "paper_id": "p1"}])
def test_returned_citation_requires_its_actual_evidence_identity(citation: dict[str, Any]) -> None:
    """参数为已返回但缺少Evidence身份的引用；不能把无可解析引用误当成合法的空引用路线。"""
    turn = _turn([], evidence_records=[{"evidence_id": "e1", "paper_id": "p1"}], citations=[citation])
    result = grade_workflow_record(_record([turn]))
    assert _metrics(result)["citation_reference_validity"]["value"] == 0
    assert _metrics(result)["workflow_rule_compliance"]["value"] == 0


def test_unknown_cache_provenance_does_not_count_as_new_generation() -> None:
    """无参数；缓存身份未采集不能靠schema合法对象推断模型生成成功。"""
    events = _structure()
    events[1]["payload"].pop("cache_hit")
    result = grade_workflow_record(_record([_turn(events)]))
    assert _metrics(result)["structured_final_compliance"]["status"] == "missing"
    assert result["structure"]["final_distribution"] == {"missing": 1}


def test_bad_cache_is_not_a_model_generation_failure() -> None:
    """无参数；只有坏缓存而无新模型响应时，不将缓存对象错误计为模型格式零分。"""
    result = grade_workflow_record(_record([_turn(_structure(good=False, cached=True), failure=True)]))
    assert result["structure"]["final_distribution"] == {"cache_invalid": 1}
    assert _metrics(result)["structured_final_compliance"]["status"] == "missing"


def test_business_coverage_retains_unvalidated_operation() -> None:
    """无参数；一个计划业务校验通过不能遮盖另一个计划未取得业务校验观察。"""
    events = _structure()
    events[0]["payload"]["schema"] = "ResearchPlanDraft"
    events.extend([_event("structured_operation_start", operation_id="lost", schema="ResearchPlanDraft"),
                   _event("structured_business_validation", operation_id="op", valid=True)])
    result = grade_workflow_record(_record([_turn(events)]))
    assert _metrics(result)["structured_business_observation_coverage"]["value"] == 0.5
    assert result["structure"]["business_validation_missing_operation_ids"] == ["lost"]


def test_local_type_error_at_transport_boundary_is_not_provider_failure() -> None:
    """无参数；请求序列化或本地调用类型错误不能因为transport捕获点被归到API。"""
    from app.services.llm_observation import http_error_facts

    facts = http_error_facts(TypeError("request cannot be serialized"), boundary="transport")
    event = _event("model_response_usage", actor="product", status="failed", **facts)
    result = grade_workflow_record(_record([_turn([event], failure=True)]))
    assert result["fault_distribution"]["system"] == 1
    assert "product_api" not in result["fault_distribution"]


def test_sidecar_rejects_conflicting_generic_and_agent_snapshot() -> None:
    """无参数；不能用不同输入的通用投影签名覆盖Agent实际输入身份。"""
    record = _record([_turn([])])
    record["run_record"]["input_snapshot"] = snapshot({"question": "other"})
    with pytest.raises(ValueError, match="input/truth snapshot mismatch"):
        grade_workflow_record(record)
