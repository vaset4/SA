"""规划循环决策纯函数单测：放行回边的判据是「这一步有没有带来新东西」（AC1）。

决策器已不读任何额度：放行与否取决于 `budget.history` 里有没有上一次同阶段同动作
、且观察摘要完全相同的留痕。所以本文件的写法是「先推进出一份 history，再拿相同 /
不同的观察调 decide_next」，而不是摆一个计数上界。
"""

from __future__ import annotations

import json

from app.models.research_planner_contracts import EvidenceSufficiencyReport, FacetSufficiencyReport, FinalAnswerEligibilityReport
from o2_evidence_test_utils import direct_evidence

from app.models.research_planner_loop_contracts import (
    PlannerDecision,
    PlannerLoopBudget,
    PlannerObservation,
)
from app.services.internal_research.research_planner_loop_controller import (
    advance_budget,
    decide_next,
    init_budget,
    material_fact_keys,
    observe_execution,
)


def test_restoring_previously_seen_results_is_not_new_progress() -> None:
    """无参数；A/B 成果交替失效和恢复时，累计历史不把旧成果重新算成进展。"""
    first = _execute_obs(has_repairable_gap=True).model_copy(update={"fulfilled_requirement_keys": ["f:A"]})
    second = first.model_copy(update={"fulfilled_requirement_keys": ["f:B"]})
    budget = _with_previous(first, action="repair_evidence", stage="execute")
    budget = advance_budget(budget, PlannerDecision(action="replan"), stage="execute", observation=second)
    assert decide_next(observation=first, budget=budget).action == "terminate_honest"


def test_same_source_with_changed_group_id_is_not_material_progress() -> None:
    """无参数；同一固定正文仅更换材料分组与证据编号，不增加来源进展。"""
    first = material_fact_keys([direct_evidence("group-one")], ["paper-a"])
    second = material_fact_keys([direct_evidence("group-two")], ["paper-a"])
    assert first == second
    previous = _execute_obs(has_repairable_gap=True).model_copy(update={"material_fact_keys": first})
    budget = _with_previous(previous, action="repair_evidence", stage="execute")
    assert decide_next(observation=previous.model_copy(update={"material_fact_keys": second}), budget=budget).action == "terminate_honest"


def test_only_unfulfilled_paper_materials_and_locators_count() -> None:
    """无参数；比较只缺 B 时，A 的额外正文和新目录不能为继续研究提供依据。"""
    report = EvidenceSufficiencyReport(
        pass_index=1, missing_required_facet_ids=["compare"], repair_recommended=True,
        facet_reports=[FacetSufficiencyReport(facet_id="compare", required=True, satisfied=False,
                                             target_coverage={"paper-a": 1, "paper-b": 0}, missing_paper_ids=["paper-b"])],
    )
    observation = observe_execution(
        eligibility=FinalAnswerEligibilityReport(can_call_provider=False), sufficiency=report,
        missing_evidence=[], scoped_paper_ids=["paper-a", "paper-b"], has_answerable_evidence=True,
        evidence_records=[direct_evidence("more-a")], locator_keys=[json.dumps(["paper-a", "section", "New"])]
    )
    assert observation.material_fact_keys == observation.locator_fact_keys == []


def _budget(
    *,
    total_steps: int = 0,
    replan_count: int = 0,
    evidence_repair_count: int = 0,
    probe_count: int = 0,
    history: list | None = None,
) -> PlannerLoopBudget:
    return PlannerLoopBudget(
        total_steps=total_steps,
        replan_count=replan_count,
        evidence_repair_count=evidence_repair_count,
        probe_count=probe_count,
        history=history or [],
    )


def _with_previous(
    observation: PlannerObservation,
    *,
    action: str,
    stage: str,
) -> PlannerLoopBudget:
    """推进出「上一次同类动作」的留痕：判等的基准就是这条 history。"""
    return advance_budget(
        _budget(),
        PlannerDecision(action=action),
        stage=stage,
        observation=observation,
    )


def _validate_obs(
    *,
    ok: bool,
    error_code: str = "",
    issue_codes: list[str] | None = None,
) -> PlannerObservation:
    return PlannerObservation(
        phase="validate",
        ok=ok,
        error_code=error_code,
        issue_codes=issue_codes or [],
    )


def _execute_obs(
    *,
    ok: bool = False,
    has_repairable_gap: bool = False,
    has_answerable_evidence: bool = False,
    scoped_in_index: bool = True,
    issue_codes: list[str] | None = None,
    blocked_unit_ids: list[str] | None = None,
) -> PlannerObservation:
    return PlannerObservation(
        phase="execute",
        ok=ok,
        has_repairable_gap=has_repairable_gap,
        has_answerable_evidence=has_answerable_evidence,
        scoped_in_index=scoped_in_index,
        issue_codes=issue_codes or [],
        blocked_unit_ids=blocked_unit_ids or [],
    )


# --- validate 分支 ---

def test_validate_ok_continues() -> None:
    decision = decide_next(observation=_validate_obs(ok=True), budget=_budget())
    assert decision.action == "continue"


def test_validate_fail_without_previous_replans() -> None:
    # 第一次撞上这个失败面：没有任何基准可比，算「有新东西」，回 analyze 换角度
    decision = decide_next(
        observation=_validate_obs(ok=False, error_code="invalid_facet_id"),
        budget=_budget(),
    )
    assert decision.action == "replan"


def test_validate_same_failure_face_second_time_terminates_fail() -> None:
    # 同一错误码、同一 issue 集第二次出现：重试只会重复同一份计划，硬失败
    failure = _validate_obs(
        ok=False, error_code="explicit_facet_uncovered", issue_codes=["facet_missing"],
    )
    budget = _with_previous(failure, action="replan", stage="validate")

    decision = decide_next(observation=failure, budget=budget)

    assert decision.action == "terminate_fail"


def test_validate_changed_error_code_replans() -> None:
    # 失败面变了（错误码不同）：说明这一轮换的角度确实撞上了别的问题，放行重规划
    previous = _validate_obs(ok=False, error_code="invalid_facet_id")
    budget = _with_previous(previous, action="replan", stage="validate")

    decision = decide_next(
        observation=_validate_obs(ok=False, error_code="cyclic_dependency"),
        budget=budget,
    )

    assert decision.action == "replan"


def test_validate_changed_issue_set_replans() -> None:
    # 同错误码但 issue 集变了（模型补的声明换了一批）：仍算新局面，放行
    previous = _validate_obs(ok=False, error_code="plan_invalid", issue_codes=["a"])
    budget = _with_previous(previous, action="replan", stage="validate")

    decision = decide_next(
        observation=_validate_obs(ok=False, error_code="plan_invalid", issue_codes=["a", "b"]),
        budget=budget,
    )

    assert decision.action == "replan"


def test_validate_issue_order_does_not_count_as_new() -> None:
    # 集合比较：同一批 issue 只是顺序不同不算新东西，否则同一局面能靠乱序反复要一轮
    previous = _validate_obs(ok=False, error_code="plan_invalid", issue_codes=["a", "b"])
    budget = _with_previous(previous, action="replan", stage="validate")

    decision = decide_next(
        observation=_validate_obs(ok=False, error_code="plan_invalid", issue_codes=["b", "a"]),
        budget=budget,
    )

    assert decision.action == "terminate_fail"


# --- execute 分支 ---

def test_execute_ok_continues_to_synthesis() -> None:
    decision = decide_next(observation=_execute_obs(ok=True), budget=_budget())
    assert decision.action == "continue"


def test_execute_new_repairable_gap_repairs() -> None:
    # 有可定向补检索的硬缺口且是头一次见：放行一次补证据
    decision = decide_next(
        observation=_execute_obs(
            has_repairable_gap=True, issue_codes=["direct_text_missing"],
            blocked_unit_ids=["u1"],
        ),
        budget=_budget(),
    )
    assert decision.action == "repair_evidence"


def test_execute_same_repairable_gap_second_time_does_not_repair() -> None:
    # 同一缺口集第二次出现：再补一轮只会重复同一次检索，不再放行 repair
    gap = _execute_obs(
        has_repairable_gap=True, issue_codes=["direct_text_missing"],
        blocked_unit_ids=["u1"],
    )
    budget = _with_previous(gap, action="repair_evidence", stage="execute")

    decision = decide_next(observation=gap, budget=budget)

    assert decision.action == "terminate_honest"


def test_execute_same_gap_with_evidence_delivers_partial() -> None:
    # 缺口没变化但手上已有可用证据：进综合交付部分结果，不诚实收尾
    gap = _execute_obs(
        has_repairable_gap=True, has_answerable_evidence=True,
        issue_codes=["direct_text_missing"],
    )
    budget = _with_previous(gap, action="repair_evidence", stage="execute")

    decision = decide_next(observation=gap, budget=budget)

    assert decision.action == "continue"


def test_execute_changed_unit_ids_do_not_create_progress() -> None:
    # 缺口集变了（换了一批受阻单元）：新局面，放行补证据
    previous = _execute_obs(
        has_repairable_gap=True, issue_codes=["direct_text_missing"],
        blocked_unit_ids=["u1"],
    )
    budget = _with_previous(previous, action="repair_evidence", stage="execute")

    decision = decide_next(
        observation=_execute_obs(
            has_repairable_gap=True, issue_codes=["direct_text_missing"],
            blocked_unit_ids=["u1", "u2"],
        ),
        budget=budget,
    )

    assert decision.action == "terminate_honest"


def test_execute_repair_and_replan_share_one_baseline() -> None:
    # 上一次放行的是 replan、这一次想 repair：同一份缺口集换动作再要一次也要被拦住
    gap = _execute_obs(
        has_repairable_gap=True, issue_codes=["direct_text_missing"],
        blocked_unit_ids=["u1"],
    )
    budget = _with_previous(gap, action="replan", stage="execute")

    decision = decide_next(observation=gap, budget=budget)

    assert decision.action == "terminate_honest"


def test_execute_empty_but_in_index_replans() -> None:
    # 库里确有内容、只是没命中：头一次见到这个缺口，换角度重规划
    decision = decide_next(
        observation=_execute_obs(scoped_in_index=True, issue_codes=["target_coverage_missing"]),
        budget=_budget(),
    )
    assert decision.action == "replan"


def test_execute_not_in_index_terminates_honest_without_replan() -> None:
    # scoped 论文库里根本没内容 → 不浪费一轮 replan，直接诚实收尾
    decision = decide_next(
        observation=_execute_obs(scoped_in_index=False),
        budget=_budget(),
    )
    assert decision.action == "terminate_honest"


def test_execute_not_in_index_needs_no_fresh_gap_to_stop() -> None:
    # 库里没内容时当面就是「无出路」：即便缺口集与上一次不同也不再绕，直接诚实收尾
    budget = _with_previous(
        _execute_obs(has_repairable_gap=True, issue_codes=["direct_text_missing"]),
        action="repair_evidence",
        stage="execute",
    )

    decision = decide_next(
        observation=_execute_obs(scoped_in_index=False, issue_codes=["target_coverage_missing"]),
        budget=budget,
    )

    assert decision.action == "terminate_honest"


# --- 推进步数（并入原 test_planner_budget_state 的行为用例）---

def test_init_budget_is_all_zero() -> None:
    budget = init_budget()
    assert budget.total_steps == 0
    assert budget.replan_count == 0
    assert budget.evidence_repair_count == 0
    assert budget.probe_count == 0
    assert budget.history == []


def test_advance_increments_total_steps() -> None:
    advanced = advance_budget(init_budget(), PlannerDecision(action="continue"), stage="validate")
    assert advanced.total_steps == 1
    assert advanced.replan_count == 0
    assert advanced.evidence_repair_count == 0
    assert advanced.probe_count == 0


def test_advance_analyze_thinking_step_counts_only_total() -> None:
    # analyze 是纯思考步（decision=None）：只 +1 总步数并留痕，动作记 continue
    advanced = advance_budget(init_budget(), None, stage="analyze")
    assert advanced.total_steps == 1
    assert advanced.history[0].action == "continue"


def test_advance_replan_increments_replan_count_only() -> None:
    advanced = advance_budget(_budget(), PlannerDecision(action="replan"), stage="validate")
    assert advanced.total_steps == 1
    assert advanced.replan_count == 1
    assert advanced.evidence_repair_count == 0
    assert advanced.probe_count == 0


def test_advance_repair_increments_evidence_repair_count_only() -> None:
    advanced = advance_budget(_budget(), PlannerDecision(action="repair_evidence"), stage="execute")
    assert advanced.total_steps == 1
    assert advanced.replan_count == 0
    assert advanced.evidence_repair_count == 1
    assert advanced.probe_count == 0


def test_advance_probe_increments_probe_count_only() -> None:
    advanced = advance_budget(_budget(), PlannerDecision(action="probe_context"), stage="analyze")
    assert advanced.total_steps == 1
    assert advanced.probe_count == 1
    assert advanced.replan_count == 0
    assert advanced.evidence_repair_count == 0


def test_advance_is_immutable() -> None:
    budget = _budget()
    advance_budget(budget, PlannerDecision(action="replan"), stage="validate")
    # 原对象不被就地修改
    assert budget.total_steps == 0
    assert budget.replan_count == 0
    assert budget.history == []


def test_advance_appends_history_step() -> None:
    observation = _validate_obs(ok=False, error_code="invalid_facet_id")
    advanced = advance_budget(
        _budget(),
        PlannerDecision(action="replan", reason="重规划"),
        stage="validate",
        observation=observation,
    )

    assert len(advanced.history) == 1
    step = advanced.history[0]
    assert step.step_index == 0
    assert step.stage == "validate"
    assert step.action == "replan"
    assert step.observation["error_code"] == "invalid_facet_id"


def test_history_step_index_tracks_total_steps() -> None:
    first = advance_budget(_budget(), PlannerDecision(action="replan"), stage="analyze")
    second = advance_budget(first, PlannerDecision(action="continue"), stage="validate")
    assert [step.step_index for step in second.history] == [0, 1]


# --- 有界性：判据是「有没有新东西」，不是步数上界 ---

def test_progress_keeps_going_past_the_old_step_bound() -> None:
    # 每一步都真带来新东西时循环不该被任何步数上界掐断：旧实现的 6 步触顶在这里
    # 会提前 terminate_fail，判据换成判等后应一路 replan
    budget = init_budget()
    for step in range(12):
        observation = _validate_obs(
            ok=False, error_code="plan_invalid", issue_codes=[f"issue_{step}"],
        )
        decision = decide_next(observation=observation, budget=budget)
        assert decision.action == "replan", f"第 {step} 步有进展却被拦下"
        budget = advance_budget(budget, decision, stage="validate", observation=observation)

    assert budget.total_steps == 12
    assert budget.replan_count == 12


def test_execute_progress_keeps_going_past_the_old_step_bound() -> None:
    # 执行侧同理：缺口集每轮都变（换了受阻单元）就一直放行补证据
    budget = init_budget()
    for step in range(10):
        observation = _execute_obs(
            has_repairable_gap=True,
            issue_codes=[f"gap_{step}"],
            blocked_unit_ids=[f"u{step}"],
        ).model_copy(update={"material_fact_keys": [f"material-{index}" for index in range(step + 1)]})
        decision = decide_next(observation=observation, budget=budget)
        assert decision.action == "repair_evidence", f"第 {step} 步缺口变了却不再补"
        budget = advance_budget(budget, decision, stage="execute", observation=observation)

    assert budget.total_steps == 10
    assert budget.evidence_repair_count == 10


def test_stalled_validation_converges_within_a_few_steps() -> None:
    # 恒同失败面：第二次就收敛，不再有额度式的多轮白转
    budget = init_budget()
    observation = _validate_obs(ok=False, error_code="explicit_facet_uncovered")
    actions: list[str] = []
    for _ in range(10):
        decision = decide_next(observation=observation, budget=budget)
        actions.append(decision.action)
        if decision.action.startswith("terminate"):
            break
        budget = advance_budget(budget, decision, stage="validate", observation=observation)

    assert actions == ["replan", "terminate_fail"]
    assert budget.total_steps == 1


def test_stalled_execution_converges_to_honest() -> None:
    # 恒同缺口集且无可用证据：第二次就诚实收尾
    budget = init_budget()
    observation = _execute_obs(
        has_repairable_gap=True, issue_codes=["direct_text_missing"], blocked_unit_ids=["u1"],
    )
    actions: list[str] = []
    for _ in range(10):
        decision = decide_next(observation=observation, budget=budget)
        actions.append(decision.action)
        if decision.action.startswith("terminate"):
            break
        budget = advance_budget(budget, decision, stage="execute", observation=observation)

    assert actions == ["repair_evidence", "terminate_honest"]
