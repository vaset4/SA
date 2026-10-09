"""Planner 有界 ReAct 循环的全链路集成测（AC4 / AC5 + 循环收敛）。

三类验证：

1. 同一失败面的收敛：用真实 planner 节点 + 路由 + 回边搭最小图，恒返回同一份校验
   失败草案。断言循环靠「这一步没带来新东西」在有限步内收敛到 terminate_fail
   ——这验的是「每节点显式读旧 budget +1 回写」（get_type_hints 契约测验不到），
   以及判等真把同一失败面拦下来；旧实现靠步数额度触顶，现在没有额度。
2. 诚实收尾（AC5）：检索落空 + 无可用证据 → execute 决策先按新硬缺口放行一轮补检索，
   缺口无变化后 terminate_honest，synthesize 返回不足交付报告，保留实际读取范围、
   attempted_queries 和 local_evidence_execution=attempted，不把未执行计划当成已查事实。
3. 检索观察回流（AC4）：检索落空但库内有内容 → 观察带 scoped_in_index=True，决策给出
   回边（补检索）；库内无内容时同一观察带 scoped_in_index=False 且理由指向库内为空。
"""

from __future__ import annotations

from o2_evidence_test_utils import scope
from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph

from app.models.models import RAGState
from app.models.research_contracts import TurnDepthAssessment
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch, ResearchPlanStrategyDraft
from app.models.research_planner_loop_contracts import PlannerProbeDecision
from app.nodes import rag_node
from app.services.internal_research.research_planner_pipeline import (
    execute_planner_state,
    synthesize_planner_state,
)
from research_planner_strategy_test_utils import validate_strategy_fixture, strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- 恒失败草案：显式 facet 未被任何单元覆盖，backfill 修不了（只补 inferred） ---

def _persistently_invalid_draft() -> dict[str, Any]:
    # unit 只覆盖 f1，却声明了未覆盖的显式 facet f2 → explicit_facet_uncovered 硬失败
    return make_draft(
        [make_unit("a", facets=["f1"])],
        explicit_facets=[make_facet("f1"), make_facet("f2")],
    )


@dataclass
class _LoopSettings:
    """最小配置对象：仅带单元数上限与循环总超时，供节点读配置。"""

    rag_research_planner_max_units: int = 8
    rag_planner_loop_total_timeout_seconds: float = 0.0


@dataclass
class _AlwaysBadLLM:
    """结构化输出替身：计划草案恒返回同一个校验失败草案，探测决策恒判不用探。

    模拟「模型反复产同一份坏计划」：失败面逐轮完全一致，正是判等要拦的局面。
    """

    draft: dict[str, Any]

    def with_structured_output(self, model: Any) -> "_AlwaysBadStructured":
        return _AlwaysBadStructured(model=model, draft=self.draft)


@dataclass
class _AlwaysBadStructured:
    model: Any
    draft: dict[str, Any]

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self.model is ResearchPlanStrategyDraft:
            return ResearchPlanStrategyDraft.model_validate(strategy_fixture(self.draft))
        if self.model is ResearchClaimStrategyDraftBatch:
            return ResearchClaimStrategyDraftBatch(claims=[])
        if self.model is PlannerProbeDecision:
            return PlannerProbeDecision(needs_probe=False, reason="上下文已足够")
        raise AssertionError(f"unexpected structured output model: {self.model}")


def _analysis_input() -> dict[str, Any]:
    """进图初始状态：深度已定在 lookup，跳过首次分析（意图 / 深度 / 上下文装载）。

    深度与评估结果直接摆进状态是既有做法（`_first_analysis_delta` 见到 turn_depth 即
    返回空增量）：本文件审的是循环回边的收敛与控制流形状，不是首次分析本身。
    """
    return {
        "user_query": "text f1 text f2",
        "turn_depth": "lookup",
        "turn_depth_assessment": TurnDepthAssessment(
            depth="lookup",
            reason_code="paper_evidence_needed",
            requires_paper_evidence=True,
            resolved_query="q",
        ),
    }


def _build_planner_loop_graph():
    """用真实 planner 节点 + 路由 + 回边搭最小可跑图（入口直接进 analyze）。

    走真实 LangGraph dict 合并语义（无 reducer 键未返回则保持旧值），因此能验出
    「某节点漏回写 planner_budget → total_steps 停滞」的隐患。
    """
    graph = StateGraph(RAGState)
    graph.add_node("planner_analyze", rag_node.planner_analyze_node)
    graph.add_node("planner_validate", rag_node.planner_validate_node)
    graph.add_node("planner_execute", rag_node.planner_execute_node)
    graph.add_node("planner_synthesize", rag_node.planner_synthesize_node)
    graph.add_node("write_back", lambda _state: {})
    graph.add_edge(START, "planner_analyze")
    graph.add_conditional_edges(
        "planner_analyze",
        rag_node.route_after_planner_analyze,
        {
            "planner_validate": "planner_validate",
            "planner_analyze": "planner_analyze",
            "planner_synthesize": "planner_synthesize",
            "write_back": "write_back",
        },
    )
    graph.add_conditional_edges(
        "planner_validate",
        rag_node.route_after_planner_validate,
        {
            "planner_execute": "planner_execute",
            "planner_analyze": "planner_analyze",
            "write_back": "write_back",
        },
    )
    graph.add_conditional_edges(
        "planner_execute",
        rag_node.route_after_planner_execute,
        {
            "planner_synthesize": "planner_synthesize",
            "planner_analyze": "planner_analyze",
            "planner_execute": "planner_execute",
            "write_back": "write_back",
        },
    )
    graph.add_conditional_edges(
        "planner_synthesize",
        rag_node.route_after_planner_synthesize,
        {"write_back": "write_back"},
    )
    graph.add_edge("write_back", END)
    return graph.compile()


def test_persistent_validation_failure_converges_without_a_step_bound(monkeypatch) -> None:
    # 恒失败驱动循环：断言它靠「没有新东西」在有限步内收敛到硬失败，而不是靠步数上界
    monkeypatch.setattr(rag_node, "llm", _AlwaysBadLLM(draft=_persistently_invalid_draft()))
    monkeypatch.setattr(rag_node, "settings", _LoopSettings())
    compiled = _build_planner_loop_graph()

    final = _run(compiled.ainvoke(_analysis_input(), {"recursion_limit": 25}))

    budget = final["planner_budget"]
    # 每节点显式回写生效：analyze → validate(replan) → analyze → validate(硬失败) 共 4 步
    assert budget["total_steps"] == 4
    # 同一失败面只放行过一次重规划，第二次就收敛，未无限打转
    assert budget["replan_count"] == 1
    assert final["planner_loop_decision"] == "terminate_fail"
    assert final.get("error_code") == "explicit_facet_uncovered"
    trace = [(step["stage"], step["action"]) for step in final["planner_recovery_trace"]]
    assert trace == [
        ("analyze", "continue"),
        ("validate", "replan"),
        ("analyze", "continue"),
        ("validate", "terminate_fail"),
    ]


def test_same_failure_face_does_not_trigger_a_second_replan(monkeypatch) -> None:
    # 同一失败面反复出现：第二次校验不再回边，直接硬失败（旧版 max_replans 额度语义已删）
    monkeypatch.setattr(rag_node, "llm", _AlwaysBadLLM(draft=_persistently_invalid_draft()))
    monkeypatch.setattr(rag_node, "settings", _LoopSettings())
    compiled = _build_planner_loop_graph()

    final = _run(compiled.ainvoke(_analysis_input(), {"recursion_limit": 25}))

    assert final["planner_loop_decision"] == "terminate_fail"
    # 恰好一次 replan：多一次说明判等没生效，零次说明第一次失败就不给换角度的机会
    assert final["planner_budget"]["replan_count"] == 1
    assert final["planner_budget"]["total_steps"] == 4


@dataclass
class _RaisingLLM:
    """结构化输出恒抛异常，模拟 analyze 阶段的结构化硬失败。"""

    def with_structured_output(self, model: Any) -> "_RaisingStructured":
        return _RaisingStructured()


@dataclass
class _RaisingStructured:
    async def ainvoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("structured output totally failed")


def test_analyze_structured_hard_failure_routes_to_write_back(monkeypatch) -> None:
    # R5：analyze 结构化硬失败必须直达 write_back，不拖着失败载荷穿过 validate
    monkeypatch.setattr(rag_node, "llm", _RaisingLLM())
    monkeypatch.setattr(rag_node, "settings", _LoopSettings())
    compiled = _build_planner_loop_graph()

    final = _run(compiled.ainvoke(_analysis_input(), {"recursion_limit": 25}))

    # 硬失败即终态，决策为 terminate_fail，未产出 planner_draft（没进 validate）
    assert final["planner_loop_decision"] == "terminate_fail"
    assert final.get("error_code")
    assert not final.get("planner_draft")


# --- AC5 诚实收尾：检索落空 + 无可用证据 → terminate_honest → 不足交付 ---

async def _empty_adapter(unit: Any, context: Any) -> EvidenceReadResult:
    """参数为正式读取申请；返回已执行且确实未命中的读取事实。"""
    return EvidenceReadResult((), KnowledgeReadReport(operation="chunks", status="complete"))


def _executable_plan() -> dict[str, Any]:
    result = validate_strategy_fixture(make_draft([make_unit("a", facets=["f1"], paper_ids=["paper-a"])]))
    assert result.plan is not None
    return result.plan.model_dump(mode="json")


@dataclass
class _NoClaimLLM:
    def with_structured_output(self, model: Any) -> "_NoClaimStructured":
        return _NoClaimStructured(model=model)


@dataclass
class _NoClaimStructured:
    model: Any

    async def ainvoke(self, messages: list[Any]) -> Any:
        return ResearchClaimStrategyDraftBatch(claims=[])


def _read_request(unit: Any, _context: Any) -> dict[str, Any]:
    """参数为离线授权后的实际申请；返回与正式检索相同的请求身份形状。"""
    return {"operation": "search_chunks", "queries": list(unit.evidence_policy.locator_queries),
            "structured_query": {"paper_ids": list(unit.evidence_policy.target_paper_ids)}}


def _execute_pass(state: dict[str, Any], index_probe: Any) -> dict[str, Any]:
    """跑一次 execute 阶段并把增量并回状态，返回本次增量供逐项断言。"""
    delta = _run(execute_planner_state(
        state,
        llm=_NoClaimLLM(),
        settings=_LoopSettings(),
        adapters={"local_direct_text": _empty_adapter},
        request_resolver=_read_request,
        index_probe=index_probe,
    ))
    state.update(delta)
    return delta


def test_scoped_empty_repairs_once_then_terminates_honest_then_insufficient() -> None:
    """无参数；验证实际空读收敛后保留原问题和取证事实，交付报告明确不足。"""
    # 检索落空但有可定向补检索的硬缺口 → 先按判等放行一轮补证据；缺口无变化后诚实收尾
    state: dict[str, Any] = {
        "user_query": "LSTTN 的创新点是什么",
        "research_plan": _executable_plan(),
        "run_id": "react-offline-run", "project_scope": scope(),
    }

    first = _execute_pass(state, index_probe=None)
    # 头一次见到这个硬缺口：观察带上缺口种类与受阻单元，判等放行补检索
    assert first["planner_observation"]["has_repairable_gap"] is True
    assert "target_coverage_missing" in first["planner_observation"]["issue_codes"]
    assert first["planner_observation"]["blocked_unit_ids"] == ["u1"]
    assert first["planner_loop_decision"] == "repair_evidence"
    assert rag_node.route_after_planner_execute(first) == "planner_execute"

    second = _execute_pass(state, index_probe=None)

    # 再跑一轮缺口集一模一样 → 判等拦住补检索，无可用证据落到诚实收尾
    assert second["planner_loop_decision"] == "terminate_honest"
    assert rag_node.route_after_planner_execute(second) == "planner_synthesize"
    assert state["planner_budget"]["evidence_repair_count"] == 1
    assert state["planner_budget"]["total_steps"] == 2
    assert state["source_status"]["local_evidence_execution"] == "attempted"

    # synthesize 统一生成不足交付报告，无业务输入时不调用模型
    honest = _run(synthesize_planner_state(state, llm=_NoClaimLLM(), paper_title_resolver=None))

    assert honest["final_delivery_report"]["status"] == "insufficient"
    assert honest["source_status"]["planner_final"] == "skipped"
    assert not honest.get("error_code")
    assert honest["source_status"]["local_evidence_execution"] == "attempted"
    # 终态只报告实际完成读取的论文范围
    assert [item["paper_id"] for item in honest["scoped_papers"]] == ["paper-a"]
    # 实际首轮查询与补证查询均可追查，保留已经完成的读取凭据
    assert "attention mechanism" in honest["attempted_queries"]
    # 正文说人话、不糊检索式（本用户不懂技术）
    assert "LSTTN" in honest["answer"]
    for query in honest["attempted_queries"]:
        assert query not in honest["answer"]


def test_insufficient_delivery_does_not_promote_unexecuted_plan_queries() -> None:
    """无参数；未执行计划的目标和检索式不能在不足说明中冒称实际读取。"""
    plan = _executable_plan()
    plan["units"][0]["evidence_policy"]["locator_queries"] = ["never-executed-query"]
    state = {"user_query": "LSTTN 的创新点是什么", "research_plan": plan,
             "research_unit_results": [], "evidence_records": [], "planner_read_receipts": [],
             "project_scope": scope(), "planner_loop_decision": "terminate_honest"}

    honest = _run(synthesize_planner_state(state, llm=_NoClaimLLM()))

    assert honest["final_delivery_report"]["status"] == "insufficient"
    assert honest["attempted_queries"] == []
    assert honest["scoped_papers"] == []
    assert honest["source_status"].get("local_evidence_execution") != "attempted"
    assert "LSTTN" in honest["answer"]
    assert "never-executed-query" not in honest["answer"]


@pytest.mark.parametrize(("outcome", "call_count", "keep_request"), [
    ("empty", 1, False), ("empty", 0, True), ("failed", 1, True), ("cancelled", 1, True),
])
def test_insufficient_delivery_uses_saved_call_facts(
    outcome: str, call_count: int, keep_request: bool,
) -> None:
    """参数为保存的读取结果、调用数和请求可见性；只投影实际已调用请求，不借计划补齐。"""
    state: dict[str, Any] = {"user_query": "LSTTN 的创新点是什么", "research_plan": _executable_plan(),
                             "run_id": "react-offline-run", "project_scope": scope()}
    _execute_pass(state, index_probe=None)
    original = state["planner_read_receipts"][0]
    receipt = {**original, "outcome": outcome, "actual_call_count": call_count,
               "request": original["request"] if keep_request else None}
    state = {**state, "planner_read_receipts": [receipt], "planner_loop_decision": "terminate_honest"}

    honest = _run(synthesize_planner_state(state, llm=_NoClaimLLM()))

    known_attempt = bool(call_count and keep_request)
    assert honest["final_delivery_report"]["status"] == "insufficient"
    assert honest["attempted_queries"] == (["attention mechanism"] if known_attempt else [])
    assert [item["paper_id"] for item in honest["scoped_papers"]] == (["paper-a"] if known_attempt else [])
    assert honest["source_status"]["local_evidence_execution"] == ("attempted" if call_count else "not_run")


def test_scoped_empty_keeps_the_gap_visible_for_a_second_look() -> None:
    # AC4：检索落空但库内有内容（探针 True）→ 观察把「库内有内容」如实带出去，
    # 决策据此给出回边；回边落地后缺口无变化即收敛，不再白转
    state: dict[str, Any] = {
        "user_query": "text f1 text f2",
        "research_plan": _executable_plan(),
        "run_id": "react-offline-run", "project_scope": scope(),
    }

    async def _in_index_probe(_scoped_ids: list[str]) -> bool:
        return True

    first = _execute_pass(state, index_probe=_in_index_probe)

    assert first["planner_observation"]["scoped_in_index"] is True
    assert first["planner_observation"]["ok"] is False
    assert first["planner_loop_decision"] == "repair_evidence"
    # 回边目标：补检索原地再入 execute（不再回 analyze 换角度）
    assert rag_node.route_after_planner_execute(first) == "planner_execute"

    second = _execute_pass(state, index_probe=_in_index_probe)

    assert second["planner_loop_decision"] == "terminate_honest"
    assert state["planner_budget"]["total_steps"] == 2
    assert state["planner_budget"]["replan_count"] == 0


def test_scoped_empty_out_of_index_is_reported_as_empty_library() -> None:
    # 探针报「库内无内容」时同一观察带 scoped_in_index=False，且收敛到与库内有内容
    # 时同样的步数：明知补不动的场景不额外多绕一轮
    state: dict[str, Any] = {
        "user_query": "text f1 text f2",
        "research_plan": _executable_plan(),
        "run_id": "react-offline-run", "project_scope": scope(),
    }

    async def _empty_index_probe(_scoped_ids: list[str]) -> bool:
        return False

    first = _execute_pass(state, index_probe=_empty_index_probe)

    assert first["planner_observation"]["scoped_in_index"] is False
    assert first["planner_observation"]["ok"] is False

    second = _execute_pass(state, index_probe=_empty_index_probe)

    assert second["planner_loop_decision"] == "terminate_honest"
    assert state["planner_budget"]["total_steps"] == 2
    assert state["planner_budget"]["replan_count"] == 0
