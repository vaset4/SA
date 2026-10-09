"""Planner ReAct 循环的 RAGState 桥接层：步数读写 + 观察/决策落 state。

决策器与观察归一是纯函数（research_planner_loop_controller），本模块负责把它们
与 LangGraph 的 dict 态对接：从 state 读旧步数快照、推进后整体回写、把决策/观察/留痕
序列化进 state。这里是 design §3.1.1 钉死的「读旧值 → advance → 整体回写」唯一入口，
每个循环节点都必须经此回写 planner_budget，漏写则本回合步数统计失真。
"""

from __future__ import annotations

from typing import Any

from app.models.research_planner_loop_contracts import (
    PlannerDecision,
    PlannerLoopBudget,
    PlannerObservation,
)
from app.services.internal_research.research_planner_loop_controller import (
    advance_budget,
    init_budget,
)

# state 里承载循环步数/观察/决策/留痕的键名，与 models.py RAGState 声明一致
_BUDGET_KEY = "planner_budget"
_OBSERVATION_KEY = "planner_observation"
_DECISION_KEY = "planner_loop_decision"
_TRACE_KEY = "planner_recovery_trace"

# 探测累积结果在 state 里的键名，与 models.py RAGState 的 planner_probe_results 声明一致。
# 写方在 pipeline（探测轮回写）、读方在 plan_prompt（渲染进草案提示词），两处共用本常量，
# 免得同一个字段名在两个模块里各写一份字符串、改一处漏一处
PROBE_RESULTS_STATE_KEY = "planner_probe_results"


def read_budget(state: dict[str, Any]) -> PlannerLoopBudget:
    """从 state 读回步数快照；首轮（无快照）返回计数归零的新快照。

    参数:
        state: 当前 RAGState 字典态
    返回:
        当前循环的 PlannerLoopBudget
    """
    raw = state.get(_BUDGET_KEY)
    if isinstance(raw, dict) and raw:
        return PlannerLoopBudget.model_validate(raw)
    return init_budget()


def advance_and_serialize(
    *,
    budget: PlannerLoopBudget,
    stage: str,
    decision: PlannerDecision | None,
    observation: PlannerObservation | None,
) -> dict[str, Any]:
    """推进一步步数并把步数/决策/观察/留痕序列化成 state 增量。

    这是节点回写的统一出口：analyze 传 decision=None（纯思考步只 +1 总步数），
    validate/execute 传决策后按动作累加计数。返回的 dict 直接并进节点 return，
    保证 planner_budget 整体回写（无 reducer 的 Dict 键，漏写即步数统计失真）。

    参数:
        budget: 推进前的步数快照
        stage: 本步阶段名（analyze/validate/execute）
        decision: 本步决策；analyze 思考步传 None
        observation: 本步观察，序列化进 state 供 replan 回灌 analyzer，
            并作为下一次「有没有新东西」判等的比较基准
    返回:
        含 planner_budget / planner_loop_decision / planner_observation /
        planner_recovery_trace 的 state 增量字典
    """
    advanced = advance_budget(budget, decision, stage=stage, observation=observation)
    payload: dict[str, Any] = {
        _BUDGET_KEY: advanced.model_dump(mode="json"),
        _TRACE_KEY: [step.model_dump(mode="json") for step in advanced.history],
        # 思考步（decision=None）清空决策：避免上一轮 validate/execute 的陈旧决策
        # 残留在 state 里污染 analyze 之后的路由判定
        _DECISION_KEY: decision.action if decision is not None else "",
    }
    if observation is not None:
        payload[_OBSERVATION_KEY] = observation.model_dump(mode="json")
    return payload


def read_decision(state: dict[str, Any]) -> str:
    """读回上一步决策动作，供路由函数选边。缺省空串（路由据此走默认成功边）。"""
    return str(state.get(_DECISION_KEY) or "")


def read_observation(state: dict[str, Any]) -> PlannerObservation | None:
    """读回上一轮观察，供 analyzer 重规划时回灌反馈。无观察返回 None。"""
    raw = state.get(_OBSERVATION_KEY)
    if isinstance(raw, dict) and raw:
        return PlannerObservation.model_validate(raw)
    return None


def clear_decision() -> dict[str, Any]:
    """清空上一步决策，避免陈旧决策污染下一次路由判定。"""
    return {_DECISION_KEY: ""}
