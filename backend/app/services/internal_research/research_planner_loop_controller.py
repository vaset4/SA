"""Planner ReAct 循环的决策核心（纯函数）+ 观察归一 + 步数推进。

三块职责，全部无 I/O、同输入同输出，便于穷举单测：
1. 观察归一：把校验结果 / 执行充分性报告压成决策器可读的 `PlannerObservation`。
2. 步数推进：`init_budget` 起始、`advance_budget` 每次节点转移读旧值 +1 回写（不可变）。
3. 决策：`decide_next` 据观察与留痕产出封闭动作枚举，对齐 design.md §4 决策表。

放行回边比较本次要求、真实来源范围与累计已见事实；已恢复的旧产物不重复记作收益。
计数只用于观察，不作为研究额度。计划前探测与执行尝试共享事实依据，换动作不能重置。

决策器不做任何 I/O：需要查库的判断（scoped 论文是否在索引里有内容）由观察层
在图节点侧填入 `PlannerObservation.scoped_in_index`，本模块只消费布尔量。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from app.models.research_contracts import MissingEvidence
from app.models.research_planner_contracts import (
    EvidenceSufficiencyReport,
    FinalAnswerEligibilityReport,
    PlannerFailure,
    PlannerValidationResult,
)
from app.models.research_planner_loop_contracts import (
    PlannerDecision,
    PlannerLoopBudget,
    PlannerLoopStep,
    PlannerObservation,
)

# 观察摘要进 trace 时的单字段截断长度，避免整段校验栈灌进留痕
_SUMMARY_MAX_CHARS = 240

# 判「无进展」时回看的动作：只有这两类动作会因「有新东西」被放行回边，
# 上一次同类动作的观察就是本次要与之比较的基准
_REPLAN_ACTION = "replan"
_REPAIR_ACTION = "repair_evidence"

# 校验阶段的续跑只有 replan 一种；执行阶段补检索与换角度重规划吃的是同一份缺口集，
# 两者共用一条基准，否则同一缺口集能靠「换个动作再要一次」绕过判等、白走一轮
_EXECUTE_CONTINUATION_ACTIONS = (_REPAIR_ACTION, _REPLAN_ACTION)


def init_budget() -> PlannerLoopBudget:
    """构建首轮步数快照：计数全部归零，无任何上界。

    返回:
        计数全为 0 的新 PlannerLoopBudget
    """
    return PlannerLoopBudget()


def advance_budget(
    budget: PlannerLoopBudget,
    decision: PlannerDecision | None,
    *,
    stage: str,
    observation: PlannerObservation | None = None,
) -> PlannerLoopBudget:
    """推进一次循环转移：总步数 +1，按决策动作累加对应计数，并留痕。

    这是 design §3.1.1 钉死的「读旧值 → +1 → 整体回写」入口：analyze 是纯思考步
    （decision=None，只 +1 总步数并留痕）；validate/execute 产决策后按动作累加对应
    计数。`evidence_repair_count` 只在决策器放行 repair_evidence（即再入 execute）时
    +1，execute 节点内既有那一轮补检索不经过本函数、不计数。`probe_count` 只在放行
    probe_context（出计划前补上下文、回 analyze）时 +1。计数全部只作观察记录，
    没有任何上界，也不参与决策器放行（放行看观察是否带来新东西）。返回全新对象
    （frozen 契约保证不可变）。

    参数:
        budget: 转移前的步数快照
        decision: 刚产出的决策；analyze 思考步传 None
        stage: 本步阶段名（analyze/validate/execute）
        observation: 本步观察，摘要后进留痕
    返回:
        计数递增并追加一条 history 的新 PlannerLoopBudget
    """
    action = decision.action if decision is not None else "continue"
    reason = decision.reason if decision is not None else ""
    # 1. 本步留痕：step_index 用递增前的 total_steps，方便对齐"第几步"
    step = PlannerLoopStep(
        step_index=budget.total_steps,
        stage=stage,
        action=action,
        reason=reason,
        observation=_observation_summary(observation),
    )
    # 2. 总步数每次转移都 +1；replan / repair / probe 仅在对应决策时各自 +1
    return budget.model_copy(update={
        "total_steps": budget.total_steps + 1,
        "replan_count": budget.replan_count + (1 if action == "replan" else 0),
        "evidence_repair_count": budget.evidence_repair_count + (1 if action == "repair_evidence" else 0),
        "probe_count": budget.probe_count + (1 if action == "probe_context" else 0),
        "history": [*budget.history, step],
    })


def decide_next(
    *,
    observation: PlannerObservation,
    budget: PlannerLoopBudget,
) -> PlannerDecision:
    """据阶段观察与历史留痕产出下一步动作（纯函数，对齐 design §4 决策表）。

    参数:
        observation: 归一化的阶段观察（validate 或 execute）
        budget: 当前步数快照；本函数只读它的 history（上一次同阶段观察），不读计数
    返回:
        封闭动作枚举 + 理由的 PlannerDecision
    """
    if observation.phase == "validate":
        return _decide_after_validate(observation, budget)
    return _decide_after_execute(observation, budget)


def decide_action_admission(observation: PlannerObservation, budget: PlannerLoopBudget) -> PlannerDecision:
    """参数为动作校验及历史；按真实输入签名区分错误，同码不同动作仍可修正。"""
    if observation.ok:
        return PlannerDecision(action="continue", reason="当前动作输入有效，执行后交回模型决策")
    repeated = any(step.stage == "validate" and not step.observation.get("ok")
                   and step.observation.get("pending_work_keys") == observation.pending_work_keys
                   and step.observation.get("error_code") == observation.error_code for step in budget.history)
    return PlannerDecision(action="terminate_fail" if repeated else "replan",
                           reason="相同输入重复失败" if repeated else "回填动作错误供模型修正")


def _decide_after_validate(
    observation: PlannerObservation,
    budget: PlannerLoopBudget,
) -> PlannerDecision:
    """校验阶段决策：自洽即取证；失败面有变化才重规划，否则硬失败。"""
    # 计划自洽，进入取证
    if observation.ok:
        return PlannerDecision(action="continue", reason="计划自洽，进入取证")
    # 失败面与上一次重规划时不同 → 回 analyze 换角度重规划
    if _brings_new_information(budget, observation=observation):
        return PlannerDecision(
            action="replan",
            reason=f"计划校验失败({observation.error_code})，失败面有变化，重规划",
        )
    # 同样的失败面再犯一次：重试只会重复同一份计划，计划始终不自洽 → 硬失败
    return PlannerDecision(
        action="terminate_fail",
        reason=f"计划校验失败({observation.error_code})且失败面与上次相同，不再重规划",
    )


def _decide_after_execute(
    observation: PlannerObservation,
    budget: PlannerLoopBudget,
) -> PlannerDecision:
    """执行阶段决策：已满足则综合，存在首次工作或真实新增事实才继续，否则带缺口收束。"""
    # 必需要求与依赖已经满足，进入现有综合；终答资格报告继续作为风险说明
    if observation.ok:
        return PlannerDecision(action="continue", reason="必需要求与依赖已满足，进入现有综合")
    # 1. 【研究循环】【动作准入】缺口本身不赋予续跑资格，先核对累计事实与已做尝试
    if observation.has_repairable_gap and _brings_new_information(
        budget, observation=observation,
    ):
        return PlannerDecision(action="repair_evidence", reason="当前事实支持尚未尝试的定向补证")
    # 库内确有可检索内容、只是本次检索式没命中，且缺口集有变化 → 换角度重规划
    if observation.scoped_in_index and _brings_new_information(
        budget, observation=observation,
    ):
        return PlannerDecision(action="replan", reason="要求未满足，当前事实允许调整一次研究方法")
    # 缺口没有变化但已拿到可用证据 → 带部分证据进综合（终态由取证结果改判为部分完成）
    if observation.has_answerable_evidence:
        return PlannerDecision(action="continue", reason="缺口无变化但已有可用证据，进入综合交付部分结果")
    # 范围内确实查过但无任何可用证据 → 诚实收尾
    return PlannerDecision(action="terminate_honest", reason=_honest_reason(observation))


def _brings_new_information(
    budget: PlannerLoopBudget, *, observation: PlannerObservation,
) -> bool:
    """参数为累计历史、拟执行动作和实际观察；返回是否存在尚未尝试的事实依据。"""
    summaries = [step.observation for step in budget.history
                 if step.observation.get("facts_version") == observation.facts_version]
    if observation.phase == "validate":
        signature = (observation.error_code, tuple(sorted(observation.issue_codes)))
        return not any(
            step.stage == "validate" and step.action == "replan"
            and (step.observation.get("error_code"), tuple(sorted(step.observation.get("issue_codes", [])))) == signature
            for step in budget.history
        )
    attempted = {key for item in summaries for key in item.get("attempted_work_keys", [])}
    attempted.update(key for step in budget.history if step.action in _EXECUTE_CONTINUATION_ACTIONS
                     for key in step.observation.get("pending_work_keys", []))
    if set(observation.pending_work_keys) - attempted:
        return True
    continuations = [step.observation for step in budget.history
                     if step.action in _EXECUTE_CONTINUATION_ACTIONS
                     and step.observation.get("facts_version") == observation.facts_version]
    if not continuations:
        return True
    # 1. 【研究循环】【累计进展】恢复已见产物、换编号或换动作都不产生新的继续资格
    for field in ("fulfilled_requirement_keys", "locator_fact_keys"):
        seen = {key for item in summaries for key in item.get(field, [])}
        if set(getattr(observation, field)) - seen:
            return True
    seen_material = [key for item in summaries for key in item.get("material_fact_keys", [])]
    return any(not _material_seen(key, seen_material) for key in observation.material_fact_keys)


def _material_seen(key: str, seen: Sequence[str]) -> bool:
    """参数为来源区间及累计来源键；返回该固定内容范围是否已经完整取得。"""
    if key in seen:
        return True
    if not key.startswith("["):
        return False
    current = json.loads(key)
    if not isinstance(current, list) or len(current) != 5:
        return False
    spans = [json.loads(item) for item in seen if item.startswith("[")]
    intervals = sorted((item[3], item[4]) for item in spans
                       if isinstance(item, list) and len(item) == 5 and item[:3] == current[:3])
    covered = current[3]
    for start, end in intervals:
        if start > covered:
            break
        covered = max(covered, end)
        if covered >= current[4]:
            return True
    return False


def material_fact_keys(records: Sequence[Mapping[str, Any]], paper_ids: Sequence[str]) -> list[str]:
    """参数为真实材料和未满足对象；返回规范来源范围键，证据编号和摘要不参与判等。"""
    keys: set[str] = set()
    for record in records:
        paper = str(record.get("paper_id") or "")
        if paper_ids and paper not in paper_ids:
            continue
        for span in record.get("source_spans") or []:
            source = [paper, span.get("batch_id")]
            fragments = span.get("source_fragments") or []
            if fragments:
                for fragment in fragments:
                    keys.add(json.dumps([*source, fragment["source_item"], fragment["source_start"],
                                         fragment["source_end"]], ensure_ascii=False))
            else:
                keys.add(json.dumps([*source, f"{span.get('chunk_id')}:{span.get('field')}",
                                     span.get("start"), span.get("end")], ensure_ascii=False))
        if not record.get("source_spans"):
            value = {name: record.get(name) for name in
                     ("paper_id", "source_type", "chunk_id", "section_path", "summary", "excerpt")}
            keys.add(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return sorted(keys)


def fact_digest(value: Any) -> str:
    """参数为可序列化任务事实；返回稳定摘要，供定位引用和要求版本关联。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _honest_reason(observation: PlannerObservation) -> str:
    """诚实收尾的理由：区分"库内无内容"与"库内有但没命中"。"""
    if not observation.scoped_in_index:
        return "scoped 论文库内无可检索内容，直接诚实收尾"
    return "范围内多轮检索无命中，诚实收尾"


def observe_validation(result: PlannerValidationResult) -> PlannerObservation:
    """把计划校验结果归一成观察：通过=ok，失败带错误码与 issue 码集合。

    参数:
        result: validate_or_repair_plan 的返回
    返回:
        phase=validate 的 PlannerObservation
    """
    # 校验通过：无缺口，决策器将放行 continue
    if result.is_valid:
        return PlannerObservation(phase="validate", ok=True, summary="计划校验通过")
    # 校验失败：暴露首个错误码 + 全部 issue 码，供 replan 时回灌 analyzer
    failure = result.failure
    return PlannerObservation(
        phase="validate",
        ok=False,
        error_code=failure.code,
        issue_codes=_validation_issue_codes(failure),
        summary=_compact(failure.message),
    )


def observe_execution(
    *,
    eligibility: FinalAnswerEligibilityReport,
    sufficiency: EvidenceSufficiencyReport,
    missing_evidence: list[MissingEvidence],
    scoped_paper_ids: list[str],
    has_answerable_evidence: bool,
    scoped_in_index: bool | None = True,
    evidence_records: Sequence[Mapping[str, Any]] = (),
    locator_keys: Sequence[str] = (),
    pending_work_keys: Sequence[str] = (),
) -> PlannerObservation:
    """把执行后的充分性/资格报告归一成观察。

    复用既有 derive_missing_evidence 的 attempted_queries，不重算检索式。
    scoped_in_index 由图节点侧查库填入（决策器保持纯函数），缺省 True。

    参数:
        eligibility: 现有终答风险报告，不作为任务完成判据或新的生成硬闸门
        sufficiency: 证据充分性报告（含各单元硬缺口）
        missing_evidence: 缺失证据记录（携带 attempted_queries）
        scoped_paper_ids: 本轮 scoped 论文范围
        has_answerable_evidence: 是否已拿到任意可引用证据（由节点据 records 判定）
        scoped_in_index: scoped 论文在向量库里是否确有可检索内容
    返回:
        phase=execute 的 PlannerObservation
    """
    return PlannerObservation(
        phase="execute",
        ok=not sufficiency.missing_required_facet_ids and not sufficiency.blocking_unit_ids,
        requirements_version=sufficiency.requirements_version,
        fulfilled_requirement_keys=_fulfilled_keys(sufficiency),
        material_fact_keys=material_fact_keys(evidence_records, _missing_targets(sufficiency)),
        locator_fact_keys=_related_locators(locator_keys, _missing_targets(sufficiency)),
        pending_work_keys=list(pending_work_keys),
        gap_keys=sorted({gap.requirement_key or gap.gap_type for unit in sufficiency.unit_reports
                         for gap in unit.hard_gaps}),
        issue_codes=_gap_type_codes(sufficiency),
        has_repairable_gap=sufficiency.repair_recommended,
        blocked_unit_ids=sorted(set(sufficiency.blocking_unit_ids)),
        has_answerable_evidence=has_answerable_evidence,
        scoped_in_index=scoped_in_index,
        attempted_queries=_attempted_queries(missing_evidence),
        scoped_paper_ids=list(scoped_paper_ids),
        summary=_execution_summary(sufficiency),
    )


def _validation_issue_codes(failure: PlannerFailure) -> list[str]:
    """从失败 trace 里抽取全部校验 issue 码（去重保序）。"""
    raw = failure.trace.get("validation_errors") if isinstance(failure.trace, dict) else None
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    codes: list[str] = []
    for entry in raw:
        code = str(entry.get("code") or "").strip() if isinstance(entry, dict) else ""
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _gap_type_codes(report: EvidenceSufficiencyReport) -> list[str]:
    """收集执行后全部硬缺口类型（去重排序），作为 execute 观察的 issue 码。"""
    types = {gap.gap_type for unit_report in report.unit_reports for gap in unit_report.hard_gaps}
    return sorted(types)


def _attempted_queries(missing_evidence: list[MissingEvidence]) -> list[str]:
    """展平缺失证据里实际尝试过的检索式（去重保序），供诚实收尾复用。"""
    seen: set[str] = set()
    queries: list[str] = []
    for item in missing_evidence:
        for query in item.get("attempted_queries", []) or []:
            text = str(query or "").strip()
            if text and text not in seen:
                seen.add(text)
                queries.append(text)
    return queries


def _execution_summary(sufficiency: EvidenceSufficiencyReport) -> str:
    """参数为统一满足报告；返回任务实际完成情况，资格风险仍由独立报告说明。"""
    if not sufficiency.missing_required_facet_ids and not sufficiency.blocking_unit_ids:
        return "必需要求与依赖已满足"
    gap_count = sum(len(unit_report.hard_gaps) for unit_report in sufficiency.unit_reports)
    return f"必需要求或依赖尚未满足，硬缺口 {gap_count} 个"


def _fulfilled_keys(report: EvidenceSufficiencyReport) -> list[str]:
    """参数为统一满足报告；返回已取得的必需要求/对象，保留局部对象进展。"""
    return list(report.fulfilled_requirement_keys)


def _missing_targets(report: EvidenceSufficiencyReport) -> list[str]:
    """参数为统一报告；返回未满足要求涉及的对象，不把整个授权库当作必查集合。"""
    return sorted({paper for facet in report.facet_reports if facet.required and not facet.satisfied
                   for paper in (facet.missing_paper_ids or list(facet.target_coverage))})


def _related_locators(keys: Sequence[str], paper_ids: Sequence[str]) -> list[str]:
    """参数为真实定位键与缺失对象；仅保留未满足对象可使用的定位事实。"""
    if not paper_ids:
        return list(keys)
    return [key for key in keys if json.loads(key)[0] in paper_ids]


def _observation_summary(observation: PlannerObservation | None) -> dict:
    """参数为本步观察；返回含事实键与尝试依据的历史投影，供累计判等与排查。"""
    if observation is None:
        return {}
    return {**observation.model_dump(mode="json"), "summary": _compact(observation.summary)}


def _compact(text: str) -> str:
    """压缩空白并截断，防止整段技术文案进摘要。"""
    compact = " ".join(str(text or "").split())
    if len(compact) > _SUMMARY_MAX_CHARS:
        return f"{compact[:_SUMMARY_MAX_CHARS]}..."
    return compact
