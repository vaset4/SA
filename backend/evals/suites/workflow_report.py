"""工作流、执行槽位和检索评分故障的独立报告。作者：xxx。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from evals.contracts import read_snapshot

from .contracts import SuiteSummary

RowRenderer = Callable[..., str]


def execution_coverage(summary: SuiteSummary) -> dict[str, Any]:
    """参数为冻结汇总；返回所有预期槽位的原执行状态，不以worker完成冒充流程完成。"""
    manifest = summary.inputs.manifest
    saved = read_snapshot(manifest.execution_snapshot)["runs"] if manifest.execution_snapshot else []
    indexed = {(row["case_id"], row["case_version"], row["repeat_id"]): row for row in saved}
    runs = {run.run.run_id: run for run in summary.inputs.runs}
    slots = []
    for slot in manifest.slots:
        actual = indexed.get((slot.case_id, slot.case_version, slot.repeat_id))
        observed = read_snapshot(runs[slot.run_id].run.observation_snapshot) if slot.run_id in runs else {}
        slots.append({"case_id": slot.case_id, "repeat_id": slot.repeat_id, "run_id": slot.run_id,
                      "worker_status": actual["status"] if actual else "historical_status_missing",
                      "product_status": observed.get("status"), "error": actual.get("error") if actual else None})
    return {"expected": len(slots), "frozen_runs": len(runs), "worker_statuses": dict(Counter(
        row["worker_status"] for row in slots)), "slots": slots}


def workflow_lines(summary: SuiteSummary, row: RowRenderer) -> list[str]:
    """参数为汇总和安全表格渲染器；展示全槽位、API阻塞与系统规则的独立分母。"""
    if summary.inputs.manifest.measurement_policy_snapshot is None:
        return []
    coverage = execution_coverage(summary)
    lines = ["## 执行覆盖与工作流", "", f"预期槽位：{coverage['expected']}；已冻结运行：{coverage['frozen_runs']}。",
             f"Worker 原始状态：{coverage['worker_statuses']}。worker completed 只表示产物交付，不表示流程合规。", "",
             "| 案例 / 重复 | Worker 状态 | 产品状态 | 错误 |", "|---|---|---|---|"]
    for slot in coverage["slots"]:
        lines.append(row(f"{slot['case_id']} / {slot['repeat_id']}", slot["worker_status"], slot["product_status"], slot["error"]))
    workflow = [read_snapshot(score.detail_snapshot) for score in summary.inputs.scores if score.grading.grader == "agent_workflow"]
    faults = Counter(fault["category"] for detail in workflow for fault in detail["faults"])
    blocked = [detail["case_id"] for detail in workflow if _api_blocked(detail)]
    lines.extend(["", f"流程未完成且存在未恢复产品 API 故障：{len(blocked)}/{len(workflow)}；案例：{blocked}。",
                  f"故障事实分类：{dict(faults)}。一次执行可同时包含 API 故障和系统处理缺陷，故障数不等于题数。", "",
                  "| Agent 案例 | 流程结果 | 规则覆盖 | 结构化首次 / 最终 | 工具状态 | 产品逻辑调用 / HTTP尝试 / 重试 |",
                  "|---|---|---|---|---|---|"])
    for detail in workflow:
        efficiency = detail["efficiency"]
        tools = detail["tools"]
        tool_summary = {**tools["distribution"], "skipped": tools["skipped_count"],
                        "operations": tools["operation_count"], "attempts": len(tools["attempts"]),
                        "repeated_requests": tools["repeated_requests"]}
        lines.append(row(detail["case_id"], detail["workflow"]["outcome"], detail["workflow"]["rule_coverage"],
                         f"{detail['structure']['first_distribution']} / {detail['structure']['final_distribution']}",
                         tool_summary,
                         f"{efficiency['logical_model_call_count']} / {efficiency['http_attempt_count']} / {efficiency['http_retry_count']}"))
    return [*lines, "", "流程完成、流程规则符合和错误处理正确性分别计算；观察缺失不算通过，正确处理外部失败也不算流程完成。", ""]


def _api_blocked(detail: dict[str, Any]) -> bool:
    """参数为流程明细；排除同一结构化操作已恢复及已完成流程，不把一次HTTP拒绝算整轮阻塞。"""
    if detail["workflow"]["outcome"] == "completed":
        return False
    resolved = {(item["turn"], item["operation_id"]) for item in detail["structure"]["operations"]
                if item["final_status"] in {"direct", "repaired", "cache_valid"}}
    return any(fault["category"] == "product_api" and not fault.get("recovered")
               and (fault.get("turn"), fault.get("operation_id")) not in resolved for fault in detail["faults"])


def retrieval_lines(summary: SuiteSummary, row: RowRenderer) -> list[str]:
    """参数为汇总和渲染器；逐次列出CP/CR输入来源与独立judge失败，保留成功子样本。"""
    if summary.inputs.manifest.measurement_policy_snapshot is None:
        return []
    lines = ["## 检索调用与 Judge 覆盖", "", *grading_coverage_lines(summary, row),
             "| 案例 / 轮次 / 调用 | 通道 | 原始查询 | 指标 | 状态 | 值 | 原因 | Judge HTTP尝试 / 失败 |",
             "|---|---|---|---|---|---:|---|---|"]
    for score in summary.inputs.scores:
        if score.grading.grader != "ragas_frozen":
            continue
        detail = read_snapshot(score.detail_snapshot)
        for sample in detail["samples"]:
            audit = sample.get("audit", {})
            attempts = [event["payload"] for event in audit.get("observation", {}).get("evidence_observations", [])
                        if event.get("stage") == "model_response_usage"]
            failures = sum(item.get("status") == "failed" or item.get("protocol_status") == "failed" for item in attempts)
            metric = sample["result"]
            lines.append(row(f"{detail['case_id']} / {sample['turn']} / {sample['call_id']}", detail["kind"],
                             sample["retrieval_query"], sample["metric"], metric["status"], metric["value"],
                             metric["reason"], f"{len(attempts)} / {failures}" if attempts else audit.get("error", audit.get("status"))))
    return [*lines, "", "Judge API 故障只使对应检索评分缺失；产品结果与另一项已完成评分均保留。合法章节直读无排名检索时不适用；采集失败为missing，真实空召回为0。", ""]


def grading_coverage_lines(summary: SuiteSummary, row: RowRenderer) -> list[str]:
    """参数为冻结汇总及表格渲染器；返回全槽评分状态、每指标已测分母和明确attempt引用。"""
    scores = {score.run_id: score for score in summary.inputs.scores if score.grading.grader == "ragas_frozen"}
    planner_runs = {score.run_id for score in summary.inputs.scores if score.kind == "planner"}
    slots = [slot for slot in summary.inputs.manifest.slots if slot.run_id not in planner_runs]
    states = Counter(read_snapshot(scores[slot.run_id].detail_snapshot).get("grading_status", "completed")
                     if slot.run_id in scores else "not_started" for slot in slots)
    lines = [f"评分状态（含未执行产品）：{dict(states)}；预定槽位：{len(slots)}。", "",
             "| 指标 | 已测 / 预定槽位 | 明确不适用 | 未开始 | 缺失或无效 |", "|---|---:|---:|---:|---:|"]
    for name in ("context_precision", "context_recall"):
        statuses = Counter(metric.status for score in scores.values() for metric in score.grading.measurements if metric.name == name)
        lines.append(row(name, f"{statuses['measured']} / {len(slots)}", statuses["not_measured"], states["not_started"],
                         statuses["missing"] + statuses["invalid_input"]))
    for source in summary.inputs.manifest.sources:
        if source.ragas_path is not None:
            lines.append(f"- 评分引用：{source.source_id} → {source.ragas_path}；SHA-256={source.ragas_sha256}")
    return [*lines, ""]
