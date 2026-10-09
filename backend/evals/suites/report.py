"""输出可追溯的离线套件主表、逐题结果和条件比较。作者：xxx。"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from evals.contracts import read_snapshot

from .contracts import SuiteInput, SuiteReport, SuiteSummary
from .workflow_report import retrieval_lines, workflow_lines

_MAIN_METRICS = (
    ("RAGAS 上下文精确率", "context_precision"),
    ("RAGAS 上下文召回率", "context_recall"),
    ("RAGAS 回答忠实性", "faithfulness"),
    ("RAGAS 事实正确性（F1）", "factual_correctness"),
    ("规划约束（现有结构诊断）", "plan_valid_rate"),
    ("所需原文召回", "evidence_group_recall_at_k"),
    ("跨论文取证覆盖", "cross_paper_coverage"),
    ("取证任务完成", "task_success"),
)
_WORKFLOW_METRICS = (
    ("流程完成", "workflow_completion"), ("流程规则符合", "workflow_rule_compliance"),
    ("异常处理正确性", "error_handling_correctness"),
    ("结构化观察覆盖", "structured_observation_coverage"),
    ("结构化业务观察覆盖", "structured_business_observation_coverage"),
    ("结构化首次合规", "structured_first_response_compliance"),
    ("结构化最终合规", "structured_final_compliance"),
    ("结构化业务合规", "structured_business_compliance"),
    ("工具完成", "tool_completion"), ("工具有效返回", "tool_effective_return"),
    ("RAGAS 上下文精确率", "context_precision"), ("RAGAS 上下文召回率", "context_recall"),
    ("所需原文召回", "evidence_group_recall_at_k"), ("跨论文取证覆盖", "cross_paper_coverage"),
)


def _cell(value: Any) -> str:
    """参数为展示值；转义外部文本和表格符号，未采集值保持空缺符号。"""
    if value is None:
        return "—"
    return html.escape(str(value)).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _row(*values: Any) -> str:
    """参数为一行表格值；返回转义后的 Markdown 行。"""
    return "| " + " | ".join(_cell(value) for value in values) + " |"


def _source_label(key: dict[str, Any]) -> str:
    """参数为真实分组键；展示来源、运行方式、审核状态和阶段。"""
    return " / ".join(str(key[name]) for name in ("source_type", "run_mode", "review_status", "stage", "backend"))


def _gap_lines(summary: SuiteSummary) -> list[str]:
    """参数为汇总；优先展示漏跑、检查失败及无法补写的历史条件。"""
    lines = ["## 覆盖缺口与检查", "", "| 案例 / 重复 | 项目 | 状态 | 原因 |", "|---|---|---|---|"]
    for slot in summary.inputs.manifest.slots:
        if slot.run_id is None:
            lines.append(_row(f"{slot.case_id} / {slot.repeat_id}", "运行", "missing", "尚无实际运行"))
    for check in summary.checks:
        lines.append(_row(f"{check.case_id} / {check.repeat_id}", check.name, check.status, check.reason))
    for run in summary.inputs.runs:
        missing = [condition.name for condition in run.conditions if condition.status == "missing"]
        if missing:
            lines.append(_row(run.case.case_id, "历史比较条件", "missing", ", ".join(missing)))
    lines.extend(["", "身份缺失影响比较资格；历史条件按原记录展示，不使用当前机器配置补写。", ""])
    return lines


def _main_lines(summary: SuiteSummary) -> list[str]:
    """参数为汇总；按冻结策略展示指标，保留不同通道及缺失分母。"""
    workflow = summary.inputs.manifest.measurement_policy_snapshot is not None
    selected = _WORKFLOW_METRICS if workflow else _MAIN_METRICS
    lines = ["## 工作流与检索指标" if workflow else "## 质量指标与项目诊断", "", "| 检查目标 | 指标 | 来源 / 模式 / 阶段 | 结果 | 完整测量题数 / 本组题数 |",
             "|---|---|---|---:|---:|"]
    for label, metric in selected:
        groups = [group for group in summary.groups if group.metric == metric]
        if not groups:
            lines.append(_row(label, metric, "未提供适用评分", None, "未测量"))
        for group in groups:
            lines.append(_row(label, metric, _source_label(read_snapshot(group.key_snapshot)), group.mean,
                              f"{group.complete_case_count}/{len(group.cases)}"))
    lines.extend(["", "结果仅针对本组完整测量题；全部计划、失败和未执行槽位另列，不能把条件均值当作全样本通过率。",
                  "RAGAS Context Precision 使用有参考答案的平均精度变体；规划结构诊断不代表开放任务规划质量。",
                  "candidate 表示参考标注未经人工审核；agent_live / retrieval_live 表示实际产品运行。自动评分不等于人工金标准确率。",
                  "检索主结果以独立 retrieval_live 通道为准；Agent 子查询评分单列，不能把不同子任务对整份参考答案的召回率当作主任务召回率。",
                  "fixture 与 replay 保持各自来源标签；缺失观察不补空、不补分。", ""])
    return lines


def _measurement_lines(summary: SuiteSummary) -> list[str]:
    """参数为汇总；保留逐题逐调用原指标、真实分母、失败和未测原因。"""
    lines = ["## 原始评分明细", "", "| 案例 / 运行 | 阶段 / 调用 | 查询 | 指标 | 状态 | 值 | 分子 / 分母 | 原因 |",
             "|---|---|---|---|---|---:|---|---|"]
    for score in summary.inputs.scores:
        detail = read_snapshot(score.detail_snapshot)
        for metric in score.grading.measurements:
            counts = None if metric.denominator is None else f"{metric.numerator}/{metric.denominator}"
            lines.append(_row(f"{score.grading.case_id} / {score.run_id}", f"{score.stage} / {score.call_id}",
                              detail.get("query"), metric.name, metric.status, metric.value, counts, metric.reason))
    lines.extend(["", "## 分层辅助指标", "", "| 指标 | 来源 / 模式 / 阶段 | 题级均值 | 四态观测计数 |",
                  "|---|---|---:|---|"])
    for group in summary.groups:
        counts = ", ".join(f"{state.status}={state.count}" for state in group.states)
        lines.append(_row(group.metric, _source_label(read_snapshot(group.key_snapshot)), group.mean, counts))
    return [*lines, ""]


def _repetition_lines(summary: SuiteSummary) -> list[str]:
    """参数为汇总；单独显示至少一次成功与全部成功，漏跑保持未知。"""
    lines = ["## 重复运行", "", "| 案例 | 来源 | 预期次数 | 已测次数 | 成功次数 | 单次成功率 | 至少一次成功 | 全部成功 |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for result in summary.repetitions:
        lines.append(_row(result.case_id, result.metric + " / " + _source_label(read_snapshot(result.key_snapshot)), result.expected_runs,
                          result.measured_runs, result.successful_runs, result.single_run_success_rate,
                          result.any_success, result.all_success))
    return [*lines, ""]


def _comparison_lines(report: SuiteReport) -> list[str]:
    """参数为报告；不可比条件或指标显示原因，不渲染不存在的差值。"""
    lines = ["## 基线比较", ""]
    comparison = report.comparison
    if comparison is None:
        return [*lines, "本次未请求比较。", ""]
    lines.extend([f"资格：{comparison.status}", f"声明变化项：{', '.join(comparison.request.changed_factors) or '无'}", ""])
    lines.extend(f"- {_cell(reason)}" for reason in comparison.reasons)
    lines.extend(["", "| 指标 | 来源 | 状态 | 当前 | 基线 | 差值 | 原因 |", "|---|---|---|---:|---:|---:|---|"])
    for delta in comparison.deltas:
        lines.append(_row(delta.metric, _source_label(read_snapshot(delta.key_snapshot)), delta.status,
                          delta.current, delta.baseline, delta.delta, ", ".join(delta.reasons)))
        paired = read_snapshot(delta.pairs_snapshot)
        if "paired_cases" in paired:
            lines.append(f"配对覆盖 {delta.metric}：{paired['paired_cases']} / {paired['expected_cases']}；排除明细：{_cell(paired['excluded'])}")
    return [*lines, "", "质量变化作为趋势展示；低质量分数本身不新增 P0 阈值。", ""]


def _provenance_lines(summary: SuiteSummary) -> list[str]:
    """参数为汇总；列出来源校验和、实际运行状态、任务覆盖与已采集的资源用量。"""
    lines = ["## 产物来源和运行记录", ""]
    for reference in summary.inputs.manifest.sources:
        lines.append(f"- {_cell(reference.source_id)}：{_cell(reference.report_path)}；SHA-256={reference.report_sha256}")
    lines.extend(["", "| 案例 / 运行 | 任务与论文覆盖 | 实际终态 / 错误 | usage / cost / latency |",
                  "|---|---|---|---|"])
    for run in summary.inputs.runs:
        observed = read_snapshot(run.run.observation_snapshot)
        topology = read_snapshot(run.topology_snapshot)
        state = {name: observed[name] for name in ("status", "turn_outcome", "failure_reason") if name in observed}
        resources = {name: observed[name] for name in ("usage", "cost", "latency") if name in observed}
        lines.append(_row(f"{run.case.case_id} / {run.run.run_id}", json.dumps(topology, ensure_ascii=False),
                          json.dumps(state, ensure_ascii=False) if state else None,
                          json.dumps(resources, ensure_ascii=False) if resources else None))
    return [*lines, "", "未采集的用量和耗时保持缺失；完整冻结合同、分母及逐题比较保留在 report.json。", ""]


def build_suite_report(inputs: SuiteInput, summary: SuiteSummary) -> SuiteReport:
    """参数为已校验输入和派生汇总；输入身份一致时返回可重放报告。"""
    if inputs != summary.inputs:
        raise ValueError("summary belongs to another suite input")
    return SuiteReport(summary=summary)


def render_suite_report(report: SuiteReport) -> str:
    """参数为结构化报告；返回中文 Markdown，保持来源、缺口和原始计数可见。"""
    summary = report.summary
    manifest = summary.inputs.manifest
    lines = ["# 离线评测套件报告", "", f"套件：{_cell(manifest.suite_id)}；用途：{manifest.purpose}",
             f"数据版本：{_cell(manifest.dataset_version)}；独立逻辑案例：{len({slot.case_id for slot in manifest.slots})}；预期运行：{len(manifest.slots)}；实际运行：{len(summary.inputs.runs)}", ""]
    if manifest.purpose == "online_discovery":
        lines.extend(["本报告仅保留问题发现明细，不计算离线质量平均和默认门禁。", ""])
    if manifest.measurement_policy_snapshot is not None:
        lines.extend([f"冻结测评策略：{_cell(manifest.measurement_policy_snapshot)}", ""])
    if manifest.grading_attempt_snapshot is not None:
        lines.extend([f"本次评分清单：{_cell(manifest.grading_attempt_snapshot)}", ""])
    lines.extend(workflow_lines(summary, _row))
    lines.extend(_gap_lines(summary))
    lines.extend(_main_lines(summary))
    lines.extend(retrieval_lines(summary, _row))
    lines.extend(_measurement_lines(summary))
    lines.extend(_repetition_lines(summary))
    lines.extend(_comparison_lines(report))
    lines.extend(_provenance_lines(summary))
    return "\n".join(lines)


def write_suite_report(output_dir: Path, report: SuiteReport) -> None:
    """参数为新目录和冻结报告；保存 JSON、Markdown 和原始评分 CSV，拒绝覆盖历史产物。"""
    json_text = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    markdown = render_suite_report(report)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json_text, encoding="utf-8", newline="\n")
    (output_dir / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    _write_results_csv(output_dir / "results.csv", report.summary.inputs)


def render_execution_report(runs: list[dict[str, Any]]) -> str:
    """参数为每个预期槽位的真实执行结果；即使全部失败也输出覆盖和错误，不伪造评分。"""
    from collections import Counter

    counts = Counter(run["status"] for run in runs)
    lines = ["# 评测执行覆盖", "", f"预期槽位：{len(runs)}；实际状态：{dict(counts)}。", "",
             "完成只表示worker交付产物，流程与检索指标见独立冻结评分。未形成评分的API责任保持未知。", "",
             "| 作业 / 案例 / 重复 | 状态 | 错误 | 耗时（秒） |", "|---|---|---|---:|"]
    for run in runs:
        lines.append(_row(f"{run['job_id']} / {run['case_id']} / {run['repeat_id']}",
                          run["status"], run.get("error"), run.get("elapsed_seconds")))
    return "\n".join([*lines, ""])


def _write_results_csv(path: Path, inputs: SuiteInput) -> None:
    """参数为新文件与已校验输入；逐 case/run 导出原始测量及缺失槽位，空值保持空白。"""
    fields = ("case_id", "case_version", "repeat_id", "run_id", "run_mode", "review_status",
              "source_id", "stage", "call_id", "grader", "grader_version", "metric", "status",
              "value", "numerator", "denominator", "reason")
    runs = {item.run.run_id: item for item in inputs.runs}
    slots = {slot.run_id: slot for slot in inputs.manifest.slots if slot.run_id is not None}
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        # 1. 【评测套件】【评分导出】直接复制冻结指标，不重新评分或用默认分数替换缺失
        for score in inputs.scores:
            run, slot = runs[score.run_id], slots[score.run_id]
            base = {"case_id": slot.case_id, "case_version": slot.case_version, "repeat_id": slot.repeat_id,
                    "run_id": score.run_id, "run_mode": run.run.run_mode, "review_status": run.review_status,
                    "source_id": score.source_id, "stage": score.stage, "call_id": score.call_id,
                    "grader": score.grading.grader, "grader_version": score.grading.grader_version}
            for metric in score.grading.measurements:
                writer.writerow({**base, "metric": metric.name, "status": metric.status, "value": metric.value,
                                 "numerator": metric.numerator, "denominator": metric.denominator, "reason": metric.reason})
        # 2. 【评测套件】【槽位保留】没有冻结运行的计划槽位仍保留，具体执行失败见 execution.json
        for slot in inputs.manifest.slots:
            if slot.run_id is None:
                writer.writerow({"case_id": slot.case_id, "case_version": slot.case_version,
                                 "repeat_id": slot.repeat_id, "status": "missing", "reason": "no_frozen_run_artifact"})
        if inputs.manifest.measurement_policy_snapshot is not None:
            unscored = {source.source_id for source in inputs.manifest.sources if source.kind != "planner" and source.ragas_path is None}
            unscored_runs = {score.run_id for score in inputs.scores if score.source_id in unscored}
            for run in inputs.runs:
                if run.run.run_id not in unscored_runs:
                    continue
                slot = slots[run.run.run_id]
                for name in ("context_precision", "context_recall"):
                    writer.writerow({"case_id": slot.case_id, "case_version": slot.case_version, "repeat_id": slot.repeat_id,
                                     "run_id": slot.run_id, "metric": name, "status": "missing", "reason": "grading_not_started"})


def report_exit_code(report: SuiteReport) -> int:
    """参数为已生成报告；缺运行、适用 P0 缺失/失败或比较无资格时返回非零。"""
    if any(slot.run_id is None for slot in report.summary.inputs.manifest.slots):
        return 1
    if any(check.status in {"failed", "missing"} for check in report.summary.checks):
        return 1
    if report.comparison is not None and report.comparison.status == "incomparable":
        return 1
    return 0
