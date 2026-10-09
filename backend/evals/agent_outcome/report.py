"""Agent 评测的分层报告和不可覆盖产物写入。作者：xxx。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import agent_record_contracts
from .evidence import collect_episode_facts, evaluate_predicates
from .scoring import AGENT_GRADER_VERSION, grade_agent_record

_STATUSES = ("measured", "not_measured", "missing", "invalid_input")


def build_agent_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """参数为冻结运行记录；返回逐题结果、状态覆盖和来源分层汇总。"""
    rows: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        case, run = agent_record_contracts(record)
        grading = grade_agent_record(record)
        facts = collect_episode_facts(case, run)
        rows.append({
            "case_id": case.case_id,
            "case_version": case.case_version,
            "title_zh": case.title_zh,
            "task_family": case.task_family,
            "review_status": case.review_status,
            "run_mode": run.run_mode,
            "run_status": run.status,
            "grading_result": grading,
            "missing_fields": list(run.missing_fields),
            "predicate_results": [item.model_dump(mode="json") for item in evaluate_predicates(case, facts)],
            "covered_span_ids": sorted(facts.covered_spans),
            "missing_span_ids": sorted(facts.missing_spans),
            "citation_issues": list(facts.delivered.issues),
        })
    return {
        "schema_version": "agent-outcome-report-v1",
        "grader": "agent_outcome",
        "grader_version": AGENT_GRADER_VERSION,
        "row_count": len(rows),
        "rows": rows,
        "coverage": _coverage(rows),
        "breakdowns": _breakdowns(rows),
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """参数为逐题报告行；按指标保留四类测量状态计数。"""
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for metric in row["grading_result"]["measurements"]:
            counters[metric["name"]][metric["status"]] += 1
    return {
        name: {status: counts[status] for status in _STATUSES}
        for name, counts in sorted(counters.items())
    }


def _breakdowns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """参数为逐题报告行；按运行环境和审核状态分开汇总，不合并均值。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["run_mode"], row["review_status"])].append(row)
    result: list[dict[str, Any]] = []
    for (run_mode, review_status), group in sorted(groups.items()):
        result.append({
            "run_mode": run_mode,
            "review_status": review_status,
            "case_count": len(group),
            "metrics": _group_metrics(group),
        })
    return result


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """参数为同一来源分组；返回仅基于 measured 样本的均值和完整覆盖计数。"""
    values: dict[str, list[float]] = defaultdict(list)
    states: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for metric in row["grading_result"]["measurements"]:
            name = metric["name"]
            states[name][metric["status"]] += 1
            if metric["status"] == "measured" and isinstance(metric.get("value"), (int, float)):
                values[name].append(float(metric["value"]))
    return {
        name: {
            "mean": sum(values[name]) / len(values[name]) if values[name] else None,
            "measured_count": len(values[name]),
            "statuses": {status: states[name][status] for status in _STATUSES},
        }
        for name in sorted(states)
    }


def render_agent_report(report: Mapping[str, Any]) -> str:
    """参数为结构化报告；返回面向复核的 Markdown，不泄漏评分真值快照。"""
    lines = [
        "# Agent Outcome 评测报告", "", "本报告仅展示冻结运行的派生判断；候选案例不会自动升级为人工金标。", "",
        f"案例数：{report.get('row_count', 0)}；评分版本：{report['grader_version']}", "", "## 分层汇总", "",
    ]
    for group in report.get("breakdowns", []):
        lines.append(f"- {group['run_mode']} / {group['review_status']}：{group['case_count']} 个案例")
        for name, metric in group["metrics"].items():
            lines.append(f"  - {name}: mean={metric['mean']}, measured={metric['measured_count']}, statuses={metric['statuses']}")
    lines.extend(["", "## 逐题结果", "", "| 案例 | 类型 | 模式 | 状态 | 指标 |", "|---|---|---|---|---|"])
    for row in report.get("rows", []):
        values = ", ".join(f"{item['name']}={item['status']}:{item.get('value')}" for item in row["grading_result"]["measurements"])
        lines.append(f"| {row['case_id']} | {row['task_family']} | {row['run_mode']} | {row['run_status']} | {values} |")
        for item in row.get("predicate_results", []):
            if item["status"] != "measured" or item["value"] != 1.0:
                lines.append(f"\n- {row['case_id']} / {item['name']}：{item['status']}，{item['reason']}")
        if row.get("missing_span_ids") or row.get("citation_issues"):
            lines.append(f"\n- 缺少片段：{row.get('missing_span_ids', [])}；出处问题：{row.get('citation_issues', [])}")
    return "\n".join(lines) + "\n"


def write_agent_report(output_dir: Path, report: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    """参数为输出目录、报告和冻结记录；写入 JSON/Markdown/JSONL，拒绝覆盖不同产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    frozen_text = "".join(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    _write_once(output_dir / "report.json", report_text)
    _write_once(output_dir / "report.md", render_agent_report(report))
    _write_once(output_dir / "frozen.jsonl", frozen_text)


def _write_once(path: Path, content: str) -> None:
    """参数为目标路径和内容；目标不存在时写入，内容不同则显式失败。"""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"agent report artifact differs: {path}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")
