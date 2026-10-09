"""Oracle 配对诊断报告和冻结产物。作者：xxx。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import OraclePairRecord


def build_oracle_report(records: Sequence[OraclePairRecord]) -> dict[str, Any]:
    """参数为冻结配对记录；返回只含诊断的结构化报告，不生成质量门禁分数。"""
    rows = [record.model_dump(mode="json") for record in records]
    labels: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        attribution = row["attribution"]
        statuses[attribution["status"]] = statuses.get(attribution["status"], 0) + 1
        for label in attribution["labels"]:
            labels[label] = labels.get(label, 0) + 1
    return {
        "schema_version": "oracle-report-v1",
        "diagnostic_only": True,
        "quality_gate_eligible": False,
        "pair_count": len(rows),
        "attribution_label_counts": dict(sorted(labels.items())),
        "attribution_status_counts": dict(sorted(statuses.items())),
        "pairs": rows,
    }


def render_oracle_report(report: Mapping[str, Any]) -> str:
    """参数为结构化 Oracle 报告；返回可读诊断 Markdown。"""
    lines = [
        "# Oracle Retrieval 配对诊断报告",
        "",
        "本报告仅描述可观察阶段症状，不证明唯一因果根因，也不进入默认质量门禁。",
        "",
        f"配对数：{report.get('pair_count', 0)}",
        "",
        "## 归因状态",
        "",
    ]
    for name, count in (report.get("attribution_status_counts") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## 归因候选", ""])
    for name, count in (report.get("attribution_label_counts") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## 配对记录", "", "| Pair | Actual | Oracle | Labels |", "|---|---|---|---|"])
    for row in report.get("pairs", []):
        labels = ", ".join(row["attribution"]["labels"])
        lines.append(f"| {row['pair']['pair_id']} | {row['actual']['status']} | {row['oracle']['status']} | {labels} |")
    return "\n".join(lines) + "\n"


def write_oracle_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    """参数为输出目录和结构化报告；写入 JSON/Markdown，拒绝覆盖不同内容。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_once(output_dir / "report.json", json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    _write_once(output_dir / "report.md", render_oracle_report(report))


def _write_once(path: Path, content: str) -> None:
    """参数为目标路径和内容；只允许幂等重写。"""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"oracle report artifact differs: {path}")
        return
    path.write_text(content, encoding="utf-8", newline="\n")
