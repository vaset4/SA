"""Run deterministic replay evaluation for research planner payloads."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _bootstrap_backend_path() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    backend_root_str = str(backend_root)
    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)


if __package__ in {None, ""}:
    _bootstrap_backend_path()

from evals.research_planner.metrics import (  # noqa: E402
    CATEGORY_NAMES,
    DEPTH_ORDER,
    FAILURE_STAGES,
    METRIC_NAMES,
    MINIMAL_DEPTH,
    NOT_MEASURED,
    score_records,
)
from evals.research_planner.live_fixture import build_live_records  # noqa: E402
from evals.contracts import CONTRACT_VERSION, PROJECTION_VERSION, measurement_lines  # noqa: E402

DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "research_planner_eval.jsonl"
DEFAULT_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
MIN_RECORDS = 20
REQUIRED_RECORD_FIELDS = ("id", "category", "question", "reference_answer", "expected", "replay")
REQUIRED_CONTRACT_FIELDS = ("golden_evidence", "critical_facts", "rubric")

# 合规类硬闸门（prd.md §13.2 判定列 / §13.3 闸门策略）。
# 这三项衡量的是约束有没有被违反，没有「比较好」的中间态，所以各自要么达标要么判失败；
# 准确率类只记基线不进这张表，为通过闸门去调阈值就等于让评测追认实现。
# mode=min 表示「不得低于」，mode=max 表示「不得高于」
COMPLIANCE_GATES: tuple[tuple[str, str, float], ...] = (
    ("citation_verifiability_rate", "min", 1.0),
    ("memory_injection_compliance_rate", "min", 1.0),
    ("stale_memory_leak_rate", "max", 0.0),
)


def load_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        _validate_record(payload, line_number=line_number)
        records.append(payload)
    if not records:
        raise ValueError(f"research planner eval dataset is empty: {path}")
    return records


def run_evaluation(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    write_report: bool = False,
) -> dict[str, Any]:
    records = load_dataset(dataset_path)
    live_records = build_live_records(records)
    scored = score_records(live_records)
    report = _build_report(
        dataset_path=Path(dataset_path),
        records=live_records,
        scored=scored,
    )
    if not write_report:
        return report
    return _write_reports(report, output_dir=Path(output_dir))


def evaluate_compliance_gate(
    metrics: Mapping[str, Any], *, measurement_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """按合规硬闸门判这次评测通不通过。

    参数:
        metrics: 本次评测的聚合指标，取值可能是样本均值也可能是 NOT_MEASURED
        measurement_coverage: 新报告逐指标覆盖，已适用样本存在缺口时禁止仅凭均值放行
    返回:
        {"passed": bool, "violations": [...]}；违规项逐条记指标名、要求与实际值，
        报告与 CLI 都据它说明红在哪
    """
    violations = [
        violation
        for violation in (_gate_violation(metrics, gate) for gate in COMPLIANCE_GATES)
        if violation is not None
    ]
    for name, _mode, _bound in COMPLIANCE_GATES:
        counts = (measurement_coverage or {}).get(name, {})
        if counts.get("missing_count", 0) or counts.get("invalid_input_count", 0):
            violations.append({"metric": name, "requirement": "complete applicable observations",
                               "actual": counts, "reason": "incomplete_measurement"})
    return {"passed": not violations, "violations": violations}


def _gate_violation(metrics: Mapping[str, Any], gate: tuple[str, str, float]) -> dict[str, Any] | None:
    """判单条闸门。

    参数:
        metrics: 聚合指标
        gate: (指标名, min/max, 阈值)
    返回:
        违规明细；达标时返回 None。**测不出即判不通过** —— 把「没测」当合格放过，
        闸门会永远绿着，而合规类指标测不出本身就说明样本集或投影有漏
    """
    name, mode, bound = gate
    actual = metrics.get(name, NOT_MEASURED)
    requirement = f"{'>=' if mode == 'min' else '<='} {bound}"
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return {"metric": name, "requirement": requirement, "actual": actual, "reason": "not_measured"}
    if (mode == "min" and actual < bound) or (mode == "max" and actual > bound):
        return {"metric": name, "requirement": requirement, "actual": actual, "reason": "threshold"}
    return None


def _validate_record(record: Any, *, line_number: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: record must be an object")
    _require_fields(record, REQUIRED_RECORD_FIELDS, line_number=line_number)
    # 三个契约字段每条样本都得有键，但只有要论文事实的样本要求非空：
    # minimal 回合本轮不检索论文，硬要金标证据等于给它编一份取不到的期望
    _require_keys(record, REQUIRED_CONTRACT_FIELDS, line_number=line_number)
    _require_fields(record, _contract_fields_for(record), line_number=line_number)
    if record["category"] not in CATEGORY_NAMES:
        raise ValueError(f"line {line_number}: invalid category {record['category']!r}")
    if not isinstance(record.get("expected"), dict):
        raise ValueError(f"line {line_number}: expected must be an object")
    if not isinstance(record.get("replay"), dict):
        raise ValueError(f"line {line_number}: replay must be an object")
    expected_stage = str(record["expected"].get("failure_stage") or "none").strip()
    if expected_stage not in FAILURE_STAGES:
        raise ValueError(f"line {line_number}: invalid failure_stage {expected_stage!r}")
    # 预期深度是可选标注，但写错档位会让深度判断准确率整列掉到 0，看起来像深度策略全错，
    # 所以在读样本时就拒掉；没标这个键是当前的正常状态，不在此处要求补齐
    expected_depth = str(record["expected"].get("depth") or "").strip()
    if expected_depth and expected_depth not in DEPTH_ORDER:
        raise ValueError(f"line {line_number}: invalid expected depth {expected_depth!r}")


def _require_fields(record: dict[str, Any], fields: tuple[str, ...], *, line_number: int) -> None:
    missing = [field for field in fields if not _present(record.get(field))]
    if missing:
        raise ValueError(f"line {line_number}: missing required fields {missing}")


def _require_keys(record: dict[str, Any], fields: tuple[str, ...], *, line_number: int) -> None:
    """只查键在不在，不管值空不空。"""
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"line {line_number}: missing required fields {missing}")


def _contract_fields_for(record: dict[str, Any]) -> tuple[str, ...]:
    """这条样本里必须非空的契约字段。

    参数:
        record: 一条评测样本，按 expected.depth 分档
    返回:
        必须非空的字段名；minimal 只要求 rubric，其余深度三项齐全
    """
    expected = record.get("expected")
    depth = str(expected.get("depth") or "").strip() if isinstance(expected, dict) else ""
    if depth == MINIMAL_DEPTH:
        return ("rubric",)
    return REQUIRED_CONTRACT_FIELDS


def _build_report(
    *,
    dataset_path: Path,
    records: list[dict[str, Any]],
    scored: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": _utc_run_id(),
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.stem,
        "adapter": "planner_live_fixture",
        "contract_version": CONTRACT_VERSION,
        "projection_version": PROJECTION_VERSION,
        "sample_count": len(records),
        "category_count": len({record["category"] for record in records}),
        "metric_names": list(METRIC_NAMES),
        "metrics": scored["metrics"],
        "planner_metrics": scored["metrics"],
        "compliance_gate": evaluate_compliance_gate(
            scored["metrics"], measurement_coverage=scored["coverage_summary"]["measurements"]),
        "records": scored["records"],
        "coverage_summary": scored["coverage_summary"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_reports(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    run_dir = output_dir / str(report["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "report.json"
    markdown_path = run_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_build_markdown_report(report), encoding="utf-8")
    return {
        **report,
        "output_dir": str(run_dir),
        "json_report_path": str(json_path),
        "markdown_report_path": str(markdown_path),
    }


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Research Planner Replay Eval",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- dataset_path: `{report['dataset_path']}`",
        f"- adapter: `{report['adapter']}`",
        f"- sample_count: {report['sample_count']}",
        "",
        "## Planner Metrics",
        "",
    ]
    lines.extend(_metric_lines(report.get("metrics", {})))
    lines.extend(["", "## Compliance Gate", ""])
    lines.extend(_gate_lines(report.get("compliance_gate", {})))
    lines.extend(["", "## Coverage", ""])
    lines.extend(_coverage_lines(report.get("coverage_summary", {})))
    lines.extend(measurement_lines(report.get("coverage_summary", {}).get("measurements", {})))
    lines.extend(["", "## Records", ""])
    for record in report.get("records", []):
        lines.extend(_record_lines(record))
    return "\n".join(lines).strip() + "\n"


def _metric_lines(metrics: dict[str, Any]) -> list[str]:
    """
    渲染指标行。

    参数:
        metrics: 指标名 → 取值，取值可能是样本均值也可能是 NOT_MEASURED
    返回:
        markdown 列表行；键缺失时落到 NOT_MEASURED 而不是 0.0，
        0.0 会把「这项没数据」显示成「这项全错」
    """
    return [f"- {name}: {metrics.get(name, NOT_MEASURED)}" for name in METRIC_NAMES]


def _gate_lines(gate: dict[str, Any]) -> list[str]:
    """
    渲染合规闸门结论。

    参数:
        gate: evaluate_compliance_gate 的返回
    返回:
        markdown 列表行；不通过时逐条列出指标、要求与实际值，让红因一眼可见
    """
    lines = [f"- passed: {bool(gate.get('passed'))}"]
    for violation in gate.get("violations") or []:
        lines.append(
            f"- violation: `{violation['metric']}` requires {violation['requirement']}, "
            f"actual `{violation['actual']}` ({violation['reason']})"
        )
    return lines


def _coverage_lines(summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for section in ("categories", "failure_stages"):
        values = summary.get(section) if isinstance(summary, dict) else {}
        if isinstance(values, dict):
            lines.append(f"- {section}: {json.dumps(values, ensure_ascii=False, sort_keys=True)}")
    return lines


def _record_lines(record: dict[str, Any]) -> list[str]:
    """参数为单条评分；返回实际/预期失败及冻结记录来源、缺口的可读摘要。"""
    return [
        f"### {record.get('id', '')}",
        "",
        f"- category: `{record.get('category', '')}`",
        f"- failure_stage: `{record.get('failure_stage', '')}`",
        f"- expected_failure_stage: `{record.get('expected_failure_stage', '')}`",
        f"- run_mode: `{record.get('run_record', {}).get('run_mode', '')}`",
        f"- provenance: `{record.get('run_record', {}).get('provenance', '')}`",
        f"- missing_fields: {record.get('run_record', {}).get('missing_fields', [])}",
        f"- question: {record.get('question', '')}",
        "",
    ]


def _present(value: Any) -> bool:
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return value is not None and str(value).strip() != ""


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace(".", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StudyAssistant research planner replay eval")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to planner JSONL dataset")
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPERIMENTS_DIR), help="Directory for reports")
    parser.add_argument("--report", action="store_true", help="Write report.json and report.md")
    args = parser.parse_args()
    result = run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        write_report=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 合规类不达标就让这次运行非零退出：闸门守的是数据集整体表现，
    # 与单元测试里的门禁断言各守一头（prd.md:405 与 implement.md:193 的口径都要）
    gate = result.get("compliance_gate") or {}
    if not gate.get("passed"):
        for violation in gate.get("violations") or []:
            print(
                f"compliance gate failed: {violation['metric']} requires {violation['requirement']}, "
                f"actual {violation['actual']} ({violation['reason']})",
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
