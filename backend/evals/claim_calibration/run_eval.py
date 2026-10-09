"""Claim/Citation 校准数据校验、重放与报告入口。作者：xxx。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_backend_path() -> None:
    """无参数；脚本直接运行时注入 backend 模块路径。"""
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


if __package__ in {None, ""}:
    _bootstrap_backend_path()

from evals.claim_calibration.contracts import CalibrationRecord  # noqa: E402
from evals.claim_calibration.report import build_calibration_report, write_calibration_report  # noqa: E402


def load_records(path: Path) -> tuple[CalibrationRecord, ...]:
    """参数为 JSONL 路径；返回逐行严格校验的版本化校准记录。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[CalibrationRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number}: calibration record must be an object")
        records.append(CalibrationRecord.model_validate(payload))
    if not records:
        raise ValueError("claim calibration dataset is empty")
    return tuple(records)


def run_evaluation(path: Path, *, adapter: str, output: Path | None = None) -> dict[str, Any]:
    """参数为校准文件、运行模式和输出路径；返回校验或分层报告。"""
    records = load_records(path)
    if adapter == "validate":
        return {
            "schema_version": "claim-calibration-validation-v1", "record_count": len(records),
            "case_ids": [item.extraction.case_id for item in records],
            "human_quality_denominator": sum(item.extraction.human_quality_eligible for item in records),
        }
    if adapter not in {"annotation", "replay", "report"}:
        raise ValueError(f"unsupported claim calibration adapter: {adapter}")
    report = build_calibration_report(records)
    if adapter == "annotation":
        report = {**report, "adapter": "annotation", "judge_results_are_diagnostic": True}
    if output is not None:
        write_calibration_report(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    """无参数；返回校准命令行解析器。"""
    parser = argparse.ArgumentParser(description="Run Claim/Citation human calibration evaluation")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter", required=True, choices=("validate", "annotation", "replay", "report"))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """参数为命令行参数；打印报告并以明确错误返回非零状态。"""
    args = _parser().parse_args(argv)
    try:
        report = run_evaluation(args.dataset, adapter=args.adapter, output=args.output)
    except Exception as exc:
        print(f"claim calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
