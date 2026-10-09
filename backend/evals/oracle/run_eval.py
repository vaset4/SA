"""Oracle 配对诊断命令入口。作者：xxx。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .contracts import OraclePairRecord
from .report import build_oracle_report, write_oracle_report


def load_pair_records(path: Path) -> tuple[OraclePairRecord, ...]:
    """参数为冻结配对 JSONL；返回校验后的记录，不调用模型或工具。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[OraclePairRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"oracle pair at line {line_number} must be an object")
        records.append(OraclePairRecord.model_validate(payload))
    if not records:
        raise ValueError("oracle pair file must contain at least one record")
    return tuple(records)


def validate_pairs(path: Path) -> dict[str, Any]:
    """参数为冻结配对 JSONL；返回配对数量和身份校验结果。"""
    records = load_pair_records(path)
    return {
        "schema_version": "oracle-validation-v1",
        "pair_count": len(records),
        "pair_ids": [record.pair.pair_id for record in records],
        "diagnostic_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    """参数为命令行参数；返回成功或失败退出码。"""
    parser = argparse.ArgumentParser(description="Run Oracle retrieval attribution diagnostics")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter", required=True, choices=("validate", "replay"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.adapter == "validate":
            result = validate_pairs(args.dataset)
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            report = build_oracle_report(load_pair_records(args.dataset))
            if args.output_dir is not None:
                write_oracle_report(args.output_dir, report)
            else:
                result = report
        if args.output_dir is None:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    except Exception as exc:
        print(f"oracle evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
