"""Agent Outcome 评测命令入口和四种执行模式。作者：xxx。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Literal, cast

from .contracts import AgentCase, agent_record_contracts
from .environment import AgentEnvironment, GraphFactory
from .loader import load_agent_cases
from .report import build_agent_report, write_agent_report
from .runner import run_episode_sync


def validate_dataset(dataset_path: Path) -> dict[str, Any]:
    """参数为 Agent 场景路径；返回仅含案例结构和版本的校验报告。"""
    cases = load_agent_cases(dataset_path)
    families = sorted({case.task_family for case in cases})
    return {
        "schema_version": "agent-outcome-validation-v1",
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "task_families": families,
        "review_statuses": sorted({case.review_status for case in cases}),
    }


def load_frozen_runs(path: Path) -> list[dict[str, Any]]:
    """参数为冻结 JSONL 路径；返回校验身份后的运行记录，不调用被测系统。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"frozen agent record at line {line_number} must be an object")
        agent_record_contracts(payload)
        records.append(payload)
    if not records:
        raise ValueError("frozen agent run file must contain at least one record")
    return records


def run_evaluation(
    dataset_path: Path,
    *,
    adapter: str,
    output_dir: Path | None = None,
    environment_factory: Callable[[AgentCase], AgentEnvironment] | None = None,
    graph_factory: GraphFactory | None = None,
) -> dict[str, Any]:
    """参数为数据路径、模式、输出目录和可替换环境；返回结构化评测报告。"""
    if adapter == "validate":
        report = validate_dataset(dataset_path)
        return _write_validation(report, output_dir)
    if adapter == "replay":
        records = load_frozen_runs(dataset_path)
    elif adapter in {"fixture", "live"}:
        cases = load_agent_cases(dataset_path)
        records = _run_cases(cases, adapter, environment_factory, graph_factory)
    else:
        raise ValueError(f"unsupported agent adapter: {adapter}")
    report = build_agent_report(records)
    if output_dir is not None:
        write_agent_report(output_dir, report, records)
    return report


def _write_validation(report: dict[str, Any], output_dir: Path | None) -> dict[str, Any]:
    """参数为校验结果和可选目录；写入纯验证报告。"""
    if output_dir is None:
        return report
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation.json"
    content = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"validation artifact differs: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8", newline="\n")
    return report


def _run_cases(
    cases: tuple[AgentCase, ...],
    adapter: str,
    environment_factory: Callable[[AgentCase], AgentEnvironment] | None,
    graph_factory: GraphFactory | None,
) -> list[dict[str, Any]]:
    """参数为冻结案例、适配器和环境工厂；串行运行每个独立环境。"""
    if environment_factory is None and graph_factory is None:
        raise ValueError(f"{adapter} execution requires an environment or graph factory")
    records: list[dict[str, Any]] = []
    for case in cases:
        if environment_factory is not None:
            environment = environment_factory(case)
        else:
            if graph_factory is None:
                raise ValueError(f"{adapter} execution requires a graph factory")
            environment = AgentEnvironment(graph_factory=graph_factory)
        mode = cast(Literal["agent_fixture", "agent_live"], f"agent_{adapter}")
        records.append(run_episode_sync(case, environment, run_mode=mode))
    return records


def _parser() -> argparse.ArgumentParser:
    """无参数；返回 Agent 评测命令行解析器。"""
    parser = argparse.ArgumentParser(description="Run Agent Outcome evaluation")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter", required=True, choices=("validate", "fixture", "live", "replay"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--graph-factory", help="module:attribute graph factory for fixture/live")
    return parser


def main(argv: list[str] | None = None) -> int:
    """参数为命令行参数；返回成功或失败退出码并保留明确错误。"""
    args = _parser().parse_args(argv)
    try:
        graph_factory = _load_graph_factory(args.graph_factory) if args.graph_factory else None
        report = run_evaluation(
            args.dataset, adapter=args.adapter, output_dir=args.output_dir,
            graph_factory=graph_factory,
        )
    except Exception as exc:
        print(f"agent evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.output_dir is None:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _load_graph_factory(path: str) -> GraphFactory:
    """参数为 module:attribute 路径；返回可调用图工厂，格式错误显式失败。"""
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("graph factory path must use module:attribute format")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("graph factory must be callable")
    return cast(GraphFactory, factory)


if __name__ == "__main__":
    raise SystemExit(main())
