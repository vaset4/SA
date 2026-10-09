"""准备固定案例、实际执行及离线汇总比较的统一入口。作者：xxx。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, get_args

if TYPE_CHECKING:
    from evals.suites.contracts import SuiteInput


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """参数为可选命令行；解析准备、运行或只读报告命令及明确的比较因素。"""
    from evals.suites.contracts import ChangeFactor

    parser = argparse.ArgumentParser(description="汇总冻结评分并核对基线比较资格")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "report", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True, help="冻结套件清单 JSON")
        if name != "validate":
            command.add_argument("--output-dir", type=Path, required=True, help="尚不存在的报告目录")
        if name == "compare":
            command.add_argument("--baseline", type=Path, required=True, help="基线套件清单 JSON")
            command.add_argument("--changed-factors", nargs="*", choices=get_args(ChangeFactor), required=True,
                                 help="比较前声明可变化因素；仅提供选项而不跟值表示不允许变化")
    prepare = commands.add_parser("prepare", help="只读校验原文、业务归属及真实索引，固定案例")
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run", help="串行独立运行固定案例并保存报告")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--baseline", type=Path)
    run.add_argument("--changed-factors", nargs="*", choices=get_args(ChangeFactor))
    run.add_argument("--resume", action="store_true", help="在已有运行目录上续跑，认领上次已跑完的槽位")
    score = commands.add_parser("score", help="只对已冻结产品产物补做 CP/CR 评分，不重新运行产品")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--frozen", type=Path, help="单条冻结运行 JSONL 文件")
    source.add_argument("--manifest", type=Path, help="套件清单；只评分尚未开始的冻结槽位")
    score.add_argument("--output", type=Path, required=True, help="新评分产物路径；已存在文件拒绝覆盖")
    score.add_argument("--kind", choices=("retrieval", "agent"))
    score.add_argument("--measurement-policy", type=Path, help="单条评分必填；套件使用已冻结策略")
    score.add_argument("--retry-failed", action="store_true", help="套件显式追加失败评分的新attempt，保留原失败")
    replay = commands.add_parser("regrade-workflow", help="纯离线复评冻结Agent工作流；新目录保存版本和来源SHA，不调用模型")
    replay.add_argument("--frozen", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--grader-version", choices=("agent-workflow-v1", "agent-workflow-v2"), required=True)
    args = parser.parse_args(argv)
    if args.command == "score" and args.frozen is not None and (args.kind is None or args.measurement_policy is None):
        parser.error("single score requires --kind and --measurement-policy")
    return args


def execute_command(args: argparse.Namespace, inputs: SuiteInput) -> int:
    """参数为命令和已校验输入；输出实际报告，返回明确的校验或比较结果。"""
    from evals.suites.adapters import load_suite_sources
    from evals.suites.aggregate import aggregate_suite
    from evals.suites.compare import compare_suites
    from evals.suites.contracts import ComparisonRequest
    from evals.suites.report import build_suite_report, report_exit_code, write_suite_report

    summary = aggregate_suite(inputs)
    report = build_suite_report(inputs, summary)
    if args.command == "compare":
        baseline = aggregate_suite(load_suite_sources(args.baseline))
        request = ComparisonRequest(changed_factors=tuple(args.changed_factors))
        report = report.model_copy(update={"comparison": compare_suites(summary, baseline, request)})
    if args.command != "validate":
        write_suite_report(args.output_dir, report)
    exit_code = report_exit_code(report)
    print(json.dumps({"suite_id": inputs.manifest.suite_id, "status": "completed" if exit_code == 0 else "incomplete_or_failed",
                      "command": args.command, "runs": len(inputs.runs), "groups": len(summary.groups),
                      "comparison": None if report.comparison is None else report.comparison.status,
                      "output_dir": str(args.output_dir) if args.command != "validate" else None}, ensure_ascii=False))
    return exit_code


def execute_live_command(args: argparse.Namespace) -> int:
    """参数为准备或运行命令；产物先落盘再返回实际状态，错误不会变成质量分数。"""
    import asyncio

    from evals.suites.execute import execute_suite, load_prepared
    from evals.suites.execution_contracts import ExecutionRequest
    from evals.suites.prepare import load_profile, prepare_suite, write_new_json
    from evals.suites.runtime import LivePreparationRuntime

    resume = args.command == "run" and args.resume
    if args.output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite run directory: {args.output_dir}")
    if args.command == "prepare":
        try:
            profile = load_profile(args.profile)
        except ValueError as exc:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            write_new_json(args.output_dir / "preparation-failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            raise
        prepared = asyncio.run(prepare_suite(profile, args.output_dir, runtime=LivePreparationRuntime()))
        print(json.dumps({"status": "prepared", "prepared_id": prepared.prepared_id, "path": str(args.output_dir / "prepared.json")}, ensure_ascii=False))
        return 0
    if args.baseline is not None and args.changed_factors is None:
        raise ValueError("比较前必须显式提供 --changed-factors；仅提供选项而不跟值表示同设置复跑")
    request = ExecutionRequest(output_dir=args.output_dir, baseline=args.baseline,
                               changed_factors=tuple(args.changed_factors or ()), resume=resume)
    manifest = execute_suite(load_prepared(args.prepared), request)
    result = json.loads((manifest.parent / "result.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": "completed" if result["exit_code"] == 0 else "incomplete_or_failed", "manifest": str(manifest), **result}, ensure_ascii=False))
    return int(result["exit_code"])


def execute_score_command(args: argparse.Namespace) -> int:
    """参数为单条冻结产物、输出路径和评分策略；只调用 judge，不重新执行产品。"""
    from evals.contracts import read_snapshot
    from evals.suites.policy import MeasurementPolicy
    from evals.suites.prepare import sha256_file
    from evals.suites.scoring import score_attempt, score_suite
    from evals.suites.adapters import load_suite_sources

    if args.manifest is not None:
        if args.kind is not None or args.measurement_policy is not None:
            raise ValueError("suite_score_uses_frozen_kind_and_policy")
        path = score_suite(args.manifest, output=args.output, retry_failed=args.retry_failed)
        inputs = load_suite_sources(path)
        failed = any(source.kind != "planner" and source.ragas_path is None for source in inputs.manifest.sources)
        failed = failed or any(slot.run_id is None for slot in inputs.manifest.slots)
        failed = failed or any(read_snapshot(score.detail_snapshot).get("grading_status") == "failed"
                               for score in inputs.scores if score.grading.grader == "ragas_frozen")
        print(json.dumps({"manifest": str(path), "status": "incomplete_or_failed" if failed else "completed"}, ensure_ascii=False))
        return int(failed)
    if args.kind is None or args.measurement_policy is None or args.retry_failed:
        raise ValueError("single_score_requires_kind_and_policy_without_retry_failed")

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite grading artifact: {args.output}")
    lines = [line for line in args.frozen.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("score requires exactly one frozen record")
    record = read_snapshot(lines[0])
    policy = MeasurementPolicy.model_validate_json(args.measurement_policy.read_text(encoding="utf-8-sig"))
    result = score_attempt(record, kind=args.kind, policy=policy, output=args.output,
                           provenance={"frozen_path": str(args.frozen), "frozen_sha256": sha256_file(args.frozen)})
    status = result.get("grading_status", "completed")
    print(json.dumps({"status": status, "output": str(args.output), "run_id": result["run_id"],
                      "grading_error": result.get("grading_error")}, ensure_ascii=False))
    return 0 if status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    """参数为可选命令行；仅 prepare/run 连接实际依赖，其余命令保持只读行为。"""
    backend_root = str(Path(__file__).resolve().parents[2])
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    from evals.suites.adapters import load_suite_sources

    args = parse_args(argv)
    try:
        if args.command == "regrade-workflow":
            from evals.suites.workflow_replay import regrade_workflow

            path = regrade_workflow(args.frozen, output_dir=args.output_dir, grader_version=args.grader_version)
            print(json.dumps({"status": "graded", "report": str(path), "grader_version": args.grader_version}, ensure_ascii=False))
            return 0
        if args.command == "score":
            return execute_score_command(args)
        if args.command in {"prepare", "run"}:
            return execute_live_command(args)
        return execute_command(args, load_suite_sources(args.manifest))
    except Exception as exc:
        status = "failed" if args.command in {"prepare", "run"} else "invalid_input"
        print(json.dumps({"status": status, "error_type": type(exc).__name__, "error": f"【评测套件】【{args.command}】{exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
