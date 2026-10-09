"""串行独立进程执行固定案例，冻结真实结果并复用既有汇总比较。作者：xxx。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from evals.agent_outcome.contracts import AgentCase
from evals.agent_outcome.report import build_agent_report, render_agent_report
from evals.contracts import MetricResult, build_grading, create_case, read_snapshot, record_contracts, snapshot, snapshot_hash
from evals.evidence.corpus import load_json_object, validate_case_assets
from evals.evidence.report import build_evidence_report, render_evidence_report

from .contracts import ComparisonRequest, IdentityCondition, RunConditions, RunSlot, SuiteReport
from .execution_contracts import DEFAULT_WORKER_TIMEOUT_SECONDS, ExecutionRequest, PreparedCase, PreparedJob, PreparedSuite, ProducedRun
from .identity import RuntimeBindings, build_run_manifest, capture_conditions, verify_conditions, verify_model_requests, write_recorded_identity
from .prepare import WORKSPACE_ROOT, profile_path, write_new_json, write_replaced_json
from .policy import MeasurementPolicy
from .runtime import AgentRuntime, LivePreparationRuntime, RetrievalRuntime, export_secrets, run_agent_job, run_retrieval_job, sanitize_frozen_record

PLANNER_SUITE_GRADER_VERSION = "planner-structure-suite-v1"
PLANNER_SUITE_METRICS = frozenset({"plan_valid_rate"})
CANCELLATION_GRACE_SECONDS = 15.0
PROCESS_TERMINATION_TIMEOUT_SECONDS = 15.0
WORKER_STOP_POLL_SECONDS = 0.25


class ExportLogFormatter(logging.Formatter):
    """只在本次worker日志导出时脱敏，异常对象与原日志记录保持不变。"""

    def __init__(self, wrapped: logging.Formatter | None, *, secrets: tuple[str, ...]) -> None:
        """参数为原格式器及实际凭证；保留原日志格式，仅过滤导出文本。"""
        super().__init__()
        self.wrapped = wrapped or logging.Formatter()
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        """参数为日志及异常链；返回脱敏文本，避免Formatter缓存原始异常污染其他输出。"""
        rendered = self.wrapped.format(copy(record))
        return sanitize_frozen_record({"message": rendered}, secrets=self.secrets)["message"]


def load_prepared(path: Path) -> PreparedSuite:
    """参数为固定输入文件；校验整个内容身份后返回，不读取当前候选补写历史题目。"""
    return PreparedSuite.model_validate(load_json_object(path))


def _slot_key(job: PreparedJob, slot: RunSlot) -> str:
    """参数为事前任务和槽位；返回与槽位目录、日志文件一致的稳定键。"""
    return snapshot_hash(snapshot({"job": job.job_id, "case": slot.case_id, "repeat": slot.repeat_id}))


def execute_job(job: PreparedJob, slot: RunSlot, *, prepared_path: Path, output_dir: Path,
                timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS) -> ProducedRun:
    """参数为事前任务、槽位和新目录；通过显式 argv 启动独立进程，完成信息仅走小文件。"""
    key = _slot_key(job, slot)
    target = output_dir / "runs" / key
    logfile = output_dir / "logs" / f"{key}.log"
    argv = [sys.executable, "-m", "evals.suites.execute", "--worker", "--prepared", str(prepared_path.resolve()),
            "--job-id", job.job_id, "--case-id", slot.case_id, "--repeat-id", slot.repeat_id, "--output-dir", str(target.resolve())]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    started = time.perf_counter()
    try:
        with logfile.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(argv, cwd=WORKSPACE_ROOT / "backend", env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                       start_new_session=os.name != "nt")
            try:
                process.wait(timeout=timeout_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "timed_out"
                try:
                    cleanup = _stop_worker(process, target=target)
                except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                    cleanup = {"pid": process.pid, "cleanup_error": f"{type(cleanup_error).__name__}:{cleanup_error}",
                               "worker_exit_code": process.poll()}
                return ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version,
                                   repeat_id=slot.repeat_id, status=status, error=f"worker_{status}",
                                   elapsed_seconds=time.perf_counter() - started, cleanup_snapshot=snapshot(cleanup))
        result_path = target / "produced.json"
        if not result_path.is_file():
            raise RuntimeError(f"worker_exit={process.returncode}; completion_artifact_missing; log={logfile}")
        produced = ProducedRun.model_validate(load_json_object(result_path))
        _verify_worker_result(produced, job, slot=slot, target=target)
        if process.returncode != 0 and produced.status == "completed":
            return produced.model_copy(update={"status": "failed", "error": f"worker_exit={process.returncode}; completion_status_conflict"})
        return produced.model_copy(update={"elapsed_seconds": time.perf_counter() - started})
    except (OSError, ValueError, RuntimeError) as exc:
        return ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version,
                           repeat_id=slot.repeat_id, status="failed", error=f"{type(exc).__name__}:{exc}")


def _stop_worker(process: subprocess.Popen[Any], *, target: Path) -> dict[str, Any]:
    """参数为本次子进程及产物目录；先请求取消，超时再终止其进程树，返回实际清理状态。"""
    cleanup: dict[str, Any] = {"pid": process.pid, "cancel_requested": target.is_dir(), "forced": False}
    if target.is_dir():
        write_new_json(target / "cancel-request.json", {"reason": "suite_worker_deadline_or_user_cancel"})
    try:
        process.wait(timeout=CANCELLATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        cleanup["forced"] = True
        if os.name == "nt":
            result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                                    encoding="oem", timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS, check=False,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            cleanup["process_tree_exit_code"] = result.returncode
            cleanup["process_tree_output"] = result.stdout + result.stderr
        else:
            getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
        process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
    cleanup["worker_exit_code"] = process.returncode
    leases = target / "active-leases.json"
    if leases.is_file():
        try:
            asyncio.run(_release_saved_leases(leases))
            cleanup["lease_release"] = "completed"
        except Exception as exc:
            cleanup["lease_release"] = f"failed:{type(exc).__name__}:{exc}"
    return cleanup


async def _release_saved_leases(path: Path) -> None:
    """参数为本次进程已经持久化的租约文件；释放准确占用，不清理用户业务数据。"""
    from app.repositories.upload_publication_repository import get_upload_publication_repository

    for scope in load_json_object(path)["scopes"]:
        await get_upload_publication_repository().release_snapshot(scope["publication"])


def _verify_worker_result(produced: ProducedRun, job: PreparedJob, *, slot: RunSlot, target: Path) -> None:
    """参数为子进程完成信息和声明槽位；防止错误结果、旧产物或目录外产物填入本次运行。"""
    expected = (job.job_id, slot.case_id, slot.case_version, slot.repeat_id)
    actual = (produced.job_id, produced.case_id, produced.case_version, produced.repeat_id)
    if expected != actual:
        raise ValueError("worker_completed_another_execution_slot")
    for value in (produced.report_path, produced.frozen_path, produced.identity_path):
        if value is not None and (not Path(value).resolve().is_relative_to(target.resolve()) or not Path(value).is_file()):
            raise ValueError("worker_artifact_missing_or_outside_slot")


def build_report_from_manifest(manifest_path: Path, request: ExecutionRequest) -> SuiteReport:
    """参数为本次清单和可选比较请求；复用已有离线汇总及比较算法。"""
    from .adapters import load_suite_sources
    from .aggregate import aggregate_suite
    from .compare import compare_suites
    from .report import build_suite_report

    inputs = load_suite_sources(manifest_path)
    summary = aggregate_suite(inputs)
    report = build_suite_report(inputs, summary)
    if request.baseline is not None:
        baseline = aggregate_suite(load_suite_sources(request.baseline))
        comparison = compare_suites(summary, baseline, ComparisonRequest(changed_factors=request.changed_factors))
        report = report.model_copy(update={"comparison": comparison})
    return report


def execute_suite(prepared: PreparedSuite, request: ExecutionRequest) -> Path:
    """参数为固定输入和本次请求；串行执行全部槽位，保存真实产物后汇总比较。"""
    output = request.output_dir.resolve()
    started = time.perf_counter()
    prepared_path = _open_run_directory(output, prepared, resume=request.resume)
    produced: list[ProducedRun] = []
    resumed: list[str] = []
    _run_all_slots(prepared, output=output, prepared_path=prepared_path,
                   produced=produced, resumed=resumed, started=started)
    return _finalize_suite(prepared, request, output=output, produced=produced)


def _open_run_directory(output: Path, prepared: PreparedSuite, *, resume: bool) -> Path:
    """参数为运行目录、固定输入和是否续跑；新建目录，或核对续跑目录里的固定输入身份。"""
    prepared_path = output / "prepared.json"
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        (output / "runs").mkdir()
        (output / "logs").mkdir()
        write_new_json(prepared_path, prepared.model_dump(mode="json"))
        return prepared_path
    if not output.is_dir():
        raise FileNotFoundError(f"续跑目录不存在: {output}")
    if not prepared_path.is_file():
        raise FileNotFoundError(f"续跑目录缺少固定输入: {prepared_path}")
    if not (output / "runs").is_dir() or not (output / "logs").is_dir():
        # 只有真正跑过的目录才可能有槽位残留；prepare 目录误传进来要立刻停，不空跑整轮
        raise FileNotFoundError(f"续跑目录不是一次实际执行: {output}")
    if load_prepared(prepared_path) != prepared:
        # 续跑必须复用同一份固定输入；换了题集就是新一轮，不能混进旧目录
        raise ValueError("resume_prepared_differs_from_directory")
    return prepared_path


def _write_execution_ledger(output: Path, *, prepared: PreparedSuite, produced: list[ProducedRun],
                            started: float, suite_status: str, abort_reason: str | None, resumed: list[str]) -> None:
    """参数为运行目录与本轮进度；原子重写执行账本，中断后单看该文件即可还原进度。

    读取侧判定：suite_status 为 finished 表示整轮结束；仍为 in_progress 时，
    abort_reason 为 user_interrupt 表示人为中断，为 None 表示进程被外部终止（关机、强杀），
    两种情况都保留真实槽位状态，不把未跑完的槽位改写成成功。
    """
    write_replaced_json(output / "execution.json", {
        "prepared_id": prepared.prepared_id,
        "suite_status": suite_status,
        "abort_reason": abort_reason,
        "suite_seconds": time.perf_counter() - started,
        "worker_timeout_seconds": prepared.profile.worker_timeout_seconds,
        "resumed_from_previous_run": list(resumed),
        "runs": [run.model_dump(mode="json") for run in produced],
    })


def _claim_previous_slot(job: PreparedJob, slot: RunSlot, *, output_dir: Path) -> tuple[ProducedRun | None, str | None]:
    """参数为事前任务、槽位和已有运行目录；上次已留下终态记录的槽位直接认领，不重复调用真实依赖。

    completed 与 failed 都算终态：上次真实跑出的失败记录也是证据，认领后不会因重跑被覆盖；
    cancelled 只是被中断的半成品，没有可复用的完整轨迹，必须重跑。
    """
    target = output_dir / "runs" / _slot_key(job, slot)
    result_path = target / "produced.json"
    if not result_path.is_file():
        return None, "no_previous_record"
    try:
        produced = ProducedRun.model_validate(load_json_object(result_path))
    except Exception as exc:
        return None, f"unreadable_previous_record:{type(exc).__name__}"
    if produced.status not in {"completed", "failed"}:
        return None, f"previous_slot_{produced.status}"
    try:
        _verify_worker_result(produced, job, slot=slot, target=target)
    except ValueError as exc:
        return None, f"previous_slot_identity:{exc}"
    return produced, None


def _archive_incomplete_slot(target: Path, *, logfile: Path) -> Path | None:
    """参数为上次中断留下的槽目录及日志；重命名保留现场，让本次执行能独占新建同名目录。"""
    if not target.exists():
        return None
    attempt = 1
    while target.with_name(f"{target.name}.attempt-{attempt}").exists():
        attempt += 1
    archived = target.rename(target.with_name(f"{target.name}.attempt-{attempt}"))
    if logfile.is_file():
        logfile.rename(logfile.with_name(f"{logfile.stem}.attempt-{attempt}{logfile.suffix}"))
    return archived


def _execute_or_claim(job: PreparedJob, slot: RunSlot, *, prepared_path: Path, output: Path,
                      timeout_seconds: float) -> tuple[ProducedRun, bool]:
    """参数为一个槽位和运行目录；上次的终态记录直接认领，其余让开残留目录后真实执行。"""
    claimed, reason = _claim_previous_slot(job, slot, output_dir=output)
    label = f"{job.job_id} / {slot.case_id} / {slot.repeat_id}"
    if claimed is not None:
        print(f"【评测套件】【续跑】认领上次结果 {claimed.status} {label}", flush=True)
        return claimed, True
    archived = _archive_incomplete_slot(output / "runs" / _slot_key(job, slot),
                                        logfile=output / "logs" / f"{_slot_key(job, slot)}.log")
    if archived is not None:
        print(f"【评测套件】【续跑】归档未完成残留 {archived.name}（{reason}）{label}", flush=True)
    return execute_job(job, slot, prepared_path=prepared_path, output_dir=output, timeout_seconds=timeout_seconds), False


def _read_single_frozen_record(path: Path) -> dict[str, Any]:
    """参数为单槽冻结JSONL；返回唯一产品记录，拒绝拼接或挑选多条评分输入。"""
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("frozen_artifact_must_contain_one_record")
    return read_snapshot(lines[0])


def _score_completed_run(prepared: PreparedSuite, job: PreparedJob, produced: ProducedRun) -> ProducedRun:
    """参数为已完成产品运行；只从冻结产物独立评分，评分失败不改变产品终态。"""
    policy = prepared.profile.measurement_policy
    if (prepared.profile.grading_timing == "deferred" or policy is None or job.kind not in {"retrieval", "agent"} or produced.status != "completed"
            or produced.grading_status != "not_started" or produced.frozen_path is None):
        return produced
    from evals.ragas.frozen_bridge import build_failed_score, score_frozen_record

    kind = "agent" if job.kind == "agent" else "retrieval"
    target = Path(produced.frozen_path).resolve().parent
    started = time.perf_counter()
    try:
        record = _read_single_frozen_record(Path(produced.frozen_path))
        ragas = asyncio.run(score_frozen_record(record, kind=kind, policy=policy))
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        logging.getLogger(__name__).error("【评测套件】【独立评分】冻结产物评分失败 error_type=%s", type(exc).__name__)
        record = record if "record" in locals() else _read_single_frozen_record(Path(produced.frozen_path))
        ragas = build_failed_score(record, kind=kind, policy=policy, error=error)
    safe = sanitize_frozen_record(ragas, secrets=export_secrets(job.kind))
    write_replaced_json(target / "ragas.json", safe)
    write_replaced_json(target / "grading-resources.json", {"grading_seconds": time.perf_counter() - started})
    grading_error = safe.get("grading_error")
    grading_status = "failed" if safe.get("grading_status") == "failed" else "completed"
    write_replaced_json(target / "grading-status.json", {"status": grading_status, "error": grading_error})
    scored = produced.model_copy(update={"grading_status": grading_status, "grading_error": grading_error})
    write_replaced_json(target / "produced.json", scored.model_dump(mode="json"))
    return scored


def _run_all_slots(prepared: PreparedSuite, *, output: Path, prepared_path: Path,
                   produced: list[ProducedRun], resumed: list[str], started: float) -> None:
    """参数为固定输入和运行目录；逐槽执行或认领，每槽结束刷新账本，中断先留痕再上抛。"""
    cancelled = False
    total = sum(len(job.slots) for job in prepared.jobs)
    try:
        for job in prepared.jobs:
            for slot in job.slots:
                print(f"【评测套件】【执行】{len(produced) + 1}/{total} {job.job_id} / {slot.case_id} / {slot.repeat_id}", flush=True)
                if cancelled:
                    # 上一槽真实取消后，剩余槽位如实记为未跑，不偷偷补默认结果
                    result, claimed = ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version,
                                                  repeat_id=slot.repeat_id, status="not_run",
                                                  error="suite_cancelled_before_slot"), False
                else:
                    result, claimed = _execute_or_claim(job, slot, prepared_path=prepared_path, output=output,
                                                        timeout_seconds=prepared.profile.worker_timeout_seconds)
                    # 认领的是上一轮已定性的结果，不再参与本轮清理判断，否则上次的残留会级联停掉本次剩余槽位
                    cleanup_failed = (not claimed and result.cleanup_snapshot is not None
                                      and "cleanup_error" in read_snapshot(result.cleanup_snapshot))
                    cancelled = result.status == "cancelled" or cleanup_failed
                produced.append(result)
                if claimed:
                    resumed.append(f"{job.job_id}:{slot.case_id}:{slot.repeat_id}:{result.status}")
                _write_execution_ledger(output, prepared=prepared, produced=produced, started=started,
                                        suite_status="in_progress", abort_reason=None, resumed=resumed)
                if not cancelled and prepared.profile.measurement_policy is not None:
                    scored = _score_completed_run(prepared, job, result)
                    if scored != result:
                        produced[-1] = scored
                        _write_execution_ledger(output, prepared=prepared, produced=produced, started=started,
                                                suite_status="in_progress", abort_reason=None, resumed=resumed)
    except BaseException as exc:
        # 中断也必须留下可读进度：人为取消与外部终止用 abort_reason 区分，异常继续上抛不吞
        reason = "user_interrupt" if isinstance(exc, KeyboardInterrupt) else None
        _write_execution_ledger(output, prepared=prepared, produced=produced, started=started,
                                suite_status="in_progress", abort_reason=reason, resumed=resumed)
        raise
    _write_execution_ledger(output, prepared=prepared, produced=produced, started=started,
                            suite_status="finished", abort_reason=None, resumed=resumed)


def _finalize_suite(prepared: PreparedSuite, request: ExecutionRequest, *, output: Path,
                    produced: list[ProducedRun]) -> Path:
    """参数为固定输入和实际产物；写出执行报告与汇总比较，没有完整产物时明确失败。"""
    from .report import render_execution_report, report_exit_code, write_suite_report

    # 清单、汇总与结果都由本次实际产物派生；续跑在同一目录重算，先清掉上次版本，否则独占写入会拒改
    (output / "manifest.json").unlink(missing_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    shutil.rmtree(output / "summary", ignore_errors=True)
    (output / "execution-report.md").write_text(
        render_execution_report([run.model_dump(mode="json") for run in produced]), encoding="utf-8", newline="\n")
    manifest_path = build_run_manifest(prepared, tuple(produced), output_dir=output)
    if manifest_path is None:
        write_new_json(output / "result.json", {"exit_code": 1, "reason": "no_complete_run_artifacts", "execution_path": "execution.json"})
        raise RuntimeError(f"没有可形成清单的实际产物；具体失败见 {output / 'execution.json'}")
    report = build_report_from_manifest(manifest_path, request)
    write_suite_report(output / "summary", report)
    failed = any(run.status != "completed" for run in produced)
    write_new_json(output / "result.json", {"exit_code": 1 if failed else report_exit_code(report),
                                             "manifest_path": "manifest.json", "report_path": "summary/report.md"})
    return manifest_path


async def _verify_live_inputs(prepared: PreparedSuite, live: LivePreparationRuntime, *, case: PreparedCase, kind: str) -> None:
    """参数为固定输入和实际依赖；执行前后核对归属、原文和完整索引，不自行修补业务数据。"""
    profile = prepared.profile
    actual = await live.resolve_scope(profile, paper_ids=tuple(item.paper_id for item in profile.paper_bindings))
    if snapshot(actual) != prepared.scope_snapshot:
        raise ValueError("prepared_business_scope_changed")
    await live.verify_bindings(profile)
    record = read_snapshot(case.record_snapshot)
    inputs = read_snapshot(AgentCase.model_validate(record).input_snapshot) if kind == "agent" else read_snapshot(create_case(record).input_snapshot)
    scope = await live.resolve_scope(profile, paper_ids=tuple(inputs["project_scope"]["paper_ids"]))
    if snapshot(scope) != snapshot(inputs["project_scope"]):
        raise ValueError("prepared_case_scope_changed")
    index, issues = await live.export_index(prepared.sources)
    expected_storage = read_snapshot(prepared.initial_snapshot).get("index_storage_identity")
    if snapshot({"issues": issues}) != prepared.mapping_issues_snapshot or index != prepared.index or expected_storage != live.index_storage_identity:
        raise ValueError("prepared_actual_index_changed_or_mapping_incomplete")
    benchmark = read_snapshot(AgentCase.model_validate(record).truth_snapshot)["evidence_benchmark"] if kind == "agent" else record["evidence_benchmark"]
    from evals.evidence.contracts import EvidenceBenchmark

    assets = validate_case_assets(EvidenceBenchmark.model_validate(benchmark), uploads_dir=profile_path(profile.uploads_root), paper_dir=profile_path(profile.paper_root))
    if any(row["status"] != "measured" for row in assets):
        raise ValueError(f"prepared_source_artifacts_changed:{snapshot({'assets': assets})}")


async def _execute_case(job: PreparedJob, case: PreparedCase, *, bindings: RuntimeBindings, output: Path) -> dict[str, Any]:
    """参数为固定入口和案例；只按声明入口调用现役执行器，不在错误时退回 fixture。"""
    raw = read_snapshot(case.record_snapshot)
    if job.kind == "planner_fixture":
        from evals.research_planner.live_fixture import build_live_records_async, run_planner_nodes

        return dict((await build_live_records_async([raw], runner=run_planner_nodes))[0])
    live = bindings.live
    if live is None:
        raise RuntimeError("live_dependencies_not_bound")
    if job.kind == "retrieval":
        from app.services.retrieval_service import search_chunks
        from .runtime import search_declared_retrieval
        from functools import partial

        contract = create_case(raw)
        scope = read_snapshot(contract.input_snapshot)["project_scope"]
        scope = live.execution_scope(scope)
        search = search_chunks if job.retrieval_method is None else partial(search_declared_retrieval, method=job.retrieval_method)
        runtime = RetrievalRuntime(store=live.store, scope_snapshot=snapshot(scope), search=search,
                                   top_k=bindings.prepared.profile.retrieval_top_k, index_id=bindings.prepared.index.index_snapshot_id,
                                   backend=bindings.prepared.index.retrieval_backend)
        with patch.object(live.store, "ensure_collection", live.require_existing_index):
            return await run_retrieval_job(contract, runtime)
    from app.nodes.rag_node import create_rag_node

    runtime_agent = AgentRuntime(prepared=bindings.prepared, output_dir=output, live=live, graph_factory=create_rag_node)
    return await run_agent_job(AgentCase.model_validate(raw), runtime_agent)


def _write_case_artifacts(record: dict[str, Any], job: PreparedJob, *, output: Path,
                          policy: MeasurementPolicy | None = None) -> tuple[Path, Path]:
    """参数为真实冻结记录和执行入口；复用原评分器保存 JSON、Markdown 及原始 JSONL。"""
    workflow = job.kind == "agent" and policy is not None
    if job.kind == "agent" and policy is not None:
        from evals.agent_outcome.workflow import grade_workflow_record

        report = {"schema_version": "agent-workflow-report-v1", "measurement_policy": policy.model_dump(mode="json"),
                  "rows": [grade_workflow_record(record, grader_version=policy.workflow_grader_version)]}
        markdown = "# Agent 工作流测评\n\n流程完成、规则符合和结构化结果分别保存在 workflow.json；不评价最终回答质量。\n"
    elif job.kind == "agent":
        report = build_agent_report([record])
        markdown = render_agent_report(report)
    elif job.kind == "retrieval":
        report = build_evidence_report([record], grading_policy="candidate_diagnostic")
        markdown = "\n".join(render_evidence_report(report)) + "\n"
    else:
        from evals.research_planner.metrics import score_records

        original = score_records([record])
        write_new_json(output / "planner-detail.json", original)
        report = _planner_structure_report(record, original)
        markdown = "# Planner 固定样例结果\n\n套件事前固定比较 plan_valid_rate；完整原始诊断保留在 planner-detail.json。\n"
    case, run = record_contracts(record)
    frozen = {**record, "evidence_integrity": {"schema_version": "evidence-integrity-v1",
              "case_sha256": snapshot_hash(snapshot(case.model_dump(mode="json"))),
              "run_sha256": snapshot_hash(snapshot(run.model_dump(mode="json")))}}
    report_path, frozen_path = output / ("workflow.json" if workflow else "report.json"), output / "frozen.jsonl"
    write_new_json(report_path, report)
    with frozen_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(snapshot(frozen) + "\n")
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)
    return report_path, frozen_path


def _planner_structure_report(record: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    """参数为实际记录和原评分；按套件事前固定的结构诊断范围生成独立版本，不补辅助真值。"""
    row = original["records"][0]
    selected = [MetricResult.model_validate(item) for item in row["grading_result"]["measurements"] if item["name"] in PLANNER_SUITE_METRICS]
    grading = build_grading(record, selected, grader="research_planner", grader_version=PLANNER_SUITE_GRADER_VERSION)
    return {"measurement_policy": {"version": PLANNER_SUITE_GRADER_VERSION, "enabled_metrics": sorted(PLANNER_SUITE_METRICS)},
            "records": [{"eval_case": row["eval_case"], "run_record": row["run_record"], "grading_result": grading}]}


def _execution_failure(record: dict[str, Any]) -> str | None:
    """参数为实际记录；只将执行错误判作运行失败，不把低分或诚实未完成当作程序成功。"""
    _case, run = record_contracts(record)
    observed = read_snapshot(run.observation_snapshot)
    if run.missing_fields:
        return "actual_observation_missing:" + ",".join(run.missing_fields)
    if observed.get("failure_reason"):
        return str(observed["failure_reason"])
    if observed.get("status") in {"failed", "cancelled", "timed_out"}:
        return f"actual_episode_status:{observed['status']}"
    return None


async def _post_conditions(
    bindings: RuntimeBindings, before: tuple[IdentityCondition, ...], *, case: PreparedCase, record: dict[str, Any],
) -> tuple[tuple[IdentityCondition, ...], tuple[IdentityCondition, ...], tuple[str, ...]]:
    """参数为真实依赖和运行前身份；保存运行后漂移，验证失败使比较条件明确失效。"""
    try:
        if bindings.live is not None:
            await _verify_live_inputs(bindings.prepared, bindings.live, case=case, kind=bindings.kind)
        after = capture_conditions(bindings)
        errors = (*verify_conditions(before, after), *verify_model_requests(record, before))
    except Exception as exc:
        after = ()
        safe = sanitize_frozen_record({"error": str(exc)}, secrets=export_secrets(bindings.kind))["error"]
        errors = (f"runtime_conditions_not_verified:{type(exc).__name__}:{safe}",)
    if not errors:
        return before, after, errors
    # 1. 【评测套件】【条件漂移】保留前后原始身份，外部环境条件标缺失以阻止提升计算
    conditions = tuple(IdentityCondition(name=item.name, status="missing", reason=";".join(errors))
                       if item.name == "environment" else item for item in before)
    return conditions, after, errors


async def run_worker(prepared: PreparedSuite, job: PreparedJob, *, slot: RunSlot, output: Path) -> ProducedRun:
    """参数为固定任务和槽位；执行一题并封存实际条件，失败也写小型完成记录供父进程读取。"""
    output.mkdir(parents=True, exist_ok=False)
    secrets = export_secrets(job.kind)
    handler = logging.FileHandler(output / "runtime.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    formatters = {item: item.formatter for item in logging.getLogger().handlers}
    for item, formatter in formatters.items():
        item.setFormatter(ExportLogFormatter(formatter, secrets=secrets))
    try:
        result = await _run_cancellable_worker(prepared, job, slot=slot, output=output)
    except asyncio.CancelledError:
        result = ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version,
                             repeat_id=slot.repeat_id, status="cancelled", error="suite_cancel_request_received")
    except Exception as exc:
        logging.getLogger(__name__).exception("【评测套件】【执行失败】case=%s", slot.case_id)
        result = ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version,
                             repeat_id=slot.repeat_id, status="failed", error=f"{type(exc).__name__}:{exc}")
    finally:
        for item, formatter in formatters.items():
            item.setFormatter(formatter)
        logging.getLogger().removeHandler(handler)
        handler.close()
    result = ProducedRun.model_validate(sanitize_frozen_record(result.model_dump(mode="json"), secrets=secrets))
    write_new_json(output / "produced.json", result.model_dump(mode="json"))
    return result


async def _run_cancellable_worker(prepared: PreparedSuite, job: PreparedJob, *, slot: RunSlot, output: Path) -> ProducedRun:
    """参数为固定槽位和目录；监听父进程取消文件，等待实际任务清理后传播取消结果。"""
    task = asyncio.create_task(_run_worker_case(prepared, job, slot=slot, output=output))
    try:
        while not task.done():
            if (output / "cancel-request.json").exists():
                task.cancel()
                break
            await asyncio.wait({task}, timeout=WORKER_STOP_POLL_SECONDS)
        return await task
    finally:
        if not task.done():
            task.cancel()
            await task


async def _run_worker_case(prepared: PreparedSuite, job: PreparedJob, *, slot: RunSlot, output: Path) -> ProducedRun:
    """参数为固定案例槽位；先核对条件，再执行评分，最后将身份绑定到真实文件。"""
    started = time.perf_counter()
    case = next(case for case in job.cases if (case.case_id, case.case_version) == (slot.case_id, slot.case_version))
    live = LivePreparationRuntime() if job.kind != "planner_fixture" else None
    bindings = RuntimeBindings(prepared=prepared, kind=job.kind, live=live, fixture_snapshot=case.record_snapshot,
                               retrieval_method=job.retrieval_method)
    if live is not None:
        await _verify_live_inputs(prepared, live, case=case, kind=job.kind)
    before = capture_conditions(bindings)
    write_new_json(output / "conditions-before.json", [item.model_dump(mode="json") for item in before])
    try:
        if live is not None:
            inputs = read_snapshot(AgentCase.model_validate(read_snapshot(case.record_snapshot)).input_snapshot) if job.kind == "agent" else read_snapshot(create_case(read_snapshot(case.record_snapshot)).input_snapshot)
            await live.acquire_scope(inputs["project_scope"])
            write_new_json(output / "active-leases.json", {"scopes": list(live.active_scopes.values())})
        episode_started = time.perf_counter()
        record = await _execute_case(job, case, bindings=bindings, output=output)
        record = sanitize_frozen_record(record, secrets=export_secrets(job.kind))
        episode_seconds = time.perf_counter() - episode_started
    finally:
        if live is not None:
            await live.release_scopes()
    write_new_json(output / "product-record.json", record)
    write_new_json(output / "resources.json", {"episode_seconds": episode_seconds,
                                               "preparation_and_episode_seconds": time.perf_counter() - started})
    report, frozen = _write_case_artifacts(record, job, output=output, policy=prepared.profile.measurement_policy)
    grading_status, grading_error = ("not_started", None)
    if prepared.profile.measurement_policy is None:
        grading_status, grading_error = await _score_legacy_product_record(record, job, output=output)
    conditions, after, errors = await _post_conditions(bindings, before, case=case, record=record)
    write_new_json(output / "conditions-after.json", {"conditions": [item.model_dump(mode="json") for item in after], "errors": errors})
    identity_path = output / "identity.json"
    error = ";".join(errors) or _execution_failure(record)
    produced = ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version, repeat_id=slot.repeat_id,
                           run_id=record_contracts(record)[1].run_id, report_path=str(report.resolve()), frozen_path=str(frozen.resolve()),
                           identity_path=str(identity_path.resolve()), status="failed" if error else "completed", error=error,
                           grading_status=grading_status, grading_error=grading_error)
    write_recorded_identity(produced, RunConditions(run_id=str(produced.run_id), conditions=conditions), path=identity_path)
    return produced


async def _score_legacy_product_record(
    record: dict[str, Any], job: PreparedJob, *, output: Path,
) -> tuple[str, str | None]:
    """参数为旧版产品记录；保持无测量策略历史运行的原评分时序和产物合同。"""
    if job.kind not in {"retrieval", "agent"}:
        return "not_started", None
    from evals.ragas.frozen_bridge import build_failed_score, score_frozen_record

    started = time.perf_counter()
    kind = "agent" if job.kind == "agent" else "retrieval"
    try:
        ragas = await score_frozen_record(record, kind=kind, policy=None)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        logging.getLogger(__name__).error("【评测套件】【独立评分】旧版评分失败 error_type=%s", type(exc).__name__)
        ragas = build_failed_score(record, kind=kind, policy=None, error=error)
    ragas = sanitize_frozen_record(ragas, secrets=export_secrets(job.kind))
    write_new_json(output / "ragas.json", ragas)
    write_new_json(output / "grading-resources.json", {"grading_seconds": time.perf_counter() - started})
    grading_error = ragas.get("grading_error")
    grading_status = "failed" if ragas.get("grading_status") == "failed" else "completed"
    write_new_json(output / "grading-status.json", {"status": grading_status, "error": grading_error})
    return grading_status, grading_error


def _worker_main() -> int:
    """无参数；解析内部 worker 的完整槽位，不对外提供随意替换业务图的隐藏参数。"""
    parser = argparse.ArgumentParser(description="执行一个事前声明的评测槽位")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepared = load_prepared(args.prepared)
    job = next(job for job in prepared.jobs if job.job_id == args.job_id)
    slot = next(slot for slot in job.slots if (slot.case_id, slot.repeat_id) == (args.case_id, args.repeat_id))
    result = asyncio.run(run_worker(prepared, job, slot=slot, output=args.output_dir))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(_worker_main())
