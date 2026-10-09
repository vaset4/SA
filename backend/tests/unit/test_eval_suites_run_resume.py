"""跑批进度落盘、中断留痕与断点续跑回归。作者：xxx。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.contracts import create_case, snapshot, snapshot_hash
from evals.suites.contracts import RunSlot
from evals.suites.execution_contracts import (
    ExecutionRequest, PreparedCase, PreparedJob, PreparedSuite, ProducedRun,
)
from evals.suites.execute import _slot_key, execute_suite
from evals.suites.prepare import sha256_file, write_new_json
from test_eval_suites_execution import portable_planner_record, prepared_fixture


def two_slot_prepared(root: Path) -> PreparedSuite:
    """参数为新临时目录；返回含两个真实 planner 槽位的固定输入，用于中断与续跑验证。"""
    base = prepared_fixture(root, planner_only=True)
    raw = portable_planner_record()
    case = create_case(raw)
    slots = tuple(RunSlot(case_id=case.case_id, case_version=case.case_version, repeat_id=str(index), run_id=None,
                          required_checks=("truth_leakage", "required_trace")) for index in (1, 2))
    job = PreparedJob(job_id="planner", kind="planner_fixture",
                      cases=(PreparedCase(case_id=case.case_id, case_version=case.case_version,
                                          record_snapshot=snapshot(raw)),), slots=slots)
    body = base.model_dump(mode="json", exclude={"prepared_id"})
    body["jobs"] = [job.model_dump(mode="json")]
    return PreparedSuite.model_validate({**body, "prepared_id": snapshot_hash(snapshot(body))})


def read_ledger(output: Path) -> dict:
    """参数为运行目录；读取执行账本，供各用例断言真实落盘进度。"""
    return json.loads((output / "execution.json").read_text(encoding="utf-8"))


def artifact_digests(ledger: dict) -> list[str]:
    """参数为执行账本；返回全部槽位实际产物指纹，用于证明续跑没有改动上次结果。"""
    return sorted(sha256_file(Path(entry[key]))
                  for entry in ledger["runs"] for key in ("report_path", "frozen_path", "identity_path")
                  if entry[key] is not None)


def test_progress_is_persisted_interrupt_is_readable_and_resume_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录与替身；首槽进度真实落盘、人为中断原因可读，续跑只补未完成槽且可反复进入。"""
    from evals.suites import execute

    prepared = two_slot_prepared(tmp_path)
    output = tmp_path / "run"
    real_execute_job = execute.execute_job
    observed: list[dict] = []
    launched: list[str] = []

    def interrupting_execute_job(job, slot, **kwargs):
        """参数为真实槽位调度实参；第二槽开始前读取账本，再强制人为中断复现整轮报废现场。"""
        if slot.repeat_id == "2":
            observed.append(read_ledger(output))
            raise KeyboardInterrupt
        return real_execute_job(job, slot, **kwargs)

    monkeypatch.setattr(execute, "execute_job", interrupting_execute_job)
    with pytest.raises(KeyboardInterrupt):
        execute_suite(prepared, ExecutionRequest(output_dir=output))
    assert observed and observed[0]["suite_status"] == "in_progress"
    assert [entry["status"] for entry in observed[0]["runs"]] == ["completed"]
    interrupted = read_ledger(output)
    assert interrupted["suite_status"] == "in_progress" and interrupted["abort_reason"] == "user_interrupt"
    assert [entry["status"] for entry in interrupted["runs"]] == ["completed"]

    def resuming_execute_job(job, slot, **kwargs):
        """参数为真实槽位调度实参；续跑只允许启动未完成槽位，已完成槽位不得再起子进程。"""
        launched.append(slot.repeat_id)
        return real_execute_job(job, slot, **kwargs)

    monkeypatch.setattr(execute, "execute_job", resuming_execute_job)
    execute_suite(prepared, ExecutionRequest(output_dir=output, resume=True))
    assert launched == ["2"]
    resumed = read_ledger(output)
    assert resumed["suite_status"] == "finished"
    assert [entry["status"] for entry in resumed["runs"]] == ["completed", "completed"]
    assert resumed["runs"][0]["run_id"] == interrupted["runs"][0]["run_id"]
    assert [entry for entry in resumed["resumed_from_previous_run"] if entry.endswith(":1:completed")]
    digests = artifact_digests(resumed)

    def refuse_execution(*args, **kwargs):
        """参数为被拦截的真实调度实参；已完成整轮再次续跑不应启动任何子进程。"""
        raise AssertionError("resume must not re-execute claimed slots")

    monkeypatch.setattr(execute, "execute_job", refuse_execution)
    execute_suite(prepared, ExecutionRequest(output_dir=output, resume=True))
    assert artifact_digests(read_ledger(output)) == digests


def test_external_termination_differs_from_user_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录与替身；非人为退出保留 in_progress 且无中断原因，可与人为取消区分。"""
    from evals.suites import execute

    prepared = two_slot_prepared(tmp_path)
    output = tmp_path / "run"

    def killed_execute_job(*args, **kwargs):
        """参数为被拦截的真实调度实参；模拟进程被外部终止，不产生任何槽位进度。"""
        raise RuntimeError("external supervisor kill")

    monkeypatch.setattr(execute, "execute_job", killed_execute_job)
    with pytest.raises(RuntimeError):
        execute_suite(prepared, ExecutionRequest(output_dir=output))
    killed = read_ledger(output)
    assert killed["suite_status"] == "in_progress" and killed["abort_reason"] is None
    assert killed["runs"] == []


def test_leftover_slot_is_archived_and_timeout_stays_distinguishable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录与替身；残留槽位让路归档现场，超时槽位如实记账且不冒充完成。"""
    from evals.suites import execute

    prepared = two_slot_prepared(tmp_path)
    output = tmp_path / "run"
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()
    write_new_json(output / "prepared.json", prepared.model_dump(mode="json"))
    job = prepared.jobs[0]
    leftover = output / "runs" / _slot_key(job, job.slots[1])
    leftover.mkdir()
    (leftover / "partial-events.jsonl").write_text('{"partial":true}\n', encoding="utf-8")
    real_execute_job = execute.execute_job

    def timed_out_first_slot(job, slot, **kwargs):
        """参数为真实槽位调度实参；首槽返回超时结果，第二槽走真实子进程。"""
        if slot.repeat_id == "1":
            return ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version,
                               repeat_id=slot.repeat_id, status="timed_out", error="worker_timed_out")
        return real_execute_job(job, slot, **kwargs)

    monkeypatch.setattr(execute, "execute_job", timed_out_first_slot)
    execute_suite(prepared, ExecutionRequest(output_dir=output, resume=True))
    archived = leftover.with_name(leftover.name + ".attempt-1")
    assert (archived / "partial-events.jsonl").read_text(encoding="utf-8") == '{"partial":true}\n'
    assert (leftover / "produced.json").is_file()
    ledger = read_ledger(output)
    assert [entry["status"] for entry in ledger["runs"]] == ["timed_out", "completed"]


def test_cancelled_slot_leaves_remaining_slots_explicitly_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录与替身；真实取消后剩余槽位如实记为未跑，不用成功产物顶替。"""
    from evals.suites import execute

    prepared = two_slot_prepared(tmp_path)
    output = tmp_path / "run"
    job = prepared.jobs[0]

    def cancelled_execute_job(_job, slot, **_kwargs):
        """参数为真实槽位调度实参；首槽返回真实取消状态，后续槽位不应再被调度。"""
        assert slot.repeat_id == "1"
        return ProducedRun(job_id=job.job_id, case_id=slot.case_id, case_version=slot.case_version,
                           repeat_id=slot.repeat_id, status="cancelled", error="worker_cancelled")

    monkeypatch.setattr(execute, "execute_job", cancelled_execute_job)
    with pytest.raises(RuntimeError, match="没有可形成清单的实际产物"):
        execute_suite(prepared, ExecutionRequest(output_dir=output))
    ledger = read_ledger(output)
    assert [entry["status"] for entry in ledger["runs"]] == ["cancelled", "not_run"]
    assert ledger["runs"][1]["error"] == "suite_cancelled_before_slot"
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["reason"] == "no_complete_run_artifacts"
