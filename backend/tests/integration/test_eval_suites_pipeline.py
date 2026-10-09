"""从实际冻结文件到离线 CLI 报告的套件集成验证。作者：xxx。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from evals.suites.contracts import SuiteReport
from evals.suites.report import render_suite_report
from test_eval_suites_contracts_adapters import (
    make_agent_record, make_evidence_record, make_planner_record, make_slots,
    measured, write_manifest, write_source,
)

_CLI_TIMEOUT_SECONDS = 15
_ROOT = Path(__file__).resolve().parents[3]
_ENTRY = _ROOT / "backend" / "evals" / "suites" / "run_eval.py"


def _cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """参数为显式 CLI 参数；调用真实入口并限制单次子进程时间。"""
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "SSLKEYLOGFILE": ""}
    return subprocess.run([sys.executable, "-X", "utf8", str(_ENTRY), *arguments], cwd=_ROOT, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=_CLI_TIMEOUT_SECONDS)


def _planner_manifest(root: Path, *, value: float, model: str | None = None) -> Path:
    """参数为目录、fixture 成绩及已记录模型；生成比较入口使用的实际文件。"""
    records = (make_planner_record(metrics=(measured("plan_valid_rate", value),)),)
    source = write_source(root, "planner", records, with_identity=model is not None,
                          identity_overrides={"model": model} if model is not None else None)
    return write_manifest(root, (source,), make_slots(records))


def test_cli_writes_auditable_report_from_three_real_shapes(tmp_path: Path) -> None:
    """参数为目录；读取三类真实评分器产物，保留原文件并写出可重放报告。"""
    planner, evidence, agent = make_planner_record(), make_evidence_record(), make_agent_record()
    sources = (write_source(tmp_path, "planner", (planner,)), write_source(tmp_path, "evidence", (evidence,)),
               write_source(tmp_path, "agent", (agent,)))
    originals = {(tmp_path / source.report_path): (tmp_path / source.report_path).read_bytes() for source in sources}
    manifest = write_manifest(tmp_path, sources, make_slots((planner, evidence, agent)))
    output = tmp_path / "report-output"
    result = _cli(["report", "--manifest", str(manifest), "--output-dir", str(output)])
    assert result.returncode == 0, result.stderr
    report = SuiteReport.model_validate_json((output / "report.json").read_text(encoding="utf-8"))
    assert len(report.summary.inputs.runs) == 3
    assert {score.kind for score in report.summary.inputs.scores} == {"planner", "evidence", "agent"}
    assert all(path.read_bytes() == content for path, content in originals.items())
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert markdown == render_suite_report(report)
    assert "跨论文取证覆盖" in markdown and "历史比较条件" in markdown
    assert "RAGAS Context Precision 使用有参考答案的平均精度变体" in markdown
    assert "candidate 表示参考标注未经人工审核" in markdown
    assert (output / "results.csv").is_file()


def test_cli_compare_records_declared_change_and_delta(tmp_path: Path) -> None:
    """参数为目录；声明仅变更模型时，真实 CLI 输出配对成绩与原始运行身份。"""
    baseline = _planner_manifest(tmp_path / "before", value=0, model="model-a")
    current = _planner_manifest(tmp_path / "after", value=1, model="model-b")
    output = tmp_path / "comparison"
    result = _cli(["compare", "--manifest", str(current), "--baseline", str(baseline),
                   "--changed-factors", "model", "--output-dir", str(output)])
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    comparison = report["comparison"]
    assert comparison["status"] == "comparable"
    assert comparison["request"]["changed_factors"] == ["model"]
    assert comparison["deltas"][0]["delta"] == 1


def test_cli_missing_identity_keeps_incomparable_report(tmp_path: Path) -> None:
    """参数为目录；无历史模型条件时写出原因报告，并以非零状态结束比较。"""
    baseline = _planner_manifest(tmp_path / "before", value=0)
    current = _planner_manifest(tmp_path / "after", value=1)
    output = tmp_path / "incomparable"
    result = _cli(["compare", "--manifest", str(current), "--baseline", str(baseline),
                   "--changed-factors", "model", "--output-dir", str(output)])
    assert result.returncode == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["deltas"] == []
    assert any("identity_missing:model" in reason for reason in report["comparison"]["reasons"])
    assert "identity_missing:model" in (output / "report.md").read_text(encoding="utf-8")


def test_cli_rejects_output_overwrite(tmp_path: Path) -> None:
    """参数为目录；重复输出必须显式失败，原报告字节保持不变。"""
    manifest = _planner_manifest(tmp_path, value=0)
    output = tmp_path / "immutable-report"
    arguments = ["report", "--manifest", str(manifest), "--output-dir", str(output)]
    assert _cli(arguments).returncode == 0
    original = (output / "report.json").read_bytes()
    result = _cli(arguments)
    assert result.returncode == 1
    assert json.loads(result.stderr)["status"] == "invalid_input"
    assert (output / "report.json").read_bytes() == original


def test_cli_missing_source_is_an_explicit_input_error(tmp_path: Path) -> None:
    """参数为目录；已声明的原报告丢失时不能生成模板成功报告。"""
    manifest = _planner_manifest(tmp_path, value=1)
    (tmp_path / "planner-report.json").unlink()
    output = tmp_path / "not-created"
    result = _cli(["report", "--manifest", str(manifest), "--output-dir", str(output)])
    assert result.returncode == 1
    assert json.loads(result.stderr)["status"] == "invalid_input"
    assert not output.exists()


def test_cli_validate_does_not_write_report(tmp_path: Path) -> None:
    """参数为目录；只读校验入口验证现有产物，不启动执行或写报告目录。"""
    manifest = _planner_manifest(tmp_path, value=1)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = _cli(["validate", "--manifest", str(manifest)])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "completed"
    assert json.loads(result.stdout)["output_dir"] is None
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
