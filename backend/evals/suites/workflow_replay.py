"""【流程评测】【离线复评】新增版本化报告，不改原冻结产物或执行身份。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from evals.agent_outcome.workflow import WORKFLOW_GRADER_VERSIONS, grade_workflow_record
from evals.contracts import snapshot, snapshot_hash

from .policy import workflow_policy


def workflow_code_identity() -> dict[str, Any]:
    """无参数；冻结本机评测源码身份，不初始化模型或补造原产品运行条件。"""
    root = Path(__file__).resolve().parents[1]
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*.py"))}
    return {"scope": "backend/evals/**/*.py", "files": files, "sha256": snapshot_hash(snapshot(files))}


def regrade_workflow(frozen: Path, *, output_dir: Path, grader_version: str) -> Path:
    """参数为原冻结JSONL、新目录及明确评分版本；返回新报告路径，全程不执行产品或judge。"""
    if grader_version not in WORKFLOW_GRADER_VERSIONS:
        raise ValueError(f"unsupported workflow grader version: {grader_version}")
    frozen, output_dir = frozen.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite workflow regrading directory: {output_dir}")
    content = frozen.read_bytes()
    records = [json.loads(line) for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("workflow regrading requires frozen JSON objects")
    before = workflow_code_identity()
    rows = [grade_workflow_record(record, grader_version=grader_version) for record in records]
    after = workflow_code_identity()
    source_hash = hashlib.sha256(content).hexdigest()
    if before != after or hashlib.sha256(frozen.read_bytes()).hexdigest() != source_hash:
        raise ValueError("workflow regrading input_or_code_changed_during_attempt")
    policy_version: Literal["workflow-retrieval-v1", "workflow-retrieval-v2"] = (
        "workflow-retrieval-v1" if grader_version == "agent-workflow-v1" else "workflow-retrieval-v2")
    provenance = {"schema_version": "workflow-regrading-v1", "source_frozen_path": str(frozen),
                  "source_frozen_sha256": source_hash, "grader_version": grader_version,
                  "code_identity": before, "execution_replayed": False, "provider_called": False}
    report = {"schema_version": "agent-workflow-report-v1", "measurement_policy": workflow_policy(version=policy_version).model_dump(mode="json"),
              "regrading": provenance, "rows": rows}
    # 1. 【流程评测】【新产物】校验后独占创建新目录；报告仅引用原输入，不复制或改写旧执行身份
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "workflow.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    identity = {**provenance, "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output_dir / "provenance.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(_render_regrading(report), encoding="utf-8")
    return path


def _render_regrading(report: dict[str, Any]) -> str:
    """参数为本次复评报告；返回明确区分完成、披露及假成功方向的人工可读表格。"""
    lines = ["# 工作流离线复评", "", f"评分版本：`{report['regrading']['grader_version']}`。",
             f"原快照SHA256：`{report['regrading']['source_frozen_sha256']}`。", "",
             "原快照和旧分数保持不变；本次未运行产品或调用模型。",
             "false_success：0为未观察到不实完成，1为明确完成矛盾，missing为证据不足；不是错误处理分数的取反。",
             "仅核对正式状态、报告及可验证披露，不评价自由正文语义或回答质量。", "",
             "| 案例 | 运行 | 指标 | 状态 | 值 | 原因 |", "|---|---|---|---|---|---|"]
    names = {"workflow_completion", "workflow_rule_compliance", "error_handling_correctness", "false_success"}
    for row in report["rows"]:
        for metric in row["measurements"]:
            if metric["name"] in names:
                values = (row["case_id"], row["run_id"], metric["name"], metric["status"], metric.get("value"), metric["reason"])
                lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    return "\n".join(lines) + "\n"
