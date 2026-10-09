"""首次分析失败的用户可见性与评分版本回归。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from app.models.research_planner_contracts import PlannerFailure
from app.services.internal_research.research_planner_conversion import build_planner_failure_response
from evals.agent_outcome.contracts import load_agent_case
from evals.agent_outcome.runner import run_episode_sync
from evals.agent_outcome.scoring import grade_agent_record
from evals.contracts import MetricResult, build_grading
from test_agent_outcome_runner import _environment, _record


class _EarlyFailureGraph:
    """透传现役失败转换器生成的载荷，不调用外部服务。"""

    def __init__(self, values: dict[str, Any]) -> None:
        """参数为本次实际形状的失败载荷；保存独立副本。"""
        self.values = deepcopy(values)

    async def astream(self, inputs: dict[str, Any], *, stream_mode: list[str]):
        """参数为图输入和流模式；返回没有完成turn_outcome的首次分析失败。"""
        assert stream_mode == ["updates", "values"]
        yield "updates", {"planner_analyze": self.values}
        yield "values", self.values


def _failed_record(code: str, *, patch: dict[str, Any] | None = None) -> dict[str, Any]:
    """参数为已知失败码和实际观察变体；返回经过runner的冻结记录。"""
    failure = PlannerFailure(code=code, message="403 Forbidden: 模型服务拒绝本次请求", local_evidence_execution="not_run")
    values = {**build_planner_failure_response(failure), "planner_loop_decision": "terminate_fail",
              "evidence_records": [], "citations": [], **(patch or {})}
    return run_episode_sync(load_agent_case(_record()), _environment(lambda: _EarlyFailureGraph(values)))


@pytest.mark.parametrize("code", ["planner_depth_assessment_failed", "planner_context_load_failed",
                                 "memory_read_failed", "planner_structured_output_failed"])
def test_early_failure_keeps_visible_error_and_new_grader_identity(code: str) -> None:
    """参数为首次分析失败码；明确错误与无伪证据可判诚实失败，旧评分不可被原地改写。"""
    record = _failed_record(code)
    old = MetricResult(name="failure_handling_correctness", status="measured", value=0.0,
                       numerator=0.0, denominator=1.0, reason="legacy_terminal_not_recognized")
    record["grading_result"] = build_grading(record, [old], grader="agent_outcome", grader_version="agent-outcome-v2")
    original = deepcopy(record)
    graded = grade_agent_record(record)
    metrics = {item["name"]: item for item in graded["measurements"]}
    assert metrics["failure_handling_correctness"]["value"] == 1.0
    assert metrics["task_success"]["value"] == 0.0
    assert graded["grader_version"] == "agent-outcome-v3"
    assert record == original
    assert record["grading_result"]["grader_version"] == "agent-outcome-v2"


@pytest.mark.parametrize("patch", [
    {"answer": "本题已经得到正确结果"},
    {"error_summary": ""},
    {"citations": [{"evidence_id": "invented"}]},
    {"turn_outcome": {"status": "completed", "termination_reason": "done"}},
    {"planner_loop_decision": "continue"},
])
def test_failure_code_alone_cannot_receive_honest_failure_credit(patch: dict[str, Any]) -> None:
    """参数为隐瞒错误、伪造引用或矛盾终态；失败码本身不得自动取得正确处理分数。"""
    graded = grade_agent_record(_failed_record("planner_depth_assessment_failed", patch=patch))
    metric = next(item for item in graded["measurements"] if item["name"] == "failure_handling_correctness")
    assert metric["value"] == 0.0


def test_unknown_failed_code_does_not_expand_terminal_contract() -> None:
    """无参数；未核对的任意failed后缀不进入已知首次分析失败合同。"""
    graded = grade_agent_record(_failed_record("unverified_custom_failed"))
    metric = next(item for item in graded["measurements"] if item["name"] == "failure_handling_correctness")
    assert metric["value"] == 0.0


def test_suite_rejects_comparing_v2_and_v3_grading(tmp_path: Path) -> None:
    """参数为临时目录；同一次冻结运行换评分语义版本也不能被套件当作产品收益比较。"""
    from evals.agent_outcome.report import build_agent_report
    from evals.suites.adapters import load_suite_sources
    from evals.suites.aggregate import aggregate_suite
    from evals.suites.compare import compare_suites
    from evals.suites.contracts import ComparisonRequest
    from test_eval_suites_contracts_adapters import make_agent_record, make_slots, write_manifest, write_source

    records = (make_agent_record(),)
    summaries = []
    for version in ("agent-outcome-v2", "agent-outcome-v3"):
        report = build_agent_report(records)
        report["grader_version"] = version
        report["rows"][0]["grading_result"]["grader_version"] = version
        root = tmp_path / version
        source = write_source(root, "agent", records, report=report, with_identity=True)
        path = write_manifest(root, (source,), make_slots(records))
        summaries.append(aggregate_suite(load_suite_sources(path)))
    result = compare_suites(summaries[1], summaries[0], ComparisonRequest(changed_factors=()))
    assert result.status == "incomparable"
    assert any("grader_or_projection_changed" in reason for reason in result.reasons)
