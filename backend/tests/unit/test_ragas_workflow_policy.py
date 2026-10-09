"""工作流口径仅调度检索及适用性边界回归。作者：xxx。"""

from __future__ import annotations

from typing import Any

import pytest

from evals.contracts import read_snapshot, snapshot
from evals.ragas.frozen_bridge import score_frozen_record
from evals.suites.policy import workflow_policy
from test_ragas_frozen_bridge import _event, _metrics, _record, _Scorer


@pytest.mark.asyncio
async def test_policy_never_calls_answer_judge_and_keeps_partial_retrieval_failure() -> None:
    """Agent有真实回答时仍只调用CP/CR；CP失败不能清除CR或补评分。"""
    called = []

    async def scorer(metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为指标及样本；模拟单个检索judge API失败，返回另一项真实测试值。"""
        called.append(metric)
        assert "response" not in sample
        if metric == "context_precision":
            raise ConnectionError("judge transport unavailable")
        return {"value": 0.6}

    result = await score_frozen_record(_record(agent=True, generation=True), kind="agent",
                                       scorer=scorer, policy=workflow_policy())
    assert called == ["context_precision", "context_recall"]
    assert set(_metrics(result)) == set(called)
    assert _metrics(result)["context_precision"]["status"] == "missing"
    assert _metrics(result)["context_recall"]["value"] == 0.6
    assert result["grader_version"] == "ragas-retrieval-v2"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty", "missing", "negative"])
async def test_empty_missing_and_negative_retrieval_have_distinct_results(mode: str) -> None:
    """真实空召回是零，采集缺失没有分数，负例没有正证据召回分母。"""
    record, scorer = _record(), _Scorer()
    observed = read_snapshot(record["run_record"]["observation_snapshot"])
    observed["evidence_observations"][0]["payload"]["hits"] = []
    if mode == "missing":
        observed.pop("evidence_observations")
    if mode == "negative":
        truth = read_snapshot(record["eval_case"]["truth_snapshot"])
        record["eval_case"]["truth_snapshot"] = snapshot({**truth, "answerability": "insufficient_evidence"})
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="retrieval", scorer=scorer, policy=workflow_policy())
    assert not scorer.calls
    expected = {"empty": "measured", "missing": "missing", "negative": "not_measured"}[mode]
    assert _metrics(result)["context_recall"]["status"] == expected
    assert _metrics(result)["context_recall"]["value"] == (0 if mode == "empty" else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("source,failed_capture,status", [
    ("local_section_text", False, "not_measured"),
    ("local_direct_text", False, "missing"),
    ("local_section_text", True, "missing"),
])
async def test_only_proven_section_read_is_not_applicable(source: str, failed_capture: bool, status: str) -> None:
    """仅完整且合法章节直读不做检索；应检索路径缺观察和采集失败不套用不适用。"""
    record, scorer = _record(agent=True), _Scorer()
    observed = read_snapshot(record["run_record"]["observation_snapshot"])
    turn = observed["turns"][0]
    capture = read_snapshot(turn["evidence_observation_snapshot"])
    capture["evidence_observations"] = [_event("tool_execution_result", {
        "source_type": source, "status": "success", "tool_call_id": "read", "tool_attempt": 1,
        "parameters": {"required_sections": ["Methods"], "target_paper_ids": ["paper"]},
    })]
    if failed_capture:
        capture["evidence_capture_errors"] = [{"stage": "query_raw", "error": "capture unavailable"}]
    turn["evidence_observation_snapshot"] = snapshot(capture)
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="agent", scorer=scorer, policy=workflow_policy())
    assert not scorer.calls
    assert _metrics(result)["context_recall"]["status"] == status


@pytest.mark.asyncio
async def test_section_only_turn_does_not_hide_other_turn_retrieval() -> None:
    """首轮章节直读不进入召回分母，后轮真实查询仍独立评分。"""
    record, scorer = _record(agent=True), _Scorer()
    observed = read_snapshot(record["run_record"]["observation_snapshot"])
    first = observed["turns"][0]
    second = {**first, "turn": 2}
    captured = read_snapshot(first["evidence_observation_snapshot"])
    captured["evidence_observations"] = [_event("tool_execution_result", {
        "source_type": "local_section_text", "status": "success", "tool_call_id": "section", "tool_attempt": 1})]
    first["evidence_observation_snapshot"] = snapshot(captured)
    observed["turns"].append(second)
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="agent", scorer=scorer, policy=workflow_policy())
    assert len(scorer.calls) == 2
    assert _metrics(result)["context_recall"]["value"] == 0.5
    assert any(row["turn"] == 1 and row["result"]["status"] == "not_measured" for row in result["samples"])


@pytest.mark.asyncio
async def test_section_read_retry_recovery_remains_without_ranked_retrieval() -> None:
    """真实章节直读首次失败后重试成功仍不产生排名检索分母；失败尝试保留原观察。"""
    record, scorer = _record(agent=True), _Scorer()
    observed = read_snapshot(record["run_record"]["observation_snapshot"])
    capture = read_snapshot(observed["turns"][0]["evidence_observation_snapshot"])
    base = {"source_type": "local_section_text", "tool_call_id": "section"}
    capture["evidence_observations"] = [
        _event("tool_execution_result", {**base, "tool_attempt": 1, "status": "failed", "error_type": "ReadTimeout"}),
        {**_event("tool_execution_result", {**base, "tool_attempt": 2, "status": "success"}), "event_id": "retry"},
    ]
    observed["turns"][0]["evidence_observation_snapshot"] = snapshot(capture)
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="agent", scorer=scorer, policy=workflow_policy())
    assert _metrics(result)["context_recall"]["status"] == "not_measured"
    assert not scorer.calls
    assert capture["evidence_observations"][0]["payload"]["status"] == "failed"
