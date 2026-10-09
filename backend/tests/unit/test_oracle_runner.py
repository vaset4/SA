"""Oracle 配对执行、隔离和诊断报告回归。作者：xxx。"""

from __future__ import annotations

from typing import Any

from evals.oracle.contracts import make_pair_spec
from evals.oracle.report import build_oracle_report
from evals.oracle.runner import run_oracle_pair


class _ReviewedBenchmark:
    """由测试 monkeypatch 的最小真值占位对象。"""

    review_status = "reviewed"


def _spec() -> Any:
    """无参数；返回目标证据和控制条件均已冻结的配对合同。"""
    return make_pair_spec(
        pair_id="pair-001", case_id="case-001", case_version="v1",
        input_payload={"question": "查找原文", "paper_ids": ["paper-1"]},
        truth_payload={"reference_answer": "secret answer"},
        control_payload={"target_evidence_ids": ["span-1"], "prompt_hash": "prompt-v1", "temperature": 0},
        index_snapshot_id="index-001",
    )


def test_pair_keeps_actual_and_oracle_namespaces_separate(monkeypatch) -> None:
    """无参数；配对执行保持输入一致、证据来源分离和诊断-only 报告。"""
    import evals.oracle.runner as runner

    monkeypatch.setattr(runner, "build_oracle_evidence", lambda benchmark, index_snapshot_id: ({"evidence_id": "span-1", "excerpt": "source"},))
    spec = _spec()
    calls: list[tuple[str, str, int]] = []

    def actual_executor(payload, *, retrieval_mode, oracle_evidence, cache_namespace):
        assert payload["question"] == "查找原文"
        assert retrieval_mode == "actual"
        assert oracle_evidence == ()
        calls.append((retrieval_mode, cache_namespace, len(oracle_evidence)))
        return {"observation": {"stage_observations": [{"stage": "query_raw", "items": []}], "response": ""}}

    def oracle_executor(payload, *, retrieval_mode, oracle_evidence, cache_namespace):
        assert payload == {"question": "查找原文", "paper_ids": ["paper-1"]}
        assert retrieval_mode == "oracle"
        assert oracle_evidence[0]["evidence_id"] == "span-1"
        calls.append((retrieval_mode, cache_namespace, len(oracle_evidence)))
        return {"observation": {"stage_observations": [{"stage": "query_raw", "items": [{"evidence_id": "span-1"}]}], "response": "oracle"}}

    pair = run_oracle_pair(spec, benchmark=_ReviewedBenchmark(), actual_executor=actual_executor, oracle_executor=oracle_executor)
    assert calls == [("actual", "pair-001:actual", 0), ("oracle", "pair-001:oracle", 1)]
    report = build_oracle_report([pair])
    assert report["diagnostic_only"] is True
    assert report["quality_gate_eligible"] is False
    assert report["attribution_label_counts"] == {"retrieval_gap": 1}
