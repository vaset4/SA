"""Oracle 阶段症状归因回归。作者：xxx。"""

from __future__ import annotations

import json
from typing import Any

from evals.oracle.attribution import attribute_pair
from evals.oracle.contracts import OracleSideRun, make_pair_spec


def _pair() -> Any:
    """无参数；返回声明目标 Evidence 的配对合同。"""
    return make_pair_spec(
        pair_id="pair-1", case_id="case-1", case_version="v1",
        input_payload={"question": "Q"}, truth_payload={},
        control_payload={"target_evidence_ids": ["span-1"]}, index_snapshot_id="idx-1",
    )


def _side(side: str, stages: list[dict], response: str = "answer") -> OracleSideRun:
    """参数为一侧名称、阶段观察和答案；返回冻结侧运行。"""
    spec = _pair()
    return OracleSideRun(
        side=side, run_id=f"run-{side}", run_mode="oracle" if side == "oracle" else "live",
        retrieval_mode="oracle" if side == "oracle" else "actual",
        input_snapshot=spec.input_snapshot, observation_snapshot=json.dumps({"stage_observations": stages, "response": response}),
        cache_namespace=spec.oracle_cache_namespace if side == "oracle" else spec.actual_cache_namespace,
        status="completed",
    )


def test_retrieval_gap_is_reported_when_actual_misses_oracle_target() -> None:
    """无参数；Actual 未召回而 Oracle 召回目标时报告检索缺口。"""
    spec = _pair()
    actual = _side("actual", [{"stage": "query_raw", "items": []}])
    oracle = _side("oracle", [{"stage": "query_raw", "items": [{"evidence_id": "span-1"}]}])
    result = attribute_pair(actual, oracle, spec=spec)
    assert result.labels == ("retrieval_gap",)


def test_selection_gap_is_reported_when_target_does_not_reach_model_context() -> None:
    """无参数；Actual 已召回但没有进入模型上下文时报告选择缺口。"""
    spec = _pair()
    actual = _side("actual", [{"stage": "query_raw", "items": [{"evidence_id": "span-1"}]}])
    oracle = _side("oracle", [{"stage": "query_raw", "items": [{"evidence_id": "span-1"}]}, {"stage": "final_input", "items": [{"evidence_id": "span-1"}]}])
    result = attribute_pair(actual, oracle, spec=spec)
    assert result.labels == ("selection_gap",)


def test_missing_stage_observation_is_unknown_or_missing() -> None:
    """无参数；缺少阶段观察不能被最终答案反推成具体根因。"""
    spec = _pair()
    actual = _side("actual", [])
    oracle = _side("oracle", [{"stage": "final_input", "items": [{"evidence_id": "span-1"}]}])
    result = attribute_pair(actual, oracle, spec=spec)
    assert result.status in {"missing", "measured"}
    assert result.labels in {(), ("unknown",)}
