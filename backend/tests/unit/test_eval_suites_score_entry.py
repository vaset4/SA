"""独立评分 CLI 的口径入口回归。作者：xxx。"""

from __future__ import annotations

import pytest

from evals.suites.run_eval import parse_args


def test_score_requires_explicit_measurement_policy() -> None:
    """无参数；独立评分缺少 MeasurementPolicy 时必须在 CLI 入口拒绝。"""
    with pytest.raises(SystemExit) as exc_info:
        parse_args([
            "score",
            "--frozen",
            "frozen.jsonl",
            "--output",
            "grading.json",
            "--kind",
            "retrieval",
        ])
    assert exc_info.value.code == 2
