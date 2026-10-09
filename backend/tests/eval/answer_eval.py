"""回答评估骨架：跑全链路，评答案与证据的一致性。

用法：
    python -m pytest tests/eval/answer_eval.py -v --flag-baseline

注意：真正的端到端评估需要 LLM 运行时和检索服务。
当前仅验证评估框架可加载、样本格式正确。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "intent_samples"


def _load_samples(intent_type: str) -> list[dict]:
    path = FIXTURES_DIR / f"{intent_type}.jsonl"
    if not path.exists():
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class TestAnswerEvalSkeleton:
    """回答评估骨架（需要 LLM 运行时才能真正跑全链路）。

    当前仅验证样本格式和评估框架可加载。
    """

    @pytest.mark.parametrize("intent_type", ["lookup", "compare", "explore"])
    def test_sample_has_forbidden_patterns(self, intent_type: str):
        samples = _load_samples(intent_type)
        assert len(samples) >= 20
        for i, sample in enumerate(samples):
            assert "forbidden_claim_patterns" in sample, f"样本 {i} 缺少 forbidden_claim_patterns"
            assert isinstance(sample["forbidden_claim_patterns"], list)

    def test_eval_framework_importable(self):
        from app.services.internal_research.intent_classifier import classify_intent
        from app.services.internal_research.focus_resolver import resolve_focus
        from app.services.internal_research.answer_brief_builder import build, serialize_brief_to_prompt_text
        assert callable(classify_intent)
        assert callable(resolve_focus)
        assert callable(build)
        assert callable(serialize_brief_to_prompt_text)
