"""检索评估骨架：对每个样本跑协调层，断言 direct_text 召回率与 missing_evidence 准确率。

用法：
    python -m pytest tests/eval/retrieval_eval.py -v

注意：这是骨架版基线，样本量小（每类 20 条），容易过拟合规则。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.services.internal_research.intent_classifier import classify_intent

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "intent_samples"

_DIRECT_TEXT_RECALL_THRESHOLD = 0.7
_MISSING_EVIDENCE_PRECISION_THRESHOLD = 0.8


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


class TestIntentClassificationAccuracy:
    """规则版意图识别准确率评估。"""

    @pytest.mark.parametrize("intent_type", ["lookup", "compare", "explore"])
    def test_rule_accuracy_per_intent(self, intent_type: str):
        samples = _load_samples(intent_type)
        assert len(samples) >= 20, f"{intent_type} 样本不足 20 条"

        correct = 0
        for sample in samples:
            result = classify_intent(sample["question"])
            if result["intent"] == sample["expected_intent"]:
                correct += 1

        accuracy = correct / len(samples)
        assert accuracy >= 0.8, (
            f"{intent_type} 规则版准确率 {accuracy:.2%} < 80% "
            f"(correct={correct}/{len(samples)})"
        )


class TestRetrievalEvalSkeleton:
    """检索评估骨架（需要运行时环境才能真正跑协调层）。

    当前仅验证样本格式正确、字段完整。
    真正的端到端评估需要 Milvus/LightRAG/Neo4j 运行时。
    """

    @pytest.mark.parametrize("intent_type", ["lookup", "compare", "explore"])
    def test_sample_format(self, intent_type: str):
        samples = _load_samples(intent_type)
        assert len(samples) >= 20
        required_fields = {"question", "expected_intent", "expected_focus", "expected_paper_coverage", "forbidden_claim_patterns"}
        for i, sample in enumerate(samples):
            missing = required_fields - set(sample.keys())
            assert not missing, f"样本 {i} 缺少字段: {missing}"

    def test_all_intents_have_samples(self):
        for intent_type in ("lookup", "compare", "explore"):
            samples = _load_samples(intent_type)
            assert len(samples) >= 20, f"{intent_type} 样本不足"
