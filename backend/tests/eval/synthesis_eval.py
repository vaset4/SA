"""synthesis_eval: 整理评估骨架。

规则断言为主，覆盖多源证据合并、缺失证据识别、forbidden_claims 默认条目。
LLM 评分通过 SYNTHESIS_EVAL_USE_LLM=1 启用（可选附加）。

用法:
    python -m pytest backend/tests/eval/synthesis_eval.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synthesis_samples"


def _load_samples(intent: str) -> list[dict[str, Any]]:
    path = FIXTURES_DIR / f"{intent}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_mock_answer_brief(
    *,
    intent: str,
    paper_ids: list[str],
    has_direct_text: bool = True,
    missing_papers: list[str] | None = None,
) -> dict[str, Any]:
    from app.models.research_contracts import AnswerBrief, EvidenceRecord, MissingEvidence

    usable: list[EvidenceRecord] = []
    if has_direct_text:
        for pid in paper_ids:
            if pid not in (missing_papers or []):
                usable.append(EvidenceRecord(
                    evidence_id=f"ev-test-{pid}",
                    source="milvus",
                    layer="direct_text",
                    paper_id=pid,
                    chunk_id=f"{pid}_0_0",
                    parent_id=f"{pid}_0",
                    section="Methods",
                    page_num=3,
                    excerpt=f"Sample excerpt from {pid}",
                    expanded_text=f"Full text from {pid}",
                    summary="",
                    dimensions=[],
                    confidence=0.85,
                    usable_as_citation=True,
                ))

    usable.append(EvidenceRecord(
        evidence_id="ev-test-graph",
        source="lightrag",
        layer="graph_hint",
        paper_id="",
        chunk_id="",
        parent_id="",
        section="",
        page_num=0,
        excerpt="Graph hint context",
        expanded_text="",
        summary="Graph hint summary",
        dimensions=[],
        confidence=0.5,
        usable_as_citation=False,
    ))

    missing: list[MissingEvidence] = []
    for pid in (missing_papers or []):
        missing.append(MissingEvidence(
            paper_id=pid,
            dimension="direct_text",
            reason="no_hit_in_milvus",
            attempted_queries=[],
        ))

    brief = AnswerBrief(
        intent=intent,
        focus="project_scope",
        usable_evidence=usable,
        missing_evidence=missing,
        uncertainties=["sample uncertainty"],
        answer_guidance="Graph hint summary. Use direct text evidence for factual claims.",
        forbidden_claims=["不要把图线索说成原文事实"],
    )
    return brief


LOOKUP_SAMPLES = _load_samples("lookup")
COMPARE_SAMPLES = _load_samples("compare")
EXPLORE_SAMPLES = _load_samples("explore")


class TestSynthesisMultiSourceMerge:
    @pytest.mark.parametrize("sample", LOOKUP_SAMPLES, ids=[s["query"][:40] for s in LOOKUP_SAMPLES])
    def test_lookup_has_direct_text(self, sample):
        brief = _build_mock_answer_brief(
            intent="lookup",
            paper_ids=sample["paper_ids"],
            has_direct_text=sample.get("expected_direct_text", True),
        )
        direct_text_records = [
            e for e in brief["usable_evidence"] if e.get("layer") == "direct_text"
        ]
        assert len(direct_text_records) >= 1, "lookup should have at least one direct_text evidence"

    @pytest.mark.parametrize("sample", LOOKUP_SAMPLES, ids=[s["query"][:40] for s in LOOKUP_SAMPLES])
    def test_lookup_graph_hint_in_guidance(self, sample):
        brief = _build_mock_answer_brief(
            intent="lookup",
            paper_ids=sample["paper_ids"],
        )
        assert "graph" in brief["answer_guidance"].lower() or "hint" in brief["answer_guidance"].lower()

    @pytest.mark.parametrize("sample", EXPLORE_SAMPLES, ids=[s["query"][:40] for s in EXPLORE_SAMPLES])
    def test_explore_has_direct_text(self, sample):
        brief = _build_mock_answer_brief(
            intent="explore",
            paper_ids=sample["paper_ids"],
            has_direct_text=sample.get("expected_direct_text", True),
        )
        direct_text_records = [
            e for e in brief["usable_evidence"] if e.get("layer") == "direct_text"
        ]
        assert len(direct_text_records) >= 1


class TestSynthesisMissingEvidence:
    @pytest.mark.parametrize("sample", COMPARE_SAMPLES, ids=[s["query"][:40] for s in COMPARE_SAMPLES])
    def test_compare_missing_identified(self, sample):
        brief = _build_mock_answer_brief(
            intent="compare",
            paper_ids=sample["paper_ids"],
            missing_papers=sample.get("expected_missing", []),
        )
        expected_missing = set(sample.get("expected_missing", []))
        actual_missing = {m["paper_id"] for m in brief["missing_evidence"]}
        assert expected_missing <= actual_missing, (
            f"Expected missing papers {expected_missing} not found in {actual_missing}"
        )


class TestSynthesisForbiddenClaims:
    @pytest.mark.parametrize(
        "intent,samples",
        [("lookup", LOOKUP_SAMPLES), ("compare", COMPARE_SAMPLES), ("explore", EXPLORE_SAMPLES)],
    )
    def test_forbidden_claims_present(self, intent, samples):
        for sample in samples[:3]:
            brief = _build_mock_answer_brief(
                intent=intent,
                paper_ids=sample["paper_ids"],
            )
            assert brief["forbidden_claims"], "forbidden_claims should have at least one entry"
            assert any(
                "图" in claim or "graph" in claim.lower() or "事实" in claim
                for claim in brief["forbidden_claims"]
            )


@pytest.mark.skipif(
    not os.environ.get("SYNTHESIS_EVAL_USE_LLM"),
    reason="LLM eval disabled (set SYNTHESIS_EVAL_USE_LLM=1 to enable)",
)
class TestSynthesisLLMScoring:
    def test_llm_scoring_placeholder(self):
        pytest.skip("LLM scoring not yet implemented")
