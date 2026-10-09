"""answer_brief_builder 单元测试。"""

import pytest

from app.services.internal_research.answer_brief_builder import (
    build,
    serialize_brief_to_prompt_text,
)


def _make_evidence(paper_id, layer="direct_text", usable=True):
    return {
        "evidence_id": f"ev_{paper_id}",
        "source": "milvus",
        "layer": layer,
        "paper_id": paper_id,
        "chunk_id": f"chunk_{paper_id}",
        "excerpt": "some text",
        "confidence": 0.9,
        "usable_as_citation": usable,
    }


class TestBuild:
    def test_basic_build(self):
        state = {
            "evidence_records": [_make_evidence("p1")],
            "missing_evidence": [],
            "intent": "lookup",
            "focus": {"turn_focus": ["p1"], "fallback_reason": ""},
        }
        brief = build(state)
        assert brief["intent"] == "lookup"
        assert len(brief["usable_evidence"]) == 1
        assert "不要把图线索说成原文事实" in brief["forbidden_claims"][0]

    def test_graph_hint_not_in_usable(self):
        state = {
            "evidence_records": [
                _make_evidence("p1", layer="graph_hint", usable=False),
                _make_evidence("p2", layer="direct_text", usable=True),
            ],
            "missing_evidence": [],
            "intent": "explore",
            "focus": {"turn_focus": ["p1", "p2"], "fallback_reason": ""},
        }
        brief = build(state)
        assert len(brief["usable_evidence"]) == 1
        assert brief["usable_evidence"][0]["paper_id"] == "p2"
        assert "图线索" in brief["answer_guidance"]

    def test_compare_coverage_forbidden_claims(self):
        state = {
            "evidence_records": [_make_evidence("p1")],
            "missing_evidence": [{"paper_id": "p2", "dimension": "method", "reason": "no hit"}],
            "intent": "compare",
            "focus": {"turn_focus": ["p1", "p2"], "fallback_reason": ""},
        }
        brief = build(state)
        forbidden = " ".join(brief["forbidden_claims"])
        assert "p2" in forbidden

    def test_fallback_reason_in_uncertainties(self):
        state = {
            "evidence_records": [_make_evidence("p1")],
            "missing_evidence": [],
            "intent": "lookup",
            "focus": {"turn_focus": ["p1"], "fallback_reason": "lookup_no_focus"},
        }
        brief = build(state)
        assert any("lookup_no_focus" in u for u in brief["uncertainties"])

    def test_empty_evidence(self):
        state = {
            "evidence_records": [],
            "missing_evidence": [],
            "intent": "explore",
            "focus": {"turn_focus": [], "fallback_reason": "explore_default"},
        }
        brief = build(state)
        assert brief["usable_evidence"] == []
        assert len(brief["forbidden_claims"]) >= 1


class TestSerializeBriefToPromptText:
    def test_contains_key_phrases(self):
        brief = {
            "intent": "compare",
            "focus": "焦点论文: p1, p2",
            "usable_evidence": [_make_evidence("p1")],
            "missing_evidence": [{"paper_id": "p2", "dimension": "method", "reason": "no hit"}],
            "uncertainties": ["焦点降级：compare_insufficient_focus"],
            "answer_guidance": "图线索仅供参考，不可作为事实引用。",
            "forbidden_claims": ["不要把图线索说成原文事实。"],
        }
        text = serialize_brief_to_prompt_text(brief)
        assert "Intent: compare" in text
        assert "可用证据: 1 条" in text
        assert "缺失证据: 1 项" in text
        assert "不确定性" in text
        assert "FORBIDDEN" in text
        assert "不要把图线索说成原文事实" in text

    def test_minimal_brief(self):
        brief = {
            "intent": "",
            "focus": "",
            "usable_evidence": [],
            "missing_evidence": [],
            "uncertainties": [],
            "answer_guidance": "",
            "forbidden_claims": [],
        }
        text = serialize_brief_to_prompt_text(brief)
        assert "可用证据: 0 条" in text
