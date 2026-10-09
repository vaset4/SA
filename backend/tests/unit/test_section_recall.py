from __future__ import annotations

import pytest

from evals.ragas import section_recall


def test_plain_golden_evidence_rejects_id_like_labels() -> None:
    records = [
        {
            "question": "Q",
            "paper_ids": ["paper-a"],
            "golden_evidence": ["chunk_paper_a_0_0"],
        }
    ]

    with pytest.raises(ValueError, match="text spans"):
        section_recall.validate_plain_golden_evidence(records)


def test_section_recall_is_stable_when_chunk_ids_change() -> None:
    records = [
        {
            "question": "adaptive adjacency hidden dependency",
            "paper_ids": ["paper-a"],
            "golden_evidence": ["hidden spatial dependencies"],
        }
    ]
    sections_by_paper = {
        "paper-a": [
            {
                "title": "3.2 Adaptive Matrix",
                "section_path": "3.2 Adaptive Matrix",
                "text": "The model discovers hidden spatial dependencies from traffic data.",
            }
        ]
    }
    chunk = {
        "chunk_id": "chunk-a",
        "parent_id": "parent-a",
        "section_path": "3.2 Adaptive Matrix",
        "embedding_text": "adaptive adjacency hidden spatial dependencies",
        "text": "The model discovers hidden spatial dependencies from traffic data.",
        "full_text": "The model discovers hidden spatial dependencies from traffic data.",
    }

    first = section_recall.evaluate_section_recall(
        records=records,
        sections_by_paper=sections_by_paper,
        chunks_by_paper={"paper-a": [chunk]},
    )
    second = section_recall.evaluate_section_recall(
        records=records,
        sections_by_paper=sections_by_paper,
        chunks_by_paper={"paper-a": [{**chunk, "chunk_id": "renumbered"}]},
    )

    assert first["section_recall"] == 1.0
    assert second["section_recall"] == 1.0
    assert first["records"][0]["expected_sections"] == second["records"][0]["expected_sections"]


def test_section_recall_uses_sections_not_full_text_substrings() -> None:
    records = [
        {
            "question": "temporal convolution",
            "paper_ids": ["paper-a"],
            "golden_evidence": ["gated temporal convolution"],
        }
    ]
    sections_by_paper = {
        "paper-a": [
            {
                "title": "3.3 Temporal",
                "section_path": "3.3 Temporal",
                "text": "The section explains gated temporal convolution.",
            },
            {
                "title": "3.4 Framework",
                "section_path": "3.4 Framework",
                "text": "Neighbor section.",
            },
        ]
    }
    chunks_by_paper = {
        "paper-a": [
            {
                "chunk_id": "framework",
                "parent_id": "merged-parent",
                "section_path": "3.4 Framework",
                "embedding_text": "temporal convolution framework",
                "text": "Neighbor section.",
                "full_text": "The section explains gated temporal convolution. Neighbor section.",
            }
        ]
    }

    result = section_recall.evaluate_section_recall(
        records=records,
        sections_by_paper=sections_by_paper,
        chunks_by_paper=chunks_by_paper,
    )

    assert result["section_recall"] == 0.0
