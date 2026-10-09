"""Planner direct-text structural preference tests."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.internal_research.research_planner_direct_text import rank_direct_text_hits


def test_required_sections_rank_section_path_match_first() -> None:
    hits = [
        _hit("c1", section="Discussion", section_path="5 Discussion"),
        _hit("c2", section="Method", section_path="Paper / 3 Method / 3.1 Model"),
    ]

    ranked = rank_direct_text_hits(hits, _policy(required_sections=["method"]))

    assert [hit["chunk_id"] for hit in ranked] == ["c2", "c1"]


def test_required_sections_use_section_when_path_missing() -> None:
    hits = [
        _hit("c1", section="Introduction", section_path=""),
        _hit("c2", section="Evaluation Results", section_path=""),
    ]

    ranked = rank_direct_text_hits(hits, _policy(required_sections=["results"]))

    assert [hit["chunk_id"] for hit in ranked] == ["c2", "c1"]


def test_preferred_block_types_rank_without_filtering() -> None:
    hits = [
        _hit("c1", block_type="paragraph"),
        _hit("c2", block_type="table"),
    ]

    ranked = rank_direct_text_hits(hits, _policy(preferred_block_types=["table"]))

    assert [hit["chunk_id"] for hit in ranked] == ["c2", "c1"]
    assert len(ranked) == 2


def test_evidence_dimensions_add_deterministic_hints() -> None:
    method_hits = [
        _hit("c1", section_path="7 Appendix"),
        _hit("c2", section_path="3 Architecture"),
    ]
    result_hits = [
        _hit("c3", section_path="3 Method", block_type="paragraph"),
        _hit("c4", section_path="4 Experiments", block_type="table"),
    ]
    definition_hits = [
        _hit("c5", section_path="4 Experiments", block_type="table"),
        _hit("c6", section_path="2 Preliminaries", block_type="formula"),
    ]

    assert rank_direct_text_hits(method_hits, _policy(evidence_dimensions=["mechanism"]))[0]["chunk_id"] == "c2"
    assert rank_direct_text_hits(result_hits, _policy(evidence_dimensions=["metric"]))[0]["chunk_id"] == "c4"
    assert rank_direct_text_hits(definition_hits, _policy(evidence_dimensions=["definition"]))[0]["chunk_id"] == "c6"


def test_chinese_method_dimension_adds_method_section_hint() -> None:
    hits = [
        _hit("c1", section_path="4 Experiments"),
        _hit("c2", section_path="3 Architecture"),
    ]

    ranked = rank_direct_text_hits(hits, _policy(evidence_dimensions=["融合模型架构思想"]))

    assert ranked[0]["chunk_id"] == "c2"


def test_unrecognized_dimension_does_not_change_order() -> None:
    hits = [_hit("c1", section_path="A"), _hit("c2", section_path="B")]

    ranked = rank_direct_text_hits(hits, _policy(evidence_dimensions=["custom unexplained axis"]))

    assert [hit["chunk_id"] for hit in ranked] == ["c1", "c2"]


def test_method_architecture_query_prefers_model_components_over_training_setup() -> None:
    hits = [
        _hit(
            "loss",
            section_path="3 Method",
            preview="LSTTN uses a pretraining loss function on masked input data.",
        ),
        _hit(
            "dataset",
            section_path="3 Method",
            preview="STD-PLM partitions the dataset into training, validation, and testing sets with 6:2:2.",
        ),
        _hit(
            "architecture",
            section_path="3 Method",
            preview=(
                "The framework contains a temporal encoder, spatial graph module, "
                "attention layer, decoder, and spatiotemporal data flow."
            ),
        ),
    ]

    ranked = rank_direct_text_hits(
        hits,
        _policy(required_sections=["method"], evidence_dimensions=["architecture mechanism"]),
    )

    assert [hit["chunk_id"] for hit in ranked] == ["architecture", "loss", "dataset"]


def _policy(
    *,
    required_sections: list[str] | None = None,
    preferred_block_types: list[str] | None = None,
    evidence_dimensions: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        target_paper_ids=["paper-a"],
        required_sections=required_sections or [],
        preferred_block_types=preferred_block_types or [],
        evidence_dimensions=evidence_dimensions or [],
    )


def _hit(
    chunk_id: str,
    *,
    section: str = "Method",
    section_path: str = "3 Method",
    block_type: str = "paragraph",
    preview: str | None = None,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "parent_id": chunk_id,
        "paper_id": "paper-a",
        "section": section,
        "section_path": section_path,
        "block_type": block_type,
        "preview": preview or chunk_id,
        "score": 0.9,
    }
