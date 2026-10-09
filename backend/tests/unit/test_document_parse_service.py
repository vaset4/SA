from __future__ import annotations

from app.services.ingest import document_parse_service


def test_extract_abstract_skips_affiliation_paragraph() -> None:
    full_text = (
        "Graph WaveNet for Deep Spatial-Temporal Graph Modeling\n\n"
        "1 Centre for Artificial Intelligence, FEIT, University of Technology Sydney, Australia "
        "2 Faculty of Information Technology, Monash University, Australia\n\n"
        "Traffic forecasting is a core problem in intelligent transportation systems. "
        "This paper proposes a graph-based spatio-temporal model that captures hidden dependencies "
        "and improves forecasting accuracy across multiple benchmarks.\n\n"
        "Introduction\n\n"
        "More body text."
    )

    abstract = document_parse_service._extract_abstract_from_text(full_text)

    assert "University of Technology Sydney" not in abstract
    assert abstract.startswith("Traffic forecasting is a core problem")


def test_extract_abstract_prefers_explicit_abstract_section() -> None:
    full_text = (
        "Sample Title\n\n"
        "Abstract\n"
        "We study retrieval augmented generation for traffic forecasting.\n"
        "The paper focuses on robust evidence grounding.\n\n"
        "Keywords: traffic forecasting, rag"
    )

    abstract = document_parse_service._extract_abstract_from_text(full_text)

    assert "retrieval augmented generation" in abstract.lower()
    assert "Keywords" not in abstract
