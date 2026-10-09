from app.services.graph.research_object_normalizer import (
    build_entity_key,
    dedupe_preserve_order,
    normalize_entity_name,
)


def test_normalize_entity_name_trims_case_and_whitespace() -> None:
    assert normalize_entity_name("  Transformer  ") == "transformer"
    assert normalize_entity_name("Top-1   Accuracy") == "top-1 accuracy"


def test_build_entity_key_is_stable() -> None:
    assert build_entity_key("default", "Method", " Transformer ") == "default:Method:transformer"


def test_dedupe_preserve_order_filters_blank_and_duplicates() -> None:
    assert dedupe_preserve_order(["Transformer", " transformer ", "", "BERT"]) == ["Transformer", "BERT"]
