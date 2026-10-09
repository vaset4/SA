from app.services.graph.academic_schema import (
    ACADEMIC_DOC_EXTRACTION_PROMPT,
    ACADEMIC_ENTITY_EXTRACTION_EXAMPLES,
    ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ACADEMIC_ENTITY_TYPES,
)


def test_academic_entity_types_are_fixed() -> None:
    assert ACADEMIC_ENTITY_TYPES == ("Method", "Dataset", "Metric", "Task")


def test_prompts_include_all_entity_types() -> None:
    prompt_text = "\n".join(
        [
            ACADEMIC_DOC_EXTRACTION_PROMPT,
            ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT,
            *ACADEMIC_ENTITY_EXTRACTION_EXAMPLES,
        ]
    )

    for entity_type in ACADEMIC_ENTITY_TYPES:
        assert entity_type in prompt_text


def test_prompts_include_anti_hallucination_rules() -> None:
    prompt_text = "\n".join(
        [
            ACADEMIC_DOC_EXTRACTION_PROMPT,
            ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT,
        ]
    )

    for keyword in (
        "at least twice",
        "DO NOT extract author names",
        "empty",
        "valid JSON",
    ):
        assert keyword in prompt_text
