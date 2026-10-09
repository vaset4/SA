"""Research planner contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models.research_planner_contracts import (
    ClaimPolicy,
    EvidencePolicy,
    ResearchPlanDraft,
    ResearchUnitResult,
)
from research_planner_test_utils import make_draft, make_unit


def test_free_research_action_accepted() -> None:
    unit = make_unit(
        action="analyze why the proposed method may be unstable in real traffic anomaly scenarios",
        operation_family=None,
    )
    plan = ResearchPlanDraft.model_validate(make_draft([unit]))
    assert plan.units[0].research_action.startswith("analyze why")


def test_new_phrasing_no_enum_change() -> None:
    unit = make_unit(
        goal="Design a follow-up validation idea",
        action="propose an evidence-grounded validation plan for transfer robustness",
        operation_family="creative_analysis",
    )
    plan = ResearchPlanDraft.model_validate(make_draft([unit]))
    assert plan.units[0].operation_family == "creative_analysis"


def test_operation_family_optional() -> None:
    plan = ResearchPlanDraft.model_validate(make_draft([make_unit(operation_family=None)]))
    assert plan.units[0].operation_family is None


def test_output_role_top_level_optional_and_strict() -> None:
    missing = ResearchPlanDraft.model_validate(make_draft([make_unit()]))
    explicit = ResearchPlanDraft.model_validate(make_draft([
        make_unit(source_type="inference_from_units", output_role="final_recommendation")
    ]))

    assert missing.units[0].output_role is None
    assert explicit.units[0].output_role == "final_recommendation"
    with pytest.raises(ValidationError):
        ResearchPlanDraft.model_validate(make_draft([make_unit(output_role="answer")]))


def test_output_role_not_allowed_inside_expected_output() -> None:
    unit = make_unit()
    unit["expected_output"]["output_role"] = "evidence"

    with pytest.raises(ValidationError):
        ResearchPlanDraft.model_validate(make_draft([unit]))


def test_unit_status_enum() -> None:
    valid = ResearchUnitResult(unit_id="u1", status="evidence_found")
    assert valid.status == "evidence_found"
    with pytest.raises(ValidationError):
        ResearchUnitResult(unit_id="u1", status="answered")


def test_allowed_claim_types_enum() -> None:
    policy = ClaimPolicy(allowed_claim_types=["paper_supported"])
    assert policy.allowed_claim_types == ["paper_supported"]
    with pytest.raises(ValidationError):
        ClaimPolicy(allowed_claim_types=["unsupported_fact"])


def test_preferred_block_types_optional_and_strict() -> None:
    omitted = EvidencePolicy(source_type="local_direct_text")
    explicit = EvidencePolicy(source_type="local_direct_text", preferred_block_types=["table", "formula"])

    assert omitted.preferred_block_types == []
    assert explicit.preferred_block_types == ["table", "formula"]
    with pytest.raises(ValidationError):
        EvidencePolicy(source_type="local_direct_text", preferred_block_types=["chart"])


def test_planner_config_has_no_runtime_chain_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """工程边界配置项保留，但选择运行模型的两个开关必须已从配置层消失。"""
    for key in (
        "RAG_RESEARCH_PLANNER_MAX_UNITS",
        "RAG_PLANNER_LIGHTRAG_LOCATOR_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.rag_research_planner_max_units == 8
    assert settings.rag_planner_lightrag_locator_enabled is False
    assert "rag_research_planner_enabled" not in Settings.model_fields
    assert "rag_use_evidence_records" not in Settings.model_fields
