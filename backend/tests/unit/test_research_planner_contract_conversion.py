"""Research planner contract conversion-boundary tests."""

from __future__ import annotations

import json

import pytest

from app.models.research_planner_contracts import ResearchClaim, ResearchUnitResult
from app.services.internal_research.research_planner_conversion import (
    downgrade_to_compat_payload,
    ingest_legacy_evidence_records,
    reject_compat_payload_readback,
    to_message_payload,
)
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft


def test_round_trip_shape() -> None:
    plan = _valid_plan()
    results = [_supported_result()]
    payload = to_message_payload(plan, results)
    json.dumps(payload)
    assert payload["research_plan"]["units"][0]["unit_id"] == "u1"
    assert payload["research_unit_results"][0]["claims"][0]["claim_type"] == "paper_supported"


def test_downgrade_to_compat_payload() -> None:
    plan = _valid_plan()
    payload = downgrade_to_compat_payload(plan, [_supported_result()])
    assert payload["synthesis_state"]["status"] == "ready"
    assert payload["answer_brief"]["intent"] == "single_unit"
    assert "answer_brief" in payload


def test_no_readback() -> None:
    with pytest.raises(ValueError, match="derived sinks"):
        reject_compat_payload_readback({"synthesis_state": {}})


def test_legacy_evidence_ingest() -> None:
    record = {"evidence_id": "e1", "source": "milvus", "usable_as_citation": True}
    ingested = ingest_legacy_evidence_records([record])
    assert ingested == [record]
    assert ingested[0] is not record


def _valid_plan():
    result = validate_strategy_fixture(make_draft())
    assert result.plan is not None
    return result.plan


def _supported_result() -> ResearchUnitResult:
    claim = ResearchClaim(
        claim_id="c1",
        text="The paper uses attention.",
        claim_type="paper_supported",
        validation_status="validated",
        supporting_evidence_ids=["e1"],
    )
    return ResearchUnitResult(
        unit_id="u1",
        status="evidence_found",
        summary="found",
        evidence_ids=["e1"],
        claims=[claim],
    )
