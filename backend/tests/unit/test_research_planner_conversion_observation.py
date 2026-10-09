"""【研究主张】【转换观察】真实provider到reasoner保留原模型操作身份。作者：xxx。"""

import asyncio
import json

from app.models.research_planner_contracts import ResearchUnit
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.services.internal_research.research_planner_claim_provider import LLMClaimDraftProvider
from app.services.internal_research.research_planner_consumer_budget import evidence_text_field
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from o2_evidence_test_utils import direct_evidence
from research_planner_test_utils import make_unit
from test_llm_contract_standardization import response

pytest_plugins = ["test_llm_contract_standardization"]


def test_provider_version_binding_links_business_validation_to_original_operation(runtime):
    """参数为离线HTTP替身及真实trace；验证版本投影后的业务校验仍关联原schema操作。"""
    client, post, _cache, trace = runtime
    record = direct_evidence("conversion-observation")
    field, text = evidence_text_field(record)
    raw = {"claims": [{"text": "明确原文支持", "claim_type": "paper_supported",
        "supporting_evidence_ids": [record["evidence_id"]], "addressed_facet_ids": ["f1"],
        "supporting_spans": [{"evidence_id": record["evidence_id"], "field": field,
                              "start": 0, "end": len(text), "quote": text}]}]}
    post.side_effect = None
    post.return_value = response(json.dumps(raw, ensure_ascii=False))
    unit = ResearchUnit.model_validate(make_unit("u1", allowed_claims=["paper_supported"]))
    reasoner = ResearchUnitReasoner(draft_provider=LLMClaimDraftProvider(llm=client))
    outcome = asyncio.run(reasoner.reason(unit, [record], []))
    assert outcome.claims, outcome.errors
    events = [json.loads(item) for item in trace.evidence_observations]
    source = next(item["payload"] for item in events if item["stage"] == "structured_response_validation")
    converted = next(item["payload"] for item in events if item["stage"] == "structured_conversion")
    business = next(item["payload"] for item in events if item["stage"] == "structured_business_validation")
    assert source["schema"] == ResearchClaimStrategyDraftBatch.__name__
    assert source["output_fingerprint"] == converted["input_fingerprint"]
    assert converted["input_fingerprint"] != converted["output_fingerprint"]
    assert business["input_fingerprint"] == converted["output_fingerprint"]
    assert business["conversion_ids"] == [converted["conversion_id"]]
    assert business["operation_id"] == converted["operation_id"] == source["operation_id"]
    assert outcome.claims[0].supporting_spans[0].quote == text
    assert outcome.claims[0].supporting_spans[0].evidence_version
    assert "evidence_version" not in raw["claims"][0]["supporting_spans"][0]
