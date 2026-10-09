"""Planner 评测输入、观察和评分隔离回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import copy
import json

from evals.contracts import grading_projection, record_contracts
from evals.research_planner import live_fixture, run_eval
from evals.research_planner.metrics import score_record, score_records
from app.models.knowledge_access_contracts import EvidenceReadResult
from app.models.research_planner_contracts import ResearchUnit
from app.services.internal_research.research_planner_claim_provider import LLMClaimDraftProvider
from app.services.internal_research.research_planner_executor import PlannerExecutionContext, _call_adapter
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner


def _record() -> dict:
    """无参数；返回带独立预期和替身种子的旧样本。"""
    return {
        "id": "integrity-1", "question": "Which module?", "category": "single_paper_lookup",
        "reference_answer": "private answer", "golden_evidence": ["private-evidence"],
        "critical_facts": [{"must_include": ["private answer"]}],
        "rubric": {"requires_citation": True},
        "expected": {"depth": "lookup", "failure_stage": "none", "min_units": 1,
                     "source_types": ["local_direct_text"], "statuses": ["evidence_found"]},
        "replay": {"answer": "fixture answer", "research_plan": {"units": []},
                   "research_unit_results": [{"unit_id": "seed"}], "planner_trace": {}},
    }


def _measurement(scored: dict, name: str) -> dict:
    """参数为单条评分和指标名；返回其测量详情。"""
    return next(item for item in scored["grading_result"]["measurements"] if item["name"] == name)


def test_fixture_environment_is_separate_from_actual_observation() -> None:
    """无参数；验证 runner 只收到执行输入，冻结结果保留实际观察且不修改样本。"""
    record = _record()
    original = copy.deepcopy(record)
    observed = {"answer": "actual answer", "research_plan": None,
                "research_unit_results": [], "planner_trace": {"code": "planner_analysis_failed"}}

    async def runner(state, **dependencies):
        assert "private answer" not in json.dumps(state)
        assert "private-evidence" not in json.dumps(state)
        assert "expected" not in dependencies["llm"].record
        state["project_scope"]["paper_ids"].append("runtime mutation")
        return copy.deepcopy(observed)

    live = asyncio.run(live_fixture.build_live_records_async([record], runner=runner))[0]
    case, run = record_contracts(live)
    projection = grading_projection(case, run)
    assert record == original
    assert projection["observation"] == observed
    assert "runtime mutation" not in json.loads(run.input_snapshot)["project_scope"]["paper_ids"]
    assert json.loads(run.fixture_snapshot)["seed"] == original["replay"]
    assert projection["truth"]["expected"] == original["expected"]
    assert run.run_mode == "planner_live_fixture"


def test_fixture_preloads_declared_context_and_restores_external_ports() -> None:
    """无参数；正式分析读取夹具上下文并完成minimal正文，退出后还原所有读取工厂。"""
    from app.repositories import research_memory_repository
    from app.services.internal_research import research_planner_pipeline as pipeline

    record = _record()
    record["expected"]["depth"] = "minimal"
    factories = (pipeline.get_paper_repository, pipeline.get_milvus_store,
                 pipeline.get_research_object_query_service, research_memory_repository._repository)
    result = live_fixture.build_live_records([record])[0]["replay"]
    assert result["turn_depth"] == "minimal"
    assert result["answer"] == "fixture answer"
    assert any("pdformer" in line for line in result["_paper_list_lines"])
    assert any("Fixture" in line for line in result["_section_structure_lines"])
    assert factories == (pipeline.get_paper_repository, pipeline.get_milvus_store,
                         pipeline.get_research_object_query_service, research_memory_repository._repository)


def test_frozen_observation_is_scored_after_legacy_alias_changes() -> None:
    """无参数；验证修改可读别名不能替换冻结观察，序列化后仍可重评分。"""
    scored = score_record(_record())
    restored = json.loads(json.dumps(scored))
    restored["replay"] = {"turn_depth": "lookup"}
    restored["expected"] = {"depth": "lookup"}
    regraded = score_record(restored)
    assert regraded["grading_result"] == scored["grading_result"]
    assert regraded["run_record"] == scored["run_record"]


def test_expected_synthesis_failure_does_not_overwrite_actual_result() -> None:
    """无参数；预期综合失败不能把成功的 runner 返回伪造成失败。"""
    record = _record()
    record["expected"]["failure_stage"] = "synthesis"
    observed = {"answer": "real output", "research_unit_results": [], "planner_trace": {}}

    async def runner(_state, **_dependencies):
        return observed

    live = asyncio.run(live_fixture.build_live_records_async([record], runner=runner))[0]
    assert live["replay"] == observed
    assert score_record(live)["planner_failure_visibility"] == 0.0


def test_measurements_distinguish_missing_not_applicable_invalid_and_zero() -> None:
    """无参数；四种状态分别报告，实际未升级为测得的零分。"""
    base = _record()
    missing = score_record({**base, "expected": {"should_escalate_depth": True}, "replay": {}})
    skipped = score_record({**base, "expected": {"should_escalate_depth": False}, "replay": {}})
    invalid = score_record({**base, "expected": {"should_escalate_depth": "yes"}, "replay": {}})
    zero = score_record({**base, "expected": {"should_escalate_depth": True}, "replay": {"depth_transitions": []}})
    name = "depth_escalation_recall"
    assert _measurement(missing, name)["status"] == "missing"
    assert _measurement(skipped, name)["status"] == "not_measured"
    assert _measurement(invalid, name)["status"] == "invalid_input"
    assert _measurement(zero, name) == {
        "name": name, "status": "measured", "reason": "depth_escalation_observed",
        "value": 0.0, "numerator": 0.0, "denominator": 1.0,
    }


def test_missing_applicable_citation_observation_blocks_partial_gate() -> None:
    """无参数；部分样本有满分也不能掩盖另一条适用样本缺失 Evidence。"""
    base = _record()
    claim = {"claim_type": "paper_supported", "supporting_evidence_ids": ["e1"],
             "validation_status": "validated"}
    observed = {"research_unit_results": [{"claims": [claim]}],
                "evidence_records": [{"evidence_id": "e1", "usable_as_citation": True}]}
    complete = {**base, "replay": observed}
    partial = {**base, "id": "partial", "replay": {"research_unit_results": [{"claims": [claim]}]}}
    scored = score_records([complete, partial])
    coverage = scored["coverage_summary"]["measurements"]
    assert coverage["citation_verifiability_rate"]["measured_count"] == 1
    assert coverage["citation_verifiability_rate"]["missing_count"] == 1
    metrics = {**scored["metrics"], "memory_injection_compliance_rate": 1.0, "stale_memory_leak_rate": 0.0}
    gate = run_eval.evaluate_compliance_gate(metrics, measurement_coverage=coverage)
    assert not gate["passed"]
    assert any(item["metric"] == "citation_verifiability_rate" and item["reason"] == "incomplete_measurement"
               for item in gate["violations"])


def test_citation_requires_boolean_eligibility_and_claim_type() -> None:
    """无参数；字符串 false 不能充当可引用真值，未知主张类型不能退出分母。"""
    claim = {"claim_type": "paper_supported", "supporting_evidence_ids": ["e1"], "validation_status": "validated"}
    observed = {"research_unit_results": [{"claims": [claim]}],
                "evidence_records": [{"evidence_id": "e1", "usable_as_citation": "false"}]}
    result = score_record({**_record(), "replay": observed})
    assert _measurement(result, "citation_verifiability_rate")["status"] == "invalid_input"
    del claim["claim_type"]
    result = score_record({**_record(), "replay": observed})
    assert _measurement(result, "citation_verifiability_rate")["status"] == "missing"


def test_fixture_adapters_preserve_seed_labels_and_use_new_read_identities() -> None:
    """无参数；模拟取证经过正式适配器入口和主张生成，种子标签不被重写或冒充实际证据 ID。"""
    record = _record()
    record["replay"]["research_unit_results"] = [{"unit_id": "u1", "evidence_ids": ["seed-e1"], "claims": [{
        "text": "Fixture finding", "claim_type": "paper_supported", "supporting_evidence_ids": ["seed-e1"],
    }]}]
    original = copy.deepcopy(record)
    environment = live_fixture._fixture_environment(record)
    fixture = live_fixture._RecordFixture(record=environment)
    unit = ResearchUnit.model_validate(live_fixture._complete_unit({}, 1, previous_units=[]))
    result = asyncio.run(_call_adapter(fixture._direct_text_adapter, unit, PlannerExecutionContext(question="test")))
    assert isinstance(result, EvidenceReadResult)
    evidence_id = result.records[0]["evidence_id"]
    assert evidence_id.startswith("ev-r1-") and evidence_id != "seed-e1"
    reasoner = ResearchUnitReasoner(draft_provider=LLMClaimDraftProvider(llm=live_fixture._FixtureLLM(environment)))
    outcome = asyncio.run(reasoner.reason(unit, [dict(item) for item in result.records], []))
    assert outcome.claims[0].supporting_evidence_ids == [evidence_id]
    metadata_unit = ResearchUnit.model_validate(live_fixture._complete_unit(
        {"evidence_policy": {"source_type": "local_metadata"}}, 1, previous_units=[],
    ))
    metadata = asyncio.run(_call_adapter(fixture._metadata_adapter, metadata_unit, PlannerExecutionContext(question="test")))
    assert metadata.records[0]["layer"] == "metadata"
    assert metadata.read_report.status == "not_applicable"
    assert record == original and environment["seed"] == original["replay"]
