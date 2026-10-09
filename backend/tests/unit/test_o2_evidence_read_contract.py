"""O2 正式执行、补证登记与读取报告贯通回归，作者：xxx。"""
import asyncio
from copy import deepcopy

import pytest

from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport
from app.models.research_planner_contracts import ResearchClaimDraft, ResearchUnitResult, RepairQuery, RepairQueryPlan
from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials
from app.services.internal_research.research_planner_executor import EvidenceCapabilityError, ResearchPlannerExecutor
from app.services.internal_research.research_planner_evidence_repair import (
    build_evidence_sufficiency_report, execute_with_repair, merge_repair_evidence, rerun_affected_units,
)
from app.services.internal_research.research_planner_plan_prompt import build_planner_prompt
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from research_planner_strategy_test_utils import validate_strategy_fixture
from o2_evidence_test_utils import material, read_result
from research_planner_test_utils import make_draft, make_unit


def _plan(minimum=1, *, source_type="local_direct_text"):
    """参数为已有最低引用数；返回正式校验后的正文单元计划。"""
    unit = make_unit()
    unit["evidence_policy"]["source_type"] = source_type
    if source_type == "local_section_text":
        unit["evidence_policy"]["required_sections"] = ["Method"]
    unit["evidence_policy"]["min_citations_per_target"] = minimum
    outcome = validate_strategy_fixture(make_draft([unit]))
    assert outcome.plan is not None
    return outcome.plan


def test_executor_and_repair_count_groups_but_observe_actual_records():
    """无参数；同组范围扩展在执行和补证中都只算一份材料，实际记录数仍为二。"""
    records = convert_knowledge_materials((material("first"), material("first and more")))
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": lambda *_: read_result(records)})
    plan = _plan(minimum=2)
    execution = asyncio.run(executor.execute_with_evidence(plan))
    assert execution.results[0].status == "insufficient_evidence"
    sufficiency = build_evidence_sufficiency_report(
        plan=plan, results=execution.results, evidence_records=execution.evidence_records, pass_index=1,
    )
    report = sufficiency.unit_reports[0]
    assert report.target_coverage == {"paper-a": 1}
    assert next(gap for gap in report.hard_gaps if gap.gap_type == "target_coverage_missing").found_count == 1
    merge, merged = merge_repair_evidence(first_records=records[:1], repair_records=records[1:], query_plans=[])
    assert len(merged) == 2
    assert merge.pass_counts["merged_records"] == 2
    assert merge.pass_counts["repair_records"] == 1


def test_partial_read_report_reaches_reasoning_outside_evidence():
    """无参数；部分读取说明独立送给主张提供方并保存单元 metadata，不进入证据池。"""
    records = convert_knowledge_materials((material(),))
    report = KnowledgeReadReport(operation="sections", status="partial", requested=("Method", "Results"),
                                 remaining=({"locator": "Results", "reason": "character_budget"},))
    seen = []

    def provider(unit, evidence, dependencies, *, read_reports):
        """参数为单元、正文、依赖与独立读取说明；返回只引用实际正文的离线主张。"""
        seen.append((evidence, read_reports))
        return [ResearchClaimDraft(text="Supported method.", claim_type="paper_supported",
                                   supporting_evidence_ids=[evidence[0]["evidence_id"]])]

    executor = ResearchPlannerExecutor(
        adapters={"local_direct_text": lambda *_: EvidenceReadResult(tuple(records), report)},
        reasoner=ResearchUnitReasoner(draft_provider=provider),
    )
    execution = asyncio.run(executor.execute_with_evidence(_plan()))
    assert seen[0][0] == records
    assert seen[0][1] == [report.to_dict()]
    assert execution.results[0].metadata["read_reports"] == [report.to_dict()]
    assert execution.evidence_records == records


def test_same_id_changed_content_is_recorded_as_repair_conflict():
    """无参数；补证遇同 ID 不同正文会记录冲突，既有证据正文保持原样。"""
    first = convert_knowledge_materials((material(),))[0]
    conflicting = deepcopy(first)
    conflicting["excerpt"] = "changed"
    report, records = merge_repair_evidence(first_records=[first], repair_records=[conflicting], query_plans=[])
    assert report.conflict_records[0]["reason"] == "duplicate_evidence_conflict:excerpt"
    assert records == [first]


def test_repair_rerun_keeps_initial_and_supplemental_read_reports():
    """无参数；补证重跑同时持有首轮和补读报告，不因重新推理丢掉部分读取事实。"""
    plan = _plan()
    unit_id = plan.units[0].unit_id
    record = convert_knowledge_materials((material(),))[0]
    initial = {"operation": "chunks", "status": "partial"}
    extra = {"operation": "chunks", "status": "complete"}
    first = ResearchUnitResult(unit_id=unit_id, status="insufficient_evidence", metadata={"read_reports": [initial]})
    repair = ResearchUnitResult(unit_id="repair-1", status="evidence_found", evidence_ids=[record["evidence_id"]],
                                metadata={"read_reports": [extra]})
    query_plan = RepairQueryPlan(source_report_pass=1, linked_unit_id=unit_id, repair_unit_id="repair-1",
                                 queries=[RepairQuery(query="method", source_gap_id="gap-1")])
    seen = []

    def provider(unit, evidence, dependencies, *, read_reports):
        """参数为补证重跑事实；记录读取报告并返回正文支持主张。"""
        seen.extend(read_reports)
        return [ResearchClaimDraft(text="Supported method.", claim_type="paper_supported",
                                   supporting_evidence_ids=[record["evidence_id"]])]

    results = asyncio.run(rerun_affected_units(
        plan=plan, first_results=[first], evidence_records=[record], repair_results=[repair],
        query_plans=[query_plan], executor=ResearchPlannerExecutor(reasoner=ResearchUnitReasoner(draft_provider=provider)),
    ))
    assert seen == [initial, extra]
    assert results[0].metadata["read_reports"] == [initial, extra]


def test_adapter_list_return_is_an_explicit_contract_error():
    """无参数；生产适配器不能继续沿用列表/新合同两套协议。"""
    executor = ResearchPlannerExecutor(adapters={"local_direct_text": lambda *_: []})
    with pytest.raises(EvidenceCapabilityError, match="EvidenceReadResult"):
        asyncio.run(executor.execute_with_evidence(_plan()))


def test_repair_reasoning_failure_keeps_completed_read_reports_for_replanning():
    """无参数；补读成功后推理失败仍保留两次读取事实，下一轮不会把未读部分说成不存在。"""
    plan = _plan()
    record = convert_knowledge_materials((material(),))[0]
    first_report = KnowledgeReadReport(
        operation="chunks", status="complete", requested=("initial-query",),
    )
    repair_report = KnowledgeReadReport(
        operation="chunks", status="partial",
        remaining=({"chunk_id": "v3:batch:1", "reason": "character_budget"},),
    )

    def adapter(unit, context):
        """参数为正式或补读单元及上下文；返回各次实际材料与不同读取报告。"""
        if unit.unit_id == plan.units[0].unit_id:
            return EvidenceReadResult((), first_report)
        return EvidenceReadResult((record,), repair_report)

    def provider(unit, evidence, dependencies, *, read_reports):
        """参数为实际取证成果；只在原单元补读后的推理阶段模拟显式失败。"""
        if unit.unit_id == plan.units[0].unit_id:
            raise RuntimeError("rerun provider unavailable")
        return [ResearchClaimDraft(text="Supported method.", claim_type="paper_supported",
                                   supporting_evidence_ids=[record["evidence_id"]])]

    executor = ResearchPlannerExecutor(
        adapters={"local_direct_text": adapter}, reasoner=ResearchUnitReasoner(draft_provider=provider),
    )
    outcome = asyncio.run(execute_with_repair(plan=plan, executor=executor))
    result = outcome.results[0]
    assert outcome.repair_status == "failed" and result.status == "blocked"
    assert result.claims == [] and outcome.evidence_records == [record]
    assert result.metadata["read_reports"] == [first_report.to_dict(), repair_report.to_dict()]
    prompt = build_planner_prompt({
        "planner_observation": {"summary": "rerun failed"},
        "research_unit_results": [result.model_dump(mode="json")],
    })
    assert "initial-query" in prompt and "character_budget" in prompt


@pytest.mark.parametrize("source_type", ["local_direct_text", "local_section_text"])
@pytest.mark.parametrize("missing", ["read_contract_version", "material_group_id", "source_spans", "version_and_layer"])
def test_live_body_adapter_rejects_missing_source_contract(missing, source_type):
    """实时正文缺版本、分组或范围均明确失败，不能包装后作为旧记录混进新运行。"""
    record = convert_knowledge_materials((material(),))[0]
    if missing == "version_and_layer":
        del record["read_contract_version"]
        del record["layer"]
    else:
        del record[missing]
    executor = ResearchPlannerExecutor(adapters={source_type: lambda *_: read_result([record])})
    with pytest.raises(EvidenceCapabilityError):
        asyncio.run(executor.execute_with_evidence(_plan(source_type=source_type)))
