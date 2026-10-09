"""O2 章节正式适配器、执行器、引用与额度合同测试，作者：xxx。"""
from __future__ import annotations

import asyncio

import pytest

from app.nodes import rag_node
from app.services.internal_research.research_planner_executor import EvidenceCapabilityError, PlannerUnitExecutor
from app.services.internal_research.research_planner_support import record_citable
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer
from research_planner_strategy_test_utils import validate_strategy_fixture
from app.services.knowledge_access_service import resolve_section_paths
from o2_evidence_test_utils import bind_publication, scope
from research_planner_test_utils import explicit_claim_provider, make_draft, make_unit

_SOURCE = "local_section_text"
_METHOD = "Paper / 3.1 Method"
_OTHER = "Paper / 4 Results"


def _row(index, path, text, *, block_type="paragraph"):
    """参数为序号、路径、正文及块类型；返回完整发布正文和故意截短的索引显示。"""
    return {"chunk_id": f"v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:{index}", "parent_id": f"parent-{index}", "paper_id": "paper-a",
            "section_path": path, "section": path.rsplit(" / ", 1)[-1], "block_type": block_type,
            "text": text, "full_text": text, "display_text": "index preview", "page_num": index + 1}


def _plan(sections=None):
    """参数为章节定位；返回经过正式校验的正文研究计划。"""
    unit = make_unit(source_type=_SOURCE, allowed_claims=["paper_supported"])
    unit["evidence_policy"]["required_sections"] = ["method"] if sections is None else sections
    result = validate_strategy_fixture(make_draft([unit]))
    assert result.plan is not None
    return result.plan


def _execute(monkeypatch, rows, *, sections=None, budget=24000, failure=None, project_scope=None):
    """参数为发布材料、章节和额度；通过正式章节适配器与执行器返回计划和执行报告。"""
    reads = bind_publication(monkeypatch, rows, failure=failure)
    monkeypatch.setattr(rag_node.settings, "rag_planner_section_text_max_chars", budget)
    state = {"project_scope": scope() if project_scope is None else project_scope}
    plan = _plan(sections)
    adapters = rag_node._planner_optional_adapters(state)
    report = asyncio.run(PlannerUnitExecutor(adapters=adapters,
        reasoner=ResearchUnitReasoner(draft_provider=explicit_claim_provider)).execute_with_evidence(plan))
    return plan, report, reads


def test_section_source_contract_and_output_role():
    """无参数；章节正文保持引用资格，但不能冒充最终推荐单元。"""
    plan = _plan()
    assert plan.units[0].output_role == "evidence"
    assert plan.units[0].claim_policy.citation_required is True
    unit = make_unit(source_type=_SOURCE, output_role="final_recommendation")
    assert validate_strategy_fixture(make_draft([unit])).failure is not None


@pytest.mark.parametrize("selector", ["method", "3.1", _METHOD])
def test_section_reads_published_body_and_all_source_spans(monkeypatch, selector):
    """参数为 patch；同节多个块按发布顺序拼正文，完整范围不缩成首块页码。"""
    rows = [_row(1, _METHOD, "second"), _row(0, _METHOD, "first")]
    _, report, reads = _execute(monkeypatch, rows, sections=[selector])
    assert report.capability_results == {_SOURCE: "success"}
    assert report.results[0].status == "evidence_found"
    record = report.evidence_records[0]
    assert record["excerpt"] == "first\n\nsecond"
    assert [span["chunk_id"] for span in record["source_spans"]] == ["v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", "v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:1"]
    assert all(span["field"] == "full_text" for span in record["source_spans"])
    assert record["source_location_status"] == "legacy_chunk"
    assert len(reads) == 1
    assert report.results[0].metadata["read_reports"][0]["status"] == "complete"


def test_partial_report_is_separate_from_citable_evidence(monkeypatch):
    """参数为 patch；额度不足的后续原子结构保留未读定位，不生成截断伪证据。"""
    rows = [_row(0, _METHOD, "first"), _row(1, _OTHER, "|a|b|\n|1|2|", block_type="table")]
    _, report, _ = _execute(monkeypatch, rows, sections=[_METHOD, _OTHER], budget=8)
    assert len(report.evidence_records) == 1
    assert report.evidence_records[0]["excerpt"] == "first"
    read_report = report.results[0].metadata["read_reports"][0]
    assert read_report["status"] == "partial"
    assert read_report["remaining"]
    assert report.results[0].evidence_ids == [report.evidence_records[0]["evidence_id"]]


def test_first_atomic_structure_cannot_be_cut_or_empty(monkeypatch):
    """参数为 patch；表格没有足够额度时明确失败，不能返回半表或资料不存在。"""
    rows = [_row(0, _METHOD, "| a | b |\n| 1 | 2 |", block_type="table")]
    with pytest.raises(EvidenceCapabilityError, match="KnowledgeReadBudgetError"):
        _execute(monkeypatch, rows, budget=3)


def test_long_plain_text_partial_read_has_exact_returned_span(monkeypatch):
    """参数为 patch；普通正文的部分读取范围对应实际返回字符，未读事实独立保存。"""
    _, report, _ = _execute(monkeypatch, [_row(0, _METHOD, "a" * 100)], budget=30)
    record = report.evidence_records[0]
    assert len(record["excerpt"]) == 30
    assert record["source_spans"][0]["end"] == 30
    assert report.results[0].metadata["read_reports"][0]["status"] == "partial"


def test_citation_and_synthesis_preserve_range_identity(monkeypatch):
    """参数为 patch；正文新 ID 与完整来源范围贯穿主张和综合引用，历史来源不重算。"""
    plan, report, _ = _execute(monkeypatch, [_row(0, _METHOD, "The paper uses attention.")])
    ids = [record["evidence_id"] for record in report.evidence_records]
    assert {record["evidence_id"] for record in report.evidence_records if record_citable(record)} == set(ids)
    assert report.results[0].claims[0].supporting_evidence_ids == ids
    payload = synthesize_planner_answer(plan, report.results, project_paper_count=1,
        evidence_records=report.evidence_records, final_answer_draft={
            "contract_version": "research-planner-final-v3", "status": "answered", "unresolved_items": [],
            "blocks": [{"block_id": "b1", "text": "The paper uses attention.", "content_kind": "fact",
                "facet_ids": ["f1"], "section_ids": [],
                "claim_ids": [report.results[0].claims[0].claim_id], "evidence_ids": ids}],
        })
    assert payload["citations"][0]["source_spans"] == report.evidence_records[0]["source_spans"]


def test_ambiguous_section_is_failed_instead_of_empty(monkeypatch):
    """参数为 patch；短路径歧义不默取首节，也不让未执行的读取冒充资料不存在。"""
    rows = [_row(0, _METHOD, "one"), _row(1, "Paper / 5 Method Details", "two")]
    with pytest.raises(EvidenceCapabilityError, match="歧义"):
        _execute(monkeypatch, rows, sections=["method"])


def test_absent_section_is_empty_with_distinct_read_report(monkeypatch):
    """参数为 patch；完成定位查询但无命中可记 empty，并保存未找到的真实原因。"""
    rows = [_row(0, _METHOD, "one")]
    _, report, _ = _execute(monkeypatch, rows, sections=["9.9"])
    assert report.evidence_records == []
    assert report.capability_results == {_SOURCE: "empty"}
    remaining = report.results[0].metadata["read_reports"][0]["remaining"]
    assert {item["reason"] for item in remaining} == {"not_found"}


def test_partial_material_survives_an_ambiguous_remaining_locator(monkeypatch):
    """参数为 patch；已读到明确章节时保留正文，其余歧义单独报告为部分读取。"""
    rows = [_row(0, _METHOD, "one"), _row(1, "Paper / 5 Method Details", "two")]
    _, report, _ = _execute(monkeypatch, rows, sections=[_METHOD, "method"])
    assert report.evidence_records[0]["excerpt"] == "one"
    assert report.results[0].metadata["read_reports"][0]["status"] == "partial"


def test_missing_required_sections_skips_reader(monkeypatch):
    """参数为 patch；计划缺少定位时保留 skipped，未执行不伪造空读取。"""
    _, report, reads = _execute(monkeypatch, [], sections=[])
    assert report.capability_results == {_SOURCE: "skipped"}
    assert reads == []


def test_publication_failure_remains_hard_and_visible(monkeypatch):
    """参数为 patch；正文产物故障保持既有硬失败语义。"""
    with pytest.raises(EvidenceCapabilityError, match="manifest invalid"):
        _execute(monkeypatch, [_row(0, _METHOD, "one")], failure=RuntimeError("manifest invalid"))


def test_missing_owner_or_snapshot_rejects_before_read(monkeypatch):
    """参数为 patch；权限或快照不完整时不得回读当前版本或索引预览。"""
    with pytest.raises(EvidenceCapabilityError):
        _execute(monkeypatch, [_row(0, _METHOD, "one")], project_scope={"paper_ids": ["paper-a"]})


def test_number_boundary_and_full_path_resolution():
    """无参数；章节编号不命中 3.10，完整路径可以消除短路径歧义。"""
    paths = [_METHOD, "Paper / 3.10 Method", "Paper / A.3.1 Appendix"]
    assert resolve_section_paths(paths, ["3.1"])[0].paths == (_METHOD,)
    assert resolve_section_paths(paths, ["method"])[0].status == "ambiguous"
    assert resolve_section_paths(paths, [_METHOD])[0].status == "matched"
