"""读取说明经主张与最终交付入口透传的离线回归；作者：xxx。"""

from __future__ import annotations

from typing import Any
from copy import deepcopy
import json
import socket

import pytest

from app.models.research_planner_contracts import ResearchClaimDraftBatch, ResearchPlan, ResearchUnit, ResearchUnitResult
from app.middleware import trace_context
from app.services.internal_research.research_planner_claim_provider import LLMClaimDraftProvider
from app.services.internal_research.research_planner_final_contracts import FINAL_CONTRACT_VERSION, PlannerFinalAnswerInput
from app.services.internal_research.research_planner_final_provider import LLMPlannerFinalAnswerProvider, prepare_final_input
from app.services.internal_research.research_planner_consumer_budget import CLAIM_PROMPT_VERSION, prepare_consumer_input
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from app.services.internal_research.research_planner_plan_prompt import build_action_prompt
from research_planner_test_utils import make_draft, make_facet, make_unit
from app.services.internal_research.evidence_identity import make_read_evidence_id

_TABLE = "| Method | Score |\n| --- | --- |\n| A | 0.8 |\n| B | 0.9 |"
_REPORT = {
    "operation": "sections", "status": "partial", "resolved": ["Results"],
    "remaining": [{"section_path": "Results", "reason": "atomic_structure_budget", "chunk_id": "v3:b:2"}],
}


@pytest.fixture(autouse=True)
def _isolate_external_services(monkeypatch):
    """参数为替换器；模型、HTTP、数据库和检索意外外联直接失败，无返回值。"""
    import httpx
    import requests
    from app.services import database

    def reject(*_args, **_kwargs):
        """参数为意外外部调用；立即报错以暴露离线替身遗漏，无正常返回。"""
        raise AssertionError("read-report tests must not access external services")

    monkeypatch.setattr(httpx.AsyncClient, "send", reject)
    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(requests.Session, "request", reject)
    monkeypatch.setattr(database, "get_pool", reject)
    monkeypatch.setattr(socket, "create_connection", reject)




def test_planner_prefers_full_section_paths_for_unambiguous_reads() -> None:
    """无参数；规划提示优先复制完整章节路径，短定位只能在唯一匹配时使用。"""
    prompt = build_action_prompt({})
    assert "copy the full section path" in prompt
    assert "only when it uniquely identifies" in prompt
    assert "put one short section word per entry" not in prompt




class _PromptCapture:
    """只记录正式 provider 送出的消息，不连接模型服务。"""

    def __init__(self, response: Any) -> None:
        """参数为预置结构化返回；初始化消息记录列表。"""
        self.response = response
        self.prompts: list[str] = []

    def with_structured_output(self, _schema: Any) -> _PromptCapture:
        """参数为生产模型合同；返回当前隔离的消息记录端口。"""
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        """参数为正式提示消息；保存实际正文并返回预置结果。"""
        self.prompts.append(str(messages[0].content))
        return self.response


def _record() -> dict[str, Any]:
    """无参数；返回含可核验发布范围的表格材料。"""
    record = {
        "paper_id": "paper-a", "chunk_id": "v3:b:0",
        "layer": "direct_text", "usable_as_citation": True, "excerpt": _TABLE,
        "read_contract_version": 1, "material_group_id": "mg-results", "source_location_status": "exact",
        "source_spans": [{"paper_id": "paper-a", "batch_id": "b", "chunk_id": "v3:b:0", "field": "text",
                          "start": 0, "end": len(_TABLE), "text_start": 0, "text_end": len(_TABLE)}],
    }
    record["evidence_id"] = make_read_evidence_id(paper_id="paper-a", material_group_id="mg-results",
                                               source_spans=record["source_spans"], text=_TABLE)
    return record


@pytest.mark.asyncio
async def test_reasoner_delivers_partial_read_report_without_synthetic_evidence() -> None:
    """无参数；reasoner 经真实 provider 将部分读取单独送模，表格和证据身份保持。"""
    capture = _PromptCapture(ResearchClaimDraftBatch(claims=[]))
    reasoner = ResearchUnitReasoner(draft_provider=LLMClaimDraftProvider(llm=capture))
    unit = ResearchUnit.model_validate(make_unit())
    result = await reasoner.reason(unit, [_record()], [], read_reports=[_REPORT])
    assert result.claims == []
    prompt = capture.prompts[0]
    evidence, report = prompt.split("Reading reports", 1)
    assert _TABLE in evidence
    assert "atomic_structure_budget" not in evidence
    assert "atomic_structure_budget" in report and "not evidence" in report
    assert "v3:b:0" in evidence and _record()["evidence_id"] in evidence


@pytest.mark.asyncio
async def test_dependency_and_final_prompts_keep_saved_reading_limits() -> None:
    """无参数；依赖主张及最终生成都保留单元读取报告，不修改历史证据身份。"""
    unit = ResearchUnit.model_validate(make_unit(deps=[{"unit_id": "dependency", "required": True}]))
    dependency = ResearchUnitResult(unit_id="dependency", status="evidence_found", metadata={"read_reports": [_REPORT]})
    claim_capture = _PromptCapture(ResearchClaimDraftBatch(claims=[]))
    prepared = prepare_consumer_input([_record()], [], evidence_max_chars=10000, prompt_version=CLAIM_PROMPT_VERSION)
    await LLMClaimDraftProvider(llm=claim_capture)(unit, [_record()], [dependency], prepared_input=prepared)
    assert "atomic_structure_budget" in claim_capture.prompts[0]
    final_capture = _PromptCapture(json.dumps({"contract_version": FINAL_CONTRACT_VERSION, "blocks": []}))
    payload = _final_material_input(_record(), _REPORT)
    await LLMPlannerFinalAnswerProvider(llm=final_capture).generate(payload)
    prompt = final_capture.prompts[0]
    assert _TABLE in prompt and "atomic_structure_budget" in prompt
    assert _record()["evidence_id"] in prompt and "v3:b:0" in prompt
    assert payload.evidence_records[0]["evidence_id"] == _record()["evidence_id"]


def _final_material_input(record: dict[str, Any], report: dict[str, Any]) -> PlannerFinalAnswerInput:
    """参数为明确取得材料和读取报告；返回一次准备的材料任务输入，不凭空添加主张。"""
    unit = make_unit("u1")
    unit["expected_output"].update(deliverable_kind="materials", required_claim_types=[])
    facet = {**make_facet("f1"), "deliverable_kind": "materials"}
    plan = ResearchPlan.model_validate(make_draft([unit], explicit_facets=[facet]))
    result = ResearchUnitResult(unit_id="u1", status="evidence_found", evidence_ids=[record["evidence_id"]],
                                metadata={"read_reports": [report]})
    return prepare_final_input(PlannerFinalAnswerInput(question="读取实际材料", plan=plan,
                                unit_results=[result], evidence_records=[record]))


@pytest.mark.asyncio
async def test_many_short_source_spans_do_not_displace_body_or_duplicate_report_ranges(monkeypatch) -> None:
    """多来源短正文经正式主张和定稿入口仍完整送模，原始范围保留于保存报告和观察。"""
    from app.config import get_settings
    from app.services.internal_research.evidence_identity import make_read_evidence_id

    monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", 1600)
    part = "实际证据。\n"
    text = part * 100
    spans = [{"paper_id": "paper-a", "batch_id": "a" * 32, "chunk_id": f"v3:{'a' * 32}:{index}",
              "field": "text", "start": 0, "end": len(part),
              "text_start": index * len(part), "text_end": (index + 1) * len(part),
              "source_fragments": [{"source_item": f"item-{index}", "source_start": 0, "source_end": len(part),
                                    "text_start": 0, "text_end": len(part), "page_num": index % 3 + 1}]}
             for index in range(100)]
    record = {**_record(), "excerpt": text, "source_spans": spans}
    record["evidence_id"] = make_read_evidence_id(
        paper_id="paper-a", material_group_id=record["material_group_id"], source_spans=spans, text=text,
    )
    report = {**_REPORT, "read_spans": spans, "used_chars": len(text)}
    original_report = deepcopy(report)
    unit = ResearchUnit.model_validate(make_unit())
    dependency = ResearchUnitResult(unit_id="dependency", status="evidence_found", metadata={"read_reports": [report]})
    claim_capture = _PromptCapture(ResearchClaimDraftBatch(claims=[]))
    final_capture = _PromptCapture(json.dumps({"contract_version": FINAL_CONTRACT_VERSION, "blocks": []}))
    ctx = trace_context.TraceContext(request_id="many-spans")
    token = trace_context.set_trace_ctx(ctx)
    try:
        prepared = prepare_consumer_input([record], [], evidence_max_chars=1600, prompt_version=CLAIM_PROMPT_VERSION)
        await LLMClaimDraftProvider(llm=claim_capture)(unit, [record], [], read_reports=[report], prepared_input=prepared)
        await LLMPlannerFinalAnswerProvider(llm=final_capture).generate(_final_material_input(record, report))
    finally:
        trace_context.reset_trace_ctx(token)
    for prompt in [claim_capture.prompts[0], final_capture.prompts[0]]:
        assert text in prompt
        assert "atomic_structure_budget" in prompt
        assert "read_spans" not in prompt and "source_fragments" not in prompt
        assert "range_count=100" in prompt and "pages=[1, 2, 3]" in prompt
    events = [json.loads(row) for row in ctx.evidence_observations]
    sources = [row["payload"] for row in events if row["stage"] == "model_evidence_text"]
    assert len(sources) == 2
    assert all(source["source_text"] == text and source["source_spans"] == spans for source in sources)
    assert report == original_report and dependency.metadata["read_reports"] == [original_report]
