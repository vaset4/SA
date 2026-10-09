"""【模型合同】【公共调用方】通过真实节点入口核对严格响应与错误传播；作者：xxx。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models.structured_outputs import AnswerEvaluation, CitedAnswer, PaperAnalysisOutput
from app.nodes import document_grading, paper_analyzer, query_parser, sufficiency_grader
from app.services import llm_service, paper_metadata_service, research_memory_curator_service
from app.services.ingest import upload_metadata_service
from app.services.internal_research import research_memory_writeback


class ModelClient:
    """替换 HTTP，仍执行正式消息转换和严格 Pydantic 解析。"""

    def __init__(self, payload):
        """参数为本地固定模型输出；保存实际请求消息以核对合同。"""
        self.content = json.dumps(payload, ensure_ascii=False)
        self.messages = []

    def invoke(self, messages, **_kwargs):
        """参数为正式请求消息；返回可控 JSON，不执行外部 IO。"""
        self.messages.append(messages)
        return llm_service.LLMResponse(content=self.content, raw={})

    def with_structured_output(self, schema):
        """参数为调用方真实模型；返回正式公共适配器。"""
        return llm_service.StructuredOutputLLM(self, schema)


@pytest.fixture(autouse=True)
def zero_repair(monkeypatch):
    """参数为替身管理器；关闭原生格式且不回修，使文本合同和首次错误可直接核对。"""
    monkeypatch.setattr(llm_service, "_structured_output_native_enabled", lambda: False)
    monkeypatch.setattr(llm_service, "_structured_output_repair_max", lambda: 0)


def test_query_entry_preserves_filters_and_rejects_string_year(monkeypatch):
    """查询解析遵守类型，错误年份不能被转换或吞成无过滤查询。"""
    monkeypatch.setattr(query_parser, "_heuristic_parse", lambda _: None)
    valid = ModelClient({"year_min": 2020, "clean_query": "attention"})
    monkeypatch.setattr(query_parser, "get_llm", lambda: valid)
    assert asyncio.run(query_parser.parse_query("search")).year_min == 2020
    assert sum(message["content"].count('"title": "QueryFilters"') for message in valid.messages[0]) == 1
    invalid = ModelClient({"year_min": "2020", "clean_query": "attention"})
    monkeypatch.setattr(query_parser, "get_llm", lambda: invalid)
    with pytest.raises(llm_service.StructuredOutputError):
        asyncio.run(query_parser.parse_query("search"))
    for _query, example in query_parser.QUERY_FILTER_EXAMPLES:
        assert query_parser.QueryFilters.model_validate_json(example.model_dump_json(), strict=True) == example


def test_sufficiency_error_is_not_reported_as_evidence_insufficiency(monkeypatch):
    """充分性旧字符串布尔直接失败，不能伪装成一次已完成的证据判定。"""
    monkeypatch.setattr(sufficiency_grader, "get_settings", lambda: SimpleNamespace(rag_sufficiency_grader_enabled=True))
    monkeypatch.setattr(sufficiency_grader, "get_llm", lambda: ModelClient({"sufficient": "false", "reason": "missing"}))
    with pytest.raises(llm_service.StructuredOutputError):
        asyncio.run(sufficiency_grader.sufficiency_grader_node({"query": "question"}))


def test_document_grade_entry_uses_schema_and_rejects_alias(monkeypatch):
    """相关性分类正常结果不变，未约定别名不能被归一化。"""
    monkeypatch.setattr(document_grading, "_settings", lambda: SimpleNamespace(document_grading_enabled=True))
    monkeypatch.setattr(document_grading, "get_llm", lambda: ModelClient({"binary_score": "relevant", "reasoning": "same topic"}))
    relevant, irrelevant = asyncio.run(document_grading.grade_documents("question", [{"text": "paper"}]))
    assert relevant[0]["grade"] == "relevant" and irrelevant == []
    monkeypatch.setattr(document_grading, "get_llm", lambda: ModelClient({"binary_score": "yes", "reasoning": "same topic"}))
    with pytest.raises(llm_service.StructuredOutputError):
        asyncio.run(document_grading.grade_documents("question", [{"text": "paper"}]))


def test_paper_analysis_uses_model_contract_and_keeps_original_text():
    """论文分析的正文空白和算法围栏保留，分析入口的合同与输出模型一致。"""
    method = ' \n```python\nprint("中文")\n```\n '
    payload = {"problem": "p", "method": method, "contribution": ["c"], "limitation": "l", "keywords": ["k"],
               "related_work": [], "tldr": "summary", "methodology_type": "experimental", "methodology_detail": "details"}
    client = ModelClient(payload)
    result = paper_analyzer._invoke_analysis("question", llm_client=client)
    assert result["method"] == method
    contract = json.dumps(PaperAnalysisOutput.model_json_schema(), ensure_ascii=False, indent=2)
    assert contract in client.messages[0][0]["content"]


def test_metadata_raw_entry_keeps_zero_repair_and_strict_lists():
    """上传独立补全仍只发一次；作者字符串不能被转换成列表或伪装成功。"""
    missing = {"title": "", "authors": [], "published_date": "", "abstract": ""}
    client = ModelClient({"title": "title", "authors": ["A"], "published_date": "2025-01-01", "abstract": "abstract"})
    result = upload_metadata_service.complete_metadata("document", missing, client=client)
    assert result["title"] == "title" and len(client.messages) == 1
    assert '"additionalProperties": false' in client.messages[0]
    invalid = ModelClient({"authors": "A"})
    with pytest.raises(ValidationError):
        upload_metadata_service.complete_metadata("document", missing, client=invalid)
    assert len(invalid.messages) == 1


def test_category_structure_failure_does_not_become_empty_category(monkeypatch):
    """分类字段违约可见，不能把非法响应报告为补全成功但分类为空。"""
    monkeypatch.setattr(llm_service, "get_llm", lambda: ModelClient({"category": 1}))
    with pytest.raises(llm_service.StructuredOutputError):
        paper_metadata_service._infer_category("document", {})


def test_memory_curator_preserves_quote_and_body(monkeypatch):
    """候选整理保留原句和正文空白，后续精确来源核验继续使用原字符。"""
    quote = " \n原文\n "
    payload = {"candidates": [{"title": "title", "body": quote, "memory_type": "decision_note",
                               "evidence": [{"paper_id": "paper", "quote": quote}]}]}
    client = ModelClient(payload)
    monkeypatch.setattr(research_memory_curator_service, "_curator_llm", lambda: client)
    result = asyncio.run(research_memory_curator_service._invoke_curator_llm({"papers": []}))
    assert result.candidates[0].body == quote and result.candidates[0].evidence[0].quote == quote


def test_memory_writeback_preserves_body_and_strict_policy_flags():
    """写回通过正式记忆复核包装，字符串政策标记不能变成有效许可。"""
    proposal = {"source_claim_id": "c", "title": "title", "body": " body\n ", "memory_type": "decision_note",
                "durable_project_value": True}
    result = asyncio.run(research_memory_writeback._propose_candidates({}, claims=[], llm=ModelClient({"proposals": [proposal]})))
    assert result.proposals[0].body == " body\n "
    with pytest.raises(llm_service.StructuredOutputError):
        asyncio.run(research_memory_writeback._propose_candidates({}, claims=[], llm=ModelClient({
            "proposals": [{**proposal, "durable_project_value": "true"}],
        })))


@pytest.mark.parametrize(("schema", "payload"), [
    (CitedAnswer, {"answer": "answer", "confidence": 0.9, "confidence_reasoning": "reason"}),
    (CitedAnswer, {"answer": "answer", "confidence": "high", "confidence_reasoning": "reason",
                   "citations": [{"paper_id": "paper", "page_num": "1"}]}),
    (AnswerEvaluation, {"relevance": "high", "faithfulness": "high", "completeness": "complete",
                        "quality_score": "90%"}),
])
def test_retained_public_contracts_do_not_coerce_bad_fields(schema, payload):
    """保留的引用回答和评分模型拒绝旧分数、枚举和页码强转格式。"""
    with pytest.raises(ValidationError):
        llm_service.parse_structured_response(json.dumps(payload), schema)
