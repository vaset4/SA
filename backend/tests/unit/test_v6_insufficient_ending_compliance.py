"""【研究交付】【不足收尾】真实图核对项目范围、可读说明和四态边界。作者：xxx。"""

from __future__ import annotations

from dataclasses import dataclass, field
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import get_settings
from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport
from app.models.research_contracts import TurnDepthAssessment
from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.models.research_planner_loop_contracts import PlannerProbeDecision
from app.nodes import rag_node
from app.services.internal_research.research_planner_synthesis import synthesize_planner_answer, synthesize_planner_answer_async
from research_planner_strategy_test_utils import strategy_fixture, validate_strategy_fixture
from research_planner_test_utils import make_draft, make_unit

_USER_QUESTION = "这些论文怎么用注意力机制"
_DRAFT_QUESTION = "How do the papers use attention?"
_DRAFT_GOAL = "Verify the attention evidence"
_PROJECT_PAPER_IDS = ("paper-a", "paper-b", "paper-c")


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch):
    """参数为替身；在客户端边界禁止模型、数据库和检索外联，不拦Windows内部管道。"""
    import httpx
    import requests
    from app.services import database

    monkeypatch.setattr(httpx.AsyncClient, "send", _reject_network)
    monkeypatch.setattr(httpx.Client, "send", _reject_network)
    monkeypatch.setattr(requests.Session, "request", _reject_network)
    monkeypatch.setattr(database, "get_pool", _reject_network)
    monkeypatch.setattr(socket, "create_connection", _reject_network)


def _reject_network(*args, **kwargs):
    """参数为意外网络连接；立即失败以暴露隔离遗漏，无正常返回。"""
    raise AssertionError("insufficient-ending tests must not connect to external services")


def _draft(paper_ids=()) -> dict[str, Any]:
    """参数为草案目标论文；返回正常研究要求，项目数量仍来自实际授权范围。"""
    unit = make_unit("lookup", goal=_DRAFT_GOAL)
    unit["evidence_policy"]["target_paper_ids"] = list(paper_ids)
    draft = {**make_draft([unit]), "question": _DRAFT_QUESTION}
    draft["coverage_review"]["explicit_facets"][0]["user_text_span"] = _USER_QUESTION
    return draft


@dataclass
class _StructuredCall:
    """【研究交付】【模型替身】返回明确预置结构化结果。"""

    value: Any

    async def ainvoke(self, _messages):
        """参数为真实提示；返回指定模型边界结果，不改动研究流程。"""
        return self.value


@dataclass
class _PlannerLLM:
    """【研究交付】【模型隔离】无业务输入时最终生成一旦被调用立即失败。"""

    text_calls: int = 0
    paper_ids: tuple[str, ...] = ()

    def with_structured_output(self, schema):
        """参数为正式模型合同；仅提供规划、探测及空主张三种明确替身。"""
        values = {ResearchPlanStrategyDraft: ResearchPlanStrategyDraft.model_validate(strategy_fixture(_draft(self.paper_ids[:1]))),
                  ResearchClaimStrategyDraftBatch: ResearchClaimStrategyDraftBatch(claims=[]),
                  PlannerProbeDecision: PlannerProbeDecision(needs_probe=False)}
        if schema not in values:
            raise AssertionError(f"unexpected model schema: {schema}")
        return _StructuredCall(values[schema])

    async def ainvoke(self, _messages):
        """参数为最终提示；无合法输入时不应调用，调用事实保留后直接报错。"""
        self.text_calls += 1
        raise AssertionError("no business input is available for final generation")


@dataclass
class _WriteBackProbe:
    """【研究交付】【写回边界】隔离外部保存并记录写回前的正式交付事实。"""

    seen: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, state):
        """参数为实际图终态；保存观察副本并返回空更新，不访问数据库。"""
        self.seen.append(dict(state))
        return {}


async def _empty_adapter(_unit, _context):
    """参数为正式读取申请；返回合法空读取，不把模型失败伪装为无证据。"""
    return EvidenceReadResult((), KnowledgeReadReport(operation="chunks", status="complete"))


async def _index_probe(_ids):
    """参数为实际项目范围；隔离向量库探测，本例没有可检索内容。"""
    return False


async def _run_graph(monkeypatch, paper_ids):
    """参数为替身及真实项目论文范围；从正式子图入口运行至写回并返回状态与调用事实。"""
    llm, writeback = _PlannerLLM(paper_ids=paper_ids), _WriteBackProbe()
    monkeypatch.setattr(rag_node, "rag_init_node", lambda _state: {})
    monkeypatch.setattr(rag_node, "write_back_node", writeback)
    monkeypatch.setattr(rag_node, "_llm", lambda: llm)
    monkeypatch.setattr(rag_node, "_settings", get_settings)
    monkeypatch.setattr(rag_node, "_planner_adapters", lambda _state: {"local_direct_text": _empty_adapter})
    monkeypatch.setattr(rag_node, "_planner_optional_adapters", lambda _state: {})
    monkeypatch.setattr(rag_node, "_scoped_index_probe", _index_probe)
    monkeypatch.setattr(rag_node, "_resolve_paper_titles", lambda _ids: {})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: SimpleNamespace())
    batches = {paper_id: f"{index:032x}" for index, paper_id in enumerate(paper_ids, 1)}
    project_scope = {"user_id": "offline-user", "paper_ids": list(paper_ids),
                     "local_retrieval_allowed": True, "publication": {"user_id": "offline-user",
                     "lease_id": "offline-lease", "batch_ids": list(batches.values()), "paper_batches": batches}}
    state = {"run_id": "offline-ending", "user_query": _USER_QUESTION,
             "structured_query": {"paper_ids": list(paper_ids), "project_scope": project_scope},
             "project_scope": project_scope,
             "turn_depth": "research",
             "turn_depth_assessment": TurnDepthAssessment(depth="research", reason_code="cross_paper_comparison",
                requires_paper_evidence=True, resolved_query=_USER_QUESTION, needs_project_memory=False)}
    result = await rag_node.create_rag_node().ainvoke(state)
    return result, llm, writeback


@pytest.mark.asyncio
@pytest.mark.parametrize("paper_ids", [_PROJECT_PAPER_IDS])
async def test_real_graph_keeps_insufficient_delivery_and_scope_based_readable_explanation(monkeypatch, paper_ids):
    """参数为真实项目范围；图上产生不足报告，正文按范围说明下一步且不泄漏内部规划物。"""
    result, llm, writeback = await _run_graph(monkeypatch, paper_ids)
    answer = result["answer"]
    assert result["research_unit_results"], {key: result.get(key) for key in
        ("error_code", "failure_reason", "research_plan", "planner_plan_trace", "source_status_reasons", "answer")}
    assert result["final_delivery_report"]["status"] == "insufficient"
    assert result["source_status"]["planner_final"] == "skipped"
    assert llm.text_calls == 0
    assert writeback.seen[-1]["final_delivery_report"] == result["final_delivery_report"]
    assert answer.strip() and _DRAFT_QUESTION not in answer and _DRAFT_GOAL not in answer
    assert "remaining direct-text gap" not in answer and "validated claims" not in answer
    assert "1 个研究点" not in answer and "阻塞原因" not in answer
    assert "还没有可核对的论文" not in answer
    assert "调整问题" in answer or "补充" in answer
    assert result["research_plan"]["question"] == _USER_QUESTION
    assert result["source_status_reasons"]["planner_final"]


@pytest.mark.asyncio
async def test_empty_scope_cannot_be_mislabeled_as_executed_insufficient_research(monkeypatch):
    """参数为外部替身；空授权范围无法执行正文检索，保留执行前失败且不伪造交付报告。"""
    result, llm, _writeback = await _run_graph(monkeypatch, ())
    assert result["research_unit_results"] == []
    assert result["error_code"] == "planner_execution_before_read_failed"
    assert result.get("final_delivery_report") is None
    assert result["final_delivery_applicability"] == "pre_execution_failure"
    assert result["answer"].strip() and llm.text_calls == 0


@pytest.mark.parametrize("paper_count", [0, len(_PROJECT_PAPER_IDS)])
def test_sync_no_input_keeps_visible_description_without_claim_summary(paper_count):
    """参数为真实论文数；无成果时正文可读、报告不足，不称为已经完成的主张摘要。"""
    plan = validate_strategy_fixture(_draft()).plan
    assert plan is not None
    result = synthesize_planner_answer(plan, [], project_paper_count=paper_count)
    assert result["answer"].strip()
    assert result["final_delivery_report"]["status"] == "insufficient"
    assert "已保存的研究成果" not in result["answer"]
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_both_entries_require_actual_project_paper_count():
    """无参数；两个正式入口漏传项目论文数都必须报错，不能以默认值制造项目事实。"""
    plan = validate_strategy_fixture(_draft()).plan
    assert plan is not None
    with pytest.raises(TypeError):
        synthesize_planner_answer(plan, [])
    with pytest.raises(TypeError):
        await synthesize_planner_answer_async(plan, [])
