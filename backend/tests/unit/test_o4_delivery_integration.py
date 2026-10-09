"""O4 真实出口、记忆准入和跨层交付回归；作者：xxx。"""

from types import SimpleNamespace
from typing import get_type_hints
import json
import socket

import pytest

from app.models.models import AgentState, RAGState
from app.services.internal_research.research_planner_final_contracts import FinalDeliveryReport
from app.services.internal_research.research_memory_writeback import build_turn_outcome, write_back_turn_memory
from app.services.rag_trace_service import build_turn_level_fields
from app.services.internal_research import research_planner_pipeline as pipeline


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """参数为测试替换器；任何意外真实网络连接均显式失败。"""
    def reject(*_args, **_kwargs):
        """参数不使用；阻止模型、数据库和检索实际外联。"""
        raise AssertionError("O4 离线测试禁止真实网络连接")

    # 1. Windows asyncio 的 socketpair 使用回环连接，外联守卫放在真实客户端和数据库边界
    import httpx
    import requests
    from app.services import database

    monkeypatch.setattr(httpx.AsyncClient, "send", reject)
    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(requests.Session, "request", reject)
    monkeypatch.setattr(database, "get_pool", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


class ForbiddenModel:
    """任何意外模型调用都直接使离线测试失败。"""

    def with_structured_output(self, *_args, **_kwargs):
        """参数不使用；禁止被拒绝回合触发候选模型。"""
        raise AssertionError("不允许外部模型调用")


def delivery_state(status: str, *, evidence_not_found: bool = False) -> dict:
    """参数为真实交付状态；返回有正文和已有主张的测试状态。"""
    report = FinalDeliveryReport(attempt_id="attempt-1", entrypoint="normal", status=status,
                                 generation_status="completed", evidence_not_found=evidence_not_found)
    return {"turn_depth": "research", "runtime_chain": "planner", "answer": "失败也可以保留的正文 😀",
            "planner_loop_decision": "continue", "source_status": {"planner": "ok"},
            "final_delivery_applicability": "research", "final_delivery_report": report.model_dump(mode="json"),
            "final_generation_attempt": {"attempt_id": "attempt-1"}, "final_citation_bindings": []}


@pytest.mark.parametrize("status,turn", [("complete", "completed"), ("partial", "incomplete"),
                                         ("insufficient", "incomplete"), ("failed", "failed")])
def test_turn_outcome_consumes_final_delivery_without_changing_evidence(status, turn):
    """参数为交付四态；回合按报告映射，取证能力不被反向改写。"""
    state = delivery_state(status)
    outcome = build_turn_outcome(state, memory_writeback={"status": "skipped"})
    assert outcome["status"] == turn
    assert state["source_status"]["planner"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "insufficient"])
async def test_failure_body_does_not_enter_success_memory(status):
    """参数为失败/不足；正式记忆入口必须在模型调用前拒绝。"""
    result = await write_back_turn_memory(delivery_state(status), request_id="req-1", llm=ForbiddenModel())
    assert result["status"] == "skipped"
    assert any(f"final_delivery_{status}" in reason for reason in result["reason_codes"])


@pytest.mark.asyncio
async def test_missing_report_becomes_failure_before_committed_memory_admission(monkeypatch):
    """无参数；正式收尾把缺失报告记为失败，候选写回仍等待答案真实提交。"""
    from app.services.internal_research import research_memory_writeback as memory

    state = delivery_state("complete")
    del state["final_delivery_report"]
    async def writeback(current, **_kwargs):
        """参数为待写回状态；尚未提交答案时调用模型或候选写回即失败。"""
        raise AssertionError("memory admission must wait for the result commit")

    monkeypatch.setattr(memory, "write_back_turn_memory", writeback)
    result = await build_turn_level_fields(state)
    assert result["final_delivery_report"]["status"] == "failed"
    assert result["final_delivery_report"]["reasons"][0]["code"] == "missing_delivery_report"
    assert result["memory_writeback"]["reason_codes"] == ["awaiting_result_commit"]
    assert result["turn_outcome"]["status"] == "failed"


@pytest.mark.asyncio
async def test_research_report_is_checked_even_when_depth_is_missing():
    """无参数；深度缺失不使已进入研究的出口漏掉缺报告合同错误。"""
    result = await build_turn_level_fields({"final_delivery_applicability": "research"})
    assert result["final_delivery_report"]["status"] == "failed"
    assert result["source_status"]["planner_final"] == "failed"
    assert "turn_outcome" not in result
    assert "memory_writeback" not in result


@pytest.mark.asyncio
async def test_execution_failure_and_pre_execution_failure_have_distinct_applicability(monkeypatch):
    """无参数；两个正式入口的失败清楚区分是否已进入研究执行。"""
    failed = await pipeline.execute_planner_state({}, llm=ForbiddenModel(), settings=SimpleNamespace(), adapters={})
    assert failed["final_delivery_applicability"] == "pre_execution_failure"
    assert failed.get("final_delivery_report") is None

    async def fail_analysis(*_args):
        """模拟首次上下文读取硬失败，不调用外部服务。"""
        raise RuntimeError("离线上下文失败")

    monkeypatch.setattr(pipeline, "_first_analysis_delta", fail_analysis)
    early = await pipeline.analyze_planner_state({}, ForbiddenModel(), SimpleNamespace())
    assert early["final_delivery_applicability"] == "pre_execution_failure"
    assert "final_delivery_report" not in early


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["authorization", "adapter", "inference"])
async def test_execution_applicability_comes_from_actual_work_boundary(boundary):
    """参数为请求拒绝、真实读取故障或纯推理故障；正式执行入口按实际工作事实决定交付适用性。"""
    from research_planner_test_utils import make_draft, make_unit
    from o2_evidence_test_utils import direct_evidence, read_result, scope

    unit = make_unit("u1", output_role="evidence")
    units = [unit]
    if boundary == "inference":
        unit["expected_output"].update(deliverable_kind="materials", required_claim_types=[])
        unit["claim_policy"].update(allowed_claim_types=[], citation_required=False)
        unit = make_unit("u2", source_type="inference_from_units", output_role="synthesis",
                         allowed_claims=["model_prior_hypothesis"], operation_family="creative_analysis",
                         deps=[{"unit_id": "u1", "required_product": "materials", "required": True}])
        unit["claim_policy"]["citation_required"] = False
        unit["evidence_policy"]["target_paper_ids"] = []
        units.append(unit)
    state = {"research_plan": {**make_draft(units), "contract_version": "research-planner-v4"},
             "run_id": "offline-run", "turn_depth": "research",
             "project_scope": scope()}
    calls = []

    async def resolver(_unit, _context):
        """参数为正式读取请求；授权失败发生于读取和模型调用前。"""
        if boundary == "authorization":
            raise PermissionError("离线请求无读取权限")
        return None

    async def adapter(_unit, _context):
        """参数为已获准的请求；记录真实适配器调用后抛出基础设施故障。"""
        if boundary == "inference":
            return read_result([direct_evidence()])
        calls.append("adapter")
        raise RuntimeError("离线适配器执行失败")

    class FailedModel:
        """纯推理允许进入模型边界，实际模型调用失败不得误记为未执行。"""

        def with_structured_output(self, *_args, **_kwargs):
            """参数为真实主张合同；返回隔离的失败模型自身。"""
            return self

        async def ainvoke(self, _messages):
            """参数为实际推理输入；记录模型调用并显式失败。"""
            calls.append("inference")
            raise RuntimeError("离线纯推理执行失败")

    result = await pipeline.execute_planner_state(state, llm=FailedModel(), settings=SimpleNamespace(),
        adapters={"local_direct_text": adapter}, request_resolver=resolver)
    if boundary == "authorization":
        assert calls == []
        assert result["final_delivery_applicability"] == "pre_execution_failure"
        assert result.get("final_delivery_report") is None
    else:
        assert calls == [boundary], result.get("error_summary")
        assert result["final_delivery_applicability"] == "research"
        assert result["final_delivery_report"]["status"] == "failed"
        assert result["final_delivery_report"]["generation_status"] == "not_called"


@pytest.mark.asyncio
async def test_previous_research_applicability_survives_later_input_failure():
    """无参数；同运行已进入研究后再次校验失败，仍按研究执行失败交付，不回退为执行前。"""
    result = await pipeline.execute_planner_state({"final_delivery_applicability": "research"},
        llm=ForbiddenModel(), settings=SimpleNamespace(), adapters={})
    assert result["final_delivery_applicability"] == "research"
    assert result["final_delivery_report"]["status"] == "failed"


def test_delivery_fields_survive_both_graph_state_contracts():
    """无参数；实际主图与子图都声明交付载体，避免 LangGraph 静默丢键。"""
    expected = {"final_delivery_report", "final_citation_bindings", "final_generation_attempt", "final_delivery_applicability"}
    for state in (AgentState, RAGState):
        assert expected <= get_type_hints(state).keys()


def test_api_and_stream_keep_exact_answer_and_delivery():
    """无参数；实际聊天投影与流式完成载体保持同一次正文及交付四态。"""
    from app.api.facades.graph_facade import _cross_paper_done_payload
    from app.api.routes.chat_api import _completed_qa_outcome

    expected = {"complete": "answered", "partial": "partial_answer", "insufficient": "insufficient_answer", "failed": "planner_failed"}
    for status, outcome in expected.items():
        state = {**delivery_state(status), "answer": "  😀 正文\r\n"}
        response = _cross_paper_done_payload(state, "session-1")
        assert response["answer"] == state["answer"]
        assert response["final_delivery_report"] == state["final_delivery_report"]
        assert _completed_qa_outcome(state) == outcome
    assert _completed_qa_outcome(delivery_state("insufficient", evidence_not_found=True)) == "evidence_not_found"


@pytest.mark.asyncio
async def test_honest_termination_still_delivers_available_materials():
    """无参数；正式综合入口在诚实收尾动作下仍调用最终模型组织合法材料。"""
    from research_planner_test_utils import make_draft, make_unit

    draft = make_draft([make_unit("u1", output_role="evidence")])
    draft["contract_version"] = "research-planner-v4"
    draft["units"][0]["expected_output"].update(deliverable_kind="materials", required_claim_types=[], required_answer_sections=[])
    draft["units"][0]["claim_policy"].update(allowed_claim_types=[], citation_required=False)
    draft["coverage_review"]["explicit_facets"][0].update(deliverable_kind="materials", required_paper_ids=["paper-a"])
    record = {"evidence_id": "ev-material", "paper_id": "paper-a", "layer": "direct_text",
              "usable_as_citation": True, "excerpt": "保存原文片段", "chunk_id": "old-chunk"}

    class Model:
        """捕获正式 provider 的离线模型调用。"""
        calls = 0

        async def ainvoke(self, _messages):
            """参数为真实提示；返回材料正文载体。"""
            self.calls += 1
            return SimpleNamespace(content=json.dumps({"contract_version": "research-planner-final-draft-v4", "blocks": [
                {"text": "已整理原文材料", "product_kind": "materials", "facet_ids": ["f1"],
                 "section_ids": [], "evidence_ids": ["ev-material"]}]}))

    model = Model()
    result = await pipeline.synthesize_planner_state({"research_plan": draft,
        "research_unit_results": [{"unit_id": "u1", "status": "evidence_found", "evidence_ids": ["ev-material"]}],
        "evidence_records": [record], "planner_loop_decision": "terminate_honest",
        "final_delivery_applicability": "research"}, model)
    assert model.calls == 1
    assert "已整理原文材料" in result["answer"]
    assert result["final_delivery_report"]["entrypoint"] == "research_honest"
    assert result["final_delivery_report"]["status"] in {"complete", "partial"}


@pytest.mark.asyncio
async def test_real_chat_route_persists_original_markdown_and_unclipped_sources(monkeypatch):
    """无参数；正式聊天路由的回答、来源与消息元数据使用同份交付，完整外部替身隔离。"""
    from app.api.routes import chat_api
    from app.api.schemas.research import CrossPaperQARequest

    state = {**delivery_state("failed"), "answer": "  😀 本次正文\r\n",
             "evidence_records": [{"evidence_id": "ev1", "expanded_text": "文" * 5000}], "citations": []}
    from result_delivery_test_utils import Connection, InlineExecutor, repository
    from app.services.result_delivery_service import deliver_result
    connection = Connection({"run_id": "route-run", "user_id": "user-1", "conversation_id": "conv-1"})
    saved = []

    async def context(**_kwargs):
        """参数为请求身份；提供已隔离的真实路由上下文形状。"""
        return chat_api._CrossPaperContext(conversation_id="conv-1", session_id="session-1", project_id="project-1",
            scoped_ids=["paper-1"], filter_expr="", dialogue_mode="research", run=None)

    async def graph(*_args):
        """参数为正式图输入；返回本次产物替身，避免真实模型或检索。"""
        return state, "session-1"

    async def persist(**kwargs):
        """参数为正式消息写入；保存可断言的同次字段。"""
        result = await deliver_result(**kwargs, repository=repository(connection))
        saved.append(result["message_patch"])
        return result

    async def create_operation(**_kwargs):
        """参数为新操作；返回真实保存仓储已知的测试运行。"""
        return connection.data["runs"]["route-run"], True

    monkeypatch.setattr(chat_api.rag_run_service, "create_operation", create_operation)
    monkeypatch.setattr(chat_api, "get_rag_run_executor_registry", lambda: InlineExecutor())
    monkeypatch.setattr(chat_api, "_build_cross_paper_context", context)
    monkeypatch.setattr(chat_api, "invoke_graph", graph)
    monkeypatch.setattr(chat_api, "deliver_result", persist)
    monkeypatch.setattr(chat_api, "get_settings", lambda: SimpleNamespace(rag_expose_unified_contract=True, graph_timeout_seconds=30))
    response = await chat_api.cross_paper_qa(CrossPaperQARequest(query="查看来源"), user={"user_id": "user-1"})
    assert response["answer"] == state["answer"] == saved[0]["content"]
    assert len(response["evidence_records"][0]["expanded_text"]) == 5000
    assert saved[0]["final_generation_attempt"] == state["final_generation_attempt"]
    assert saved[0]["final_delivery_report"]["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_graph_entry_clears_new_question_and_preserves_same_run_resume(monkeypatch, resume):
    """参数为新问题或同运行恢复；正式 facade 入口清理旧交付，恢复仅刷新授权范围。"""
    from app.api.facades import graph_facade
    from app.services import guardrails_trace_service
    from langgraph.types import Command

    previous = {**delivery_state("partial"), "run_id": "old-run", "research_plan": {"question": "旧问题"},
                "planner_final_candidate": {"attempt_id": "old-attempt", "answer": "尚未提交的正文"}}
    captured = []

    class Graph:
        """只替换图计算，真实输入组装与恢复分支均运行。"""

        async def aget_state(self, _config):
            """参数为线程配置；返回旧 checkpoint 或待恢复 checkpoint。"""
            return SimpleNamespace(values=previous, interrupts=("ask",) if resume else ())

        async def ainvoke(self, graph_input, **_kwargs):
            """参数为正式图输入；记录清理/恢复载荷，不调用实际模型或数据库。"""
            captured.append(graph_input)
            return {"report": "离线结果"}

    async def graph():
        """无参数；返回隔离图。"""
        return Graph()

    async def no_trace(result, **_kwargs):
        """参数为图结果；隔离外部观测持久化。"""
        return result

    monkeypatch.setattr(graph_facade, "get_shared_agent", graph)
    monkeypatch.setattr(graph_facade, "_resolve_session_id", lambda *_args: "session-1")
    monkeypatch.setattr(graph_facade, "resolve_thread_id", lambda *_args, **_kwargs: "thread-1")
    monkeypatch.setattr(graph_facade, "_agent_config", lambda *_args: {})
    monkeypatch.setattr(graph_facade.observability, "record_timed_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guardrails_trace_service, "attach_short_circuit_trace", no_trace)
    monkeypatch.setattr(graph_facade, "get_settings", lambda: SimpleNamespace(graph_timeout_seconds=1))
    await graph_facade.invoke_graph("新的回答或澄清", {"session_id": "session-1"},
                                   execution_kind="resume" if resume else "new")
    current = captured[0]
    if resume:
        assert isinstance(current, Command)
        assert current.resume == "新的回答或澄清"
        assert current.update is None
        assert previous["final_delivery_report"]["status"] == "partial"
        assert previous["planner_final_candidate"] == {"attempt_id": "old-attempt", "answer": "尚未提交的正文"}
    else:
        assert current["run_id"] != previous["run_id"]
        assert current["final_delivery_report"] is None
        assert current["final_generation_attempt"] is None
        assert current["final_citation_bindings"] == []
        assert current["planner_final_blocks"] == []
        assert current["planner_final_candidate"] == {}
        assert current["turn_outcome"] is None
