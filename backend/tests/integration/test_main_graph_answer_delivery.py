"""【研究交付】【主图贯通】真实主图、RAG、流式与保存边界的隔离回归。作者：xxx。"""

from copy import deepcopy
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import agent
from app.api.facades import graph_facade
from app.api.routes import chat_api
from app.api.schemas.research import CrossPaperQARequest
from app.models.research_action_contracts import ResearchDecisionOutput
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from app.nodes import rag_node
from app.services import upload_read_service
from app.services import internal_research_service
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.llm_service import StructuredOutputError
from app.services.result_delivery_service import deliver_result, read_delivery_response
from o2_evidence_test_utils import direct_evidence, read_result
from result_delivery_test_utils import Connection, repository
from result_delivery_test_utils import offline_memory_callback  # noqa: F401
from .test_incremental_research import ScriptedModel, state


class DeliveryModel(ScriptedModel):
    """【外部模型替身】成功形成原文结论，或两次读取后抛出真实结构化错误类型。"""

    def __init__(self, records, *, fail_after_reads):
        """参数为固定材料与失败位置；不访问任何真实模型。"""
        super().__init__(records)
        self.fail_after_reads = fail_after_reads
        self.analysis_messages = []

    def with_structured_output(self, schema):
        """参数为正式模型合同；第三次决策模拟读取后的结构错误，其他阶段使用明确响应。"""
        if self.fail_after_reads and schema is ResearchDecisionOutput and len(self.steps) == 2:
            async def fail(messages):
                raise StructuredOutputError("structured_output_failed", "decision.execute.research_intent: extra_forbidden",
                                            user_summary="研究动作字段不符合契约", details={"repair_count": 1})
            return SimpleNamespace(ainvoke=fail)
        structured = super().with_structured_output(schema)
        if schema is not ResearchClaimStrategyDraftBatch:
            return structured

        async def observe_analysis(messages):
            """参数为执行器交给分析器的真实上下文；保留原始正文后调用明确离线模型。"""
            self.analysis_messages.append(deepcopy(messages))
            return await structured.ainvoke(messages)

        return SimpleNamespace(ainvoke=observe_analysis)

    async def ainvoke(self, messages):
        """参数为最终生成上下文；保留 Unicode 和前后空白以验证正文未被重建。"""
        response = await super().ainvoke(messages)
        raw = json.loads(response.content)
        raw["blocks"][0]["text"] = "  😀 The paper uses attention.\n\n"
        return SimpleNamespace(content=json.dumps(raw, ensure_ascii=False))


def install_workflow(monkeypatch, *, fail_after_reads):
    """参数为外部端口替换器与失败位置；返回实际编译主图和子图输出观察。"""
    records = [direct_evidence("first"), direct_evidence("second", text=(
        "The paper uses attention. Additional details identify the second section."))]
    model = DeliveryModel(records, fail_after_reads=fail_after_reads)
    observed = dict(reads=[], rag=[], model=model, records=records)

    async def initialize(current):
        return {}

    async def first_analysis(current, llm):
        return {"turn_depth": "research", "turn_depth_assessment": {"depth": "research", "requires_paper_evidence": True}}

    async def adapter(unit, context):
        observed["reads"].append(unit.unit_id)
        return read_result([records[len(observed["reads"]) - 1]])

    async def resolve(*args):
        return None

    async def trace_writeback(current):
        observed["rag"].append(deepcopy(current))
        return {}

    monkeypatch.setattr(pipeline, "_first_analysis_delta", first_analysis)
    monkeypatch.setattr(rag_node, "rag_init_node", initialize)
    monkeypatch.setattr(rag_node, "write_back_node", trace_writeback)
    monkeypatch.setattr(rag_node, "_llm", lambda: model)
    monkeypatch.setattr(rag_node, "_planner_adapters", lambda current: {"local_direct_text": adapter})
    monkeypatch.setattr(rag_node, "_planner_optional_adapters", lambda current: {})
    monkeypatch.setattr(rag_node, "_planner_read_request", resolve)
    monkeypatch.setattr(internal_research_service.paper_db, "get_paper", lambda paper_id: {"id": paper_id, "title": "Offline paper"})
    monkeypatch.setattr(graph_facade.observability, "record_timed_event", lambda *args, **kwargs: None)
    workflow = agent.build_agent_workflow().compile(checkpointer=InMemorySaver())
    return workflow, observed


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_after_reads", [False, True])
async def test_main_graph_stream_and_save_keep_same_answer(monkeypatch, fail_after_reads):
    """参数为成功或读后失败；真实主图结果经事件流与正式事务保存保持同一正文和状态。"""
    workflow, observed = install_workflow(monkeypatch, fail_after_reads=fail_after_reads)
    connection = Connection()
    events = []

    async def shared_agent():
        return workflow

    async def append_event(**kwargs):
        events.append(deepcopy(kwargs["payload"]))

    async def save(**kwargs):
        return await deliver_result(**kwargs, repository=repository(connection))

    @asynccontextmanager
    async def publication_lifetime(scope):
        yield scope

    monkeypatch.setattr(graph_facade, "get_shared_agent", shared_agent)
    monkeypatch.setattr(upload_read_service, "publication_scope_lifetime", publication_lifetime)
    monkeypatch.setattr(chat_api, "begin_stream_trace_context", lambda **kwargs: None)
    monkeypatch.setattr(chat_api.rag_run_service, "append_event", append_event)
    monkeypatch.setattr(chat_api, "deliver_result", save)
    extra = {**state(), "session_id": "session", "conversation_id": "conversation", "thread_id": "answer-thread", "intent": "rag_qa"}
    context = chat_api._CrossPaperContext("conversation", "session", "project", ["paper-a"], "", "research",
                                         connection.data["runs"]["target"], graph_extra=extra)
    saved = await chat_api._execute_cross_paper_stream_run(
        req=CrossPaperQARequest(query="first second", session_id="session"), ctx=context, user_id="user")
    expected = observed["rag"][-1]
    snapshot = await workflow.aget_state(graph_facade._agent_config("session", "conversation", "answer-thread"))
    assert snapshot.values["answer"] == snapshot.values["report"] == expected["answer"]
    assert observed["reads"] == ["u1", "u2"]
    assert_analysis_consumes_returned_materials(observed, expected, fail_after_reads=fail_after_reads)
    assert "".join(event["content"] for event in events if event.get("type") == "token") == expected["answer"]
    reopened = read_delivery_response(connection.data["runs"]["target"])
    assert saved["persistence_status"] == "saved"
    assert saved["message_patch"]["content"] == reopened["answer"] == expected["answer"]
    for key in ("final_delivery_report", "final_citation_bindings", "final_generation_attempt"):
        assert saved["message_patch"][key] == reopened[key] == expected[key]
    status = "failed" if fail_after_reads else "complete"
    assert reopened["final_delivery_report"]["status"] == status
    # 1. 【研究交付】【状态区分】保存操作完成不把失败研究报告改写为成功
    assert connection.data["runs"]["target"]["status"] == "completed"
    assert reopened.get("error_code") == expected.get("error_code")
    assert chat_api._completed_qa_outcome(reopened) == ("planner_failed" if fail_after_reads else "answered")
    if fail_after_reads:
        from app.services.internal_research.research_memory_writeback import _blocking_terminal_label
        assert _blocking_terminal_label(reopened) == "final_delivery_failed"
    assert bool(reopened["final_citation_bindings"]) is (not fail_after_reads)
    await assert_next_question_clears_answer(workflow)


def assert_analysis_consumes_returned_materials(observed, result, *, fail_after_reads):
    """参数为主图观察、结果和失败位置；分析器须消费两轮原文，结构失败不能伪造分析。"""
    model = observed["model"]
    if fail_after_reads:
        assert model.analysis_messages == []
        assert len(result["research_unit_results"]) == 2
        return
    assert len(model.analysis_messages) == 1
    context = "\n".join(message.content for message in model.analysis_messages[0])
    for record in observed["records"]:
        assert record["evidence_id"] in context
        assert record["excerpt"] in context
    dependencies = result["research_plan"]["units"][2]["dependencies"]
    assert [dependency["unit_id"] for dependency in dependencies] == observed["reads"]
    assert all(dependency["required_product"] == "materials" and dependency["required_claim_types"] == []
               for dependency in dependencies)
    assert len(result["research_unit_results"]) == 3
    assert result["research_unit_results"][2]["claims"][0]["claim_type"] == "paper_supported"
    assert len(result["final_citation_bindings"]) == 2
    assert model.steps == []


async def assert_next_question_clears_answer(workflow):
    """参数为已有答案的同线程主图；新问题短路时也不能保留前轮答案。"""
    new_state, _, _ = graph_facade._build_agent_state("", {"session_id": "session", "conversation_id": "conversation",
                                                           "thread_id": "answer-thread"})
    assert "answer" in new_state and new_state["answer"] is None
    result = await workflow.ainvoke(new_state, config=graph_facade._agent_config("session", "conversation", "answer-thread"))
    assert result["answer"] is None
    assert result["guardrails_action"] == "reject"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["resume", "retry"])
async def test_resume_input_does_not_apply_new_question_answer_reset(kind):
    """参数为恢复方式；新问题默认值不能进入恢复命令覆盖已保存正文。"""
    saved = {"answer": "已保存正文", "report": "已保存正文"}
    updates = []

    async def get_state(config):
        return SimpleNamespace(values=saved, interrupts=({"question": "继续吗"},) if kind == "resume" else ())

    async def update_state(config, update):
        updates.append(update)

    current, session_id, conversation_id = graph_facade._build_agent_state("继续", {"session_id": "session"})
    graph_input = await graph_facade._graph_input(agent=SimpleNamespace(aget_state=get_state, aupdate_state=update_state),
        user_query="继续", state=current, session_id=session_id, conversation_id=conversation_id,
        thread_id=current["thread_id"], execution_kind=kind)
    if kind == "resume":
        assert graph_input.resume == "继续"
        assert graph_input.update is None or "answer" not in graph_input.update
    else:
        assert graph_input is None
    assert all("answer" not in update for update in updates)
    assert saved["answer"] == "已保存正文"
