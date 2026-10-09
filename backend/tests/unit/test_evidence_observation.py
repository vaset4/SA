"""真实执行观察的隔离、顺序、截断和失败回归。作者：xxx。"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock, Mock

import pytest

from app.config import get_settings
from app.middleware import trace_context as tracing
from app.models.knowledge_access_contracts import (
    KnowledgeMaterial, KnowledgeReadReport, KnowledgeReadRequest, KnowledgeReadResult,
)
from app.models.research_planner_contracts import ResearchPlan, ResearchUnit, ResearchUnitResult
from app.services.internal_research import research_planner_claim_provider as claim_provider
from app.services.internal_research import research_planner_final_provider as final_provider
from app.services.internal_research.research_planner_final_contracts import FINAL_CONTRACT_VERSION, PlannerFinalAnswerInput
from app.services.llm_service import OpenAICompatibleLLM
from evals.evidence.live_capture import CapturedRunFailure, capture_live_run
from research_planner_test_utils import make_draft, make_facet, make_unit, supporting_spans

_THREAD_COUNT = 24
# 消费侧正文总额度：主张证据与定稿摘录两处共用，两段用例按它算边界
_BUDGET_LIMIT = 8000
_FINAL_TEXT_CHARS = 700
_FINAL_COUNT = 12


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """参数为补丁工具；封锁未替换的网络和数据库边界，并断言无意外外联。"""
    blocked = Mock(side_effect=AssertionError("观察回归禁止真实外联"))
    for target in ("httpx.Client.send", "httpx.AsyncClient.send", "requests.sessions.Session.request",
                   "app.services.database.get_pool", "socket.create_connection"):
        monkeypatch.setattr(target, blocked)
    yield
    assert blocked.call_count == 0


@pytest.fixture
def trace() -> Iterator[tracing.TraceContext]:
    """无参数；提供独立请求 trace，测试结束恢复此前上下文。"""
    ctx = tracing.TraceContext(request_id="request-test")
    token = tracing.set_trace_ctx(ctx)
    try:
        yield ctx
    finally:
        tracing.reset_trace_ctx(token)


def _events(ctx: tracing.TraceContext, stage: str) -> list[dict[str, Any]]:
    """参数为 trace 与阶段；返回独立解码的该阶段事件。"""
    return [event for item in ctx.evidence_observations if (event := json.loads(item))["stage"] == stage]


def _evidence(evidence_id: str = "ev-1", *, text: str = "actual source") -> dict[str, Any]:
    """参数为证据 ID 和正文；返回明确的流程测试直接文本证据。"""
    return {
        "evidence_id": evidence_id, "paper_id": "paper-a", "chunk_id": evidence_id,
        "parent_id": evidence_id, "layer": "direct_text", "source": "milvus",
        "section": "Method", "expanded_text": text, "excerpt": text,
        "usable_as_citation": True, "verification_status": "verified",
    }


def _range_evidence(text: str) -> dict[str, Any]:
    """参数为正文；返回带两页实际范围承诺的新版证据，不改变旧保存证据测试形状。"""
    from app.services.internal_research.evidence_identity import make_read_evidence_id

    record = _evidence(text=text)
    middle = len(text) // 2
    spans = [{"paper_id": "paper-a", "batch_id": "fixture-batch", "chunk_id": f"c{page}",
              "field": "full_text", "start": 0, "end": end - start, "text_start": start, "text_end": end,
              "source_fragments": [{"source_item": f"s{page}", "source_start": 0, "source_end": end - start,
                                    "text_start": 0, "text_end": end - start, "page_num": page}]}
             for page, (start, end) in enumerate(((0, middle), (middle, len(text))), start=1)]
    record.update(read_contract_version=1, material_group_id="fixture-material-group", source_spans=spans,
                  source_location_status="exact")
    record["evidence_id"] = make_read_evidence_id(
        paper_id="paper-a", material_group_id=record["material_group_id"], source_spans=spans, text=text,
    )
    return record


def _unit() -> ResearchUnit:
    """无参数；返回复用现有测试构造器的有效研究单元。"""
    return ResearchUnit.model_validate(make_unit("unit-a"))


def _thread_capture(label: str) -> str:
    """参数为线程任务标签；记录真实传播的请求上下文并返回本次 scope ID。"""
    with tracing.evidence_scope("retrieval_query") as scope:
        assert scope is not None
        tracing.record_evidence_observation("thread_hit", {"label": label})
        ctx = tracing.get_trace_ctx()
        assert ctx is not None
        assert ctx.begin_call().evidence_scope_id == scope.scope_id
        return scope.scope_id


def test_scope_failure_restores_parent_and_duplicate_request_id_isolated(trace: tracing.TraceContext) -> None:
    """参数为请求 trace；子 scope 异常后恢复父身份，同名新请求不能继承关联。"""
    failure = ValueError("retrieval failed")
    with tracing.evidence_scope("planner_execute") as parent:
        assert parent is not None
        with pytest.raises(ValueError) as caught:
            with tracing.evidence_scope("planner_unit", unit_id="unit-a") as unit:
                raise failure
        assert caught.value is failure
        assert trace.begin_call().evidence_scope_id == parent.scope_id
        other = tracing.TraceContext(request_id=trace.request_id)
        token = tracing.set_trace_ctx(other)
        try:
            assert other.begin_call().evidence_scope_id is None
            with tracing.evidence_scope("planner_execute") as independent:
                assert independent is not None and independent.parent_scope_id is None
        finally:
            tracing.reset_trace_ctx(token)
    assert trace.begin_call().evidence_scope_id is None
    failed = _events(trace, "scope_failed")[0]
    assert failed["scope"]["scope_id"] == unit.scope_id
    assert failed["scope"]["parent_scope_id"] == parent.scope_id
    assert failed["payload"] == {"error_type": "ValueError", "error": "retrieval failed"}


async def _parallel_request(request_id: str) -> tracing.TraceContext:
    """参数为请求 ID；在独立 async 请求和工作线程内采集，结束后返回该 trace。"""
    ctx = tracing.TraceContext(request_id=request_id)
    token = tracing.set_trace_ctx(ctx)
    try:
        with tracing.evidence_scope("planner_unit", unit_id=request_id):
            await asyncio.to_thread(_thread_capture, request_id)
        return ctx
    finally:
        tracing.reset_trace_ctx(token)


@pytest.mark.asyncio
async def test_async_requests_and_to_thread_isolate_observations(trace: tracing.TraceContext) -> None:
    """参数为上层 trace；验证并发请求和线程传播不会交换记录或覆盖上层上下文。"""
    contexts = await asyncio.gather(_parallel_request("request-a"), _parallel_request("request-b"))
    assert tracing.get_trace_ctx() is trace
    assert trace.evidence_observations == ()
    for ctx in contexts:
        rows = _events(ctx, "thread_hit")
        assert len(rows) == 1
        assert rows[0]["request_id"] == rows[0]["payload"]["label"] == ctx.request_id
        assert rows[0]["scope"]["unit_id"] == ctx.request_id
        starts = _events(ctx, "scope_start")
        assert starts[1]["scope"]["parent_scope_id"] == starts[0]["scope"]["scope_id"]


@pytest.mark.asyncio
async def test_concurrent_thread_records_and_call_ids_are_not_lost(trace: tracing.TraceContext) -> None:
    """参数为共享请求 trace；并发线程追加必须保留全部事件和唯一客户端调用 ID。"""
    scopes = await asyncio.gather(*(asyncio.to_thread(_thread_capture, str(i)) for i in range(_THREAD_COUNT)))
    assert len(set(scopes)) == _THREAD_COUNT
    assert len(_events(trace, "thread_hit")) == _THREAD_COUNT
    assert len(_events(trace, "scope_completed")) == _THREAD_COUNT
    assert len({call.call_id for call in trace.model_calls}) == _THREAD_COUNT


def test_capture_preserves_actual_order_missing_fields_and_immutable_text(trace: tracing.TraceContext) -> None:
    """参数为 trace；重复命中保留原位，字段缺失不补造，任意 metadata 不进入受控正文。"""
    hits: list[dict[str, Any]] = [
        {"chunk_id": "same", "display_text": "first full body", "score": 0.2, "metadata": {"truth": object()}},
        {"chunk_id": "same", "preview": "second preview"},
    ]
    tracing.record_retrieval_hits("query_raw", hits)
    hits[0]["display_text"] = "changed after capture"
    rows = _events(trace, "query_raw")[0]["payload"]["hits"]
    assert rows == [
        {"rank": 1, "chunk_id": "same", "display_text": "first full body", "score": 0.2},
        {"rank": 2, "chunk_id": "same", "preview": "second preview"},
    ]
    assert trace.evidence_capture_errors == ()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), object()])
def test_serialization_failure_has_error_instead_of_empty_success(trace: tracing.TraceContext, bad_value: Any) -> None:
    """参数为 trace 和非法值；非有限值或不可序列化正文不能伪造成成功空命中。"""
    tracing.record_retrieval_hits("query_raw", [{"score": bad_value}])
    assert _events(trace, "query_raw") == []
    assert json.loads(trace.evidence_capture_errors[0])["stage"] == "query_raw"
    tracing.record_retrieval_hits("query_filtered", [])
    assert _events(trace, "query_filtered")[0]["payload"]["hits"] == []


def test_no_trace_does_not_create_request_or_coerce_invalid_payload() -> None:
    """无参数；普通无 trace 调用不创建伪造请求，也不为观察检查改变业务行为。"""
    assert tracing.get_trace_ctx() is None
    with tracing.evidence_scope("planner_execute") as scope:
        tracing.record_retrieval_hits("query_raw", [{"score": object()}])
    assert scope is None
    assert tracing.get_trace_ctx() is None


def _retrieval_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ResearchUnit, Any]:
    """参数为补丁工具；返回真实检索编排所需状态、单元与明确的向量存储替身。"""
    from app.nodes import rag_node

    raw = [
        {"paper_id": "paper-b", "chunk_id": "excluded", "preview": "excluded", "score": 0.99},
        {"paper_id": "paper-a", "chunk_id": "c1", "parent_id": "p1", "section": "Discussion",
         "preview": "discussion", "display_text": "full discussion", "score": 0.8},
        {"paper_id": "paper-a", "chunk_id": "c2", "parent_id": "p2", "section": "Method",
         "preview": "method mechanism", "display_text": "full method mechanism", "score": 0.6},
    ]
    second = [raw[2], {**raw[2], "chunk_id": "c3", "score": 0.5}]
    store = SimpleNamespace(hybrid_search=Mock(side_effect=deepcopy([raw, second, raw, second])))
    monkeypatch.setattr(rag_node, "get_vector_store", Mock(return_value=store))
    monkeypatch.setattr(rag_node, "_settings", Mock(return_value=SimpleNamespace(rag_top_k=8)))
    monkeypatch.setattr(rag_node.paper_db, "get_paper", Mock(return_value={"paper_id": "paper-a"}))
    monkeypatch.setattr(rag_node, "read_knowledge", _published_read_fixture)
    scope = {"user_id": "fixture-user", "local_retrieval_allowed": True,
             "paper_ids": ["paper-a", "paper-b"], "filter_expr": 'paper_id in ["paper-a", "paper-b"]',
             "publication": {"user_id": "fixture-user", "lease_id": "fixture-lease", "batch_ids": ["fixture-batch"],
                             "paper_batches": {"paper-a": "fixture-batch"}}}
    state = {"project_scope": scope, "structured_query": {"project_scope": scope}, "filter_expr": "additional-filter"}
    unit = _unit()
    unit = unit.model_copy(update={"evidence_policy": unit.evidence_policy.model_copy(update={"locator_queries": ["first", "second"]})})
    return state, unit, store


async def _published_read_fixture(request: KnowledgeReadRequest) -> KnowledgeReadResult:
    """参数为真实适配器读取请求；只替代发布正文边界，返回故意不同于索引预览的材料。"""
    assert request.scope["publication"]["lease_id"] == "fixture-lease"
    assert request.paper_ids == ("paper-a",) and request.granularity == "child"
    assert request.chunk_ids == ("c2", "c3", "c1")
    materials = []
    for parent, chunk_ids in (("p2", ("c2", "c3")), ("p1", ("c1",))):
        parts, spans, offset = [], [], 0
        for chunk_id in chunk_ids:
            text = f"published source body for {chunk_id}"
            if parts:
                offset += 2
            spans.append({"paper_id": "paper-a", "batch_id": "fixture-batch", "chunk_id": chunk_id,
                          "field": "text", "start": 0, "end": len(text),
                          "text_start": offset, "text_end": offset + len(text)})
            parts.append(text)
            offset += len(text)
        materials.append(KnowledgeMaterial(
            paper_id="paper-a", batch_id="fixture-batch", group_kind="parent", group_locator=parent,
            material_group_id=f"fixture-group-{parent}", text="\n\n".join(parts), source_spans=tuple(spans),
            chunk_id=chunk_ids[0], parent_id=parent, section_path="Method" if parent == "p2" else "Discussion",
            section="Method" if parent == "p2" else "Discussion", page_num=0,
            source_location_status="legacy_chunk", block_type="text",
        ))
    report = KnowledgeReadReport(operation="chunks", status="complete", requested=request.chunk_ids,
                                 read_spans=tuple(span for material in materials for span in material.source_spans))
    return KnowledgeReadResult(tuple(materials), report)


@pytest.mark.asyncio
async def test_real_retrieval_stages_keep_raw_filter_merge_and_rank_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身边界；观察前后检索参数与结果相同，过滤及去重前排名各自可核对。"""
    from app.nodes import rag_node

    state, unit, store = _retrieval_fixture(monkeypatch)
    baseline = await rag_node._planner_direct_text_records(state, unit, None)
    ctx = tracing.TraceContext(request_id="retrieval-request")
    token = tracing.set_trace_ctx(ctx)
    try:
        with tracing.evidence_scope("planner_unit", unit_id=unit.unit_id):
            observed = await rag_node._planner_direct_text_records(state, unit, None)
    finally:
        tracing.reset_trace_ctx(token)
    assert observed == baseline
    assert store.hybrid_search.call_args_list[:2] == store.hybrid_search.call_args_list[2:]
    raw = _events(ctx, "query_raw")
    filtered = _events(ctx, "query_filtered")
    assert [row["chunk_id"] for row in raw[0]["payload"]["hits"]] == ["excluded", "c1", "c2"]
    assert [row["rank"] for row in raw[0]["payload"]["hits"]] == [1, 2, 3]
    assert [row["chunk_id"] for row in filtered[0]["payload"]["hits"]] == ["c1", "c2"]
    assert [row["rank"] for row in filtered[0]["payload"]["hits"]] == [1, 2]
    assert raw[0]["scope"]["scope_id"] != raw[1]["scope"]["scope_id"]
    assert raw[0]["scope"]["parent_scope_id"] == raw[1]["scope"]["parent_scope_id"]
    assert [row["chunk_id"] for row in _events(ctx, "merged_hits")[0]["payload"]["hits"]] == ["c1", "c2", "c3"]
    assert [row["chunk_id"] for row in _events(ctx, "ranked_hits")[0]["payload"]["hits"]] == ["c2", "c3", "c1"]
    assert raw[0]["payload"]["hits"][2]["display_text"] == "full method mechanism"
    assert raw[0]["payload"]["retrieval_backend"] is None
    # 单元目标是 paper-a：真实查询条件在 top-k 之前就收窄到它，raw 记录的是收窄后的范围
    assert raw[0]["payload"]["scope"]["paper_ids"] == ["paper-a"]
    assert 'paper_id in ["paper-a"]' in raw[0]["payload"]["scope"]["filter_expr"]
    assert filtered[0]["payload"]["scope"]["paper_ids"] == ["paper-a"]
    assert filtered[0]["payload"]["scope"]["business_allowed_paper_ids"] == ["paper-a"]
    assert all(record["excerpt"].startswith("published source body") for record in observed.records)
    assert "published source body" not in json.dumps(raw)
    assert _events(ctx, "adapter_evidence_pool")[0]["payload"]["records"] == list(baseline.records)


def test_query_failure_and_capture_failure_keep_original_business_semantics(
    trace: tracing.TraceContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数为 trace 与存储替身；采集坏分数不改业务返回，实际检索异常保留同一实例。"""
    from app.nodes import rag_node

    state, _, store = _retrieval_fixture(monkeypatch)
    hits = [{"paper_id": "paper-a", "chunk_id": "c1", "score": float("nan")}]
    store.hybrid_search.side_effect = None
    store.hybrid_search.return_value = hits
    result = rag_node._search_chunks(store, "query", 5, structured_query=state["structured_query"])
    assert result[0] is hits[0]
    assert len(trace.evidence_capture_errors) == 2
    error = RuntimeError("actual retrieval failure")
    store.hybrid_search.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        rag_node._search_chunks(store, "query", 5, structured_query=state["structured_query"])
    assert caught.value is error
    assert _events(trace, "scope_failed")[-1]["payload"]["error"] == str(error)


def test_filtered_scope_projection_never_expands_authorized_papers(
    trace: tracing.TraceContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数为 trace 和检索替身；业务目标含未授权论文时，观察中的允许范围只能取交集。"""
    from app.nodes import rag_node

    state, _, store = _retrieval_fixture(monkeypatch)
    structured = {**state["structured_query"], "paper_ids": ["paper-a", "outside-authorization"]}
    result = rag_node._search_chunks(store, "query", 5, structured_query=structured)
    assert [hit["chunk_id"] for hit in result] == ["c1", "c2"]
    scope = _events(trace, "query_filtered")[0]["payload"]["scope"]
    assert scope["paper_ids"] == ["paper-a"]
    assert scope["business_allowed_paper_ids"] == ["outside-authorization", "paper-a"]


def test_provider_source_uses_actual_class_identity() -> None:
    """无参数；真实 Milvus 类型与其测试子类必须分别标为已知来源和未知来源。"""
    from app.nodes.rag_node import _retrieval_observation_source
    from app.services.milvus_store import MilvusStore

    original = object.__new__(MilvusStore)
    substitute = type("FixtureMilvus", (MilvusStore,), {})
    assert _retrieval_observation_source(original)["retrieval_backend"] == "milvus"
    assert _retrieval_observation_source(object.__new__(substitute))["retrieval_backend"] is None


class _LiveGraph:
    """只用于请求边界测试的图替身，显式产出或抛出真实测试执行结果。"""

    def __init__(self, *, error: Exception | None = None, invalid_call: bool = False) -> None:
        """参数为预设错误和采集异常开关；初始化收到的执行参数记录。"""
        self.error = error
        self.invalid_call = invalid_call
        self.inputs: list[dict[str, Any]] = []

    async def ainvoke(self, live_input: dict[str, Any]) -> dict[str, Any]:
        """参数为隔离后的业务输入；记录调用事实并返回回答或原始错误。"""
        self.inputs.append(deepcopy(live_input))
        with tracing.evidence_scope("claim_context", unit_id="unit-a"):
            tracing.record_evidence_observation("unit_evidence_pool", {"records": [_evidence()]})
            if self.invalid_call:
                ctx = tracing.get_trace_ctx()
                assert ctx is not None
                ctx.begin_call().input_messages = [{"role": "user", "content": object()}]
            if self.error is not None:
                raise self.error
        live_input["paper_ids"].append("mutated-by-graph")
        return {"answer": "actual answer"}


@pytest.mark.asyncio
async def test_live_capture_preserves_input_and_runtime_index_declaration(trace: tracing.TraceContext) -> None:
    """参数为上层 trace；索引声明在运行前采集，业务输入隔离且运行后恢复上层 trace。"""
    graph = _LiveGraph()
    live_input = {"user_query": "question", "paper_ids": ["paper-a"]}
    result = await capture_live_run(graph, live_input, index_snapshot_id="index-v1")
    assert result["answer"] == "actual answer"
    assert graph.inputs == [live_input] == [{"user_query": "question", "paper_ids": ["paper-a"]}]
    declaration = result["evidence_observations"][0]
    assert declaration["stage"] == "index_snapshot_declared"
    assert declaration["payload"] == {"index_snapshot_id": "index-v1", "source": "eval_declared_before_execution"}
    result["evidence_observations"].clear()
    assert tracing.get_trace_ctx() is trace and trace.evidence_observations == ()


@pytest.mark.asyncio
async def test_live_failure_retains_business_cause_when_model_export_is_invalid(trace: tracing.TraceContext) -> None:
    """参数为上层 trace；采集序列化失败不能替换原始业务错误或丢掉已有证据。"""
    original = LookupError("original graph failure")
    graph = _LiveGraph(error=original, invalid_call=True)
    with pytest.raises(CapturedRunFailure) as caught:
        await capture_live_run(graph, {"user_query": "question", "paper_ids": ["paper-a"]})
    assert caught.value.__cause__ is original
    capture = caught.value.capture
    assert capture.error_type == "LookupError" and capture.error_message == str(original)
    observation = json.loads(capture.observation_snapshot)
    assert any(event["stage"] == "unit_evidence_pool" for event in observation["evidence_observations"])
    error = observation["evidence_capture_errors"][0]
    assert error["stage"] == "model_call_observations" and error["call_id"]
    assert "answer" not in observation and tracing.get_trace_ctx() is trace


@pytest.mark.asyncio
async def test_live_capture_rejects_nested_truth_before_execution() -> None:
    """无参数；证据真值不能借由嵌套执行参数进入实际图。"""
    graph = _LiveGraph()
    with pytest.raises(ValueError, match="evidence_benchmark"):
        await capture_live_run(graph, {"user_query": "question", "options": {"evidence_benchmark": {}}})
    assert graph.inputs == []


def _client(monkeypatch: pytest.MonkeyPatch, contents: list[str]) -> tuple[OpenAICompatibleLLM, Mock]:
    """参数为补丁工具和 HTTP 返回正文；构造真实客户端，仅替换缓存及网络边界。"""
    cache = Mock()
    cache.get_llm_response.return_value = None
    monkeypatch.setattr("app.services.llm_service.get_cache", Mock(return_value=cache))
    responses = [Mock(json=Mock(return_value={"choices": [{"message": {"content": value}}]})) for value in contents]
    post = Mock(side_effect=responses)
    monkeypatch.setattr("app.services.llm_service.requests.post", post)
    client = OpenAICompatibleLLM(
        model="fixture-model", api_key="fixture-key", base_url="https://example.invalid/v1",
        temperature=0.0, timeout=1, request_exception_max_retries=0, empty_response_max_retries=0,
    )
    return client, post


@pytest.mark.asyncio
@pytest.mark.parametrize("full_text", ["a" * 20000, "🙂e\u0301\n " * 4000], ids=["ascii", "unicode_whitespace"])
async def test_claim_context_matches_real_client_messages_and_budget_boundary(
    trace: tracing.TraceContext, monkeypatch: pytest.MonkeyPatch, full_text: str,
) -> None:
    """参数为 trace 和 HTTP 替身；真实客户端仅调用一次，超总额度的尾部不进输入并写明省略。"""
    monkeypatch.setattr(get_settings(), "rag_planner_consumer_text_max_chars", _BUDGET_LIMIT)
    records = [_range_evidence(full_text)]
    unit = _unit()
    expected_prompt = claim_provider.build_claim_prompt(unit, records, [])
    assert tracing.export_evidence_observations(trace)["model_call_observations"] == []
    assert _events(trace, "model_evidence_text")[0]["scope"] is None
    client, post = _client(monkeypatch, ['{"claims": []}'])
    result = await claim_provider.LLMClaimDraftProvider(llm=client)(unit, records, [])
    assert result == [] and post.call_count == 1
    sent = post.call_args.kwargs["json"]["messages"]
    user_messages = [message for message in sent if message["role"] == "user"]
    assert user_messages == [{"role": "user", "content": expected_prompt}]
    exported = tracing.export_evidence_observations(trace)
    call = exported["model_call_observations"][0]
    assert call["input_messages"] == sent and call["error"] is None
    event = _events(trace, "model_evidence_text")[-1]
    assert event["scope"]["scope_id"] == call["evidence_scope_id"]
    # 可追溯正文就是展示段省略号前的那截：总额度扣掉元信息行之后的余量
    prefix = expected_prompt.split("Evidence records:\n", 1)[1].split("text: ", 1)[0] + "text: "
    expected_visible = full_text[: _BUDGET_LIMIT - len(prefix)].rstrip()
    assert event["payload"]["source_text"] == expected_visible
    for key in ("read_contract_version", "material_group_id", "source_spans", "source_location_status"):
        assert event["payload"][key] == records[0][key]
    rendered = event["payload"]["rendered_text"]
    assert rendered.startswith(expected_visible)
    assert "已截断" in rendered and f"完整 {len(full_text)} 字" in rendered
    assert rendered != expected_visible and rendered in user_messages[0]["content"]
    assert f"此处省略 {len(full_text) - len(expected_visible)} 字" in rendered
    assert full_text not in user_messages[0]["content"]
    call["input_messages"][0]["content"] = "changed after export"
    assert trace.model_calls[0].input_messages == sent


@pytest.mark.asyncio
async def test_final_context_applies_budget_with_visible_omission(
    trace: tracing.TraceContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数为 trace 和 HTTP 替身；额度内整条给出，超额度明写省略，不可引用证据不进输入。"""
    hidden = {**_evidence("metadata", text="FILTERED_TEXT"), "layer": "metadata"}
    records = [
        hidden,
        *[_evidence(f"ev-{i}", text="x" * _FINAL_TEXT_CHARS + "TAIL") for i in range(_FINAL_COUNT)],
    ]
    # 1. 首条正常展开，第二条明确超过当前额度，保证配置变化后仍真实覆盖裁剪与整条省略
    records[2] = _evidence("ev-1", text="x" * (_BUDGET_LIMIT * 2) + "TAIL")
    records.append(_evidence("ev-13", text="THIRTEENTH_EXCERPT"))
    unit = make_unit(output_role="evidence")
    unit["expected_output"].update(deliverable_kind="materials", required_claim_types=[])
    facet = {**make_facet("f1"), "deliverable_kind": "materials"}
    plan = ResearchPlan.model_validate(make_draft([unit], explicit_facets=[facet]))
    result = ResearchUnitResult(unit_id=unit["unit_id"], status="evidence_found",
                                evidence_ids=[record["evidence_id"] for record in records])
    payload = final_provider.prepare_final_input(PlannerFinalAnswerInput(
        question="question", plan=plan, unit_results=[result], evidence_records=records,
    ), evidence_max_chars=_BUDGET_LIMIT)
    expected_prompt = final_provider._final_answer_prompt(payload)
    answer = {"contract_version": FINAL_CONTRACT_VERSION, "blocks": [{"block_id": "material",
              "text": "actual final", "content_kind": "materials", "facet_ids": ["f1"], "section_ids": [],
              "claim_ids": [], "evidence_ids": ["ev-0"]}]}
    client, post = _client(monkeypatch, [json.dumps(answer)])
    output = await final_provider.LLMPlannerFinalAnswerProvider(llm=client).generate(payload)
    assert output.payload == answer and post.call_count == 1
    call = tracing.export_evidence_observations(trace)["model_call_observations"][0]
    assert call["input_messages"] == post.call_args.kwargs["json"]["messages"]
    assert call["input_messages"] == [{"role": "user", "content": expected_prompt}]
    # 非可引用证据不冒充正文输入；额度装不下的那条明写省略，未展开的摘录如实交代数量
    assert "FILTERED_TEXT" not in expected_prompt
    assert "已截断" in expected_prompt
    assert "[另有" in expected_prompt
    events = [row for row in _events(trace, "model_evidence_text") if row["scope"] is not None]
    assert events and all(row["scope"]["scope_id"] == call["evidence_scope_id"] for row in events)
    assert all(row["payload"]["rendered_text"] in expected_prompt for row in events)
    whole = [row for row in events if "已截断" not in row["payload"]["rendered_text"]]
    # 额度内整条原样进入观察：可追溯正文与展示文本一致、不含省略说明
    assert whole and all(row["payload"]["source_text"] == row["payload"]["rendered_text"] for row in whole)
    truncated = [row for row in events if "已截断" in row["payload"]["rendered_text"]]
    assert truncated and all(row["payload"]["source_text"] in row["payload"]["rendered_text"] for row in truncated)
    assert all(row["payload"]["source_text"] != row["payload"]["rendered_text"] for row in truncated)


@pytest.mark.asyncio
async def test_structured_output_repair_keeps_distinct_real_calls_in_context(
    trace: tracing.TraceContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数为 trace 和网络替身；结构化回修各次真实消息分别保留，不复制成单次调用。"""
    client, post = _client(monkeypatch, ["not JSON", '{"claims": []}'])
    monkeypatch.setattr("app.services.llm_service._structured_output_repair_max", Mock(return_value=1))
    await claim_provider.LLMClaimDraftProvider(llm=client)(_unit(), [_evidence()], [])
    calls = tracing.export_evidence_observations(trace)["model_call_observations"]
    assert post.call_count == len(calls) == 2
    assert calls[0]["call_id"] != calls[1]["call_id"]
    assert calls[0]["evidence_scope_id"] == calls[1]["evidence_scope_id"]
    assert len(calls[0]["input_messages"]) == 1 and len(calls[1]["input_messages"]) == 2
    assert "one JSON object validating ResearchClaimDraftBatch" in calls[0]["input_messages"][0]["content"]
    assert all(call["input_messages"] == sent.kwargs["json"]["messages"] for call, sent in zip(calls, post.call_args_list))
    assert "上一轮结构化输出未能通过校验" in calls[1]["input_messages"][1]["content"]


@pytest.mark.asyncio
async def test_production_debug_size_cap_does_not_truncate_live_export(
    trace: tracing.TraceContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数为 trace 与存储替身；生产 debug 裁剪后摘要计数仍真实，独立 live 观察保持完整。"""
    tracing.record_evidence_observation("unit_evidence_pool", {"records": [_evidence(text="body" * 100)]})
    call = trace.begin_call()
    call.input_messages = [{"role": "user", "content": "message" * 100}]
    captured = tracing.export_evidence_observations(trace)
    full = tracing.build_full_trace(trace, {})
    repository = SimpleNamespace(write_rag_trace=AsyncMock())
    monkeypatch.setattr("app.repositories.observability_repository.get_observability_repository", Mock(return_value=repository))
    monkeypatch.setattr(tracing, "OBS_TRACE_MAX_BYTES", 1)
    await tracing.safe_write_trace(full, trace.request_id, "session", "question", "user")
    assert full["debug"] == {"truncated": True, "reason": "size_cap"}
    assert full["summary"]["evidence_observation_count"] == 1
    assert full["summary"]["evidence_capture_error_count"] == 0
    assert captured == tracing.export_evidence_observations(trace)
    assert captured["model_call_observations"][0]["input_messages"][0]["content"] == "message" * 100


@pytest.mark.asyncio
async def test_repeated_execute_and_repair_keep_separate_unit_and_pool_lineage(
    trace: tracing.TraceContext, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数为 trace 和调用替身；每个运行经循环批准补证，各执行、查询和重跑保留真实身份。"""
    from app.services.internal_research.research_planner_pipeline import execute_planner_state

    state, unit, adapter, store, records = await _lineage_read_fixture(monkeypatch)
    plan = ResearchPlan.model_validate(make_draft([unit.model_dump(mode="json")]))
    state = {**state, "user_query": plan.question, "research_plan": plan.model_dump(mode="json"),
             "turn_depth": "lookup", "planner_locator_facts": [{"paper_id": "paper-a", "section_path": "Method"}]}
    answer = json.dumps({"claims": [{
        "text": "actual supported claim", "claim_type": "paper_supported",
        "supporting_evidence_ids": [record["evidence_id"] for record in records], "addressed_facet_ids": ["f1"],
        "supporting_spans": [span.model_dump(mode="json") for span in supporting_spans(records)],
    }]})
    client, post = _client(monkeypatch, [answer] * 2)
    kwargs = {"llm": client, "settings": get_settings(), "adapters": {"local_direct_text": adapter},
              "request_resolver": lambda current, _context: current.evidence_policy.model_dump(mode="json")}
    deltas: list[dict[str, Any]] = []
    # 1. 每次研究先报告真实缺口，再把循环已批准的动作与累计成果交给补证回边
    for run_id in ("lineage-run-1", "lineage-run-2"):
        current = {**state, "run_id": run_id}
        first = await execute_planner_state(current, **kwargs)
        assert not first.get("error_code"), first
        assert first["planner_loop_decision"] == "repair_evidence"
        assert first["evidence_records"] == []
        second = await execute_planner_state({**current, **first}, **kwargs)
        assert not second.get("error_code"), second
        assert {record["evidence_id"] for record in second["evidence_records"]} == {record["evidence_id"] for record in records}
        assert second["research_unit_results"][0]["status"] == "evidence_found"
        assert {receipt["identity"]["run_id"] for receipt in second["planner_read_receipts"]} == {run_id}
        deltas.extend((first, second))
    assert adapter.call_count == 4
    # 补证单元只交付材料，每个运行仅在重跑原单元时发送一次推理请求
    assert post.call_count == 2
    # 2. 逻辑读取、底层真实查询和送模调用分别计数，原查询仍保留索引命中的原始形状
    reads = _events(trace, "evidence_read_request")
    assert len(reads) == 4 and all(row["payload"]["actual_call_count"] == 1 for row in reads)
    raw_queries = _events(trace, "query_raw")
    assert len(raw_queries) == store.hybrid_search.call_count
    assert len(raw_queries) == sum(len(call.args[0].evidence_policy.locator_queries) for call in adapter.call_args_list)
    assert "published source body" not in json.dumps(raw_queries)
    assert [row["payload"]["hits"] for row in raw_queries].count([]) == 4
    starts = _events(trace, "scope_start")
    executions = [row for row in starts if row["scope"]["stage"] == "planner_execute"]
    execution_ids = [row["scope"]["scope_id"] for row in executions]
    assert len(execution_ids) == len(set(execution_ids)) == 4
    transients = [row for row in starts if row["payload"]["details"] == {"transient": True}]
    assert len(transients) == 2
    assert {row["scope"]["parent_scope_id"] for row in transients} == set(execution_ids[1::2])
    reruns = [row for row in starts if row["scope"]["stage"] == "repair_rerun"]
    assert len(reruns) == 2 and all(row["scope"]["unit_id"] == "unit-a" for row in reruns)
    assert {row["scope"]["parent_scope_id"] for row in reruns} == set(execution_ids[1::2])
    assert len(_events(trace, "repair_evidence_pool")) == 2
    assert len(_events(trace, "rerun_evidence_pool")) == 2
    pools = _events(trace, "execution_evidence_pool")
    assert len(pools) == 4
    assert [row["payload"]["records"] for row in pools] == [delta["evidence_records"] for delta in deltas]
    assert [row["scope"]["scope_id"] for row in pools] == execution_ids
    assert trace.evidence_capture_errors == ()


async def _lineage_read_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ResearchUnit, Any, Any, Any]:
    """参数为补丁工具；返回真实读取适配器及离线存储，初查为空而定向补查命中固定正文。"""
    from functools import partial

    from app.nodes import rag_node
    from app.services.internal_research.retrieval_coordinator import convert_knowledge_materials

    state, unit, store = _retrieval_fixture(monkeypatch)
    hits = [{"paper_id": "paper-a", "chunk_id": chunk, "parent_id": parent, "section": section,
             "preview": "original index preview", "score": score}
            for chunk, parent, section, score in (("c1", "p1", "Discussion", 0.8),
                                                   ("c2", "p2", "Method", 0.6), ("c3", "p2", "Method", 0.5))]
    store.hybrid_search.side_effect = lambda query, *_args, **_kwargs: [] if query in {"first", "second"} else deepcopy(hits)
    adapter = Mock(wraps=partial(rag_node._planner_direct_text_records, state))
    published = await _published_read_fixture(KnowledgeReadRequest(
        operation="chunks", scope=state["project_scope"], paper_ids=("paper-a",), chunk_ids=("c2", "c3", "c1"),
    ))
    records = [dict(record) for record in convert_knowledge_materials(published.materials)]
    return state, unit, adapter, store, records
