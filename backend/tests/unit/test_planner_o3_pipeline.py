"""O3 正式编排的成果接续、动作准入与失败快照回归。作者：xxx。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from app.api.facades import graph_facade
from app.config import get_settings
from app.middleware.trace_context import TraceContext, build_full_trace, reset_trace_ctx, set_trace_ctx
from app.models.knowledge_access_contracts import EvidenceReadResult, KnowledgeReadReport
from app.models.research_planner_contracts import ResearchClaimDraft, ResearchPlan
from app.nodes import rag_node
from app.services.internal_research import research_planner_pipeline as pipeline
from app.services.internal_research.research_planner_conversion import migrate_execution_checkpoint
from app.services.internal_research.research_planner_observability import build_planner_trace_payload
from research_planner_strategy_test_utils import validate_strategy_fixture
from o2_evidence_test_utils import direct_evidence, scope
from research_planner_test_utils import make_draft, make_facet, make_unit, supporting_spans


def _state() -> dict[str, Any]:
    """无参数；返回有明确运行、用户和固定发布版本的离线图状态。"""
    return {"run_id": "o3-run", "user_query": "Explain A Method", "project_scope": scope(),
            "planner_read_receipts": [], "evidence_records": [],
            "turn_depth": "lookup", "planner_locator_facts": [{"paper_id": "paper-a", "section_path": "Method"}]}


def _plan(*, source: str = "local_direct_text", materials: bool = False, unit_id: str = "u1") -> ResearchPlan:
    """参数为来源、交付种类及单元编号；返回经过正式校验的 A 方法要求。"""
    unit = make_unit(unit_id, source_type=source, allowed_claims=["paper_supported"])
    facet = {**make_facet("f1"), "required_paper_ids": ["paper-a"]}
    if materials:
        unit["expected_output"].update(deliverable_kind="materials", required_claim_types=[])
        unit["claim_policy"] = {"allowed_claim_types": [], "citation_required": False}
        facet["deliverable_kind"] = "materials"
    outcome = validate_strategy_fixture(make_draft([unit], explicit_facets=[facet]))
    assert outcome.plan is not None, outcome.failure
    if unit_id != "u1":
        current = outcome.plan.units[0].model_copy(update={"unit_id": unit_id})
        coverage = outcome.plan.coverage_review.model_copy(update={"unit_facet_map": {unit_id: ["f1"]}})
        return outcome.plan.model_copy(update={"units": [current], "coverage_review": coverage})
    return outcome.plan


def _request_key(unit: Any, _context: Any) -> dict[str, Any]:
    """参数为已授权的离线单元与上下文；返回固定请求身份，替代权限/租约存储端口。"""
    policy = unit.evidence_policy
    return {"paper_ids": list(policy.target_paper_ids), "sections": list(policy.required_sections),
            "queries": list(policy.locator_queries) if policy.source_type == "local_direct_text" else []}


async def _execute(state: dict[str, Any], adapter: Any, *, llm: Any = None) -> dict[str, Any]:
    """参数为当前状态、离线适配器及模型身份；调用正式执行编排并返回增量。"""
    model = llm or SimpleNamespace(model="offline", temperature=0.0, base_url="https://offline.invalid/v1")
    return await pipeline.execute_planner_state(
        state, llm=model, settings=get_settings(),
        adapters={"local_direct_text": adapter, "local_section_text": adapter}, request_resolver=_request_key,
    )




def test_replan_renumber_reuses_material_and_claim_without_false_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身注入；仅改编号的重规划保留材料与有效主张，真实调用数不增加。"""
    calls, drafts = [], []
    record = direct_evidence()

    async def adapter(unit: Any, _context: Any) -> EvidenceReadResult:
        """参数为申请；返回同一版本的真实来源形状并记录调用。"""
        calls.append(unit.unit_id)
        return EvidenceReadResult((record,), KnowledgeReadReport(operation="chunks", status="complete"))

    async def provider(unit: Any, records: Any, _dependencies: Any, **_kwargs: Any) -> list[ResearchClaimDraft]:
        """参数为当前要求与实际材料；返回绑定实际支持片段及 facet 的离线主张。"""
        drafts.append(unit.unit_id)
        return [ResearchClaimDraft(text="A uses attention", claim_type="paper_supported",
                                   supporting_evidence_ids=[records[0]["evidence_id"]],
                                   supporting_spans=supporting_spans(records, [records[0]["evidence_id"]]),
                                   addressed_facet_ids=["f1"])]

    monkeypatch.setattr(pipeline, "LLMClaimDraftProvider", lambda **_kwargs: provider)

    async def scenario() -> None:
        """无参数；执行后仅重绑编号，确认当前产物不残留旧单元关联。"""
        state = {**_state(), "research_plan": _plan().model_dump(mode="json")}
        first = await _execute(state, adapter)
        assert not first.get("error_code"), first
        first_claim = first["research_unit_results"][0]["claims"][0]
        assert first_claim["validation_status"] == "validated"
        state = {**state, **first, "research_plan": _plan(unit_id="renamed").model_dump(mode="json")}
        second = await _execute(state, adapter)
        assert not second.get("error_code"), second
        assert len(calls) == len(drafts) == 1
        assert second["research_unit_results"][0]["unit_id"] == "renamed"
        assert second["research_unit_results"][0]["metadata"]["result_reused"]
        reused_claim = second["research_unit_results"][0]["claims"][0]
        assert reused_claim["claim_id"].startswith("renamed.")
        assert reused_claim["supporting_spans"] == first_claim["supporting_spans"]
        assert second["planner_observation"]["fulfilled_requirement_keys"] == first["planner_observation"]["fulfilled_requirement_keys"]

    asyncio.run(scenario())


def test_execution_failure_keeps_completed_read_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身注入；推理失败必须保留已完成读取，失败状态和空派生产物均可见。"""
    record = direct_evidence()

    async def adapter(_unit: Any, _context: Any) -> EvidenceReadResult:
        """参数为读取上下文；返回已完成材料。"""
        return EvidenceReadResult((record,), KnowledgeReadReport(operation="chunks", status="complete"))

    async def provider(*_args: Any, **_kwargs: Any) -> Any:
        """参数沿用推理入口；在读取之后抛出可定位的真实失败替身。"""
        raise RuntimeError("reasoning failed after successful read")

    monkeypatch.setattr(pipeline, "LLMClaimDraftProvider", lambda **_kwargs: provider)
    state = {**_state(), "research_plan": _plan().model_dump(mode="json")}
    delta = asyncio.run(_execute(state, adapter))
    assert delta["planner_loop_decision"] == "terminate_fail"
    assert delta["evidence_records"] == [record]
    assert delta["planner_read_receipts"][0]["evidence_ids"] == [record["evidence_id"]]
    assert not any(result["claims"] for result in delta["research_unit_results"])


def test_cancelled_execution_snapshot_reaches_full_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为替身注入；取消保留原信号，已完成读取进入真实 full trace 的调试投影。"""
    record = direct_evidence()

    async def scenario() -> None:
        """无参数；在推理等待期间取消，并核对落盘构造器仍可读取完成材料。"""
        entered = asyncio.Event()

        async def adapter(_unit: Any, _context: Any) -> EvidenceReadResult:
            """参数为读取上下文；返回先于取消完成的材料。"""
            return EvidenceReadResult((record,), KnowledgeReadReport(operation="chunks", status="complete"))

        async def provider(*_args: Any, **_kwargs: Any) -> Any:
            """参数沿用推理入口；通知测试已进入推理并等待取消。"""
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(pipeline, "LLMClaimDraftProvider", lambda **_kwargs: provider)
        trace = TraceContext(request_id="http-trace", user_id="u1")
        token = set_trace_ctx(trace)
        try:
            task = asyncio.create_task(_execute({**_state(), "research_plan": _plan().model_dump(mode="json")}, adapter))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            payload = build_full_trace(trace, {"runtime_chain": "planner"})
            events = payload["debug"]["evidence_observations"]
            completed = next(item["payload"] for item in events if item["stage"] == "cancelled_execution_snapshot")
            assert completed["evidence_records"] == [record]
            assert completed["planner_read_receipts"][0]["identity"]["run_id"] == "o3-run"
            assert "lease_id" not in str(completed["planner_read_receipts"])
        finally:
            reset_trace_ctx(token)

    asyncio.run(scenario())








@pytest.mark.parametrize("preplan", [False, True])
def test_legacy_checkpoint_is_archived_and_replanned(preplan: bool) -> None:
    """参数为计划前/计划后中断位置；旧事实只读归档，不升级为新版成功或尝试。"""
    state = {**_state(), "research_unit_results": [{"unit_id": "old", "claims": []}],
             "planner_probe_results": [{"source_type": "local_section_text", "status": "success"}]}
    if not preplan:
        state["research_plan"] = {"question": "legacy", "units": []}
    original = deepcopy(state)
    delta = migrate_execution_checkpoint(state)
    assert state == original
    assert delta["research_unit_results"] == delta["planner_probe_results"] == []
    assert delta["planner_loop_decision"] == "replan"
    assert delta["planner_execution_history"][0]["snapshot"]["research_unit_results"] == original["research_unit_results"]


def test_new_question_creates_new_identity_and_clears_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为会话替身；同一会话新问题拥有新运行，已有 durable 身份按原值透传。"""
    monkeypatch.setattr(graph_facade, "_resolve_session_id", lambda *_args: "session-1")
    monkeypatch.setattr(graph_facade, "resolve_thread_id", lambda *_args, **_kwargs: "thread-1")
    first = graph_facade._build_agent_state("first")[0]
    second = graph_facade._build_agent_state("second")[0]
    assert first["run_id"] != second["run_id"]
    assert second["planner_read_receipts"] == second["research_unit_results"] == []
    assert graph_facade._build_agent_state("third", {"run_id": "durable-run"})[0]["run_id"] == "durable-run"


@pytest.mark.parametrize("streaming", [False, True])
def test_graph_facade_passes_business_run_identity(monkeypatch: pytest.MonkeyPatch, streaming: bool) -> None:
    """参数为图替身及调用模式；同步和流式正式 facade 均把服务端运行身份传入主图。"""
    invocation = AsyncMock(return_value={"answer": "offline answer"})
    stream = Mock(return_value=object())
    graph = SimpleNamespace(aget_state=AsyncMock(return_value=None), ainvoke=invocation, astream=stream)
    monkeypatch.setattr(graph_facade, "get_shared_agent", AsyncMock(return_value=graph))
    monkeypatch.setattr(graph_facade, "_resolve_session_id", lambda *_args: "session-1")
    monkeypatch.setattr(graph_facade, "resolve_thread_id", lambda *_args, **_kwargs: "thread-1")
    monkeypatch.setattr(graph_facade, "_agent_config", lambda *_args: {"configurable": {"thread_id": "thread-1"}})
    monkeypatch.setattr(graph_facade, "_begin_trace_context", lambda **_kwargs: None)
    monkeypatch.setattr(graph_facade.observability, "record_timed_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(graph_facade, "_finalize_graph_result", AsyncMock(return_value={"answer": "offline answer"}))
    extra = {"run_id": "durable-run", "request_id": "http-request", "project_scope": scope()}

    entry = graph_facade.open_graph_stream if streaming else graph_facade._invoke_graph
    asyncio.run(entry("Explain A Method", extra))

    captured = (stream.call_args if streaming else invocation.call_args).args[0]
    assert captured["run_id"] == "durable-run"
    assert captured["request_id"] == "http-request"
    assert captured["project_scope"] == extra["project_scope"]
    assert captured["evidence_records"] == captured["planner_read_receipts"] == []


def test_new_contract_missing_required_fields_fails_before_io() -> None:
    """无参数；畸形新版不能经历史路径被当作可执行成功。"""
    plan = _plan().model_dump(mode="json")
    del plan["coverage_review"]["explicit_facets"][0]["required"]

    async def adapter(*_args: Any) -> Any:
        """参数沿用读取入口；非法合同不得到达此处。"""
        pytest.fail("invalid contract reached adapter")

    delta = asyncio.run(_execute({**_state(), "research_plan": plan}, adapter))
    assert delta["planner_loop_decision"] == "terminate_fail"
    assert "新版研究合同无效" in delta["error_summary"]
    assert delta["source_status"]["local_evidence_execution"] == "not_run"


def test_missing_run_identity_fails_before_evidence_attempt() -> None:
    """无参数；缺少运行身份在首次调用即失败，不能宣称已启动本地检索。"""
    calls = []

    async def adapter(*_args: Any) -> Any:
        """参数沿用读取接口；记录任何不应出现的实际调用。"""
        calls.append("read")
        pytest.fail("missing run reached adapter")

    state = {**_state(), "research_plan": _plan().model_dump(mode="json")}
    del state["run_id"]
    delta = asyncio.run(_execute(state, adapter))
    assert not calls
    assert "local_evidence_execution" not in delta["source_status"]
    assert delta["planner_trace"]["read_facts"]["unknown_reason"] == "missing_execution_identity"
    assert "run_id" in delta["error_summary"]


def test_receipt_is_visible_in_persisted_planner_stage() -> None:
    """无参数；新增凭据必须通过现有 full trace 白名单，不停留在节点内部。"""
    state = {**_state(), "runtime_chain": "planner", "planner_read_receipts": [{"receipt_id": "r1"}]}
    full = build_full_trace(TraceContext(), {"runtime_chain": "planner", **build_planner_trace_payload(state)})
    stage = next(item for item in full["stages"] if item["name"] == "planner")
    assert stage["execution"]["read_receipts"] == [{"receipt_id": "r1"}]


def test_repair_back_edge_only_reads_missing_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为离线配置；比较缺 B 时回边只补 B，A 材料与结果仍保留。"""
    units, facets = [], []
    for index, paper in enumerate(("paper-a", "paper-b"), 1):
        unit = make_unit(f"u{index}", goal=f"Read {paper} method", paper_ids=[paper], facets=[f"f{index}"])
        unit["expected_output"].update(deliverable_kind="materials", required_claim_types=[])
        unit["claim_policy"] = {"allowed_claim_types": [], "citation_required": False}
        units.append(unit)
        facets.append({**make_facet(f"f{index}"), "required_paper_ids": [paper], "deliverable_kind": "materials"})
    validated = validate_strategy_fixture(make_draft(units, explicit_facets=facets))
    plan = validated.plan
    assert plan is not None, validated.failure
    state = _state()
    state["project_scope"]["paper_ids"].append("paper-b")
    state["project_scope"]["publication"]["paper_batches"]["paper-b"] = "b" * 32
    state["research_plan"] = plan.model_dump(mode="json")
    calls = []

    async def adapter(unit: Any, _context: Any) -> EvidenceReadResult:
        """参数为具体取证单元；B 首次为空、补证返回材料，A 始终有可用材料。"""
        paper = unit.evidence_policy.target_paper_ids[0]
        calls.append(paper)
        records = () if paper == "paper-b" and calls.count(paper) == 1 else (direct_evidence(paper, paper_id=paper),)
        return EvidenceReadResult(records, KnowledgeReadReport(operation="chunks", status="complete"))

    async def scenario() -> None:
        """无参数；分别运行首次取证及经过循环准入的补证，核对实际执行范围。"""
        first = await _execute(state, adapter)
        assert not first.get("error_code"), first
        assert first["planner_loop_decision"] == "repair_evidence"
        split = len(calls)
        second = await _execute({**state, **first}, adapter)
        assert not second.get("error_code"), second
        assert calls[split:] and set(calls[split:]) == {"paper-b"}
        assert {record["paper_id"] for record in second["evidence_records"]} == {"paper-a", "paper-b"}
        assert second["evidence_sufficiency_report"]["missing_required_facet_ids"] == []

    asyncio.run(scenario())


def test_production_request_resolver_renews_before_each_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为存储替身；生产请求身份忽略续租令牌，但每次申请仍核对权限和租约。"""
    from app.services import upload_read_service
    renewed = []

    async def renew(snapshot: dict[str, Any]) -> None:
        """参数为本轮快照；记录实际权限核验调用，不连接数据库。"""
        renewed.append(snapshot["lease_id"])

    monkeypatch.setattr(upload_read_service, "get_upload_publication_repository",
                        lambda: SimpleNamespace(renew_snapshot=renew))

    async def scenario() -> None:
        """无参数；同版本续租保持读取身份相同，越界目标在存储前拒绝。"""
        state = _state()
        unit = _plan(source="local_section_text", materials=True).units[0]
        first = await rag_node._planner_read_request(state, unit, None)
        state["project_scope"]["publication"]["lease_id"] = "renewed-lease"
        second = await rag_node._planner_read_request(state, unit, None)
        assert first == second
        assert renewed == ["test-lease", "renewed-lease"]
        bad = unit.model_copy(update={"evidence_policy": unit.evidence_policy.model_copy(update={"target_paper_ids": ["outside"]})})
        with pytest.raises(ValueError, match="授权范围"):
            await rag_node._planner_read_request(state, bad, None)

    asyncio.run(scenario())


def test_validated_baseline_cannot_be_lowered_by_replan() -> None:
    """无参数；已有必需对象在重规划草案中删除时正式校验拒绝替换原计划。"""
    plan = _plan()
    draft = plan.model_dump(mode="json")
    draft["coverage_review"]["explicit_facets"][0]["required_paper_ids"] = []
    state = {**_state(), "research_plan": plan.model_dump(mode="json"), "planner_draft": draft}
    delta = pipeline.validate_planner_state(state, get_settings())
    assert "research_plan" not in delta
    assert "invalid_plan_format" in delta["planner_observation"]["issue_codes"]
    assert plan.coverage_review.explicit_facets[0].required_paper_ids == ["paper-a"]


def test_main_graph_bridges_saved_execution_into_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为图依赖替身；主图恢复进入 RAG 时传递同运行的计划、材料、凭据及循环历史。"""
    from app.agents import agent
    state = {**_state(), "research_plan": _plan().model_dump(mode="json"),
             "evidence_records": [direct_evidence()], "planner_read_receipts": [{"receipt_id": "saved"}],
             "planner_budget": {"history": [{"stage": "execute"}]}}
    captured = {}

    async def inner(_graph: Any, current: Any) -> Any:
        """参数为内部图与实际桥接状态；在真实交接点保存输入后停止，避免执行外部服务。"""
        captured.update(current)
        raise RuntimeError("bridge inspected")

    monkeypatch.setattr(agent, "_log_agent_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent, "create_rag_node", lambda: object())
    monkeypatch.setattr(agent, "_run_rag_graph_stream", inner)
    with pytest.raises(RuntimeError, match="bridge inspected"):
        asyncio.run(agent.rag_node(state))
    for key in ("run_id", "research_plan", "evidence_records", "planner_read_receipts", "planner_budget"):
        assert captured[key] == state[key]






def test_index_probe_failure_keeps_execution_receipt() -> None:
    """无参数；成功返回空读取之后索引探针失败，仍需进入带凭据的正式失败出口。"""
    async def adapter(_unit: Any, _context: Any) -> EvidenceReadResult:
        """参数为取证请求；返回真实已完成但为空的读取。"""
        return EvidenceReadResult((), KnowledgeReadReport(operation="chunks", status="complete"))

    async def index_probe(_papers: Any) -> bool:
        """参数为已查范围；模拟后处理阶段的明确存储故障。"""
        raise RuntimeError("index observation failed")

    delta = asyncio.run(pipeline.execute_planner_state(
        {**_state(), "research_plan": _plan(materials=True).model_dump(mode="json")},
        llm=object(), settings=get_settings(), adapters={"local_direct_text": adapter},
        request_resolver=_request_key, index_probe=index_probe,
    ))
    assert delta["planner_loop_decision"] == "terminate_fail"
    assert delta["error_summary"] == "index observation failed"
    assert delta["planner_trace"]["message"] == delta["error_summary"]
    assert delta["planner_read_receipts"][0]["outcome"] == "empty"
    assert delta["planner_read_receipts"][0]["actual_call_count"] == 1
    full_trace = build_full_trace(TraceContext(), {"runtime_chain": "planner", **build_planner_trace_payload(delta)})
    planner = next(stage for stage in full_trace["stages"] if stage["name"] == "planner")
    assert planner["execution"]["read_receipts"] == delta["planner_read_receipts"]
