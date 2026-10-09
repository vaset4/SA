"""【研究循环】【独立审查】检查合同入口与领域追踪，外部端口保持隔离。作者：xxx。"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.middleware.trace_context import TraceContext, reset_trace_ctx, set_trace_ctx
from app.services.internal_research import research_planner_pipeline as pipeline
from o2_evidence_test_utils import read_result, scope
from result_delivery_test_utils import Connection, offline_memory_callback, repository  # noqa: F401


def state():
    """无参数；返回独立检查的可信用户问题与论文范围。"""
    return dict(user_query="first", project_scope=scope(), run_id="independent-review")


def initial():
    """无参数；返回尚未承诺未来分析步骤的正文读取决策。"""
    return dict(requested_facets=[dict(facet_id="first", label="first", source="explicit",
        user_text_span="first", required=True, deliverable_kind="claims", required_claim_types=["paper_supported"])],
        decision=dict(kind="execute", actions=[dict(goal="first", covered_facet_ids=["first"],
            input=dict(source_type="local_direct_text", target_paper_ids=["paper-a"], locator_queries=["first"]))]))


def test_incremental_checkpoint_cannot_reenter_whole_plan_validator():
    """无参数；当前增量账本缺失决策字段时保持动作错误语义，不能切换合同。"""
    current = {**state(), "planner_draft": initial()}
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    current.update(pipeline.validate_planner_state(current, settings))
    ledger = deepcopy(current["research_plan"])
    current["planner_draft"] = {}
    delta = pipeline.validate_planner_state(current, settings)
    assert delta["planner_observation"]["error_code"] == "invalid_action_format"
    assert current["research_plan"] == ledger
    assert delta.get("research_plan", ledger)["metadata"]["execution_mode"] == "incremental"


def test_action_admission_records_business_validation():
    """无参数；合法与越权动作均保存真实准入结论及错误，供完整追踪持久化。"""
    ctx = TraceContext(request_id="offline-review")
    token = set_trace_ctx(ctx)
    try:
        current = {**state(), "planner_draft": initial()}
        settings = SimpleNamespace(rag_research_planner_max_units=8)
        pipeline.validate_planner_state(current, settings)
        current["planner_draft"]["decision"]["actions"][0]["input"]["target_paper_ids"] = ["foreign"]
        pipeline.validate_planner_state(current, settings)
        events = [json.loads(row) for row in ctx.evidence_observations]
        outcomes = [row["payload"] for row in events if row["stage"] == "structured_business_validation"]
        assert [row["accepted"] for row in outcomes] == [True, False]
        assert all(row["schema"] == "InitialResearchDecision" for row in outcomes)
        assert outcomes[1]["errors"]
        assert outcomes[1]["errors"][0]["code"] == "out_of_scope_paper_id"
    finally:
        reset_trace_ctx(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", [None, "session"])
async def test_incremental_insufficient_delivery_survives_save_and_reopen(fail_at):
    """参数为保存故障点；真实空读取的账本经正式保存服务提交或暂存，重开不混写。"""
    from app.services.result_delivery_service import deliver_result, read_delivery_response
    from app.services.internal_research.research_planner_observability import build_planner_trace_payload

    current = {**state(), "planner_draft": initial()}
    settings = SimpleNamespace(rag_research_planner_max_units=8)
    model = SimpleNamespace(model="offline", temperature=0, base_url="https://offline.invalid")
    current.update(pipeline.validate_planner_state(current, settings))

    async def empty(unit, context):
        """参数为真实动作与上下文；明确返回离线空读取。"""
        return read_result([])

    current.update(await pipeline.execute_planner_state(current, llm=model, settings=settings,
        adapters={"local_direct_text": empty}))
    current["planner_draft"] = dict(decision=dict(kind="finish", allow_partial=True, reason="no evidence"))
    current.update(pipeline.validate_planner_state(current, settings))
    current.update(await pipeline.synthesize_planner_state(current, model))
    assert current["final_delivery_report"]["status"] == "insufficient"
    assert current["research_plan"]["metadata"]["pending_unit_ids"] == []
    connection = Connection(fail_at=fail_at)
    response = await deliver_result(user_id="user", run=connection.data["runs"]["target"], result=current,
        session_status="completed", repository=repository(connection))
    target = connection.data["runs"]["target"]
    if fail_at:
        assert response["persistence_status"] == "pending"
        assert target.get("answer") is None
        metadata = target["pending_artifact"]["artifact"]["metadata"]
    else:
        reopened = read_delivery_response(json.loads(json.dumps(target, default=str)))
        assert reopened["answer"] == current["answer"]
        assert reopened["final_delivery_report"] == current["final_delivery_report"]
        assert reopened["final_citation_bindings"] == []
        metadata = reopened
    assert metadata["research_plan"] == current["research_plan"]
    trace = json.loads(json.dumps(build_planner_trace_payload(current)))
    assert trace["research_plan"] == metadata["research_plan"]
    assert trace["planner_read_receipts"] == current["planner_read_receipts"]
