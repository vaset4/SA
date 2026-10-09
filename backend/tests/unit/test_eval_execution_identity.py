"""评估图入口的单次研究身份与冻结快照回归。作者：xxx。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.evaluation import eval_pipeline
from app.middleware.trace_context import get_trace_ctx
from evals.agent_outcome.contracts import AgentCase, agent_record_contracts, freeze_agent_run, load_agent_case
from evals.agent_outcome.environment import AgentEnvironment
from evals.agent_outcome.runner import collect_graph_turn, run_episode_sync
from evals.contracts import read_snapshot, snapshot
from evals.evidence.live_capture import CapturedRunFailure, capture_live_run
from evals.ragas.rag_adapter import LiveRAGAdapter
from evals.suites import runtime as suite_runtime
from evals.suites.adapters import stable_execution_input
from evals.suites.compare import compare_suites
from evals.suites.contracts import ComparisonRequest
from test_eval_suites_aggregation_comparison import make_summary
from test_eval_suites_contracts_adapters import make_agent_record
from test_eval_suites_execution import prepared_fixture


class _InputGraph:
    """只记录图边界实际输入，不连接模型、检索或数据库。"""

    def __init__(self, *, failure: Exception | None = None) -> None:
        """参数为可选外部故障；初始化本图调用及请求追踪记录，无返回值。"""
        self.inputs: list[dict[str, Any]] = []
        self.request_ids: list[str] = []
        self.failure = failure

    def _capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        """参数为真实传入载荷；保存独立副本并返回受控终态，缺少运行身份直接失败。"""
        assert UUID(payload["run_id"]).version == 4
        self.inputs.append(deepcopy(payload))
        trace = get_trace_ctx()
        if trace is not None:
            self.request_ids.append(trace.request_id)
        if self.failure is not None:
            raise self.failure
        return {"run_id": payload["run_id"], "answer": "已记录图输入", "search_results": [],
                "turn_outcome": {"status": "completed"}}

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """参数为实际非流输入；返回保留运行身份的终态，不调用外部服务。"""
        return self._capture(payload)

    async def astream(
        self, payload: dict[str, Any], *, stream_mode: list[str],
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """参数为实际流输入和流模式；返回同一次研究的节点更新及最终状态。"""
        assert stream_mode == ["updates", "values"]
        result = self._capture(payload)
        yield "updates", {"planner_execute": {"run_id": payload["run_id"]}}
        yield "values", result


def test_simple_eval_starts_distinct_research_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为图工厂替身；同题两次新评估获得不同身份，返回的原评估行结构不变。"""
    from app.nodes import rag_node

    graph = _InputGraph()
    monkeypatch.setattr(rag_node, "create_rag_node", lambda: graph)
    first = asyncio.run(eval_pipeline.run_rag_pipeline("问题"))
    second = asyncio.run(eval_pipeline.run_rag_pipeline("问题"))

    assert len(graph.inputs) == 2
    assert graph.inputs[0]["run_id"] != graph.inputs[1]["run_id"]
    assert first == second == {"question": "问题", "answer": "已记录图输入", "contexts": []}
    assert all(item["user_query"] == "问题" for item in graph.inputs)


def test_live_adapter_freezes_execution_identity_separately_from_case_and_trace() -> None:
    """无参数；实际图输入与冻结快照身份一致，同题新执行不复用研究或评分身份。"""
    sample = {"id": "case-a", "question": "问题", "reference_answer": "标注答案"}
    original = deepcopy(sample)
    graph = _InputGraph()
    adapter = LiveRAGAdapter(rag_node_factory=lambda: graph)

    records = asyncio.run(adapter._load_async([sample, sample]))

    assert sample == original
    assert records[0]["eval_case"] == records[1]["eval_case"]
    assert len({item["run_id"] for item in graph.inputs}) == 2
    assert len(set(graph.request_ids)) == 2
    for record, actual, request_id in zip(records, graph.inputs, graph.request_ids, strict=True):
        frozen = record["run_record"]
        assert read_snapshot(frozen["input_snapshot"]) == actual
        assert actual["run_id"] not in {request_id, frozen["run_id"], frozen["case_id"]}
        assert "run_id" not in read_snapshot(record["eval_case"]["input_snapshot"])


def test_live_capture_preserves_existing_identity_on_reentry_and_failure() -> None:
    """无参数；已构造输入再次采集及失败时保留运行身份，每次请求追踪仍独立。"""
    payload = LiveRAGAdapter()._build_live_input({"question": "问题"})
    original = deepcopy(payload)
    graph = _InputGraph()
    first = asyncio.run(capture_live_run(graph, payload))
    second = asyncio.run(capture_live_run(graph, payload))
    graph.failure = TimeoutError("offline source timeout")

    with pytest.raises(CapturedRunFailure) as captured:
        asyncio.run(capture_live_run(graph, payload))

    assert payload == original
    assert first["run_id"] == second["run_id"] == payload["run_id"]
    assert graph.inputs == [original, original, original]
    assert len(set(graph.request_ids)) == 3
    assert payload["run_id"] not in graph.request_ids
    assert read_snapshot(captured.value.capture.input_snapshot) == original
    assert captured.value.capture.error_type == "TimeoutError"
    assert captured.value.__cause__ is graph.failure


def _agent_setup(tmp_path: Path, graph: _InputGraph) -> tuple[AgentCase, AgentEnvironment]:
    """参数为隔离目录与图替身；构造真实 suite 范围装配和两轮新问题，不访问业务介质。"""
    prepared = prepared_fixture(tmp_path / "input")
    scope = read_snapshot(prepared.scope_snapshot)
    live = suite_runtime.LivePreparationRuntime()
    live.active_scopes[snapshot(scope)] = scope
    runtime = suite_runtime.AgentRuntime(
        prepared=prepared, output_dir=tmp_path, live=live, graph_factory=lambda: graph,
    )
    case = load_agent_case({
        "id": "identity-case", "title_zh": "身份回归", "task_family": "controlled_graph",
        "input": {"question_zh": "第一问", "paper_keys": ["native"], "project_scope": scope,
                  "structured_query": {"project_scope": scope, "paper_ids": scope["paper_ids"]}},
        "initial_conditions": {"history": []},
        "turns": [{"turn": 1, "user_message": "第一问"}, {"turn": 2, "user_message": "第二问"}],
    })
    return case, suite_runtime.build_agent_environment(runtime)


def test_suite_runtime_freezes_new_research_identity_per_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """参数为隔离目录及凭证端口替身；正式 episode 每个新问题分配身份并冻结完整实际输入。"""
    monkeypatch.setattr(suite_runtime, "export_secrets", lambda _kind: ())
    graph = _InputGraph()
    case, environment = _agent_setup(tmp_path, graph)

    record = run_episode_sync(case, environment, run_mode="agent_live")

    assert record["agent_run"]["status"] == "completed"
    turns = read_snapshot(record["agent_run"]["observation_snapshot"])["turns"]
    assert len(turns) == len(graph.inputs) == 2
    assert graph.inputs[0]["run_id"] != graph.inputs[1]["run_id"]
    assert "run_id" not in read_snapshot(case.input_snapshot)
    assert read_snapshot(record["run_record"]["input_snapshot"]) == graph.inputs[0]
    for turn, actual, request_id in zip(turns, graph.inputs, graph.request_ids, strict=True):
        assert read_snapshot(turn["input_snapshot"]) == actual
        assert read_snapshot(turn["values_snapshot"])["run_id"] == actual["run_id"]
        assert turn["events"][0]["payload"]["planner_execute"]["run_id"] == actual["run_id"]
        assert actual["run_id"] not in {request_id, record["agent_run"]["run_id"], case.case_id}
    assert graph.inputs[1]["chat_history"][-1] == {"role": "assistant", "content": "已记录图输入"}


def test_agent_environment_and_stream_preserve_existing_run_input(tmp_path: Path) -> None:
    """参数为隔离目录；已有执行输入经过环境及重复流采集保持身份，不把请求追踪当运行。"""
    graph = _InputGraph()
    case, environment = _agent_setup(tmp_path, graph)
    existing = environment.build_turn_input(case, user_message="第一问", history=[])
    original = deepcopy(existing)
    continued = AgentEnvironment(graph_factory=lambda: graph, state_builder=lambda *_args: existing)
    payload = continued.build_turn_input(case, user_message="第一问", history=[])

    first = asyncio.run(collect_graph_turn(graph, payload, turn=1, env=continued))
    second = asyncio.run(collect_graph_turn(graph, payload, turn=1, env=continued))

    assert existing == payload == original
    assert payload is not existing
    assert graph.inputs == [original, original]
    assert len(set(graph.request_ids)) == 2
    assert original["run_id"] not in graph.request_ids
    assert first["status"] == second["status"] == "completed"
    assert read_snapshot(first["input_snapshot"]) == read_snapshot(second["input_snapshot"]) == original


def test_comparison_projection_keeps_nested_identity_and_original_snapshot() -> None:
    """无参数；比较只排除根执行编号与发布租约，业务编号和原始记录不变。"""
    first = snapshot({"run_id": str(uuid4()), "user_query": "问题", "business": {"run_id": "business-a"},
                      "project_scope": {"publication": {"lease_id": "lease-a", "batch_ids": ["batch"]}}})
    second = read_snapshot(first)
    second["run_id"] = str(uuid4())
    second["project_scope"]["publication"]["lease_id"] = "lease-b"
    projected = stable_execution_input(first)

    assert projected == stable_execution_input(snapshot(second))
    assert read_snapshot(projected)["business"] == {"run_id": "business-a"}
    assert read_snapshot(first)["run_id"] != second["run_id"]
    assert read_snapshot(first)["project_scope"]["publication"]["lease_id"] == "lease-a"


@pytest.mark.parametrize("change", ["none", "question", "nested_business_id"])
def test_suite_comparison_distinguishes_execution_identity_from_business_change(
    tmp_path: Path, change: str,
) -> None:
    """参数为隔离目录及变化项；同案新研究仍可比较，实际问题或嵌套业务编号改变仍阻止比较。"""
    records = []
    for position in range(2):
        case, run = agent_record_contracts(make_agent_record())
        observations = read_snapshot(run.observation_snapshot)["turns"]
        inputs = read_snapshot(run.input_snapshot)
        inputs.update({"run_id": str(uuid4()), "business": {"run_id": "business-a"}})
        if position == 1 and change == "question":
            inputs["user_query"] = "另一个问题"
        if position == 1 and change == "nested_business_id":
            inputs["business"] = {"run_id": "business-b"}
        observations[0]["input_snapshot"] = snapshot(inputs)
        records.append(freeze_agent_run(case, observations, run_mode=run.run_mode,
                                        environment=read_snapshot(run.environment_snapshot)))
    originals = deepcopy(records)
    baseline = make_summary(tmp_path / "before", (records[0],), kind="agent")
    current = make_summary(tmp_path / "after", (records[1],), kind="agent")

    comparison = compare_suites(current, baseline, ComparisonRequest(changed_factors=()))

    assert records == originals
    assert records[0]["eval_case"] == records[1]["eval_case"]
    assert read_snapshot(records[0]["run_record"]["input_snapshot"])["run_id"] != read_snapshot(
        records[1]["run_record"]["input_snapshot"],
    )["run_id"]
    if change == "none":
        assert comparison.status == "comparable", comparison.reasons
    else:
        assert comparison.status == "incomparable"
        assert any("undeclared_change:input" in reason for reason in comparison.reasons)
