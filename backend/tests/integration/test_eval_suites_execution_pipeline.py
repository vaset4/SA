"""准备、独立进程运行和前后对比的文件闭环验证。作者：xxx。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evals.contracts import create_case, read_snapshot, snapshot
from evals.evidence.report import build_evidence_report
from evals.suites.execute import execute_suite
from evals.suites.execution_contracts import ExecutionRequest
from evals.suites.runtime import RetrievalRuntime, run_retrieval_job
from test_eval_suites_execution import prepared_fixture


def test_independent_worker_runs_and_same_condition_comparison(tmp_path: Path) -> None:
    """参数为目录；启动两次真实 worker 并比较，验证独立身份和可重建产物闭环。"""
    prepared = prepared_fixture(tmp_path / "input", planner_only=True)
    baseline = execute_suite(prepared, ExecutionRequest(output_dir=tmp_path / "baseline"))
    current = execute_suite(prepared, ExecutionRequest(output_dir=tmp_path / "current", baseline=baseline))
    before = json.loads(baseline.read_text(encoding="utf-8"))
    after = json.loads(current.read_text(encoding="utf-8"))
    assert before["slots"][0]["run_id"] != after["slots"][0]["run_id"]
    report = json.loads((current.parent / "summary" / "report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["status"] == "comparable"
    plan = next(delta for delta in report["comparison"]["deltas"] if delta["metric"] == "plan_valid_rate")
    assert plan["delta"] == 0.0
    assert (current.parent / "result.json").is_file()


def test_retrieval_entry_uses_actual_query_and_records_no_answer(tmp_path: Path) -> None:
    """参数为目录；真实检索适配层只接收问题和范围，候选真值不进入检索实参。"""
    from app.services.retrieval_service import search_chunks

    prepared = prepared_fixture(tmp_path)
    raw = read_snapshot(prepared.jobs[0].cases[0].record_snapshot)
    case = create_case(raw)
    calls = []

    class Store:
        """只替换外部向量存储，正文来自实际测试 PDF。"""

        def hybrid_search(self, query, *, top_k, filter_expr):
            """参数为真正调用实参；返回对应真实测试原文的检索载荷。"""
            calls.append((query, top_k, filter_expr))
            return [{"paper_id": "actual-paper", "chunk_id": "actual-child", "text": prepared.sources[0].blocks[0].text}]

    runtime = RetrievalRuntime(store=Store(), scope_snapshot=snapshot(raw["project_scope"]), top_k=1,
                               index_id=prepared.index.index_snapshot_id, backend="fixture", search=search_chunks)
    record = asyncio.run(run_retrieval_job(case, runtime))
    observed = read_snapshot(record["run_record"]["observation_snapshot"])
    assert len(calls) == 1 and calls[0][0] == raw["question"]
    assert observed["model_call_observations"] == [] and "response" not in observed
    assert "evidence_benchmark" not in record["run_record"]["input_snapshot"]
    report = build_evidence_report([record], grading_policy="candidate_diagnostic")
    metrics = report["records"][0]["measurements"]
    assert next(item for item in metrics if item["name"] == "evidence_group_recall_at_k")["value"] == 1.0


class _StructuredReply:
    """只控制外部模型回复，使用产品真实深度合同。"""

    def __init__(self, schema) -> None:
        """参数为产品请求的合同；保存本次结构化响应类型。"""
        self.schema = schema

    async def ainvoke(self, messages):
        """参数为真实模型消息；按现役深度/直答合同返回明确问候，不替换实际产品节点。"""
        if self.schema.__name__ == "DirectAnswerDraft":
            return self.schema(answer="你好，可以开始讨论研究问题。", knowledge_bearing=False)
        assert self.schema.__name__ == "TurnDepthProposal"
        return self.schema(depth="minimal", reason_code="no_paper_fact_needed", requires_paper_evidence=False, resolved_query="你好")


class _GreetingModel:
    """问候场景的外部模型替身；不替换图节点或写回服务。"""

    def with_structured_output(self, schema):
        """参数为实际产品合同；返回独立结构化回复端口。"""
        return _StructuredReply(schema)

    async def ainvoke(self, messages):
        """参数为真实回答请求；返回受控文本消息。"""
        return SimpleNamespace(content="你好，可以开始讨论研究问题。")


class _IndexedStore:
    """索引就绪检查端口；本问候场景不会调用正文检索。"""

    def exists(self, paper_id):
        """参数为论文 ID；仅测试准备包中的实际论文被声明为已索引。"""
        return paper_id == "actual-paper"

    def ensure_collection(self):
        """无参数；该端口必须由隔离运行时绑定只读检查。"""
        raise AssertionError("unexpected direct index initialization")

    def for_user(self, user_id: str) -> _IndexedStore:
        """参数为读取用户；只接受本测试授权身份，返回同一离线存储。"""
        assert user_id == "fixture-user"
        return self

    def for_snapshot(self, publication: dict[str, object] | None) -> _IndexedStore:
        """参数为发布快照；测试准备包没有发布租约，返回同一离线存储。"""
        assert publication is None
        return self

    def list_section_paths(self, paper_id: str) -> list[str]:
        """参数为论文编号；本测试 PDF 未标注章节名，返回实际空目录。"""
        assert paper_id == "actual-paper"
        return []


def test_real_graph_preserves_trace_and_uses_isolated_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录和外部依赖；实际产品图贯通隔离IO、workflow评分与新检索策略，不调用真实模型。"""
    from app.nodes import rag_node
    from app.services import database
    from app.services.internal_research import research_planner_pipeline as planner_pipeline
    from app.services.llm_service import OpenAICompatibleLLM
    from evals.agent_outcome.contracts import load_agent_case
    from evals.agent_outcome.workflow import grade_workflow_record
    from evals.ragas.frozen_bridge import score_frozen_record
    from evals.suites.contracts import RunSlot
    from evals.suites.execute import _write_case_artifacts
    from evals.suites.execution_contracts import PreparedCase, PreparedJob
    from evals.suites.policy import workflow_policy
    from evals.suites.runtime import AgentRuntime, run_agent_job

    prepared = prepared_fixture(tmp_path / "input")
    scope = read_snapshot(prepared.scope_snapshot)
    raw = {"id": "real-graph-greeting", "title_zh": "真实图接口验证", "task_family": "controlled_graph",
           "input": {"question_zh": "你好", "paper_keys": ["native"], "project_scope": scope, "structured_query": {"project_scope": scope, "paper_ids": scope["paper_ids"]}},
           "initial_conditions": {"history": [], "focused_paper_key": None}, "turns": [{"turn": 1, "user_message": "你好"}],
           "required_outcomes": [{"predicate": "authorized_scope_preserved"}], "allowed_terminal_states": [{"state": "completed", "task_completed": True}]}
    store = _IndexedStore()
    live = SimpleNamespace(store=store, require_existing_index=lambda: store.exists("actual-paper"), execution_scope=lambda value: value)
    monkeypatch.setattr(rag_node, "llm", _GreetingModel())
    monkeypatch.setattr(rag_node, "get_milvus_store", lambda: store)
    # 1. 【评测集成】【离线读取】焦点清单、章节目录和图谱状态各自走真实入口，仅替换外部数据端口
    paper_reader = AsyncMock(return_value=read_snapshot(prepared.initial_snapshot)["papers"])
    monkeypatch.setattr(planner_pipeline, "get_paper_repository", lambda: SimpleNamespace(get_papers_by_ids=paper_reader))
    monkeypatch.setattr(planner_pipeline, "get_milvus_store", lambda: store)
    graph_reader = AsyncMock(return_value={"meta": {"node_count": 0, "edge_count": 0}})
    monkeypatch.setattr(planner_pipeline, "get_research_object_query_service", lambda: SimpleNamespace(query_project_subgraph=graph_reader))
    # 2. 【评测集成】【意外外联】未覆盖的数据库或模型端口立即失败，不能以真实连接补齐测试夹具
    database_access = AsyncMock(side_effect=AssertionError("unexpected real database access"))
    model_http = Mock(side_effect=AssertionError("unexpected real model HTTP request"))
    monkeypatch.setattr(database, "get_pool", database_access)
    monkeypatch.setattr(OpenAICompatibleLLM, "_post_chat_completions", model_http)
    monkeypatch.setattr(OpenAICompatibleLLM, "_post_chat_completions_stream", model_http)
    output = tmp_path / "run"
    output.mkdir()
    runtime = AgentRuntime(prepared=prepared, output_dir=output, live=live, graph_factory=rag_node.create_rag_node)
    case = load_agent_case(raw)
    record = asyncio.run(run_agent_job(case, runtime))
    observed = read_snapshot(record["agent_run"]["observation_snapshot"])
    assert record["agent_run"]["status"] == "completed", observed
    values = read_snapshot(observed["turns"][0]["values_snapshot"])
    assert values["turn_depth"] == "minimal" and isinstance(values["trace"], dict)
    assert values["run_id"] == read_snapshot(observed["turns"][0]["input_snapshot"])["run_id"]
    paper_reader.assert_awaited_once_with(["actual-paper"], user_id="fixture-user")
    database_access.assert_not_called()
    model_http.assert_not_called()
    access = json.loads((output / "storage" / "access.json").read_text(encoding="utf-8"))
    assert any(item["action"] == "rag_trace" for item in access)
    assert record["agent_run"]["missing_fields"] == []
    workflow = grade_workflow_record(record)
    metrics = {row["name"]: row for row in workflow["measurements"]}
    for name in ("workflow_completion", "workflow_rule_compliance", "scope_adherence", "citation_reference_validity"):
        assert metrics[name]["value"] == 1, workflow
    assert metrics["structured_first_response_compliance"]["status"] == "missing"
    policy = workflow_policy()
    judge = AsyncMock(side_effect=AssertionError("no judge is applicable to greeting"))
    retrieval = asyncio.run(score_frozen_record(record, kind="agent", scorer=judge, policy=policy))
    assert not judge.called
    assert {row["name"] for row in retrieval["measurements"]} == {"context_precision", "context_recall"}
    assert all(row["status"] == "not_measured" for row in retrieval["measurements"])
    job = PreparedJob(job_id="actual-graph", kind="agent", cases=(PreparedCase(
        case_id=case.case_id, case_version=case.case_version, record_snapshot=snapshot(case.model_dump(mode="json"))),),
        slots=(RunSlot(case_id=case.case_id, case_version=case.case_version, repeat_id="repeat-1", run_id=None,
                       required_checks=("scope", "citation", "truth_leakage", "required_trace", "false_success")),))
    report_path, frozen_path = _write_case_artifacts(record, job, output=output, policy=policy)
    assert json.loads(report_path.read_text(encoding="utf-8"))["rows"][0]["measurements"] == workflow["measurements"]
    assert frozen_path.is_file()
    missing = deepcopy(record)
    missing_observation = read_snapshot(missing["run_record"]["observation_snapshot"])
    missing_observation["turns"][0]["events"] = []
    missing["run_record"]["observation_snapshot"] = snapshot(missing_observation)
    incomplete = asyncio.run(score_frozen_record(missing, kind="agent", scorer=judge, policy=policy))
    assert not judge.called and all(row["status"] == "missing" for row in incomplete["measurements"])


def test_worker_logs_and_failure_artifact_redact_before_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为目录和受控故障；实际worker日志与produced文件在首次写入前脱敏，原异常链保持。"""
    from evals.suites import execute

    prepared = prepared_fixture(tmp_path / "input", planner_only=True)
    secret = "fixture-worker-credential"
    original = ConnectionError(f"credential rejected: {secret}")
    cause = ValueError(f"upstream detail: {secret}")

    async def fail_worker(*args, **kwargs):
        """参数为真实worker调度实参；仅以已知凭证的受控异常替换外部执行。"""
        raise original from cause

    monkeypatch.setattr(execute, "export_secrets", lambda kind: (secret,))
    monkeypatch.setattr(execute, "_run_cancellable_worker", fail_worker)
    output = tmp_path / "run"
    job = prepared.jobs[0]
    result = asyncio.run(execute.run_worker(prepared, job, slot=job.slots[0], output=output))
    assert result.status == "failed" and secret not in result.error
    assert original.__cause__ is cause and secret in str(original)
    for filename in ("runtime.log", "produced.json"):
        contents = (output / filename).read_text(encoding="utf-8")
        assert secret not in contents and "ConnectionError" in contents and "credential rejected" in contents
    assert "ValueError" in (output / "runtime.log").read_text(encoding="utf-8")
