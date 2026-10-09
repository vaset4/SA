"""Agent Outcome 多轮流采集、失败留痕和评分回归。作者：xxx。"""

from __future__ import annotations

import json
from pathlib import Path

from evals.agent_outcome.contracts import load_agent_case
from evals.agent_outcome.environment import AgentEnvironment
from evals.agent_outcome.report import build_agent_report, write_agent_report
from evals.agent_outcome.runner import run_episode_sync
from evals.contracts import snapshot_hash
from evals.evidence.alignment import locate_span
from evals.evidence.contracts import EvidenceBenchmark, EvidenceSpan, SourceBlock, make_source_snapshot
from evals.evidence.corpus import build_index_from_chunks

_ACTUAL_TEXT = "75% random masking uses temporal continuity"


def _benchmark() -> EvidenceBenchmark:
    """无参数；建立固定测试原文和完整切块映射，不向被测图提供预期标签。"""
    source = make_source_snapshot(
        paper_id="lsttn", pdf_sha256=snapshot_hash("test-pdf"), artifact_manifest_snapshot="{}",
        blocks=(SourceBlock(block_id="body", page_num=1, section_path="", text=_ACTUAL_TEXT),),
    )
    spans = []
    for identity, quote in (("ratio", "75% random masking"), ("reason", "temporal continuity")):
        block, start, end = locate_span(source, quote)
        spans.append(EvidenceSpan(span_id=identity, source_snapshot_id=source.snapshot_id, block_id=block,
                                  start=start, end=end, quote=quote, quote_sha256=snapshot_hash(quote)))
    index, _issues = build_index_from_chunks(
        (source,), [{"paper_id": "lsttn", "chunk_id": "chunk-1", "text": _ACTUAL_TEXT}], retrieval_backend="fixture",
    )
    return EvidenceBenchmark.model_validate({
        "dataset_version": "test-v1", "review_status": "candidate", "sources": [source], "spans": spans,
        "groups": [{"group_id": "masking", "importance": "critical", "proofs": [{"span_ids": ["ratio", "reason"]}]}],
        "index_snapshots": [index], "target_paper_ids": ["lsttn"],
    })


def _environment(factory) -> AgentEnvironment:
    """参数为可控图工厂；声明本次测试实际索引身份。"""
    return AgentEnvironment(graph_factory=factory, declared={"index_snapshot_id": _benchmark().index_snapshots[0].index_snapshot_id})


def _record() -> dict:
    """无参数；返回带真实证据引用的成功场景。"""
    return {
        "id": "agent-run-001", "title_zh": "运行测试", "task_family": "evidence_task_completion",
        "review_status": "candidate", "execution_status": "not_run",
        "input": {"question_zh": "查找证据", "paper_keys": ["lsttn"], "project_scope": {
            "user_id": "test-owner", "paper_ids": ["lsttn"], "local_retrieval_allowed": True,
        }},
        "initial_conditions": {"history": [], "focused_paper_key": "lsttn"},
        "turns": [{"turn": 1, "user_message": "查找证据"}],
        "evidence_refs": [{"case_id": "rag-001", "evidence_key": "target"}],
        "evidence_benchmark": _benchmark().model_dump(mode="json"),
        "evidence_targets": {"required_span_ids": ["ratio", "reason"], "required_group_ids": ["masking"], "required_paper_ids": ["lsttn"]},
        "required_outcomes": [
            {"predicate": "all_required_evidence_observed"},
            {"predicate": "returned_sources_resolve_to_observed_evidence"},
            {"predicate": "authorized_scope_preserved"},
        ],
        "allowed_terminal_states": [{"state": "completed_with_verified_evidence", "task_completed": True}],
    }


class _SuccessGraph:
    """返回结构化证据和终态的可控图。"""

    async def astream(self, _input, *, stream_mode):
        assert stream_mode == ["updates", "values"]
        yield "updates", {"planner_execute": {"action": "collect"}}
        yield "values", {
            "answer": "真实结果",
            "turn_outcome": {"state": "completed_with_verified_evidence"},
            "evidence_records": [{"evidence_id": "e1", "chunk_id": "chunk-1", "paper_id": "lsttn", "excerpt": _ACTUAL_TEXT}],
            "citations": [{"evidence_id": "e1", "paper_id": "lsttn", "chunk_id": "chunk-1"}],
        }


class _FailureGraph:
    """抛出实际检索错误的可控图。"""

    async def astream(self, _input, *, stream_mode):
        assert stream_mode == ["updates", "values"]
        yield "updates", {"planner_execute": {"action": "collect"}}
        raise TimeoutError("fixture source unavailable")


def test_episode_preserves_actual_turn_and_scores_success() -> None:
    """无参数；验证只把实际图流和实际输入写进运行记录。"""
    case = load_agent_case(_record())
    environment = _environment(_SuccessGraph)
    record = run_episode_sync(case, environment)
    values = json.loads(record["agent_run"]["observation_snapshot"])["turns"][0]["values_snapshot"]
    assert "required_outcomes" not in values
    report = build_agent_report([record])
    measurements = report["rows"][0]["grading_result"]["measurements"]
    assert next(item for item in measurements if item["name"] == "task_success")["value"] == 1.0


def test_episode_keeps_graph_failure_without_fabricating_terminal_state() -> None:
    """无参数；图异常保留错误且任务不被伪装为完成。"""
    case = load_agent_case(_record())
    record = run_episode_sync(case, AgentEnvironment(graph_factory=_FailureGraph))
    turn = json.loads(record["agent_run"]["observation_snapshot"])["turns"][0]
    assert turn["status"] == "failed"
    assert turn["error"]["type"] == "TimeoutError"
    assert turn["terminal_state"] is None
    measurements = build_agent_report([record])["rows"][0]["grading_result"]["measurements"]
    assert next(item for item in measurements if item["name"] == "task_success")["value"] == 0.0


def test_report_artifacts_are_reproducible_and_not_overwritten(tmp_path: Path) -> None:
    """参数为 pytest 临时目录；相同快照可重复写入，改动产物必须失败。"""
    case = load_agent_case(_record())
    record = run_episode_sync(case, _environment(_SuccessGraph))
    report = build_agent_report([record])
    write_agent_report(tmp_path, report, [record])
    write_agent_report(tmp_path, report, [record])
    report_path = tmp_path / "report.json"
    report_path.write_text(report_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    try:
        write_agent_report(tmp_path, report, [record])
    except FileExistsError:
        pass
    else:
        raise AssertionError("changed report artifact was silently overwritten")


def test_environment_records_unconfigured_port() -> None:
    """无参数；未配置外部能力必须留下环境问题，不能静默跳过。"""
    from evals.agent_outcome.environment import EnvironmentPortUnavailable

    environment = AgentEnvironment(graph_factory=_SuccessGraph)
    try:
        environment.require_port("original_text")
    except EnvironmentPortUnavailable:
        pass
    else:
        raise AssertionError("missing environment port did not fail")
    assert environment.validate_after_run() == ("environment_access",)


def test_real_rag_trace_can_be_frozen() -> None:
    """无参数；真实图的 trace 应保留结构化身份，不使流采集失败。"""
    from app.middleware.observability import RAGTrace
    from evals.agent_outcome.runner import _serializable

    captured = _serializable({"trace": RAGTrace(request_id="actual-trace-001")})
    assert captured["trace"]["request_id"] == "actual-trace-001"


class _ClarificationGraph:
    """逐轮返回真实形状的澄清和结束状态，不运行外部模型。"""

    async def astream(self, payload, *, stream_mode):
        """参数为本轮输入与流模式；输出随用户后续消息变化的状态。"""
        clarified = payload["user_query"] == "使用 LSTTN"
        yield "values", {
            "answer": "已确认论文" if clarified else "请确认论文",
            "turn_outcome": {"status": "completed" if clarified else "incomplete"},
            "turn_depth_assessment": {"needs_clarification": not clarified},
        }


def test_clarification_terminal_does_not_end_declared_episode() -> None:
    """无参数；第一轮澄清终态不能吞掉事前声明的第二轮用户回复。"""
    raw = _record()
    raw["turns"].append({"turn": 2, "user_message": "使用 LSTTN"})
    record = run_episode_sync(
        load_agent_case(raw), AgentEnvironment(graph_factory=_ClarificationGraph),
    )
    turns = json.loads(record["agent_run"]["observation_snapshot"])["turns"]
    assert len(turns) == 2
    assert turns[0]["terminal_state"] == "incomplete"
    second_input = json.loads(turns[1]["input_snapshot"])
    assert second_input["chat_history"][-1] == {"role": "assistant", "content": "请确认论文"}


def test_paper_id_in_input_is_not_observed_evidence() -> None:
    """无参数；仅在输入或计划出现论文编号，不能被评为已完成取证。"""
    from evals.agent_outcome.contracts import freeze_agent_run

    raw = _record()
    raw["required_outcomes"] = [{"predicate": "requested_paper_evidence_observed"}]
    case = load_agent_case(raw)
    observations = [{
        "turn": 1, "events": [], "input_snapshot": "{}",
        "values_snapshot": json.dumps({"structured_query": {"paper_id": "lsttn"}}),
        "terminal_state": "completed", "status": "completed", "assistant_output": "",
    }]
    record = freeze_agent_run(case, observations, run_mode="agent_fixture", environment={})
    measures = build_agent_report([record])["rows"][0]["grading_result"]["measurements"]
    task = next(item for item in measures if item["name"] == "task_success")
    assert task["value"] != 1.0
