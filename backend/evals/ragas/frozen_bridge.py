"""同一次冻结产品运行的独立 RAGAS 评分桥接。作者：xxx。"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any, Awaitable, Callable, Literal

from pydantic import ValidationError

from evals.agent_outcome.workflow_execution import grade_tools
from evals.agent_outcome.workflow_facts import read_turn
from evals.contracts import (
    MetricResult, build_grading, grading_projection, read_snapshot, record_contracts,
    snapshot, snapshot_hash, validate_execution_input,
)
from evals.evidence.alignment import EvidenceDataError
from evals.evidence.report import _stage_observations
from evals.suites.policy import MeasurementPolicy, RETRIEVAL_GRADER_VERSION, RETRIEVAL_METRICS

logger = logging.getLogger(__name__)
GRADER_VERSION = "ragas-frozen-v1"
RAGAS_VERSION = "0.4.3"
METRIC_VARIANTS = {
    "context_precision": "LLMContextPrecisionWithReference",
    "context_recall": "LLMContextRecall",
    "faithfulness": "Faithfulness",
    "factual_correctness": "FactualCorrectness(mode=f1,beta=1.0,atomicity=low,coverage=low)",
}
Scorer = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _unmeasured(name: str, reason: str, *, status: str = "missing") -> MetricResult:
    """参数为指标、原因与状态；返回不携带伪造分数的测量。"""
    return MetricResult.model_validate({"name": name, "status": status, "reason": reason})


def _turns(projection: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """参数为冻结投影和通道；返回独立轮次数据，保留真实空回答与缺失差异。"""
    observed = projection["observation"]
    if kind == "retrieval":
        return [{"turn": 1, "input": projection["input"], "observation": observed}]
    turns = []
    for turn in observed.get("turns", []):
        inputs = read_snapshot(turn["input_snapshot"])
        validate_execution_input(inputs)
        values = read_snapshot(turn["values_snapshot"])
        capture = read_snapshot(turn["evidence_observation_snapshot"])
        turns.append({"turn": turn["turn"], "input": inputs, "observation": capture,
                      "response": values.get("answer"), "status": turn.get("status"), "values": values,
                      "raw_turn": turn})
    return turns


def _stages(turn: dict[str, Any]) -> tuple[list[Any], MetricResult | None]:
    """参数为真实轮次；复用证据校验，返回阶段观察及显式采集错误。"""
    try:
        return [stage for stage, _ in _stage_observations(turn["observation"])], None
    except EvidenceDataError as exc:
        return [], _unmeasured("capture", exc.reason, status=exc.status)
    except ValidationError:
        return [], _unmeasured("capture", "observation_schema_invalid", status="invalid_input")


def _text_input(name: str, sample: dict[str, Any]) -> MetricResult | None:
    """参数为指标和最小输入；拒绝缺失或畸形正文，真实空回答没有可评分主张。"""
    for field, value in sample.items():
        if value is None:
            return _unmeasured(name, f"{field}_missing")
        if field == "retrieved_contexts":
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                return _unmeasured(name, "context_text_invalid", status="invalid_input")
        elif not isinstance(value, str):
            return _unmeasured(name, f"{field}_invalid", status="invalid_input")
        elif not value.strip():
            status = "not_measured" if field == "response" else "invalid_input"
            return _unmeasured(name, f"{field}_empty", status=status)
    return None


def _stage_input(name: str, stage: Any) -> tuple[list[str] | None, MetricResult | None]:
    """参数为指标及已校验阶段；返回该阶段正文，不回读父块或追加黄金证据。"""
    if stage.status != "complete" or stage.items is None:
        return None, _unmeasured(name, stage.reason or f"{stage.stage}_{stage.status}")
    if any(item.text_snapshot is None for item in stage.items):
        return None, _unmeasured(name, "observed_context_text_missing")
    return [item.text_snapshot for item in stage.items], None


def _jobs(projection: dict[str, Any], kind: str, *, retrieval_only: bool = False) -> list[dict[str, Any]]:
    """参数为冻结投影和通道；生成逐查询及最终轮回答的指标作业与适用性。"""
    turns = _turns(projection, kind)
    reference = projection["truth"].get("reference_answer")
    negative = projection["truth"].get("answerability") == "insufficient_evidence"
    jobs: list[dict[str, Any]] = []
    last_stages, last_error = [], None
    # 1. 【RAGAS】【检索评分】各查询保留自身顺序，不能拼接为虚构排名
    for turn in turns:
        stages, error = _stages(turn)
        last_stages, last_error = stages, error
        raw = [stage for stage in stages if stage.stage == "query_raw"]
        for name in RETRIEVAL_METRICS:
            for stage in raw or [None]:
                contexts, invalid = _stage_input(name, stage) if stage else (None, error)
                sample = {"user_input": turn["input"].get("user_query", turn["input"].get("question")),
                          "reference": reference, "retrieved_contexts": contexts}
                job = _job(name, sample, turn=turn["turn"], stage=stage,
                           issue=invalid, not_applicable=negative)
                if retrieval_only:
                    job = _retrieval_applicability(job, turn, kind=kind, stage=stage, negative=negative)
                jobs.append(job)
    if retrieval_only:
        return jobs or [_job(name, {"retrieved_contexts": None}, turn=0, not_applicable=negative)
                        for name in RETRIEVAL_METRICS]
    # 2. 【RAGAS】【回答评分】只评最后实际轮次，前轮澄清不得替代最终回答
    last = turns[-1] if turns else {"turn": 0, "input": {}}
    question = last["input"].get("user_query", last["input"].get("question"))
    response = last.get("response")
    jobs.append(_job("factual_correctness", {"response": response, "reference": reference},
                     turn=last["turn"], not_applicable=kind == "retrieval" or negative))
    finals = [stage for stage in last_stages if stage.stage == "final_input"]
    final = finals[0] if len(finals) == 1 else None
    contexts, issue = _stage_input("faithfulness", final) if final else (None, last_error)
    if len(finals) > 1:
        issue = _unmeasured("faithfulness", "final_generation_call_ambiguous")
    jobs.append(_job("faithfulness", {"user_input": question, "response": response,
                                      "retrieved_contexts": contexts},
                     turn=last["turn"], stage=final, issue=issue, not_applicable=kind == "retrieval"))
    if not turns:
        for name in ("context_precision", "context_recall"):
            jobs.append(_job(name, {"retrieved_contexts": None}, turn=0, not_applicable=negative))
    return jobs


def _retrieval_applicability(job: dict[str, Any], turn: dict[str, Any], *, kind: str,
                             stage: Any, negative: bool) -> dict[str, Any]:
    """参数为检索作业和真实轮次；区分负例、合法直读、采集缺失与真实空召回。"""
    if negative:
        return job
    result = job["result"]
    if stage is None and kind == "agent" and _observed_nonretrieval_path(turn):
        result = _unmeasured(job["metric"], "not_applicable:observed_nonretrieval_path",
                             status="not_measured").model_dump(mode="json")
    elif result is None and read_snapshot(job["input_snapshot"])["retrieved_contexts"] == []:
        result = MetricResult(name=job["metric"], status="measured", value=0.0,
                              reason="complete_retrieval_with_zero_hits").model_dump(mode="json")
    return {**job, "result": result}


def _observed_nonretrieval_path(turn: dict[str, Any]) -> bool:
    """参数为冻结轮次；完整无工具路径或成功直读不产生排名检索分母，缺采集仍需标缺失。"""
    capture = turn["observation"]
    if capture.get("evidence_capture_errors") != [] or turn.get("status") != "completed":
        return False
    events = capture.get("evidence_observations")
    if not isinstance(events, list):
        return False
    tools = [event["payload"] for event in events if event.get("stage") == "tool_execution_result"]
    if not tools:
        # 1. 【RAGAS】【检索适用性】沿用过程评测对真实路由的无工具判断，不从最终深度标签猜测
        assessment = turn["values"].get("turn_depth_assessment", {})
        terminal = turn["raw_turn"].get("terminal_state")
        direct = terminal in {"completed", "converged"} and assessment.get("requires_paper_evidence") is False
        clarification = terminal in {"incomplete", "terminate_honest"} and assessment.get("needs_clarification") is True
        if not direct and not clarification:
            return False
        metrics, _details = grade_tools((read_turn(turn["raw_turn"]),))
        return all(metric.status == "not_measured" for metric in metrics)
    final: dict[str, dict[str, Any]] = {}
    for tool in tools:
        identity, attempt = tool.get("tool_call_id"), tool.get("tool_attempt")
        if (tool.get("source_type") != "local_section_text" or not isinstance(identity, str)
                or not identity or not isinstance(attempt, int)):
            return False
        if identity not in final or final[identity]["tool_attempt"] < attempt:
            final[identity] = tool
    return bool(final) and all(tool.get("status") in {"success", "empty"} for tool in final.values())


def _job(name: str, sample: dict[str, Any], *, turn: int, stage: Any = None,
         issue: MetricResult | None = None, not_applicable: bool = False) -> dict[str, Any]:
    """参数为最小指标输入及来源；返回可重建的作业，不把参考答案传入忠实度。"""
    result = _text_input(name, sample)
    if issue is not None:
        result = _unmeasured(name, issue.reason, status=issue.status)
    if not_applicable:
        result = _unmeasured(name, "metric_not_applicable_to_case_or_channel", status="not_measured")
    return {"metric": name, "variant": METRIC_VARIANTS[name], "turn": turn,
            "call_id": stage.call_id if stage else None,
            "context_stage": stage.stage if stage else None,
            "retrieval_query": stage.query if stage else None,
            "input_snapshot": snapshot(sample), "input_hash": snapshot_hash(snapshot(sample)),
            "result": result.model_dump(mode="json") if result else None}


async def _score_job(job: dict[str, Any], scorer: Scorer) -> dict[str, Any]:
    """参数为独立作业和注入评分器；返回审计结果，judge 失败不补分。"""
    if job["result"] is not None:
        return {**job, "audit": {"status": "not_called"}}
    name = job["metric"]
    try:
        audit = await scorer(name, read_snapshot(job["input_snapshot"]))
    except Exception as exc:
        logger.error("【RAGAS】【指标评分】judge 失败 metric=%s error_type=%s", name, type(exc).__name__)
        audit = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    value = audit.get("value")
    if audit.get("error"):
        result = _unmeasured(name, "judge_failed")
    elif value is None:
        result = _unmeasured(name, "judge_score_missing")
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        result = _unmeasured(name, "judge_score_not_finite", status="invalid_input")
        audit = {**audit, "value": repr(value)}
    elif not 0 <= value <= 1:
        result = _unmeasured(name, "judge_score_out_of_range", status="invalid_input")
    else:
        result = MetricResult(name=name, value=value, status="measured", reason="ragas_judge_score")
    return {**job, "result": result.model_dump(mode="json"), "audit": audit}


def _aggregate(name: str, samples: list[dict[str, Any]]) -> MetricResult:
    """参数为指标名和各次评分；只有适用调用全部评分完成才给 episode 均值。"""
    results = [MetricResult.model_validate(row["result"]) for row in samples if row["metric"] == name]
    applicable = [item for item in results if item.status != "not_measured"]
    if not applicable:
        return _unmeasured(name, results[0].reason, status="not_measured")
    for status in ("invalid_input", "missing"):
        failures = [item for item in results if item.status == status]
        if failures:
            reason = failures[0].reason if len(results) == 1 else f"incomplete_calls:{failures[0].reason}"
            return _unmeasured(name, reason, status=status)
    return MetricResult(name=name, status="measured", value=sum(item.value for item in applicable) / len(applicable),
                        reason="episode_mean_of_actual_calls")


def _grading_error(samples: list[dict[str, Any]]) -> str | None:
    """参数为逐调用评分审计；返回 judge/API 失败摘要，普通不适用不算评分故障。"""
    errors = []
    for sample in samples:
        error = sample.get("audit", {}).get("error")
        if isinstance(error, dict):
            errors.append(f"{error.get('type', 'UnknownError')}:{error.get('message', '')}")
    return ";".join(dict.fromkeys(errors)) or None


def build_failed_score(
    record: dict[str, Any], *, kind: Literal["retrieval", "agent"], policy: MeasurementPolicy | None,
    error: str,
) -> dict[str, Any]:
    """参数为已落盘运行、评分通道及策略；生成只含缺失指标的独立评分失败产物，不改写运行记录。"""
    frozen = deepcopy(record)
    case, run = record_contracts(frozen)
    enabled = policy.ragas_metrics if policy is not None else tuple(METRIC_VARIANTS)
    jobs = _jobs(grading_projection(case, run), kind, retrieval_only=policy is not None)
    samples = [{**job, "result": _unmeasured(job["metric"], "grading_failed").model_dump(mode="json"),
                "audit": {"error": {"type": "ScoringError", "message": error}}} for job in jobs]
    measurements = [_unmeasured(name, "grading_failed") for name in enabled]
    version = RETRIEVAL_GRADER_VERSION if policy is not None else GRADER_VERSION
    grading = build_grading(frozen, measurements, grader="ragas_frozen", grader_version=version)
    return {
        "case_id": case.case_id,
        "run_id": run.run_id,
        "grader_version": version,
        "measurement_policy": policy.model_dump(mode="json") if policy is not None else None,
        "ragas_version": RAGAS_VERSION,
        "kind": kind,
        "grading_status": "failed",
        "grading_error": error,
        "grading_result": grading,
        "measurements": grading["measurements"],
        "samples": samples,
        "judge": {"status": "failed", "error": {"type": "ScoringError", "message": error}},
        "aggregation": "equal_weight_per_call_within_episode; incomplete_calls_excluded_as_episode",
    }


async def score_frozen_record(
    record: dict[str, Any], *, kind: Literal["retrieval", "agent"], scorer: Scorer | None = None,
    policy: MeasurementPolicy | None = None,
) -> dict[str, Any]:
    """参数为 generic/Agent 冻结记录、通道和可选评分器；返回独立评分审计，不运行产品图。"""
    if kind not in {"retrieval", "agent"}:
        raise ValueError("RAGAS kind must be retrieval or agent")
    frozen = deepcopy(record)
    case, run = record_contracts(frozen)
    validate_execution_input(read_snapshot(run.input_snapshot))
    if kind == "retrieval" and run.run_mode != "retrieval_live":
        raise ValueError("retrieval RAGAS requires a frozen retrieval_live run")
    if kind == "agent" and run.run_mode not in {"agent_live", "agent_fixture", "replay"}:
        raise ValueError("agent RAGAS requires a frozen Agent run")
    enabled = policy.ragas_metrics if policy is not None else tuple(METRIC_VARIANTS)
    jobs = _jobs(grading_projection(case, run), kind, retrieval_only=policy is not None)
    initialization_error = None
    if scorer is None and any(job["result"] is None for job in jobs):
        try:
            from evals.ragas.frozen_judge import ProjectRagasScorer

            scorer = ProjectRagasScorer(enabled_metrics=enabled)
        except Exception as exc:
            initialization_error = {"type": type(exc).__name__, "message": str(exc)}
            logger.error("【RAGAS】【初始化】judge 初始化失败 error_type=%s", type(exc).__name__)
    samples = []
    for job in jobs:
        if initialization_error is not None and job["result"] is None:
            result = _unmeasured(job["metric"], "judge_initialization_failed")
            samples.append({**job, "result": result.model_dump(mode="json"), "audit": {"error": initialization_error}})
            continue
        if scorer is None:
            samples.append({**job, "audit": {"status": "not_called"}})
        else:
            samples.append(await _score_job(job, scorer))
    measurements = [_aggregate(name, samples) for name in enabled]
    version = RETRIEVAL_GRADER_VERSION if policy is not None else GRADER_VERSION
    grading = build_grading(frozen, measurements, grader="ragas_frozen", grader_version=version)
    grading_error = _grading_error(samples)
    return {"case_id": case.case_id, "run_id": run.run_id, "grader_version": version,
            "measurement_policy": policy.model_dump(mode="json") if policy is not None else None,
            "ragas_version": RAGAS_VERSION, "kind": kind,
            "grading_status": "failed" if grading_error else "completed", "grading_error": grading_error,
            "grading_result": grading,
            "measurements": grading["measurements"], "samples": samples,
            "judge": deepcopy(getattr(scorer, "metadata", {"status": "not_called"})),
            "aggregation": "equal_weight_per_call_within_episode; incomplete_calls_excluded_as_episode"}
