"""冻结运行的 RAGAS 指标输入隔离回归。作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from evals.contracts import freeze_record, snapshot
from evals.ragas.frozen_bridge import build_failed_score, score_frozen_record
from evals.suites.policy import workflow_policy


def _event(stage: str, payload: dict[str, Any], *, scope_stage: str = "retrieval") -> dict[str, Any]:
    """参数为阶段、载荷和作用域类型；返回合法的真实观察替身。"""
    return {
        "schema_version": "evidence-observation-v1", "event_id": stage,
        "request_id": "request-1", "stage": stage, "observed_at_ns": 1,
        "scope": {"scope_id": scope_stage, "request_id": "request-1",
                  "parent_scope_id": None, "stage": scope_stage, "unit_id": None},
        "payload": payload,
    }


def _record(*, agent: bool = False, generation: bool = False) -> dict[str, Any]:
    """参数控制检索或 Agent 及生成观察；返回未包含衍生评分的冻结记录。"""
    question = {"question": "模型的实验结果是什么？"}
    observations = {
        "evidence_observations": [_event("query_raw", {"query": "实验结果", "hits": [
            {"text": "实际子块", "full_text": "不得补读的父全文", "rank": 1},
        ]})], "model_call_observations": [], "evidence_capture_errors": [],
    }
    if generation:
        observations["evidence_observations"].append(_event(
            "model_evidence_text", {"source_text": "真实生成摘录", "rendered_text": "引用：真实生成摘录"},
            scope_stage="final_context",
        ))
        observations["model_call_observations"].append({
            "call_id": "final-call", "evidence_scope_id": "final_context", "stage": "final_answer",
            "input_messages": [{"role": "user", "content": "问题；引用：真实生成摘录"}],
            "http_attempt_count": 1, "cache_hit": False, "error": None,
        })
    observed = observations
    if agent:
        observed = {"turns": [{"turn": 1, "input_snapshot": snapshot(question),
                               "values_snapshot": snapshot({"answer": "真实回答"}),
                               "assistant_output": "真实回答", "status": "completed",
                               "evidence_observation_snapshot": snapshot(observations)}],
                    "status": "completed"}
    return freeze_record({"id": "case-1", **question, "reference_answer": "原文参考答案"},
                         mode="agent_live" if agent else "retrieval_live",
                         execution_input=question, observation=observed)


class _Scorer:
    """记录真实传入评分器的数据，返回注入测试分数。"""

    metadata = {"provider": "test", "model": "test-judge", "ragas_version": "0.4.3"}

    def __init__(self) -> None:
        """无参数；初始化独立调用列表。"""
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为指标和 RAGAS 样本；记录副本后返回分数与用量缺失事实。"""
        self.calls.append((metric, deepcopy(sample)))
        return {"value": 0.5, "calls": [], "usage_status": "missing"}


def _metrics(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """参数为评分产物；返回以指标名索引的测量。"""
    return {item["name"]: item for item in result["measurements"]}


def test_failed_frozen_score_keeps_expected_samples_and_product_record() -> None:
    """无参数；judge失败只形成独立缺失评分，冻结产品记录和策略作业仍可重建。"""
    record = _record()
    original = deepcopy(record)
    result = build_failed_score(record, kind="retrieval", policy=workflow_policy(), error="judge unavailable")
    assert record == original
    assert result["grading_status"] == "failed"
    assert result["grading_error"] == "judge unavailable"
    assert {sample["metric"] for sample in result["samples"]} == {"context_precision", "context_recall"}
    assert all(sample["result"]["status"] == "missing" for sample in result["samples"])


@pytest.mark.asyncio
async def test_retrieval_scores_without_response_and_never_reads_full_text() -> None:
    """纯检索只需参考答案和实际子块；评分不能修改被冻结的运行。"""
    record, scorer = _record(), _Scorer()
    original = deepcopy(record)
    result = await score_frozen_record(record, kind="retrieval", scorer=scorer)
    assert record == original
    assert {name for name, _ in scorer.calls} == {"context_precision", "context_recall"}
    assert all(sample["retrieved_contexts"] == ["实际子块"] for _, sample in scorer.calls)
    assert all("response" not in sample for _, sample in scorer.calls)
    assert _metrics(result)["faithfulness"]["status"] == "not_measured"
    assert _metrics(result)["factual_correctness"]["status"] == "not_measured"
    assert result["grading_result"]["run_id"] == record["run_record"]["run_id"]


@pytest.mark.asyncio
async def test_agent_uses_exact_final_context_and_no_truth_for_faithfulness() -> None:
    """忠实度只收到真实生成摘录，不带参考答案或检索全集。"""
    scorer = _Scorer()
    result = await score_frozen_record(_record(agent=True, generation=True), kind="agent", scorer=scorer)
    samples = dict(scorer.calls)
    assert samples["faithfulness"] == {"user_input": "模型的实验结果是什么？", "response": "真实回答",
                                       "retrieved_contexts": ["真实生成摘录"]}
    assert samples["factual_correctness"] == {"response": "真实回答", "reference": "原文参考答案"}
    assert all(item["status"] == "measured" for item in _metrics(result).values())


@pytest.mark.asyncio
async def test_missing_generation_context_does_not_block_factual_correctness() -> None:
    """没有最终生成调用时忠实度缺失，事实对照仍可独立评分。"""
    scorer = _Scorer()
    result = await score_frozen_record(_record(agent=True), kind="agent", scorer=scorer)
    assert _metrics(result)["faithfulness"]["status"] == "missing"
    assert _metrics(result)["factual_correctness"]["status"] == "measured"
    assert "faithfulness" not in dict(scorer.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_empty_retrieval_is_not_missing_observation(missing: bool) -> None:
    """显式空召回传递空列表，观察缺失不得生成空列表凑分。"""
    record, scorer = _record(), _Scorer()
    observed = json.loads(record["run_record"]["observation_snapshot"])
    if missing:
        observed.pop("evidence_observations")
    else:
        observed["evidence_observations"][0]["payload"]["hits"] = []
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="retrieval", scorer=scorer)
    if missing:
        assert not scorer.calls
        assert _metrics(result)["context_recall"]["status"] == "missing"
    else:
        assert all(sample["retrieved_contexts"] == [] for _, sample in scorer.calls)
        assert _metrics(result)["context_recall"]["status"] == "measured"


@pytest.mark.asyncio
async def test_judge_failure_is_missing_and_keeps_original_run() -> None:
    """评分服务失败只产生缺失测量及错误审计，不覆盖产品回答。"""
    async def fail(metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为指标与样本；抛出可复现的评测故障。"""
        raise RuntimeError("judge unavailable")

    record = _record()
    original = deepcopy(record)
    result = await score_frozen_record(record, kind="retrieval", scorer=fail)
    assert record == original
    assert _metrics(result)["context_precision"]["status"] == "missing"
    assert result["samples"][0]["audit"]["error"]["type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_malformed_context_is_invalid_without_judge_calls() -> None:
    """结构化元数据不能转换成证据正文。"""
    record, scorer = _record(), _Scorer()
    observed = json.loads(record["run_record"]["observation_snapshot"])
    observed["evidence_observations"][0]["payload"]["hits"][0]["text"] = {"reference": "答案"}
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="retrieval", scorer=scorer)
    assert not scorer.calls
    assert _metrics(result)["context_precision"]["status"] == "invalid_input"


@pytest.mark.asyncio
async def test_negative_case_has_no_retrieval_or_factual_denominator() -> None:
    """资料不足案例不制造召回及事实对照分母，真实生成仍可独立检查忠实度。"""
    record, scorer = _record(agent=True, generation=True), _Scorer()
    truth = json.loads(record["eval_case"]["truth_snapshot"])
    record["eval_case"]["truth_snapshot"] = snapshot({**truth, "answerability": "insufficient_evidence"})
    result = await score_frozen_record(record, kind="agent", scorer=scorer)
    assert list(dict(scorer.calls)) == ["faithfulness"]
    assert _metrics(result)["context_recall"]["status"] == "not_measured"
    assert _metrics(result)["factual_correctness"]["status"] == "not_measured"


@pytest.mark.asyncio
async def test_final_text_missing_from_actual_prompt_cannot_score_faithfulness() -> None:
    """证据事件存在但文本未进入实际模型请求时，不能用它替代生成上下文。"""
    record, scorer = _record(agent=True, generation=True), _Scorer()
    observed = json.loads(record["run_record"]["observation_snapshot"])
    captured = json.loads(observed["turns"][0]["evidence_observation_snapshot"])
    captured["model_call_observations"][0]["input_messages"][0]["content"] = "不包含目标证据的真实提示词"
    observed["turns"][0]["evidence_observation_snapshot"] = snapshot(captured)
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="agent", scorer=scorer)
    assert "faithfulness" not in dict(scorer.calls)
    assert _metrics(result)["faithfulness"]["status"] == "missing"


@pytest.mark.asyncio
async def test_partial_query_failure_cannot_be_hidden_by_successful_mean() -> None:
    """第二次查询缺失时保留成功子样本，但 episode 总测量不得只平均成功调用。"""
    record, scorer = _record(), _Scorer()
    observed = json.loads(record["run_record"]["observation_snapshot"])
    missing = _event("query_raw", {"query": "第二次查询"})
    missing["event_id"] = "query-missing"
    observed["evidence_observations"].append(missing)
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="retrieval", scorer=scorer)
    assert _metrics(result)["context_precision"]["status"] == "missing"
    assert any(row["result"]["status"] == "measured" for row in result["samples"])


@pytest.mark.asyncio
async def test_scorer_cannot_mutate_frozen_input_or_forge_observation() -> None:
    """恶意评分器只获得独立最小样本，额外输出不得变成产品观察。"""
    async def mutate(metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为指标和样本；修改临时样本并返回伪造回答。"""
        sample["retrieved_contexts"].append("伪造文本")
        return {"value": 0.1, "response": "篡改回答"}

    record = _record()
    original = deepcopy(record)
    result = await score_frozen_record(record, kind="retrieval", scorer=mutate)
    assert original == record
    assert all("伪造文本" not in row["input_snapshot"] for row in result["samples"])


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1.1])
async def test_invalid_judge_scores_stay_invalid_and_json_serializable(value: Any) -> None:
    """无效数字不能成为分数，审计仍需能保存为严格 JSON。"""
    async def invalid(metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为指标和样本；返回待验证的错误值。"""
        return {"value": value}

    result = await score_frozen_record(_record(), kind="retrieval", scorer=invalid)
    assert _metrics(result)["context_recall"]["status"] == "invalid_input"
    json.dumps(result, allow_nan=False)


@pytest.mark.asyncio
async def test_agent_user_query_and_each_turn_question_are_preserved() -> None:
    """Agent 实际入口使用 user_query 时仍可评分，每轮问题与子查询均有出处。"""
    record, scorer = _record(agent=True), _Scorer()
    observed = json.loads(record["run_record"]["observation_snapshot"])
    observed["turns"][0]["input_snapshot"] = snapshot({"user_query": "用户本轮真实问题"})
    record["run_record"]["observation_snapshot"] = snapshot(observed)
    result = await score_frozen_record(record, kind="agent", scorer=scorer)
    assert dict(scorer.calls)["context_recall"]["user_input"] == "用户本轮真实问题"
    assert result["samples"][0]["retrieval_query"] == "实验结果"


@pytest.mark.asyncio
async def test_missing_reference_only_blocks_reference_metrics() -> None:
    """没有参考答案只影响依赖真值的指标，忠实度依旧使用真实生成证据。"""
    record, scorer = _record(agent=True, generation=True), _Scorer()
    record["eval_case"]["truth_snapshot"] = snapshot({})
    result = await score_frozen_record(record, kind="agent", scorer=scorer)
    assert list(dict(scorer.calls)) == ["faithfulness"]
    assert _metrics(result)["context_recall"]["reason"] == "reference_missing"


@pytest.mark.asyncio
async def test_absent_answer_is_missing_but_real_empty_answer_is_not_applicable() -> None:
    """缺失生成输出与真实空输出保留不同测量状态，均不填充替代回答。"""
    for answer, expected in ((None, "missing"), ("", "not_measured")):
        record, scorer = _record(agent=True, generation=True), _Scorer()
        observed = json.loads(record["run_record"]["observation_snapshot"])
        observed["turns"][0]["values_snapshot"] = snapshot({"answer": answer})
        record["run_record"]["observation_snapshot"] = snapshot(observed)
        result = await score_frozen_record(record, kind="agent", scorer=scorer)
        assert _metrics(result)["factual_correctness"]["status"] == expected
        assert "factual_correctness" not in dict(scorer.calls)
