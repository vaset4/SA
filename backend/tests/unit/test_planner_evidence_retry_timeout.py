"""研究规划器取证调用的重试与超时（U3 切片 6，09-12 卡）。

覆盖点：
- 基础设施故障先按配置重试，次数用尽才 failed；失败带可读原因与失败面
- 第二次尝试成功即记 success，证据记录与不重试时完全一致
- 白名单外的异常不重试，调用计数 = 1
- 适配器超时：wait_for 到点掐断，failed 带可读原因（到点抛的 TimeoutError 消息是空串）
- 图谱面超时走软失败：单元 blocked、整轮不 terminate_fail（超时发生在适配器之外，不能被包成硬失败）
- 软失败子类经重试路径不被洗成硬失败（坑 44 回归锁），可重试的 __cause__ 会被重试
- 实测的白名单覆盖两种 Milvus 到点形状：正常超时 MilvusException、极短超时 RpcError 的子孙

作者：xxx
创建时间：2026-09-14 00:00:00
"""

from __future__ import annotations

from o2_evidence_test_utils import direct_evidence, read_result, scope as publication_scope

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest
from grpc import RpcError
from grpc._channel import _InactiveRpcError
from pymilvus.exceptions import MilvusException

from app.config import get_settings
from app.models import evidence_layers as layers
from app.models.knowledge_access_contracts import EvidenceReadResult
from app.nodes import rag_node
from app.services.graph.neo4j_client import Neo4jDegradedError
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError,
    EvidenceCapabilitySoftError,
    PlannerExecutionContext,
    PlannerUnitExecutor,
    _call_adapter,
    _is_retryable,
)
from app.services.internal_research.research_planner_pipeline import (
    _adapters_for_plan,
    execute_planner_state,
)
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit

_SOURCE = "local_direct_text"
_GRAPH_SOURCE = layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT
_PAPER = "paper-a"
_TIMEOUT_REASON = "取证调用超过 0.05 秒未返回"


def _patch_retry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retries: int = 2,
    timeout: float = 5.0,
) -> None:
    """把重试次数、退避与时间上限压到测试值，退避固定为 0 避免真等。

    Args:
        monkeypatch: pytest 的打补丁夹具
        retries: 额外重试次数
        timeout: 单次调用时间上限（秒）
    """
    cfg = get_settings()
    monkeypatch.setattr(cfg, "rag_planner_evidence_max_retries", retries)
    monkeypatch.setattr(cfg, "rag_planner_evidence_retry_backoff_seconds", 0.0)
    monkeypatch.setattr(cfg, "rag_planner_evidence_timeout_seconds", timeout)


def _stepped_adapter(steps: list[Any], calls: list[str]):
    """按步执行的适配器替身：该步是异常就抛，否则当返回值返回，最后一步可重复。

    Args:
        steps: 每步的动作，异常实例或记录列表
        calls: 调用计数收集器，元素为被调单元的 unit_id
    """
    async def run(unit: Any, _context: PlannerExecutionContext) -> EvidenceReadResult:
        calls.append(unit.unit_id)
        step = steps[min(len(calls) - 1, len(steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return read_result(step)

    return run


def _sleeping_adapter(seconds: float, calls: list[str]):
    """睡过时间上限的适配器替身，用来证明 wait_for 真把它掐断。"""
    async def run(unit: Any, _context: PlannerExecutionContext) -> EvidenceReadResult:
        calls.append(unit.unit_id)
        await asyncio.sleep(seconds)
        return read_result([])

    return run


def _soft_graph_adapter(calls: list[str]):
    """照 rag_node 图谱适配器的真实形状：把已分类的图故障换成软失败信号抛出。"""
    async def run(unit: Any, _context: PlannerExecutionContext) -> EvidenceReadResult:
        calls.append(unit.unit_id)
        try:
            raise Neo4jDegradedError(reason="ServiceUnavailable")
        except Neo4jDegradedError as exc:
            raise EvidenceCapabilitySoftError(
                _GRAPH_SOURCE, f"{type(exc).__name__}: {exc}") from exc

    return run


class _HangingGraphService:
    """查询即睡死的图服务替身，模拟图库查询卡住超过时间上限。"""

    async def query_project_subgraph(self, _paper_ids: list[str]) -> dict[str, Any]:
        await asyncio.sleep(5.0)
        return {"papers": [], "nodes": [], "edges": [], "meta": {}}


def _direct_unit() -> dict[str, Any]:
    return make_unit("a", facets=["f1"])


def _structure_unit() -> dict[str, Any]:
    return make_unit(
        "graph-relation",
        source_type=_GRAPH_SOURCE,
        allowed_claims=["evidence_grounded_inference"],
        paper_ids=[_PAPER],
        facets=["f1"],
    )


def _plan(units: list[dict[str, Any]]) -> Any:
    result = validate_strategy_fixture(make_draft(units, explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    return result.plan


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="How do the papers use attention?")


def _direct_evidence() -> dict[str, Any]:
    """无参数；返回重试前后可逐字段比较的新版正文材料。"""
    return direct_evidence("e1", paper_id=_PAPER)


def _run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def test_retryable_failure_retries_until_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """基础设施故障按配置重试到次数用尽才失败：调用计数 = 次数 + 1，原因与失败面都在。"""
    _patch_retry(monkeypatch, retries=2)
    calls: list[str] = []
    plan = _plan([_direct_unit()])
    executor = PlannerUnitExecutor(
        adapters={_SOURCE: _stepped_adapter([OSError("connection refused")], calls)},
    )

    with pytest.raises(EvidenceCapabilityError) as caught:
        _run(executor.execute_with_evidence(plan, _ctx()))

    assert len(calls) == 3
    assert caught.value.source_type == _SOURCE
    assert caught.value.reason == "OSError: connection refused"
    assert isinstance(caught.value.__cause__, OSError)


def test_retryable_failure_reaches_failed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """重试次数用尽仍失败：四态如实记 failed，整轮走既有硬失败路径，没被重试吞掉。"""
    _patch_retry(monkeypatch, retries=1)
    calls: list[str] = []
    plan = _plan([_direct_unit()])
    executor = PlannerUnitExecutor(
        adapters={_SOURCE: _stepped_adapter([OSError("connection refused")], calls)},
    )

    payload = _run(execute_planner_state(
        {"run_id": "test-run", "project_scope": publication_scope(),
         "user_query": plan.question, "research_plan": plan.model_dump(mode="json")},
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=executor.adapters,
    ))

    assert len(calls) == 2
    assert payload["planner_evidence_capabilities"] == {_SOURCE: "failed"}
    assert payload["error_code"] == "planner_execution_failed"
    assert payload["source_status_reasons"][_SOURCE] == "OSError: connection refused"


def test_retry_success_keeps_records_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """第二次尝试成功记 success，证据记录与不重试时完全一致。"""
    _patch_retry(monkeypatch, retries=2)
    records = [_direct_evidence()]
    retried_calls: list[str] = []
    plain_calls: list[str] = []
    plan = _plan([_direct_unit()])

    retried = _run(PlannerUnitExecutor(
        adapters={_SOURCE: _stepped_adapter([OSError("connection refused"), records], retried_calls)},
    ).execute_with_evidence(plan, _ctx()))
    plain = _run(PlannerUnitExecutor(
        adapters={_SOURCE: _stepped_adapter([records], plain_calls)},
    ).execute_with_evidence(plan, _ctx()))

    assert len(retried_calls) == 2
    assert len(plain_calls) == 1
    assert retried.capability_results == {_SOURCE: "success"}
    assert retried.evidence_records == plain.evidence_records
    assert [result.model_dump(exclude={"metadata"}) for result in retried.results] == [
        result.model_dump(exclude={"metadata"}) for result in plain.results
    ]
    assert retried.results[0].metadata["read_reports"] == plain.results[0].metadata["read_reports"]
    assert retried.results[0].metadata["fulfillment"] == plain.results[0].metadata["fulfillment"]
    assert retried.read_receipts[0]["actual_call_count"] == 2
    assert plain.read_receipts[0]["actual_call_count"] == 1


def test_non_retryable_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """白名单外的异常不是基础设施故障，不重试：调用计数 = 1，原因照旧可读。"""
    _patch_retry(monkeypatch, retries=2)
    calls: list[str] = []
    plan = _plan([_direct_unit()])
    executor = PlannerUnitExecutor(
        adapters={_SOURCE: _stepped_adapter([ValueError("bad payload")], calls)},
    )

    with pytest.raises(EvidenceCapabilityError) as caught:
        _run(executor.execute_with_evidence(plan, _ctx()))

    assert len(calls) == 1
    assert caught.value.reason == "ValueError: bad payload"


def test_adapter_timeout_is_cut_off_with_readable_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """适配器睡过时间上限：wait_for 到点掐断并补可读原因，不真等它睡完。"""
    _patch_retry(monkeypatch, retries=0, timeout=0.05)
    calls: list[str] = []
    plan = _plan([_direct_unit()])
    executor = PlannerUnitExecutor(adapters={_SOURCE: _sleeping_adapter(5.0, calls)})

    started = time.monotonic()
    with pytest.raises(EvidenceCapabilityError) as caught:
        _run(executor.execute_with_evidence(plan, _ctx()))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert len(calls) == 1
    assert caught.value.reason == f"TimeoutError: {_TIMEOUT_REASON}"


def test_adapter_timeout_keeps_its_existing_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为打补丁夹具；无返回值，验证适配器自带超时原因与原异常对象不被改写。"""
    _patch_retry(monkeypatch, retries=0)
    failure = TimeoutError("storage RPC deadline reached")
    calls: list[str] = []
    plan = _plan([_direct_unit()])

    with pytest.raises(EvidenceCapabilityError) as caught:
        _run(_call_adapter(_stepped_adapter([failure], calls), plan.units[0], _ctx()))

    assert caught.value.__cause__ is failure
    assert caught.value.reason == "TimeoutError: storage RPC deadline reached"
    assert len(calls) == 1


def test_graph_timeout_stays_soft_and_turn_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图谱面超时发生在适配器之外，仍按辅助来源包成软失败：单元 blocked、整轮不 terminate_fail。"""
    _patch_retry(monkeypatch, retries=0, timeout=0.05)
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: _HangingGraphService())
    plan = _plan([_structure_unit()])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state()))

    # 1. 到点换成带失败面与可读原因的软失败异常上抛，不是硬失败
    with pytest.raises(EvidenceCapabilitySoftError) as caught:
        _run(_call_adapter(adapters[_GRAPH_SOURCE], plan.units[0], _ctx()))
    assert caught.value.source_type == _GRAPH_SOURCE
    assert caught.value.reason == f"TimeoutError: {_TIMEOUT_REASON}"
    # 2. 执行层消化成 blocked：整轮不写 planner_execution_failed、决策不 terminate_fail
    payload = _run(execute_planner_state(
        {"run_id": "test-run", "project_scope": publication_scope(),
         "user_query": plan.question, "research_plan": plan.model_dump(mode="json")},
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))
    assert payload.get("error_code") is None
    assert payload["planner_loop_decision"] != "terminate_fail"
    assert payload["planner_evidence_capabilities"] == {_GRAPH_SOURCE: "failed"}
    assert payload["research_unit_results"][0]["status"] == "blocked"
    assert payload["research_unit_results"][0]["unresolved_reason"] == f"TimeoutError: {_TIMEOUT_REASON}"


def test_soft_failure_survives_retry_path_and_keeps_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """软失败子类经重试路径原样上抛：不洗成硬失败，也不覆盖 __cause__ 上的原始故障链。"""
    _patch_retry(monkeypatch, retries=2)
    calls: list[str] = []
    plan = _plan([_structure_unit()])
    adapter = _soft_graph_adapter(calls)

    with pytest.raises(EvidenceCapabilitySoftError) as caught:
        _run(_call_adapter(adapter, plan.units[0], _ctx()))

    # 1. __cause__ 命中白名单（Neo4jDegradedError）→ 先重试到次数用尽
    assert len(calls) == 3
    # 2. 最终仍是软失败类原样，失败面不被重包改掉
    assert type(caught.value) is EvidenceCapabilitySoftError
    assert caught.value.source_type == _GRAPH_SOURCE
    # 3. 原始故障链保留在 __cause__ 上，没被自引用覆盖
    assert isinstance(caught.value.__cause__, Neo4jDegradedError)


def test_retryable_whitelist_covers_measured_milvus_shapes() -> None:
    """实测的两种 Milvus 到点形状都在白名单里：正常超时 MilvusException、极短超时 RpcError 的子孙。"""
    assert _is_retryable(MilvusException(message="rpc deadline exceeded"))
    assert issubclass(_InactiveRpcError, RpcError)
    assert _is_retryable(RpcError())
    assert _is_retryable(Neo4jDegradedError(reason="ServiceUnavailable"))
    assert _is_retryable(TimeoutError())
    assert _is_retryable(OSError("connection reset"))


def _state() -> dict[str, Any]:
    return {
        "user_query": "这篇论文用了什么方法",
        "focus": {"turn_focus": [_PAPER]},
        "project_scope": {"user_id": "u1", "paper_ids": [_PAPER]},
    }


@dataclass
class _Settings:
    rag_research_planner_max_units: int = 8


@dataclass
class _FakeLLM:
    def with_structured_output(self, model: Any) -> Any:
        raise AssertionError("claim drafting must not run in these tests")
