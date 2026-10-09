"""结构线索验证那一跳的时间上限与失败面（U3 审查第 2 项）。

覆盖点：
- 验证用的正文适配器睡死时被时间上限掐断，不在验证跳里永久挂住
- 掐断后的失败面记回计划里点的 local_structure_hint（不是被改面后的 local_direct_text），
  所以走软失败：该单元 blocked、四态记 failed、整轮不 terminate_fail
- 验证跳与其余取证共用同一条受控入口，基础设施故障照配置重试

本文件的单元都在计划里写了 verify_with_direct_text=true：验证跳只在计划索取时才发生，
不索取的单元压根不进验证跳，也就没有超时与重试可谈。

作者：xxx
创建时间：2026-09-15 00:00:00
"""

from __future__ import annotations

from o2_evidence_test_utils import direct_evidence, read_result, scope as publication_scope

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from app.config import get_settings
from app.models import evidence_layers as layers
from app.models.knowledge_access_contracts import EvidenceReadResult
from app.models.research_planner_contracts import ResearchUnit
from app.services.internal_research.research_planner_executor import (
    PlannerExecutionContext,
    PlannerUnitExecutor,
)
from app.services.internal_research.research_planner_pipeline import execute_planner_state
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit

_HINT_SOURCE = layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT
_DIRECT_SOURCE = layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT
_PAPER = "paper-a"
_TIMEOUT_REASON = "取证调用超过 0.05 秒未返回"


def _patch_retry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retries: int = 0,
    timeout: float = 0.05,
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


def _hint_adapter():
    """结构线索适配器替身：照图谱适配器的真实形状返回一条不可引用的线索。"""
    async def run(_unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        return read_result([{
            "evidence_id": "g-a",
            "source": "neo4j",
            "layer": "structure_hint",
            "paper_id": _PAPER,
            "summary": "graph relation",
            "usable_as_citation": False,
            "verification_status": "unverified",
        }])

    return run


def _hanging_direct_adapter(calls: list[str]):
    """睡过时间上限的正文适配器替身，用来证明验证跳真被掐断。"""
    async def run(unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        calls.append(unit.unit_id)
        await asyncio.sleep(5.0)
        return read_result([])

    return run


def _flaky_direct_adapter(calls: list[str]):
    """先抛一次连接故障、再返回可引用正文的替身，用来证明验证跳也走重试。"""
    async def run(unit: ResearchUnit, _context: PlannerExecutionContext) -> EvidenceReadResult:
        calls.append(unit.unit_id)
        if len(calls) == 1:
            raise OSError("connection refused")
        return read_result([direct_evidence("verification", paper_id=_PAPER)])

    return run


def _structure_unit() -> dict[str, Any]:
    unit = make_unit(
        "graph-relation",
        source_type=_HINT_SOURCE,
        allowed_claims=["evidence_grounded_inference"],
        paper_ids=[_PAPER],
        facets=["f1"],
    )
    # 本文件盯的就是验证那一跳，计划得先索取正文验证，否则单元压根不发这次调用
    unit["evidence_policy"]["verify_with_direct_text"] = True
    return unit


def _plan() -> Any:
    result = validate_strategy_fixture(make_draft([_structure_unit()], explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    return result.plan


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="How do the papers use attention?")


def test_verification_hop_is_cut_off_by_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证用的正文适配器睡死时被时间上限掐断：不真等它睡完，单元只落 blocked。"""
    _patch_retry(monkeypatch)
    calls: list[str] = []
    executor = PlannerUnitExecutor(adapters={
        _HINT_SOURCE: _hint_adapter(),
        _DIRECT_SOURCE: _hanging_direct_adapter(calls),
    })

    started = time.monotonic()
    report = asyncio.run(executor.execute_with_evidence(_plan(), _ctx()))
    elapsed = time.monotonic() - started

    # 1. 掐断真发生了：适配器睡 5 秒，整轮远早于此结束
    assert elapsed < 1.0
    assert len(calls) == 1
    # 2. 失败面记回计划里点的辅助能力，所以是软失败：单元 blocked、整轮没抛硬失败
    assert report.results[0].status == "blocked"
    assert report.results[0].unresolved_reason == f"TimeoutError: {_TIMEOUT_REASON}"
    assert report.capability_results == {_HINT_SOURCE: "failed"}


def test_verification_timeout_does_not_terminate_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证跳超时按辅助来源记 failed，整轮照常往下走：不写 planner_execution_failed、决策不 terminate_fail。"""
    _patch_retry(monkeypatch)
    calls: list[str] = []
    plan = _plan()

    payload = asyncio.run(execute_planner_state(
        {"run_id": "test-run", "project_scope": publication_scope(),
         "user_query": plan.question, "research_plan": plan.model_dump(mode="json")},
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters={
            _HINT_SOURCE: _hint_adapter(),
            _DIRECT_SOURCE: _hanging_direct_adapter(calls),
        },
    ))

    assert payload.get("error_code") is None
    assert payload["planner_loop_decision"] != "terminate_fail"
    assert payload["planner_evidence_capabilities"] == {_HINT_SOURCE: "failed"}
    assert payload["research_unit_results"][0]["status"] == "blocked"
    assert payload["research_unit_results"][0]["unresolved_reason"] == f"TimeoutError: {_TIMEOUT_REASON}"


def test_verification_hop_retries_infrastructure_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证跳与其余取证共用受控入口：正文侧的连接故障照配置重试，第二次成功即接着验证。"""
    _patch_retry(monkeypatch, retries=2, timeout=5.0)
    calls: list[str] = []
    executor = PlannerUnitExecutor(adapters={
        _HINT_SOURCE: _hint_adapter(),
        _DIRECT_SOURCE: _flaky_direct_adapter(calls),
    })

    report = asyncio.run(executor.execute_with_evidence(_plan(), _ctx()))

    # 1. 重试真发生在验证这一跳上，第二次拿到了正文
    assert len(calls) == 2
    # 2. 线索取证本身仍记 success：验证跳的重试不改写它的四态
    assert report.capability_results == {_HINT_SOURCE: "success"}
    # 3. 正文到位后走既有验证门，不再是 blocked
    assert report.results[0].status != "blocked"


@dataclass
class _Settings:
    rag_research_planner_max_units: int = 8


@dataclass
class _FakeLLM:
    def with_structured_output(self, model: Any) -> Any:
        raise AssertionError("claim drafting must not run in these tests")
