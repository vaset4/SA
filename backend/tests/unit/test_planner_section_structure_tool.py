"""研究规划器章节结构取证适配器测试（U3 切片 1，09-12 卡）。

覆盖点：
- `local_section_structure` 进严格枚举：带该 source_type 的草案过既有校验，不再落非法值
- 适配器按单元绑定论文 ID 逐篇走 `list_section_paths` 窄查询，不连真实 Milvus
- 返回的章节结构记录不被 `_filter_records` 过滤，有章节的单元结果不是 insufficient_evidence
- 论文零章节是如实的空：该篇不产记录、不抛错
- 计划没选该 source_type 时可选适配器不被启用，向量库零调用

作者：xxx
创建时间：2026-09-14 00:00:00
"""

from __future__ import annotations

from o2_evidence_test_utils import read_result

import asyncio
from typing import Any

import pytest

from app.models import evidence_layers as layers
from app.models.research_planner_contracts import ResearchClaimDraft
from app.nodes import rag_node
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilityError,
    PlannerExecutionContext,
    PlannerUnitExecutor,
)
from app.services.internal_research.research_planner_pipeline import _adapters_for_plan
from app.services.internal_research.research_planner_reasoner import ResearchUnitReasoner
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit

_SOURCE = layers.PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE
_PAPER_A = "paper-a"
_PAPER_B = "paper-b"


class _StubVectorStore:
    """按 paper_id 返回预置章节路径的向量库替身，记录每次窄查询与被绑定的读取身份。"""

    def __init__(self, sections: dict[str, list[str]]) -> None:
        self._sections = sections
        self.queried: list[str] = []
        self.bound_user_ids: list[str] = []
        self.bound_snapshots: list[dict | None] = []

    def for_user(self, user_id: str) -> "_StubVectorStore":
        # 留痕本轮绑进来的身份，供断言取证读库前确实带上了用户；替身无需真副本，仍返回自己
        self.bound_user_ids.append(user_id)
        return self

    def for_snapshot(self, snapshot: dict | None) -> "_StubVectorStore":
        # 留痕本轮盖章的发布快照，供断言直读沿用了同一版本；替身无需真副本，仍返回自己
        self.bound_snapshots.append(snapshot)
        return self

    def list_section_paths(self, paper_id: str) -> list[str]:
        self.queried.append(paper_id)
        return self._sections.get(paper_id, [])


class _StubReasoner:
    """把过滤后的证据原样编成一条推断草案，记录每次拿到的证据 ID。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, unit, evidence, dependency_results, *, read_reports=None, prepared_input=None) -> list[ResearchClaimDraft]:
        """参数为实际证据及已准备输入；返回本测试的固定草案，不调用模型。"""
        evidence_ids = [str(record.get("evidence_id")) for record in evidence]
        self.calls.append(evidence_ids)
        return [
            ResearchClaimDraft(
                text="The target paper section list is available.",
                claim_type="evidence_grounded_inference",
                supporting_evidence_ids=evidence_ids,
                reasoning_summary="Section structure came from the vector store narrow query.",
            )
        ]


def _state(paper_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "user_query": "这篇论文有哪些章节",
        "focus": {"turn_focus": paper_ids or []},
        "project_scope": {"user_id": "u1", "paper_ids": paper_ids or [], "publication": {
            "user_id": "u1", "lease_id": "fixture-lease", "batch_ids": ["b" * 32],
            "paper_batches": {paper_id: "b" * 32 for paper_id in (paper_ids or [])},
        }},
    }


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="这篇论文有哪些章节")


def _section_unit(*, paper_ids: list[str], facets: list[str] | None = None) -> dict[str, Any]:
    unit = make_unit(
        "sec",
        goal="List the section structure of the target paper",
        action="read the target paper section list",
        source_type=_SOURCE,
        allowed_claims=["evidence_grounded_inference"],
        paper_ids=paper_ids,
        facets=facets or ["f1"],
    )
    unit["claim_policy"]["citation_required"] = False
    return unit


def _metadata_unit() -> dict[str, Any]:
    unit = make_unit(
        "meta",
        source_type="local_metadata",
        allowed_claims=["evidence_grounded_inference"],
        paper_ids=[_PAPER_A],
        facets=["f1"],
    )
    unit["claim_policy"]["citation_required"] = False
    return unit


def _plan(units: list[dict[str, Any]]) -> Any:
    result = validate_strategy_fixture(make_draft(units, explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    return result.plan


def _metadata_adapter():
    async def run(unit, context) -> list[dict[str, Any]]:
        return read_result([
            {
                "evidence_id": f"metadata-{_PAPER_A}",
                "source": layers.EVIDENCE_SOURCE_INVENTORY,
                "layer": layers.EVIDENCE_LAYER_METADATA,
                "paper_id": _PAPER_A,
                "summary": "Paper A",
                "usable_as_citation": False,
            }
        ])

    return run


def test_section_structure_source_type_passes_existing_validation() -> None:
    result = validate_strategy_fixture(make_draft([_section_unit(paper_ids=[_PAPER_A])]))

    assert result.failure is None
    assert result.plan is not None
    assert result.plan.units[0].evidence_policy.source_type == _SOURCE


def test_section_structure_unit_cannot_claim_final_answer_role() -> None:
    unit = _section_unit(paper_ids=[_PAPER_A])
    unit["output_role"] = "final_recommendation"

    result = validate_strategy_fixture(make_draft([unit], explicit_facets=[make_facet("f1")]))

    # 1. 只出章节名单、出不了可引用证据的单元，不能自称产出最终答案的单元；
    #    这道门与 local_metadata 等既有本地来源同一套，漏了新类型就等于开了个口子
    assert result.failure is not None
    assert result.failure.code == "invalid_plan_format"
    assert "output_role" in result.failure.message


def test_section_structure_unit_defaults_to_evidence_role() -> None:
    unit = _section_unit(paper_ids=[_PAPER_A])
    unit.pop("output_role", None)

    result = validate_strategy_fixture(make_draft([unit], explicit_facets=[make_facet("f1")]))

    # 2. 模型漏给 output_role 时按本地取证单元补成 evidence，与既有本地来源一致
    assert result.failure is None
    assert result.plan.units[0].output_role == "evidence"


def test_section_structure_read_carries_the_round_publication_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """章节名单属直读取证，必须盖本轮发布快照，不能各查各的「当前最新版本」。"""
    store = _StubVectorStore({_PAPER_A: ["1 Introduction"]})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    snapshot = {
        "lease_id": "lease-1",
        "user_id": "u1",
        "batch_ids": ["batch-a"],
        "paper_batches": {_PAPER_A: "batch-a"},
    }
    state = {**_state([_PAPER_A]), "project_scope": {
        "user_id": "u1", "paper_ids": [_PAPER_A], "publication": snapshot}}
    plan = _plan([_section_unit(paper_ids=[_PAPER_A])])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(state))

    asyncio.run(
        PlannerUnitExecutor(
            adapters=adapters,
            reasoner=ResearchUnitReasoner(draft_provider=_StubReasoner()),
        ).execute_with_evidence(plan, _ctx())
    )

    # 快照与身份盖在同一个句柄上：直读与向量检索读同一版本
    assert store.bound_snapshots == [snapshot]
    assert store.bound_user_ids == ["u1"]


def test_section_structure_missing_snapshot_rejects_before_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为 patch；正式目录读取缺发布快照时不能转读当前版本。"""
    store = _StubVectorStore({_PAPER_A: ["1 Introduction"]})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    state = _state([_PAPER_A])
    del state["project_scope"]["publication"]
    plan = _plan([_section_unit(paper_ids=[_PAPER_A])])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(state))
    with pytest.raises(EvidenceCapabilityError, match="发布快照"):
        asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))
    assert store.bound_snapshots == []
    assert store.bound_user_ids == []


def test_section_structure_valid_empty_snapshot_stays_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为 patch；合法空发布快照保持固定空范围，不重新读取当前发布。"""
    store = _StubVectorStore({})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    snapshot = {"user_id": "u1", "lease_id": "empty-lease", "paper_batches": {}, "batch_ids": []}
    state = _state([_PAPER_A])
    state["project_scope"]["publication"] = snapshot
    plan = _plan([_section_unit(paper_ids=[_PAPER_A])])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(state))
    report = asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))
    assert store.bound_snapshots == [snapshot]
    assert report.capability_results == {_SOURCE: "empty"}


def test_section_structure_snapshot_reader_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为 patch；发布快照归属与回合身份不同则明确拒绝。"""
    store = _StubVectorStore({})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    state = _state([_PAPER_A])
    state["project_scope"]["publication"]["user_id"] = "another-user"
    plan = _plan([_section_unit(paper_ids=[_PAPER_A])])
    with pytest.raises(EvidenceCapabilityError, match="发布快照"):
        asyncio.run(PlannerUnitExecutor(adapters=rag_node._planner_optional_adapters(state)).execute_with_evidence(plan))
    assert store.bound_snapshots == []


def test_section_structure_records_reach_the_unit_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StubVectorStore({
        _PAPER_A: ["1 Introduction", "2 Method"],
        _PAPER_B: ["1 Introduction"],
    })
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan([_section_unit(paper_ids=[_PAPER_A, _PAPER_B])])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state()))

    assert _SOURCE in adapters
    report = asyncio.run(
        PlannerUnitExecutor(
            adapters=adapters,
            reasoner=ResearchUnitReasoner(draft_provider=_StubReasoner()),
        ).execute_with_evidence(plan, _ctx())
    )

    # 1. 按单元绑定论文 ID 逐篇窄查询，不多发查询也不落到别的论文
    assert store.queried == [_PAPER_A, _PAPER_B]
    # 2. 读库前先绑本轮用户身份：论文 ID 来自计划草案，缺身份仓储层的归属谓词会整条跳过
    assert store.bound_user_ids == ["u1"]
    # 3. 记录过了 _filter_records 才会出现在结果 evidence_ids 里
    result = report.results[0]
    # 4. 【知识取证】【定位资料】目录可保留用于定位，不能单独支持正文推断
    assert result.status == "insufficient_evidence"
    assert result.evidence_ids == [
        f"section-structure-{_PAPER_A}",
        f"section-structure-{_PAPER_B}",
    ]
    # 4. 记录形状与非引用身份
    record = next(
        item for item in report.evidence_records
        if item["evidence_id"] == f"section-structure-{_PAPER_A}"
    )
    assert record["source"] == layers.EVIDENCE_SOURCE_MILVUS
    assert record["layer"] == layers.EVIDENCE_LAYER_STRUCTURE_HINT
    assert record["usable_as_citation"] is False
    assert record["summary"] == "1 Introduction | 2 Method"


def test_zero_sections_yield_no_records_and_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StubVectorStore({_PAPER_A: []})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan([_section_unit(paper_ids=[_PAPER_A])])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state()))

    report = asyncio.run(
        PlannerUnitExecutor(
            adapters=adapters,
            reasoner=ResearchUnitReasoner(draft_provider=_StubReasoner()),
        ).execute_with_evidence(plan, _ctx())
    )

    # 查得到但零章节是如实的空：不产记录、不抛错，单元如实报取证不足
    assert store.queried == [_PAPER_A]
    assert report.evidence_records == []
    assert report.results[0].status == "insufficient_evidence"
    assert report.results[0].evidence_ids == []


def test_missing_reader_identity_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _StubVectorStore({_PAPER_A: ["1 Introduction"]})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan([_section_unit(paper_ids=[_PAPER_A])])
    scopeless = {**_state(), "project_scope": {"paper_ids": [_PAPER_A]}}
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(scopeless))

    # 缺身份时论文归属无从校验：带取证面的失败上抛，不拿计划里的论文 ID 裸读，向量库零查询
    with pytest.raises(EvidenceCapabilityError) as caught:
        asyncio.run(
            PlannerUnitExecutor(
                adapters=adapters,
                reasoner=ResearchUnitReasoner(draft_provider=_StubReasoner()),
            ).execute_with_evidence(plan, _ctx())
        )
    assert store.queried == []
    assert store.bound_user_ids == []
    assert caught.value.source_type == _SOURCE
    assert "读取身份" in caught.value.reason


def test_adapter_not_selected_when_plan_does_not_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StubVectorStore({_PAPER_A: ["1 Introduction"]})
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    plan = _plan([_metadata_unit()])
    # 计划单元的论文详情取数用替身：本用例只盯「没点名的 section_structure 不被启用」，
    # 详情这边不该真去读论文库
    adapters = _adapters_for_plan(
        plan,
        {},
        {**rag_node._planner_optional_adapters(_state()), "local_metadata": _metadata_adapter()},
    )

    assert _SOURCE not in adapters
    report = asyncio.run(
        PlannerUnitExecutor(
            adapters=adapters,
            reasoner=ResearchUnitReasoner(draft_provider=_StubReasoner()),
        ).execute_with_evidence(plan, _ctx())
    )

    assert report.results[0].status == "insufficient_evidence"
    assert store.queried == []
