"""结构线索单元的正文核实按计划索取测试（U4 切片 2，09-12 卡）。

覆盖点：
- 计划没索取正文核实时整轮向量检索调用计数为 0，且不因此失败：线索作已知事实进推理
- 计数落在真实调用点上（产线直接正文适配器 + 向量库替身的 hybrid_search），不看日志
- 索取核实时该计数 > 0，核实通过后线索升级成可引用主张，验证门行为不变
- 实时严格合同保留 evidence_granularity / verify_with_direct_text，不做字段清洗

作者：xxx
创建时间：2026-09-15 00:00:00
"""

from __future__ import annotations

from o2_evidence_test_utils import bind_publication, scope as publication_scope

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any
from unittest.mock import Mock

import pytest

from app.models.research_planner_contracts import (
    ResearchClaimDraft,
)
from app.nodes import rag_node
from app.services.internal_research.research_planner_pipeline import (
    _adapters_for_plan,
    execute_planner_state,
)
from app.models.research_planner_draft_contracts import ResearchClaimStrategyDraftBatch
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit

_PAPER = "paper-a"
_PARENT = "parent_chunk"
_METHOD_NODE_ID = "entity_method_1"


class _StubVectorStore:
    """只记向量检索次数的向量库替身：命中固定一条，回读父块本轮用不到。"""

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits
        self.hybrid_search = Mock(side_effect=lambda *args, **kwargs: deepcopy(self._hits))

    def for_user(self, _user_id: str) -> "_StubVectorStore":
        return self

    def for_snapshot(self, _snapshot: dict | None) -> "_StubVectorStore":
        return self


class _StubGraphService:
    """返回一条固定关系边的图服务替身，供结构线索适配器取到非空线索。"""

    async def query_project_subgraph(self, _paper_ids: list[str]) -> dict[str, Any]:
        return {
            "papers": [{"paper_id": _PAPER, "title": "PDFormer"}],
            "nodes": [{"id": _METHOD_NODE_ID, "label": "Method", "name": "Graph Attention Network"}],
            "edges": [{"source": f"paper:{_PAPER}", "target": _METHOD_NODE_ID, "type": "USES_METHOD"}],
            "meta": {"paper_count": 1, "node_count": 1, "edge_count": 1},
        }


def _hit() -> dict[str, Any]:
    return {
        "paper_id": _PAPER,
        "chunk_id": "v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0",
        "text": "graph attention in method", "full_text": "graph attention in method",
        "parent_id": "p-c1",
        "section": "Method",
        "preview": "graph attention in method",
        "score": 0.5,
    }


def _state() -> dict[str, Any]:
    scope = {
        "user_id": "u1",
        "local_retrieval_allowed": True,
        "paper_ids": [_PAPER],
        "filter_expr": f'paper_id in ["{_PAPER}"]',
    }
    return {
        "structured_query": {"project_scope": scope},
        "filter_expr": "",
        "project_scope": publication_scope(),
        "focus": {"turn_focus": [_PAPER]},
    }


def test_structure_question_without_verification_makes_no_vector_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, store = _structure_turn(monkeypatch, verify_with_direct_text=False)

    # 1. 计划没索取正文核实：整轮一次向量检索都不发，补证阶段也不替计划改主意去补
    assert store.hybrid_search.call_count == 0
    # 2. 不发检索不等于这一轮失败：线索作为已知事实进推理，单元如实出结果
    assert payload.get("error_code") is None
    assert payload["planner_loop_decision"] != "terminate_fail"
    result = payload["research_unit_results"][0]
    assert result["status"] == "evidence_found"
    # 3. 线索不可引用，主张最高只到推断级
    assert [claim["claim_type"] for claim in result["claims"]] == ["evidence_grounded_inference"]
    assert all(record["usable_as_citation"] is False for record in payload["evidence_records"])


def test_structure_question_asking_for_verification_does_call_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, store = _structure_turn(monkeypatch, verify_with_direct_text=True)

    # 索取正文核实的单元照旧多发检索，核实通过后线索升级成可引用主张，验证门行为不变
    assert store.hybrid_search.call_count > 0
    result = payload["research_unit_results"][0]
    assert result["status"] == "evidence_found"
    assert [claim["claim_type"] for claim in result["claims"]] == ["paper_supported"]


def test_schema_keeps_evidence_granularity_and_verification_fields() -> None:
    """无参数；实时严格合同保留明确的正文粒度和核实请求，不经字段清洗。"""
    from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft
    from research_planner_strategy_test_utils import strategy_fixture
    unit = make_unit(source_type="local_structure_hint", allowed_claims=["evidence_grounded_inference"])
    unit["evidence_policy"]["verify_with_direct_text"] = True
    strategy = strategy_fixture(make_draft([unit]))
    parsed = ResearchPlanStrategyDraft.model_validate(strategy, strict=True)
    assert parsed.units[0].evidence_policy.verify_with_direct_text is True
    direct = make_unit()
    direct["evidence_policy"]["evidence_granularity"] = _PARENT
    parsed = ResearchPlanStrategyDraft.model_validate(strategy_fixture(make_draft([direct])), strict=True)
    assert parsed.units[0].evidence_policy.evidence_granularity == _PARENT


def _structure_turn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verify_with_direct_text: bool,
) -> tuple[dict[str, Any], _StubVectorStore]:
    """跑一整轮结构线索取证，返回 (执行载荷, 向量库替身)。

    向量检索计数落在真实调用点上：直接正文适配器用的是产线那一份，命中都由
    store.hybrid_search 产出，所以计数为 0 就是真的一跳都没发。

    Args:
        monkeypatch: pytest 的打桩夹具
        verify_with_direct_text: 计划是否在该单元上索取正文核实
    Returns:
        tuple: (execute_planner_state 的返回载荷, 向量库替身)
    """
    bind_publication(monkeypatch, [_hit()])
    store = _StubVectorStore([_hit()])
    monkeypatch.setattr(rag_node, "get_vector_store", Mock(return_value=store))
    monkeypatch.setattr(rag_node.paper_db, "get_paper", Mock(return_value={"paper_id": _PAPER}))
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: _StubGraphService())
    state = _state()
    plan = _structure_plan(verify_with_direct_text=verify_with_direct_text)
    adapters = _adapters_for_plan(
        plan,
        {"local_direct_text": partial(rag_node._planner_direct_text_records, state)},
        rag_node._planner_optional_adapters(state),
    )
    payload = asyncio.run(execute_planner_state(
        {**state, "run_id": "test-run", "user_query": plan.question, "research_plan": plan.model_dump(mode="json")},
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))
    return payload, store


def _structure_plan(*, verify_with_direct_text: bool) -> Any:
    """拼一个单结构线索单元的计划，核实开关按参数写在单元上。"""
    unit = make_unit(
        "hint",
        goal="Verify the graph relation between the paper and its method",
        action="query the research object graph for relation triples",
        source_type="local_structure_hint",
        allowed_claims=["paper_supported"] if verify_with_direct_text else ["evidence_grounded_inference"],
        paper_ids=[_PAPER],
    )
    unit["claim_policy"]["citation_required"] = False
    unit["evidence_policy"]["verify_with_direct_text"] = verify_with_direct_text
    result = validate_strategy_fixture(make_draft([unit], explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    return result.plan


@dataclass
class _Settings:
    rag_research_planner_max_units: int = 8


@dataclass
class _FakeLLM:
    def with_structured_output(self, model: Any) -> "_StructuredOutput":
        return _StructuredOutput(model=model)


@dataclass
class _StructuredOutput:
    model: Any

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self.model is not ResearchClaimStrategyDraftBatch:
            raise AssertionError(f"unexpected structured output model: {self.model}")
        # 模型一律先起草可引用主张，能不能站住由确定性校验判：证据不可引用时会被降成推断级，
        # 这条降级路径正是「线索不索取核实」时该走的，桩掉它就测不出真实终态
        return ResearchClaimStrategyDraftBatch.model_validate({"claims": [
            ResearchClaimDraft(
                text="Evidence supports this unit.",
                claim_type="paper_supported",
                supporting_evidence_ids=_evidence_ids_in_prompt(messages),
            ).model_dump()
        ]})


def _evidence_ids_in_prompt(messages: list[Any]) -> list[str]:
    """从 claim drafting 提示词里解析出在场证据的 id 行，模拟只引用在场证据的模型行为。

    Args:
        messages: 传给 claim drafting 的消息列表
    Returns:
        list[str]: 提示词里出现过的证据 ID，保序去重
    """
    text = "\n".join(str(getattr(message, "content", "")) for message in messages)
    ids: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("id: "):
            continue
        evidence_id = stripped[len("id: "):].strip()
        if evidence_id and evidence_id not in ids:
            ids.append(evidence_id)
    return ids
