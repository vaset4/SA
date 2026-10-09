"""研究规划器取图谱关系三元组适配器测试（U3 切片 5，09-12 卡）。

覆盖点：
- 按目标论文真查研究对象图（桩掉图服务），不再读主链上恒为空的 `graph_evidence`
- 每条边一条三元组记录：端点在文案前映射成人可读名字，节点 id 不裸奔
- 记录不可引用（`usable_as_citation=False`）、不进引用池，能过 `_filter_records`
- 证据 ID 基于边内容确定性生成，同一关系重复查到仍是同一个 ID
- 超条数上限：取前 N 条 + 截断概览（含命中总数），概览不可引用
- 四态各可达：有边 success / 零边 empty / 查询抛异常 → failed 且**软失败**：
  单元 blocked 带可读原因、整轮不 terminate_fail；图库取不到不冒到硬失败 except
- Milvus 侧能力的硬失败语义不变（由 test_planner_evidence_capabilities 覆盖）
- 计划没点名该类型时可选适配器不被启用，图服务零调用
- 规划提示词挂着该枚举值并带用法指导句（缺了它模型侧永远不启用这项能力）

本文件的单元一律显式索取正文核实（verify_with_direct_text=true），因为这里锁的是
「关系线索核实过才升级」这道门；不索取核实的结构线索走通用收尾，那条路另有覆盖。

作者：xxx
创建时间：2026-09-14 00:00:00
"""

from __future__ import annotations

from o2_evidence_test_utils import read_result

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from app.models import evidence_layers as layers
from app.nodes import rag_node
from app.services.internal_research.research_planner_executor import (
    EvidenceCapabilitySoftError,
    PlannerExecutionContext,
    PlannerUnitExecutor,
    _call_adapter,
    _record_matches_source,
)
from app.services.internal_research.research_planner_pipeline import (
    _adapters_for_plan,
    execute_planner_state,
)
from app.services.internal_research.research_planner_plan_prompt import build_planner_prompt
from research_planner_strategy_test_utils import validate_strategy_fixture
from research_planner_test_utils import make_draft, make_facet, make_unit

_SOURCE = layers.PLANNER_SOURCE_LOCAL_STRUCTURE_HINT
_PAPER = "upload_87b14b95"
_TITLE = "PDFormer"
_PAPER_NODE = f"paper:{_PAPER}"
_CAP_SETTING = "rag_planner_graph_relation_max_rows"
# 关系三元组记录形状（工作卡定死）：每条边一条，不得混进引用池
_RECORD_FIELDS = {
    "evidence_id", "source", "layer", "paper_id",
    "summary", "excerpt", "usable_as_citation", "provenance",
}
_METHOD_NODE = {"id": "entity_method_1", "label": "Method", "name": "Graph Attention Network"}
_DATASET_NODE = {"id": "entity_dataset_1", "label": "Dataset", "name": "PeMSD7"}


class _StubGraphService:
    """固定返回预置子图的图服务替身，记录查询入参。"""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload if payload is not None else _payload([])
        self.calls: list[list[str]] = []

    async def query_project_subgraph(self, paper_ids: list[str]) -> dict[str, Any]:
        self.calls.append(list(paper_ids))
        return self._payload


class _FailingGraphService:
    """查询即抛错的图服务替身，模拟 neo4j 连不上 / 查询超时。"""

    async def query_project_subgraph(self, paper_ids: list[str]) -> dict[str, Any]:
        raise RuntimeError("neo4j unavailable")


def _payload(
    edges: list[dict[str, Any]],
    *,
    nodes: list[dict[str, Any]] | None = None,
    papers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """照 query_project_subgraph 的真实返回形状拼子图载荷。"""
    return {
        "papers": [{"paper_id": _PAPER, "title": _TITLE}] if papers is None else papers,
        "nodes": nodes or [],
        "edges": edges,
        "meta": {
            "paper_count": 1,
            "node_count": len(nodes or []),
            "edge_count": len(edges),
        },
    }


def _edge(target: str, relation: str = "USES_METHOD") -> dict[str, Any]:
    return {"source": _PAPER_NODE, "target": target, "type": relation}


def _state(paper_ids: list[str]) -> dict[str, Any]:
    return {
        "run_id": "graph-relation-test-run",
        "user_query": "这篇论文用了什么方法、什么数据集",
        "focus": {"turn_focus": paper_ids},
        "project_scope": {"user_id": "u1", "paper_ids": paper_ids, "publication": {
            "user_id": "u1", "lease_id": "graph-fixture-lease", "batch_ids": ["b" * 32],
            "paper_batches": {paper_id: "b" * 32 for paper_id in paper_ids},
        }},
    }


def _ctx() -> PlannerExecutionContext:
    return PlannerExecutionContext(question="这篇论文用了什么方法、什么数据集")


def _structure_unit() -> dict[str, Any]:
    unit = make_unit(
        "graph-relation",
        goal="Find graph relations between the target paper and its research objects",
        action="query the research object graph for relation triples",
        source_type=_SOURCE,
        allowed_claims=["evidence_grounded_inference"],
        paper_ids=[_PAPER],
        facets=["f1"],
    )
    # 本文件的断言都落在「关系线索要拿正文核实过才升级」这道门上，所以计划显式索取正文核实；
    # 不索取核实的单元不进验证器，那条路由 test_research_planner_executor 覆盖
    unit["evidence_policy"]["verify_with_direct_text"] = True
    return unit


def _metadata_unit() -> dict[str, Any]:
    unit = make_unit(
        "meta",
        source_type="local_metadata",
        allowed_claims=["evidence_grounded_inference"],
        paper_ids=[_PAPER],
        facets=["f1"],
    )
    unit["claim_policy"]["citation_required"] = False
    return unit


def _plan(units: list[dict[str, Any]]) -> Any:
    result = validate_strategy_fixture(make_draft(units, explicit_facets=[make_facet("f1")]))
    assert result.plan is not None
    return result.plan


def _run_plan(
    units: list[dict[str, Any]],
    service: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """用真实可选适配器执行计划，返回 (plan, report)。"""
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: service)
    plan = _plan(units)
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state([_PAPER])))
    report = asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))
    return plan, report


def _record_with(report: Any, provenance: str) -> dict[str, Any]:
    return next(item for item in report.evidence_records if item.get("provenance") == provenance)


def _empty_adapter() -> Any:
    """零命中的直接正文适配器替身：补证重跑时拿不到可引用的直接正文。"""

    async def run(unit: Any, _context: PlannerExecutionContext) -> list[dict[str, Any]]:
        return read_result([])

    return run


def test_edges_become_named_relation_triples(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubGraphService(_payload(
        [_edge("entity_method_1", "USES_METHOD"), _edge("entity_dataset_1", "USES_DATASET")],
        nodes=[_METHOD_NODE, _DATASET_NODE],
    ))

    _, report = _run_plan([_structure_unit()], service, monkeypatch)

    # 1. 按单元绑定论文真查子图，不再读主链上恒为空的 graph_evidence
    assert service.calls == [[_PAPER]]
    assert report.capability_results == {_SOURCE: "success"}
    # 2. 每条边一条三元组记录，端点名字与关系类型都进文案，形状与工作卡定死的一致
    records = [item for item in report.evidence_records if item["provenance"] == "graph_relation_query"]
    assert [record["summary"] for record in records] == [
        f"{_TITLE} USES_METHOD Graph Attention Network",
        f"{_TITLE} USES_DATASET PeMSD7",
    ]
    record = records[0]
    assert set(record) == _RECORD_FIELDS
    assert record["excerpt"] == record["summary"]
    assert record["source"] == layers.EVIDENCE_SOURCE_NEO4J
    assert record["layer"] == layers.EVIDENCE_LAYER_STRUCTURE_HINT
    assert record["paper_id"] == _PAPER
    # 3. 图谱关系不可引用：能过执行层按来源的认领，但绝不进引用池
    assert record["usable_as_citation"] is False
    assert _record_matches_source(record, _SOURCE) is True
    assert _record_matches_source(record, layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT) is False
    # 4. 节点 id 不裸奔：文案里只有名字，模型看不到裸 id
    assert "entity_method_1" not in record["summary"]
    assert record["evidence_id"] in report.results[0].evidence_ids
    # 5. 既有收尾门不动：结构提示要直接正文验证通过才升级，只有图记录时单元仍取证不足
    assert report.results[0].status == "insufficient_evidence"


def test_relation_evidence_ids_are_stable_across_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload([_edge("entity_method_1")], nodes=[_METHOD_NODE])

    _, first = _run_plan([_structure_unit()], _StubGraphService(payload), monkeypatch)
    _, second = _run_plan([_structure_unit()], _StubGraphService(payload), monkeypatch)

    # 同一关系重复查到仍是同一个证据 ID（AC 28 要求稳定，不用 uuid4）
    first_ids = sorted(item["evidence_id"] for item in first.evidence_records)
    assert first_ids == sorted(item["evidence_id"] for item in second.evidence_records)
    assert all(evidence_id.startswith("ev-") for evidence_id in first_ids)


def test_adapter_ignores_stale_graph_evidence_state(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubGraphService(_payload([_edge("entity_method_1")], nodes=[_METHOD_NODE]))
    state = {
        **_state([_PAPER]),
        "graph_evidence": [{"summary": "stale hint from another chain", "paper_id": _PAPER}],
    }
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: service)
    plan = _plan([_structure_unit()])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(state))

    report = asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))

    # 主链上那份 graph_evidence 不再被读：结果只来自真查图，陈旧提示一条都不出现
    assert service.calls == [[_PAPER]]
    assert all("stale hint" not in item["summary"] for item in report.evidence_records)


def test_edge_without_paper_endpoint_gets_empty_paper_id(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubGraphService(_payload(
        [{"source": "entity_a", "target": "entity_b", "type": "BASED_ON"}],
        nodes=[
            {"id": "entity_a", "label": "Method", "name": "Transformer"},
            {"id": "entity_b", "label": "Task", "name": "Traffic Forecasting"},
        ],
        papers=[],
    ))

    _, report = _run_plan([_structure_unit()], service, monkeypatch)

    # 端点上没有论文 id 时 paper_id 如实留空串，记录本身不因范围过滤被丢掉
    record = _record_with(report, "graph_relation_query")
    assert record["paper_id"] == ""
    assert record["summary"] == "Transformer BASED_ON Traffic Forecasting"
    assert record["evidence_id"] in report.results[0].evidence_ids


def test_over_cap_keeps_first_edges_and_reports_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubGraphService(_payload(
        [_edge("entity_method_1"), _edge("entity_method_2"), _edge("entity_method_3")],
        nodes=[
            {"id": "entity_method_1", "label": "Method", "name": "Node 1"},
            {"id": "entity_method_2", "label": "Method", "name": "Node 2"},
            {"id": "entity_method_3", "label": "Method", "name": "Node 3"},
        ],
    ))
    monkeypatch.setattr(rag_node.settings, _CAP_SETTING, 2)

    _, report = _run_plan([_structure_unit()], service, monkeypatch)

    # 1. 超上限时取前 N 条，不静默截断也不打乱返回顺序
    edge_records = [item for item in report.evidence_records if item["provenance"] == "graph_relation_query"]
    assert [record["summary"] for record in edge_records] == [
        f"{_TITLE} USES_METHOD Node 1",
        f"{_TITLE} USES_METHOD Node 2",
    ]
    # 2. 读取额度单独报告，关系证据池只有真实返回的两条关系
    read_report = report.results[0].metadata["read_reports"][0]
    assert read_report["status"] == "partial"
    assert read_report["remaining"][0]["hit_count"] == 3
    assert read_report["remaining"][0]["returned_count"] == 2
    assert len(report.evidence_records) == 2
    assert report.capability_results == {_SOURCE: "success"}


def test_graph_target_outside_authorization_is_rejected_before_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标越界时在真查之前就拒掉：底层图服务零调用，失败面与原因都和「图库无数据」可区分。"""
    service = _StubGraphService(_payload([_edge("entity_method_1")], nodes=[_METHOD_NODE]))
    unit = _plan([_structure_unit()]).units[0]
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: service)
    # 授权范围只有 own_a，单元却点名查 foreign_b（模型给的候选，不是授权凭据）
    state = {"focus": {"turn_focus": ["own_a"]}, "project_scope": {"user_id": "u1", "paper_ids": ["own_a"]}}

    with pytest.raises(EvidenceCapabilitySoftError) as caught:
        asyncio.run(rag_node._planner_structure_records(state, unit, _ctx()))

    assert service.calls == []
    assert caught.value.source_type == _SOURCE
    assert "不在本轮授权范围" in caught.value.reason
    assert _PAPER in caught.value.reason


def test_graph_target_without_any_authorized_scope_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """授权范围读不到时按失败关闭处理：验不了授权就不查图，不放行到「没有范围约束」的查询。"""
    service = _StubGraphService(_payload([_edge("entity_method_1")], nodes=[_METHOD_NODE]))
    unit = _plan([_structure_unit()]).units[0]
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: service)
    # 只带焦点没带 project_scope：身份与归属的真相源缺席，不能当成「不限制」
    state = {"focus": {"turn_focus": [_PAPER]}}

    with pytest.raises(EvidenceCapabilitySoftError) as caught:
        asyncio.run(rag_node._planner_structure_records(state, unit, _ctx()))

    assert service.calls == []
    assert "不在本轮授权范围" in caught.value.reason


def test_reachable_graph_without_edges_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubGraphService(_payload([]))

    _, report = _run_plan([_structure_unit()], service, monkeypatch)

    # 图库能连上但零边是如实的空：不冒称成功，也不产概览
    assert service.calls == [[_PAPER]]
    assert report.evidence_records == []
    assert report.capability_results == {_SOURCE: "empty"}
    assert report.results[0].status == "insufficient_evidence"
    assert report.results[0].unresolved_reason == "no structure hints found"


def test_unauthorized_graph_target_blocks_unit_and_keeps_turn_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """越界目标是软失败：该单元 blocked 带可读原因、四态记 failed，整轮不 terminate_fail。

    与「图库连不上」失败面不同，但两者都只阻塞该单元；这一点必须与权限静默剔除区分开。
    """
    service = _StubGraphService(_payload([_edge("entity_method_1")], nodes=[_METHOD_NODE]))
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: service)
    plan = _plan([_structure_unit()])
    state = {"focus": {"turn_focus": ["own_a"]}, "project_scope": {"user_id": "u1", "paper_ids": ["own_a"]}}
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(state))

    report = asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))

    assert service.calls == []
    assert report.capability_results == {_SOURCE: "failed"}
    assert report.results[0].status == "blocked"
    assert "不在本轮授权范围" in report.results[0].unresolved_reason


def test_structure_hint_matcher_claims_graph_records_only() -> None:
    """结构线索只认图库来源：章节结构名单与正文截断概览同为结构线索层但源是 milvus，不得被认领。

    今天筛选逐个适配器跑、记录池不混，认宽了也不发作；一旦有人把记录合并后再筛就立刻串味，
    所以这里按源把边界钉死。
    """
    # 1. 图谱关系记录：源是 neo4j，照旧被认领
    graph_relation = {
        "evidence_id": "ev-graph-1",
        "source": layers.EVIDENCE_SOURCE_NEO4J,
        "layer": layers.EVIDENCE_LAYER_STRUCTURE_HINT,
        "paper_id": _PAPER,
        "provenance": "graph_relation_query",
        "usable_as_citation": False,
    }
    # 2. 章节结构名单：同为结构线索层，但源是 milvus，归 local_section_structure
    section_structure = {
        "evidence_id": f"section-structure-{_PAPER}",
        "source": layers.EVIDENCE_SOURCE_MILVUS,
        "layer": layers.EVIDENCE_LAYER_STRUCTURE_HINT,
        "paper_id": _PAPER,
        "usable_as_citation": False,
    }
    # 3. 正文截断概览：历史伪概览既不是关系来源，也不再属于正文证据
    section_text_truncation = {
        "evidence_id": "ev-section-text-truncation",
        "source": layers.EVIDENCE_SOURCE_MILVUS,
        "layer": layers.EVIDENCE_LAYER_STRUCTURE_HINT,
        "paper_id": "",
        "provenance": "section_path_truncation",
        "usable_as_citation": False,
    }

    assert _record_matches_source(graph_relation, _SOURCE) is True
    assert _record_matches_source(section_structure, _SOURCE) is False
    assert _record_matches_source(section_text_truncation, _SOURCE) is False
    # 4. 各归各家：真实目录材料保持原来源，历史截断概览不再认领为正文
    assert _record_matches_source(
        section_structure, layers.PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE
    ) is True
    assert _record_matches_source(section_text_truncation, layers.PLANNER_SOURCE_LOCAL_SECTION_TEXT) is False


def test_graph_query_failure_is_soft_and_turn_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: _FailingGraphService())
    plan = _plan([_structure_unit()])
    unit = plan.units[0]
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state([_PAPER])))

    # 1. 适配器把查询异常换成软失败信号：带取证面与可读原因，原始异常链保留
    with pytest.raises(EvidenceCapabilitySoftError) as caught:
        asyncio.run(rag_node._planner_structure_records(_state([_PAPER]), unit, _ctx()))
    assert caught.value.source_type == _SOURCE
    assert caught.value.reason == "RuntimeError: neo4j unavailable"
    assert isinstance(caught.value.__cause__, RuntimeError)
    # 2. _call_adapter 不把软失败重包成硬失败，原样上抛
    with pytest.raises(EvidenceCapabilitySoftError):
        asyncio.run(_call_adapter(adapters[_SOURCE], unit, _ctx()))
    # 3. 执行层消化软失败：单元 blocked 带可读原因、四态如实记 failed，整轮不抛异常
    report = asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))
    assert report.capability_results == {_SOURCE: "failed"}
    assert report.results[0].status == "blocked"
    assert report.results[0].unresolved_reason == "RuntimeError: neo4j unavailable"
    # 4. 走到 execute 阶段仍不是硬失败：不写 planner_execution_failed、决策不 terminate_fail
    payload = asyncio.run(execute_planner_state(
        {**_state([_PAPER]), "user_query": plan.question, "research_plan": plan.model_dump(mode="json")},
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))
    assert payload.get("error_code") != "planner_execution_failed"
    assert payload["planner_loop_decision"] != "terminate_fail"
    assert payload["source_status"]["planner"] == "executed"
    assert payload["planner_evidence_capabilities"] == {_SOURCE: "failed"}
    # 5. 失败在单元结果里显式可见：blocked + 可读原因，读的时候看这一处
    assert payload["research_unit_results"][0]["status"] == "blocked"
    assert payload["research_unit_results"][0]["unresolved_reason"] == "RuntimeError: neo4j unavailable"


def test_graph_relations_survive_the_execute_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubGraphService(_payload([_edge("entity_method_1")], nodes=[_METHOD_NODE]))
    plan, _ = _run_plan([_structure_unit()], service, monkeypatch)
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state([_PAPER])))

    payload = asyncio.run(execute_planner_state(
        {**_state([_PAPER]), "user_query": plan.question, "research_plan": plan.model_dump(mode="json")},
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))

    # 有边就是四态 success，且关系记录留在本轮证据池里；补证阶段拿不到直接正文时
    # 单元如实停在取证不足，而不是硬失败
    assert payload["planner_evidence_capabilities"] == {_SOURCE: "success"}
    assert payload.get("error_code") is None
    assert payload["planner_loop_decision"] != "terminate_fail"
    assert [item["provenance"] for item in payload["evidence_records"]] == ["graph_relation_query"]
    assert payload["research_unit_results"][0]["status"] == "insufficient_evidence"


def test_repair_rerun_keeps_graph_records_non_citable(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubGraphService(_payload([_edge("entity_method_1")], nodes=[_METHOD_NODE]))
    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: service)
    plan = _plan([_structure_unit()])
    # 直接正文补检索在册但零命中：补证重跑走 _rerun_structure_hint，图记录仍不可引用
    base = {layers.PLANNER_SOURCE_LOCAL_DIRECT_TEXT: _empty_adapter()}
    adapters = _adapters_for_plan(plan, base, rag_node._planner_optional_adapters(_state([_PAPER])))

    state = {**_state([_PAPER]), "user_query": plan.question, "research_plan": plan.model_dump(mode="json")}
    initial = asyncio.run(execute_planner_state(
        state,
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))
    assert initial["planner_loop_decision"] == "repair_evidence"
    payload = asyncio.run(execute_planner_state(
        {**state, **initial},
        llm=_FakeLLM(),
        settings=_Settings(),
        adapters=adapters,
    ))

    assert payload["planner_evidence_capabilities"] == {_SOURCE: "success", "local_direct_text": "empty"}
    assert payload["research_unit_results"][0]["status"] == "insufficient_evidence"
    assert payload["research_unit_results"][0]["unresolved_reason"] == (
        "structure_hint still lacks direct-text verification after repair"
    )
    # 关系记录不可引用，补证也升不了级；它们照旧留在证据池里供推断
    assert [item["provenance"] for item in payload["evidence_records"]] == ["graph_relation_query"]


def test_adapter_not_selected_when_plan_does_not_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数为外部依赖替身；只声明论文详情时实际读取详情，图谱能力不得被启用。"""
    service = _StubGraphService()
    metadata_calls = []

    def lookup(paper_id: str, user_id: str) -> dict[str, Any]:
        """参数为论文和读取者；记录隔离的详情调用，返回真实形状的论文元数据。"""
        metadata_calls.append((paper_id, user_id))
        return {"paper_id": paper_id, "title": _TITLE, "abstract": "Paper metadata fixture"}

    monkeypatch.setattr(rag_node, "get_research_object_query_service", lambda: service)
    monkeypatch.setattr(rag_node.paper_db, "get_paper", lookup)
    plan = _plan([_metadata_unit()])
    adapters = _adapters_for_plan(plan, {}, rag_node._planner_optional_adapters(_state([_PAPER])))

    assert _SOURCE not in adapters
    asyncio.run(PlannerUnitExecutor(adapters=adapters).execute_with_evidence(plan, _ctx()))

    assert service.calls == []
    assert metadata_calls == [(_PAPER, "u1")]


def test_prompt_guides_structure_hint_and_lists_it_in_the_enum() -> None:
    prompt = build_planner_prompt({})

    # 1. 来源枚举由实时模型统一提供，提示只负责图关系的业务含义
    from app.models.research_planner_draft_contracts import ResearchPlanStrategyDraft
    schema = ResearchPlanStrategyDraft.model_json_schema()
    assert "local_structure_hint" in str(schema)
    # 2. 用法指导句必须在场，且指向研究对象图的关系三元组
    assert "local_structure_hint for known graph relations" in prompt
    assert "Direct-text verification of graph relations" in prompt


@dataclass
class _Settings:
    rag_research_planner_max_units: int = 8


@dataclass
class _FakeLLM:
    def with_structured_output(self, model: Any) -> Any:
        raise AssertionError("claim drafting must not run in these tests")
