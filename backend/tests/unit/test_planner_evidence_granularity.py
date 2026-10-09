"""O2 子块/父块权威读取与定位检索保序测试，作者：xxx。"""
from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest

from app.models.research_contracts import RetrievalScopeError
from app.models.research_planner_contracts import ResearchUnit
from app.nodes import rag_node
from o2_evidence_test_utils import bind_publication, scope
from research_planner_test_utils import make_unit


def _row(index: int, *, parent: str | None = None, paper_id: str = "paper-a") -> dict[str, Any]:
    """参数为文档序号和可选父身份；返回预览刻意不同于发布正文的离线块。"""
    return {"chunk_id": f"v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:{index}", "parent_id": parent or f"parent-{index}",
            "paper_id": paper_id, "section": "Method", "section_path": "Paper / Method",
            "block_type": "paragraph", "page_num": index + 1,
            "text": f"published child {index}", "full_text": f"published parent {index}",
            "preview": f"clipped preview {index}", "score": 0.5}


def _unit(*, granularity="child_chunk", queries=None, paper_ids=None) -> ResearchUnit:
    """参数为粒度、定位词和目标；返回正式研究单元。"""
    unit = ResearchUnit.model_validate(make_unit(paper_ids=paper_ids or ["paper-a"]))
    return unit.model_copy(update={"evidence_policy": unit.evidence_policy.model_copy(update={
        "evidence_granularity": granularity, "locator_queries": queries or ["attention mechanism"],
    })})


def _fixture(monkeypatch, rows, *, limit=5, failure=None):
    """参数为发布块与额度；返回调用正式知识读取、仅存储端口为替身的回合。"""
    project_scope = scope()
    state = {"project_scope": project_scope, "structured_query": {"project_scope": project_scope}, "filter_expr": ""}
    store = Mock(spec=["hybrid_search"])
    store.hybrid_search.side_effect = lambda *args, **kwargs: deepcopy(rows)
    monkeypatch.setattr(rag_node, "get_vector_store", lambda: store)
    monkeypatch.setattr(rag_node.paper_db, "get_paper", lambda *args: {"paper_id": "paper-a"})
    monkeypatch.setattr(rag_node.settings, "rag_auto_expand_parent_limit", limit)
    reads = bind_publication(monkeypatch, rows, failure=failure)
    return state, store, reads


def test_child_granularity_reads_authoritative_text_once_per_batch(monkeypatch):
    """参数为 patch；子块默认读取真实 text，多块共用一次批次装载。"""
    rows = [_row(0), _row(1)]
    state, _, reads = _fixture(monkeypatch, rows)
    result = asyncio.run(rag_node._planner_direct_text_records(state, _unit(), None))
    assert [r["excerpt"] for r in result.records] == ["published child 0", "published child 1"]
    assert all(r["source_spans"][0]["field"] == "text" for r in result.records)
    assert len(reads) == 1
    assert reads[0][1]["publication"] == state["project_scope"]["publication"]
    assert result.read_report.status == "complete"


def test_parent_granularity_preserves_full_text_and_explicit_limit(monkeypatch):
    """参数为 patch；选中父块保持完整正文，超额定位不混成子块也不伪造证据。"""
    rows = [_row(0), _row(1), _row(2)]
    rows[0]["full_text"] = "| a | b |\n| 1 | 2 |\n" + "x" * 5000
    state, _, reads = _fixture(monkeypatch, rows, limit=2)
    result = asyncio.run(rag_node._planner_direct_text_records(state, _unit(granularity="parent_chunk"), None))
    assert [r["excerpt"] for r in result.records] == [rows[0]["full_text"], rows[1]["full_text"]]
    assert all(r["source_spans"][0]["field"] == "full_text" for r in result.records)
    assert result.read_report.status == "partial"
    assert result.read_report.remaining[-1]["chunk_id"] == "v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:2"
    assert len(reads) == 1


def test_same_parent_has_one_material_multiple_child_spans(monkeypatch):
    """参数为 patch；同一父块多个子块按实际范围合成一条材料，不吞掉第二段。"""
    rows = [_row(0, parent="shared"), _row(1, parent="shared")]
    rows[1]["full_text"] = rows[0]["full_text"]
    state, _, _ = _fixture(monkeypatch, rows)
    child = asyncio.run(rag_node._planner_direct_text_records(state, _unit(), None))
    assert len(child.records) == 1
    assert child.records[0]["excerpt"] == "published child 0\n\npublished child 1"
    assert len(child.records[0]["source_spans"]) == 2
    parent = asyncio.run(rag_node._planner_direct_text_records(state, _unit(granularity="parent_chunk"), None))
    assert len(parent.records) == 1
    assert parent.records[0]["excerpt"] == rows[0]["full_text"]


def test_parent_identity_is_stable_across_different_hit_children(monkeypatch):
    """参数为 patch；同父完整正文固定来源身份，同时保留各次实际命中分数。"""
    rows = [_row(0, parent="shared"), _row(1, parent="shared")]
    rows[1]["full_text"] = rows[0]["full_text"]
    rows[0]["score"], rows[1]["score"] = 0.4, 0.8
    state, store, _ = _fixture(monkeypatch, rows)
    store.hybrid_search.side_effect = lambda *args, **kwargs: [deepcopy(rows[0])]
    first = asyncio.run(rag_node._planner_direct_text_records(state, _unit(granularity="parent_chunk"), None))
    store.hybrid_search.side_effect = lambda *args, **kwargs: [deepcopy(rows[1])]
    second = asyncio.run(rag_node._planner_direct_text_records(state, _unit(granularity="parent_chunk"), None))
    assert first.records[0]["source_spans"] == second.records[0]["source_spans"]
    assert first.records[0]["evidence_id"] == second.records[0]["evidence_id"]
    assert first.records[0]["confidence"] == 0.4
    assert second.records[0]["confidence"] == 0.8


def test_missing_reader_or_snapshot_is_explicit_failure(monkeypatch):
    """参数为 patch；读取权限或固定版本缺失时不能把预览作为成功正文。"""
    state, _, reads = _fixture(monkeypatch, [_row(0)])
    state["project_scope"] = {"paper_ids": ["paper-a"]}
    with pytest.raises(RetrievalScopeError):
        asyncio.run(rag_node._planner_direct_text_records(state, _unit(), None))
    assert reads == []


def test_publication_failure_is_not_hidden(monkeypatch):
    """参数为 patch；发布产物故障明确上抛，原始检索命中不能掩盖失败。"""
    state, _, reads = _fixture(monkeypatch, [_row(0)], failure=LookupError("manifest missing"))
    with pytest.raises(LookupError, match="manifest missing"):
        asyncio.run(rag_node._planner_direct_text_records(state, _unit(), None))
    assert len(reads) == 1


def test_target_is_pushed_into_top_k_scope_and_never_widens(monkeypatch):
    """参数为 patch；模型范围先与授权求交再送搜索，不能查询后才过滤。"""
    state, store, _ = _fixture(monkeypatch, [])
    state["project_scope"]["paper_ids"].append("paper-b")
    asyncio.run(rag_node._planner_direct_text_records(state, _unit(paper_ids=["paper-b", "outside"]), None))
    pushed = store.hybrid_search.call_args.kwargs["filter_expr"]
    assert 'paper_id in ["paper-b"]' in pushed
    assert "outside" not in pushed


def test_locator_queries_are_concurrent_but_merge_in_request_order(monkeypatch):
    """参数为 patch；同批查询并发，读取候选仍按声明顺序。"""
    rows = [_row(0), _row(1)]
    state, store, _ = _fixture(monkeypatch, rows)
    rendezvous = threading.Barrier(2, timeout=5)
    def search(query, *args, **kwargs):
        """参数为定位词；等待另一查询后返回对应离线候选。"""
        rendezvous.wait()
        return [deepcopy(rows[int(query[-1])])]
    store.hybrid_search.side_effect = search
    result = asyncio.run(rag_node._planner_direct_text_records(state, _unit(queries=["q0", "q1"]), None))
    assert [r["chunk_id"] for r in result.records] == ["v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:0", "v3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:1"]
