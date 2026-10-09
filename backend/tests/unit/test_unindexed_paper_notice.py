"""Milvus exists 检测单元测试（Phase 1 Slice D 的现役部分）。

未索引论文的回答段渲染与标题解析随旧问答路径（execute_analysis）一起摘除，
现役只剩检测：`rag_node` 用它把未进向量库的论文写进 RAGState.unindexed_paper_ids。

涵盖 PRD R3.1 / R3.4：
- 检测 milvus exists==False 的 paper_id
- 触发后 emit structured log event=rag_unindexed_papers_detected
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.services.rag_answer_service import detect_unindexed_papers



class _FakeMilvusStore:
    def __init__(self, indexed_ids: set[str]) -> None:
        self._indexed_ids = indexed_ids
        self.calls: List[str] = []

    def exists(self, paper_id: str) -> bool:
        self.calls.append(paper_id)
        return paper_id in self._indexed_ids


def test_detect_unindexed_papers_collects_zero_chunk_ids() -> None:
    """部分 paper_id 没有 chunk 时，必须被收集进 unindexed 列表。"""
    store = _FakeMilvusStore(indexed_ids={"upload_c41ff1c7", "upload_2292fcc2"})
    paper_ids = [
        "upload_c41ff1c7",
        "upload_2292fcc2",
        "upload_b49aabbf",
        "upload_61085f5a",
    ]
    unindexed = detect_unindexed_papers(paper_ids, milvus_store=store)
    assert unindexed == ["upload_b49aabbf", "upload_61085f5a"]
    # 同时验证每个 paper_id 都查过一次（顺序探查，可观测）
    assert store.calls == paper_ids


def test_detect_unindexed_papers_empty_when_all_indexed() -> None:
    """所有 paper_id 都已索引时，unindexed 必须为空列表（不是 None）。"""
    paper_ids = ["upload_a", "upload_b"]
    store = _FakeMilvusStore(indexed_ids=set(paper_ids))
    unindexed = detect_unindexed_papers(paper_ids, milvus_store=store)
    assert unindexed == []


def test_detect_unindexed_papers_empty_input_returns_empty() -> None:
    """空输入直接返回空列表，不应调用 milvus。"""
    store = _FakeMilvusStore(indexed_ids=set())
    unindexed = detect_unindexed_papers([], milvus_store=store)
    assert unindexed == []
    assert store.calls == []


# ---------------------------------------------------------------------------
# 结构化日志
# ---------------------------------------------------------------------------


def test_unindexed_papers_emit_structured_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """检测到未索引论文时，必须 emit event=rag_unindexed_papers_detected
    且字段包含 count / paper_ids。"""
    import app.services.rag_answer_service as svc

    captured: List[Dict[str, Any]] = []

    def fake_log(current_logger, event: str, *, level: int = 20, **fields: Any) -> None:
        captured.append({"event": event, "fields": fields})

    monkeypatch.setattr(svc, "log_structured_event", fake_log)

    store = _FakeMilvusStore(indexed_ids={"upload_a"})
    paper_ids = ["upload_a", "upload_b", "upload_c"]
    unindexed = svc.detect_unindexed_papers(paper_ids, milvus_store=store)
    assert unindexed == ["upload_b", "upload_c"]

    events = [e for e in captured if e["event"] == "rag_unindexed_papers_detected"]
    assert len(events) == 1, f"expected one detection event, got {captured!r}"
    fields = events[0]["fields"]
    assert fields["count"] == 2
    assert fields["paper_ids"] == ["upload_b", "upload_c"]


def test_no_log_event_when_no_unindexed(monkeypatch: pytest.MonkeyPatch) -> None:
    """全部已索引时不应发探测日志事件（避免噪音）。"""
    import app.services.rag_answer_service as svc

    captured: List[Dict[str, Any]] = []

    def fake_log(current_logger, event: str, *, level: int = 20, **fields: Any) -> None:
        captured.append({"event": event, "fields": fields})

    monkeypatch.setattr(svc, "log_structured_event", fake_log)

    store = _FakeMilvusStore(indexed_ids={"upload_a", "upload_b"})
    svc.detect_unindexed_papers(["upload_a", "upload_b"], milvus_store=store)
    events = [e for e in captured if e["event"] == "rag_unindexed_papers_detected"]
    assert events == []
