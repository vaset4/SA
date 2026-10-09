"""研究规划器论文详情取证适配器测试（C3 / F04，09-24 卡）。

覆盖点：
- 已授权但尚未索引的论文也能按需读到真实摘要与已存分析，不需要向量索引成功
- 有分析时按字段并列给出，缺失字段如实不出现，不拿摘要冒充分析
- 按 ID 直读前先核对本轮授权范围与读取人：越界目标不读库，也不算别人的论文
- 元数据记录不可引用：它只当规划已知事实，不产出原文引用

作者：xxx
创建时间：2026-09-25 00:00:00
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from app.models import evidence_layers as layers
from app.models.research_contracts import RetrievalScopeError
from app.models.research_planner_contracts import ResearchUnit
from app.nodes import rag_node
from research_planner_test_utils import make_unit

_PAPER = "paper-a"
_OTHER = "paper-b"
_USER_ID = "u1"


def _unit(*, paper_ids: list[str]) -> ResearchUnit:
    return ResearchUnit.model_validate(make_unit("meta", source_type="local_metadata", paper_ids=paper_ids))


def _state(paper_ids: list[str], *, user_id: str = _USER_ID) -> dict[str, Any]:
    project_scope: dict[str, Any] = {"paper_ids": paper_ids}
    if user_id:
        project_scope["user_id"] = user_id
    return {"user_query": "这篇论文讲了什么", "project_scope": project_scope}


def _paper_record(
    paper_id: str,
    *,
    title: str = "一篇论文",
    abstract: str = "摘要正文",
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"paper_id": paper_id, "title": title, "abstract": abstract, "analysis": analysis}


def _run(monkeypatch: pytest.MonkeyPatch, state: dict[str, Any], unit: ResearchUnit, records: list[dict[str, Any]]):
    lookup = Mock(side_effect=lambda paper_id, user_id=None: next(
        (record for record in records if record["paper_id"] == paper_id), None
    ))
    monkeypatch.setattr(rag_node.paper_db, "get_paper", lookup)
    result = asyncio.run(rag_node._planner_metadata_records(state, unit, None))
    return list(result.records), lookup


def test_unindexed_paper_still_yields_real_abstract_and_stored_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """论文只有摘要、没建索引时同样读得到详情：这条路不依赖向量索引或图谱。"""
    paper = _paper_record(
        _PAPER,
        abstract="完整摘要正文",
        analysis={"problem": "要解决的问题", "method": "提出的方法"},
    )

    records, _ = _run(monkeypatch, _state([_PAPER]), _unit(paper_ids=[_PAPER]), [paper])

    assert len(records) == 1
    excerpt = records[0]["excerpt"]
    assert "完整摘要正文" in excerpt
    assert "方法: 提出的方法" in excerpt
    # 元数据只当规划已知事实：不可引用，不进原文引用池
    assert records[0]["usable_as_citation"] is False
    assert records[0]["layer"] == layers.EVIDENCE_LAYER_METADATA
    assert records[0]["source"] == layers.EVIDENCE_SOURCE_INVENTORY


def test_abstract_is_not_truncated_to_a_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    """摘要给完整原文：结论常在五百字之后，预览式截断会让模型判错要不要读这篇。"""
    long_abstract = "开头" + "中" * 600 + "结论"
    paper = _paper_record(_PAPER, abstract=long_abstract)

    records, _ = _run(monkeypatch, _state([_PAPER]), _unit(paper_ids=[_PAPER]), [paper])

    assert "结论" in records[0]["excerpt"]
    assert "已截断" not in records[0]["excerpt"]


def test_missing_analysis_is_not_faked_from_the_abstract(monkeypatch: pytest.MonkeyPatch) -> None:
    """库里没有分析时如实不带分析字段，不拿摘要冒充分析。"""
    paper = _paper_record(_PAPER, abstract="只有摘要", analysis=None)

    records, _ = _run(monkeypatch, _state([_PAPER]), _unit(paper_ids=[_PAPER]), [paper])

    assert records[0]["excerpt"] == "只有摘要"


def test_read_binds_the_round_user_so_foreign_papers_are_not_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读库要带上本轮读取人：不带时仓储层的归属谓词整条跳过，模型点谁的 ID 都能读到。"""
    paper = _paper_record(_PAPER)

    _, lookup = _run(monkeypatch, _state([_PAPER]), _unit(paper_ids=[_PAPER]), [paper])

    assert lookup.call_args_list == [((_PAPER, _USER_ID), {})]


def test_out_of_scope_target_is_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型点到授权范围外的论文时既不读库也不产记录，收窄只往下走。"""
    in_scope = _paper_record(_PAPER)
    outside = _paper_record(_OTHER)

    records, lookup = _run(
        monkeypatch,
        _state([_PAPER]),
        _unit(paper_ids=[_PAPER, _OTHER]),
        [in_scope, outside],
    )

    assert [record["paper_id"] for record in records] == [_PAPER]
    assert all(call.args[0] != _OTHER for call in lookup.call_args_list)


def test_missing_reader_identity_fails_instead_of_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """取不到读取人时显式失败，不拿模型给的 ID 裸读。"""
    lookup = Mock(return_value=None)
    monkeypatch.setattr(rag_node.paper_db, "get_paper", lookup)

    with pytest.raises(RetrievalScopeError):
        asyncio.run(rag_node._planner_metadata_records(_state([_PAPER], user_id=""), _unit(paper_ids=[_PAPER]), None))

    assert lookup.call_args_list == []
