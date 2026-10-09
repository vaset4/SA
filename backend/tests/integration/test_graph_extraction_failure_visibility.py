"""可选图谱服务异常保持可见，作者：xxx。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.ingest import document_graph_service
from app.services.graph.research_object_llm_extractor import ExtractionFailure
from app.services.lightrag_insert_status import LightRAGInsertFailure


@pytest.mark.asyncio
async def test_neo4j_extraction_failure_is_not_reported_as_success(monkeypatch):
    """研究对象提取失败直接传播，不伪造可用图谱。"""
    insert = AsyncMock(side_effect=ExtractionFailure("paper", 4, "invalid JSON"))
    monkeypatch.setattr(document_graph_service, "get_research_object_graph_service",
        lambda: SimpleNamespace(upsert_paper_subgraph=insert))
    with pytest.raises(ExtractionFailure):
        await document_graph_service.insert_to_neo4j_graph({"paper_id": "paper", "user_id": "a"}, "full text")


@pytest.mark.asyncio
async def test_lightrag_insert_failure_is_not_reported_as_success(monkeypatch):
    """LightRAG 未完成写入时直接传播，独立图谱任务可记录真实暂停原因。"""
    from app.services import lightrag_service

    insert = AsyncMock(side_effect=LightRAGInsertFailure("insert incomplete: 524"))
    monkeypatch.setattr(lightrag_service, "get_lightrag_service", lambda: SimpleNamespace(insert=insert))
    with pytest.raises(LightRAGInsertFailure, match="524"):
        await document_graph_service.insert_to_lightrag("paper", "full text", library_id="library")
