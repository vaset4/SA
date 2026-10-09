from __future__ import annotations

import asyncio
import os

import pytest

os.environ["SSLKEYLOGFILE"] = ""

import app.api.routes.papers_api as papers_api_module
from app.services.ingest import document_graph_service, upload_pipeline
import app.services.lightrag_service as lightrag_service


@pytest.mark.asyncio
async def test_insert_to_lightrag_uses_paper_id_as_doc_id(monkeypatch):
    calls: list[tuple[str, str, str]] = []

    class FakeService:
        async def insert(self, text: str, paper_id: str = "", *, library_id: str = "default") -> None:
            calls.append((text, paper_id, library_id))

    monkeypatch.setattr(lightrag_service, "get_lightrag_service", lambda: FakeService())

    await document_graph_service.insert_to_lightrag("paper_123", "full text" * 3, library_id="lib_a")

    assert calls == [("full textfull textfull text", "paper_123", "lib_a")]


@pytest.mark.asyncio
async def test_optional_graph_uses_published_paper_identity(monkeypatch):
    """独立图谱使用核验原文和服务端文献库，不再次解析或生成新论文身份。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.models.upload_models import UploadExecution

    execution = UploadExecution("task", "worker", "paper", "owner", "parse", "batch", "published", "graph")
    parsed = {"document": {"full_text": "published text", "chunks": [], "metadata": {}}}
    insert = AsyncMock()
    repository = SimpleNamespace(get_document=AsyncMock(return_value={"library_id": "owned-library"}))
    monkeypatch.setattr(upload_pipeline, "get_upload_repository", lambda: repository)
    monkeypatch.setattr(upload_pipeline, "get_settings", lambda: SimpleNamespace(graph_rag_enabled=True))
    monkeypatch.setattr(document_graph_service, "insert_to_lightrag", insert)
    result = await upload_pipeline._build_graph(parsed, execution)
    assert result == {"paper_id": "paper", "graph_status": "completed"}
    insert.assert_awaited_once_with("paper", "published text", library_id="owned-library", chunks=[], metadata={})


@pytest.mark.asyncio
async def test_delete_paper_enqueues_owner_aware_task(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_request_deletion(paper_id: str, *, user_id: str):
        captured.update({"paper_id": paper_id, "user_id": user_id})
        return {"task_id": "task-delete-1", "task_type": "paper_delete", "status": "queued"}

    monkeypatch.setattr(papers_api_module, "get_user_id", lambda _user: "user-1")
    monkeypatch.setattr(papers_api_module.paper_facade, "request_paper_deletion_or_raise", fake_request_deletion)

    result = await papers_api_module.delete_paper("paper_1", user={})
    route = next(item for item in papers_api_module.router.routes if item.path == "/papers/{paper_id}" and "DELETE" in item.methods)

    assert route.status_code == 202
    assert result["task_id"] == "task-delete-1"
    assert captured == {"paper_id": "paper_1", "user_id": "user-1"}


@pytest.mark.asyncio
async def test_lightrag_service_insert_and_delete_use_document_id(monkeypatch):
    service = lightrag_service.LightRAGService()

    class FakeRag:
        def __init__(self) -> None:
            self.insert_calls: list[tuple[str, list[str]]] = []
            self.delete_calls: list[tuple[str, bool]] = []
            self.doc_status = self

        async def ainsert(self, text: str, ids: list[str] | None = None) -> str:
            self.insert_calls.append((text, ids or []))
            return "track-1"

        async def get_by_id(self, doc_id: str):
            assert doc_id == "paper_3"
            return {"status": "processed", "error_msg": ""}

        async def adelete_by_doc_id(self, doc_id: str, delete_llm_cache: bool = False):
            self.delete_calls.append((doc_id, delete_llm_cache))
            return type(
                "DeletionResult",
                (),
                {
                    "status": "success",
                    "message": "deleted",
                    "doc_id": doc_id,
                    "status_code": 200,
                    "file_path": None,
                },
            )()

    fake_rag = FakeRag()
    monkeypatch.setattr(lightrag_service.settings, "graph_rag_enabled", True, raising=False)
    runtime = service._get_runtime("lib_graph")
    runtime.rag = fake_rag
    runtime.initialized = True

    await service.insert("body text", paper_id="paper_3", library_id="lib_graph")
    result = await service.delete_by_doc_id("paper_3", library_id="lib_graph")

    assert fake_rag.insert_calls == [("[Library ID: lib_graph]\n[Paper ID: paper_3]\n\nbody text", ["paper_3"])]
    assert fake_rag.delete_calls == [("paper_3", True)]
    assert result["status"] == "success"
