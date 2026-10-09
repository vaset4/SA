from __future__ import annotations

import asyncio
import os
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

os.environ["SSLKEYLOGFILE"] = ""

import app.api.routes.task_api as task_api
from app.api.schemas.tasks import ResearchTaskRequest
from app.api.facades import session_identity_facade, upload_facade
from app.api.facades.session_identity_facade import ConversationIdentity


def test_create_research_task_route_delegates_to_service(monkeypatch) -> None:
    async def fake_resolve_conversation_identity(**kwargs):  # noqa: ANN003
        assert kwargs["user_id"] == "anonymous"
        return ConversationIdentity(conversation_id=None, session_id=kwargs["session_id"])

    async def fake_create_research_task(query, session_id=None, *, user_id):  # noqa: ANN001
        return {
            "task_id": "task-1",
            "task_type": "deep_research",
            "status": "queued",
            "progress": 0,
            "message": "deep research 任务已入队",
            "payload": {"query": query, "session_id": session_id, "user_id": user_id},
            "cancel_requested": False,
            "created_at": "2026-03-23T00:00:00+00:00",
            "updated_at": "2026-03-23T00:00:00+00:00",
        }

    monkeypatch.setattr(session_identity_facade, "resolve_conversation_identity", fake_resolve_conversation_identity)
    monkeypatch.setattr(task_api.task_service, "create_research_task", fake_create_research_task)

    payload = asyncio.run(
        task_api.create_research_task(
            ResearchTaskRequest(query="研究 LangGraph interrupt", session_id="session-1"),
            user={},
        )
    )

    assert payload["task_type"] == "deep_research"
    assert payload["payload"]["session_id"] == "session-1"


def test_create_research_task_route_hides_cross_user_session(monkeypatch) -> None:
    """越权或不存在的 research session 必须统一映射为 404，且不能创建任务。"""
    called = False

    async def fail_resolve_conversation_identity(**kwargs):  # noqa: ANN003
        raise LookupError("会话不存在")

    async def fail_create_research_task(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal called
        called = True
        raise AssertionError("越权会话不能创建任务")

    monkeypatch.setattr(
        session_identity_facade,
        "resolve_conversation_identity",
        fail_resolve_conversation_identity,
    )
    monkeypatch.setattr(task_api.task_service, "create_research_task", fail_create_research_task)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            task_api.create_research_task(
                ResearchTaskRequest(query="研究 LangGraph interrupt", session_id="session-b"),
                user={"user_id": "user-a"},
            )
        )

    assert exc_info.value.status_code == 404
    assert called is False


def test_create_upload_task_route_maps_value_error(monkeypatch) -> None:
    def fail_validate(filename, content_type):  # noqa: ANN001
        raise ValueError("不支持的 MIME 类型: text/plain")

    monkeypatch.setattr(task_api.paper_facade, "validate_upload", fail_validate)

    upload = UploadFile(filename="paper.txt", file=BytesIO(b"hello"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(task_api.create_upload_task(upload, user={}, library_id=None, project_id=None))

    assert exc_info.value.status_code == 415


def test_create_upload_task_route_starts_embedded_worker(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_create_upload_task(filename, raw, *, library_id=None, project_id=None, user_id="anonymous"):  # noqa: ANN001
        captured.update(
            {
                "filename": filename,
                "raw": raw,
                "library_id": library_id,
                "project_id": project_id,
                "user_id": user_id,
            }
        )
        return {
            "task_id": "task-upload-1",
            "task_type": "upload_ingest",
            "status": "queued",
            "progress": 0,
            "message": "上传入库任务已入队",
            "payload": {},
            "cancel_requested": False,
            "created_at": "2026-03-23T00:00:00+00:00",
            "updated_at": "2026-03-23T00:00:00+00:00",
        }

    worker_calls: list[list[str] | None] = []

    monkeypatch.setattr(task_api.paper_facade, "validate_upload", lambda filename, content_type: None)
    monkeypatch.setattr(task_api.task_service, "create_upload_task", fake_create_upload_task)
    monkeypatch.setattr(
        task_api.embedded_worker_service,
        "ensure_embedded_worker_running",
        lambda *, task_types=None: worker_calls.append(task_types),
    )
    monkeypatch.setattr(task_api, "get_user_id", lambda user: "user-1")

    upload = UploadFile(filename="paper.pdf", file=BytesIO(b"%PDF-1.4"))

    payload = asyncio.run(task_api.create_upload_task(upload, user={}, library_id="lib-1", project_id="proj-1"))

    assert payload["task_id"] == "task-upload-1"
    assert captured == {
        "filename": "paper.pdf",
        "raw": b"%PDF-1.4",
        "library_id": "lib-1",
        "project_id": "proj-1",
        "user_id": "user-1",
    }
    assert worker_calls == [["upload_ingest"]]


def test_create_upload_task_rejects_oversized_upload_before_staging(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_create_upload_task(*args, **kwargs):  # noqa: ANN001
        calls.append("create")
        return {"task_id": "task-upload-1"}

    monkeypatch.setattr(task_api.paper_facade, "validate_upload", lambda filename, content_type: None)
    monkeypatch.setattr(task_api.task_service, "create_upload_task", fake_create_upload_task)
    monkeypatch.setattr(upload_facade, "get_settings", lambda: type("S", (), {"max_upload_size_mb": 1})())

    upload = UploadFile(filename="paper.pdf", file=BytesIO(b"%PDF-" + (b"a" * upload_facade.BYTES_PER_MIB)))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(task_api.create_upload_task(upload, user={}, library_id=None, project_id=None))

    assert exc_info.value.status_code == 413
    assert calls == []
