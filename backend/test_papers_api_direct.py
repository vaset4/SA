from __future__ import annotations

import asyncio
import os
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

os.environ["SSLKEYLOGFILE"] = ""

import app.api.routes.papers_api as papers_api
from app.api.facades import upload_facade


def test_upload_paper_route_uses_default_library_when_missing(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def ok_validate(filename, content_type):  # noqa: ANN001
        observed["filename"] = filename
        observed["content_type"] = content_type

    async def fake_ingest(
        filename,
        raw,
        library_id,
        project_id=None,
        *,
        user_id="anonymous",
        run_analysis=False,
    ):  # noqa: ANN001
        observed["ingest_filename"] = filename
        observed["ingest_size"] = len(raw)
        observed["library_id"] = library_id
        observed["project_id"] = project_id
        observed["user_id"] = user_id
        observed["run_analysis"] = run_analysis
        return {"paper_id": "upload_1", "library_id": library_id or "default"}

    monkeypatch.setattr(papers_api.paper_facade, "validate_upload", ok_validate)
    monkeypatch.setattr(papers_api.paper_facade, "ingest_uploaded_paper", fake_ingest)

    upload = UploadFile(
        filename="paper.pdf",
        file=BytesIO(b"%PDF-1.4 test"),
        headers={"content-type": "application/pdf"},
    )

    payload = asyncio.run(papers_api.upload_paper(upload, library_id=None, project_id=None, user={}))

    assert payload["paper_id"] == "upload_1"
    assert payload["library_id"] == "default"
    assert observed["library_id"] is None
    assert observed["project_id"] is None
    assert observed["run_analysis"] is False


def test_upload_paper_route_rejects_oversized_upload_before_ingest(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_ingest(*args, **kwargs):  # noqa: ANN001
        calls.append("ingest")
        return {"paper_id": "upload_1"}

    monkeypatch.setattr(papers_api.paper_facade, "validate_upload", lambda filename, content_type: None)
    monkeypatch.setattr(papers_api.paper_facade, "ingest_uploaded_paper", fake_ingest)
    monkeypatch.setattr(upload_facade, "get_settings", lambda: type("S", (), {"max_upload_size_mb": 1})())

    upload = UploadFile(
        filename="paper.pdf",
        file=BytesIO(b"%PDF-" + (b"a" * upload_facade.BYTES_PER_MIB)),
        headers={"content-type": "application/pdf"},
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(papers_api.upload_paper(upload, library_id=None, project_id=None, user={}))

    assert exc_info.value.status_code == 413
    assert calls == []
