"""不可变引用正文读取的隔离回归，作者：xxx。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.upload_models import ArtifactIntegrityError
from app.services import upload_read_service
from app.services.ingest.upload_artifacts import write_json_artifact
from source_file_test_ports import install_file_ports


@pytest.fixture
def published_reference(tmp_path, monkeypatch):
    """提供已退役但仍有正文的旧发布清单，不连接业务库或向量服务。"""
    install_file_ports(monkeypatch)
    chunk_id = "v3:" + "a" * 32 + ":0"
    chunk = {"chunk_id": chunk_id, "paper_id": "paper", "batch_id": "a" * 32,
        "full_text": "引用当时的完整正文", "parent_id": "parent"}
    reference = write_json_artifact(tmp_path, {"format": 3, "batch_id": "a" * 32, "paper_id": "paper", "user_id": "owner", "chunks": [chunk]})
    lookup = AsyncMock(return_value={"paper_id": "paper", "state": "retired", "manifest_ref": reference})
    monkeypatch.setattr(upload_read_service, "get_settings", lambda: SimpleNamespace(data_dir=str(tmp_path)))
    monkeypatch.setattr(upload_read_service, "get_upload_publication_repository", lambda: SimpleNamespace(content_reference=lookup))
    monkeypatch.setattr(upload_read_service.paper_db, "_run", asyncio.run)
    return {"chunk": chunk, "reference": reference, "lookup": lookup}


def test_old_citation_reads_original_content_after_vector_retirement(published_reference):
    """清理旧向量不影响引用原文；正文读取继续携带当前账号。"""
    fixture = published_reference
    assert upload_read_service.read_published_chunk(fixture["chunk"]["chunk_id"], user_id="owner") == fixture["chunk"]
    assert fixture["lookup"].await_count == 2
    fixture["lookup"].assert_awaited_with("a" * 32, user_id="owner", connection="isolated-file-connection")


def test_changed_manifest_never_falls_back_to_another_version(published_reference):
    """旧清单内容损坏时明确失败，不能换成新版本或给出伪造正文。"""
    fixture = published_reference
    Path(fixture["reference"]["path"]).write_text('{"chunks": []}', encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        upload_read_service.read_published_chunk(fixture["chunk"]["chunk_id"], user_id="owner")
