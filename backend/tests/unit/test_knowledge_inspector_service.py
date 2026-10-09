"""Unit tests for the chunk/vector inspector service."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.routes import knowledge_inspector_api
from app.models.knowledge_inspector_contracts import ChunkDetailResponse, ChunkMetadata
from app.services import knowledge_inspector_service as service
from types import SimpleNamespace
from unittest.mock import Mock


class FakeInspectorStore:
    def __init__(self, rows: list[dict] | None = None, missing_read_ids: set[str] | None = None) -> None:
        self.rows = rows or []
        self.missing_read_ids = missing_read_ids or set()

    def get_chunk_row(self, chunk_id: str) -> dict | None:
        return next((row for row in self.rows if row.get("chunk_id") == chunk_id), None)

    def list_chunk_rows_by_paper(self, paper_id: str) -> list[dict]:
        return [row for row in self.rows if row.get("paper_id") == paper_id]

    def read_chunk(self, chunk_id: str, already_read: set[str]) -> dict:
        if chunk_id in self.missing_read_ids:
            return {"skipped": True, "full_text": f"(chunk {chunk_id} not found)"}
        return {"skipped": False, "full_text": "parent text"}


def paper_provider(paper_id: str):
    if paper_id == "paper-1":
        return {"paper_id": paper_id, "title": "Paper One", "is_vectorized": True}
    if paper_id == "vectorized-empty":
        return {"paper_id": paper_id, "title": "Empty", "is_vectorized": True}
    return None


def row(**overrides):
    base = {
        "chunk_id": "chunk-1",
        "paper_id": "paper-1",
        "parent_id": "parent-1",
        "embedding_text": "embedding text",
        "display_text": "display text",
        "full_text": "full parent text",
        "section": "Methods",
        "section_path": "3 Methods",
        "page_num": 2,
        "block_type": "paragraph",
        "asset_ref": "",
        "parser_mode": "mineru_online",
    }
    return {**base, **overrides}


def test_chunk_detail_defaults_to_previews_and_missing_fields():
    store = FakeInspectorStore([row(section_path="", page_num=0)])

    detail = service.get_chunk_detail(
        "chunk-1",
        store=store,
        paper_provider=paper_provider,
    )

    assert detail.vector_row_found is True
    assert detail.paper and detail.paper.title == "Paper One"
    assert "section_path" in detail.missing_fields
    assert "page_num" in detail.missing_fields
    assert all(text.text is None for text in detail.text_fields)


def test_chunk_detail_full_text_reports_truncation():
    store = FakeInspectorStore([row(full_text="abcdef")])

    detail = service.get_chunk_detail(
        "chunk-1",
        include_text="full",
        max_text_chars=3,
        store=store,
        paper_provider=paper_provider,
    )

    full_text = next(field for field in detail.text_fields if field.name == "full_text")
    assert full_text.text == "abc"
    assert full_text.returned_chars == 3
    assert full_text.total_chars == 6
    assert full_text.truncated is True


def test_missing_chunk_detail_is_explicit_not_found_shape():
    detail = service.get_chunk_detail(
        "missing",
        store=FakeInspectorStore([]),
        paper_provider=paper_provider,
    )

    assert detail.vector_row_found is False
    assert "chunk_id" in detail.missing_fields


def test_paper_chunk_map_groups_by_section_then_parent():
    rows = [
        row(chunk_id="c2", parent_id="p2", section_path="4 Results", page_num=5),
        row(chunk_id="c1", parent_id="p1", section_path="3 Methods", page_num=2),
        row(chunk_id="c3", parent_id="p1", section_path="3 Methods", page_num=3),
    ]

    chunk_map = service.get_paper_chunk_map(
        "paper-1",
        store=FakeInspectorStore(rows),
        paper_provider=paper_provider,
    )

    assert chunk_map.map_type == "metadata_grouped"
    assert [section.section_path for section in chunk_map.sections] == ["3 Methods", "4 Results"]
    assert chunk_map.sections[0].parents[0].parent_id == "p1"
    assert [child.chunk_id for child in chunk_map.sections[0].parents[0].chunks] == ["c1", "c3"]


def test_audit_reports_bad_rows_and_parent_inconsistency():
    rows = [
        row(chunk_id="dup", parent_id="parent-1", full_text="full A", section_path="", page_num=0),
        row(chunk_id="dup", parent_id="parent-1", full_text="full B", display_text=""),
        row(chunk_id="missing-read", parent_id="", full_text=""),
    ]

    audit = service.get_paper_chunk_audit(
        "paper-1",
        store=FakeInspectorStore(rows, missing_read_ids={"missing-read"}),
        paper_provider=paper_provider,
    )
    codes = {issue.code for issue in audit.issues}

    assert audit.summary.status == "failed"
    assert "duplicate_chunk_id" in codes
    assert "missing_parent_id" in codes
    assert "missing_full_text" in codes
    assert "missing_section_path" in codes
    assert "invalid_page_num" in codes
    assert "inconsistent_parent_full_text" in codes
    assert "read_chunk_missing" in codes


def test_audit_ignores_intentional_embedding_section_prefix():
    rows = [
        row(
            embedding_text="3 Methods\nBody **text** with _formatting_.",
            display_text="Body text with formatting.",
            section_path="3 Methods",
        ),
    ]

    audit = service.get_paper_chunk_audit(
        "paper-1",
        store=FakeInspectorStore(rows),
        paper_provider=paper_provider,
    )

    assert "embedding_display_text_drift" not in {issue.code for issue in audit.issues}


def test_audit_reports_real_embedding_display_text_drift():
    rows = [
        row(
            embedding_text="3 Methods\nCompletely unrelated vector content that does not match the displayed row.",
            display_text="Short shown text.",
            section_path="3 Methods",
        ),
    ]

    audit = service.get_paper_chunk_audit(
        "paper-1",
        store=FakeInspectorStore(rows),
        paper_provider=paper_provider,
    )

    assert "embedding_display_text_drift" in {issue.code for issue in audit.issues}


def test_audit_reports_vectorized_paper_without_rows():
    audit = service.get_paper_chunk_audit(
        "vectorized-empty",
        store=FakeInspectorStore([]),
        paper_provider=paper_provider,
    )

    assert audit.summary.status == "failed"
    assert audit.issues[0].code == "vectorized_paper_without_rows"


def test_audit_reports_milvus_rows_without_paper_metadata():
    audit = service.get_paper_chunk_audit(
        "paper-missing",
        store=FakeInspectorStore([row(paper_id="paper-missing")]),
        paper_provider=paper_provider,
    )

    assert audit.summary.status == "failed"
    assert audit.issues[0].code == "paper_missing_in_db"


def test_upload_audit_summary_retains_structural_findings(monkeypatch):
    """审计结果保留真实状态和问题数量，不依赖旧同步上传返回体。"""
    audit = service.get_paper_chunk_audit("paper-1", store=FakeInspectorStore([row()]), paper_provider=paper_provider)
    monkeypatch.setattr(service, "get_paper_chunk_audit", lambda paper_id: audit)
    summary = service.build_upload_chunk_audit_summary("paper-1")
    assert summary["chunk_audit_status"] == audit.summary.status
    assert summary["chunk_audit_error_count"] == audit.summary.error_count
    assert summary["chunk_audit_warning_count"] == audit.summary.warning_count


@pytest.mark.asyncio
async def test_api_missing_chunk_returns_404(monkeypatch):
    def fake_detail(*args, **kwargs):
        return ChunkDetailResponse(
            chunk_id="missing",
            vector_row_found=False,
            metadata=ChunkMetadata(),
        )

    monkeypatch.setattr(knowledge_inspector_api, "get_milvus_store",
        lambda: SimpleNamespace(for_user=Mock(return_value=object())))
    monkeypatch.setattr(knowledge_inspector_api.inspector, "get_chunk_detail", fake_detail)
    with pytest.raises(HTTPException) as exc:
        await knowledge_inspector_api.get_chunk_detail(
            "missing",
            user={"role": "developer", "user_id": "u1"},
        )

    assert exc.value.status_code == 404


def test_upload_audit_failure_is_explicit(monkeypatch):
    """审计 IO 失败不能被包装为已完成上传或空的成功审计。"""
    def fail_audit(paper_id):
        """模拟不可用的实际审计依赖。"""
        raise RuntimeError("milvus unavailable")

    monkeypatch.setattr(service, "get_paper_chunk_audit", fail_audit)
    with pytest.raises(RuntimeError, match="milvus unavailable"):
        service.build_upload_chunk_audit_summary("paper-1")
