from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from app.api.routes import system_api
from app.services import lightrag_service
from app.services import paper_file_cleanup_service
from app.services import paper_orphan_reconciliation_service as reconciliation


def test_extract_paper_id_from_lightrag_payload() -> None:
    payload = {
        "content": "[Library ID: default]\n[Paper ID: upload_demo]\n\nDemo Paper"
    }

    assert lightrag_service.extract_paper_id_from_payload(payload) == "upload_demo"


@pytest.mark.asyncio
async def test_reconciliation_defaults_to_dry_run_and_applies_exact_orphans(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    uploads = data_dir / "uploads"
    uploads.mkdir(parents=True)
    valid_file = uploads / "paper-1.pdf"
    orphan_file = uploads / "paper-stale.pdf"
    valid_file.write_bytes(b"valid")
    orphan_file.write_bytes(b"orphan")
    (uploads / "paper-1.mineru").mkdir()
    orphan_artifacts = uploads / "paper-stale.mineru"
    orphan_artifacts.mkdir()
    calls: dict[str, list] = {"milvus": [], "lightrag": [], "neo4j": [], "memory": []}

    class FakePaperRepository:
        async def list_reconciliation_records(self):
            return [{
                "paper_id": "paper-1",
                "user_id": "user-1",
                "library_id": "default",
                "pdf_path": str(valid_file),
                "deletion_status": "active",
                "deletion_task_status": None,
            }]

    class FakeLibraryRepository:
        async def list_libraries(self):
            return [{"library_id": "default"}]

    class FakeMilvusStore:
        def list_paper_ids(self):
            return ["paper-1", "paper-stale"]

        def delete_paper(self, paper_id: str):
            calls["milvus"].append(paper_id)

        def flush_collection(self):
            calls["milvus"].append("flush")

    class FakeLightRAGService:
        def list_document_records(self, *, library_id=None):
            assert library_id == "default"
            return [
                {"doc_id": "paper-1", "paper_id": "paper-1"},
                {"doc_id": "doc-stale", "paper_id": "paper-stale"},
            ]

        async def delete_by_doc_id(self, doc_id: str, *, library_id=None):
            calls["lightrag"].append((library_id, doc_id))
            return {"status": "success"}

    class FakeNeo4jService:
        async def list_paper_ids(self):
            return ["paper-1", "paper-stale"]

        async def delete_paper_subgraph(self, paper_id: str):
            calls["neo4j"].append(paper_id)
            return {"enabled": True}

    class FakeMemoryRepository:
        async def list_orphan_evidence_refs(self, valid_paper_ids):
            assert valid_paper_ids == {"paper-1"}
            return [{"ref_id": "ref-stale", "memory_id": "memory-1", "paper_id": "paper-stale", "user_id": "user-1"}]

        async def remove_orphan_evidence_refs(self, refs):
            calls["memory"].extend(ref["ref_id"] for ref in refs)
            return {
                "deleted_evidence_refs": len(refs),
                "updated_memory_ids": ["memory-1"],
                "stale_memory_ids": ["memory-1"],
            }

    monkeypatch.setattr(reconciliation, "get_paper_repository", lambda: FakePaperRepository())
    monkeypatch.setattr(reconciliation, "get_library_repository", lambda: FakeLibraryRepository())
    monkeypatch.setattr(reconciliation, "get_milvus_store", lambda: FakeMilvusStore())
    monkeypatch.setattr(reconciliation, "get_lightrag_service", lambda: FakeLightRAGService())
    monkeypatch.setattr(reconciliation, "get_neo4j_maintenance_service", lambda: FakeNeo4jService())
    monkeypatch.setattr(reconciliation, "get_research_memory_repository", lambda: FakeMemoryRepository())
    monkeypatch.setattr(
        paper_file_cleanup_service,
        "get_settings",
        lambda: SimpleNamespace(data_dir=str(data_dir)),
    )

    dry_run = await reconciliation.reconcile_paper_orphans()

    assert dry_run["mode"] == "dry_run"
    assert [dry_run[key]["orphan_count"] for key in ("milvus", "lightrag", "neo4j", "files", "memory_evidence_refs")] == [1, 1, 1, 2, 1]
    assert calls == {"milvus": [], "lightrag": [], "neo4j": [], "memory": []}
    assert orphan_file.exists() and orphan_artifacts.exists()

    applied = await reconciliation.reconcile_paper_orphans(apply=True)

    assert applied["mode"] == "apply"
    assert calls == {
        "milvus": ["paper-stale", "flush"],
        "lightrag": [("default", "doc-stale")],
        "neo4j": ["paper-stale"],
        "memory": ["ref-stale"],
    }
    assert not orphan_file.exists() and not orphan_artifacts.exists()
    assert valid_file.exists() and (uploads / "paper-1.mineru").exists()


@pytest.mark.asyncio
async def test_reconciliation_route_requires_developer_and_forwards_apply(monkeypatch) -> None:
    applied: list[bool] = []

    async def fake_reconcile_paper_orphans_payload(*, apply: bool):
        applied.append(apply)
        return {"mode": "apply" if apply else "dry_run"}

    monkeypatch.setattr(system_api.system_facade, "reconcile_paper_orphans_payload", fake_reconcile_paper_orphans_payload)

    with pytest.raises(HTTPException) as exc_info:
        await system_api.reconcile_paper_orphans(False, {"user_id": "user-1", "role": "user"})
    assert exc_info.value.status_code == 403

    payload = await system_api.reconcile_paper_orphans(True, {"user_id": "dev-1", "role": "developer"})

    assert payload == {"mode": "apply"}
    assert applied == [True]
