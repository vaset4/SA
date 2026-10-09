from __future__ import annotations

import pytest
from types import SimpleNamespace

from app.models.research_contracts import ScopeOwnershipError
from app.services import project_scope_service
from app.services import upload_read_service


async def _fake_publication_snapshot(paper_ids, *, user_id, allow_empty):
    """隔离发布仓储端口；不为范围单测连接业务数据库。"""
    return {"lease_id": "scope-test", "user_id": user_id,
        "batch_ids": ["batch-" + paper_id for paper_id in paper_ids],
        "paper_batches": {paper_id: "batch-" + paper_id for paper_id in paper_ids},
        "collection_names": ["isolated"], "embedding_fingerprints": ["isolated"]}


async def _fake_require_project(project_id: str, *, user_id: str = "anonymous") -> dict:
    return {
        "project_id": project_id,
        "default_library_id": "default",
        "user_id": user_id,
    }


async def _fake_project_scope(project_id: str, *, user_id: str = "anonymous") -> dict:
    return {
        "project_id": project_id,
        "library_ids": ["default"],
        "paper_ids": ["paper-in-project"],
        "excluded_paper_ids": [],
        "year_from": None,
        "year_to": None,
        "task_filters": [],
        "method_filters": [],
    }


class _FakePaperRepository:
    """记录批量归属查询，避免测试退回逐篇兼容调用。"""

    def __init__(self, owned_paper_ids: set[str]) -> None:
        self.owned_paper_ids = owned_paper_ids
        self.calls: list[tuple[list[str], str]] = []

    async def get_papers_by_ids(self, paper_ids: list[str], *, user_id: str) -> list[dict]:
        self.calls.append((paper_ids, user_id))
        return [
            {"paper_id": paper_id, "user_id": user_id}
            for paper_id in paper_ids
            if paper_id in self.owned_paper_ids
        ]


def _patch_scope(monkeypatch, owned_paper_ids: set[str]) -> _FakePaperRepository:
    monkeypatch.setattr(upload_read_service, "get_upload_publication_repository",
        lambda: SimpleNamespace(snapshot=_fake_publication_snapshot))
    repository = _FakePaperRepository(owned_paper_ids)
    monkeypatch.setattr(project_scope_service, "require_project", _fake_require_project)
    monkeypatch.setattr(project_scope_service, "get_project_scope", _fake_project_scope)
    monkeypatch.setattr(project_scope_service, "get_paper_repository", lambda: repository)
    monkeypatch.setattr(
        project_scope_service.library_service,
        "require_library",
        lambda *args, **kwargs: _fake_require_project("library", user_id="user-1"),
    )
    return repository


@pytest.mark.asyncio
async def test_build_project_context_accepts_explicit_project_paper(monkeypatch) -> None:
    _patch_scope(monkeypatch, {"paper-in-project"})

    context = await project_scope_service.build_project_context(
        "project-1",
        requested_paper_ids=["paper-in-project"],
        user_id="user-1",
    )

    assert context["paper_ids"] == ["paper-in-project"]
    assert context["accepted_paper_ids"] == ["paper-in-project"]
    assert context["rejected_paper_ids"] == []


@pytest.mark.asyncio
async def test_build_project_context_accepts_user_owned_paper_outside_project(monkeypatch) -> None:
    _patch_scope(monkeypatch, {"paper-in-project", "paper-outside-project"})

    context = await project_scope_service.build_project_context(
        "project-1",
        requested_paper_ids=["paper-outside-project"],
        user_id="user-1",
    )

    assert context["paper_ids"] == ["paper-outside-project"]
    assert 'paper_id == "paper-outside-project"' in context["filter_expr"]
    assert context["accepted_paper_ids"] == ["paper-outside-project"]
    assert context["rejected_paper_ids"] == []


@pytest.mark.asyncio
async def test_build_project_context_rejects_unowned_explicit_paper(monkeypatch) -> None:
    _patch_scope(monkeypatch, {"paper-in-project"})

    with pytest.raises(ScopeOwnershipError) as exc_info:
        await project_scope_service.build_project_context(
            "project-1",
            requested_paper_ids=["paper-private"],
            user_id="user-1",
        )

    assert str(exc_info.value) == "论文范围不存在或无权访问"


@pytest.mark.asyncio
async def test_explicit_scope_uses_one_owner_aware_batch_query(monkeypatch) -> None:
    repository = _patch_scope(monkeypatch, {"paper-1", "paper-2"})

    context = await project_scope_service.build_project_context(
        None,
        requested_paper_ids=["paper-2", "paper-1"],
        user_id="user-1",
    )

    assert context["paper_ids"] == ["paper-2", "paper-1"]
    assert repository.calls == [(["paper-2", "paper-1"], "user-1")]
