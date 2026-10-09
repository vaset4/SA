import asyncio

import pytest

from app.services import project_service
from app.services.project_scope_service import build_scope_filter_expr


@pytest.fixture(autouse=True)
def _isolate_project_service_unit_dependencies(monkeypatch):
    """隔离 project 单测的数据库、论文库与 library schema 外部依赖。"""
    monkeypatch.setattr(project_service.paper_db, "init_db", lambda: None)
    monkeypatch.setattr(
        project_service.paper_db,
        "get_paper",
        lambda paper_id, user_id=None: {"paper_id": paper_id, "title": paper_id},
    )
    monkeypatch.setattr(
        project_service.library_service,
        "ensure_library_schema",
        lambda: asyncio.sleep(0),
    )
    monkeypatch.setattr(project_service.library_service, "require_library", _require_library)
    monkeypatch.setattr(
        project_service.project_activity_service,
        "ensure_project_activity_schema",
        lambda: asyncio.sleep(0),
    )


class FakeProjectRepository:
    def __init__(self) -> None:
        self.project = None
        self.scope = None
        self.paper_links: list[dict] = []

    async def ensure_schema(self) -> None:
        return None

    async def create_project(self, payload: dict):
        self.project = {
            "project_id": "proj_demo",
            "name": payload["name"],
            "description": payload.get("description", ""),
            "owner": payload.get("owner", "system"),
            "status": payload.get("status", "active"),
            "default_library_id": payload.get("default_library_id", "default"),
            "created_at": "2026-03-26T00:00:00+00:00",
            "updated_at": "2026-03-26T00:00:00+00:00",
        }
        return self.project

    async def get_project(self, project_id: str, *, user_id: str | None = None):
        return self.project if self.project and self.project["project_id"] == project_id else None

    async def list_projects(self, *, status: str | None = None, user_id: str | None = None):
        return [self.project] if self.project else []

    async def update_project(self, project_id: str, payload: dict, *, user_id: str | None = None):
        if self.project is None:
            return None
        self.project = {**self.project, **payload}
        return self.project

    async def delete_project(self, project_id: str, *, user_id: str | None = None):
        if self.project and self.project["project_id"] == project_id:
            self.project = None
            return True
        return False

    async def get_scope(self, project_id: str):
        return self.scope

    async def upsert_scope(self, project_id: str, payload: dict):
        self.scope = {
            "project_id": project_id,
            "library_ids": payload.get("library_ids", []),
            "paper_ids": payload.get("paper_ids", []),
            "excluded_paper_ids": payload.get("excluded_paper_ids", []),
            "year_from": payload.get("year_from"),
            "year_to": payload.get("year_to"),
            "task_filters": payload.get("task_filters", []),
            "method_filters": payload.get("method_filters", []),
            "created_at": "",
            "updated_at": "",
        }
        return self.scope

    async def link_paper(self, project_id: str, paper_id: str, *, role: str = "core", note: str = ""):
        link = {"project_id": project_id, "paper_id": paper_id, "role": role, "note": note}
        self.paper_links.append(link)
        return link

    async def unlink_paper(self, project_id: str, paper_id: str) -> None:
        self.paper_links = [item for item in self.paper_links if item["paper_id"] != paper_id]

    async def list_project_papers(self, project_id: str):
        return list(self.paper_links)

    async def link_session(self, project_id: str, session_id: str):
        return {"project_id": project_id, "session_id": session_id}

    async def list_project_sessions(self, project_id: str):
        return []


async def _require_library(library_id: str | None, *, owner: str | None = None):
    return {"library_id": library_id or "default"}


def test_create_project_initializes_scope(monkeypatch) -> None:
    repository = FakeProjectRepository()
    monkeypatch.setattr(project_service, "_schema_ready", False)
    monkeypatch.setattr(project_service, "get_project_repository", lambda: repository)
    monkeypatch.setattr(project_service.library_service, "ensure_library_schema", lambda: asyncio.sleep(0))
    monkeypatch.setattr(project_service.library_service, "require_library", _require_library)

    project = asyncio.run(project_service.create_project({"name": "NLP 项目", "default_library_id": "default"}))
    scope = asyncio.run(project_service.get_project_scope(project["project_id"]))

    assert project["project_id"] == "proj_demo"
    assert scope["library_ids"] == ["default"]


def test_add_project_paper_syncs_scope(monkeypatch) -> None:
    repository = FakeProjectRepository()
    repository.project = {
        "project_id": "proj_demo",
        "name": "NLP 项目",
        "description": "",
        "owner": "system",
        "status": "active",
        "default_library_id": "default",
        "created_at": "",
        "updated_at": "",
    }
    repository.scope = {
        "project_id": "proj_demo",
        "library_ids": ["default"],
        "paper_ids": [],
        "excluded_paper_ids": [],
        "year_from": None,
        "year_to": None,
        "task_filters": [],
        "method_filters": [],
        "created_at": "",
        "updated_at": "",
    }
    monkeypatch.setattr(project_service, "_schema_ready", True)
    monkeypatch.setattr(project_service, "get_project_repository", lambda: repository)

    asyncio.run(project_service.add_project_paper("proj_demo", "paper_1"))
    scope = asyncio.run(project_service.get_project_scope("proj_demo"))
    assert scope["paper_ids"] == ["paper_1"]


def test_build_scope_filter_expr_includes_paper_ids_and_year() -> None:
    filter_expr = build_scope_filter_expr(
        {
            "project_id": "proj_demo",
            "library_ids": ["default"],
            "paper_ids": ["paper_1", "paper_2"],
            "excluded_paper_ids": [],
            "year_from": 2020,
            "year_to": 2024,
            "task_filters": [],
            "method_filters": [],
        }
    )
    assert 'paper_id == "paper_1"' in filter_expr
    assert "year >= 2020" in filter_expr
    assert "year <= 2024" in filter_expr


def test_attach_session_resolves_conversation_id_to_real_session(monkeypatch) -> None:
    repository = FakeProjectRepository()
    repository.project = {
        "project_id": "proj_demo",
        "name": "NLP 项目",
        "description": "",
        "owner": "system",
        "status": "active",
        "default_library_id": "default",
        "created_at": "",
        "updated_at": "",
    }
    linked: dict[str, str] = {}

    async def fake_link_session(project_id: str, session_id: str):
        linked["project_id"] = project_id
        linked["session_id"] = session_id
        return {"project_id": project_id, "session_id": session_id}

    repository.link_session = fake_link_session  # type: ignore[method-assign]

    monkeypatch.setattr(project_service, "_schema_ready", True)
    monkeypatch.setattr(project_service, "get_project_repository", lambda: repository)
    monkeypatch.setattr(project_service.paper_db, "get_session", lambda session_id, user_id=None: None if session_id == "conv-demo" else {"session_id": session_id})
    monkeypatch.setattr(
        project_service.conversation_service,
        "get_conversation",
        lambda conversation_id, user_id="anonymous": asyncio.sleep(0, result={"id": conversation_id, "session_id": "real-session-1"}),
    )

    result = asyncio.run(project_service.attach_session("proj_demo", "conv-demo"))

    assert linked["session_id"] == "real-session-1"
    assert result["session_id"] == "real-session-1"


def test_add_project_paper_appends_activity(monkeypatch) -> None:
    repository = FakeProjectRepository()
    repository.project = {
        "project_id": "proj_demo",
        "name": "NLP 项目",
        "description": "",
        "owner": "system",
        "status": "active",
        "default_library_id": "default",
        "created_at": "",
        "updated_at": "",
    }
    repository.scope = {
        "project_id": "proj_demo",
        "library_ids": ["default"],
        "paper_ids": [],
        "excluded_paper_ids": [],
        "year_from": None,
        "year_to": None,
        "task_filters": [],
        "method_filters": [],
        "created_at": "",
        "updated_at": "",
    }
    captured: list[dict] = []

    async def fake_append_activity(project_id: str, event_type: str, payload: dict):
        captured.append({"project_id": project_id, "event_type": event_type, "payload": payload})
        return {"project_id": project_id, "event_type": event_type, "payload": payload}

    monkeypatch.setattr(project_service, "_schema_ready", True)
    monkeypatch.setattr(project_service, "get_project_repository", lambda: repository)
    monkeypatch.setattr(project_service.project_activity_service, "append_activity", fake_append_activity)
    monkeypatch.setattr(project_service.paper_db, "get_paper", lambda paper_id, user_id=None: {"paper_id": paper_id, "title": "Test Paper"})

    asyncio.run(project_service.add_project_paper("proj_demo", "paper_1"))

    assert captured == [
        {
            "project_id": "proj_demo",
            "event_type": "paper_added",
            "payload": {"paper_id": "paper_1", "title": "Test Paper"},
        }
    ]


def test_attach_session_appends_internal_research_activity(monkeypatch) -> None:
    repository = FakeProjectRepository()
    repository.project = {
        "project_id": "proj_demo",
        "name": "NLP 项目",
        "description": "",
        "owner": "system",
        "status": "active",
        "default_library_id": "default",
        "created_at": "",
        "updated_at": "",
    }
    captured: list[dict] = []

    async def fake_append_activity(project_id: str, event_type: str, payload: dict):
        captured.append({"project_id": project_id, "event_type": event_type, "payload": payload})
        return {"project_id": project_id, "event_type": event_type, "payload": payload}

    monkeypatch.setattr(project_service, "_schema_ready", True)
    monkeypatch.setattr(project_service, "get_project_repository", lambda: repository)
    monkeypatch.setattr(project_service.project_activity_service, "append_activity", fake_append_activity)
    monkeypatch.setattr(
        project_service.paper_db,
        "get_session",
        lambda session_id, user_id=None: {
            "session_id": session_id,
            "topic": "研究会话",
            "paper_ids": ["paper-1", "paper-2"],
        },
    )

    asyncio.run(project_service.attach_session("proj_demo", "session-1"))

    assert captured == [
        {
            "project_id": "proj_demo",
            "event_type": "internal_research_started",
            "payload": {
                "session_id": "session-1",
                "topic": "研究会话",
                "paper_ids": ["paper-1", "paper-2"],
            },
        }
    ]


def test_attach_session_rejects_unreadable_link_write(monkeypatch) -> None:
    """关联 INSERT 后读不到关系时必须报错，不能把空字典当成成功。"""
    repository = FakeProjectRepository()
    repository.project = {
        "project_id": "proj_demo",
        "name": "NLP 项目",
        "owner": "user-1",
        "status": "active",
        "default_library_id": "default",
    }

    async def fake_link_session(project_id: str, session_id: str) -> dict:
        """模拟数据库写后无法读回关联。"""
        return {}

    repository.link_session = fake_link_session  # type: ignore[method-assign]
    monkeypatch.setattr(project_service, "_schema_ready", True)
    monkeypatch.setattr(project_service, "get_project_repository", lambda: repository)

    with pytest.raises(RuntimeError, match="关联写入后无法读取"):
        asyncio.run(project_service.attach_session(
            "proj_demo",
            "session-1",
            user_id="user-1",
            session_row={"session_id": "session-1", "user_id": "user-1"},
        ))


def test_delete_project_removes_owned_project(monkeypatch) -> None:
    repository = FakeProjectRepository()
    repository.project = {
        "project_id": "proj_demo",
        "name": "NLP 项目",
        "description": "",
        "owner": "system",
        "status": "active",
        "default_library_id": "default",
        "created_at": "",
        "updated_at": "",
    }
    monkeypatch.setattr(project_service, "_schema_ready", True)
    monkeypatch.setattr(project_service, "get_project_repository", lambda: repository)

    asyncio.run(project_service.delete_project("proj_demo"))

    assert repository.project is None
