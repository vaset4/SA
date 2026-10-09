from __future__ import annotations

import app.services.conversation_service as conversation_service
import app.services.paper_db as paper_db


def _paper_payload() -> dict:
    return {
        "paper_id": "paper-1",
        "arxiv_id": "2403.00001",
        "doi": None,
        "title": "Repository Layer Paper",
        "authors": ["Author"],
        "abstract": "abstract",
        "pdf_url": "",
        "pdf_path": None,
        "categories": ["cs.AI"],
        "published_date": "2026-03-24",
        "source": "test",
        "analysis": None,
        "tags": ["phase7"],
    }


def test_paper_db_delegates_to_repositories(monkeypatch):
    events: list[str] = []

    class FakePaperRepository:
        async def ensure_schema(self) -> None:
            events.append("ensure_schema")

        async def add_or_update_paper(self, paper: dict) -> str:
            events.append(f"add:{paper['paper_id']}")
            return paper["paper_id"]

        async def get_paper(self, paper_id: str, user_id: str | None = None) -> dict:
            del user_id
            events.append(f"get:{paper_id}")
            return _paper_payload()

    class FakeNoteRepository:
        async def add_note(
            self,
            user_id: str,
            paper_id: str | None,
            content: str,
            **_: object,
        ) -> str:
            events.append(f"note:{user_id}:{paper_id}:{content}")
            return "note-1"

        async def list_notes(self, user_id: str, paper_id: str | None = None, **_: object) -> list[dict]:
            return [{"note_id": "note-1", "user_id": user_id, "paper_id": paper_id, "content": "memo"}]

        async def delete_notes_by_paper(self, paper_id: str) -> None:
            events.append(f"delete-notes:{paper_id}")

    class FakeNotificationRepository:
        async def list_notifications(self, user_id: str, limit: int = 50) -> list[dict]:
            events.append(f"notifications:{user_id}:{limit}")
            return [{"log_id": "log-1", "read": False}]

        async def set_read_state(self, log_id: str, user_id: str, read: bool) -> bool:
            events.append(f"read:{user_id}:{log_id}:{read}")
            return True

    class FakeTaskRepository:
        async def create_task_run(
            self,
            task_type: str,
            payload: dict,
            message: str,
            *,
            task_id: str | None,
            user_id: str,
        ) -> dict:
            events.append(f"task:{task_type}")
            return {
                "task_id": task_id or "task-1",
                "user_id": user_id,
                "task_type": task_type,
                "status": "queued",
                "progress": 0,
                "message": message,
                "payload": payload,
                "result": None,
                "error": None,
                "cancel_requested": False,
                "worker_id": None,
                "lease_until": None,
                "created_at": "2026-03-24T00:00:00+00:00",
                "updated_at": "2026-03-24T00:00:00+00:00",
                "started_at": None,
                "finished_at": None,
            }

    class FakeObservabilityRepository:
        async def write_log(
            self,
            event_type: str,
            node_name: str,
            query: str,
            result_count: int,
            latency_ms: int,
            session_id: str | None,
            user_id: str | None = None,
        ) -> None:
            events.append(f"log:{event_type}")

    class FakeUserRepository:
        async def ensure_schema(self) -> None:
            events.append("ensure_user_schema")

    monkeypatch.setattr(paper_db, "get_user_repository", lambda: FakeUserRepository())
    monkeypatch.setattr(paper_db, "get_paper_repository", lambda: FakePaperRepository())
    monkeypatch.setattr(paper_db, "get_note_repository", lambda: FakeNoteRepository())
    monkeypatch.setattr(paper_db, "get_notification_repository", lambda: FakeNotificationRepository())
    monkeypatch.setattr(paper_db, "get_task_repository", lambda: FakeTaskRepository())
    monkeypatch.setattr(paper_db, "get_observability_repository", lambda: FakeObservabilityRepository())
    paper_db._initialized = False

    assert paper_db.add_paper(_paper_payload()) == "paper-1"
    assert paper_db.get_paper("paper-1")["title"] == "Repository Layer Paper"
    assert paper_db.add_note("paper-1", "memo") == "note-1"
    assert paper_db.get_notes("paper-1")[0]["content"] == "memo"
    task = paper_db.create_task_run(
        "deep_research",
        {"query": "x"},
        message="queued",
        task_id="task-1",
        user_id="user-1",
    )
    assert task["task_id"] == "task-1"
    paper_db.write_log("push_notification", "node", "query", 1, 20, user_id="user-1")
    assert paper_db.list_notifications("user-1")[0]["log_id"] == "log-1"
    assert paper_db.set_notification_read_state("log-1", "user-1", True) is True
    assert events.count("ensure_schema") == 1
    paper_db.close_db()


def test_conversation_service_supports_repository_injection():
    events: list[str] = []

    class FakeConversationRepository:
        async def ensure_schema(self) -> None:
            events.append("ensure_schema")

        async def list_conversations(self, limit: int, offset: int, sort_by: str) -> list[dict]:
            events.append(f"list:{limit}:{offset}:{sort_by}")
            return [{"id": "conv-1", "title": "T", "is_pinned": False, "tags": [], "associated_papers": []}]

        async def get_conversation(self, conversation_id: str) -> dict:
            events.append(f"get:{conversation_id}")
            return {"id": conversation_id, "title": "T", "messages": [], "associated_papers": []}

        async def create_conversation(self, conversation_id: str, title: str, associated_papers: list[str] | None) -> dict:
            return {"id": conversation_id, "title": title, "associated_papers": associated_papers or []}

        async def update_conversation(self, conversation_id: str, **_: object) -> bool:
            events.append(f"update:{conversation_id}")
            return True

        async def delete_conversation(self, conversation_id: str) -> bool:
            events.append(f"delete:{conversation_id}")
            return True

        async def add_message(
            self,
            conversation_id: str,
            message_id: str,
            role: str,
            content: str,
            title: str | None,
            tone: str | None,
        ) -> dict:
            return {
                "id": message_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "title": title,
                "tone": tone,
            }

        async def set_messages(self, conversation_id: str, messages: list[dict]) -> bool:
            events.append(f"set:{conversation_id}:{len(messages)}")
            return True

        async def add_paper_to_conversation(self, conversation_id: str, paper_id: str) -> bool:
            events.append(f"add-paper:{conversation_id}:{paper_id}")
            return True

        async def remove_paper_from_conversation(self, conversation_id: str, paper_id: str) -> bool:
            events.append(f"remove-paper:{conversation_id}:{paper_id}")
            return True

    db = conversation_service.ConversationDB(repository=FakeConversationRepository())
    assert db.list_conversations(limit=10, offset=0, sort_by="updated_at")[0]["id"] == "conv-1"
    assert db.get_conversation("conv-1")["id"] == "conv-1"
    assert db.create_conversation("conv-2", "New", ["paper-1"])["associated_papers"] == ["paper-1"]
    assert db.update_conversation("conv-1", title="Renamed") is True
    assert db.delete_conversation("conv-1") is True
    assert db.add_message("conv-1", "msg-1", "user", "hello")["id"] == "msg-1"
    assert db.set_messages("conv-1", [{"id": "msg-1", "role": "user", "content": "hello"}]) is True
    assert db.add_paper_to_conversation("conv-1", "paper-1") is True
    assert db.remove_paper_from_conversation("conv-1", "paper-1") is True
    assert events[0] == "ensure_schema"
    conversation_service.close_conversation_db()
