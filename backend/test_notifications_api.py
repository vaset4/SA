from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

os.environ["SSLKEYLOGFILE"] = ""

from app.api.server import app
import app.runtime.lifespan as runtime_lifespan
import app.services.paper_db as paper_db


def _db_file(tmp_path) -> str:
    return str(tmp_path / "papers.db")


def _insert_push_notification(db_file: str) -> str:
    paper_db.write_log(
        event_type="push_notification",
        node_name="recommendation",
        query="test notification",
        result_count=1,
        latency_ms=12,
    )
    with sqlite3.connect(db_file) as connection:
        row = connection.execute(
            "SELECT log_id FROM logs WHERE event_type='push_notification' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    return row[0]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    class NoopRuntime:
        async def initialize(self):
            return None

        async def close(self) -> None:
            return None

    noop_runtime = NoopRuntime()
    monkeypatch.setattr(runtime_lifespan, "get_agent_runtime", lambda: noop_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_db_runtime", lambda: noop_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_conversation_runtime", lambda: noop_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_runtime_container", lambda: object())
    monkeypatch.setattr(paper_db.settings, "data_dir", str(tmp_path))
    paper_db.init_db()
    with TestClient(app) as test_client:
        yield test_client, _db_file(tmp_path)


def test_mark_notification_read_persists_to_database(client):
    test_client, db_file = client
    log_id = _insert_push_notification(db_file)

    before = test_client.get("/notifications")
    assert before.status_code == 200
    before_payload = before.json()
    before_item = next(item for item in before_payload["notifications"] if item["log_id"] == log_id)
    assert before_item["read"] is False

    response = test_client.patch(f"/notifications/{log_id}", json={"read": True})
    assert response.status_code == 200
    assert response.json() == {"log_id": log_id, "read": True}

    after = test_client.get("/notifications")
    assert after.status_code == 200
    after_payload = after.json()
    after_item = next(item for item in after_payload["notifications"] if item["log_id"] == log_id)
    assert after_item["read"] is True

    with sqlite3.connect(db_file) as connection:
        row = connection.execute(
            "SELECT is_read FROM logs WHERE log_id = ?",
            (log_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == 1
