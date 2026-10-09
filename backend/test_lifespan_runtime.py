from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
import pytest

import app.runtime.lifespan as runtime_lifespan
from app.services import jwt_service


@pytest.fixture(autouse=True)
def isolate_upload_startup(monkeypatch):
    """生命周期测试隔离队列端口，禁止启动业务上传消费。"""
    monkeypatch.setattr(runtime_lifespan, "start_upload_worker", lambda: None)


class RecordingRuntime:
    def __init__(self, events: list[str], name: str, result=None) -> None:
        self._events = events
        self._name = name
        self._result = result

    async def initialize(self):
        self._events.append(f"{self._name}:initialize")
        return self._result

    async def close(self) -> None:
        self._events.append(f"{self._name}:close")


def test_lifespan_initializes_and_closes_resources_in_order(monkeypatch) -> None:
    events: list[str] = []
    container = object()
    db_runtime = RecordingRuntime(events, "db")
    conversation_runtime = RecordingRuntime(events, "conversation")
    agent_runtime = RecordingRuntime(events, "agent", result=object())

    async def fake_warm_optional_resources() -> None:
        events.append("optional:warm")

    monkeypatch.setattr(runtime_lifespan, "get_runtime_container", lambda: container)
    monkeypatch.setattr(runtime_lifespan, "get_db_runtime", lambda: db_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_conversation_runtime", lambda: conversation_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_agent_runtime", lambda: agent_runtime)
    monkeypatch.setattr(runtime_lifespan, "_warm_optional_resources", fake_warm_optional_resources)
    monkeypatch.setattr(runtime_lifespan, "validate_startup_auth_configuration", lambda: None)
    monkeypatch.setattr(runtime_lifespan, "start_upload_worker", lambda: events.append("upload:start"))

    app = FastAPI()
    app.state = SimpleNamespace()

    async def exercise_lifespan() -> None:
        async with runtime_lifespan.build_app_lifespan()(app):
            assert app.state.runtime_container is container
            assert app.state.optional_warmup_task is not None

    asyncio.run(exercise_lifespan())

    assert events == [
        "db:initialize",
        "conversation:initialize",
        "agent:initialize",
        "upload:start",
        "optional:warm",
        "agent:close",
        "conversation:close",
        "db:close",
    ]


def test_lifespan_does_not_block_on_slow_optional_warmup(monkeypatch) -> None:
    events: list[str] = []
    container = object()
    db_runtime = RecordingRuntime(events, "db")
    conversation_runtime = RecordingRuntime(events, "conversation")
    agent_runtime = RecordingRuntime(events, "agent", result=object())

    async def fake_warm_optional_resources() -> None:
        await asyncio.sleep(0.2)
        events.append("optional:warm")

    monkeypatch.setattr(runtime_lifespan, "get_runtime_container", lambda: container)
    monkeypatch.setattr(runtime_lifespan, "get_db_runtime", lambda: db_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_conversation_runtime", lambda: conversation_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_agent_runtime", lambda: agent_runtime)
    monkeypatch.setattr(runtime_lifespan, "_warm_optional_resources", fake_warm_optional_resources)
    monkeypatch.setattr(runtime_lifespan, "validate_startup_auth_configuration", lambda: None)

    app = FastAPI()
    app.state = SimpleNamespace()

    async def exercise_lifespan() -> float:
        started_at = time.perf_counter()
        async with runtime_lifespan.build_app_lifespan()(app):
            return time.perf_counter() - started_at

    import time

    elapsed = asyncio.run(exercise_lifespan())

    assert elapsed < 0.1


def test_lifespan_rejects_default_secret_before_runtime_init(monkeypatch) -> None:
    events: list[str] = []
    db_runtime = RecordingRuntime(events, "db")
    conversation_runtime = RecordingRuntime(events, "conversation")
    agent_runtime = RecordingRuntime(events, "agent", result=object())
    settings = type(
        "S",
        (),
        {
            "app_env": "prod",
            "auth_enabled": True,
            "jwt_secret_key": jwt_service.DEFAULT_DEV_SECRET,
        },
    )()

    monkeypatch.setattr(runtime_lifespan, "get_settings", lambda: settings)
    monkeypatch.setattr(runtime_lifespan, "get_runtime_container", lambda: object())
    monkeypatch.setattr(runtime_lifespan, "get_db_runtime", lambda: db_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_conversation_runtime", lambda: conversation_runtime)
    monkeypatch.setattr(runtime_lifespan, "get_agent_runtime", lambda: agent_runtime)

    app = FastAPI()
    app.state = SimpleNamespace()

    async def exercise_lifespan() -> None:
        async with runtime_lifespan.build_app_lifespan()(app):
            return None

    with pytest.raises(Exception) as exc_info:
        asyncio.run(exercise_lifespan())

    assert exc_info.value.status_code == 500
    assert events == []
