from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI

import app.runtime as runtime_module
import app.runtime.lifespan as runtime_lifespan


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


def test_configure_windows_event_loop_policy_skips_non_windows() -> None:
    class FakeAsyncio:
        class WindowsSelectorEventLoopPolicy:
            pass

        class SelectorEventLoop:
            pass

        def __init__(self) -> None:
            self.policy = None
            self.new_loop = object()

        def set_event_loop_policy(self, policy) -> None:
            self.policy = policy

        def new_event_loop(self):
            return self.new_loop

    fake_asyncio = FakeAsyncio()

    changed = runtime_module.configure_windows_event_loop_policy(
        asyncio_module=fake_asyncio,
        platform_name="linux",
    )

    assert changed is False
    assert fake_asyncio.policy is None


def test_create_compatible_event_loop_uses_selector_on_windows() -> None:
    class FakeAsyncio:
        class SelectorEventLoop:
            pass

        def new_event_loop(self):
            raise AssertionError("Windows should use SelectorEventLoop directly")

    loop = runtime_module.create_compatible_event_loop(
        asyncio_module=FakeAsyncio(),
        platform_name="win32",
    )

    assert isinstance(loop, FakeAsyncio.SelectorEventLoop)


def test_get_uvicorn_loop_config_uses_factory_on_windows() -> None:
    assert (
        runtime_module.get_uvicorn_loop_config(platform_name="win32")
        is runtime_module.create_compatible_event_loop
    )


def test_get_uvicorn_loop_config_keeps_auto_elsewhere() -> None:
    assert runtime_module.get_uvicorn_loop_config(platform_name="linux") == "auto"
