from __future__ import annotations

import asyncio
import importlib
import sys
import types


def test_shared_agent_is_singleton_and_can_close(monkeypatch) -> None:
    calls: list[str] = []

    fake_agents_pkg = types.ModuleType("app.agents")
    fake_agents_pkg.__path__ = []
    fake_graph_factory = types.ModuleType("app.agents.graph_factory")
    fake_graph_factory.create_compiled_graph = lambda checkpointer=None: {"compiled": checkpointer}  # noqa: ARG005
    fake_database = types.ModuleType("app.services.database")
    fake_database.get_database_url = lambda: "postgresql://example"
    fake_checkpoint = types.ModuleType("app.services.checkpoint_service")

    async def fake_read_graph_snapshot(agent, session_id):  # noqa: ARG001
        return {"session_id": session_id}

    fake_checkpoint.read_graph_snapshot = fake_read_graph_snapshot
    monkeypatch.setitem(sys.modules, "app.agents", fake_agents_pkg)
    monkeypatch.setitem(sys.modules, "app.agents.graph_factory", fake_graph_factory)
    monkeypatch.setitem(sys.modules, "app.services.database", fake_database)
    monkeypatch.setitem(sys.modules, "app.services.checkpoint_service", fake_checkpoint)
    sys.modules.pop("app.api.agent_singleton", None)
    agent_singleton = importlib.import_module("app.api.agent_singleton")

    class FakeManager:
        async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            calls.append("manager:closed")

    async def fake_build_agent():
        calls.append("build")
        return {"graph": "ok"}, FakeManager()

    monkeypatch.setattr(agent_singleton, "_build_agent", fake_build_agent)
    agent_singleton._agent = None
    agent_singleton._checkpoint_manager = None

    async def exercise() -> None:
        first = await agent_singleton.get_shared_agent()
        second = await agent_singleton.get_shared_agent()
        assert first == {"graph": "ok"}
        assert second is first
        await agent_singleton.close_shared_agent()

    asyncio.run(exercise())

    assert calls == ["build", "manager:closed"]
    assert agent_singleton._agent is None
    assert agent_singleton._checkpoint_manager is None
