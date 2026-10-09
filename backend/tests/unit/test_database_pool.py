from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import app.services.database as database_module


class FakePool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_get_pool_reuses_pool_within_same_loop(monkeypatch) -> None:
    created_pools: list[FakePool] = []

    async def fake_create_pool(**_: object) -> FakePool:
        pool = FakePool(name=f"pool-{len(created_pools) + 1}")
        created_pools.append(pool)
        return pool

    monkeypatch.setattr(database_module, "_pools", {})
    monkeypatch.setattr(database_module, "_load_asyncpg", lambda: SimpleNamespace(create_pool=fake_create_pool))

    async def exercise() -> None:
        pool1 = await database_module.get_pool()
        pool2 = await database_module.get_pool()
        assert pool1 is pool2
        assert len(created_pools) == 1
        await database_module.close_pool()
        assert pool1.closed is True

    asyncio.run(exercise())
    assert database_module._pools == {}


def test_get_pool_isolated_between_event_loops(monkeypatch) -> None:
    created_pools: list[FakePool] = []
    worker_ready = threading.Event()
    worker_can_close = threading.Event()
    worker_pool: dict[str, FakePool] = {}

    async def fake_create_pool(**_: object) -> FakePool:
        pool = FakePool(name=f"pool-{len(created_pools) + 1}")
        created_pools.append(pool)
        return pool

    monkeypatch.setattr(database_module, "_pools", {})
    monkeypatch.setattr(database_module, "_load_asyncpg", lambda: SimpleNamespace(create_pool=fake_create_pool))

    def worker() -> None:
        async def worker_job() -> None:
            worker_pool["value"] = await database_module.get_pool()
            worker_ready.set()
            assert worker_can_close.wait(timeout=5)
            await database_module.close_pool()

        asyncio.run(worker_job())

    thread = threading.Thread(target=worker)
    thread.start()
    assert worker_ready.wait(timeout=5)

    async def main_job() -> None:
        main_pool = await database_module.get_pool()
        assert main_pool is not worker_pool["value"]
        assert len(database_module._pools) == 2
        await database_module.close_pool()
        assert main_pool.closed is True
        assert len(database_module._pools) == 1

    asyncio.run(main_job())

    worker_can_close.set()
    thread.join(timeout=5)

    assert worker_pool["value"].closed is True
    assert len(created_pools) == 2
    assert database_module._pools == {}
