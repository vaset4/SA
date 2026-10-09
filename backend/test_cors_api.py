from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["SSLKEYLOGFILE"] = ""

import app.api.server as server_module
import app.runtime.lifespan as runtime_lifespan
from app.api.server import app


ALLOWED_LOOPBACK_ORIGINS = [
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost:3001",
]
DISALLOWED_ORIGIN = "http://192.168.0.106:3001"


def _preflight_headers(origin: str) -> dict[str, str]:
    return {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }


@pytest.fixture()
def client(monkeypatch):
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

    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("origin", ALLOWED_LOOPBACK_ORIGINS)
@pytest.mark.parametrize("path", ["/chat", "/search/arxiv"])
def test_preflight_accepts_loopback_origin_on_any_local_port(client, path, origin):
    response = client.options(path, headers=_preflight_headers(origin))

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize("path", ["/chat", "/search/arxiv"])
def test_preflight_rejects_non_loopback_origin(client, path):
    response = client.options(path, headers=_preflight_headers(DISALLOWED_ORIGIN))

    assert response.status_code == 400
