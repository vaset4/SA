from __future__ import annotations

import asyncio
import inspect
import os

import httpx
import pytest
from fastapi import FastAPI
from fastapi.params import Depends

os.environ["SSLKEYLOGFILE"] = ""

import app.api.facades.citation_facade as citation_facade
import app.api.routes.citation_api as citation_api
from app.middleware.auth import get_current_user_with_rate_limit


def test_citation_get_declares_authentication_dependency() -> None:
    signature = inspect.signature(citation_api.citations)
    user_default = signature.parameters["user"].default

    assert isinstance(user_default, Depends)
    assert user_default.dependency is get_current_user_with_rate_limit


def test_citation_get_rejects_anonymous_request() -> None:
    app = FastAPI()
    app.include_router(citation_api.router)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/citations/paper-1")

    response = asyncio.run(request())

    assert response.status_code == 401


def test_citation_route_passes_user_scope(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_payload(paper_id, *, depth=1, user_id="anonymous", include_graph_paths=False):  # noqa: ANN001
        captured.update(
            {
                "paper_id": paper_id,
                "depth": depth,
                "user_id": user_id,
                "include_graph_paths": include_graph_paths,
            }
        )
        return {"paper_id": paper_id}

    monkeypatch.setattr(citation_api, "get_citation_tree_payload", fake_payload)

    payload = asyncio.run(
        citation_api.citations(
            "paper-1",
            depth=2,
            include_graph_paths=True,
            user={"user_id": "user-1"},
        )
    )

    assert payload == {"paper_id": "paper-1"}
    assert captured == {
        "paper_id": "paper-1",
        "depth": 2,
        "user_id": "user-1",
        "include_graph_paths": True,
    }


def test_citation_route_maps_cross_user_scope_to_not_found(monkeypatch) -> None:
    async def fake_payload(*args, **kwargs):  # noqa: ANN001
        raise LookupError("论文不存在")

    monkeypatch.setattr(citation_api, "get_citation_tree_payload", fake_payload)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(citation_api.citations("paper-2", user={"user_id": "user-1"}))

    assert exc_info.value.status_code == 404


def test_citation_facade_checks_owner_before_fetching_tree(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        citation_facade.paper_facade,
        "get_paper_or_none",
        lambda paper_id, *, user_id="anonymous": None,
    )

    def fake_tree(paper_id, depth=1):  # noqa: ANN001
        calls.append(paper_id)
        return {"paper_id": paper_id}

    monkeypatch.setattr(citation_facade, "get_citation_tree", fake_tree)

    with pytest.raises(LookupError):
        asyncio.run(citation_facade.get_citation_tree_payload("paper-2", user_id="user-1"))

    assert calls == []


def test_citation_facade_returns_owner_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        citation_facade.paper_facade,
        "get_paper_or_none",
        lambda paper_id, *, user_id="anonymous": {"paper_id": paper_id, "user_id": user_id},
    )
    monkeypatch.setattr(
        citation_facade,
        "get_citation_tree",
        lambda paper_id, depth=1: {"paper_id": paper_id, "depth": depth},
    )

    payload = asyncio.run(citation_facade.get_citation_tree_payload("paper-1", depth=2, user_id="user-1"))

    assert payload == {"paper_id": "paper-1", "depth": 2}
