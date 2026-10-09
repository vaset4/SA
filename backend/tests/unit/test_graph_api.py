from __future__ import annotations

import pytest

import app.api.routes.graph_api as graph_api
from app.api.schemas.controversy import ControversyMapRequest
from app.api.schemas.graph_admin import Neo4jBackfillLibraryRequest, Neo4jRebuildPaperRequest


class FakeEvolutionService:
    async def get_evolution(self, query):
        return {
            "focus": {
                "key": query["paper_id"] or query["method"],
                "label": "Focus",
                "node_type": query["view_mode"],
                "view_mode": query["view_mode"],
            },
            "timeline": [],
            "nodes": [],
            "edges": [],
            "stats": {"node_count": 0},
            "filters": query,
        }

    async def get_evolution_export(self, query):
        return await self.get_evolution(query)

    async def get_evolution_entry(self, paper_id: str, *, project_id: str = "", resolved_scope=None):
        return {
            "focus": {"key": paper_id, "label": paper_id, "node_type": "paper", "view_mode": "paper"},
            "recommended_filters": {"project_id": project_id},
            "target_url": f"/projects/{project_id}/graph" if project_id else "/graph",
        }


class FakeControversyService:
    async def build_map(self, query):
        return {
            "conflicts": [],
            "gaps": [],
            "overview": "ok",
            "scope": query,
        }


class FakeMaintenanceService:
    async def rebuild_paper_subgraph(self, paper_id: str):
        return {"paper_id": paper_id, "node_count": 3, "edge_count": 2}

    async def backfill_library(self, library_id: str):
        return {"library_id": library_id, "paper_count": 2}


class FakePaperGraphService:
    async def get_paper_graph(self, *, project_id: str = "", library_id: str = "", resolved_scope=None, limit: int = 200):
        return {
            "nodes": [{"id": "paper:p1", "paper_id": "p1", "title": "Paper 1", "year": 2024, "library_id": library_id or "default"}],
            "edges": [],
            "meta": {
                "project_id": project_id,
                "library_id": library_id,
                "node_count": 1,
                "edge_count": 0,
                "scoped_paper_count": len((resolved_scope or {}).get("paper_ids", [])),
                "relation_counts": {},
            },
        }


def _setup(monkeypatch) -> None:
    async def fake_resolve_project_scope(project_id, *, user_id: str = "anonymous"):
        return {"project_id": project_id, "paper_ids": ["p1"]}

    monkeypatch.setattr(graph_api, "get_evolution_graph_service", lambda: FakeEvolutionService())
    monkeypatch.setattr(graph_api, "get_controversy_map_service", lambda: FakeControversyService())
    monkeypatch.setattr(graph_api, "get_neo4j_maintenance_service", lambda: FakeMaintenanceService())
    monkeypatch.setattr(graph_api, "get_paper_graph_service", lambda: FakePaperGraphService())
    monkeypatch.setattr(graph_api, "resolve_project_scope", fake_resolve_project_scope)
    monkeypatch.setattr(graph_api.paper_db, "get_paper", lambda paper_id, user_id=None: {"paper_id": paper_id, "title": "Paper 1"})


@pytest.mark.asyncio
async def test_graph_evolution_route(monkeypatch) -> None:
    _setup(monkeypatch)
    response = await graph_api.graph_evolution(
        view_mode="paper",
        paper_id="p1",
        method="",
        task="",
        dataset="",
        from_year=None,
        to_year=None,
        depth=1,
        project_id="proj_demo",
        user={"user_id": "tester"},
    )
    assert response["filters"]["paper_id"] == "p1"
    assert response["filters"]["project_id"] == "proj_demo"
    assert response["filters"]["resolved_scope"]["paper_ids"] == ["p1"]


@pytest.mark.asyncio
async def test_controversy_map_route(monkeypatch) -> None:
    _setup(monkeypatch)
    response = await graph_api.controversy_map(
        ControversyMapRequest(paper_ids=["p1", "p2"], topic="qa", project_id="proj_demo"),
        user={"user_id": "tester"},
    )
    assert response["overview"] == "ok"
    assert response["scope"]["project_id"] == "proj_demo"
    assert response["scope"]["resolved_scope"]["paper_ids"] == ["p1"]


@pytest.mark.asyncio
async def test_neo4j_rebuild_and_backfill_routes(monkeypatch) -> None:
    _setup(monkeypatch)
    rebuild = await graph_api.rebuild_paper_graph(
        Neo4jRebuildPaperRequest(paper_id="p1"),
        user={"user_id": "tester"},
    )
    backfill = await graph_api.backfill_library_graph(
        Neo4jBackfillLibraryRequest(library_id="default"),
        user={"user_id": "tester"},
    )
    assert rebuild["paper_id"] == "p1"
    assert backfill["library_id"] == "default"


@pytest.mark.asyncio
async def test_paper_graph_route(monkeypatch) -> None:
    _setup(monkeypatch)
    payload = await graph_api.paper_graph(
        project_id="proj_demo",
        library_id="default",
        limit=120,
        user={"user_id": "tester"},
    )
    assert payload["meta"]["project_id"] == "proj_demo"
    assert payload["meta"]["library_id"] == "default"
    assert payload["meta"]["scoped_paper_count"] == 1
