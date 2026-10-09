import asyncio

from app.services.graph.paper_graph_service import PaperGraphService


class FakeClient:
    def __init__(self, edge_rows, node_rows=None, *, relation_types=None, paper_keys=None):
        self.enabled = True
        self._edge_rows = edge_rows
        self._node_rows = node_rows or []
        self._relation_types = relation_types or ["USES_METHOD", "USES_DATASET"]
        self._paper_keys = paper_keys or ["paper_id", "title", "year", "published_date", "library_id"]
        self.calls = []

    async def execute_read(self, query: str, parameters: dict):
        self.calls.append((query, parameters))
        if "AS relation_types" in query:
            return [{"relation_types": self._relation_types}]
        if "AS paper_keys" in query:
            return [{"paper_keys": self._paper_keys}]
        if "shared_items" in query:
            return self._edge_rows
        return self._node_rows


def test_get_paper_graph_aggregates_shared_research_objects() -> None:
    client = FakeClient(
        [
            {
                "source": {
                    "paper_id": "p1",
                    "title": "Paper 1",
                    "year": 2022,
                    "published_date": "2022-05-01",
                    "library_id": "default",
                },
                "target": {
                    "paper_id": "p2",
                    "title": "Paper 2",
                    "year": 2023,
                    "published_date": "2023-08-09",
                    "library_id": "default",
                },
                "relation": "USES_METHOD",
                "shared_items": ["Transformer", "Self-Attention", "Transformer"],
                "weight": 2,
            }
        ]
    )
    service = PaperGraphService(client=client)

    payload = asyncio.run(
        service.get_paper_graph(
            project_id="proj_demo",
            resolved_scope={"paper_ids": ["p1", "p2"]},
            limit=100,
        )
    )

    assert payload["nodes"] == [
        {
            "id": "paper:p1",
            "paper_id": "p1",
            "title": "Paper 1",
            "year": 2022,
            "published_date": "2022-05-01",
            "library_id": "default",
        },
        {
            "id": "paper:p2",
            "paper_id": "p2",
            "title": "Paper 2",
            "year": 2023,
            "published_date": "2023-08-09",
            "library_id": "default",
        },
    ]
    assert payload["edges"] == [
        {
            "source": "paper:p1",
            "target": "paper:p2",
            "type": "USES_METHOD",
            "weight": 2,
            "shared_items": ["Transformer", "Self-Attention"],
        }
    ]
    assert payload["meta"] == {
        "project_id": "proj_demo",
        "library_id": "",
        "node_count": 2,
        "edge_count": 1,
        "scoped_paper_count": 2,
        "relation_counts": {"USES_METHOD": 1},
    }
    edge_call = next((call for call in client.calls if "shared_items" in call[0]), None)
    assert edge_call is not None
    assert edge_call[1]["paper_ids"] == ["p1", "p2"]
    assert edge_call[1]["paper_limit"] == 2


def test_get_paper_graph_returns_scoped_nodes_when_no_edges() -> None:
    client = FakeClient(
        [],
        node_rows=[
            {
                "p": {
                    "paper_id": "p9",
                    "title": "Lone Paper",
                    "year": 2024,
                    "published_date": "2024-01-12",
                    "library_id": "default",
                }
            },
        ],
    )
    service = PaperGraphService(client=client)

    payload = asyncio.run(
        service.get_paper_graph(
            library_id="default",
            resolved_scope={"paper_ids": ["p9"]},
            limit=20,
        )
    )

    assert payload["nodes"] == [
        {
            "id": "paper:p9",
            "paper_id": "p9",
            "title": "Lone Paper",
            "year": 2024,
            "published_date": "2024-01-12",
            "library_id": "default",
        }
    ]
    assert payload["edges"] == []
    assert payload["meta"]["scoped_paper_count"] == 1


def test_get_paper_graph_skips_missing_relations_and_falls_back_to_published_date_sort() -> None:
    client = FakeClient(
        [],
        node_rows=[
            {
                "p": {
                    "paper_id": "p9",
                    "title": "Lone Paper",
                    "published_date": "2024-01-12",
                    "library_id": "default",
                }
            },
        ],
        relation_types=["AUTHORED_BY", "REPORTS_METRIC"],
        paper_keys=["paper_id", "title", "published_date", "library_id"],
    )
    service = PaperGraphService(client=client)

    payload = asyncio.run(
        service.get_paper_graph(
            library_id="default",
            resolved_scope={"paper_ids": ["p9"]},
            limit=20,
        )
    )

    assert not any("shared_items" in query for query, _ in client.calls)
    node_query = next(query for query, _ in client.calls if "RETURN p" in query)
    assert "coalesce(p.published_date, '') DESC, p.paper_id" in node_query
    assert "coalesce(p.year, 0)" not in node_query
    assert payload["nodes"] == [
        {
            "id": "paper:p9",
            "paper_id": "p9",
            "title": "Lone Paper",
            "year": 2024,
            "published_date": "2024-01-12",
            "library_id": "default",
        }
    ]


def test_query_project_subgraph_reuses_project_scope_semantics() -> None:
    client = FakeClient(
        [
            {
                "source": {
                    "paper_id": "p1",
                    "title": "Paper 1",
                    "year": 2022,
                    "published_date": "2022-05-01",
                    "library_id": "default",
                },
                "target": {
                    "paper_id": "p2",
                    "title": "Paper 2",
                    "year": 2023,
                    "published_date": "2023-08-09",
                    "library_id": "default",
                },
                "relation": "USES_METHOD",
                "shared_items": ["Transformer"],
                "weight": 1,
            }
        ]
    )
    service = PaperGraphService(client=client)

    payload = asyncio.run(
        service.query_project_subgraph(
            project_id="proj_demo",
            paper_ids=["p1", "p2"],
            limit=50,
        )
    )

    assert payload["meta"]["project_id"] == "proj_demo"
    edge_call = next(call for call in client.calls if "shared_items" in call[0])
    assert edge_call[1]["paper_ids"] == ["p1", "p2"]
