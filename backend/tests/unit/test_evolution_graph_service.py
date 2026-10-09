import asyncio

from app.services.graph.evolution_graph_service import EvolutionGraphService


class FakeNode(dict):
    def __init__(self, label: str, **props):
        super().__init__(props)
        self.labels = {label}


class FakeRelation:
    def __init__(self, relation_type: str) -> None:
        self.type = relation_type


class FakeClient:
    def __init__(self, rows):
        self.enabled = True
        self._rows = rows

    async def execute_read(self, query: str, parameters: dict):
        return self._rows.pop(0)


def test_paper_view_builds_timeline_and_stats() -> None:
    client = FakeClient(
        [
            [
                {
                    "source": FakeNode("Paper", paper_id="p1", title="Paper One", year=2022),
                    "rel": FakeRelation("CITES"),
                    "target": FakeNode("Paper", paper_id="p2", title="Paper Two", year=2021),
                }
            ],
            [
                {
                    "source": FakeNode("Paper", paper_id="p1", title="Paper One", year=2022),
                    "rel": FakeRelation("USES_METHOD"),
                    "target": FakeNode("Method", name="Transformer", entity_key="transformer"),
                },
                {
                    "source": FakeNode("Paper", paper_id="p2", title="Paper Two", year=2021),
                    "rel": FakeRelation("STUDIES_TASK"),
                    "target": FakeNode("Task", name="Question Answering", entity_key="qa"),
                },
            ],
        ]
    )
    service = EvolutionGraphService(client=client)

    payload = asyncio.run(
        service.get_evolution(
            {
                "view_mode": "paper",
                "paper_id": "p1",
                "method": "",
                "task": "",
                "dataset": "",
                "from_year": None,
                "to_year": None,
                "depth": 2,
                "project_id": "",
                "resolved_scope": None,
            }
        )
    )

    assert payload["focus"]["key"] == "p1"
    assert payload["stats"]["node_count"] == 4
    assert payload["stats"]["relations"]["CITES"] == 1
    assert payload["timeline"][0]["paper_id"] == "p2"
    assert payload["timeline"][1]["paper_id"] == "p1"


def test_method_view_respects_scope_and_year_filter() -> None:
    client = FakeClient(
        [
            [
                {
                    "source": FakeNode("Paper", paper_id="p1", title="Paper One", year=2020),
                    "rel": FakeRelation("USES_METHOD"),
                    "target": FakeNode("Method", name="GraphSAGE", entity_key="graphsage"),
                },
                {
                    "source": FakeNode("Paper", paper_id="p2", title="Paper Two", year=2024),
                    "rel": FakeRelation("USES_METHOD"),
                    "target": FakeNode("Method", name="GraphSAGE", entity_key="graphsage"),
                },
            ],
            [
                {
                    "source": FakeNode("Paper", paper_id="p2", title="Paper Two", year=2024),
                    "rel": FakeRelation("USES_DATASET"),
                    "target": FakeNode("Dataset", name="PubMed", entity_key="pubmed"),
                }
            ],
        ]
    )
    service = EvolutionGraphService(client=client)

    payload = asyncio.run(
        service.get_evolution(
            {
                "view_mode": "method",
                "paper_id": "",
                "method": "GraphSAGE",
                "task": "",
                "dataset": "",
                "from_year": 2021,
                "to_year": 2025,
                "depth": 1,
                "project_id": "proj_demo",
                "resolved_scope": {
                    "project_id": "proj_demo",
                    "library_ids": ["default"],
                    "paper_ids": ["p2"],
                    "excluded_paper_ids": [],
                    "year_from": None,
                    "year_to": None,
                    "task_filters": [],
                    "method_filters": ["GraphSAGE"],
                },
            }
        )
    )

    paper_ids = {node["paper_id"] for node in payload["nodes"] if node["node_type"] == "paper"}
    assert paper_ids == {"p2"}
    assert payload["filters"]["resolved_scope"]["paper_ids"] == ["p2"]
    assert payload["stats"]["node_types"]["dataset"] == 1
