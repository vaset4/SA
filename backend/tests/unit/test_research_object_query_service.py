import asyncio

from app.services.graph.research_object_query_service import ResearchObjectQueryService


class FakeClient:
    def __init__(self, rows):
        self.enabled = True
        self._rows = rows

    async def execute_read(self, query: str, parameters: dict):
        if "RETURN p" in query and "type(r)" not in query:
            paper = self._rows[0]["p"] if self._rows else None
            return [{"p": paper}] if paper else []
        return self._rows


def test_get_paper_subgraph_supports_dict_rows() -> None:
    client = FakeClient(
        [
            {
                "p": {
                    "paper_id": "upload_1",
                    "title": "Chain of Thought",
                    "library_id": "default",
                    "_label": "Paper",
                },
                "r": {
                    "type": "USES_METHOD",
                },
                "n": {
                    "entity_key": "method:cot",
                    "name": "Chain-of-thought prompting",
                    "_label": "Method",
                },
            }
        ]
    )
    service = ResearchObjectQueryService(client=client)

    payload = asyncio.run(service.get_paper_subgraph("upload_1"))

    assert payload["paper"]["paper_id"] == "upload_1"
    assert payload["nodes"] == [
        {
            "id": "method:cot",
            "label": "Method",
            "name": "Chain-of-thought prompting",
        }
    ]
    assert payload["edges"] == [
        {
            "source": "paper:upload_1",
            "target": "method:cot",
            "type": "USES_METHOD",
        }
    ]
    assert payload["meta"]["library_id"] == "default"
    assert payload["meta"]["node_count"] == 1
    assert payload["meta"]["edge_count"] == 1


def test_get_paper_subgraph_returns_empty_visible_graph_when_only_hidden_relations_exist() -> None:
    client = FakeClient(
        [
            {
                "p": {
                    "paper_id": "upload_1",
                    "title": "Chain of Thought",
                    "library_id": "default",
                    "_label": "Paper",
                },
                "r": {
                    "type": "AUTHORED_BY",
                },
                "n": {
                    "entity_key": "author:wei",
                    "name": "Jason Wei",
                    "_label": "Author",
                },
            }
        ]
    )
    service = ResearchObjectQueryService(client=client)

    async def fake_execute_read(query: str, parameters: dict):
        if "RETURN p" in query and "type(r)" not in query:
            return [{"p": client._rows[0]["p"]}]
        if "type(r) IN" in query:
            return []
        return client._rows

    client.execute_read = fake_execute_read  # type: ignore[method-assign]

    payload = asyncio.run(service.get_paper_subgraph("upload_1"))

    assert payload["paper"]["title"] == "Chain of Thought"
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_query_project_subgraph_aggregates_multiple_papers() -> None:
    client = FakeClient(
        [
            {
                "p": {
                    "paper_id": "upload_1",
                    "title": "Chain of Thought",
                    "library_id": "default",
                    "_label": "Paper",
                },
                "r": {
                    "type": "USES_METHOD",
                },
                "n": {
                    "entity_key": "method:cot",
                    "name": "Chain-of-thought prompting",
                    "_label": "Method",
                },
            }
        ]
    )
    service = ResearchObjectQueryService(client=client)

    async def fake_get_paper_subgraph(paper_id: str):
        if paper_id == "upload_1":
            return {
                "paper": {"paper_id": "upload_1", "title": "Chain of Thought"},
                "nodes": [{"id": "method:cot", "label": "Method", "name": "Chain-of-thought prompting"}],
                "edges": [{"source": "paper:upload_1", "target": "method:cot", "type": "USES_METHOD"}],
                "meta": {"library_id": "default", "node_count": 1, "edge_count": 1},
            }
        return {
            "paper": {"paper_id": "upload_2", "title": "Tree of Thoughts"},
            "nodes": [{"id": "method:tot", "label": "Method", "name": "Tree-of-thought prompting"}],
            "edges": [{"source": "paper:upload_2", "target": "method:tot", "type": "USES_METHOD"}],
            "meta": {"library_id": "default", "node_count": 1, "edge_count": 1},
        }

    service.get_paper_subgraph = fake_get_paper_subgraph  # type: ignore[method-assign]

    payload = asyncio.run(service.query_project_subgraph(["upload_1", "upload_2"]))

    assert payload["meta"]["paper_count"] == 2
    assert payload["meta"]["node_count"] == 2
    assert payload["meta"]["edge_count"] == 2
    assert payload["meta"]["library_ids"] == ["default"]


def _driver_data_shape_rows(relation_type: object = "STUDIES_TASK") -> list[dict]:
    """
    构造 neo4j 驱动 `record.data()` 摊平后的真实行形状。

    Args:
        relation_type: 三元关系列表的中段取值，默认真实关系类型字符串。

    Returns:
        单行列表：`p` 为论文属性字典，`r` 为 `[起点属性, 关系类型, 终点属性]` 三元列表，
        `n` 为实体属性字典（无 `_label`/`labels`，标签只藏在 `entity_key` 中段）。
    """
    paper = {
        "paper_id": "upload_abdedbbc",
        "title": "Spatial-Temporal Imputation",
        "library_id": "default",
    }
    entity = {
        "aliases": ["ST imputation"],
        "entity_key": "default:Task:spatial-temporal imputation",
        "library_id": "default",
        "name": "spatial-temporal imputation",
        "normalized_name": "spatial-temporal imputation",
    }
    return [{"p": paper, "r": [paper, relation_type, entity], "n": entity}]


def test_get_paper_subgraph_reads_real_driver_data_shape() -> None:
    """
    真实形状下关系类型与节点标签都要取到真值。

    既有的 dict 形状测试喂的是生产里从不出现的假数据（`r` 带 `type`、`n` 带 `_label`），
    正是它把「关系类型恒为空串、节点标签恒为 Unknown」这个缺陷藏住了。
    """
    service = ResearchObjectQueryService(client=FakeClient(_driver_data_shape_rows()))

    payload = asyncio.run(service.get_paper_subgraph("upload_abdedbbc"))

    assert payload["edges"] == [
        {
            "source": "paper:upload_abdedbbc",
            "target": "default:Task:spatial-temporal imputation",
            "type": "STUDIES_TASK",
        }
    ]
    assert payload["nodes"] == [
        {
            "id": "default:Task:spatial-temporal imputation",
            "label": "Task",
            "name": "spatial-temporal imputation",
        }
    ]
    assert payload["meta"]["node_count"] == 1
    assert payload["meta"]["edge_count"] == 1


def test_query_project_subgraph_keeps_edges_in_real_driver_data_shape() -> None:
    """聚合层不许再把真实形状的边当残缺丢掉（事故直接症状：有边却返回 0 条）。"""
    service = ResearchObjectQueryService(client=FakeClient(_driver_data_shape_rows()))

    payload = asyncio.run(service.query_project_subgraph(["upload_abdedbbc"]))

    assert payload["edges"] == [
        {
            "source": "paper:upload_abdedbbc",
            "target": "default:Task:spatial-temporal imputation",
            "type": "STUDIES_TASK",
        }
    ]
    assert payload["meta"]["edge_count"] == 1
    assert payload["meta"]["node_count"] == 1


def test_relation_with_non_string_middle_segment_falls_back_and_edge_is_dropped() -> None:
    """三元列表的中段不是字符串时回落空串，聚合层照旧丢弃该边——严格性不许放松。"""
    service = ResearchObjectQueryService(
        client=FakeClient(_driver_data_shape_rows(relation_type=None))
    )

    payload = asyncio.run(service.get_paper_subgraph("upload_abdedbbc"))
    assert payload["edges"][0]["type"] == ""

    aggregated = asyncio.run(service.query_project_subgraph(["upload_abdedbbc"]))
    assert aggregated["edges"] == []
    assert aggregated["meta"]["edge_count"] == 0
