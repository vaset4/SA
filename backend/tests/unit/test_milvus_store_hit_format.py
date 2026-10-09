from __future__ import annotations

from app.services.milvus_store import MilvusStore


class _FakeHitWithId(dict):
    @property
    def id(self) -> str:
        return "chunk-from-id-property"


def test_format_hit_reads_chunk_id_from_top_level_field() -> None:
    store = MilvusStore()

    hit = {
        "chunk_id": "chunk-top-level",
        "distance": 0.91,
        "entity": {
            "parent_id": "parent-1",
            "text": "method preview",
            "paper_id": "paper-1",
            "section": "Method",
            "page_num": 4,
        },
    }

    formatted = store._format_hit(hit)

    assert formatted["chunk_id"] == "chunk-top-level"
    assert formatted["parent_id"] == "parent-1"


def test_format_hit_reads_chunk_id_from_hit_id_property() -> None:
    store = MilvusStore()

    hit = _FakeHitWithId(
        {
            "distance": 0.88,
            "entity": {
                "parent_id": "parent-2",
                "text": "experiment preview",
                "paper_id": "paper-2",
                "section": "Experiment",
                "page_num": 6,
            },
        }
    )

    formatted = store._format_hit(hit)

    assert formatted["chunk_id"] == "chunk-from-id-property"
    assert formatted["parent_id"] == "parent-2"
