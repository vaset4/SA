from __future__ import annotations

import pytest

from app.services.milvus_store import MilvusStore
from app.services.vector_store import VectorStore


def test_vector_store_read_chunk_skips_when_parent_already_read() -> None:
    store = VectorStore()
    store._loaded = True
    store._chunks = {
        "chunk-1": {
            "parent_id": "parent-1",
            "full_text": "full text 1",
            "paper_id": "paper-1",
            "section": "Method",
            "page_num": 3,
        },
        "chunk-2": {
            "parent_id": "parent-1",
            "full_text": "full text 1",
            "paper_id": "paper-1",
            "section": "Method",
            "page_num": 3,
        },
    }

    already_read: set[str] = set()
    first = store.read_chunk("chunk-1", already_read)
    second = store.read_chunk("chunk-2", already_read)

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert "parent:parent-1" in already_read


def test_milvus_store_read_chunk_skips_when_parent_already_read(monkeypatch) -> None:
    """同一父块的第二个子块再读时返回占位符，不重复消耗 token。

    MilvusStore 的读取要绑用户（正文按 owner 授权取），且它不改传入的集合，
    而是把本次读到的键放进返回值的 read_keys 由调用方并进去
    （knowledge_inspector_service 就是这个用法）。
    """
    rows = {
        "chunk-1": {
            "chunk_id": "chunk-1",
            "paper_id": "paper-1",
            "parent_id": "parent-1",
            "full_text": "full text 1",
            "section": "Method",
            "page_num": 3,
        },
        "chunk-2": {
            "chunk_id": "chunk-2",
            "paper_id": "paper-1",
            "parent_id": "parent-1",
            "full_text": "full text 1",
            "section": "Method",
            "page_num": 3,
        },
    }

    def _fake_read_published_chunk(chunk_id: str, *, user_id: str, publication=None) -> dict:
        """参数为引用、账号和快照；验证独立历史读取没有伪造请求快照。"""
        assert user_id == "u1"
        assert publication is None
        return dict(rows[chunk_id])

    monkeypatch.setattr(
        "app.services.upload_read_service.read_published_chunk",
        _fake_read_published_chunk,
    )
    store = MilvusStore().for_user("u1")

    already_read: set[str] = set()
    first = store.read_chunk("chunk-1", already_read)
    already_read.update(first["read_keys"])
    second = store.read_chunk("chunk-2", already_read)

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert "parent:parent-1" in already_read


def test_milvus_store_read_chunk_requires_identity() -> None:
    """没绑用户时显式报错，不能默默按空身份去读别人的正文。"""
    store = MilvusStore()

    with pytest.raises(LookupError):
        store.read_chunk("chunk-1", set())
