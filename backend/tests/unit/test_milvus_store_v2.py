"""milvus_store v2 schema 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.milvus_uri = "http://localhost:19530"
    s.milvus_collection = "paper_chunks"
    s.milvus_collection_v2 = "study_assistant_chunks_v2"
    s.embedding_dimension = 768
    return s


def test_direct_paper_write_cannot_delete_published_result():
    """未持有独立批次和发布事务的旧入口明确拒绝，不能先删除已有索引。"""
    from app.services.milvus_store import MilvusStore
    store = MilvusStore(collection_name="unit_collection")
    with patch.object(store, "delete_paper") as delete, patch.object(store, "_get_client") as client:
        with pytest.raises(RuntimeError, match="独立批次"):
            store.add_paper("paper", [])
        delete.assert_not_called()
        client.assert_not_called()


def test_private_batch_upserts_and_reads_exact_batch_without_deleting_paper():
    """写入失败可重放同一私有载荷，核验读取不得混入其他版本。"""
    from app.services.milvus_store import MilvusStore
    store = MilvusStore(collection_name="unit_collection")
    rows = [{"chunk_id": "c", "batch_id": "b", "paper_id": "paper"}]
    client = MagicMock()
    client.upsert.return_value = {"upsert_count": 1}
    client.query_iterator.return_value.next.side_effect = [rows, []]
    with patch.object(store, "_get_client", return_value=client), patch.object(store, "ensure_collection"), patch.object(store, "_current_batch", return_value="published-batch"):
        assert store.write_upload_batch("b", rows) == rows
    client.delete.assert_not_called()
    assert client.query_iterator.call_args.kwargs["filter"] == 'batch_id == "b"'
    client.query_iterator.return_value.close.assert_called_once()


def test_private_batch_partial_ack_is_not_success():
    """供应商只确认部分写入时，不能登记核验成功。"""
    from app.services.milvus_store import MilvusStore
    store = MilvusStore(collection_name="unit_collection")
    client = MagicMock()
    client.upsert.return_value = {"upsert_count": 0}
    with patch.object(store, "_get_client", return_value=client), patch.object(store, "ensure_collection"), patch.object(store, "_current_batch", return_value="published-batch"):
        with pytest.raises(RuntimeError, match="完整批次"):
            store.write_upload_batch("b", [{"batch_id": "b"}])
    client.query_iterator.assert_not_called()


def test_format_hit_uses_display_text():
    from app.services.milvus_store import MilvusStore

    store = MilvusStore(collection_name="unit_collection")
    hit = {
        "chunk_id": "c1",
        "distance": 0.9,
        "entity": {
            "chunk_id": "c1",
            "parent_id": "p1",
            "display_text": "Display content here",
            "embedding_text": "embedding only",
            "section": "Methods",
            "page_num": 3,
            "paper_id": "paper_1",
            "block_type": "table",
            "section_path": "3 Methods",
            "parser_mode": "mineru_online",
        },
    }
    result = store._format_hit(hit)
    assert "Display content" in result["preview"]
    assert result["block_type"] == "table"
    assert result["section_path"] == "3 Methods"
    assert result["display_text"] == "Display content here"


def test_count_paper_chunks_queries_by_paper_id():
    from app.services.milvus_store import _MILVUS_IO_TIMEOUT_SECONDS, MilvusStore

    store = MilvusStore(collection_name="unit_collection")
    iterator = MagicMock()
    iterator.next.side_effect = [[{"chunk_id": "a"}], [{"chunk_id": "b"}], []]

    with patch.object(store, "_get_client") as mock_client, \
         patch.object(store, "ensure_collection"), patch.object(store, "_current_batch", return_value="published-batch"):
        mock_client.return_value.query_iterator.return_value = iterator
        count = store.count_paper_chunks('paper"1')

    assert count == 2
    mock_client.return_value.query_iterator.assert_called_once_with(
        collection_name="unit_collection",
        filter='batch_id == "published-batch"',
        output_fields=["chunk_id"],
        batch_size=1000,
        timeout=_MILVUS_IO_TIMEOUT_SECONDS,
    )
    iterator.close.assert_called_once_with()


def test_get_chunk_row_reads_all_inspector_fields():
    from app.services.milvus_store import MilvusStore

    store = MilvusStore(collection_name="unit_collection")

    with patch.object(store, "_get_client") as mock_client, \
         patch.object(store, "ensure_collection"), patch.object(store, "_current_batch", return_value="published-batch"):
        mock_client.return_value.query.return_value = [{"chunk_id": "c1", "paper_id": "p1"}]
        row = store.get_chunk_row("c1")

    assert row == {"chunk_id": "c1", "paper_id": "p1"}
    call = mock_client.return_value.query.call_args
    assert call.kwargs["ids"] == ["c1"]
    assert "full_text" in call.kwargs["output_fields"]
    assert "parser_mode" in call.kwargs["output_fields"]


def test_list_chunk_rows_by_paper_uses_iterator_and_closes():
    from app.services.milvus_store import MilvusStore

    store = MilvusStore(collection_name="unit_collection")
    iterator = MagicMock()
    iterator.next.side_effect = [[{"chunk_id": "a"}], [{"chunk_id": "b"}], []]

    with patch.object(store, "_get_client") as mock_client, \
         patch.object(store, "ensure_collection"), patch.object(store, "_current_batch", return_value="published-batch"):
        mock_client.return_value.query_iterator.return_value = iterator
        rows = store.list_chunk_rows_by_paper('paper"1')

    assert rows == [{"chunk_id": "a"}, {"chunk_id": "b"}]
    mock_client.return_value.query_iterator.assert_called_once()
    assert mock_client.return_value.query_iterator.call_args.kwargs["filter"] == 'batch_id == "published-batch"'
    iterator.close.assert_called_once_with()


def _forbid_current_batch(monkeypatch) -> None:
    """把现存批次查询整条路封死；盖章后若仍走这条路会直接抛错而非静默换版本。"""
    from app.repositories import upload_publication_repository
    from app.services import paper_db

    def forbidden(*args, **kwargs):
        raise AssertionError("【检索】【固定批次】盖章后不应再查当前发布批次")

    monkeypatch.setattr(upload_publication_repository, "get_upload_publication_repository", forbidden)
    monkeypatch.setattr(paper_db, "_run", forbidden)


def test_snapshot_bound_read_uses_snapshot_batch(monkeypatch):
    """【检索】【固定批次】盖章后直读用快照里的批次，同一轮不混用新旧索引。"""
    from app.services.milvus_store import MilvusStore

    _forbid_current_batch(monkeypatch)
    store = MilvusStore(collection_name="unit_collection").for_snapshot({"paper_batches": {"paper-1": "batch-snap"}})
    iterator = MagicMock()
    iterator.next.side_effect = [[{"section_path": "1 Intro"}], []]

    with patch.object(store, "_get_client") as mock_client, patch.object(store, "ensure_collection"):
        mock_client.return_value.query_iterator.return_value = iterator
        assert store.list_section_paths("paper-1") == ["1 Intro"]

    assert mock_client.return_value.query_iterator.call_args.kwargs["filter"] == 'batch_id == "batch-snap"'


def test_snapshot_without_paper_returns_empty_without_fallback(monkeypatch):
    """【检索】【固定批次】论文不在快照里即本轮未授权，返回空且不回退查当前批次。"""
    from app.services.milvus_store import MilvusStore

    _forbid_current_batch(monkeypatch)
    store = MilvusStore(collection_name="unit_collection").for_snapshot(
        {"paper_batches": {"other-paper": "batch-other"}},
    )

    with patch.object(store, "_get_client") as mock_client, patch.object(store, "ensure_collection"):
        assert store.list_section_paths("paper-1") == []
        assert store.read_section_chunks("paper-1", ["1 Intro"]) == []
        assert store.list_chunk_rows_by_metadata("paper-1") == []
        mock_client.assert_not_called()


def test_unbound_store_still_reads_current_batch(monkeypatch):
    """【检索】【固定批次】未盖章的调用方（如知识检查器）仍读当前已发布批次。"""
    import asyncio
    from types import SimpleNamespace

    from app.repositories import upload_publication_repository
    from app.services import paper_db
    from app.services.milvus_store import MilvusStore

    calls: list[tuple[str, str | None]] = []

    async def fake_current_batch(paper_id, *, user_id=None):
        calls.append((paper_id, user_id))
        return "batch-live"

    monkeypatch.setattr(upload_publication_repository, "get_upload_publication_repository",
        lambda: SimpleNamespace(current_batch=fake_current_batch))
    monkeypatch.setattr(paper_db, "_run", asyncio.run)

    assert MilvusStore(collection_name="unit_collection").for_user("user-1")._current_batch("paper-1") == "batch-live"
    assert MilvusStore(collection_name="unit_collection").for_snapshot(None)._current_batch("paper-1") == "batch-live"
    assert MilvusStore(collection_name="unit_collection").for_snapshot({})._current_batch("paper-1") == "batch-live"
    assert calls == [("paper-1", "user-1"), ("paper-1", None), ("paper-1", None)]


def test_snapshot_binding_keeps_user_identity_and_shared_singleton():
    """【检索】【固定批次】盖章不覆盖已绑定身份，也不污染共享实例。"""
    from app.services.milvus_store import MilvusStore

    snapshot = {"paper_batches": {"paper-1": "batch-snap"}}
    base = MilvusStore(collection_name="unit_collection")
    bound = base.for_user("user-1").for_snapshot(snapshot)
    reversed_binding = base.for_snapshot(snapshot).for_user("user-2")

    assert (bound._reader_user_id, bound._reader_snapshot) == ("user-1", snapshot)
    assert (reversed_binding._reader_user_id, reversed_binding._reader_snapshot) == ("user-2", snapshot)
    assert (base._reader_user_id, base._reader_snapshot) == (None, None)


def test_flush_collection_flushes_current_collection():
    from app.services.milvus_store import _MILVUS_IO_TIMEOUT_SECONDS, MilvusStore

    store = MilvusStore(collection_name="unit_collection")

    with patch.object(store, "_get_client") as mock_client, \
         patch.object(store, "ensure_collection"), patch.object(store, "_current_batch", return_value="published-batch"):
        store.flush_collection()

    mock_client.return_value.flush.assert_called_once_with(
        collection_name="unit_collection",
        timeout=_MILVUS_IO_TIMEOUT_SECONDS,
    )
