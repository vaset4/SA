"""模型切换时的索引身份、共享编码和集合维度回归，作者：xxx。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.config import get_embedding_identity
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.research_contracts import RetrievalScopeError
from app.services import lightrag_service, milvus_store, upload_read_service, vector_store
from app.services.ingest import upload_index_service
from app.services.ingest.upload_artifacts import fingerprint


@pytest.fixture
def model_settings(tmp_path):
    """返回仅含测试地址的临时模型配置；不读写用户现有索引。"""
    return SimpleNamespace(
        data_dir=str(tmp_path), embedding_mode="api", embedding_model="unit-model",
        embedding_dimension=3, embedding_api_base_url="https://models.example.test/v1",
        embedding_api_key="unit-only-key", milvus_collection_v2="unit-only-collection",
    )


@pytest.mark.parametrize("mode", ["api", "local"])
def test_faiss_storage_uses_current_identity(monkeypatch, model_settings, tmp_path, mode):
    """指定模式后返回对应目录；API 隔离，local 保留旧文件位置。"""
    model_settings.embedding_mode = mode
    monkeypatch.setattr(vector_store, "settings", model_settings)
    expected = tmp_path / "vector_store"
    if mode == "api":
        expected = expected / get_embedding_identity(model_settings)
    assert vector_store._store_dir() == expected
    assert expected.is_dir()


@pytest.mark.parametrize("mode", ["api", "local"])
def test_lightrag_identity_preserves_library_isolation(monkeypatch, model_settings, tmp_path, mode):
    """模式目录之下继续按资料库隔离，默认资料库保持其原有相对位置。"""
    model_settings.embedding_mode = mode
    monkeypatch.setattr(lightrag_service, "settings", model_settings)
    service = lightrag_service.LightRAGService()
    expected = tmp_path / "lightrag_index"
    if mode == "api":
        expected = expected / get_embedding_identity(model_settings)
    assert service._get_runtime(DEFAULT_LIBRARY_ID).working_dir == expected
    assert service._get_runtime("library-a").working_dir == expected / "library-a"
    assert service._get_runtime("library-b").working_dir == expected / "library-b"
    assert not expected.exists()


def test_faiss_api_store_uses_configured_dimension_and_shared_encoder(monkeypatch, model_settings):
    """API 模式在无本地模型加载器时建立指定维度索引并转换共享编码结果。"""
    fake_faiss = SimpleNamespace(IndexFlatIP=Mock(return_value=SimpleNamespace(ntotal=0)))
    encoder = SimpleNamespace(embed_texts=Mock(return_value=[[0.6, 0.8, 0.0]]))
    monkeypatch.setattr(vector_store, "settings", model_settings)
    monkeypatch.setattr(vector_store, "HAS_NUMPY", True)
    monkeypatch.setattr(vector_store, "HAS_FAISS", True)
    monkeypatch.setattr(vector_store, "faiss", fake_faiss, raising=False)
    monkeypatch.setattr(vector_store, "get_embedding_client", lambda: encoder)
    store = vector_store.VectorStore()
    store._ensure_loaded()
    fake_faiss.IndexFlatIP.assert_called_once_with(3)
    assert not hasattr(store, "_get_model")
    result = store._encode(["测试文本"])
    assert result.dtype == np.float32
    assert result.tolist() == pytest.approx(np.asarray([[0.6, 0.8, 0.0]], dtype=np.float32))
    encoder.embed_texts.assert_called_once_with(["测试文本"])
    assert store._encode([]).shape == (0, 3)
    encoder.embed_texts.assert_called_once()


def test_upload_identity_keeps_local_digest_and_api_separate(monkeypatch, model_settings):
    """上传发布保留原 local 指纹，切为 API 后拒绝混用相同模型名的历史向量。"""
    monkeypatch.setattr(upload_index_service, "get_settings", lambda: model_settings)
    api_identity = upload_index_service.embedding_identity()
    assert api_identity == get_embedding_identity(model_settings)
    model_settings.embedding_mode = "local"
    legacy_identity = fingerprint({"model": "unit-model", "dimension": 3, "normalized": True})
    assert upload_index_service.embedding_identity() == legacy_identity
    assert api_identity != legacy_identity


def test_local_publication_requires_reprocessing_after_api_switch(monkeypatch, model_settings):
    """同一已发布批次在 local 可读，切 API 后先拒绝旧向量，再要求显式重新处理。"""
    model_settings.embedding_mode = "local"
    monkeypatch.setattr(upload_index_service, "get_settings", lambda: model_settings)
    authorize = AsyncMock()
    monkeypatch.setattr(upload_read_service, "authorize_publication_read", authorize)
    monkeypatch.setattr(upload_read_service.paper_db, "_run", asyncio.run)
    snapshot = {"user_id": "owner", "batch_ids": ["old-batch"],
        "collection_names": ["unit-collection"],
        "embedding_fingerprints": [upload_index_service.embedding_identity()]}
    scope = {"user_id": "owner", "filter_expr": 'paper_id == "paper"', "publication": snapshot}
    assert "old-batch" in upload_read_service.require_publication_scope(scope, collection_name="unit-collection")
    authorize.assert_awaited_once_with(snapshot, user_id="owner")
    authorize.reset_mock()
    model_settings.embedding_mode = "api"
    with pytest.raises(RetrievalScopeError, match="重新处理"):
        upload_read_service.require_publication_scope(scope, collection_name="unit-collection")
    authorize.assert_not_called()


def test_milvus_embedding_cache_is_partitioned_by_current_model_identity(monkeypatch, model_settings):
    """同一文本切换模型模式时重新编码；同身份的重复文本只编码一次。"""
    saved = {}
    cache = SimpleNamespace(get_embedding=saved.get, set_embedding=saved.__setitem__)
    encoder = SimpleNamespace(embed_texts=Mock(side_effect=[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]))
    monkeypatch.setattr(milvus_store, "settings", model_settings)
    monkeypatch.setattr(milvus_store, "get_cache", lambda: cache)
    monkeypatch.setattr(milvus_store, "get_embedding_client", lambda: encoder)
    store = milvus_store.MilvusStore()
    assert store._encode(["same", "same"]) == [[1.0, 0.0, 0.0]] * 2
    assert store._encode(["same"]) == [[1.0, 0.0, 0.0]]
    api_key = get_embedding_identity(model_settings) + ":same"
    model_settings.embedding_mode = "local"
    assert store._encode(["same"]) == [[0.0, 1.0, 0.0]]
    assert set(saved) == {api_key, get_embedding_identity(model_settings) + ":same"}
    assert encoder.embed_texts.call_count == 2


@pytest.mark.parametrize("dimension", [2, 4, None])
def test_milvus_dimension_mismatch_does_not_mutate_collection(monkeypatch, model_settings, dimension):
    """已有集合维度失配时显式失败，不创建、删除、写入或标记 schema 已通过。"""
    client = Mock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {"fields": [
        {"name": "batch_id"}, {"name": "dense_vector", "params": {"dim": dimension}},
    ]}
    monkeypatch.setattr(milvus_store, "settings", model_settings)
    store = milvus_store.MilvusStore()
    monkeypatch.setattr(store, "_get_client", lambda: client)
    with pytest.raises(RuntimeError, match="向量维度.*重新处理"):
        store.ensure_collection()
    assert not store._schema_checked
    assert [call[0] for call in client.mock_calls] == ["has_collection", "describe_collection"]


def test_milvus_matching_dimension_accepts_existing_collection(monkeypatch, model_settings):
    """匹配的已有集合只做只读核对，保留批次字段迁移约束和原有数据。"""
    client = Mock()
    client.has_collection.return_value = True
    client.describe_collection.return_value = {"fields": [
        {"name": "batch_id"}, {"name": "dense_vector", "params": {"dim": "3"}},
    ]}
    monkeypatch.setattr(milvus_store, "settings", model_settings)
    store = milvus_store.MilvusStore()
    monkeypatch.setattr(store, "_get_client", lambda: client)
    store.ensure_collection()
    assert store._schema_checked
    assert [call[0] for call in client.mock_calls] == ["has_collection", "describe_collection"]
