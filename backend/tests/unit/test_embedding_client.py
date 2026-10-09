from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import sys
from types import SimpleNamespace

import pytest

import app.services.embedding_client as embedding_client_module


@pytest.fixture(autouse=True)
def local_model_dependency(monkeypatch):
    """注入本地模型包替身；返回空模型模块，测试不会下载权重。"""
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=None))


class _FakeVectors:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


class _FakeModel:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def encode(self, texts, **kwargs):  # noqa: ANN001
        self.calls.append((list(texts), kwargs))
        return _FakeVectors(self._payload)


def test_embedding_client_loads_local_model_and_encodes_batch(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_model = _FakeModel([[1, 2], [3, 4]])

    def fake_sentence_transformer(model_name, **kwargs):  # noqa: ANN001
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(sys.modules["sentence_transformers"], "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(embedding_client_module, "configure_offline_env", lambda: captured.setdefault("offline", True))
    monkeypatch.setattr(embedding_client_module, "resolve_local_model_source", lambda name: f"/resolved/{name}")

    client = embedding_client_module.EmbeddingClient(
        model_name="BAAI/bge-m3",
        batch_size=7,
        dimension=2,
        device="cpu",
        mode="local",
    )
    vectors = client.embed_texts(["alpha", "beta"])

    assert captured == {
        "offline": True,
        "model_name": "/resolved/BAAI/bge-m3",
        "kwargs": {
            "local_files_only": True,
            "device": "cpu",
        },
    }
    assert fake_model.calls == [(
        ["alpha", "beta"],
        {
            "batch_size": 7,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        },
    )]
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


def test_embedding_client_wraps_single_vector_response(monkeypatch) -> None:
    monkeypatch.setattr(
        sys.modules["sentence_transformers"],
        "SentenceTransformer",
        lambda *args, **kwargs: _FakeModel([1, 2, 3]),
    )
    monkeypatch.setattr(embedding_client_module, "configure_offline_env", lambda: None)
    monkeypatch.setattr(embedding_client_module, "resolve_local_model_source", lambda name: name)
    client = embedding_client_module.EmbeddingClient(
        model_name="BAAI/bge-m3",
        batch_size=7,
        dimension=3,
        device="cpu",
        mode="local",
    )

    assert client.embed_texts(["alpha"]) == [[1.0, 2.0, 3.0]]


def test_embedding_client_rejects_dimension_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        sys.modules["sentence_transformers"],
        "SentenceTransformer",
        lambda *args, **kwargs: _FakeModel([[1, 2, 3]]),
    )
    monkeypatch.setattr(embedding_client_module, "configure_offline_env", lambda: None)
    monkeypatch.setattr(embedding_client_module, "resolve_local_model_source", lambda name: name)

    client = embedding_client_module.EmbeddingClient(
        model_name="BAAI/bge-m3",
        batch_size=7,
        dimension=2,
        device="cpu",
        mode="local",
    )

    with pytest.raises(RuntimeError, match="Embedding 维度不匹配"):
        client.embed_texts(["alpha"])


def test_embedding_client_retries_after_initial_load_failure(monkeypatch) -> None:
    attempts = {"count": 0}
    fake_model = _FakeModel([[1, 2]])

    def flaky_sentence_transformer(*args, **kwargs):  # noqa: ANN002, ANN003
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("transient load failure")
        return fake_model

    monkeypatch.setattr(sys.modules["sentence_transformers"], "SentenceTransformer", flaky_sentence_transformer)
    monkeypatch.setattr(embedding_client_module, "configure_offline_env", lambda: None)
    monkeypatch.setattr(embedding_client_module, "resolve_local_model_source", lambda name: name)

    client = embedding_client_module.EmbeddingClient(
        model_name="BAAI/bge-m3",
        batch_size=7,
        dimension=2,
        device="cpu",
        mode="local",
    )

    with pytest.raises(RuntimeError, match="无法从本地缓存加载 embedding 模型"):
        client.embed_texts(["alpha"])

    assert client.embed_texts(["beta"]) == [[1.0, 2.0]]
    assert attempts["count"] == 2


def test_embedding_client_initializes_model_once_under_concurrent_calls(monkeypatch) -> None:
    factory_calls = {"count": 0}
    fake_model = _FakeModel([[1, 2]])
    start_barrier = threading.Barrier(3)

    def slow_sentence_transformer(*args, **kwargs):  # noqa: ANN002, ANN003
        factory_calls["count"] += 1
        time.sleep(0.1)
        return fake_model

    monkeypatch.setattr(sys.modules["sentence_transformers"], "SentenceTransformer", slow_sentence_transformer)
    monkeypatch.setattr(embedding_client_module, "configure_offline_env", lambda: None)
    monkeypatch.setattr(embedding_client_module, "resolve_local_model_source", lambda name: name)

    client = embedding_client_module.EmbeddingClient(
        model_name="BAAI/bge-m3",
        batch_size=7,
        dimension=2,
        device="cpu",
        mode="local",
    )

    def run_embed():
        start_barrier.wait()
        return client.embed_texts(["alpha"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_embed)
        future_b = executor.submit(run_embed)
        start_barrier.wait()
        assert future_a.result() == [[1.0, 2.0]]
        assert future_b.result() == [[1.0, 2.0]]

    assert factory_calls["count"] == 1
