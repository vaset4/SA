from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
import sys
from types import SimpleNamespace

import app.services.reranker as reranker_module


def test_reranker_raises_when_model_unavailable(monkeypatch) -> None:
    reranker = reranker_module.Reranker()
    monkeypatch.setattr(reranker_module.settings, "rerank_enabled", True)

    def fail_predict(query, documents):  # noqa: ANN001
        raise RuntimeError("local reranker cache missing")

    monkeypatch.setattr(reranker, "_predict", fail_predict)

    documents = [
        {"paper_id": "p1", "preview": "A"},
        {"paper_id": "p2", "preview": "B"},
        {"paper_id": "p3", "preview": "C"},
    ]

    with pytest.raises(RuntimeError, match="local reranker cache missing"):
        reranker.rerank("test", documents, top_k=2)


def test_local_rerank_uses_model_override_and_preserves_scores(monkeypatch):
    """显式 local 模式懒加载 CrossEncoder，优先取 preview 并保留原模型分数。"""
    captured = {}

    class LocalModel:
        """记录本地模型构造参数和候选文本。"""

        def __init__(self, source, **kwargs):
            """保存模型来源与加载参数，无返回值。"""
            captured["source"] = source
            captured["kwargs"] = kwargs

        def predict(self, pairs):
            """记录查询与文档对，返回不经转换的本地模型分值。"""
            captured["pairs"] = pairs
            return [-2.3, 4.1]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=LocalModel))
    monkeypatch.setattr(reranker_module, "configure_offline_env", lambda: captured.setdefault("offline", True))
    monkeypatch.setattr(reranker_module, "resolve_local_model_source", lambda name: "/local/" + name)
    config = SimpleNamespace(rerank_mode="local", rerank_enabled=True,
        vector_rerank_model="selected-model", rerank_model="legacy-model")
    reranker = reranker_module.Reranker(config=config)
    documents = [{"id": 1, "preview": "preview", "text": "text"}, {"id": 2, "text": "fallback"}]
    assert captured == {}
    result = reranker.rerank("query", documents, 1)
    assert result == [{"id": 2, "text": "fallback", "rerank_score": 4.1, "vector_rerank_score": 4.1}]
    assert captured == {"source": "/local/selected-model", "kwargs": {"local_files_only": True},
        "offline": True, "pairs": [("query", "preview"), ("query", "fallback")]}


def test_local_rerank_rejects_missing_scores(monkeypatch):
    """本地模型少返回分数也必须失败，不能由 zip 静默丢失候选材料。"""
    reranker = reranker_module.Reranker(config=SimpleNamespace(rerank_enabled=True))
    monkeypatch.setattr(reranker, "_predict", lambda query, docs: [0.3])
    with pytest.raises(RuntimeError, match="数量"):
        reranker.rerank("q", [{"text": "a"}, {"text": "b"}], 1)


def test_concurrent_local_rerank_initializes_one_model(monkeypatch):
    """两个检索线程首次重排共享同一份本地权重，不重复初始化模型。"""
    started = threading.Event()
    second_started = threading.Event()
    released = threading.Event()
    models = []

    class LocalModel:
        """阻塞首次模型加载，暴露同时进入构造器的竞争。"""

        def __init__(self, source, **kwargs):
            """接收模型参数，记录实际实例，等待测试释放权重加载。"""
            models.append(self)
            (started if len(models) == 1 else second_started).set()
            assert released.wait(timeout=2)

        def predict(self, pairs):
            """接收文本对，返回每条材料的固定测试分数。"""
            return [0.5 for _ in pairs]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=LocalModel))
    monkeypatch.setattr(reranker_module, "configure_offline_env", lambda: None)
    monkeypatch.setattr(reranker_module, "resolve_local_model_source", lambda name: name)
    reranker = reranker_module.Reranker(config=SimpleNamespace(
        rerank_mode="local", rerank_enabled=True, rerank_model="test-model"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(reranker.rerank, "q", [{"text": "a"}], 1)
        try:
            assert started.wait(timeout=1)
            second = executor.submit(reranker.rerank, "q", [{"text": "b"}], 1)
            duplicated = second_started.wait(timeout=0.2)
        finally:
            released.set()
        assert first.result(timeout=1)[0]["rerank_score"] == 0.5
        assert second.result(timeout=1)[0]["rerank_score"] == 0.5
    assert not duplicated
    assert len(models) == 1
