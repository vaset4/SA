"""模型 API 的请求、材料映射和失败合同测试，作者：xxx。"""

from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
import sys
from types import SimpleNamespace

import pytest
import requests

from app.services.embedding_client import EmbeddingClient
from app.services.reranker import Reranker
from app.services.provider_request_queue import ProviderRequestCancelled, provider_request_context


def model_config(**changes):
    """接收测试覆盖字段，返回不含实际服务凭证的两种模型配置。"""
    return SimpleNamespace(**({
        "embedding_mode": "api", "embedding_model": "BAAI/bge-m3",
        "embedding_batch_size": 2, "embedding_dimension": 2, "embedding_device": "cpu",
        "embedding_api_base_url": "https://models.example/v1", "embedding_api_key": "test-token",
        "embedding_api_timeout_seconds": 2, "rerank_mode": "api", "rerank_enabled": True,
        "rerank_model": "BAAI/bge-reranker-v2-m3", "rerank_endpoint": "https://models.example/v1/rerank",
        "rerank_api_key": "test-token", "rerank_api_timeout_seconds": 2,
    } | changes))


class Response:
    """提供可关闭的固定 HTTP 响应，用于协议异常测试。"""

    def __init__(self, payload, status=200):
        """保存测试 JSON 载荷与状态码；初始响应未关闭。"""
        self.payload = payload
        self.status_code = status
        self.closed = False

    def json(self):
        """返回测试载荷，或抛出预设 JSON 解码异常。"""
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def close(self):
        """标记响应已关闭，无返回值。"""
        self.closed = True


class Transport:
    """记录请求并返回固定响应，确保每次失败不会自动重试。"""

    def __init__(self, response):
        """保存响应或异常，创建请求记录列表。"""
        self.response = response
        self.calls = []

    def __call__(self, endpoint, **kwargs):
        """接收地址与请求参数，记录调用后返回响应或抛出异常。"""
        self.calls.append((endpoint, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ModelHandler(BaseHTTPRequestHandler):
    """在测试自有回环端口接收真实模型 HTTP 请求。"""

    def do_POST(self):
        """读取模型请求并按测试队列返回 JSON，保存请求路径与认证头。"""
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append((self.path, self.headers.get("Authorization"), payload))
        body = json.dumps(self.server.replies.pop(0)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """关闭测试服务器默认终端访问日志，不处理产品日志。"""


@pytest.fixture
def model_server():
    """创建自有回环 HTTP 服务并返回实例，测试结束停止线程和监听。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    server.calls = []
    server.replies = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_actual_http_batches_and_rerank_preserve_material_identity(model_server):
    """真实 HTTP 验证批次请求、乱序向量、完整重排分数与原候选材料对应。"""
    model_server.replies = [
        {"data": [{"index": 1, "embedding": [0, 7]}, {"index": 0, "embedding": [3, 4]}]},
        {"data": [{"index": 0, "embedding": [-5, 0]}]},
        {"results": [{"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4}, {"index": 1, "relevance_score": 0.2}]},
    ]
    base = f"http://127.0.0.1:{model_server.server_port}/v1"
    config = model_config(embedding_api_base_url=base, rerank_endpoint=base + "/rerank",
        vector_rerank_model="custom-rerank-model")
    with requests.Session() as session:
        session.trust_env = False
        vectors = EmbeddingClient(config=config, post=session.post).embed_texts(["甲", "乙", "丙"])
        documents = [{"id": "a", "preview": "短文", "text": "长文"}, {"id": "b", "text": "乙"},
            {"id": "c", "preview": "丙"}]
        unchanged = deepcopy(documents)
        ranked = Reranker(config=config, post=session.post).rerank("查询", documents, top_k=2)
    assert vectors == [[0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]]
    assert documents == unchanged
    assert [(item["id"], item["rerank_score"], item["vector_rerank_score"]) for item in ranked] == [
        ("c", 0.9, 0.9), ("a", 0.4, 0.4)]
    assert model_server.calls == [
        ("/v1/embeddings", "Bearer test-token", {"model": "BAAI/bge-m3", "input": ["甲", "乙"],
            "encoding_format": "float"}),
        ("/v1/embeddings", "Bearer test-token", {"model": "BAAI/bge-m3", "input": ["丙"],
            "encoding_format": "float"}),
        ("/v1/rerank", "Bearer test-token", {"model": "custom-rerank-model", "query": "查询",
            "documents": ["短文", "乙", "丙"], "top_n": 3, "return_documents": False}),
    ]


@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 2], [-1, 0], [False, 1], [0.0, 1], [None, 1]])
@pytest.mark.parametrize("operation", ["embedding", "rerank"])
def test_invalid_result_indices_fail_without_partial_results(indices, operation):
    """索引缺失、重复、越界或类型错误均不能被当作部分成功。"""
    entries = [{"index": index, "embedding": [1, 0], "relevance_score": 0.5} for index in indices]
    transport = Transport(Response({"data": entries, "results": entries}))
    with pytest.raises(RuntimeError, match="数量不匹配|索引"):
        if operation == "embedding":
            EmbeddingClient(config=model_config(), post=transport).embed_texts(["a", "b"])
        else:
            Reranker(config=model_config(), post=transport).rerank("q", [{"text": "a"}, {"text": "b"}], 1)
    assert len(transport.calls) == 1
    assert transport.response.closed


@pytest.mark.parametrize("vector", [[1], [0, 0], [True, 1], ["1", 2], [math.nan, 1],
    [math.inf, 1], [-math.inf, 1], [10 ** 400, 1], None])
def test_invalid_vectors_are_rejected(vector):
    """维度、零向量和非有限或非数值元素失败，不补零也不返回错误维度。"""
    transport = Transport(Response({"data": [{"index": 0, "embedding": vector}]}))
    with pytest.raises(RuntimeError, match="Embedding"):
        EmbeddingClient(config=model_config(), post=transport).embed_texts(["a"])


def test_large_finite_vectors_normalize_without_overflow():
    """有限但平方溢出的数值仍可正确归一化。"""
    transport = Transport(Response({"data": [{"index": 0, "embedding": [1e308, 1e308]}]}))
    vector = EmbeddingClient(config=model_config(), post=transport).embed_texts(["a"])[0]
    assert math.hypot(*vector) == pytest.approx(1)


@pytest.mark.parametrize("score", [None, True, "0.5", math.nan, math.inf, -math.inf, 10 ** 400])
def test_invalid_rerank_scores_fail(score):
    """非法分数不能参与候选排序，也不被替换成默认分数。"""
    transport = Transport(Response({"results": [{"index": 0, "relevance_score": score}]}))
    with pytest.raises(RuntimeError, match="Rerank API 返回"):
        Reranker(config=model_config(), post=transport).rerank("q", [{"text": "a"}], 1)


@pytest.mark.parametrize("operation", ["embedding", "rerank"])
@pytest.mark.parametrize("response,expected", [
    (Response({}, 401), "HTTP 401"), (Response({}, 429), "HTTP 429"),
    (Response({}, 500), "HTTP 500"), (Response({}, 503), "HTTP 503"),
    (Response({}, 302), "HTTP 302"), (Response(ValueError("private-body")), "无效 JSON"),
    (Response([]), "JSON 对象"), (requests.Timeout("private-token private-url"), "请求超时"),
    (requests.ConnectionError("private-token private-url"), "ConnectionError"),
])
def test_request_failures_are_explicit_and_redacted(operation, response, expected):
    """认证、限流、服务错误、超时和坏 JSON 只报安全上下文且不重试。"""
    transport = Transport(response)
    with pytest.raises((RuntimeError, TimeoutError), match=expected) as caught:
        if operation == "embedding":
            EmbeddingClient(config=model_config(), post=transport).embed_texts(["a"])
        else:
            Reranker(config=model_config(), post=transport).rerank("q", [{"text": "a"}], 1)
    assert "private-" not in str(caught.value)
    assert len(transport.calls) == 1
    assert transport.calls[0][1]["timeout"] == 2
    assert transport.calls[0][1]["allow_redirects"] is False
    if isinstance(response, Response):
        assert response.closed


@pytest.mark.parametrize("operation", ["embedding", "rerank"])
def test_missing_credentials_fail_before_io_and_empty_input_needs_no_config(operation):
    """空输入无需服务配置；有材料时独立密钥缺失必须在 HTTP 前失败。"""
    config = model_config(embedding_api_key="", rerank_api_key="")
    transport = Transport(AssertionError("不得调用服务"))
    if operation == "embedding":
        client = EmbeddingClient(config=config, post=transport)
        assert client.embed_texts([]) == []
        with pytest.raises(ValueError, match="密钥"):
            client.embed_texts(["a"])
    else:
        client = Reranker(config=config, post=transport)
        assert client.rerank("q", [], 1) == []
        with pytest.raises(ValueError, match="密钥"):
            client.rerank("q", [{"text": "a"}], 1)
    assert not transport.calls


def test_disabled_rerank_keeps_original_order_without_api_config():
    """显式关闭重排时原始顺序和分数字段保持不变且无需密钥。"""
    transport = Transport(AssertionError("不得调用服务"))
    config = model_config(vector_rerank_enabled=False, rerank_api_key="")
    documents = [{"id": 1}, {"id": 2}]
    assert Reranker(config=config, post=transport).rerank("q", documents, 1) == [{"id": 1}]
    assert not transport.calls


@pytest.mark.parametrize("operation", ["embedding", "rerank"])
def test_local_mode_without_optional_dependency_fails_explicitly(monkeypatch, operation):
    """显式 local 缺少可选模型包时明确失败，不能转调 API。"""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    transport = Transport(AssertionError("不得调用 API"))
    config = model_config(embedding_mode="local", rerank_mode="local")
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        if operation == "embedding":
            EmbeddingClient(config=config, post=transport).embed_texts(["a"])
        else:
            Reranker(config=config, post=transport).rerank("q", [{"text": "a"}], 1)
    assert not transport.calls


@pytest.mark.parametrize("operation", ["embedding", "rerank"])
@pytest.mark.parametrize("entry", [None, [], "not-an-object", {}])
def test_result_structure_is_validated(operation, entry):
    """结果项必须是带完整索引的对象，非法结构不得进入排序或向量存储。"""
    transport = Transport(Response({"data": [entry], "results": [entry]}))
    with pytest.raises(RuntimeError, match="格式错误|索引"):
        if operation == "embedding":
            EmbeddingClient(config=model_config(), post=transport).embed_texts(["a"])
        else:
            Reranker(config=model_config(), post=transport).rerank("q", [{"text": "a"}], 1)


@pytest.mark.parametrize("cancelled", [True, False])
def test_model_request_rejects_cancelled_or_expired_execution_before_send(cancelled):
    """共享运行已取消或总期限届满时不再发送模型请求，也不占用任何 LLM 额度。"""
    cancellation = threading.Event()
    if cancelled:
        cancellation.set()
    deadline = None if cancelled else time.monotonic() - 1
    transport = Transport(AssertionError("不得发送已失效的请求"))
    with provider_request_context(cancel_event=cancellation, deadline=deadline):
        with pytest.raises(ProviderRequestCancelled if cancelled else TimeoutError):
            Reranker(config=model_config(), post=transport).rerank("q", [{"text": "a"}], 1)
    assert not transport.calls


def test_model_request_timeout_uses_remaining_execution_deadline():
    """实际 HTTP 超时不能超过所属运行的剩余期限，成功响应仍正常关闭。"""
    transport = Transport(Response({"results": [{"index": 0, "relevance_score": 0.5}]}))
    with provider_request_context(cancel_event=threading.Event(), deadline=time.monotonic() + 1):
        Reranker(config=model_config(), post=transport).rerank("q", [{"text": "a"}], 1)
    assert 0 < transport.calls[0][1]["timeout"] <= 1
    assert transport.response.closed
