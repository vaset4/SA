"""Milvus、FAISS 和 LightRAG 共用的 API / 本地向量客户端。"""

from __future__ import annotations

from collections.abc import Callable
import math
from threading import Lock
from typing import Any

from app.config import get_settings
from app.services.hf_local import configure_offline_env, resolve_local_model_source
from app.services.model_api_transport import ordered_model_results, request_model_api

settings = get_settings()
_CLIENT: "EmbeddingClient | None" = None

class EmbeddingClient:
    """统一封装远程或本地向量生成，调用方不承担模型协议差异。"""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        batch_size: int | None = None,
        dimension: int | None = None,
        device: str | None = None,
        mode: str | None = None,
        config: Any = None,
        post: Callable[..., Any] | None = None,
    ) -> None:
        """接收可选模型参数、配置和 HTTP 依赖，创建客户端；此时不加载模型或发请求。"""
        self._settings = settings if config is None else config
        self._mode = self._settings.embedding_mode if mode is None else mode
        self._model_name = (model_name if model_name is not None else self._settings.embedding_model).strip()
        self._batch_size = int(batch_size if batch_size is not None else self._settings.embedding_batch_size)
        self._dimension = int(dimension if dimension is not None else self._settings.embedding_dimension)
        self._device = (device if device is not None else self._settings.embedding_device).strip()
        if self._mode not in {"api", "local"}:
            raise ValueError("EMBEDDING_MODE 必须是 api 或 local")
        if not self._model_name or self._batch_size < 1 or self._dimension < 1:
            raise ValueError("Embedding 模型名不能为空，批大小和维度必须为正数")
        self._post = post
        self._model = None
        self._model_lock = Lock()

    @property
    def dimension(self) -> int:
        """返回当前模型的期望向量维度，供索引结构和响应核验使用。"""
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """接收文本列表，按配置生成同顺序浮点向量；空列表不访问任何模型。"""
        if not texts:
            return []
        if self._mode == "api":
            return self._embed_api(texts)
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        normalized = self._normalize_vectors(vectors)
        if len(normalized) != len(texts):
            raise RuntimeError(
                f"Embedding 返回数量不匹配: expected={len(texts)} actual={len(normalized)}"
            )
        for vector in normalized:
            self._validate_vector(vector)
        return normalized

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """按批发送文本，校验每条远程向量并归一化；返回按原文本顺序排列的向量。"""
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Embedding 输入必须是文本列表")
        options = {} if self._post is None else {"post": self._post}
        vectors = []
        for offset in range(0, len(texts), self._batch_size):
            batch = texts[offset:offset + self._batch_size]
            payload = request_model_api(
                self._settings.embedding_api_base_url.rstrip("/") + "/embeddings",
                self._settings.embedding_api_key,
                {"model": self._model_name, "input": batch, "encoding_format": "float"},
                timeout=self._settings.embedding_api_timeout_seconds, operation="Embedding", **options,
            )
            # 1. 【模型服务】【向量映射】服务可乱序返回，但每条输入必须有且仅有一条向量
            entries = ordered_model_results(payload.get("data"), count=len(batch), operation="Embedding")
            for entry in entries:
                vector = self._validate_vector(entry.get("embedding"))
                scale = max(abs(value) for value in vector)
                if scale == 0:
                    raise RuntimeError("Embedding API 返回零向量，无法归一化")
                scaled = [value / scale for value in vector]
                norm = math.hypot(*scaled)
                vectors.append([value / norm for value in scaled])
        return vectors

    def _validate_vector(self, payload: Any) -> list[float]:
        """接收一条模型向量，验证维度与有限数值后返回浮点列表。"""
        if not isinstance(payload, list) or len(payload) != self._dimension:
            raise RuntimeError(f"Embedding 维度不匹配: expected={self._dimension}")
        if any(type(value) not in {int, float} for value in payload):
            raise RuntimeError("Embedding 返回格式错误：向量必须包含有限数字")
        try:
            vector = [float(value) for value in payload]
        except OverflowError:
            raise RuntimeError("Embedding 返回格式错误：向量数值超出范围") from None
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("Embedding 返回格式错误：向量必须包含有限数字")
        return vector

    def _get_model(self):
        """
        延迟加载 sentence-transformers 模型。

        只有第一次真正需要 embedding 时才加载本地模型，这样启动期不会被大模型权重
        初始化拖慢。这里必须保证并发安全，避免 LightRAG / Milvus 在同一进程里
        竞态触发多次模型初始化。
        """
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            # 1. 【模型服务】【本地加载】只有显式 local 模式才设置 HF 环境并导入模型依赖
            configure_offline_env()
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("缺少 sentence-transformers 依赖，无法启用本地 embedding") from exc
            model_source = resolve_local_model_source(self._model_name)
            try:
                self._model = SentenceTransformer(model_source, local_files_only=True, device=self._device)
            except Exception as exc:
                raise RuntimeError(f"无法从本地缓存加载 embedding 模型: {self._model_name}") from exc
        return self._model

    def _normalize_vectors(self, payload: Any) -> list[list[float]]:
        """把底层模型返回统一收敛成 `list[list[float]]`。"""
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if not isinstance(payload, list):
            raise RuntimeError("Embedding 服务返回格式错误：期望二维浮点数组")
        if not payload:
            return []
        if all(isinstance(item, (int, float)) for item in payload):
            return [[float(item) for item in payload]]
        vectors: list[list[float]] = []
        for item in payload:
            if not isinstance(item, list) or not all(isinstance(value, (int, float)) for value in item):
                raise RuntimeError("Embedding 服务返回格式错误：期望二维浮点数组")
            vectors.append([float(value) for value in item])
        return vectors


def get_embedding_client() -> EmbeddingClient:
    """返回进程级 embedding 客户端单例。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = EmbeddingClient()
    return _CLIENT
