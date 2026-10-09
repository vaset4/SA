"""
API / 本地 Cross-Encoder 重排序服务。

这个模块位于 RAG 检索链的“召回之后、最终截断之前”阶段，负责对已经拿到的
候选文档再做一次语义重排。它不是主检索器，不负责召回：
  - 主召回仍由 Milvus 提供 chunk 级结果
  - LightRAG 的纯图谱聚合结果通常不走这里的 chunk 重排
  - `rag_node.py` 在拿到合并候选后按需调用这里做增强排序

当前输入是 `query + document list`，输出仍是文档字典列表，只是附加
`rerank_score` 并按该分数重新排序。
"""

from __future__ import annotations

from collections.abc import Callable
import math
from threading import Lock
from typing import Any, List, Optional

from app.config import get_settings, get_vector_rerank_model, is_vector_rerank_enabled
from app.services.hf_local import configure_offline_env, resolve_local_model_source
from app.services.model_api_transport import ordered_model_results, request_model_api

settings = get_settings()
_RERANKER: Optional["Reranker"] = None


class Reranker:
    """Cross-Encoder 重排器单例包装。"""

    def __init__(self, *, config: Any = None, post: Callable[..., Any] | None = None) -> None:
        """接收可选配置与 HTTP 依赖，创建重排器；不发请求且延迟加载本地模型。"""
        self._settings = settings if config is None else config
        self._post = post
        self._model = None
        self._model_lock = Lock()
        self._model_disabled = False

    def rerank(self, query: str, documents: List[dict], top_k: int) -> List[dict]:
        """
        对候选文档做重排并截断到 `top_k`。

        query 为检索问题，documents 为候选材料，top_k 为输出数量；返回带分数的材料列表。

        当 `vector_rerank_enabled=false` 时，这里只做简单前 `top_k` 截断，
        等价于关闭应用层向量重排增强。
        """
        if not documents:
            return []
        limited = documents[:top_k]
        if not is_vector_rerank_enabled(self._settings):
            # 1. 【模型服务】【重排开关】关闭时保留原召回顺序，不要求任何模型服务配置
            return limited
        scores = self._predict(query, documents)
        if len(scores) != len(documents):
            raise RuntimeError("Rerank 返回分数数量与候选材料不匹配")
        ranked = [
            {
                **document,
                "rerank_score": float(score),
                "vector_rerank_score": float(score),
            }
            for document, score in zip(documents, scores, strict=True)
        ]
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_k]

    def _predict(self, query: str, documents: List[dict]) -> List[float]:
        """
        把 `query-document` 对送入 Cross-Encoder，返回逐条相关性分数。

        这里优先取 `preview`，因为它是检索后最稳定、最短的文本表示；没有 preview 才退回 text。
        """
        texts = [item.get("preview") or item.get("text") or "" for item in documents]
        if not isinstance(query, str) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Rerank 查询与候选材料必须是文本")
        mode = self._settings.rerank_mode
        if mode == "api":
            return self._predict_api(query, texts)
        if mode != "local":
            raise ValueError("RERANK_MODE 必须是 api 或 local")
        model = self._get_model()
        pairs = [(query, text) for text in texts]
        scores = model.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)

    def _predict_api(self, query: str, texts: list[str]) -> list[float]:
        """接收查询和候选文本，请求完整分数并按原候选索引返回有限浮点值。"""
        model_name = get_vector_rerank_model(self._settings).strip()
        if not model_name:
            raise ValueError("Rerank 模型名不能为空")
        options = {} if self._post is None else {"post": self._post}
        payload = request_model_api(self._settings.rerank_endpoint, self._settings.rerank_api_key,
            {"model": model_name, "query": query, "documents": texts,
             "top_n": len(texts), "return_documents": False},
            timeout=self._settings.rerank_api_timeout_seconds, operation="Rerank", **options)
        # 1. 【模型服务】【重排映射】完整取回候选分数，之后才由业务 top_k 截断
        entries = ordered_model_results(payload.get("results"), count=len(texts), operation="Rerank")
        scores = []
        for entry in entries:
            score = entry.get("relevance_score")
            if isinstance(score, bool) or not isinstance(score, (float, int)):
                raise RuntimeError("Rerank API 返回无效相关性分数")
            try:
                score = float(score)
            except OverflowError:
                raise RuntimeError("Rerank API 返回相关性分数超出范围") from None
            if not math.isfinite(score):
                raise RuntimeError("Rerank API 返回非有限相关性分数")
            scores.append(score)
        return scores

    def _get_model(self):
        """无参数；并发首次重排只加载一次本地 CrossEncoder，返回共享模型。"""
        if self._model is not None:
            return self._model
        # 1. 【模型服务】【本地加载】检索线程共用权重，锁内复查防止同时初始化多份模型
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._model_disabled:
                raise RuntimeError("reranker 模型不可用")
            configure_offline_env()
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("缺少 sentence-transformers 依赖，无法启用 reranker") from exc
            try:
                model_name = get_vector_rerank_model(self._settings)
                self._model = CrossEncoder(
                    resolve_local_model_source(model_name),
                    local_files_only=True,
                )
            except Exception as exc:
                self._model_disabled = True
                raise RuntimeError(
                    f"无法从本地缓存加载 vector_rerank: {get_vector_rerank_model(self._settings)}"
                ) from exc
        return self._model


def get_reranker() -> Reranker:
    """返回全局重排器单例；与 embedding/llm 一样，避免重复加载本地模型。"""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER
