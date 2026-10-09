"""
LightRAG 图谱增强服务封装。

这个模块提供 GraphRAG 能力，供 `rag_node.py` 在 `graph` 或 `fusion`
 路径下按需调用。当前系统里它是 Milvus 主检索链之外的增强层，不是默认的
 chunk 主索引来源：
  - Milvus 负责主链 chunk 命中与 preview 检索
  - LightRAG 负责图谱增强和聚合式文本回答
  - `library_ingest.py` 会在主链入库后异步把长文本送入这里构建图谱知识

它的定位更像“每个 library 一个本地 GraphRAG runtime”，而不是一个中心化远程服务。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.prompt import PROMPTS as LIGHTRAG_PROMPTS
    from lightrag.utils import wrap_embedding_func_with_attrs
except ImportError as exc:
    LightRAG = QueryParam = initialize_pipeline_status = LIGHTRAG_PROMPTS = wrap_embedding_func_with_attrs = None
    _LIGHTRAG_IMPORT_ERROR = exc
else:
    _LIGHTRAG_IMPORT_ERROR = None

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import get_embedding_identity, get_settings
from app.models.library_models import DEFAULT_LIBRARY_ID
from app.services.lightrag_domain_prompt import (
    build_addon_params,
    install_domain_prompts,
    reset_domain_prompt_flag,
)
from app.services.embedding_client import get_embedding_client
from app.services.lightrag_operations import (
    LightRAGInsertMixin,
    LightRAGMaintenanceMixin,
    LightRAGQueryMixin,
    extract_paper_id_from_payload,
)
from app.services.llm_service import get_llm

settings = get_settings()
logger = logging.getLogger(__name__)
SearchMode = Literal["local", "global", "hybrid", "naive"]
MAX_EMBEDDING_TOKENS = 8192
_SERVICE: "LightRAGService | None" = None


def _require_dependencies() -> None:
    """显式校验 GraphRAG 所需依赖，缺失时直接暴露错误而不是静默降级。"""
    if _LIGHTRAG_IMPORT_ERROR is not None:
        raise RuntimeError("缺少 lightrag-hku 依赖，无法启用 GraphRAG") from _LIGHTRAG_IMPORT_ERROR


def _install_domain_prompts() -> bool:
    return install_domain_prompts(LIGHTRAG_PROMPTS)


def _build_history_messages(history_messages: list[dict[str, Any]] | None) -> list[Any]:
    """把历史消息字典转成 LangChain message，供 LightRAG 的 LLM 回调复用。"""
    messages: list[Any] = []
    for item in history_messages or []:
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        if not content:
            continue
        if role == "system":
            messages.append(SystemMessage(content=content))
            continue
        if role == "assistant":
            messages.append(AIMessage(content=content))
            continue
        messages.append(HumanMessage(content=content))
    return messages


class LightRAGService(LightRAGInsertMixin, LightRAGQueryMixin, LightRAGMaintenanceMixin):
    """
    LightRAG 单例服务。

    这里封装的是图谱检索增强层，不直接替代 Milvus 的 chunk 向量索引。
    `query()` 返回的是 LightRAG 聚合后的文本结果，而不是 chunk 列表。
    """

    def __init__(self) -> None:
        """初始化运行时容器；无参数，按配置隔离 API 向量空间并保留 local 目录。"""
        self._runtimes: dict[str, "_LightRAGRuntime"] = {}
        # LightRAG 会在本地工作目录下维护自己的索引、图谱和中间状态文件。
        self._working_root = Path(settings.data_dir) / "lightrag_index"
        # 1. 【图谱检索】【向量隔离】同名模型的云端向量不复用本地已有图谱索引
        if settings.embedding_mode == "api":
            self._working_root = self._working_root / get_embedding_identity(settings)

    def reset_runtimes(self, reason: str = "") -> None:
        """清空 runtime；切换领域 prompt 设置后必须调用或重启服务。"""
        logger.info("LightRAG runtimes reset: count=%d reason=%s", len(self._runtimes), reason)
        self._runtimes.clear()
        reset_domain_prompt_flag()

    def _ensure_enabled(self) -> None:
        """统一检查 GraphRAG 开关，避免未启用时误初始化。"""
        if not settings.graph_rag_enabled:
            raise RuntimeError("GraphRAG 未启用，请先将 graph_rag_enabled 设为 true")

    def _build_embedding_func(self):
        """构造给 LightRAG 使用的异步 embedding 回调。"""
        client = get_embedding_client()
        embedding_dim = int(client.dimension)

        @wrap_embedding_func_with_attrs(
            embedding_dim=embedding_dim,
            max_token_size=MAX_EMBEDDING_TOKENS,
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            if not texts:
                return np.empty((0, embedding_dim), dtype=np.float32)
            batch_size = max(
                1,
                min(
                    len(texts) or 1,
                    int(settings.embedding_batch_size),
                    int(getattr(settings, "lightrag_embedding_batch_size", 4)),
                ),
            )
            started_at = time.monotonic()
            logger.info(
                "LightRAG embedding start: text_count=%d batch_size=%d model=%s device=%s",
                len(texts),
                batch_size,
                settings.embedding_model,
                settings.embedding_device,
            )
            vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                vectors.extend(await asyncio.to_thread(client.embed_texts, batch))
            duration_ms = int((time.monotonic() - started_at) * 1000)
            logger.info(
                "LightRAG embedding done: text_count=%d batch_size=%d duration_ms=%d",
                len(texts),
                batch_size,
                duration_ms,
            )
            return np.asarray(vectors, dtype=np.float32)

        return embedding_func

    @staticmethod
    def _build_llm_func():
        """构造给 LightRAG 使用的异步 LLM 回调，底层仍复用项目统一 LLM 服务。"""
        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **_: Any,
        ) -> str:
            llm = get_llm()
            messages = _build_history_messages(history_messages)
            if system_prompt:
                messages.insert(0, SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            response = await llm.ainvoke(messages)
            return response.content

        return llm_model_func

    def _normalize_library_id(self, library_id: str | None) -> str:
        candidate = (library_id or "").strip()
        return candidate or DEFAULT_LIBRARY_ID

    def _working_dir_for(self, library_id: str) -> Path:
        if library_id == DEFAULT_LIBRARY_ID:
            return self._working_root
        return self._working_root / library_id

    def _get_runtime(self, library_id: str | None) -> "_LightRAGRuntime":
        normalized = self._normalize_library_id(library_id)
        runtime = self._runtimes.get(normalized)
        if runtime is None:
            # runtime 以 library 为粒度隔离，避免不同资料库共享同一份图谱目录。
            runtime = _LightRAGRuntime(
                library_id=normalized,
                working_dir=self._working_dir_for(normalized),
            )
            self._runtimes[normalized] = runtime
        return runtime

    async def initialize(self, library_id: str | None = None) -> None:
        """初始化 LightRAG 存储与运行时对象；只在首次使用时执行一次。"""
        self._ensure_enabled()
        runtime = self._get_runtime(library_id)
        if runtime.initialized:
            return
        async with runtime.init_lock:
            if runtime.initialized:
                return
            _require_dependencies()
            runtime.working_dir.mkdir(parents=True, exist_ok=True)
            domain_prompt_enabled = _install_domain_prompts()
            runtime.rag = LightRAG(
                working_dir=str(runtime.working_dir),
                llm_model_func=self._build_llm_func(),
                embedding_func=self._build_embedding_func(),
                # CPU 上 bge-m3 embedding 较慢，提升 worker 超时避免 60s timeout
                # max_execution_timeout = default_embedding_timeout * 2 = 240s
                default_embedding_timeout=120,
                addon_params=build_addon_params(domain_prompt_enabled),
            )
            await runtime.rag.initialize_storages()
            # LightRAG 内部还有一层 pipeline status，全局初始化缺失时后续 insert/query 会报错。
            await initialize_pipeline_status()
            runtime.initialized = True
            logger.info(
                "LightRAG 初始化完成: library=%s dir=%s insert_concurrency=%d embedding_batch_size=%d",
                runtime.library_id,
                runtime.working_dir,
                int(getattr(settings, "lightrag_insert_max_concurrency", 1)),
                int(getattr(settings, "lightrag_embedding_batch_size", 4)),
            )

    async def is_healthy(self, *, library_id: str | None = None) -> bool:
        """健康检查：禁用时视为健康，启用时要求已完成初始化且 `_rag` 可用。"""
        if not settings.graph_rag_enabled:
            return True
        runtime = self._get_runtime(library_id)
        return runtime.initialized and runtime.rag is not None


@dataclass
class _LightRAGRuntime:
    """
    单个 library 的 LightRAG 运行时。

    之所以单独抽出 runtime，而不是把状态平铺在 service 上，是因为 LightRAG 自身有
    明确的“工作目录 + 已初始化对象 + 并发控制”三件套，需要按 library 隔离管理。
    """

    library_id: str
    working_dir: Path
    rag: Any = None
    initialized: bool = False
    init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    insert_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(
            max(1, int(getattr(settings, "lightrag_insert_max_concurrency", 1)))
        )
    )


def get_lightrag_service() -> LightRAGService:
    """返回全局 LightRAG 服务单例。"""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = LightRAGService()
    return _SERVICE
