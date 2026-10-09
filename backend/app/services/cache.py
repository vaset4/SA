"""
Phase 4 的 L1 内存缓存实现。

当前文件只实现进程内缓存，不是完整的 Redis/L2 缓存层。项目里虽然有些地方会
探测 `cache.l2` 或按配置判断 Redis 能力，但这些二级缓存逻辑不在本文件完整
落地。

三类缓存的当前职责如下：
  - `embedding`: 供向量检索链复用 embedding 结果，典型调用方是 `milvus_store.py`
  - `search`: 供 `rag_node.py` 复用 `milvus / graph / fusion` 检索结果
  - `llm`: 供 `llm_service.py` 复用模型响应文本

这里的设计重点不是“全局缓存平台”，而是给主调用链提供一个廉价、可预测的 L1：
  - 进程重启即失效
  - 不跨 worker 共享
  - 通过深拷贝避免调用方意外篡改缓存对象
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.config import get_settings

settings = get_settings()

EMBEDDING = "embedding"
SEARCH = "search"
LLM = "llm"


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLMemoryCache:
    """简单的 LRU + TTL 内存缓存，作为各业务缓存类别的底层容器。"""

    def __init__(self, maxsize: int, ttl_seconds: int) -> None:
        """保存容量和 TTL 配置，并初始化按访问顺序维护的有序字典。"""
        self._maxsize = max(1, int(maxsize))
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """读取缓存命中值；命中后刷新 LRU 顺序，并返回深拷贝避免外部篡改。"""
        self._purge_expired(key)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return deepcopy(entry.value)

    def set(self, key: str, value: Any) -> None:
        """写入缓存值，保存深拷贝并在超出容量时淘汰最旧条目。"""
        self._entries[key] = CacheEntry(
            value=deepcopy(value),
            expires_at=time.monotonic() + self._ttl_seconds,
        )
        self._entries.move_to_end(key)
        self._evict_if_needed()

    def clear(self) -> None:
        """清空当前类别的全部缓存项。"""
        self._entries.clear()

    def delete(self, key: str) -> None:
        """删除单个缓存项；不存在时静默忽略。"""
        self._entries.pop(key, None)

    def size(self) -> int:
        """返回当前未过期条目数。"""
        self._purge_all_expired()
        return len(self._entries)

    def _evict_if_needed(self) -> None:
        """容量超限时按 LRU 规则淘汰最旧条目。"""
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    def _purge_expired(self, key: str) -> None:
        """按 key 懒清理过期项。"""
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.expires_at <= time.monotonic():
            self._entries.pop(key, None)

    def _purge_all_expired(self) -> None:
        """扫描并移除全部过期项，主要用于统计场景。"""
        now = time.monotonic()
        expired_keys = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired_keys:
            self._entries.pop(key, None)


class CacheManager:
    """
    统一管理 embedding/search/llm 三类 L1 缓存。

    这是当前缓存主实现，`settings.cache_enabled` 是总开关。这里没有完整实现
    Redis 二级缓存协议，因此注释和调用方都应把它理解为“进程内缓存管理器”。
    """

    def __init__(self) -> None:
        """按配置初始化三类缓存实例和命中统计。"""
        self._caches = {
            EMBEDDING: TTLMemoryCache(
                settings.cache_embedding_maxsize,
                settings.cache_embedding_ttl_seconds,
            ),
            SEARCH: TTLMemoryCache(
                settings.cache_search_maxsize,
                settings.cache_search_ttl_seconds,
            ),
            LLM: TTLMemoryCache(
                settings.cache_llm_maxsize,
                settings.cache_llm_ttl_seconds,
            ),
        }
        self._stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "search_hits": 0,
            "search_misses": 0,
            "llm_hits": 0,
            "llm_misses": 0,
        }
        # 这里的统计只反映当前进程自启动以来的命中情况，不做持久化或跨实例汇总。

    @staticmethod
    def _hash(text: str) -> str:
        """
        对原始 key 做稳定哈希，避免把长文本直接暴露在内存索引里。

        搜索和 LLM prompt 的 key 往往很长，先哈希后再存储可以降低内存索引开销，
        也让调试输出不会直接泄露完整问题文本。
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embedding(self, query: str) -> list[float] | None:
        """读取 embedding 缓存，主要服务向量检索链。"""
        return self._get(EMBEDDING, query)

    def set_embedding(self, query: str, embedding: list[float]) -> None:
        """写入 embedding 缓存。"""
        self._set(EMBEDDING, query, embedding)

    def get_search(self, query: str, filter_expr: str = "", mode: str = "milvus") -> Any | None:
        """读取搜索缓存；按检索模式隔离 key，避免不同来源结果串用。"""
        return self._get(SEARCH, f"{mode}|{query}|{filter_expr}")

    def set_search(self, query: str, filter_expr: str, results: Any, mode: str = "milvus") -> None:
        """写入搜索缓存；value 可以是列表或分源 payload。"""
        self._set(SEARCH, f"{mode}|{query}|{filter_expr}", results)

    def get_llm_response(self, prompt_key: str) -> str | None:
        """读取 LLM 响应缓存，供 `llm_service.py` 直接复用。"""
        return self._get(LLM, prompt_key)

    def set_llm_response(self, prompt_key: str, response: str) -> None:
        """写入 LLM 响应缓存。"""
        self._set(LLM, prompt_key, response)

    def delete_llm_response(self, prompt_key: str) -> None:
        """删除单个 LLM 响应缓存，供坏缓存自恢复使用。"""
        self._delete(LLM, prompt_key)

    def clear_all(self) -> None:
        """清空全部 L1 缓存类别。"""
        for cache in self._caches.values():
            cache.clear()

    def get_stats(self) -> dict:
        """返回命中/未命中统计和各缓存类别当前大小。"""
        return {
            **self._stats,
            "embedding_size": self._caches[EMBEDDING].size(),
            "search_size": self._caches[SEARCH].size(),
            "llm_size": self._caches[LLM].size(),
        }

    def _get(self, category: str, key: str) -> Any | None:
        """
        带总开关与统计更新的统一读取入口。

        这里故意把“总开关判断、哈希、命中统计”集中在一处，避免上层调用方各自散落
        一套缓存判断逻辑。
        """
        if not settings.cache_enabled:
            return None
        result = self._caches[category].get(self._hash(key))
        stat_key = f"{category}_hits" if result is not None else f"{category}_misses"
        self._stats[stat_key] += 1
        return result

    def _set(self, category: str, key: str, value: Any) -> None:
        """带总开关的统一写入入口。"""
        if not settings.cache_enabled:
            return
        self._caches[category].set(self._hash(key), value)

    def _delete(self, category: str, key: str) -> None:
        """带总开关的统一删除入口。"""
        if not settings.cache_enabled:
            return
        self._caches[category].delete(self._hash(key))


_cache: CacheManager | None = None


def get_cache() -> CacheManager:
    """
    返回进程级缓存管理器单例。

    主链所有缓存访问都应该汇聚到这里，避免不同模块各自创建独立缓存实例导致命中率
    被意外打散。
    """
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache
