"""
未建索引论文检测服务。

这个文件原先承载 RAG 最终回答生成（`execute_analysis` 那条链）。回答生成已由
planner 链的 synthesize 阶段接手，旧链路整体摘除，只留下这一处仍被现役图调用的检测：
`rag_node` 在取证前用它筛出还没进向量库的论文，写进 `unindexed_paper_ids`。
"""

from __future__ import annotations

import logging
from typing import List, Protocol

from app.middleware.observability import log_structured_event

logger = logging.getLogger(__name__)


class _MilvusExistsClient(Protocol):
    def exists(self, paper_id: str) -> bool:  # pragma: no cover - 协议
        ...


def detect_unindexed_papers(
    paper_ids: List[str],
    *,
    milvus_store: _MilvusExistsClient,
) -> List[str]:
    """筛出尚未在 Milvus 主集合中存在向量索引的 paper_id。

    返回保留输入顺序的列表。空输入直接返回空列表，避免无意义日志噪音。
    检测到至少 1 个未索引论文时按 R3.4 发结构化日志事件。
    """
    if not paper_ids:
        return []
    unindexed: List[str] = [pid for pid in paper_ids if not milvus_store.exists(pid)]
    if unindexed:
        log_structured_event(
            logger,
            "rag_unindexed_papers_detected",
            count=len(unindexed),
            paper_ids=unindexed,
        )
    return unindexed
