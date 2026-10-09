"""
RAG 请求级结构化追踪。

这个模块属于可观测性辅助层，不参与主业务决策。当前职责是：
  - 为单次 RAG 请求维护一份轻量 trace 上下文
  - 记录检索、重排、分级、生成、评估等阶段耗时
  - 把结构化 trace 异步写入业务库的 `rag_traces` 表

它与 `rag_node.py` 的关系是：
  - `rag_node.py` 负责在各阶段调用 `checkpoint()` 和更新指标字段
  - 这里负责 trace 结构定义、序列化和持久化
"""

from __future__ import annotations

import asyncio
# 导入 json，用于把结构化字段压成稳定日志字符串。
import json
# 导入 logging，用于在 trace 持久化失败时记警告日志。
import logging
# 导入 time，用于统计耗时。
import time
# 导入 uuid，为每次请求生成稳定 request_id。
import uuid
# dataclass + field 让 trace 结构更轻量直观。
from dataclasses import dataclass, field
# Any 用于描述 trace 字典里可能出现的动态值。
from typing import Any

# 复用业务数据库服务，把 trace 作为业务日志写入 PostgreSQL。
from app.services import paper_db

# 模块级 logger，仅用于记录 trace 写入失败等辅助日志。
logger = logging.getLogger(__name__)
# 当前按这些阶段名统计耗时；需与 `rag_node.py` 里打点名称保持一致。
_LATENCY_STAGES = ("retrieval", "rerank", "grading", "generation", "evaluation")

# 这个文件定义的是“请求级 trace 结构”，而不是中间件意义上的 HTTP middleware。
# 历史命名保留了 observability/middleware 概念，但实际用途更接近 tracing helper。


def _default_retrieval_metrics() -> dict[str, int]:
    """返回检索阶段指标的默认结构。"""
    return {
        "hybrid_results": 0,
        "reranked": 0,
        "graded_relevant": 0,
        "graded_irrelevant": 0,
    }


def _normalize_log_value(value: Any) -> Any:
    """把日志字段压成可稳定序列化的轻量值。"""
    if value is None:
        return None
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        normalized = {
            str(key): _normalize_log_value(item)
            for key, item in value.items()
            if _normalize_log_value(item) is not None
        }
        return normalized or None
    if isinstance(value, (list, tuple, set)):
        items = [_normalize_log_value(item) for item in value]
        items = [item for item in items if item is not None]
        return items or None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) > 240:
        return text[:237].rstrip() + "..."
    return text


def log_structured_event(
    current_logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """统一输出简洁的结构化日志事件。"""
    payload = {}
    for key, value in fields.items():
        normalized = _normalize_log_value(value)
        if normalized is not None:
            payload[str(key)] = normalized
    current_logger.log(
        level,
        "event=%s fields=%s",
        event,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def write_observation_event(
    event_type: str,
    node_name: str,
    *,
    query: str = "",
    result_count: int = 0,
    latency_ms: int = 0,
    session_id: str | None = None,
) -> None:
    """统一写入结构化观测事件。"""
    try:
        paper_db.write_log(
            event_type=event_type,
            node_name=node_name,
            query=query,
            result_count=result_count,
            latency_ms=latency_ms,
            session_id=session_id,
        )
    except Exception as exc:  # pragma: no cover - 仅记录异常
        logger.warning(
            "写入观测事件失败 event_type=%s session_id=%s: %s",
            event_type,
            session_id,
            exc,
            exc_info=True,
        )


def record_timed_event(
    event_type: str,
    node_name: str,
    started_at: float,
    *,
    query: str = "",
    result_count: int = 0,
    session_id: str | None = None,
) -> None:
    """按开始时间自动计算耗时并写入观测事件。"""
    latency_ms = max(int((time.monotonic() - started_at) * 1000), 0)
    write_observation_event(
        event_type,
        node_name,
        query=query,
        result_count=result_count,
        latency_ms=latency_ms,
        session_id=session_id,
    )


@dataclass(slots=True)
class RAGTrace:
    """
    单次 RAG 请求的最小追踪上下文。

    它不是业务状态本身，而是围绕一次请求采集的旁路观测数据。
    """

    # 当前请求所属的会话 ID，通常与 LangGraph thread_id 对齐。
    session_id: str = ""
    # 原始用户查询。
    query: str = ""
    # 每次请求独立生成的 request_id，便于日志和数据库追踪。
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # 查询改写后得到的候选 query 列表。
    rewritten_queries: list[str] = field(default_factory=list)
    # 本次请求最终采用的检索模式。
    mode: str = ""
    # 当前对话模式。
    dialogue_mode: str = ""
    # 当前图检索模式。
    graph_mode: str = ""
    # 检索路径选择原因。
    retrieval_reasoning: str = ""
    # 各数据源状态，用于排障。
    source_status: dict[str, str] = field(default_factory=dict)
    # 本次请求实际运行的 RAG 链路。
    runtime_chain: str = ""
    # 请求最终收束结果，例如 cache_hit / rag_hit / rag_miss / all_sources_failed。
    outcome: str = ""
    # 当前项目范围 ID。
    project_id: str = ""
    # 当前请求绑定的论文范围。
    paper_ids: list[str] = field(default_factory=list)
    # 当前论文焦点模式。
    paper_focus_mode: str = ""
    # 整理层状态。
    synthesis_status: str = ""
    # 当前回答附带的证据层角色。
    evidence_layers: dict[str, str] = field(default_factory=dict)
    # 本次请求各证据层实际参与状态。
    evidence_layer_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 检索阶段的计数指标。
    retrieval: dict[str, int] = field(default_factory=_default_retrieval_metrics)
    # Self-RAG 迭代次数。
    self_rag_iterations: int = 0
    # 最终质量评估分。
    final_quality_score: float | None = None
    # 最终答案包含的引用条数。
    citations_count: int = 0
    # 05-24: 是否执行了 graph_prefetch 前置节点。
    graph_prefetched: bool = False
    # 05-24: 是否经过 sufficiency_grader 判定。
    sufficiency_graded: bool = False
    # 各检查点时间戳，内部使用，不暴露在 repr 中。
    _checkpoints: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    # 请求起始时间，用于统计总耗时。
    _start_time: float = field(default_factory=lambda: time.monotonic(), init=False, repr=False)

    def checkpoint(self, name: str) -> None:
        """记录一个命名检查点的当前时间戳。"""
        self._checkpoints[name] = time.monotonic()

    def _latency_ms(self, start_name: str, end_name: str) -> int:
        """根据两个检查点计算耗时；缺起点则记为 0，缺终点则截到当前时刻。"""
        start = self._checkpoints.get(start_name)
        if start is None:
            return 0
        end = self._checkpoints.get(end_name)
        if end is None:
            end = time.monotonic()
        return max(int((end - start) * 1000), 0)

    def to_dict(self) -> dict[str, Any]:
        """把 trace 转成可写入数据库/日志系统的结构化字典。"""
        # 先生成总耗时。
        latency_ms = {
            "total": max(int((time.monotonic() - self._start_time) * 1000), 0),
        }
        # 再按约定阶段补齐各阶段耗时。
        for stage in _LATENCY_STAGES:
            latency_ms[stage] = self._latency_ms(f"{stage}_start", f"{stage}_end")
        return {
            # 请求级唯一标识。
            "request_id": self.request_id,
            # 会话标识。
            "session_id": self.session_id,
            # 原始查询。
            "query": self.query,
            # 查询改写结果。
            "rewritten_queries": list(self.rewritten_queries),
            # 路由与数据源状态。
            "mode": self.mode,
            "dialogue_mode": self.dialogue_mode,
            "graph_mode": self.graph_mode,
            "retrieval_reasoning": self.retrieval_reasoning,
            "source_status": dict(self.source_status),
            "runtime_chain": self.runtime_chain,
            "outcome": self.outcome,
            "scope": {
                "project_id": self.project_id,
                "paper_ids": list(self.paper_ids),
                "paper_focus_mode": self.paper_focus_mode,
            },
            "synthesis": {
                "status": self.synthesis_status,
                "evidence_layers": dict(self.evidence_layers),
                "evidence_layer_status": dict(self.evidence_layer_status),
            },
            # 检索相关指标。
            "retrieval": dict(self.retrieval),
            # 生成/收敛阶段相关指标。
            "generation": {
                "self_rag_iterations": self.self_rag_iterations,
                "final_quality_score": self.final_quality_score,
                "citations_count": self.citations_count,
            },
            # 各阶段耗时。
            "latency_ms": latency_ms,
        }


async def save_rag_trace(trace: RAGTrace) -> None:
    """
    把 trace 写入业务库。

    这里写入失败不会反向影响主回答链，只记录 warning。
    """

    try:
        # 当前 trace 通过 `paper_db.write_rag_trace(...)` 落到 PostgreSQL `rag_traces` 表。
        await asyncio.to_thread(
            paper_db.write_rag_trace,
            request_id=trace.request_id,
            session_id=trace.session_id,
            query=trace.query,
            trace_data=trace.to_dict(),
        )
    except Exception as exc:  # pragma: no cover - 仅记录异常
        # 可观测性失败不阻断主流程，只记日志暴露问题。
        logger.warning(
            "保存 RAG trace 失败 request_id=%s session_id=%s: %s",
            trace.request_id,
            trace.session_id,
            exc,
            exc_info=True,
        )


async def persist_rag_trace(trace: RAGTrace | None) -> None:
    """在当前请求路径显式等待 trace 保存。"""
    if trace is None:
        return
    await save_rag_trace(trace)
