"""
可观测 trace 上下文与安全写入。

职责：
  - 定义 TraceContext dataclass，通过 contextvars 在 RAG 请求链路中传递
  - 提供 safe_write_trace() 安全写入函数，失败不打断主流程
  - 定义 size cap 常量和截断逻辑
  - 定义 model_call 记录结构和 sanitize 函数
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

from app.config import get_settings

logger = logging.getLogger(__name__)

# 1. 【观测】【记录大小】使用本副本配置，保留 .env 中已有的记录大小设置
OBS_TRACE_MAX_BYTES = get_settings().obs_trace_max_bytes
OBS_CALL_INPUT_MAX = get_settings().obs_call_input_max
OBS_CALL_OUTPUT_MAX = get_settings().obs_call_output_max
EVIDENCE_OBSERVATION_VERSION = "evidence-observation-v1"
RETRIEVAL_OBSERVED_FIELDS = (
    "paper_id", "chunk_id", "parent_id", "section", "section_path", "page_num",
    "score", "text", "preview", "full_text", "display_text", "block_type", "parser_mode",
)
MODEL_EVIDENCE_ID_FIELDS = (
    "evidence_id", "paper_id", "chunk_id", "parent_id", "layer", "section", "section_path", "page_num",
    "read_contract_version", "material_group_id", "source_spans", "source_location_status",
)
_SERIALIZATION_ERRORS = (TypeError, ValueError, OverflowError, RecursionError)

REDACT_KEYS = frozenset({
    "authorization", "api_key", "base_url", "headers", "api_base",
})

_trace_ctx: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "_trace_ctx", default=None
)


@dataclass
class ModelCallRecord:
    """单次 LLM 调用的采集记录。"""

    call_id: str = ""
    node: str = "unknown"
    stage: str = "unknown"
    request_id: str = ""
    evidence_scope_id: str | None = None
    structured_operation_id: str | None = None
    input_messages: list[dict[str, Any]] = field(default_factory=list)
    input_total_chars: int = 0
    output_text: str | None = None
    output_chars: int = 0
    structured_parse: str = "n/a"
    latency_ms: int = 0
    retry_count: int = 0
    http_attempt_count: int = 0
    retry_events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cache_hit: bool = False
    started_at: float = field(default_factory=time.monotonic)

    def finalize(self, output: str | None = None, error: Exception | None = None, *, secret: str = "") -> None:
        """参数为正文、异常及仅用于脱敏的凭证；记录耗时和脱敏错误，不保存凭证或修改原异常。"""
        from app.services.llm_observation import redact_error

        self.latency_ms = max(int((time.monotonic() - self.started_at) * 1000), 0)
        if error is not None:
            self.error = f"{type(error).__name__}: {redact_error(error, secret=secret)[:200]}"
            self.output_text = None
            self.output_chars = 0
        elif output is not None:
            self.output_text = output.replace(secret, "[REDACTED]") if secret else output
            self.output_chars = len(output)


@dataclass
class TraceContext:
    """请求级 trace 上下文，通过 contextvars 在 async 链路中自动传播。"""

    request_id: str = ""
    session_id: str = ""
    user_id: str = ""
    node: str = "unknown"
    stage: str = "unknown"
    call_seq: int = 0
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    evidence_observations: tuple[str, ...] = ()
    evidence_capture_errors: tuple[str, ...] = ()
    _observation_lock: Any = field(default_factory=Lock, init=False, repr=False, compare=False)

    def next_call_id(self) -> str:
        """无参数；返回本请求内唯一调用 ID，线程传播时仍保留原有递增格式。"""
        with self._observation_lock:
            self.call_seq += 1
            return f"{self.request_id}:call-{self.call_seq}"

    def begin_call(self) -> ModelCallRecord:
        """无参数；返回统一登记的调用记录，使用当前任务标签及本请求的证据关联。"""
        scope = _current_evidence_scope(self)
        node, stage = _current_trace_labels(self)
        record = ModelCallRecord(
            call_id=self.next_call_id(),
            node=node,
            stage=stage,
            request_id=self.request_id,
            evidence_scope_id=scope.scope_id if scope else None,
        )
        self.model_calls.append(record)
        return record


def get_trace_ctx() -> TraceContext | None:
    return _trace_ctx.get(None)


def set_trace_ctx(ctx: TraceContext) -> contextvars.Token:
    return _trace_ctx.set(ctx)


def reset_trace_ctx(token: contextvars.Token) -> None:
    _trace_ctx.reset(token)


@dataclass(frozen=True, kw_only=True)
class EvidenceScope:
    """保存本次执行的请求、父调用、阶段和研究单元身份，不包含业务真值。"""

    scope_id: str
    request_id: str
    parent_scope_id: str | None
    stage: str
    unit_id: str | None


_evidence_scope: contextvars.ContextVar[EvidenceScope | None] = contextvars.ContextVar(
    "evidence_scope", default=None,
)
_evidence_scope_owner: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "evidence_scope_owner", default=None,
)


def _current_evidence_scope(ctx: TraceContext) -> EvidenceScope | None:
    """参数为当前 trace；仅返回属于该请求实例的关联，避免重复请求 ID 串联。"""
    return _evidence_scope.get() if _evidence_scope_owner.get() == id(ctx) else None


@contextmanager
def evidence_scope(
    stage: str, *, unit_id: str | None = None, details: Any = None,
) -> Iterator[EvidenceScope | None]:
    """参数为阶段、研究单元与真实执行信息；返回临时关联，结束或异常后恢复父关联。"""
    ctx = get_trace_ctx()
    if ctx is None:
        yield None
        return
    parent = _current_evidence_scope(ctx)
    scope = EvidenceScope(
        scope_id=str(uuid4()), request_id=ctx.request_id,
        parent_scope_id=parent.scope_id if parent else None, stage=stage,
        unit_id=unit_id if unit_id is not None else (parent.unit_id if parent else None),
    )
    token = _evidence_scope.set(scope)
    owner_token = _evidence_scope_owner.set(id(ctx))
    try:
        record_evidence_observation("scope_start", {"details": details})
        yield scope
    except BaseException as exc:
        record_evidence_observation("scope_failed", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    else:
        record_evidence_observation("scope_completed", {})
    finally:
        _evidence_scope.reset(token)
        _evidence_scope_owner.reset(owner_token)


def record_evidence_capture_error(
    stage: str, exc: Exception, *, trace_ctx: TraceContext | None = None,
    scope_id: str | None = None, call_id: str | None = None,
) -> None:
    """参数为采集阶段、异常及可选调用身份；留存明确失败，不修改业务返回或异常。"""
    ctx = trace_ctx if trace_ctx is not None else get_trace_ctx()
    if ctx is None:
        return
    scope = _current_evidence_scope(ctx)
    error = {
        "stage": stage, "request_id": ctx.request_id,
        "scope_id": scope_id if scope_id is not None else (scope.scope_id if scope else None),
        "call_id": call_id, "error_type": type(exc).__name__, "reason": str(exc),
    }
    frozen = json.dumps(error, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with ctx._observation_lock:
        ctx.evidence_capture_errors = (*ctx.evidence_capture_errors, frozen)
    logger.error("【评测观察】【采集失败】%s", error, exc_info=exc)


def record_evidence_observation(stage: str, payload: dict[str, Any]) -> None:
    """参数为真实阶段和数据；保存请求内不可变 JSON，非法序列化输入只登记采集失败。"""
    ctx = get_trace_ctx()
    if ctx is None:
        return
    scope = _current_evidence_scope(ctx)
    event = {
        "schema_version": EVIDENCE_OBSERVATION_VERSION, "event_id": str(uuid4()),
        "request_id": ctx.request_id, "scope": asdict(scope) if scope else None,
        "stage": stage, "observed_at_ns": time.monotonic_ns(), "payload": payload,
    }
    try:
        frozen = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except _SERIALIZATION_ERRORS as exc:
        record_evidence_capture_error(stage, exc)
        return
    with ctx._observation_lock:
        ctx.evidence_observations = (*ctx.evidence_observations, frozen)


def record_retrieval_hits(
    stage: str, hits: list[dict[str, Any]], *, query: str = "", scope: Any = None,
    retrieval_backend: str | None = None, provider_type: str | None = None,
) -> None:
    """参数为实际有序命中、查询及生效范围；保留各返回阶段真实位置与受控文本字段。"""
    if get_trace_ctx() is None:
        return
    try:
        if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
            raise TypeError("retrieval hits must be a list of objects")
        rows = [
            {"rank": rank, **{key: hit[key] for key in RETRIEVAL_OBSERVED_FIELDS if key in hit}}
            for rank, hit in enumerate(hits, start=1)
        ]
    except _SERIALIZATION_ERRORS as exc:
        record_evidence_capture_error(stage, exc)
        return
    record_evidence_observation(stage, {
        "query": query, "scope": scope, "hits": rows,
        "retrieval_backend": retrieval_backend, "provider_type": provider_type,
    })


def record_model_evidence(record: dict[str, Any], source_text: str, *, rendered_text: str) -> None:
    """参数为本次提示词证据、可追溯正文和展示文本；只记录已筛选截断的内容。"""
    if get_trace_ctx() is None:
        return
    if not isinstance(record, dict) or not isinstance(source_text, str) or not isinstance(rendered_text, str):
        record_evidence_capture_error("model_evidence_text", TypeError("invalid model evidence shape"))
        return
    payload = {key: record[key] for key in MODEL_EVIDENCE_ID_FIELDS if key in record}
    record_evidence_observation(
        "model_evidence_text", {**payload, "source_text": source_text, "rendered_text": rendered_text},
    )


def export_evidence_observations(ctx: TraceContext) -> dict[str, Any]:
    """参数为请求 trace；返回独立未裁剪观察，模型消息无效时保留缺失及调用级采集错误。"""
    calls = []
    for call in ctx.model_calls:
        payload = {
            "call_id": call.call_id, "evidence_scope_id": call.evidence_scope_id,
            "structured_operation_id": call.structured_operation_id,
            "node": call.node, "stage": call.stage, "input_messages": call.input_messages,
            "input_total_chars": call.input_total_chars, "error": call.error,
            "output_text": call.output_text, "output_chars": call.output_chars,
            "cache_hit": call.cache_hit, "http_attempt_count": call.http_attempt_count,
            "retry_count": call.retry_count, "retry_events": call.retry_events,
        }
        try:
            calls.append(json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)))
        except _SERIALIZATION_ERRORS as exc:
            record_evidence_capture_error(
                "model_call_observations", exc, trace_ctx=ctx,
                scope_id=call.evidence_scope_id, call_id=call.call_id,
            )
    return {
        "evidence_observations": [json.loads(item) for item in ctx.evidence_observations],
        "model_call_observations": calls,
        "evidence_capture_errors": [json.loads(item) for item in ctx.evidence_capture_errors],
    }


@dataclass(frozen=True, kw_only=True)
class _TraceLabels:
    """保存任务局部的不可变标签，owner 标识所属请求实例。"""

    owner: TraceContext
    node: str
    stage: str


_trace_labels: contextvars.ContextVar[_TraceLabels | None] = contextvars.ContextVar(
    "_trace_labels", default=None,
)


def _current_trace_labels(ctx: TraceContext) -> tuple[str, str]:
    """参数为请求实例；返回本任务的有效标签，没有同实例标签时使用请求默认值。"""
    labels = _trace_labels.get()
    # 1. 【调用跟踪】【阶段归因】同名请求也必须按实例隔离，避免继承旧请求标签
    if labels is not None and labels.owner is ctx:
        return labels.node, labels.stage
    return ctx.node, ctx.stage


class _TraceLabelScope:
    """临时设置当前任务的模型调用标签，退出时恢复该任务的父标签。"""

    __slots__ = ("_node", "_stage", "_token")

    def __init__(self, node: str | None, stage: str | None) -> None:
        """参数为可选节点和阶段；初始化标签作用域，未指定字段继承进入时的有效标签。"""
        self._node = node
        self._stage = stage
        self._token: contextvars.Token[_TraceLabels | None] | None = None

    def __enter__(self) -> TraceContext | None:
        """无参数；设置任务局部标签并返回原请求上下文，未启用跟踪时返回 None。"""
        ctx = get_trace_ctx()
        if ctx is None:
            return None
        node, stage = _current_trace_labels(ctx)
        # 1. 【调用跟踪】【阶段归因】并发单元只替换自己的标签，请求调用记录继续统一汇总
        self._token = _trace_labels.set(_TraceLabels(
            owner=ctx,
            node=self._node if self._node is not None else node,
            stage=self._stage if self._stage is not None else stage,
        ))
        return ctx

    def __exit__(self, exc_type, exc, tb) -> None:
        """参数为作用域退出时的异常信息；恢复父标签，返回 None 保留原异常语义。"""
        if self._token is not None:
            _trace_labels.reset(self._token)
            self._token = None


def trace_label(node: str | None = None, stage: str | None = None) -> _TraceLabelScope:
    """参数为可选节点和阶段；返回任务局部标签作用域，省略字段继承当前有效标签。"""
    return _TraceLabelScope(node, stage)


def truncate_content(text: str, max_bytes: int) -> str:
    if len(text.encode("utf-8", errors="replace")) <= max_bytes:
        return text
    keep = max_bytes // 4
    mid = f"\n[...truncated {len(text) - keep * 2} chars...]\n"
    return text[:keep] + mid + text[-keep:]


def sanitize_call_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k.lower() not in REDACT_KEYS}


def model_call_to_summary(record: ModelCallRecord) -> dict[str, Any]:
    input_msgs = []
    for msg in record.input_messages:
        preview = str(msg.get("content", ""))[:120]
        input_msgs.append({
            "role": msg.get("role", "unknown"),
            "char_count": len(str(msg.get("content", ""))),
            "preview": preview,
        })
    output_preview = (record.output_text or "")[:200]
    return {
        "call_id": record.call_id,
        "evidence_scope_id": record.evidence_scope_id,
        "node": record.node,
        "stage": record.stage,
        "input": {"messages": input_msgs, "total_chars": record.input_total_chars},
        "output": {
            "text_preview": output_preview,
            "char_count": record.output_chars,
            "structured_parse": record.structured_parse,
        },
        "latency_ms": record.latency_ms,
        "retry_count": record.retry_count,
        "http_attempt_count": record.http_attempt_count,
        "retry_events": list(record.retry_events),
        "error": record.error,
        "cache_hit": record.cache_hit,
    }


def model_call_to_debug(record: ModelCallRecord) -> dict[str, Any]:
    input_full = []
    for msg in record.input_messages:
        content = str(msg.get("content", ""))
        input_full.append({
            "role": msg.get("role", "unknown"),
            "content": truncate_content(content, OBS_CALL_INPUT_MAX),
        })
    output_full = truncate_content(record.output_text or "", OBS_CALL_OUTPUT_MAX)
    return {
        "call_id": record.call_id,
        "input_full": input_full,
        "output_full": output_full,
    }


async def safe_write_trace(
    trace_data: dict[str, Any],
    request_id: str,
    session_id: str,
    query: str,
    user_id: str,
) -> None:
    """参数为完整 trace 及归属；保存响应证据，裁剪其他 debug 时不修改原对象，失败显式记录。"""
    from app.repositories.observability_repository import get_observability_repository

    try:
        stored_trace = trace_data
        payload = json.dumps(trace_data, ensure_ascii=False, default=str)
        payload_bytes = len(payload.encode("utf-8"))
        if payload_bytes > OBS_TRACE_MAX_BYTES:
            # 1. 【模型服务】【响应留存】完整响应及校验事件独立于可裁剪 debug，失败现场不能随大 trace 丢失
            stored_trace = {**trace_data, "debug": {"truncated": True, "reason": "size_cap"}}
        repo = get_observability_repository()
        await repo.write_rag_trace(
            request_id=request_id,
            session_id=session_id,
            query=query,
            trace_data=stored_trace,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(
            "【研究观察】【记录写入】trace 写入失败 request_id=%s session_id=%s: %s",
            request_id,
            session_id,
            exc,
            exc_info=True,
        )


def build_trace_summary(
    trace_ctx: TraceContext,
    rag_trace_dict: dict[str, Any],
) -> dict[str, Any]:
    """构建随消息返回的裁剪版 trace summary。"""
    from app.middleware.trace_builders import build_trace_summary_payload

    return build_trace_summary_payload(trace_ctx, rag_trace_dict)


def build_full_trace(
    trace_ctx: TraceContext,
    rag_trace_dict: dict[str, Any],
) -> dict[str, Any]:
    """构建落库的完整 trace，含 stages + model_calls + debug。"""
    from app.middleware.trace_builders import build_full_trace_payload

    return build_full_trace_payload(
        trace_ctx,
        rag_trace_dict,
        call_summary_fn=model_call_to_summary,
        call_debug_fn=model_call_to_debug,
    )
