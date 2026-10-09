"""【评测观察】【模型边界】记录真实调用与校验事实，不改变产品行为。作者：xxx。"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import requests
from pydantic import BaseModel, ValidationError

from app.middleware.trace_context import (
    get_trace_ctx,
    record_evidence_capture_error,
    record_evidence_observation,
)

ERROR_PREVIEW_CHARS = 1200
_operation: ContextVar[tuple[Any, dict[str, Any], str] | None] = ContextVar(
    "structured_operation", default=None,
)
_CREDENTIAL = re.compile(
    r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?|bearer\s+|"
    r"(?:api[_-]?key|access[_-]?token|secret|password)[\"']?\s*[:=]\s*[\"']?)"
    r"[^\s\"',;}]+",
)
_URL = re.compile(r"https?://[^\s\"'<>]+")


def safe_endpoint(value: str) -> str:
    """参数为实际地址；返回移除用户信息、查询与片段的接口身份。"""
    try:
        parsed = urlsplit(value)
    except ValueError:
        # 1. 【评测观察】【地址脱敏】错误正文可能含畸形URL，采集缺失不能替代原业务异常
        record_evidence_capture_error("endpoint_redaction", ValueError("invalid endpoint syntax; address omitted"))
        return "[INVALID_ENDPOINT]"
    host = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def redact_error(value: Any, *, secret: str = "", limit: int | None = ERROR_PREVIEW_CHARS) -> str:
    """参数为错误与已知凭证；返回脱敏摘要，不保留请求认证信息。"""
    text = str(value)
    if secret:
        # 1. 【评测观察】【凭证脱敏】原HTTP正文仍保留JSON转义，先替换完整编码凭证再截断显示
        for encoded_secret in (json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]):
            text = text.replace(encoded_secret, "[REDACTED]")
        text = text.replace(secret, "[REDACTED]")
    text = _CREDENTIAL.sub(r"\1[REDACTED]", text)
    text = _URL.sub(lambda match: safe_endpoint(match.group()), text)
    return text[:limit] if limit is not None else text


def structured_identity() -> dict[str, Any]:
    """无参数；返回当前请求内的结构化操作事实副本，不跨请求继承。"""
    active = _operation.get()
    if active is None or active[0] is not get_trace_ctx():
        return {}
    return dict(active[1])


@contextmanager
def structured_operation(schema: str, *, secret: str = "") -> Iterator[dict[str, Any]]:
    """参数为schema及脱敏凭证；记录完整操作起止，返回本操作的既有路径状态。"""
    obs: dict[str, Any] = {
        "operation_id": str(uuid4()), "schema": schema, "actor": "product",
        "phase": "initial", "native_mode": "disabled", "repair_attempted": False,
        "repair_count": 0, "repair_succeeded": False,
    }
    token = _operation.set((get_trace_ctx(), obs, secret))
    record_evidence_observation("structured_operation_start", dict(obs))
    try:
        yield obs
    except BaseException as exc:
        record_evidence_observation("structured_operation_end", {
            **obs, "status": "failed", "error_type": type(exc).__name__,
            "error_message": redact_error(exc, secret=secret),
        })
        raise
    else:
        record_evidence_observation("structured_operation_end", {**obs, "status": "success", "failure_stage": None})
    finally:
        _operation.reset(token)


def bind_structured_call(call_id: str) -> str | None:
    """参数为真实逻辑调用ID；关联到当前操作并返回操作ID，不构造HTTP尝试。"""
    active = _operation.get()
    if active is None or active[0] is not get_trace_ctx():
        return None
    active[1]["call_id"] = call_id
    return str(active[1]["operation_id"])


def record_structured_path(obs: dict[str, Any], path: str) -> None:
    """参数为当前状态及已发生路径；记录缓存失效、native拒绝或结构回修事实。"""
    record_evidence_observation("structured_path", {**obs, "path": path})


def value_fingerprint(value: Any) -> str:
    """参数为实际模型或JSON值；返回内容指纹供跨节点校验关联，不写入业务对象。"""
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def record_structured_validation(
    content: str, *, schema: str, result: Any = None,
    error: Exception | None = None, parsing: dict[str, Any],
) -> None:
    """参数为实际解析输入、结果和解析事实；记录schema结果，采集失败不改变原异常。"""
    identity = structured_identity()
    if not identity or get_trace_ctx() is None:
        return
    if not content:
        record_evidence_observation("structured_response_unavailable", {
            **identity, "reason": "empty_assistant_content", "error_boundary": "api_envelope",
        })
        return
    try:
        try:
            json.loads(content)
            raw_valid = True
        except ValueError:
            raw_valid = False
        active = _operation.get()
        if active is not None:
            if identity.get("phase") == "initial":
                active[1]["original_compliance"] = error is None if not identity.get("cache_hit") else None
            active[1]["final_compliance"] = error is None
        field_errors = error.errors(include_input=False, include_url=False, include_context=False) if isinstance(error, ValidationError) else []
        if active is not None:
            field_errors = [{**item, "msg": redact_error(item["msg"], secret=active[2], limit=None)} for item in field_errors]
        record_evidence_observation("structured_response_validation", {
            **identity, **parsing, "schema": schema, "schema_valid": error is None,
            "raw_json_valid": raw_valid,
            "response_source": "cache" if identity.get("cache_hit") else "model",
            "response_text": content.replace(active[2], "[REDACTED]") if active and active[2] else content,
            "field_errors": field_errors,
            "failure_stage": ("model_schema" if raw_valid else "model_json") if error else None,
            "error_type": type(error).__name__ if error else None,
            "error_message": redact_error(error, secret=active[2]) if error and active else None,
            "output_fingerprint": value_fingerprint(result) if error is None else None,
        })
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        record_evidence_capture_error("structured_response_validation", exc)


def http_error_facts(
    error: Exception, *, response: Any = None, boundary: str, secret: str = "",
) -> dict[str, Any]:
    """参数为实际异常、已得响应及失败边界；返回脱敏错误证据，不推测供应商责任。"""
    error_response = getattr(error, "response", None)
    actual = error_response if error_response is not None else response
    status = getattr(actual, "status_code", None)
    if isinstance(error, requests.exceptions.HTTPError):
        boundary = "http_status"
    elif isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        boundary = "transport"
    facts: dict[str, Any] = {
        "http_status": status if isinstance(status, int) else None,
        "error_type": type(error).__name__, "error_boundary": boundary,
        "error_message": redact_error(error, secret=secret),
    }
    request = getattr(error, "request", None)
    request_url = getattr(actual, "url", None) or getattr(request, "url", None)
    if isinstance(request_url, str):
        facts["endpoint"] = safe_endpoint(request_url)
    # 1. 【评测观察】【API错误】只读取已经消费的错误正文，不再次消费SSE或成功响应
    body = getattr(actual, "_content", None)
    if isinstance(error, requests.exceptions.HTTPError) and isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
        facts["response_summary"] = redact_error(text, secret=secret)
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        detail = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            for key in ("code", "type", "param"):
                if isinstance(detail.get(key), (str, int)):
                    facts[f"provider_error_{key}"] = redact_error(detail[key], secret=secret)
    return facts


def model_attempt_identity(
    model: str, endpoint: str, *, call_record: Any, streaming: bool,
) -> dict[str, Any]:
    """参数为模型、接口和真实调用；返回可追溯HTTP尝试身份，不包含凭证。"""
    return {
        "actor": "product", "model": model, "endpoint": safe_endpoint(endpoint),
        "call_id": call_record.call_id if call_record else None,
        "http_attempt": call_record.http_attempt_count if call_record else None,
        "operation_id": structured_identity().get("operation_id"),
        "streaming": streaming, "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def record_plan_validation(raw: Any, result: Any, *, phase: str, schema: str | None = None) -> None:
    """参数为实际计划输入、校验结果和回修阶段；记录权限/依赖校验及可证明的操作关联。"""
    if get_trace_ctx() is None:
        return
    metadata = result.plan.metadata.model_dump(mode="json") if result.plan else result.failure.trace
    record_business_validation(raw, {
        "validator": "research_planner", "schema": schema or (type(raw).__name__ if isinstance(raw, BaseModel) else None),
        "phase": phase, "validation_stage": "plan_scope_dependencies", "valid": result.is_valid,
        "accepted": result.is_valid, "errors": metadata.get("validation_errors", []),
        "repairs": metadata.get("repairs", []),
        "accepted_plan": result.plan.model_dump(mode="json") if result.plan else None,
    })


def _matching_structured_responses(ctx: Any, *, schema: str | None, fingerprint: str) -> list[dict[str, Any]]:
    """参数为本请求 trace、schema 和对象指纹；返回真实成功解析来源，保留所有候选事件。"""
    matches = []
    for serialized in ctx.evidence_observations:
        event = json.loads(serialized)
        payload = event["payload"]
        if (event["stage"] == "structured_response_validation" and payload.get("schema_valid") is True
                and payload.get("schema") == schema and payload.get("output_fingerprint") == fingerprint
                and payload.get("operation_id")):
            matches.append(event)
    return matches


def record_structured_conversion(raw: Any, converted: Any, *, schema: str, target_schema: str | None = None) -> None:
    """参数为模型对象、转换结果及源/目标schema；记录真实指纹映射，仅唯一来源可关联操作。"""
    ctx = get_trace_ctx()
    if ctx is None:
        return
    try:
        fingerprint = value_fingerprint(raw)
        matches = _matching_structured_responses(ctx, schema=schema, fingerprint=fingerprint)
        candidates = [event["payload"]["operation_id"] for event in matches]
        record_evidence_observation("structured_conversion", {
            "actor": "product", "schema": schema, "conversion_id": str(uuid4()),
            **({"target_schema": target_schema} if target_schema is not None else {}),
            "input_fingerprint": fingerprint, "output_fingerprint": value_fingerprint(converted),
            "operation_id": candidates[0] if len(matches) == 1 else None,
            "matching_operation_ids": candidates,
            "source_validation_event_ids": [event["event_id"] for event in matches],
            "source_association": "matched" if len(matches) == 1 else "ambiguous" if matches else "missing",
        })
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        record_evidence_capture_error("structured_conversion", exc)


def record_business_validation(raw: Any, details: dict[str, Any]) -> None:
    """参数为实际校验输入与结果事实；按内容关联模型操作，歧义保留候选而不猜测。"""
    ctx = get_trace_ctx()
    if ctx is None:
        return
    try:
        fingerprint = value_fingerprint(raw)
        sources = _matching_structured_responses(ctx, schema=details.get("schema"), fingerprint=fingerprint)
        candidates = [event["payload"]["operation_id"] for event in sources]
        conversions = []
        for serialized in ctx.evidence_observations:
            event = json.loads(serialized)
            payload = event["payload"]
            if (event["stage"] == "structured_conversion"
                    and payload.get("target_schema", payload.get("schema")) == details.get("schema")
                    and payload.get("output_fingerprint") == fingerprint):
                conversions.append(payload)
                candidates.extend(payload["matching_operation_ids"])
        # 1. 【模型观察】【转换关联】同一操作的直连与映射不重复计数；缺源或多源映射不能变成唯一来源
        candidates = list(dict.fromkeys(candidates))
        ambiguous = any(item.get("operation_id") is None for item in conversions)
        record_evidence_observation("structured_business_validation", {
            "actor": "product", **details, "input_fingerprint": fingerprint,
            "operation_id": candidates[0] if len(candidates) == 1 and not ambiguous else None,
            "matching_operation_ids": candidates,
            "conversion_ids": [item["conversion_id"] for item in conversions],
        })
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        record_evidence_capture_error("structured_business_validation", exc)
