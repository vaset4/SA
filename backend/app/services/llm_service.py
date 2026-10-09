"""LLM 服务模块。"""

from __future__ import annotations

# 这是项目自研的 OpenAI-compatible LLM 客户端封装，不直接依赖官方 OpenAI SDK。
# 它负责统一 chat completions 调用、缓存、重试，以及结构化输出适配。
#
# 主调用链可以简单理解为：
#   get_llm() -> OpenAICompatibleLLM.invoke()/ainvoke()
#   -> 底层 `/chat/completions` 请求
#   -> 响应标准化 / 缓存
#   -> 上层 RAG、评估器、LightRAG 回调继续消费

import hashlib
import html
import json
import logging
import re
import threading
import time
import unicodedata
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, Generator, TypeVar, cast
from urllib.parse import urlparse
from uuid import uuid4

import requests
from langchain_core.messages import BaseMessage, convert_to_messages, convert_to_openai_messages
from pydantic import BaseModel, ValidationError

from app.services.cache import get_cache
from app.services.llm_structured_diagnostics import describe_validation_locations
from app.services.provider_request_queue import (
    check_provider_request_active,
    get_provider_queue,
    provider_request_timeout,
    retry_after_seconds,
    run_provider_thread,
    wait_for_provider_retry,
)
from app.services.llm_observation import (
    bind_structured_call,
    http_error_facts,
    model_attempt_identity,
    record_structured_path,
    record_structured_validation,
    redact_error,
    structured_operation,
    structured_identity,
    value_fingerprint,
)

if TYPE_CHECKING:
    from app.middleware.trace_context import ModelCallRecord


# 进程级单例。主图、RAG、LightRAG 都通过它共享同一组配置与缓存策略。
_llm_instance = None
_http_identity: ContextVar[dict] = ContextVar("llm_http_identity", default={})
logger = logging.getLogger(__name__)
JSON_LIKE_PREFIXES = ("{", "[")
PROVIDER_ERROR_DISPLAY_CHARS = 240
PROVIDER_PROTOCOL_PREVIEW_CHARS = 240
REQUEST_OUTPUT_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")
CHAT_COMPLETIONS_PATH = "/chat/completions"
EMPTY_RESPONSE_ERROR_PREFIX = "LLM 返回空响应体"
EMPTY_RESPONSE_MAX_RETRIES = 5  # 增加到 5 次重试
EMPTY_RESPONSE_ZERO_LENGTH_MAX_RETRIES = 1  # 上游明确返回 0 字节时快速失败
REQUEST_EXCEPTION_MAX_RETRIES = 5  # 增加到 5 次重试
# NVIDIA 免费上游实测：503 高发且错误约 1.6s 快速返回，主机节流已保证请求间隔，
# 指数退避（1→2→4→8→16s）只会空耗预算；改用固定短退避，靠节流兜底真实节奏
HTTP_RETRY_BACKOFF_SECONDS = 1.5
RETRYABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    # 【模型服务】【流式重试】分块协议断流不继承 ConnectionError，沿用现有连接重试额度
    requests.exceptions.ChunkedEncodingError,
)
RETRYABLE_LLM_EXCEPTIONS = RETRYABLE_REQUEST_EXCEPTIONS + (requests.exceptions.HTTPError, ValueError)
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 520, 522, 523, 524})
# 结构回修沿用既有一次额度，版本参与缓存与观察身份
STRUCTURED_OUTPUT_CONTRACT_VERSION = "strict-json-v2"
STRUCTURED_OUTPUT_PROMPT_VERSION = "schema-prompt-v2"
STRUCTURED_OUTPUT_REPAIR_MAX = 1
STRUCTURED_OUTPUT_ERROR_CODE = "structured_output_failed"
NATIVE_FORMAT_REJECT_MARKERS = (
    "response_format",
    "json_schema",
    "json_object",
    "unsupported",
    "not supported",
    "invalid_request",
)
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class LLMResponse:
    """标准化的 LLM 返回对象：文本内容 + 原始响应载荷。"""
    content: str
    raw: dict[str, Any]


class ProviderCooldownError(requests.exceptions.HTTPError):
    """已发生429且共享冷却写入失败；保留HTTP响应并明确禁止继续发送。"""


class ProviderHTTPError(requests.exceptions.HTTPError):
    """【模型服务】【供应商失败】沿用HTTP错误协议，分别携带安全用户摘要与技术上下文。"""

    def __init__(
        self, technical_message: str, *, response: requests.Response | None,
        user_summary: str, provider_error_type: str | int | None = None,
        provider_error_code: str | int | None = None,
    ) -> None:
        """参数为脱敏技术消息、原响应、用户摘要及实际错误标识；返回保留response/request的HTTP异常。"""
        self.technical_message = technical_message
        self.user_summary = user_summary
        self.provider_error_type = provider_error_type
        self.provider_error_code = provider_error_code
        super().__init__(technical_message, response=response)


@dataclass(frozen=True)
class _ResponseFacts:
    """【模型服务】【响应事实】一次解析完成或失败时的快照，不作为下层可变输出参数。"""

    response_text: str = ""
    actual_response_mode: str | None = None
    finish_reason: Any = None
    stream_done_received: bool | None = None
    usage: Any = None
    protocol_error: dict[str, Any] | None = None

    @classmethod
    def from_sse(cls, parts: list[str], *, usage: Any, finish_reason: Any,
                 finished: bool, failure: BaseException | None) -> _ResponseFacts:
        """参数为本次局部片段和已见终止事实；返回一次拼接的快照，不修改输入或异常。"""
        return cls(response_text="".join(parts), actual_response_mode="sse", usage=usage,
                   finish_reason=finish_reason, stream_done_received=finished,
                   protocol_error=failure.diagnostic if isinstance(failure, ProviderProtocolError) else None)


class ProviderProtocolError(RuntimeError):
    """【模型服务】【协议失败】仅表示SSE data不是合法JSON对象，不属于任何重试类型。"""

    def __init__(self, code: str, *, line_number: int, data_text: str,
                 json_error_offset: int | None = None) -> None:
        """参数为机器码、流行号、完整脱敏data及可选JSON位置；返回仅含安全摘要的异常。"""
        self.user_summary = "模型服务返回的数据损坏，本次响应未能完整接收"
        self.diagnostic = {"code": code, "line_number": line_number,
                           "data_text": data_text,
                           "data_preview": _preview_text(data_text, limit=PROVIDER_PROTOCOL_PREVIEW_CHARS),
                           "json_error_offset": json_error_offset}
        super().__init__(self.user_summary)


def provider_failure_message(exc: Exception, *, include_type: bool = False) -> str:
    """参数为原异常及其他错误是否显示类型；返回供应商安全摘要，其余错误保持原表达。"""
    # 1. 【模型服务】【失败摘要】研究入口共用客户端摘要，不重新解释供应商协议
    if isinstance(exc, (ProviderHTTPError, ProviderProtocolError)):
        return exc.user_summary
    return f"{type(exc).__name__}: {exc}" if include_type else str(exc)


def _first_response_choice(data: Any) -> dict[str, Any]:
    """参数为已解析的JSON信封或SSE对象；返回原第一选择，缺失时不构造结束原因。"""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    return choices[0]
class StructuredOutputError(ValueError):
    """
    结构化输出最终失败的稳定错误类型。

    供 planner 等业务层识别 code，避免把整段校验堆栈当唯一用户文案。
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        user_summary: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        # 1. 保存机器可读码与技术细节
        self.code = str(code or STRUCTURED_OUTPUT_ERROR_CODE)
        self.technical_message = str(message or "structured output failed")
        # 2. 用户摘要默认取技术文案；调用方可覆盖为更短中文
        self.user_summary = str(user_summary or self.technical_message)
        self.details = dict(details or {})
        super().__init__(self.technical_message)


class StructuredOutputLLM(Generic[ModelT]):
    """组合标准消息、严格 Pydantic 校验和既有一次回修，不修改模型字段含义。"""

    def __init__(self, client: "OpenAICompatibleLLM", schema: type[ModelT]) -> None:
        self._client = client
        self._schema = schema

    def invoke(self, messages: Any) -> ModelT:
        """参数为标准聊天消息；返回严格模型，格式失败仅使用现有回修额度。"""
        with structured_operation(self._schema.__name__, secret=getattr(self._client, "api_key", "")) as obs:
            # 1. 【模型服务】【合同身份】原生关闭时同样隔离 schema 与提示版本
            obs["failure_stage"] = "schema_configuration"
            obs.update({
                "contract_version": STRUCTURED_OUTPUT_CONTRACT_VERSION,
                "prompt_version": STRUCTURED_OUTPUT_PROMPT_VERSION,
                "schema_fingerprint": value_fingerprint(self._schema.model_json_schema()),
            })
            return self._invoke_structured(messages, obs)

    def _invoke_structured(self, messages: Any, obs: dict[str, Any]) -> ModelT:
        """参数为消息和操作观察；返回完整响应的严格校验结果，不刷新坏缓存增加尝试。"""
        obs["failure_stage"] = "message_boundary"
        messages = _structured_messages(messages, self._schema)
        obs["prompt_fingerprint"] = value_fingerprint(messages)
        response_format = None
        if _structured_output_native_enabled():
            obs["failure_stage"] = "schema_configuration"
            response_format = _build_native_response_format(self._schema)
            obs["native_mode"] = "attempted"
        response = self._invoke_text(messages, response_format=response_format, use_cache=True, obs=obs)
        obs["failure_stage"] = "program_conversion"
        try:
            return parse_structured_response(response.content, self._schema)
        except ValidationError as first_error:
            obs["failure_stage"] = "model_json" if any(item["type"] == "json_invalid" for item in first_error.errors()) else "model_schema"
            if response.raw.get("cache_hit"):
                self._client.invalidate_cached_response(messages, response_format=response_format)
                record_structured_path(obs, "bad_cache_invalidated")
            return self._repair_once(messages, response, first_error, obs=obs, response_format=response_format)

    async def ainvoke(self, messages: Any) -> ModelT:
        return await run_provider_thread(self.invoke, messages)

    def _invoke_text(
        self,
        messages: Any,
        *,
        response_format: dict[str, Any] | None,
        use_cache: bool,
        obs: dict[str, Any],
    ) -> LLMResponse:
        """参数为消息、原生格式、缓存开关及观察；返回响应，供应商拒绝不改变配置。"""
        obs["failure_stage"] = "provider_call"
        try:
            response = self._client.invoke(messages, use_cache=use_cache, response_format=response_format)
        except requests.exceptions.HTTPError as error:
            if response_format is not None and _is_native_format_rejected(error):
                obs["native_mode"] = "rejected"
                record_structured_path(obs, "native_rejected")
            raise
        if response_format is not None and obs.get("native_mode") == "attempted":
            obs["native_mode"] = "accepted"
        obs["cache_hit"] = bool(response.raw.get("cache_hit"))
        return response

    def _repair_once(
        self,
        messages: Any,
        response: LLMResponse,
        first_error: ValidationError,
        *,
        obs: dict[str, Any],
        response_format: dict[str, Any] | None,
    ) -> ModelT:
        """
        校验失败后最多回修一次；耗尽则抛 StructuredOutputError。

        参数:
            messages: 原始用户消息
            response: 最近一次模型输出
            first_error: 首次解析错误
            obs: 可观测字典
            response_format: 仅在 native accepted 时沿用
        返回:
            校验通过的模型实例
        """
        max_repairs = _structured_output_repair_max()
        if max_repairs < 1:
            raise self._to_structured_error(first_error, response.content, obs) from first_error
        # 1. 【模型服务】【格式回修】完整上次响应及字段错误共用一次既有额度
        repair_messages = _build_structured_repair_messages(
            messages,
            response.content,
            first_error,
            schema=self._schema,
        )
        obs["repair_attempted"] = True
        obs["repair_count"] = 1
        obs["phase"] = "repair"
        record_structured_path(obs, "schema_repair_started")
        repair_response = self._invoke_text(
            repair_messages,
            response_format=response_format,
            use_cache=False,
            obs=obs,
        )
        obs["failure_stage"] = "program_conversion"
        try:
            result = parse_structured_response(repair_response.content, self._schema)
            obs["repair_succeeded"] = True
            logger.info(
                "【模型服务】【格式回修成功】schema=%s repair_count=%s native_mode=%s",
                self._schema.__name__,
                obs["repair_count"],
                obs.get("native_mode"),
            )
            return result
        except ValidationError as second_error:
            obs["failure_stage"] = "model_json" if any(item["type"] == "json_invalid" for item in second_error.errors()) else "model_schema"
            raise self._to_structured_error(second_error, repair_response.content, obs) from second_error

    def _to_structured_error(
        self,
        error: Exception,
        content: str,
        obs: dict[str, Any],
    ) -> StructuredOutputError:
        """
        把解析/校验异常包装为带稳定 code 的 StructuredOutputError。

        参数:
            error: 原始异常
            content: 最近一次模型文本
            obs: 可观测细节
        返回:
            StructuredOutputError
        """
        secret = getattr(self._client, "api_key", "")
        preview = redact_error(_preview_text(content), secret=secret)
        technical = (
            f"LLM 结构化输出失败: schema={self._schema.__name__}; error={redact_error(error, secret=secret)}; "
            f"response_preview={preview}"
        )
        user_summary = _structured_output_user_summary(self._schema.__name__, error)
        details = dict(obs)
        details["error"] = redact_error(error, secret=secret)
        details["response_preview"] = preview
        logger.error(
            "【模型服务】【结构合同失败】schema=%s native_mode=%s repair_count=%s error=%s",
            self._schema.__name__,
            obs.get("native_mode"),
            obs.get("repair_count"),
            redact_error(error, secret=secret),
        )
        return StructuredOutputError(
            STRUCTURED_OUTPUT_ERROR_CODE,
            technical,
            user_summary=user_summary,
            details=details,
        )


class OpenAICompatibleLLM:
    """面向 OpenAI-compatible `/chat/completions` 接口的轻量客户端。"""
    _throttle_lock = threading.Lock()
    _host_next_request_at: dict[str, float] = {}

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        timeout: int,
        min_request_interval_seconds: float = 0.0,
        request_exception_max_retries: int = REQUEST_EXCEPTION_MAX_RETRIES,
        empty_response_max_retries: int = EMPTY_RESPONSE_MAX_RETRIES,
        requests_per_minute_limit: int = 0,
        retry_backoff_seconds: float = HTTP_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = self._normalize_base_url(base_url)
        self.temperature = temperature
        self.timeout = timeout
        # 有效请求间隔取"显式最小间隔"与"由 RPM 反推的间隔(60/rpm)"的较大值，
        # 让 NVIDIA 免费上游 RPM=30 这类额度既不被浪费也不被突破
        explicit_interval = max(float(min_request_interval_seconds or 0.0), 0.0)
        rpm = max(int(requests_per_minute_limit or 0), 0)
        rpm_interval = (60.0 / rpm) if rpm > 0 else 0.0
        self.min_request_interval_seconds = max(explicit_interval, rpm_interval)
        self.request_exception_max_retries = max(int(request_exception_max_retries), 0)
        self.empty_response_max_retries = max(int(empty_response_max_retries), 0)
        # 单次重试等待：不稳定但快速失败的上游取短值(靠多试扛抖动)，RPM 低的上游取长值
        self.retry_backoff_seconds = max(float(retry_backoff_seconds or 0.0), 0.0)

    def invoke(
        self,
        messages: Any,
        *,
        use_cache: bool = True,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        同步调用 chat completions，内含消息归一化、缓存和重试。

        参数:
            messages: 聊天消息
            use_cache: 是否读写响应缓存
            response_format: 可选上游强制格式（json_schema/json_object）
        返回:
            LLMResponse
        """
        # 1. 组装请求体；response_format 仅在调用方显式传入时附上
        normalized_messages = self._normalize_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
            "temperature": self.temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # 2. 中转站对非流式请求排慢车道（首字可达分钟级），默认改 SSE 流式；开关关闭即回退旧行为
        use_stream = self._streaming_enabled()
        if use_stream:
            payload["stream"] = True
        prompt_key = self._cache_key(normalized_messages, response_format=response_format)

        call_record = self._begin_trace_capture(normalized_messages)

        if use_cache:
            cached_content = get_cache().get_llm_response(prompt_key)
            if cached_content is not None:
                if call_record:
                    call_record.cache_hit = True
                    call_record.finalize(output=cached_content, secret=self.api_key)
                return LLMResponse(content=cached_content, raw={"cache_hit": True})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return self._invoke_with_retries(headers, payload, prompt_key=prompt_key,
                                         use_stream=use_stream, call_record=call_record)

    def _invoke_with_retries(self, headers: dict, payload: dict, *, prompt_key: str,
                             use_stream: bool, call_record: ModelCallRecord | None) -> LLMResponse:
        """参数为请求、缓存键及观察；按既有共享次数重试，每个HTTP尝试重新排队。"""
        max_retries = max(
            self.empty_response_max_retries,
            self.request_exception_max_retries,
        )
        retry_count = 0
        call_id = call_record.call_id if call_record else str(uuid4())
        try:
            for attempt in range(max_retries + 1):
                try:
                    llm_response = self._invoke_http_attempt(
                        headers, payload, use_stream=use_stream, call_record=call_record,
                        request_identity={"call_id": call_id, "attempt": attempt + 1},
                    )
                    get_cache().set_llm_response(prompt_key, llm_response.content)
                    if call_record:
                        call_record.retry_count = retry_count
                        call_record.finalize(output=llm_response.content, secret=self.api_key)
                    return llm_response
                except RETRYABLE_LLM_EXCEPTIONS as error:
                    retry_count += 1
                    wait_for_provider_retry(self._retry_or_raise(error, attempt))
        except Exception as exc:
            if call_record and call_record.error is None:
                call_record.retry_count = retry_count
                call_record.finalize(error=exc, secret=self.api_key)
            raise
        final_error = RuntimeError("LLM 调用异常结束：未获得可用响应")
        if call_record:
            call_record.retry_count = retry_count
            call_record.finalize(error=final_error, secret=self.api_key)
        raise final_error

    def _retry_or_raise(self, error: Exception, attempt: int) -> float:
        """参数为失败和零起始尝试号；返回原预算允许的退避，否则抛出真实错误。"""
        if isinstance(error, requests.exceptions.HTTPError):
            category = "HTTP异常"
            if not self._should_retry_http_error(error) or attempt >= self.request_exception_max_retries:
                raise error
            delay = self._retry_delay_for_http_error(error, attempt)
        elif isinstance(error, RETRYABLE_REQUEST_EXCEPTIONS):
            category = "连接重试"
            if attempt >= self.request_exception_max_retries:
                raise error
            delay = self._short_retry_delay()
        elif isinstance(error, ValueError):
            category = "空响应"
            if not self._should_retry_empty_response(error, attempt):
                raise error
            delay = 2 ** attempt
        else:
            raise error
        logger.info(
            "【模型服务】【HTTP重试】%s attempt=%s delay=%s model=%s host=%s error=%s",
            category, attempt + 1, delay, self.model, self._base_url_host(), redact_error(error, secret=self.api_key),
        )
        return delay

    async def ainvoke(
        self,
        messages: Any,
        *,
        use_cache: bool = True,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """异步包装版本，当前通过 `asyncio.to_thread` 复用同步实现。"""
        return await run_provider_thread(
            self.invoke,
            messages,
            use_cache=use_cache,
            response_format=response_format,
        )

    def with_structured_output(self, schema: type[ModelT]) -> StructuredOutputLLM[ModelT]:
        """返回结构化输出包装器，供上层直接拿 Pydantic 模型。"""
        return StructuredOutputLLM(self, schema)

    def _invoke_http_attempt(
        self, headers: dict[str, str], payload: dict[str, Any], *,
        use_stream: bool, call_record: ModelCallRecord | None,
        request_identity: dict | None = None,
    ) -> LLMResponse:
        """【LLM】【实际用量】参数为原请求、传输模式及逻辑调用；返回完整响应，记录每次传输结果及provider原始usage。"""
        # 1. 【模型服务】【发送准入】排队不计 HTTP；每次重试和结构回修重新加入队尾
        queue = get_provider_queue()
        if not queue.policy["enabled"]:
            self._throttle_request_if_needed()
        identity = request_identity or {"call_id": call_record.call_id if call_record else str(uuid4()),
                                       "attempt": call_record.http_attempt_count + 1 if call_record else 1}
        ticket = queue.acquire(identity)
        token = _http_identity.set(identity)
        error = None
        try:
            queue.mark_started(ticket)
            return self._invoke_admitted_attempt(headers, payload, use_stream=use_stream, call_record=call_record)
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                queue.release(ticket, error=error)
            finally:
                _http_identity.reset(token)

    def _invoke_admitted_attempt(
        self, headers: dict[str, str], payload: dict[str, Any], *,
        use_stream: bool, call_record: ModelCallRecord | None,
    ) -> LLMResponse:
        """参数为已获准入的原请求和调用记录；返回公开响应，同次成功/失败只记录一份事实。"""
        # 1. 【模型服务】【实际尝试】缓存不经过此边界，事实只属于当前HTTP请求
        if call_record is not None:
            call_record.http_attempt_count += 1
        identity = model_attempt_identity(
            self.model, self._chat_completions_url(), call_record=call_record, streaming=use_stream,
        )
        request_identity = _http_identity.get()
        identity = {**identity, "call_id": identity["call_id"] or request_identity.get("call_id"),
                    "http_attempt": identity["http_attempt"] or request_identity.get("attempt")}
        started = time.perf_counter()
        response = None
        facts = _ResponseFacts()
        failure: dict[str, Any] = {}
        boundary = "transport"
        try:
            response = (self._post_chat_completions_stream(headers, payload) if use_stream
                        else self._post_chat_completions(headers, payload))
            boundary = "api_envelope"
            try:
                result, facts = self._build_response(response, payload)
            finally:
                response.close()
            return result
        except BaseException as exc:
            # 2. 【模型服务】【失败事实】原异常身份与重试分类不变，只读取解析器冻结的当前快照
            facts = getattr(exc, "_llm_response_facts", facts)
            if response is None:
                response = getattr(exc, "response", None)
            failure = http_error_facts(
                cast(Exception, exc), response=response,
                boundary="stream_protocol" if isinstance(exc, ProviderProtocolError) else boundary,
                secret=self.api_key,
            )
            raise
        finally:
            # 3. 【模型服务】【响应留存】正文与用量共用同一快照，失败前实见事实不丢失
            self._record_response_observations(identity, facts, payload,
                response=response, failure=failure, duration_seconds=time.perf_counter() - started)

    def _record_response_observations(
        self, identity: dict[str, Any], facts: _ResponseFacts, payload: dict[str, Any], *,
        response: requests.Response | None, failure: dict[str, Any], duration_seconds: float,
    ) -> None:
        """参数为同次身份、快照、实际请求及结果；向既有正文/用量观察写入相同元数据，无返回值。"""
        from app.middleware.trace_context import record_evidence_observation

        usage = facts.usage if isinstance(facts.usage, dict) else None
        status = getattr(response, "status_code", None)
        content_type = getattr(response, "headers", {}).get("Content-Type")
        content_type = content_type if isinstance(content_type, str) else None
        # 1. 【模型服务】【真实终止】未提供的结束原因和未进入的解析模式保留未知
        outcome = {
            **identity, "duration_seconds": duration_seconds,
            "status": "failed" if failure else "completed",
            "http_status": status if isinstance(status, int) else None,
            "requested_stream": bool(payload.get("stream", False)),
            "actual_response_mode": facts.actual_response_mode,
            "response_content_type": content_type,
            "finish_reason": facts.finish_reason,
            "stream_done_received": facts.stream_done_received,
            "requested_output_limits": {key: {"present": key in payload, "value": payload.get(key)}
                                        for key in REQUEST_OUTPUT_LIMIT_FIELDS},
            "assistant_content_present": bool(facts.response_text),
            "protocol_status": "failed" if failure or not facts.response_text else "completed",
            "usage_status": "measured" if usage else "missing", "usage": usage,
            "protocol_error": facts.protocol_error,
            **({"error_boundary": "api_envelope", "error_type": "EmptyAssistantContent"}
               if not failure and not facts.response_text else {}),
            **failure,
        }
        # 2. 【模型服务】【响应留存】小型事实随正文穿过完整trace与size_cap，不依赖debug副本
        record_evidence_observation("model_response_content", {
            **outcome, "response_text": facts.response_text.replace(self.api_key, "[REDACTED]")
            if self.api_key else facts.response_text,
            "content_complete": not bool(failure), "response_source": "model",
        })
        record_evidence_observation("model_response_usage", outcome)
    def _post_chat_completions(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> requests.Response:
        """向配置的 OpenAI-compatible 基地址发起 `/chat/completions` 请求。"""
        request_url = self._chat_completions_url()
        response = requests.post(
            request_url,
            headers=headers,
            json=payload,
            timeout=provider_request_timeout(self.timeout, check_active=False),
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            try:
                decorated = self._decorate_http_error(error, request_url)
            finally:
                response.close()
            raise decorated from error
        return response

    def _post_chat_completions_stream(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> requests.Response:
        """以 `stream=True` 发起流式 `/chat/completions` 请求，响应体按 SSE 逐行读取。"""
        request_url = self._chat_completions_url()
        response = requests.post(
            request_url,
            headers=headers,
            json=payload,
            timeout=provider_request_timeout(self.timeout, check_active=False),
            stream=True,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            try:
                decorated = self._decorate_http_error(error, request_url)
            finally:
                response.close()
            raise decorated from error
        return response

    def _streaming_enabled(self) -> bool:
        """读取是否默认走流式传输；开关只影响传输层，配置缺失时回落常量。"""
        from app.config import get_settings

        return bool(getattr(get_settings(), "llm_streaming_enabled", True))

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}{CHAT_COMPLETIONS_PATH}"

    def _normalize_base_url(self, base_url: str) -> str:
        normalized = str(base_url or "").strip().rstrip("/")
        if normalized.endswith(CHAT_COMPLETIONS_PATH):
            normalized = normalized[: -len(CHAT_COMPLETIONS_PATH)]
        return normalized.rstrip("/")

    def _decorate_http_error(
        self,
        error: requests.exceptions.HTTPError,
        request_url: str,
    ) -> requests.exceptions.HTTPError:
        """参数为原HTTP异常和请求地址；返回安全摘要与原响应，状态、冷却和异常链由原入口保留。"""
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if response is not None and status_code == 429:
            # 1. 【模型服务】【共享冷却】最后一次失败也影响同组后续调用，不新增尝试额度
            try:
                get_provider_queue().cooldown(response.headers.get("Retry-After"), _http_identity.get())
            except Exception as coordination_error:
                from app.middleware.trace_context import record_evidence_observation

                record_evidence_observation("provider_cooldown_failed", {
                    "http_status": 429, "error_type": type(coordination_error).__name__,
                })
                logger.error("【模型服务】【共享冷却】429后协调写入失败", exc_info=True)
                failure = ProviderCooldownError(redact_error(error, secret=self.api_key), response=response)
                failure.add_note("共享冷却写入失败，禁止继续重试")
                return failure
        # 2. 【模型服务】【供应商失败】只认标准error.message，404本身不证明地址或模型配置错误
        details = _provider_error_details(response, secret=self.api_key)
        user_summary = _provider_failure_summary(status_code, details.get("message", ""))
        body_preview = self._response_body_preview(response)
        hint = ""
        if status_code == 403:
            hint = (
                "LLM provider rejected the request. Check API key, model allowlist, provider "
                "permission, or upstream proxy policy."
            )
        elif status_code == 401:
            hint = "LLM provider authentication failed. Check API key and Authorization header."
        message = (
            f"{user_summary}; {error}; {hint} requested_url={request_url}; model={self.model}; "
            f"provider_error_type={details.get('type')}; provider_error_code={details.get('code')}; "
            f"host={self._base_url_host()}; body_preview={body_preview}"
        ).strip()
        message = redact_error(message, secret=self.api_key, limit=None)
        logger.error("【模型服务】【供应商失败】%s", message)
        return ProviderHTTPError(
            message, response=response, user_summary=user_summary,
            provider_error_type=details.get("type"), provider_error_code=details.get("code"),
        )

    def _should_retry_http_error(self, error: requests.exceptions.HTTPError) -> bool:
        if isinstance(error, ProviderCooldownError):
            return False
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        return bool(status_code in RETRYABLE_HTTP_STATUS_CODES)

    def _retry_delay_for_http_error(
        self,
        error: requests.exceptions.HTTPError,
        attempt: int,
    ) -> float:
        response = getattr(error, "response", None)
        retry_after = ""
        if response is not None:
            retry_after = str(response.headers.get("Retry-After", "")).strip()
        if getattr(response, "status_code", None) == 429:
            queue = get_provider_queue()
            if queue.policy["enabled"]:
                return 0.0
            parsed = retry_after_seconds(retry_after)
            return queue.policy["cooldown_seconds"] if parsed is None else parsed
        # 1. 无 Retry-After 时用预设的固定短退避：NVIDIA 免费上游这类"不稳定但错误快返回"
        #    的供应商适合短等多试，稳定但低 RPM 的供应商则应把退避调长，故按供应商预设走
        base_delay = self.retry_backoff_seconds
        if retry_after:
            try:
                # 2. 上游显式给了 Retry-After 才尊重它，但对其封顶：Cloudflare 524/供应商
                #    503 常返回 120s 这类超大值，直接照睡会在 RAG 总预算内空转烧穿
                base_delay = max(float(retry_after), 0.0)
            except ValueError:
                pass
        return min(base_delay, self._retry_delay_cap_seconds())

    def _short_retry_delay(self) -> float:
        """连接异常/无状态码错误的固定短退避（秒），取预设退避与 sleep 硬上限的较小值。"""
        return min(self.retry_backoff_seconds, self._retry_delay_cap_seconds())

    def _retry_delay_cap_seconds(self) -> float:
        """读取单次重试 sleep 的硬上限（秒），配置缺失时回退到保守默认值。"""
        from app.config import get_settings

        return float(getattr(get_settings(), "llm_retry_delay_cap_seconds", 15.0) or 15.0)

    def _build_response(
        self, response: requests.Response, payload: dict[str, Any] | None = None,
    ) -> tuple[LLMResponse, _ResponseFacts]:
        """参数为实际HTTP响应和原请求；返回公开响应及私有快照，解析失败附带原异常抛出。"""
        # 1. 【模型服务】【实际模式】按响应头选择解析器，请求stream开关不代表实际传输
        content_type = str(response.headers.get("Content-Type") or "")
        if "text/event-stream" in content_type:
            return self._build_response_from_sse(response)
        facts = _ResponseFacts(actual_response_mode="json")
        try:
            return self._build_json_response(response, payload)
        except BaseException as exc:
            # 2. 【模型服务】【信封失败】未能得到JSON对象时仍保存已选择的解析模式，不伪造正文
            if not hasattr(exc, "_llm_response_facts"):
                setattr(exc, "_llm_response_facts", facts)
            raise

    def _build_json_response(
        self, response: requests.Response, payload: dict[str, Any] | None,
    ) -> tuple[LLMResponse, _ResponseFacts]:
        """参数为非SSE响应与原请求；返回原JSON/文本结果和实见事实，不增加解析重试。"""
        raw_text = self._extract_raw_text(response)
        if not raw_text:
            error_details = self._response_debug_details(response)
            request_info = {}
            if payload:
                messages = payload.get("messages", [])
                total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
                request_info = {"message_count": len(messages), "total_prompt_chars": total_chars,
                                "temperature": payload.get("temperature")}
            logger.error("【模型服务】【空响应】model=%s host=%s request=%s response=%s",
                         self.model, self._base_url_host(), request_info, error_details)
            raise ValueError(
                f"{EMPTY_RESPONSE_ERROR_PREFIX}: status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type', '')}, "
                f"content_length={response.headers.get('Content-Length', '')}"
            )
        try:
            data = response.json()
        except ValueError as error:
            if raw_text.startswith(JSON_LIKE_PREFIXES):
                preview = raw_text[:300]
                raise ValueError(f"LLM JSON 解析失败: {error}; body={preview}") from error
            result = LLMResponse(content=raw_text, raw={"raw_text": raw_text,
                "status_code": response.status_code, "content_type": response.headers.get("Content-Type", "")})
            return result, _ResponseFacts(response_text=raw_text, actual_response_mode="text")
        # 1. 【模型服务】【响应事实】只消费原始第一选择与usage，不以缺省stop替代未知
        first = _first_response_choice(data)
        facts = _ResponseFacts(actual_response_mode="json", finish_reason=first.get("finish_reason"),
                               usage=data.get("usage") if isinstance(data, dict) else None)
        try:
            content = self._extract_content(data)
        except BaseException as exc:
            setattr(exc, "_llm_response_facts", facts)
            raise
        return LLMResponse(content=content, raw=data), replace(facts, response_text=content)

    def _build_response_from_sse(self, response: requests.Response) -> tuple[LLMResponse, _ResponseFacts]:
        """参数为实际SSE响应；返回聚合响应与快照，空流保持既有空响应错误和重试规则。"""
        facts = self._aggregate_sse_response(response)
        aggregated: dict[str, Any] = {"choices": [{"index": 0,
            "message": {"role": "assistant", "content": facts.response_text}, "finish_reason": facts.finish_reason}]}
        if facts.usage is not None:
            aggregated["usage"] = facts.usage
        content = self._extract_content(aggregated)
        if not content:
            # 1. 【模型服务】【空流事实】DONE只证明传输终止，空正文仍保留原错误分类
            error = ValueError(f"{EMPTY_RESPONSE_ERROR_PREFIX}: stream 已正常结束但未聚合到任何内容")
            setattr(error, "_llm_response_facts", facts)
            raise error
        return LLMResponse(content=content, raw=aggregated), facts

    def _aggregate_sse_response(self, response: requests.Response) -> _ResponseFacts:
        """参数为SSE响应；返回结束快照，失败附带已见事实并原样抛出异常。"""
        # 1. 【模型服务】【流式采集】片段只在本函数局部累积，结束时一次拼接，不跨attempt共享
        content_parts: list[str] = []
        content_chars = 0
        usage: Any = None
        finish_reason: Any = None
        finished = False
        failure: BaseException | None = None
        try:
            for line_number, line in enumerate(self._iter_sse_data_lines(response), start=1):
                if not line.startswith("data:"):
                    continue
                data_text = line[len("data:"):].strip()
                if data_text == "[DONE]":
                    finished = True
                    break
                chunk = self._decode_sse_chunk(data_text, line_number=line_number)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                first = _first_response_choice(chunk)
                if first.get("finish_reason") is not None:
                    finish_reason = first["finish_reason"]
                if chunk.get("error"):
                    # 2. 【模型服务】【供应商错误】已知error事件保持连接错误语义与原重试额度
                    raise requests.exceptions.ConnectionError(
                        f"LLM 流式上游错误事件: {_preview_text(json.dumps(chunk, ensure_ascii=False))}"
                    )
                delta = first.get("delta") or {}
                content = delta.get("content")
                if content is not None:
                    part = self._response_content(content)
                    content_parts.append(part)
                    content_chars += len(part)
            if not finished:
                raise requests.exceptions.ConnectionError(
                    "LLM 流式响应未以 [DONE] 正常结束（流中断）: "
                    f"partial_chars={content_chars}"
                )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # 3. 【模型服务】【结束快照】异常不包成新类型；只附带本次已收到的信息，不读取后续DONE
            facts = _ResponseFacts.from_sse(content_parts, finish_reason=finish_reason,
                finished=finished, usage=usage, failure=failure)
            if failure is not None:
                setattr(failure, "_llm_response_facts", facts)
        return facts

    def _decode_sse_chunk(self, data_text: str, *, line_number: int) -> dict[str, Any]:
        """参数为单条data文本与流行号；返回JSON对象，坏数据以安全协议异常立即终止。"""
        try:
            chunk = json.loads(data_text)
        except json.JSONDecodeError as exc:
            # 1. 【模型服务】【坏流数据】完整脱敏片段仅进入诊断，短预览不替代错误位置后部原文
            safe_data = redact_error(data_text, secret=self.api_key, limit=None)
            raise ProviderProtocolError("invalid_sse_json", line_number=line_number,
                data_text=safe_data, json_error_offset=exc.pos) from None
        if not isinstance(chunk, dict):
            safe_data = redact_error(data_text, secret=self.api_key, limit=None)
            raise ProviderProtocolError("non_object_sse_data", line_number=line_number, data_text=safe_data)
        return chunk

    def _iter_sse_data_lines(self, response: requests.Response) -> "Generator[str, None, None]":
        """
        按字节缓冲切行后再 UTF-8 解码，产出 SSE 数据行。

        不能用 iter_lines(decode_unicode=True)：它按网络分片独立解码，
        一个多字节中文字符被劈在两个分片上就变成乱码、甚至产出断行的
        非法 JSON（线上实测）。这里先在字节层拼出完整行，再整行 UTF-8 解码。

        参数:
            response: 已建立（stream=True）的响应对象
        返回:
            完整 SSE 行（已解码）的生成器；结尾不足一行的残留也会产出
        """
        buffer = b""
        for raw_chunk in response.iter_content(chunk_size=None):
            # 1. 【模型服务】【流式取消】持续心跳不能绕过运行取消和总期限
            check_provider_request_active()
            if not raw_chunk:
                continue
            buffer += raw_chunk
            while b"\n" in buffer:
                line_bytes, _, buffer = buffer.partition(b"\n")
                yield line_bytes.rstrip(b"\r").decode("utf-8", errors="replace")
        tail = buffer.rstrip(b"\r")
        if tail.strip():
            yield tail.decode("utf-8", errors="replace")

    def _extract_raw_text(self, response: requests.Response) -> str:
        text_value = (response.text or "").strip()
        if text_value:
            return text_value
        raw_bytes = response.content or b""
        if not raw_bytes:
            return ""
        return raw_bytes.decode("utf-8", errors="ignore").strip()

    def _is_empty_response_error(self, error: ValueError) -> bool:
        return str(error).startswith(EMPTY_RESPONSE_ERROR_PREFIX)

    def _should_retry_empty_response(
        self,
        error: ValueError,
        attempt: int,
    ) -> bool:
        if not self._is_empty_response_error(error):
            return False
        retry_limit = (
            EMPTY_RESPONSE_ZERO_LENGTH_MAX_RETRIES
            if "content_length=0" in str(error)
            else self.empty_response_max_retries
        )
        return attempt < retry_limit

    def _base_url_host(self) -> str:
        parsed = urlparse(self.base_url)
        return parsed.netloc or self.base_url

    def _throttle_request_if_needed(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        host = self._base_url_host()
        sleep_seconds = 0.0
        with self._throttle_lock:
            now = time.monotonic()
            scheduled_at = self._host_next_request_at.get(host, 0.0)
            if scheduled_at > now:
                sleep_seconds = scheduled_at - now
                next_at = scheduled_at + self.min_request_interval_seconds
            else:
                next_at = now + self.min_request_interval_seconds
            self._host_next_request_at[host] = next_at
        if sleep_seconds > 0:
            logger.info(
                "LLM request throttled: sleep=%.3fs model=%s host=%s",
                sleep_seconds,
                self.model,
                host,
            )
            wait_for_provider_retry(sleep_seconds)

    def _response_debug_details(self, response: requests.Response) -> dict[str, Any]:
        """参数为HTTP响应；返回状态与公开响应头，供空正文错误定位。"""
        headers: Mapping[str, str] = response.headers or {}
        return {
            "status": response.status_code,
            "content_type": headers.get("Content-Type", ""),
            "content_length": headers.get("Content-Length", ""),
            "server": headers.get("Server", ""),
            "x_rid": headers.get("X-Rid", ""),
            "x_server": headers.get("X-Server", ""),
        }

    def _response_body_preview(self, response: requests.Response | None) -> str:
        """参数为供应商原响应；返回先完整脱敏再截断的技术预览，避免截断凭证后无法匹配。"""
        if response is None:
            return ""
        safe_text = redact_error(self._extract_raw_text(response), secret=self.api_key, limit=None)
        return _preview_text(safe_text, limit=PROVIDER_ERROR_DISPLAY_CHARS)

    def _extract_content(self, data: Any) -> str:
        """
        从兼容厂商的不同响应格式中提取文本内容。

        项目默认期待 OpenAI `choices[0].message.content` 结构，但为了兼容不同网关/
        模型代理，也接受 `content/text/output_text` 这类较常见的变体。
        """
        if not isinstance(data, dict):
            raise ValueError(f"LLM 响应不是对象: {type(data).__name__}")

        choices = data.get("choices") or []
        if choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message") or {}
            return self._response_content(message.get("content", ""))

        if "content" in data:
            return self._response_content(data["content"])
        if "text" in data:
            return self._response_content(data["text"])
        if "output_text" in data:
            return self._response_content(data["output_text"])

        raise ValueError(f"LLM 响应缺少可识别内容字段: {data}")

    def _normalize_messages(self, messages: Any) -> list[dict[str, Any]]:
        """参数为文本或标准消息；返回库转换的 OpenAI 消息，拒绝未知内容对象。"""
        return _convert_messages(messages)

    def _response_content(self, content: Any) -> str:
        """参数为供应商可见正文；只读取已支持的文本形态，错误内容类型不强转成模型回答。"""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise TypeError(f"LLM 响应正文必须是文本，实际为 {type(content).__name__}")
        parts = []
        for item in content:
            text = item
            if isinstance(item, dict):
                text = item.get("text") if "text" in item else item.get("content")
            if not isinstance(text, str):
                raise TypeError("LLM 响应内容块缺少文本字段")
            parts.append(text)
        return "\n".join(parts)

    def _cache_key(
        self,
        normalized_messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """
        基于实例配置和消息内容生成稳定缓存键。

        这里把 `id(self)` 也纳入键空间，目的是让"旧实例上的坏缓存"不会在 reset 之后
        污染到新实例；代价是缓存只在当前进程实例生命周期内复用。
        response_format 写入键空间，避免 native/纯文本路径互相污染缓存。
        """
        cache_payload = {
            "instance": f"{id(self)}",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "messages": normalized_messages,
            "response_format": response_format,
            "structured_contract": {
                key: structured_identity().get(key) for key in
                ("contract_version", "prompt_version", "schema_fingerprint")
            },
        }
        serialized = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def invalidate_cached_response(
        self,
        messages: Any,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        """按消息内容删除对应缓存项，供结构化输出坏缓存自恢复。"""
        normalized_messages = self._normalize_messages(messages)
        prompt_key = self._cache_key(normalized_messages, response_format=response_format)
        get_cache().delete_llm_response(prompt_key)

    def _begin_trace_capture(self, normalized_messages: list[dict[str, str]]) -> "ModelCallRecord | None":
        """若当前请求有 trace context，开始一条 model_call 记录。"""
        from app.middleware.trace_context import get_trace_ctx
        ctx = get_trace_ctx()
        if ctx is None:
            return None
        record = ctx.begin_call()
        record.structured_operation_id = bind_structured_call(record.call_id)
        record.input_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "").replace(self.api_key, "[REDACTED]") if self.api_key else m.get("content", "")}
            for m in normalized_messages
        ]
        record.input_total_chars = sum(len(m.get("content", "")) for m in normalized_messages)
        return record


def _preview_text(content: str, limit: int = 200) -> str:
    compact = re.sub(r"\s+", " ", (content or "").strip())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _convert_messages(messages: Any) -> list[dict[str, Any]]:
    """参数为字符串或文本消息列表；返回标准角色及内容，不把未知对象转换成字符串。"""
    source = [messages] if isinstance(messages, str) else messages
    if not isinstance(source, list):
        raise TypeError(f"不支持的 messages 类型: {type(messages).__name__}")
    for message in source:
        content = message.get("content") if isinstance(message, dict) else (
            message.content if isinstance(message, BaseMessage) else message
        )
        if not isinstance(content, (str, list)):
            raise TypeError(f"不支持的消息正文类型: {type(content).__name__}")
        if isinstance(content, list) and any(
            not isinstance(block, str) and not (
                isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            ) for block in content
        ):
            raise TypeError("当前模型入口仅接受文本消息内容")
    return convert_to_openai_messages(convert_to_messages(source), text_format="string")


def build_structured_output_prompt(prompt: str, schema: type[ModelT]) -> str:
    """参数为业务提示和模型；返回同一模型生成的完整 JSON 合同，不另列字段白名单。"""
    contract = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        f"{prompt}\n\n"
        "Return ONLY one valid JSON object. Do not wrap JSON in markdown fences.\n"
        "Do not output explanations before or after JSON.\n"
        f"Contract version: {STRUCTURED_OUTPUT_PROMPT_VERSION}\n"
        f"Schema: {schema.__name__}\nJSON Schema:\n{contract}\n"
    )


def _structured_messages(messages: Any, schema: type[BaseModel]) -> list[dict[str, Any]]:
    """参数为业务消息及模型；统一追加完整文本合同，原生关闭时仍明确字段。"""
    converted = _convert_messages(messages)
    instructions = build_structured_output_prompt("", schema).strip()
    return [{"role": "system", "content": instructions}, *converted]


def parse_structured_response(content: str, schema: type[ModelT]) -> ModelT:
    """参数为完整模型文本和严格合同；返回原值模型，ValidationError 保留字段路径及类型。"""
    parsing: dict[str, Any] = {
        "validation_mode": "strict_json", "normalization_applied": False, "normalization_changes": [],
    }
    try:
        result = schema.model_validate_json(content, strict=True)
    except ValidationError as error:
        record_structured_validation(content, schema=schema.__name__, error=error, parsing=parsing)
        raise
    record_structured_validation(content, schema=schema.__name__, result=result, parsing=parsing)
    return result


def _structured_output_native_enabled() -> bool:
    """无参数；返回已配置的原生格式策略，配置加载失败直接暴露。"""
    from app.config import get_settings

    return get_settings().structured_output_native_default


def _structured_output_repair_max() -> int:
    """无参数；返回既有回修预算，缺省沿用一次，配置错误不静默跳过。"""
    from app.config import get_settings

    return max(int(getattr(get_settings(), "structured_output_repair_max", STRUCTURED_OUTPUT_REPAIR_MAX)), 0)


def _build_native_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """参数为 Pydantic 合同；返回配置使用的原生格式，构造错误直接暴露。"""
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False},
    }


def _is_native_format_rejected(error: Exception) -> bool:
    """
    判断异常是否表示上游不支持/拒绝 response_format。

    仅 4xx 且文案命中关键词时视为 native 拒绝；429/5xx 不算。

    参数:
        error: 底层请求异常
    返回:
        True 仅标识供应商拒绝已选格式，不授权额外调用
    """
    if not isinstance(error, requests.exceptions.HTTPError):
        return False
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None or not (400 <= int(status_code) < 500) or int(status_code) == 429:
        return False
    body = ""
    if response is not None:
        try:
            body = (response.text or "")[:500]
        except Exception:
            body = ""
    haystack = f"{error} {body}".lower()
    return any(marker in haystack for marker in NATIVE_FORMAT_REJECT_MARKERS)


def _build_structured_repair_messages(
    messages: Any,
    raw_content: str,
    error: ValidationError,
    *,
    schema: type[BaseModel],
) -> list[dict[str, Any]]:
    """参数为已含合同的消息、完整响应、原错误和当前模型；返回区分校验位置与 JSON 位置的回修指令。"""
    base = _convert_messages(messages)
    errors = error.errors(include_input=False, include_url=False, include_context=False)
    fields = json.dumps(errors, ensure_ascii=False)
    locations = describe_validation_locations(raw_content, schema.model_json_schema(mode="validation"), errors)
    repair_prompt = (
        "上一轮结构化输出未能通过校验，请按已有完整 JSON Schema 修正字段并重新输出唯一 JSON。\n"
        f"原始校验错误（loc 是校验位置，可能含联合分支标签，并非 JSON 键路径；type 是错误类型）:\n{fields}\n"
        "原 JSON 定位（union_branches 是合同确认的分支标签；json_pointer 为 null 表示无法确定字段位置；"
        "缺失字段的位置表示应填写处；declared_root_field_pointer 仅指出根合同声明的同名字段，不表示自动搬移）:\n"
        f"{json.dumps(locations, ensure_ascii=False)}"
    )
    return [*base, {"role": "assistant", "content": raw_content}, {"role": "user", "content": repair_prompt}]


def _structured_output_user_summary(schema_name: str, error: Exception) -> str:
    """
    生成面向用户的短中文摘要，避免整段 ValidationError 堆栈。

    参数:
        schema_name: schema 名称
        error: 原始错误
    返回:
        1～2 句中文摘要
    """
    text = str(error)
    if isinstance(error, ValidationError) and any(item["type"] == "json_invalid" for item in error.errors()):
        return f"{schema_name} 结构化输出不是合法 JSON，已停止后续规划"
    if "校验失败" in text or "validation" in text.lower():
        return f"{schema_name} 结构化输出字段不符合契约，已停止后续规划"
    return f"{schema_name} 结构化输出失败，已停止后续规划"


def _provider_error_details(response: requests.Response | None, *, secret: str) -> dict[str, Any]:
    """参数为原HTTP响应及配置凭证；返回标准错误中的安全message/type/code，不猜非结构化原因。"""
    if response is None:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {}
    detail = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(detail, dict):
        return {}
    # 1. 【模型服务】【供应商失败】先脱敏完整文本，再折叠控制字符并转义HTML/Markdown，供正文按文字显示
    result: dict[str, Any] = {}
    message = detail.get("message")
    if isinstance(message, str):
        safe_text = redact_error(message, secret=secret, limit=None)
        safe_text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in safe_text)
        safe_text = " ".join(safe_text.split())
        safe_text = html.escape(safe_text, quote=False)
        safe_text = re.sub(r"([\\`*_\[\]()!#|~])", r"\\\1", safe_text)
        result["message"] = _preview_text(safe_text, limit=PROVIDER_ERROR_DISPLAY_CHARS)
    # 2. 【模型服务】【供应商失败】type/code仅作诊断事实，不维护服务商专用关键词分类
    for key in ("type", "code"):
        value = detail.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            result[key] = redact_error(value, secret=secret) if isinstance(value, str) else value
    return result


def _provider_failure_summary(status_code: int | None, reason: str = "") -> str:
    """参数为真实HTTP状态和已处理的供应商说明；返回短用户摘要，无说明时明确原因未知。"""
    prefix = f"模型服务返回{status_code}" if status_code is not None else "模型服务请求失败"
    return f"{prefix}。服务商说明：{reason}" if reason else f"{prefix}，具体原因未明确"


def classify_llm_upstream_error(exc: Exception) -> tuple[int, str] | None:
    """参数为上游异常；返回本应用HTTP状态和用户原因，未知错误返回None，不改变重试分类。"""
    # 1. 【模型服务】【协议失败】502是本应用的映射，不改写供应商实际HTTP状态
    if isinstance(exc, ProviderProtocolError):
        return 502, exc.user_summary
    if isinstance(exc, requests.exceptions.Timeout):
        return 504, "上游模型服务响应超时，请稍后重试"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return 502, "无法连接上游模型服务，请检查网络或稍后重试"
    if isinstance(exc, requests.exceptions.HTTPError):
        return _classify_provider_http_error(exc)
    return None


def _classify_provider_http_error(exc: requests.exceptions.HTTPError) -> tuple[int, str] | None:
    """参数为真实HTTP异常；返回既有状态/摘要映射，未识别状态保持None。"""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    # 1. 【模型服务】【供应商失败】404先按真实状态处理，不从URL或供应商关键词改写原因
    if status_code == 404:
        summary = exc.user_summary if isinstance(exc, ProviderHTTPError) else _provider_failure_summary(404)
        return 502, summary
    message = str(exc)
    if status_code == 429 or "ResourceExhausted" in message or "rate" in message.lower():
        return 503, "上游模型服务当前繁忙（额度或并发已满），请稍后重试"
    if status_code == 524 or "timeout occurred" in message.lower():
        return 504, "上游模型服务处理超时，请简化问题或稍后重试"
    if status_code in (500, 502, 503):
        return 503, "上游模型服务暂时不可用，请稍后重试"
    if status_code == 401:
        return 502, "上游模型服务鉴权失败，请检查 API Key 配置"
    return None


def get_llm() -> OpenAICompatibleLLM:
    """
    返回全局 LLM 单例。

    单例会绑定当前配置中的 `llm_model_name / llm_api_key / llm_base_url`。
    如配置变更，需要调用 `reset_llm()` 后重新初始化。

    之所以做成进程级单例，是因为：
      - 调用方很多，没必要在每个请求里重复创建客户端
      - L1 LLM 缓存以这里为入口，单例更容易统一命中
      - `lightrag_service.py` 的异步回调也需要复用同一实例配置
    """
    global _llm_instance
    if _llm_instance is None:
        from app.config import get_settings

        settings = get_settings()
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY 为空，无法初始化 LLM 服务")
        if not settings.llm_base_url:
            raise ValueError("LLM_BASE_URL 为空，无法初始化 LLM 服务")
        _llm_instance = OpenAICompatibleLLM(
            model=settings.llm_model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
            min_request_interval_seconds=getattr(settings, "llm_min_request_interval_seconds", 0.0),
            request_exception_max_retries=getattr(settings, "llm_http_max_retries", REQUEST_EXCEPTION_MAX_RETRIES),
            empty_response_max_retries=getattr(settings, "llm_http_max_retries", EMPTY_RESPONSE_MAX_RETRIES),
            requests_per_minute_limit=getattr(settings, "llm_requests_per_minute_limit", 0),
            # 与预设热切换闭环：LLM_RETRY_BACKOFF_SECONDS → 固定短/长退避
            retry_backoff_seconds=getattr(
                settings,
                "llm_retry_backoff_seconds",
                HTTP_RETRY_BACKOFF_SECONDS,
            ),
        )
        print(f"LLM服务初始化成功! 模型: {settings.llm_model_name}")
    return _llm_instance


def reset_llm():
    """重置全局 LLM 单例，供配置变更或测试场景重新初始化。"""
    global _llm_instance
    _llm_instance = None
    print("LLM实例已重置")
