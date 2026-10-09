"""RAGAS 0.4.3 的项目模型适配及独立 judge 审计。作者：xxx。"""

from __future__ import annotations

import asyncio
import importlib.metadata
import time
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from app.services.llm_service import OpenAICompatibleLLM
from app.services.provider_request_queue import queue_policy_identity
from evals.contracts import read_snapshot, snapshot
from evals.ragas.evals import _ProjectRagasLLM, _load_project_settings
from evals.ragas.frozen_bridge import METRIC_VARIANTS, RAGAS_VERSION


def _sanitize(value: Any, *, secrets: tuple[str, ...]) -> Any:
    """参数为 judge 审计和配置秘密；返回去除敏感字段及凭证字串的副本。"""
    private = {"authorization", "api_key", "base_url", "headers", "api_base", "access_token"}
    if isinstance(value, dict):
        return {key: _sanitize(item, secrets=secrets) for key, item in value.items() if key.lower() not in private}
    if isinstance(value, list):
        return [_sanitize(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


class _JudgeClient(OpenAICompatibleLLM):
    """在项目传输边界捕获真实请求体，并隔离旧客户端的无条件缓存写入。"""

    def __init__(self, **options: Any) -> None:
        """参数为项目客户端设置；初始化本次独有缓存键和请求列表。"""
        super().__init__(**options)
        self.judge_cache_key = f"ragas-judge:{uuid4()}"
        self.requests: list[dict[str, Any]] = []

    def _cache_key(self, normalized_messages: list[dict[str, str]], *, response_format: Any = None) -> str:
        """参数为实际消息和格式；返回本次 judge 专用键，不能命中或覆盖产品缓存。"""
        return self.judge_cache_key

    def _post_chat_completions(self, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        """参数为传输头和请求体；仅捕获真实请求体，返回项目传输结果。"""
        self.requests.append(read_snapshot(snapshot(payload)))
        return super()._post_chat_completions(headers, payload)

    def _post_chat_completions_stream(self, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        """参数为流式传输头和请求体；仅捕获真实请求体，返回原始响应流。"""
        self.requests.append(read_snapshot(snapshot(payload)))
        return super()._post_chat_completions_stream(headers, payload)


class _AuditedProjectLLM(_ProjectRagasLLM):
    """复用项目模型传输，每次 judge 调用保存独立请求、响应和 provider 用量。"""

    def __init__(self, settings: Any) -> None:
        """参数为项目设置；初始化评测模型和本次审计列表，不输出密钥。"""
        super().__init__(
            model=settings.llm_model_name, api_key=settings.llm_api_key,
            base_url=settings.llm_base_url, temperature=0.0, timeout=settings.llm_timeout,
            min_request_interval_seconds=settings.ragas_llm_min_request_interval_seconds,
        )
        self.calls: list[dict[str, Any]] = []

    def _build_client(self, *, temperature: float) -> _JudgeClient:
        """参数为当前指标温度；构造保留项目真实传输和重试语义的审计客户端。"""
        return _JudgeClient(model=self.model, api_key=self._api_key, base_url=self._base_url,
                            temperature=temperature, timeout=self._timeout,
                            min_request_interval_seconds=self._min_request_interval_seconds,
                            **self._transport_options())

    async def _ainvoke_once(self, *, payload: Any, temperature: float) -> str:
        """参数为 RAGAS 提示词与温度；返回原始文本，同时捕获真实响应而非估算用量。"""
        client = self._build_client(temperature=temperature)
        audit: dict[str, Any] = {"requests": client.requests, "status": "running"}
        self.calls.append(audit)
        started = time.perf_counter()
        try:
            response = await client.ainvoke(payload, use_cache=False)
            audit.update({"status": "completed", "response": response.raw, "output": response.content,
                          "usage": response.raw.get("usage"),
                          "usage_status": "measured" if isinstance(response.raw.get("usage"), dict) else "missing"})
            return response.content
        except asyncio.CancelledError:
            audit.update({"status": "cancelled", "usage": None, "usage_status": "missing"})
            raise
        except Exception as exc:
            audit.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
                          "usage": None, "usage_status": "missing"})
            failed_response = getattr(exc, "response", None)
            if failed_response is not None:
                audit["failed_response"] = {"status_code": failed_response.status_code, "body": failed_response.text}
            raise
        finally:
            from app.services.cache import get_cache

            # 1. 【RAGAS】【缓存隔离】旧客户端始终写缓存，只清理本次 judge 专用键
            get_cache().delete_llm_response(client.judge_cache_key)
            audit["elapsed_seconds"] = time.perf_counter() - started


class ProjectRagasScorer:
    """逐指标调用固定 RAGAS 变体，所有调用使用独立 trace 和禁用产品响应缓存。"""

    def __init__(self, settings: Any = None, *, enabled_metrics: tuple[str, ...] = tuple(METRIC_VARIANTS)) -> None:
        """参数为可选设置；固定版本和指标参数，只有实际需要评分时才初始化。"""
        actual_version = importlib.metadata.version("ragas")
        if actual_version != RAGAS_VERSION:
            raise RuntimeError(f"RAGAS version mismatch: expected {RAGAS_VERSION}, got {actual_version}")
        self.settings = settings if settings is not None else _load_project_settings()
        if not enabled_metrics or any(name not in METRIC_VARIANTS for name in enabled_metrics):
            raise ValueError("unsupported RAGAS measurement selection")
        self.enabled_metrics = enabled_metrics
        self.metadata = {
            "provider": "studyassistant-project-llm", "model": self.settings.llm_model_name,
            "provider_host": urlsplit(self.settings.llm_base_url).hostname,
            "ragas_version": actual_version, "default_temperature": 0.0,
            "timeout_seconds": self.settings.llm_timeout,
            "min_request_interval_seconds": self.settings.ragas_llm_min_request_interval_seconds,
            "provider_queue": queue_policy_identity(_load_project_settings()),
            "metric_variants": {name: METRIC_VARIANTS[name] for name in enabled_metrics}, "embedding": None,
            "resource_role": "judge", "cache_enabled": False,
            "cache_isolation": "unique_judge_key_deleted_after_call",
            "cost": None, "cost_status": "missing", "cost_reason": "no_frozen_price_schedule",
        }

    async def __call__(self, metric: str, sample: dict[str, Any]) -> dict[str, Any]:
        """参数为指标名和隔离样本；返回数值、原始 judge 交互及独立 trace。"""
        if metric not in self.enabled_metrics:
            raise ValueError(f"metric_disabled_by_frozen_policy:{metric}")
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
        )
        from ragas.run_config import RunConfig

        from app.middleware.trace_context import TraceContext, export_evidence_observations, reset_trace_ctx, set_trace_ctx, trace_label

        factories = {"context_precision": LLMContextPrecisionWithReference, "context_recall": LLMContextRecall,
                     "faithfulness": Faithfulness, "factual_correctness": FactualCorrectness}
        llm = _AuditedProjectLLM(self.settings)
        metric_instance = factories[metric](llm=llm)
        metric_instance.init(RunConfig(timeout=self.settings.llm_timeout, max_workers=1))
        trace = TraceContext(request_id=f"ragas-judge-{uuid4()}", node="ragas_judge", stage=metric)
        token = set_trace_ctx(trace)
        started = time.perf_counter()
        result: dict[str, Any]
        try:
            # 1. 【RAGAS】【资源隔离】judge trace 不进入被测产品的调用或 token 统计
            with trace_label("ragas_judge", metric):
                value = await metric_instance.single_turn_ascore(
                    SingleTurnSample(**sample), timeout=self.settings.llm_timeout,
                )
            result = {"value": float(value)}
        except Exception as exc:
            result = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        finally:
            reset_trace_ctx(token)
        captured = export_evidence_observations(trace)
        # 2. 【RAGAS】【角色归属】此独立 trace 只执行评分请求，导出副本明确归属 judge
        captured["evidence_observations"] = [
            {**event, "payload": {**event["payload"], "actor": "judge"}}
            for event in captured["evidence_observations"]
        ]
        audit = {**result, "judge": dict(self.metadata), "calls": llm.calls, "observation": captured,
                 "trace": [asdict(call) for call in trace.model_calls],
                 "elapsed_seconds": time.perf_counter() - started,
                 "cost": None, "cost_status": "missing"}
        return _sanitize(audit, secrets=(self.settings.llm_api_key, self.settings.llm_base_url))
