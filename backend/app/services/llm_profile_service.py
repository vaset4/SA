"""LLM 供应商预设服务。

负责预设的持久化(data/llm_profiles.json)、增删改查，以及"激活"——
把某个预设的连接三要素与重试规则写回 .env 并热重载配置，让运行时立即切换上游。

设计取舍：
  - 预设与激活标记都存 JSON，Key 明文存本地文件(单机自用场景)，对外一律脱敏
  - 激活时同步写回 .env，保证进程重启后仍沿用所选预设
  - 写 .env 用 dotenv.set_key 逐字段更新，保留其它无关配置不动
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from dotenv import set_key

from app.api.schemas.llm_profile import (
    LLMProfileCreate,
    LLMProfileRetryRule,
    LLMProfileUpdate,
    LLMProfileView,
    LLMTestResult,
)
from app.config import get_settings, get_settings_env_path, reset_settings
from app.services.llm_service import reset_llm

logger = logging.getLogger(__name__)

# 预设数据沿用后端工作目录，配置写回当前副本的权威文件
_PROFILE_STORE_PATH = Path("data") / "llm_profiles.json"
_ENV_PATH = get_settings_env_path()

_MASK_VISIBLE_TAIL = 4  # 脱敏时保留的尾部明文位数


def _load_store() -> dict[str, Any]:
    """读取预设存储文件；不存在或损坏时回退到空结构。"""
    if not _PROFILE_STORE_PATH.exists():
        return {"active_id": "", "profiles": []}
    try:
        raw = _PROFILE_STORE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        logger.error("LLM 预设存储读取失败，按空存储处理: path=%s", _PROFILE_STORE_PATH)
        return {"active_id": "", "profiles": []}
    if not isinstance(data, dict):
        return {"active_id": "", "profiles": []}
    data.setdefault("active_id", "")
    data.setdefault("profiles", [])
    return data


def _save_store(store: dict[str, Any]) -> None:
    """把预设存储写回磁盘。"""
    _PROFILE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILE_STORE_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _mask_key(api_key: str) -> str:
    """把 API Key 脱敏成仅露尾部若干位的展示串。"""
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= _MASK_VISIBLE_TAIL:
        return "*" * len(key)
    return f"{'*' * 6}{key[-_MASK_VISIBLE_TAIL:]}"


def _to_view(record: dict[str, Any], active_id: str) -> LLMProfileView:
    """把存储记录转换成对外视图（脱敏 Key、带激活标记）。"""
    rule = LLMProfileRetryRule(**record.get("retry_rule", {}))
    return LLMProfileView(
        id=record["id"],
        name=record.get("name", ""),
        base_url=record.get("base_url", ""),
        model_name=record.get("model_name", ""),
        api_key="",
        temperature=record.get("temperature", 0.0),
        retry_rule=rule,
        is_active=record["id"] == active_id,
        api_key_masked=_mask_key(record.get("api_key", "")),
    )


def list_profiles() -> list[LLMProfileView]:
    """列出全部预设（Key 脱敏），当前激活项带 is_active=True。"""
    store = _load_store()
    active_id = store.get("active_id", "")
    return [_to_view(item, active_id) for item in store.get("profiles", [])]


def create_profile(payload: LLMProfileCreate) -> LLMProfileView:
    """新增一个预设并返回其视图。"""
    store = _load_store()
    record = {
        "id": f"llmp_{uuid.uuid4().hex[:10]}",
        "name": payload.name,
        "base_url": payload.base_url,
        "model_name": payload.model_name,
        "api_key": payload.api_key,
        "temperature": payload.temperature,
        "retry_rule": payload.retry_rule.model_dump(),
    }
    store["profiles"].append(record)
    _save_store(store)
    return _to_view(record, store.get("active_id", ""))


def update_profile(profile_id: str, payload: LLMProfileUpdate) -> LLMProfileView:
    """编辑预设；api_key 留空表示保留原值。找不到抛 LookupError。"""
    store = _load_store()
    record = _find_record(store, profile_id)
    record["name"] = payload.name
    record["base_url"] = payload.base_url
    record["model_name"] = payload.model_name
    record["temperature"] = payload.temperature
    record["retry_rule"] = payload.retry_rule.model_dump()
    # 编辑时留空 Key 表示不改动，避免脱敏展示后误清空真实密钥
    if payload.api_key.strip():
        record["api_key"] = payload.api_key
    _save_store(store)
    # 若正在编辑的是当前激活预设，同步刷新运行时，让改动立即生效
    if store.get("active_id") == profile_id:
        _apply_to_runtime(record)
    return _to_view(record, store.get("active_id", ""))


def delete_profile(profile_id: str) -> None:
    """删除预设；不允许删除当前激活项。找不到抛 LookupError。"""
    store = _load_store()
    _find_record(store, profile_id)
    if store.get("active_id") == profile_id:
        raise ValueError("不能删除当前正在使用的预设，请先切换到其它预设")
    store["profiles"] = [item for item in store["profiles"] if item["id"] != profile_id]
    _save_store(store)


def activate_profile(profile_id: str) -> LLMProfileView:
    """激活预设：写回 .env → 重载配置 → 重置 LLM 单例，使上游立即切换。"""
    store = _load_store()
    record = _find_record(store, profile_id)
    store["active_id"] = profile_id
    _save_store(store)
    _apply_to_runtime(record)
    return _to_view(record, profile_id)


def _find_record(store: dict[str, Any], profile_id: str) -> dict[str, Any]:
    """在存储里定位预设记录，找不到抛 LookupError。"""
    for item in store.get("profiles", []):
        if item["id"] == profile_id:
            return item
    raise LookupError(f"预设不存在: {profile_id}")


def _apply_to_runtime(record: dict[str, Any]) -> None:
    """接收预设记录，将连接与重试规则持久化并立即应用到运行时；无返回值。"""
    rule = LLMProfileRetryRule(**record.get("retry_rule", {}))
    values = {
        "llm_model_name": record.get("model_name", ""), "llm_api_key": record.get("api_key", ""),
        "llm_base_url": record.get("base_url", ""), "llm_temperature": record.get("temperature", 0.0),
        "llm_timeout": rule.timeout_seconds, "llm_requests_per_minute_limit": rule.requests_per_minute,
        "llm_http_max_retries": rule.max_retries, "llm_min_request_interval_seconds": _interval_from_rpm(rule),
        "llm_retry_backoff_seconds": rule.retry_backoff_seconds,
    }
    # 1. 【LLM配置】【预设持久化】写回当前副本的权威文件，避免根配置遮住后端写入值
    for key, value in values.items():
        set_key(str(_ENV_PATH), key.upper(), str(value))
    # 2. 【LLM配置】【切换上游】显式激活覆盖本次运行的启动值，普通重启仍由部署环境优先
    reset_settings(llm_overrides=values)
    reset_llm()
    logger.info(
        "【LLM配置】【切换上游】已激活预设并热重载: name=%s base_url=%s model=%s rpm=%s max_retries=%s backoff=%s",
        record.get("name"),
        record.get("base_url"),
        record.get("model_name"),
        rule.requests_per_minute,
        rule.max_retries,
        rule.retry_backoff_seconds,
    )


def _interval_from_rpm(rule: LLMProfileRetryRule) -> float:
    """由 RPM 反推最小请求间隔(60/rpm)；rpm 为 0 时回退到 0（不额外节流）。"""
    if rule.requests_per_minute <= 0:
        return 0.0
    return round(60.0 / rule.requests_per_minute, 3)


def resolve_test_target(
    *,
    profile_id: str,
    base_url: str,
    model_name: str,
    api_key: str,
) -> tuple[str, str, str]:
    """按 profile_id 取存储中真实连接信息，显式传入的字段优先覆盖。找不到抛 LookupError。"""
    store = _load_store()
    record = _find_record(store, profile_id)
    return (
        base_url.strip() or record.get("base_url", ""),
        model_name.strip() or record.get("model_name", ""),
        api_key.strip() or record.get("api_key", ""),
    )


def test_connection(
    *,
    base_url: str,
    model_name: str,
    api_key: str,
) -> LLMTestResult:
    """向指定上游发一个极小请求探测连通性，返回通/不通与耗时。"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    started = time.monotonic()
    # 探活超时可配：慢上游(如免费 NVIDIA 网关)握手偏慢，走配置而非硬编码，便于按上游调大
    settings = get_settings()
    test_timeout = settings.llm_connection_test_timeout_seconds
    queue = None
    # 1. 【模型配置】【连接测试】当前受限凭证的探活也消耗同一额度，其他配置不占该组
    if api_key == settings.llm_api_key and base_url.rstrip('/') == settings.llm_base_url.rstrip('/'):
        from app.services.provider_request_queue import get_provider_queue

        queue = get_provider_queue()
    ticket = None
    error: BaseException | None = None
    try:
        from app.services.provider_request_queue import provider_request_timeout

        if queue is not None:
            ticket = queue.acquire({"call_id": "connection_test", "attempt": 1})
            queue.mark_started(ticket)
        # 测试请求 max_tokens=1，只判连通与鉴权，不等模型长输出
        response = requests.post(url, json=payload, headers=headers,
            timeout=provider_request_timeout(test_timeout, check_active=queue is None))
        # 2. 【模型配置】【共享冷却】先发布限流事实，再允许其他进程领取发送资格
        if response.status_code == 429 and queue is not None:
            queue.cooldown(response.headers.get("Retry-After"), {"call_id": "connection_test", "attempt": 1})
    except requests.exceptions.Timeout as exc:
        error = exc
        return LLMTestResult(ok=False, message=f"连接超时（{test_timeout}s），上游无响应")
    except TimeoutError as exc:
        error = exc
        return LLMTestResult(ok=False, message="模型服务额度等待超过连接测试期限")
    except requests.exceptions.ConnectionError as exc:
        error = exc
        return LLMTestResult(ok=False, message="无法连接，请检查 Base URL 与网络")
    except BaseException as exc:
        error = exc
        raise
    finally:
        if ticket is not None:
            assert queue is not None
            queue.release(ticket, error=error)
    latency_ms = int((time.monotonic() - started) * 1000)
    return _interpret_test_response(response, latency_ms)


def _interpret_test_response(
    response: requests.Response,
    latency_ms: int,
) -> LLMTestResult:
    """把测试请求的 HTTP 响应翻译成用户可读的连通性结论。"""
    status = response.status_code
    if 200 <= status < 300:
        return LLMTestResult(ok=True, latency_ms=latency_ms, status_code=status, message="连接正常")
    reason_map = {
        401: "鉴权失败，请检查 API Key",
        403: "无权限，请检查 Key 或模型白名单",
        404: "端点不存在，请检查 Base URL 是否为 OpenAI 兼容地址",
        429: "上游繁忙或额度已满（可用，稍后重试）",
        503: "上游繁忙（可用，稍后重试）",
    }
    message = reason_map.get(status, f"上游返回异常状态 {status}")
    # 429/503 说明连接与鉴权其实是通的，只是当前繁忙，视为"可用"
    reachable = status in (429, 503)
    return LLMTestResult(ok=reachable, latency_ms=latency_ms, status_code=status, message=message)
