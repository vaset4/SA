"""请求限流服务。"""

from __future__ import annotations

import re
import hashlib
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from app.config import get_settings

_RATE_LIMIT_PATTERN = re.compile(r"^\s*(\d+)\s*/\s*(second|minute|hour)\s*$", re.IGNORECASE)
_RATE_LIMIT_UNIT_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}
_TASK_DETAIL_PATH_PATTERN = re.compile(r"^/api/v1/tasks/[^/]+$", re.IGNORECASE)
_TASK_EVENTS_PATH_PATTERN = re.compile(r"^/api/v1/tasks/[^/]+/events$", re.IGNORECASE)
ANONYMOUS_AUTH_IDENTITY_PREFIX = "anonymous-auth"
# 这是进程内滑动窗口记录，不跨 worker 共享；更像“本机防抖阀门”，不是分布式限流系统。
_rate_limit_records: dict[str, deque[float]] = {}
_rate_limit_lock = threading.Lock()


def _settings():
    return get_settings()


def parse_rate_limit(value: str) -> tuple[int, float]:
    """把 `60/minute` 这类配置解析成 `(次数, 时间窗秒数)`。"""
    match = _RATE_LIMIT_PATTERN.match(value or "")
    if not match:
        raise ValueError(f"无效的 rate_limit 配置: {value!r}")
    count = int(match.group(1))
    if count <= 0:
        raise ValueError(f"rate_limit 次数必须大于 0: {value!r}")
    unit = match.group(2).lower()
    return count, _RATE_LIMIT_UNIT_SECONDS[unit]


def build_rate_limit_key(request: Request, user: dict) -> str:
    """
    构造限流桶 key。

    当前粒度是 `client_host + user_identity + request_path`，意味着：
      - 同一用户访问不同路径互不影响
      - 未认证请求至少还能按来源 IP 粗粒度隔离
    """
    identity = user.get("sub") or "anonymous"
    client_host = getattr(request.client, "host", None) or "unknown"
    return f"{client_host}:{identity}:{request.url.path}"


def _fingerprint(value: str) -> str:
    """为限流桶生成稳定短指纹，避免把 refresh token 等原值放进进程内 key。"""
    normalized = str(value or "").strip() or "missing"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def resolve_rate_limit_rule(request: Request, settings=None) -> tuple[int, float]:
    """
    解析当前请求该使用哪条限流规则。

    默认走全局配置，但任务详情/事件查询会被前端高频轮询，因此这里单独放宽。
    """
    active_settings = settings or _settings()
    path = request.url.path
    method = (request.method or "").upper()

    # 任务详情与事件查询会被前端持续轮询，沿用全局 60/minute 会把正常进度刷新也误杀。
    if method == "GET" and (_TASK_DETAIL_PATH_PATTERN.match(path) or _TASK_EVENTS_PATH_PATTERN.match(path)):
        return parse_rate_limit("180/minute")

    return parse_rate_limit(active_settings.rate_limit)


def _prune_requests(history: deque[float], window_seconds: float, now: float) -> None:
    """移除时间窗之外的历史请求记录，形成简单滑动窗口。"""
    threshold = now - window_seconds
    while history and history[0] <= threshold:
        history.popleft()


def enforce_rate_limit(request: Request, user: dict, settings=None) -> None:
    """
    执行请求级限流。

    认证链一般会在用户身份解析完成后调用这里，这样限流可以按“用户 + 路径”而不是只按
    IP 工作。
    """
    active_settings = settings or _settings()
    max_requests, window_seconds = resolve_rate_limit_rule(request, active_settings)
    now = time.monotonic()
    key = build_rate_limit_key(request, user)
    with _rate_limit_lock:
        history = _rate_limit_records.setdefault(key, deque())
        _prune_requests(history, window_seconds, now)
        if len(history) >= max_requests:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        history.append(now)


def enforce_anonymous_auth_rate_limit(
    request: Request,
    action: str,
    identifier: str,
    *,
    settings=None,
) -> None:
    """对匿名认证入口执行限流，按来源 IP、动作、稳定身份维度隔离。"""
    clean_action = str(action or "auth").strip().lower() or "auth"
    identity = f"{ANONYMOUS_AUTH_IDENTITY_PREFIX}:{clean_action}:{_fingerprint(identifier)}"
    enforce_rate_limit(request, {"sub": identity}, settings=settings)


def reset_rate_limiter() -> None:
    """清空进程内限流状态，主要供测试或调试场景复位。"""
    with _rate_limit_lock:
        _rate_limit_records.clear()
