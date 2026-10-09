"""Repository 共享工具。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, TypeVar

T = TypeVar("T")


def utc_now() -> datetime:
    """返回 UTC datetime 对象；仓储层统一使用 UTC，减少时区歧义。"""
    return datetime.now(timezone.utc)


def parse_datetime(value: str | datetime | None) -> datetime | None:
    """解析时间戳字符串或返回 datetime 对象。

    Args:
        value: ISO 格式字符串、datetime 对象或 None

    Returns:
        datetime 对象或 None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # 处理 ISO 格式字符串，支持 'Z' 后缀
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    return None


def dump_json(value: Any) -> str:
    """统一序列化 JSON 值；仓储层所有 JSON/JSONB 字段都应走这里。"""
    return json.dumps(value)


def load_json(value: Any, default: T) -> T:
    """统一反序列化 JSON/JSONB 字段，兼容 asyncpg 已解码和字符串两种输入。"""
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_record(row: Any) -> dict[str, Any] | None:
    """
    把 asyncpg 记录转换成普通字典。

    顺带把 datetime-like 字段转成 ISO 字符串，避免上层 service/facade 再各自处理一遍。
    """
    if row is None:
        return None
    data = dict(row)
    for key, value in list(data.items()):
        if hasattr(value, "isoformat"):
            data[key] = value.isoformat()
    return data
