"""向量与重排 API 共用的 HTTP 和索引完整性边界，作者：xxx。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

from app.config import normalize_model_api_url
from app.services.provider_request_queue import check_provider_request_active, provider_request_timeout


def request_model_api(
    endpoint: str, api_key: str, payload: dict[str, Any], *, timeout: float,
    operation: str, post: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    """接收地址、密钥、载荷和超时，返回 JSON 对象；错误仅包含操作名与状态。"""
    endpoint = normalize_model_api_url(endpoint, setting_name=f"{operation} API 地址")
    if not api_key.strip() or any(char.isspace() for char in api_key):
        raise ValueError(f"{operation} API 密钥未配置或格式错误")
    response = None
    try:
        check_provider_request_active()
        # 1. 【模型服务】【远程调用】只发送一次请求，重定向不能改变配置的服务身份
        response = post(endpoint, json=payload, timeout=provider_request_timeout(timeout), allow_redirects=False,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"{operation} API HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError(f"{operation} API 返回无效 JSON") from None
        if not isinstance(result, dict):
            raise RuntimeError(f"{operation} API 返回格式错误：期望 JSON 对象")
        return result
    except requests.Timeout:
        raise TimeoutError(f"{operation} API 请求超时") from None
    except requests.RequestException as exc:
        raise RuntimeError(f"{operation} API 请求失败：{type(exc).__name__}") from None
    finally:
        if response is not None:
            response.close()


def ordered_model_results(payload: Any, *, count: int, operation: str) -> list[dict[str, Any]]:
    """接收 API 结果数组和期望数量，验证完整唯一索引并返回原输入顺序的条目。"""
    if not isinstance(payload, list) or len(payload) != count:
        raise RuntimeError(f"{operation} API 返回数量不匹配：expected={count}")
    indexed: dict[int, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError(f"{operation} API 返回格式错误：期望结果对象")
        index = item.get("index")
        if type(index) is not int or not 0 <= index < count or index in indexed:
            raise RuntimeError(f"{operation} API 返回索引无效、重复或超出范围")
        indexed[index] = item
    return [indexed[index] for index in range(count)]
