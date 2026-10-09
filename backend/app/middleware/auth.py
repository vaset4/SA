"""
认证依赖适配层。

这个文件处在 FastAPI 依赖注入边界上，负责把 HTTP 层的鉴权输入
（Bearer Token、API Key、Request 对象）转换成下游统一使用的 `user` 结构。

它本身不实现鉴权规则，而是把规则委托给 `auth_service` / `jwt_service`；
因此这里更像一个“入口适配器”而不是完整的认证服务。
"""

from __future__ import annotations

from fastapi import Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import get_user_id, resolve_user_for_request
from app.services.jwt_service import create_access_token, create_refresh_token, verify_jwt, verify_refresh_token
from app.services.rate_limit_service import parse_rate_limit, reset_rate_limiter

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_header),
) -> dict:
    """
    解析当前请求用户，但不附带限流。

    适合登录态查询、系统配置等低频接口；真正需要限流保护的业务接口
    一般使用 `get_current_user_with_rate_limit`。
    """
    return resolve_user_for_request(
        request=request,
        bearer_credentials=bearer.credentials if bearer else None,
        api_key=api_key,
    )


async def get_current_user_with_rate_limit(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_header),
) -> dict:
    """
    解析当前请求用户，并在同一入口处附带限流判定。

    这样 route 层只需声明一个依赖，就能同时拿到用户身份和基础频控保护。
    """
    return resolve_user_for_request(
        request=request,
        bearer_credentials=bearer.credentials if bearer else None,
        api_key=api_key,
        apply_rate_limit=True,
    )


__all__ = [
    "api_key_header",
    "bearer_scheme",
    "create_access_token",
    "create_refresh_token",
    "get_user_id",
    "get_current_user",
    "get_current_user_with_rate_limit",
    "parse_rate_limit",
    "reset_rate_limiter",
    "verify_jwt",
    "verify_refresh_token",
]
