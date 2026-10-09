"""注册、登录与 token 刷新路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas.auth import LoginRequest, RefreshTokenRequest, RegisterRequest
from app.middleware.auth import get_current_user_with_rate_limit
from app.services import auth_service
from app.services.rate_limit_service import enforce_anonymous_auth_rate_limit

router = APIRouter()

# 这组接口基本都很薄，真正的认证规则在 `auth_service.py / jwt_service.py`。
# route 层主要负责：
#   - 做请求体校验
#   - 把领域异常映射成合适的 HTTP 状态码
#   - 返回统一 token payload 给前端


def _enforce_anonymous_auth_limit(request: Request, action: str, identifier: str) -> None:
    """认证匿名入口的限流适配；计数规则保留在 rate_limit_service。"""
    enforce_anonymous_auth_rate_limit(request, action, identifier)


@router.post("/auth/register")
async def register(body: RegisterRequest, request: Request):
    """
    注册一个新用户并直接返回 token。

    这样前端在注册成功后不需要再立刻额外调用一次登录接口。
    """
    _enforce_anonymous_auth_limit(request, "register", body.username)
    try:
        user = await auth_service.register_user(body.username, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return auth_service.build_token_payload(user)


@router.post("/auth/login")
async def login(body: LoginRequest, request: Request):
    """用户名密码登录；核心密码校验逻辑由 `auth_service.authenticate_user()` 承担。"""
    _enforce_anonymous_auth_limit(request, "login", body.username)
    user = await auth_service.authenticate_user(body.username, body.password)
    return auth_service.build_token_payload(user)


@router.post("/auth/refresh")
async def refresh_token(body: RefreshTokenRequest, request: Request):
    """限流后使用 refresh token 换取完整会话。

    Args:
        body: 包含 refresh token 的请求体。
        request: 当前 FastAPI 请求。

    Returns:
        与登录接口结构一致的完整会话 payload。
    """
    _enforce_anonymous_auth_limit(request, "refresh", body.refresh_token)
    return await auth_service.refresh_access_payload(body.refresh_token)


@router.post("/auth/logout")
async def logout(user: dict = Depends(get_current_user_with_rate_limit)):
    """吊销当前账号全部 refresh token。

    Args:
        user: 认证依赖解析出的当前用户。

    Returns:
        服务端完成账号级吊销的确认结果。

    Raises:
        HTTPException: API key 合成身份不属于用户会话。
    """
    user_id = auth_service.get_user_id(user)
    if user_id == "api_key_user":
        raise HTTPException(status_code=422, detail="API key 不支持用户会话退出")
    await auth_service.revoke_user_sessions(user_id)
    return {"revoked": True}


@router.get("/auth/me")
async def me(user: dict = Depends(get_current_user_with_rate_limit)):
    """返回当前登录用户；用户身份已在依赖里完成鉴权与限流。"""
    return {
        "user": {
            "user_id": auth_service.get_user_id(user),
            "username": user.get("username") or "",
            "role": user.get("role") or "user",
        }
    }
