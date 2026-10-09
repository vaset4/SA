"""JWT 编码与校验服务。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException
from jwt import ExpiredSignatureError, InvalidTokenError

from app.config import get_settings

ALGORITHM = "HS256"
DEFAULT_DEV_SECRET = "change-me-in-production"
DEFAULT_EXPIRE_HOURS = 24
DEFAULT_REFRESH_EXPIRE_DAYS = 7
AUTH_SESSION_EXPIRED_DETAIL = "登录状态已失效，请重新登录"

# 这个文件只负责 token 的“签发与验证”，不决定请求应不应该放行。
# 真正的环境保护策略在 `auth_service.py` 中，例如：
#   - 生产环境不允许默认 secret
#   - access token / refresh token 的使用场景区分


def _settings():
    return get_settings()


def get_jwt_secret_key(settings=None) -> str:
    active_settings = settings or _settings()
    return active_settings.jwt_secret_key or DEFAULT_DEV_SECRET


def get_jwt_expire_hours(settings=None) -> int:
    active_settings = settings or _settings()
    return int(active_settings.jwt_expire_hours or DEFAULT_EXPIRE_HOURS)


def get_jwt_refresh_expire_days(settings=None) -> int:
    active_settings = settings or _settings()
    return int(active_settings.jwt_refresh_expire_days or DEFAULT_REFRESH_EXPIRE_DAYS)


def is_default_secret(settings=None) -> bool:
    return get_jwt_secret_key(settings) == DEFAULT_DEV_SECRET


def _encode_token(payload: dict, *, settings=None) -> str:
    """统一使用当前 secret 和 HS256 签发 JWT。"""
    active_settings = settings or _settings()
    return jwt.encode(payload, get_jwt_secret_key(active_settings), algorithm=ALGORITHM)


def create_access_token(
    user_id: str,
    *,
    username: str = "",
    role: str = "user",
    token_version: int = 0,
    expires_delta: timedelta | None = None,
    settings=None,
) -> str:
    """生成带账号令牌版本的 access token。

    Args:
        user_id: 当前用户标识。
        username: 当前用户名。
        role: 当前角色。
        token_version: 当前账号令牌版本。
        expires_delta: 可选的 access token 有效时长。
        settings: 可选的运行配置。

    Returns:
        已签名的 access token。
    """
    active_settings = settings or _settings()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(hours=get_jwt_expire_hours(active_settings)))
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "username": username,
        "role": role,
        "token_version": token_version,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return _encode_token(payload, settings=active_settings)


def create_refresh_token(
    user_id: str,
    *,
    username: str = "",
    role: str = "user",
    token_version: int = 0,
    expires_at: int | None = None,
    expires_delta: timedelta | None = None,
    settings=None,
) -> str:
    """生成受账号版本和绝对到期时间约束的 refresh token。

    Args:
        user_id: 当前用户标识。
        username: 当前用户名。
        role: 当前角色。
        token_version: 当前账号令牌版本。
        expires_at: 轮换前 refresh token 的绝对过期时间戳。
        expires_delta: 可选的 refresh token 有效时长。
        settings: 可选的运行配置。

    Returns:
        已签名的 refresh token。
    """
    active_settings = settings or _settings()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=get_jwt_refresh_expire_days(active_settings)))
    if expires_at is not None:
        expire = min(expire, datetime.fromtimestamp(expires_at, timezone.utc))
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "username": username,
        "role": role,
        "token_version": token_version,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return _encode_token(payload, settings=active_settings)


def _decode_token(token: str, *, settings=None) -> dict:
    """解码 JWT 并归一化认证链需要的最小字段。

    Args:
        token: 需要验签和解码的 JWT。
        settings: 可选的运行配置。

    Returns:
        已归一化用户标识、角色和令牌版本的 claims。

    Raises:
        HTTPException: JWT 签名、期限、用户标识或版本字段无效。
    """
    active_settings = settings or _settings()
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret_key(active_settings),
            algorithms=[ALGORITHM],
        )
    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="JWT token 无效或已过期") from exc
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="JWT token 无效或已过期") from exc
    user_id = str(payload.get("user_id") or payload.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="JWT token 无效或已过期")
    payload["sub"] = user_id
    payload["user_id"] = user_id
    payload["username"] = str(payload.get("username") or "").strip()
    payload["role"] = str(payload.get("role") or "user").strip() or "user"
    raw_version = payload.get("token_version")
    if raw_version is not None:
        try:
            payload["token_version"] = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=401, detail="JWT token 无效或已过期") from exc
        if payload["token_version"] < 0:
            raise HTTPException(status_code=401, detail="JWT token 无效或已过期")
    return payload


def verify_jwt(token: str, settings=None) -> dict:
    """校验 access token；空 type 兼容历史 token，其他类型一律拒绝。"""
    payload = _decode_token(token, settings=settings)
    if payload.get("type") not in {"", None, "access"}:
        raise HTTPException(status_code=401, detail="JWT token 类型无效")
    payload["type"] = "access"
    return payload


def verify_refresh_token(token: str, settings=None) -> dict:
    """校验 refresh token 类型和版本合同。

    Args:
        token: 客户端提交的 refresh token。
        settings: 可选的运行配置。

    Returns:
        通过校验的 refresh token claims。

    Raises:
        HTTPException: token 无效、过期、类型错误或缺少 token_version。
    """
    try:
        payload = _decode_token(token, settings=settings)
    except HTTPException as exc:
        raise HTTPException(status_code=401, detail=AUTH_SESSION_EXPIRED_DETAIL) from exc
    if payload.get("type") != "refresh" or "token_version" not in payload:
        raise HTTPException(status_code=401, detail=AUTH_SESSION_EXPIRED_DETAIL)
    return payload
