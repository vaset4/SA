"""认证与环境保护服务。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os

from fastapi import HTTPException, Request

from app.config import get_settings
from app.repositories.user_repository import get_user_repository
from app.services.jwt_service import (
    AUTH_SESSION_EXPIRED_DETAIL,
    create_access_token,
    create_refresh_token,
    get_jwt_expire_hours,
    is_default_secret,
    verify_jwt,
    verify_refresh_token,
)
from app.services.auth_roles import (
    configured_role_for_username,
    normalize_role,
    resolve_user_role,
)
from app.services.rate_limit_service import enforce_rate_limit

DEV_USER = {"sub": "anonymous", "user_id": "anonymous", "username": "developer", "role": "developer"}
PROTECTED_ENVS = {"staging", "prod"}
PUBLIC_BIND_HOSTS = {"0.0.0.0", "::"}
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 390000

logger = logging.getLogger(__name__)

# 这个模块承接两条主链：
# 1. 请求鉴权链：`middleware/auth.py` -> 本文件 -> `jwt_service.py / rate_limit_service.py`
# 2. 登录注册链：认证路由 -> 本文件 -> `user_repository.py`
#
# 也就是说，这里既负责“当前请求是谁”，也负责“用户账号如何创建与换 token”。


def _settings():
    return get_settings()


def get_api_key(settings=None) -> str:
    active_settings = settings or _settings()
    return active_settings.api_key or ""


def get_app_env(settings=None) -> str:
    active_settings = settings or _settings()
    return str(active_settings.app_env).lower()


def is_auth_enabled(settings=None) -> bool:
    active_settings = settings or _settings()
    return bool(active_settings.auth_enabled)


def is_public_bind_host(settings=None) -> bool:
    """判断当前 host 是否会暴露到非本机网络接口。"""
    active_settings = settings or _settings()
    host = str(getattr(active_settings, "host", "127.0.0.1") or "").strip().lower()
    return host in PUBLIC_BIND_HOSTS


def validate_auth_configuration(settings=None) -> None:
    """
    在受保护环境里强制校验认证配置。

    这里是一个很关键的安全闸门：即便有人把配置写错成“关闭认证”或“默认 secret”，
    staging/prod 也会直接失败，而不是静默带病启动。
    """
    active_settings = settings or _settings()
    app_env = get_app_env(active_settings)
    if app_env not in PROTECTED_ENVS:
        if is_public_bind_host(active_settings) and not is_auth_enabled(active_settings):
            raise HTTPException(status_code=500, detail="local 环境绑定公开地址时不允许关闭认证")
        if is_public_bind_host(active_settings) and is_default_secret(active_settings):
            raise HTTPException(status_code=500, detail="local 环境绑定公开地址时禁止使用默认 JWT secret")
        return
    if not is_auth_enabled(active_settings):
        raise HTTPException(status_code=500, detail="非 local 环境不允许关闭认证")
    if is_default_secret(active_settings):
        raise HTTPException(status_code=500, detail="staging/prod 环境禁止使用默认 JWT secret")


def resolve_authenticated_user(
    bearer_credentials: str | None,
    api_key: str | None,
    settings=None,
) -> dict:
    """【认证】【请求身份】优先验证显式凭证，无凭证时再按认证模式处理。

    Args:
        bearer_credentials: 请求携带的 JWT；提供后必须校验，不能转为开发身份。
        api_key: 请求携带的 API Key。
        settings: 可选运行配置。

    Returns:
        与实际凭证一致的用户；仅免登录开发请求使用开发身份。

    Raises:
        HTTPException: 凭证无效或账号模式下缺少凭证。
    """
    active_settings = settings or _settings()
    validate_auth_configuration(active_settings)
    # 1. 【认证】【请求身份】提供凭证的请求不能因关闭强制认证而变成 anonymous
    if bearer_credentials is not None:
        return verify_jwt(bearer_credentials, settings=active_settings)
    if api_key is not None:
        configured_api_key = get_api_key(active_settings)
        if not configured_api_key or api_key != configured_api_key:
            raise HTTPException(status_code=401, detail="API Key 无效")
        return {"sub": "api_key_user", "user_id": "api_key_user", "username": "api_key_user", "role": "user"}
    # 2. 【认证】【请求身份】仅未提供凭证的本机开发请求允许使用开发身份
    if not is_auth_enabled(active_settings):
        return DEV_USER.copy()
    raise HTTPException(
        status_code=401,
        detail="请提供 JWT token (Authorization: Bearer <token>) 或 API Key (X-API-Key: <key>)",
    )


def attach_user(request: Request, user: dict) -> dict:
    """把解析出的用户挂到 `request.state`，供 route/facade 继续复用。"""
    request.state.current_user = user
    return user


def resolve_user_for_request(
    request: Request,
    bearer_credentials: str | None,
    api_key: str | None,
    apply_rate_limit: bool = False,
    settings=None,
) -> dict:
    """
    请求期统一入口：鉴权 -> 可选限流 -> 注入 request.state。

    `middleware/auth.py` 暴露给 FastAPI 的依赖最终都会收敛到这里。
    """
    active_settings = settings or _settings()
    # 1. 【认证】【请求身份】HTTP 适配器可能将格式错误的显式凭证转成 None，不能再把它当免登录请求
    if request.headers.get("authorization") is not None and bearer_credentials is None:
        raise HTTPException(status_code=401, detail="Authorization 格式无效，请使用 Bearer token")
    if request.headers.get("x-api-key") is not None and api_key is None:
        raise HTTPException(status_code=401, detail="API Key 不能为空")
    user = resolve_authenticated_user(bearer_credentials, api_key, settings=active_settings)
    if apply_rate_limit:
        enforce_rate_limit(request, user, settings=active_settings)
    return attach_user(request, user)


def get_user_id(user: dict | None) -> str:
    """【认证】【用户归属】返回已解析身份，不为缺失或矛盾的身份生成 owner。

    Args:
        user: 认证入口产生的身份，user_id 与 sub 同时存在时必须一致。

    Returns:
        非空且符合当前认证模式的用户 ID。

    Raises:
        HTTPException: 身份缺失、冲突或账号模式使用 anonymous。
    """
    payload = user if isinstance(user, dict) else {}
    user_id = payload.get("user_id")
    subject = payload.get("sub")
    if any(value is not None and not isinstance(value, str) for value in (user_id, subject)):
        raise HTTPException(status_code=401, detail="用户身份格式无效，请重新登录")
    user_id = (user_id or "").strip()
    subject = (subject or "").strip()
    if user_id and subject and user_id != subject:
        raise HTTPException(status_code=401, detail="用户身份不一致，请重新登录")
    resolved_id = user_id or subject
    if not resolved_id:
        raise HTTPException(status_code=401, detail="缺少用户身份，请重新登录")
    if resolved_id == DEV_USER["user_id"] and is_auth_enabled():
        raise HTTPException(status_code=401, detail="账号模式不接受 anonymous 身份")
    return resolved_id


def _sanitize_username(username: str) -> str:
    cleaned = str(username or "").strip()
    if len(cleaned) < 3:
        raise ValueError("用户名至少 3 个字符")
    if len(cleaned) > 64:
        raise ValueError("用户名不能超过 64 个字符")
    return cleaned


def _sanitize_password(password: str) -> str:
    cleaned = str(password or "")
    if len(cleaned) < 6:
        raise ValueError("密码至少 6 个字符")
    if len(cleaned) > 128:
        raise ValueError("密码不能超过 128 个字符")
    return cleaned


def hash_password(password: str) -> str:
    """
    生成带随机 salt 的 PBKDF2 密码摘要。

    返回格式：
      `scheme$iterations$salt$base64_digest`
    这样后续即使迭代次数升级，也能通过摘要本身携带版本信息。
    """
    raw_password = _sanitize_password(password)
    salt = base64.b64encode(os.urandom(16)).decode("ascii")
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_ITERATIONS,
    )
    return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt}${base64.b64encode(digest).decode('ascii')}"


def verify_password(password: str, password_hash: str) -> bool:
    """按存储格式重算摘要，并用常量时间比较避免明显时序泄漏。"""
    try:
        scheme, iterations, salt, encoded_digest = str(password_hash or "").split("$", 3)
    except ValueError:
        return False
    if scheme != PASSWORD_SCHEME:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password or "").encode("utf-8"),
        salt.encode("utf-8"),
        int(iterations),
    )
    return hmac.compare_digest(base64.b64encode(digest).decode("ascii"), encoded_digest)


async def ensure_user_schema() -> None:
    await get_user_repository().ensure_schema()


async def register_user(username: str, password: str) -> dict:
    """注册用户；这里先确保 schema 存在，再做唯一性校验。"""
    await ensure_user_schema()
    sanitized_username = _sanitize_username(username)
    active_settings = _settings()
    repository = get_user_repository()
    existing = await repository.get_by_username(sanitized_username)
    if existing is not None:
        raise ValueError("用户名已存在")
    if configured_role_for_username(sanitized_username, settings=active_settings):
        raise ValueError("高权限用户名需先创建普通账号，再通过配置或数据库角色授权")
    user = await repository.create_user(
        sanitized_username,
        hash_password(password),
    )
    return user


async def authenticate_user(username: str, password: str) -> dict:
    """校验账号状态和密码并返回当前用户记录。

    Args:
        username: 客户端提交的用户名。
        password: 客户端提交的明文密码。

    Returns:
        通过认证的数据库用户记录。

    Raises:
        HTTPException: 账号不存在、已禁用或密码错误。
    """
    await ensure_user_schema()
    sanitized_username = _sanitize_username(username)
    repository = get_user_repository()
    user = await repository.get_by_username(sanitized_username)
    if (
        user is None
        or user.get("is_active") is not True
        or not verify_password(password, str(user.get("password_hash") or ""))
    ):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return user


def build_token_payload(
    user: dict,
    *,
    refresh_expires_at: int | None = None,
    settings=None,
) -> dict:
    """构造前端登录或刷新接口的统一返回体。

    Args:
        user: 当前数据库用户记录。
        refresh_expires_at: 轮换前 refresh token 的绝对过期时间戳。
        settings: 可选的运行配置。

    Returns:
        包含 access token、refresh token 和当前用户的完整会话 payload。
    """
    active_settings = settings or _settings()
    user_id = str(user.get("user_id") or user.get("sub") or "").strip()
    username = str(user.get("username") or "").strip()
    role = resolve_user_role(user, settings=active_settings)
    token_version = int(user["token_version"])
    return {
        "access_token": create_access_token(
            user_id,
            username=username,
            role=role,
            token_version=token_version,
            settings=active_settings,
        ),
        "refresh_token": create_refresh_token(
            user_id,
            username=username,
            role=role,
            token_version=token_version,
            expires_at=refresh_expires_at,
            settings=active_settings,
        ),
        "token_type": "bearer",
        "expires_in": int(get_jwt_expire_hours(active_settings)) * 3600,
        "user": {
            "user_id": user_id,
            "username": username,
            "role": role,
        },
    }


async def refresh_access_payload(refresh_token: str, *, settings=None) -> dict:
    """回源校验账号状态后在原绝对期限内换发 token。

    Args:
        refresh_token: 客户端提交的 refresh token。
        settings: 可选的运行配置。

    Returns:
        使用数据库当前账号信息签发的完整会话 payload。

    Raises:
        HTTPException: token 无效或账号状态与签发时不一致。
    """
    # 1. 【认证】【令牌刷新】先确保账号状态列完成可重复迁移，再验证 token
    await ensure_user_schema()
    active_settings = settings or _settings()
    payload = verify_refresh_token(refresh_token, settings=active_settings)

    # 2. 【认证】【令牌刷新】回源比较账号状态、版本和当前角色，禁止沿用旧 claims
    user = await get_user_repository().get_by_user_id(payload["user_id"])
    current_role = resolve_user_role(user or {}, settings=active_settings)
    if (
        user is None
        or user.get("is_active") is not True
        or int(user["token_version"]) != int(payload["token_version"])
        or current_role != normalize_role(payload.get("role"))
    ):
        raise HTTPException(status_code=401, detail=AUTH_SESSION_EXPIRED_DETAIL)

    # 3. 【认证】【令牌刷新】在旧 refresh 的绝对期限内签发完整会话快照
    logger.info("【认证】【令牌刷新】账号令牌刷新成功，user_id=%s", user["user_id"])
    return build_token_payload(
        user,
        refresh_expires_at=int(payload["exp"]),
        settings=active_settings,
    )


async def revoke_user_sessions(user_id: str) -> None:
    """递增账号令牌版本，使现有 refresh token 全部失效。

    Args:
        user_id: 当前认证用户标识。

    Returns:
        None.

    Raises:
        HTTPException: 当前账号已经不存在。
    """
    # 1. 【认证】【退出吊销】通过单条原子更新递增账号令牌版本
    await ensure_user_schema()
    token_version = await get_user_repository().revoke_all_tokens(user_id)
    if token_version is None:
        raise HTTPException(status_code=401, detail=AUTH_SESSION_EXPIRED_DETAIL)

    # 2. 【认证】【退出吊销】只记录账号和新版本，禁止记录任何 token 正文
    logger.info(
        "【认证】【退出吊销】账号令牌版本已递增，user_id=%s, token_version=%s",
        user_id,
        token_version,
    )
