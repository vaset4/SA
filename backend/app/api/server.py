"""
FastAPI app 装配层。

这个文件的职责很单一：把“一个可运行的 FastAPI 应用”拼起来。
它本身不承担业务逻辑，而是集中完成以下几件事：

1. 创建 `app`
2. 绑定统一 lifespan，把启动/关闭顺序交给 runtime 层
3. 注册全量 API 路由
4. 处理跨域配置这种与业务弱相关、但和部署强相关的横切关注点

理解项目时，可以把这里看成后端的“总装车间”：
具体零件来自 `runtime/*`、`api/routes/*`、`middleware/*`，
而这里负责把它们装成一个完整进程入口。
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 某些本地抓包环境会注入不可写 SSLKEYLOGFILE，导致 aiohttp/ssl 在导入期失败。
os.environ.pop("SSLKEYLOGFILE", None)

from app.api.routes import router as api_router
from app.api.routes.research_api import research
from app.api.routes.stream_api import stream
from app.api.schemas.research import ResearchRequest
from app.config import get_settings
from app.runtime import build_app_lifespan

settings = get_settings()

LOOPBACK_HOST_ALIASES = {
    "localhost": "127.0.0.1",
    "127.0.0.1": "localhost",
}
LOCAL_DEV_ORIGIN_REGEX = r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$"


def _expand_cors_origins(raw_origins: str) -> list[str]:
    """
    展开配置里的 CORS 源，并补齐 localhost/127.0.0.1 的本地别名。

    这样做的原因是：前端开发时经常在 `localhost` 和 `127.0.0.1`
    之间切换，浏览器会把它们视为不同 origin。这里提前补齐别名，
    可以减少“明明是本地开发却被 CORS 拦住”的无效排查。

    Args:
        raw_origins: 配置里的 CORS 源字符串，逗号分隔

    Returns:
        list[str]: 已补齐 loopback 别名且去重后的 allow_origins 列表
    """
    if not raw_origins:
        return ["*"]

    origins: list[str] = []
    # 1. 先拆分配置值，并保留原始声明的 origin
    for raw_origin in raw_origins.split(","):
        origin = raw_origin.strip()
        if not origin:
            continue
        origins.append(origin)
        if origin == "*":
            return ["*"]

        parsed = urlsplit(origin)
        alias_host = LOOPBACK_HOST_ALIASES.get(parsed.hostname or "")
        if not alias_host or not parsed.scheme:
            continue

        # 2. 对 localhost/127.0.0.1 自动补齐别名，减少本地开发切换时的 CORS 干扰
        alias_origin = f"{parsed.scheme}://{alias_host}"
        if parsed.port is not None:
            alias_origin = f"{alias_origin}:{parsed.port}"
        origins.append(alias_origin)

    # 3. 最后去重，保证中间件看到的是稳定的 allow_origins 列表
    return list(dict.fromkeys(origins)) or ["*"]


# `app` 是整个后端对外暴露的核心对象。
# 真正的资源初始化顺序不写在这里，而是交给 `build_app_lifespan()` 统一管理。
app = FastAPI(
    title="Study Assistant API",
    version="1.0.0",
    lifespan=build_app_lifespan(),
)

# CORS 只在 HTTP 层生效，不影响内部 service/repository 调用。
app.add_middleware(
    CORSMiddleware,
    allow_origins=_expand_cors_origins(settings.cors_origins),
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RAG-Run-ID"],
)

# 路由在 `app.api.routes` 里统一汇总，这里只做一次性挂载。
app.include_router(api_router)

__all__ = [
    "HTTPException",
    "ResearchRequest",
    "app",
    "research",
    "stream",
]
