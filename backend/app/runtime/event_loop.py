"""
运行时启动辅助。

当前模块只有一个职责：在 Windows 环境下，把 asyncio 的默认事件循环策略切换成
更适配 psycopg / PostgreSQL async 场景的 Selector 版本。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any


def configure_windows_event_loop_policy(
    asyncio_module: Any = asyncio,
    platform_name: str | None = None,
) -> bool:
    """
    在 Windows 下切到 psycopg 更稳定的 Selector 事件循环策略。

    返回值表示本次调用是否真的执行了策略切换：
      - `True`：当前是 Windows，且存在对应策略类型
      - `False`：当前不是 Windows，或运行环境没有该策略类
    """
    platform_name = sys.platform if platform_name is None else platform_name
    policy_type = getattr(asyncio_module, "WindowsSelectorEventLoopPolicy", None)
    if not platform_name.startswith("win") or policy_type is None:
        return False
    asyncio_module.set_event_loop_policy(policy_type())
    return True


def create_compatible_event_loop(
    asyncio_module: Any = asyncio,
    platform_name: str | None = None,
):
    """
    创建与本项目异步数据库依赖兼容的事件循环。

    `uvicorn` 0.46 在 Windows 非 reload 模式下会主动创建 Proactor 事件循环，
    这会绕过上面的全局策略设置。直接启动服务时把这个 factory 传给 uvicorn，
    可以确保 psycopg 使用 Selector 事件循环。
    """
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name.startswith("win"):
        selector_loop_type = getattr(asyncio_module, "SelectorEventLoop", None)
        if selector_loop_type is not None:
            return selector_loop_type()
    return asyncio_module.new_event_loop()


def get_uvicorn_loop_config(platform_name: str | None = None):
    """
    返回传给 uvicorn.run 的 loop 配置。

    非 Windows 继续交给 uvicorn 自动选择，避免影响 Linux/macOS 下的 uvloop 等优化。
    """
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name.startswith("win"):
        return create_compatible_event_loop
    return "auto"
