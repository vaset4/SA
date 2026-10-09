"""运行时能力导出。"""

from app.runtime.event_loop import (
    configure_windows_event_loop_policy,
    create_compatible_event_loop,
    get_uvicorn_loop_config,
)


def build_app_lifespan():
    """惰性导入 lifespan，避免导入期拉起完整 agent 依赖链。"""
    from app.runtime.lifespan import build_app_lifespan as _build_app_lifespan

    return _build_app_lifespan()


def get_runtime_container():
    """惰性导入运行时容器。"""
    from app.runtime.container import get_runtime_container as _get_runtime_container

    return _get_runtime_container()


def get_agent_runtime():
    """惰性导入 agent runtime getter。"""
    from app.runtime.container import get_agent_runtime as _get_agent_runtime

    return _get_agent_runtime()


def get_db_runtime():
    """惰性导入 db runtime getter。"""
    from app.runtime.container import get_db_runtime as _get_db_runtime

    return _get_db_runtime()


def get_conversation_runtime():
    """惰性导入 conversation runtime getter。"""
    from app.runtime.container import (
        get_conversation_runtime as _get_conversation_runtime,
    )

    return _get_conversation_runtime()


__all__ = [
    "build_app_lifespan",
    "configure_windows_event_loop_policy",
    "create_compatible_event_loop",
    "get_agent_runtime",
    "get_conversation_runtime",
    "get_db_runtime",
    "get_runtime_container",
    "get_uvicorn_loop_config",
]
