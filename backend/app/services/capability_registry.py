"""能力注册层快照。"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.services.retrieval_provider_registry import (
    get_default_vector_provider_name,
    get_graph_provider_name,
    list_retrieval_providers,
)
from app.services.tool_registry import list_registered_tools

# 这个模块不真正“启用能力”，而是把当前配置和注册表整理成可展示快照。
# 常见用途是 health/system 页面、调试页、能力探测接口。


def list_model_capabilities() -> list[dict[str, Any]]:
    """列出当前模型能力；当前默认只有一套 openai-compatible LLM 配置。"""
    settings = get_settings()
    configured = bool(getattr(settings, "llm_api_key", "") and getattr(settings, "llm_base_url", ""))
    return [
        {
            "name": getattr(settings, "llm_model_name", ""),
            "provider": "openai-compatible",
            "configured": configured,
            "temperature": getattr(settings, "llm_temperature", 0.0),
            "timeout": getattr(settings, "llm_timeout", 0),
            "base_url": getattr(settings, "llm_base_url", ""),
        }
    ]


def list_retrieval_capabilities() -> list[dict[str, Any]]:
    """列出当前检索能力。"""
    return list_retrieval_providers()


def list_tool_capabilities() -> list[dict[str, Any]]:
    """列出当前工具能力；会额外标记 Tavily 这类依赖外部 key 的工具是否已配置。"""
    settings = get_settings()
    tools = []
    for item in list_registered_tools(enabled_only=False):
        tool_item = dict(item)
        if item["name"] == "tavily_search":
            tool_item["configured"] = bool(getattr(settings, "tavily_api_key", ""))
        else:
            tool_item["configured"] = True
        tools.append(tool_item)
    return tools


def capability_snapshot() -> dict[str, Any]:
    """输出当前启用能力快照。"""
    settings = get_settings()
    return {
        "models": list_model_capabilities(),
        "retrieval": {
            "default_vector_provider": get_default_vector_provider_name(),
            "graph_provider": get_graph_provider_name(),
            "providers": list_retrieval_capabilities(),
        },
        "tools": list_tool_capabilities(),
        "config": {
            "vector_backend": getattr(settings, "vector_backend", "milvus"),
            "graph_rag_enabled": bool(getattr(settings, "graph_rag_enabled", False)),
            "tavily_configured": bool(getattr(settings, "tavily_api_key", "")),
        },
    }
