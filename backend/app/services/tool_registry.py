"""项目内轻量工具注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from langgraph.prebuilt import ToolNode


@dataclass
class ToolRegistration:
    """描述一个已注册工具。"""

    name: str
    tool: Any
    enabled: bool = True
    tags: tuple[str, ...] = ()


_TOOL_REGISTRY: dict[str, ToolRegistration] = {}

# 这个注册表和 retrieval_provider_registry 的角色类似：
#   - 都是把“可选能力”集中登记
#   - 但这里登记的是 LangGraph ToolNode 可消费的工具对象
#   - 而不是向量/图检索 provider


def _normalize_tool_name(name: str | None, tool: Any) -> str:
    """解析工具名，优先使用显式 name。"""
    if name:
        return name
    candidate = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("工具必须显式提供名称，或对象上带有 name/__name__")
    return candidate


def register_tool(
    name: str | None,
    tool: Any,
    *,
    enabled: bool = True,
    tags: Iterable[str] | None = None,
) -> ToolRegistration:
    """注册或覆盖一个工具；后注册同名工具会覆盖旧定义。"""
    tool_name = _normalize_tool_name(name, tool)
    registration = ToolRegistration(
        name=tool_name,
        tool=tool,
        enabled=enabled,
        tags=tuple(tags or ()),
    )
    _TOOL_REGISTRY[tool_name] = registration
    return registration


def get_registered_tool(name: str, *, allow_disabled: bool = False) -> ToolRegistration:
    """按名称获取工具注册项。"""
    try:
        registration = _TOOL_REGISTRY[name]
    except KeyError as exc:
        raise LookupError(f"未知工具: {name}") from exc
    if not allow_disabled and not registration.enabled:
        raise RuntimeError(f"工具未启用: {name}")
    return registration


def list_registered_tools(*, enabled_only: bool = True) -> list[dict[str, Any]]:
    """列出当前注册工具。"""
    items: list[dict[str, Any]] = []
    for registration in _TOOL_REGISTRY.values():
        if enabled_only and not registration.enabled:
            continue
        items.append(
            {
                "name": registration.name,
                "enabled": registration.enabled,
                "tags": list(registration.tags),
            }
        )
    return items


def build_tool_node(
    names: list[str] | None = None,
    *,
    node_name: str = "tools",
) -> ToolNode:
    """
    按注册表构造 ToolNode。

    这一步相当于把“普通 Python 工具对象列表”提升成 LangGraph 可挂载的工具节点。
    """
    registrations: list[ToolRegistration] = []
    if names is None:
        registrations = [
            registration
            for registration in _TOOL_REGISTRY.values()
            if registration.enabled
        ]
    else:
        for name in names:
            registrations.append(get_registered_tool(name))
    return ToolNode([registration.tool for registration in registrations], name=node_name)
