"""检索 provider 注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from app.config import get_settings

ProviderKind = Literal["vector", "graph"]

# 这个注册表的价值不在“动态插件系统”，而在于给检索编排层提供一个稳定抽象：
#   - `retrieval_service.py` 不必硬编码 import Milvus / LightRAG
#   - 测试可以替换 factory
#   - 默认 provider 的选择由配置决定，而不是散落在业务代码里


@dataclass(frozen=True)
class RetrievalProviderSpec:
    """描述一个已知检索 provider。"""

    name: str
    kind: ProviderKind
    factory: Callable[[], Any]
    description: str
    tags: tuple[str, ...] = ()


def get_milvus_store() -> Any:
    """延迟导入 Milvus provider，避免模块导入期加载重依赖。"""
    from app.services.milvus_store import get_milvus_store as _get_milvus_store

    return _get_milvus_store()


def get_vector_store() -> Any:
    """延迟导入 FAISS provider。"""
    from app.services.vector_store import get_vector_store as _get_vector_store

    return _get_vector_store()


def get_lightrag_service() -> Any:
    """延迟导入 LightRAG provider，避免测试收集期提前加载。"""
    from app.services.lightrag_service import get_lightrag_service as _get_lightrag_service

    return _get_lightrag_service()


def _provider_specs() -> dict[str, RetrievalProviderSpec]:
    """按当前模块函数构造 provider 规格，方便测试替换 factory。"""
    return {
        "milvus": RetrievalProviderSpec(
            name="milvus",
            kind="vector",
            factory=get_milvus_store,
            description="Milvus 在线混合检索主链",
            tags=("vector", "online", "default"),
        ),
        "faiss": RetrievalProviderSpec(
            name="faiss",
            kind="vector",
            factory=get_vector_store,
            description="本地 FAISS 兼容向量检索",
            tags=("vector", "offline", "compat"),
        ),
        "lightrag": RetrievalProviderSpec(
            name="lightrag",
            kind="graph",
            factory=get_lightrag_service,
            description="LightRAG 图检索增强层",
            tags=("graph", "lightrag"),
        ),
    }


def _is_enabled(spec: RetrievalProviderSpec) -> bool:
    """根据当前配置判断 provider 是否启用。"""
    settings = get_settings()
    if spec.kind == "graph":
        return bool(getattr(settings, "graph_rag_enabled", False))
    return True


def _describe(spec: RetrievalProviderSpec) -> dict[str, Any]:
    """把 provider 规格转换成可序列化描述。"""
    settings = get_settings()
    is_default = False
    if spec.kind == "vector":
        is_default = spec.name == getattr(settings, "vector_backend", "milvus")
    elif spec.kind == "graph":
        is_default = spec.name == "lightrag" and bool(getattr(settings, "graph_rag_enabled", False))
    return {
        "name": spec.name,
        "kind": spec.kind,
        "enabled": _is_enabled(spec),
        "default": is_default,
        "description": spec.description,
        "tags": list(spec.tags),
    }


def get_provider_spec(name: str) -> RetrievalProviderSpec:
    """按名称获取 provider 规格。"""
    try:
        return _provider_specs()[name]
    except KeyError as exc:
        raise LookupError(f"未知检索 provider: {name}") from exc


def get_provider(name: str, *, allow_disabled: bool = False) -> Any:
    """按名称获取 provider 实例；默认会阻止拿到“已知但当前未启用”的 provider。"""
    spec = get_provider_spec(name)
    if not allow_disabled and not _is_enabled(spec):
        raise RuntimeError(f"检索 provider 未启用: {name}")
    return spec.factory()


def get_default_vector_provider_name() -> str:
    """返回默认向量 provider 名称；这里会顺带校验配置没有把 graph provider 误配进来。"""
    settings = get_settings()
    name = getattr(settings, "vector_backend", "milvus")
    spec = get_provider_spec(name)
    if spec.kind != "vector":
        raise RuntimeError(f"默认向量 provider 配置非法: {name}")
    return name


def get_default_vector_provider() -> Any:
    """返回默认向量 provider 实例。"""
    return get_provider(get_default_vector_provider_name())


def get_graph_provider_name() -> str | None:
    """返回默认图检索 provider 名称；当前实现固定映射到 LightRAG。"""
    spec = get_provider_spec("lightrag")
    return spec.name if _is_enabled(spec) else None


def get_graph_provider() -> Any | None:
    """返回默认图检索 provider 实例。"""
    name = get_graph_provider_name()
    if name is None:
        return None
    return get_provider(name)


def list_enabled_retrieval_providers(kind: ProviderKind | None = None) -> list[dict[str, Any]]:
    """列出当前已启用 provider；用于健康页、调试页或能力探测。"""
    providers: list[dict[str, Any]] = []
    for spec in _provider_specs().values():
        if kind is not None and spec.kind != kind:
            continue
        if _is_enabled(spec):
            providers.append(_describe(spec))
    return providers


def list_retrieval_providers(kind: ProviderKind | None = None) -> list[dict[str, Any]]:
    """列出全部已知 provider；与上一个函数不同，这里会保留 disabled 项。"""
    providers: list[dict[str, Any]] = []
    for spec in _provider_specs().values():
        if kind is not None and spec.kind != kind:
            continue
        providers.append(_describe(spec))
    return providers
