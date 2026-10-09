"""HuggingFace 本地缓存解析辅助。"""

from __future__ import annotations

import os
from pathlib import Path

from app.config import get_settings

_WINDOWS_FALLBACK_HOME = Path("D:/hf_cache")


def configure_offline_env() -> None:
    """无参数；本地模型加载前应用缓存与离线配置，API 路径不调用此方法。"""
    settings = get_settings()
    # 1. 【模型服务】【本地缓存】只在本地模型边界向依赖传入本副本的缓存目录
    for key, value in (("HF_HOME", settings.hf_home), ("HUGGINGFACE_HUB_CACHE", settings.huggingface_hub_cache)):
        if value:
            os.environ.setdefault(key, value)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # 某些本地抓包/调试环境会注入不可写的 SSLKEYLOGFILE，aiohttp/ssl 导入时会直接失败。
    os.environ.pop("SSLKEYLOGFILE", None)


def get_hub_cache_dir() -> Path | None:
    """无参数；读取显式进程值或本副本配置，返回 HuggingFace Hub 缓存根目录。"""
    settings = get_settings()
    explicit_hub = os.environ.get("HUGGINGFACE_HUB_CACHE", settings.huggingface_hub_cache).strip()
    if explicit_hub:
        return Path(explicit_hub)

    hf_home = os.environ.get("HF_HOME", settings.hf_home).strip()
    if hf_home:
        return Path(hf_home) / "hub"

    if _WINDOWS_FALLBACK_HOME.exists():
        return _WINDOWS_FALLBACK_HOME / "hub"

    return None


def resolve_local_model_source(model_name: str) -> str:
    """
    优先把 HuggingFace repo id 解析成真实本地 snapshot 目录。

    若本地缓存不存在，则回退成原始 model_name，让调用方自行处理异常。
    """
    raw = (model_name or "").strip()
    if not raw:
        return raw

    candidate_path = Path(raw)
    if candidate_path.exists():
        return str(candidate_path)

    hub_cache_dir = get_hub_cache_dir()
    if hub_cache_dir is None:
        return raw

    repo_cache_dir = hub_cache_dir / f"models--{raw.replace('/', '--')}"
    if not repo_cache_dir.exists():
        return raw

    ref_snapshot = _resolve_ref_snapshot(repo_cache_dir)
    if ref_snapshot is not None:
        return str(ref_snapshot)

    latest_snapshot = _resolve_latest_snapshot(repo_cache_dir)
    if latest_snapshot is not None:
        return str(latest_snapshot)

    return raw


def _resolve_ref_snapshot(repo_cache_dir: Path) -> Path | None:
    """优先按 refs/main 指向的 snapshot 解析本地目录。"""
    ref_file = repo_cache_dir / "refs" / "main"
    if not ref_file.exists():
        return None
    commit_hash = ref_file.read_text(encoding="utf-8").strip()
    if not commit_hash:
        return None
    snapshot_dir = repo_cache_dir / "snapshots" / commit_hash
    if _is_usable_snapshot_dir(snapshot_dir):
        return snapshot_dir
    return None


def _resolve_latest_snapshot(repo_cache_dir: Path) -> Path | None:
    """refs 缺失时，回退到最新可用 snapshot。"""
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = [
        item
        for item in snapshots_dir.iterdir()
        if item.is_dir() and _is_usable_snapshot_dir(item)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _is_usable_snapshot_dir(snapshot_dir: Path) -> bool:
    """判断 snapshot 目录是否具备最基本的本地加载文件。"""
    if not snapshot_dir.exists():
        return False
    required_any = [
        snapshot_dir / "config.json",
        snapshot_dir / "modules.json",
        snapshot_dir / "tokenizer.json",
    ]
    return any(path.exists() for path in required_any)
