"""论文受控上传文件与 MinerU 产物清理服务。

Author: xxx
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.config import get_settings


def _uploads_root(data_dir: str | Path | None = None) -> Path:
    """
    解析 DATA_DIR 下的受控 uploads 目录。

    Args:
        data_dir: 可选 DATA_DIR 覆盖值
    Returns:
        uploads 目录绝对路径
    """
    root = Path(data_dir or get_settings().data_dir).resolve(strict=False)
    return (root / "uploads").resolve(strict=False)


def _resolve_upload_file(raw_path: str, uploads_root: Path) -> Path:
    """
    将数据库中的上传文件路径解析为绝对路径。

    Args:
        raw_path: papers.pdf_path 保存的文件路径
        uploads_root: DATA_DIR 下的受控 uploads 目录
    Returns:
        解析后的绝对文件路径
    """
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    cwd_candidate = candidate.resolve(strict=False)
    if cwd_candidate == uploads_root or uploads_root in cwd_candidate.parents:
        return cwd_candidate
    return (uploads_root / candidate).resolve(strict=False)


def _require_managed_path(path: Path, uploads_root: Path) -> None:
    """
    校验目标位于 DATA_DIR/uploads 内且不是 uploads 根目录。

    Args:
        path: 待删除目标的绝对路径
        uploads_root: 受控上传目录绝对路径
    Returns:
        无返回值；越界时抛出 ValueError
    """
    if path == uploads_root or uploads_root not in path.parents:
        raise ValueError(f"论文文件不在 DATA_DIR 受控上传目录内: {path}")


def delete_managed_paper_files(
    paper_id: str,
    *,
    pdf_path: str | None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    幂等删除论文原文件和确定性命名的 MinerU 产物目录。

    Args:
        paper_id: 论文主键，用于定位 MinerU 产物目录
        pdf_path: 业务库保存的原文件路径
        data_dir: 可选 DATA_DIR 覆盖值，测试使用
    Returns:
        删除数量和删除对象类型
    """
    normalized_paper_id = paper_id.strip()
    if not normalized_paper_id:
        raise ValueError("paper_id 不能为空")
    uploads_root = _uploads_root(data_dir)
    targets: list[tuple[str, Path]] = []
    if str(pdf_path or "").strip():
        targets.append(("uploaded_file", _resolve_upload_file(str(pdf_path), uploads_root)))
    targets.append(("mineru_artifacts", (uploads_root / f"{normalized_paper_id}.mineru").resolve(strict=False)))
    targets.append(("upload_versions", (uploads_root / normalized_paper_id).resolve(strict=False)))

    # 1. 【论文删除】【受管路径】先核对全部真实目标，再开始删除，越界目标不能造成部分文件先被删除
    for _, target in targets:
        _require_managed_path(target, uploads_root)
    deleted_kinds: list[str] = []
    for kind, target in targets:
        if not target.exists():
            continue
        if kind == "uploaded_file":
            if not target.is_file():
                raise ValueError(f"论文原文件路径不是文件: {target}")
            target.unlink()
        else:
            if not target.is_dir():
                raise ValueError(f"MinerU 产物路径不是目录: {target}")
            shutil.rmtree(target)
        deleted_kinds.append(kind)
    return {"deleted_count": len(deleted_kinds), "deleted_kinds": deleted_kinds}


def list_orphan_managed_paths(
    papers: list[dict[str, Any]],
    *,
    data_dir: str | Path | None = None,
) -> list[str]:
    """
    枚举未被论文生命周期记录引用的受控上传文件和 MinerU 目录。

    Args:
        papers: 业务库中的全量论文生命周期记录
        data_dir: 可选 DATA_DIR 覆盖值
    Returns:
        排序后的孤儿绝对路径
    """
    uploads_root = _uploads_root(data_dir)
    if not uploads_root.exists():
        return []
    valid_paths: set[Path] = set()
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "").strip()
        raw_pdf_path = str(paper.get("pdf_path") or "").strip()
        if raw_pdf_path:
            pdf_path = _resolve_upload_file(raw_pdf_path, uploads_root)
            if pdf_path != uploads_root and uploads_root in pdf_path.parents:
                valid_paths.add(pdf_path)
        if paper_id:
            valid_paths.add((uploads_root / f"{paper_id}.mineru").resolve(strict=False))
    candidates: list[Path] = []
    for child in uploads_root.iterdir():
        if child.is_file() or (child.is_dir() and child.name.endswith(".mineru")):
            candidate = child.resolve(strict=False)
            _require_managed_path(candidate, uploads_root)
            candidates.append(candidate)
    return sorted(str(path) for path in candidates if path not in valid_paths)


def delete_managed_orphan_paths(
    paths: list[str],
    *,
    data_dir: str | Path | None = None,
) -> list[str]:
    """
    精确删除 dry-run 已枚举的受控孤儿路径。

    Args:
        paths: 待删除的受控绝对路径
        data_dir: 可选 DATA_DIR 覆盖值
    Returns:
        实际删除的路径
    """
    uploads_root = _uploads_root(data_dir)
    deleted_paths: list[str] = []
    for raw_path in paths:
        target = Path(raw_path).resolve(strict=False)
        _require_managed_path(target, uploads_root)
        if not target.exists():
            continue
        if target.is_file():
            target.unlink()
        elif target.is_dir() and target.name.endswith(".mineru"):
            shutil.rmtree(target)
        else:
            raise ValueError(f"孤儿路径不是受控上传文件或 MinerU 目录: {target}")
        deleted_paths.append(str(target))
    return deleted_paths
