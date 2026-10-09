"""从当前磁盘源码导出独立公开目录，不复制私人运行数据。作者：xxx。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

MANIFEST_NAME = "public-manifest.json"
EXPLICIT_FILES = (
    "LICENSE", "THIRD_PARTY_NOTICES.md", "docs/pdf-processing.md",
    "README.md", "DESIGN.md", ".env.example", ".gitignore", ".gitattributes",
    "docker-compose.yml", "docker-compose.local-models.yml",
    "backend/Dockerfile", "backend/requirements.txt", "backend/requirements-local.txt",
    "backend/requirements-dev.txt", "backend/.gitignore",
    "frontend/package.json", "frontend/package-lock.json", "frontend/.npmrc",
    "frontend/postcss.config.js", "frontend/tailwind.config.js",
    "frontend/docs/FRONTEND_UI_GUIDE.md",
    "docs/README.md", "docs/deployment-security.md", "docs/model-api.md",
    "docs/evaluation.md", "scripts/export_public_repo.py",
    "backend/scripts/check_chunks_consistency.py", "backend/scripts/migrate_chunks_v2.py",
    "backend/scripts/reingest_papers.py", "backend/scripts/reingest_lightrag_failed.py",
    "backend/scripts/prepare_chunk_tokenizer.py",
)
SOURCE_TREES = {
    "docs/licenses": frozenset({".txt", ".md", ".json"}),
    "backend/app": frozenset({".py"}),
    "backend/tests": frozenset({".py"}),
    "backend/evals": frozenset({".py"}),
    **{f"frontend/{name}": frozenset({".js", ".jsx", ".mjs", ".css"}) for name in (
        "app", "components", "config", "lib", "stores", "styles", "utils",
    )},
}
PUBLIC_FIXTURES = tuple(
    f"backend/tests/eval/fixtures/{family}/{intent}.jsonl"
    for family in ("intent_samples", "synthesis_samples")
    for intent in ("lookup", "compare", "explore")
)
EXCLUDED_FILES = {
    **{f"backend/tests/unit/{name}.py": "Depends on private historical research artifacts" for name in (
        "test_agent_outcome_contracts", "test_claim_draft_isolation",
        "test_llm_structured_error_locations", "test_llm_structured_error_location_review",
        "test_research_action_prompt_projection", "test_research_claim_scope_contract",
        "test_research_decision_output", "test_research_delivery_feedback",
        "test_research_review_live_contract", "test_research_opt_dataset",
        "test_ragas_private_dataset",
    )},
    "backend/tests/unit/test_chunking_report_cached_papers.py": "Requires private cached MinerU uploads",
    "backend/test_phase3_reliability.py": "Historical tests import the retired app.nodes.self_rag module",
    "backend/evals/suites/datasets/research_opt_dataset.py": "Private paper corpus and historical builder",
}
EXCLUDED_DIRECTORIES = frozenset({
    "__pycache__", ".git", ".trellis", ".codex", ".agents", ".claude", ".codegraph",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules", ".next", ".venv",
    "runs", "experiments", "outputs", "checkpoints", "datasets", "profiles",
})
EXCLUDED_CATEGORIES = (
    "Environment files and backups except the reviewed .env.example",
    "Model weights, uploaded documents, databases, caches and build artifacts",
    "Private evaluation datasets/profiles, run results and source snapshots",
    "AI/editor histories, internal plans, external checkouts and original Git metadata",
    "Unlisted files: export uses the explicit source/resource allowlist in this script",
)


def reject_link(path: Path) -> None:
    """参数为待检查路径；若为链接或 Windows 重解析点则报错，不跟随外部内容。"""
    if path.is_symlink():
        raise ValueError(f"【公开仓库】【链接检查】拒绝链接: {path}")
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"【公开仓库】【链接检查】拒绝重解析点: {path}")


def checked_source_path(source: Path, relative: str) -> Path:
    """参数为源码根目录及白名单路径；逐层拒绝链接，返回未越界的候选路径。"""
    current = source
    for part in Path(relative).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            reject_link(current)
    return current


def checked_destination(source_root: Path, destination: Path) -> tuple[Path, Path]:
    """参数为源码根目录和目标；返回核对后的绝对路径，拒绝越界、源码覆盖和非空目标。"""
    source = source_root.absolute()
    for ancestor in (*reversed(source.parents), source):
        reject_link(ancestor)
    source = source.resolve(strict=True)
    target = destination if destination.is_absolute() else source / destination
    target = Path(os.path.abspath(target))
    if not target.is_relative_to(source) or target == source:
        raise ValueError("【公开仓库】【目标检查】目标必须位于源码工作区内部")
    relative = target.relative_to(source)
    if relative.parts[0].casefold() in {"backend", "frontend", "docs", "scripts"}:
        raise ValueError("【公开仓库】【目标检查】目标不能位于源码目录内")
    current = source
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            reject_link(current)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("【公开仓库】【目标检查】目标必须不存在或为空，不覆盖已有文件")
    return source, target


def collect_files(source: Path) -> tuple[str, ...]:
    """参数为源码根目录；返回排序且去重的公开相对文件名，保留未跟踪源码。"""
    selected: set[str] = set()
    # 1. 配置、文档和人工样例仅按准确文件名公开
    for relative in (*EXPLICIT_FILES, *PUBLIC_FIXTURES):
        path = checked_source_path(source, relative)
        if path.exists() or path.is_symlink():
            reject_link(path)
            if not path.is_file():
                raise ValueError(f"【公开仓库】【文件检查】预期普通文件: {relative}")
            selected.add(relative)
    # 2. 模块目录只收录源码扩展名，避免运行产物混入包目录
    trees = {**SOURCE_TREES, "backend": frozenset()}
    for relative, extensions in trees.items():
        root = checked_source_path(source, relative)
        if not root.exists():
            continue
        reject_link(root)
        for directory, folders, files in os.walk(root, followlinks=False):
            for name in (*folders, *files):
                reject_link(Path(directory) / name)
            # 1. 【公开仓库】【许可保留】许可原文保留包内目录名称，只复制审核过的文本后缀
            folders[:] = sorted(name for name in folders if name not in EXCLUDED_DIRECTORIES or relative == "docs/licenses")
            if relative == "backend":
                folders[:] = []
            for name in files:
                path = Path(directory) / name
                key = path.relative_to(source).as_posix()
                allowed = path.suffix in extensions or (relative == "backend" and name.startswith("test_") and path.suffix == ".py")
                if allowed and key not in EXCLUDED_FILES:
                    selected.add(key)
    return tuple(sorted(selected))


def file_digest(path: Path) -> str:
    """参数为普通文件；流式读取并返回 SHA-256，避免整文件进入内存。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_manifest(source: Path, files: tuple[str, ...]) -> dict:
    """参数为源码根目录及相对路径集合；返回确定性的文件哈希和公开边界清单。"""
    return {
        "schema_version": 1,
        "manifest_note": "This generated manifest is metadata and does not hash itself.",
        "files": [{"path": name, "sha256": file_digest(source / name),
                   "size": (source / name).stat().st_size} for name in files],
        "excluded_files": dict(sorted(EXCLUDED_FILES.items())),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
    }


def export_repository(source_root: Path, destination: Path, *, dry_run: bool = False) -> dict:
    """参数为源码、目标及预览标志；返回清单，正式导出逐字节复制且验证哈希。"""
    source, target = checked_destination(source_root, destination)
    manifest = build_manifest(source, collect_files(source))
    if dry_run:
        return manifest
    # 1. 全部路径检查完成后再创建目标，源文件不会被改写
    target.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        original, copied = source / item["path"], target / item["path"]
        reject_link(original)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
        if file_digest(copied) != item["sha256"]:
            raise RuntimeError(f"【公开仓库】【复制校验】源码在导出过程中变化: {item['path']}")
    # 2. 清单自身不参与哈希，避免自引用；失败会显式保留部分目标供检查
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> int:
    """无调用参数；读取 CLI 参数，输出 JSON 清单，失败时保留异常并返回非零状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, default=Path("public-release"))
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    manifest = export_repository(options.source_root, options.destination, dry_run=options.dry_run)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
