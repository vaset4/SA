"""公开源码导出的文件边界、路径安全及一致性测试。作者：xxx。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("public_export", ROOT / "scripts/export_public_repo.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def _source(tmp_path: Path) -> Path:
    """参数为测试临时根目录；创建包含源码和私人产物的人工工作区并返回路径。"""
    root = tmp_path / "source"
    files = {
        "README.md": "public documentation",
        "LICENSE": "MIT License",
        "THIRD_PARTY_NOTICES.md": "Third-party notices",
        "docs/pdf-processing.md": "PDF usage",
        "docs/licenses/README.md": "License provenance",
        "docs/licenses/component/LICENSE.txt": "Complete component terms",
        "docs/licenses/component/node_modules/nested/LICENSE.txt": "Nested bundled component terms",
        "docs/licenses/private.key": "not a reviewed license resource",
        "DESIGN.md": "public product design required by frontend tests",
        ".env.example": "EMBEDDING_API_KEY=\n",
        ".env": "private-credential",
        "backend/app/models/paper.py": "new untracked source",
        "backend/app/__pycache__/paper.pyc": "generated cache",
        "backend/test_current.py": "portable root test",
        "backend/test_phase3_reliability.py": "historical test imports retired self_rag",
        "backend/tests/unit/test_current.py": "portable unit test",
        "backend/tests/unit/test_agent_outcome_contracts.py": "private historical test",
        "backend/tests/unit/test_chunking_report_cached_papers.py": "requires private cached uploads",
        "backend/evals/suites/run_suite.py": "generic evaluation code",
        "backend/evals/suites/datasets/research_opt_dataset.py": "private builder",
        "backend/evals/suites/runs/private.json": "private result",
        "backend/evals/ragas/datasets/private.jsonl": "private dataset",
        "backend/tests/eval/fixtures/intent_samples/lookup.jsonl": "{}\n",
        "frontend/app/page.jsx": "current frontend",
        "frontend/node_modules/secret.js": "dependency cache",
        "models/weights.bin": "private model",
        ".git/config": "old repository metadata",
        "public-release/old.txt": "old release is not input",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_export_copies_current_source_and_excludes_private_artifacts(tmp_path: Path) -> None:
    """参数为临时目录；验证公开清单、未跟踪源码、输入不变及确定性哈希，无返回值。"""
    source = _source(tmp_path)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    preview = EXPORT.export_repository(source, Path("release-new"), dry_run=True)
    assert not (source / "release-new").exists()
    manifest = EXPORT.export_repository(source, Path("release-new"))
    assert preview == manifest
    selected = {item["path"] for item in manifest["files"]}
    assert selected == {
        "LICENSE", "THIRD_PARTY_NOTICES.md", "docs/pdf-processing.md",
        "docs/licenses/README.md", "docs/licenses/component/LICENSE.txt",
        "docs/licenses/component/node_modules/nested/LICENSE.txt",
        "README.md", "DESIGN.md", ".env.example", "backend/app/models/paper.py",
        "backend/test_current.py", "backend/tests/unit/test_current.py",
        "backend/evals/suites/run_suite.py", "frontend/app/page.jsx",
        "backend/tests/eval/fixtures/intent_samples/lookup.jsonl",
    }
    for item in manifest["files"]:
        copied = source / "release-new" / item["path"]
        assert copied.read_bytes() == (source / item["path"]).read_bytes()
        assert EXPORT.file_digest(copied) == item["sha256"]
    assert all(path.read_bytes() == data for path, data in before.items())
    persisted = json.loads((source / "release-new/public-manifest.json").read_text())
    assert persisted == manifest


@pytest.mark.parametrize("destination", ["..", ".", "backend/new", "BACKEND/new", "frontend/new", "docs/new", "scripts/new"])
def test_export_rejects_outside_and_source_destinations(tmp_path: Path, destination: str) -> None:
    """参数为临时目录及危险目标；验证拒绝越界或向源码树写入，无返回值。"""
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="目标"):
        EXPORT.export_repository(source, Path(destination))


def test_export_refuses_nonempty_target_without_changing_it(tmp_path: Path) -> None:
    """参数为临时目录；验证已有公开目录不会被覆盖或清空，无返回值。"""
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="不存在或为空"):
        EXPORT.export_repository(source, Path("public-release"))
    assert (source / "public-release/old.txt").read_text() == "old release is not input"


@pytest.mark.parametrize("relative", ["frontend", "backend/app/models", "release-parent"])
def test_export_rejects_reparse_ancestors(tmp_path: Path, monkeypatch, relative: str) -> None:
    """参数为临时目录、替换工具和重解析路径；验证源/目标祖先不允许重解析，无返回值。"""
    source = _source(tmp_path)
    reparse = source / relative
    reparse.mkdir(parents=True, exist_ok=True)
    original = Path.lstat

    def fake_lstat(path: Path):
        """参数为待检查路径；仅给人工重解析路径附加 Windows 文件属性并返回元数据。"""
        result = original(path)
        if path == reparse:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="重解析点"):
        EXPORT.export_repository(source, Path("release-parent/output"))


def test_export_detects_source_change_during_copy(tmp_path: Path, monkeypatch) -> None:
    """参数为临时目录和替换工具；验证复制期间字节变化会失败且不生成成功清单。"""
    source = _source(tmp_path)
    original = EXPORT.shutil.copyfile

    def changed_copy(src: Path, dst: Path):
        """参数为测试源和目标；模拟复制内容变化，返回复制目标路径。"""
        result = original(src, dst)
        dst.write_text("changed while copying", encoding="utf-8")
        return result

    monkeypatch.setattr(EXPORT.shutil, "copyfile", changed_copy)
    with pytest.raises(RuntimeError, match="导出过程中变化"):
        EXPORT.export_repository(source, Path("release-new"))
    assert not (source / "release-new/public-manifest.json").exists()


def test_export_rejects_source_root_beneath_reparse_parent(tmp_path: Path, monkeypatch) -> None:
    """参数为临时目录和替换工具；验证源码根的祖先链接不会被 resolve 静默跟随。"""
    source = _source(tmp_path)
    original = Path.lstat

    def fake_lstat(path: Path):
        """参数为待检查路径；标记源码父目录为重解析点并返回文件元数据。"""
        result = original(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="重解析点"):
        EXPORT.export_repository(source, Path("release-new"), dry_run=True)
