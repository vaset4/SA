from __future__ import annotations

from pathlib import Path

import app.services.hf_local as hf_local_module


def test_resolve_local_model_source_prefers_ref_snapshot(monkeypatch, tmp_path) -> None:
    hub_dir = tmp_path / "hub"
    repo_dir = hub_dir / "models--BAAI--bge-m3"
    snapshot_dir = repo_dir / "snapshots" / "commit-123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("commit-123", encoding="utf-8")

    monkeypatch.setenv("HF_HOME", str(tmp_path))

    resolved = hf_local_module.resolve_local_model_source("BAAI/bge-m3")

    assert Path(resolved) == snapshot_dir


def test_resolve_local_model_source_falls_back_to_latest_snapshot(monkeypatch, tmp_path) -> None:
    hub_dir = tmp_path / "hub"
    repo_dir = hub_dir / "models--BAAI--bge-m3"
    snapshots_dir = repo_dir / "snapshots"
    older = snapshots_dir / "old"
    newer = snapshots_dir / "new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "config.json").write_text("{}", encoding="utf-8")
    (newer / "config.json").write_text("{}", encoding="utf-8")
    newer.touch()

    monkeypatch.setenv("HF_HOME", str(tmp_path))

    resolved = hf_local_module.resolve_local_model_source("BAAI/bge-m3")

    assert Path(resolved) == newer


def test_resolve_local_model_source_returns_original_when_cache_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    resolved = hf_local_module.resolve_local_model_source("BAAI/bge-m3")

    assert resolved == "BAAI/bge-m3"


def test_configure_offline_env_clears_sslkeylogfile(monkeypatch) -> None:
    monkeypatch.setenv("SSLKEYLOGFILE", r"D:\ssl_key\ssl_key.log")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    hf_local_module.configure_offline_env()

    assert hf_local_module.os.environ["HF_HUB_OFFLINE"] == "1"
    assert hf_local_module.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert "SSLKEYLOGFILE" not in hf_local_module.os.environ
