"""旧缓存脚本只读预览的回归，作者：xxx。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.reingest_papers import ReingestDeps, ReingestOptions, default_deps, run_reingest


def test_dry_run_parses_cached_papers_without_storage_write(tmp_path):
    """离线预览返回真实分块统计，入口没有业务库或向量写入端口。"""
    uploads = _cached_paper(tmp_path, "paper-a")
    parse = Mock(return_value={"file_type": "pdf", "parser_mode": "mineru_online", "chunks": [
        {"chunk_id": "c1", "parent_id": "p", "section_path": "3.1 Setup"},
        {"chunk_id": "c2", "parent_id": "p", "section_path": "3.1 Setup"},
    ]})
    result = run_reingest(ReingestOptions(uploads, (), True), ReingestDeps(parse))
    assert result[0]["action"] == "dry_run"
    assert result[0]["chunk_count"] == 2
    assert result[0]["parent_count"] == 1
    assert result[0]["sect_key_distribution"] == {"num:3": 1}
    assert "chunks" not in result[0]


def test_old_write_mode_fails_before_parsing_or_deleting(tmp_path):
    """旧无参数写入不能先删除论文；必须转到有 owner 和批次的重建任务。"""
    parse = Mock(side_effect=AssertionError("write mode must stop before IO"))
    with pytest.raises(ValueError, match="独立批次核验发布"):
        run_reingest(ReingestOptions(tmp_path, ("paper",), False), ReingestDeps(parse))
    parse.assert_not_called()


def test_default_preview_reads_cached_markdown_without_remote_parser(tmp_path, monkeypatch):
    """真实默认预览端口直接读取缓存，不能触发 MinerU 或 LLM 请求。"""
    from app.services import document_parser
    from app.services.ingest import document_parse_service

    uploads = _cached_paper(tmp_path, "paper-a")
    remote_entry = Mock(side_effect=AssertionError("preview must remain offline"))
    parse = Mock(return_value=[{"chunk_id": "c", "parent_id": "p", "section_path": "3 Method"}])
    monkeypatch.setattr(document_parse_service, "parse_document_file", remote_entry)
    monkeypatch.setattr(document_parser, "parse_markdown_text", parse)
    result = run_reingest(ReingestOptions(uploads, (), True), default_deps())
    assert result[0]["chunk_count"] == 1
    assert parse.call_args.args == ("# Paper\n\nbody", "paper-a")
    remote_entry.assert_not_called()


def _cached_paper(tmp_path: Path, paper_id: str) -> Path:
    uploads = tmp_path / "uploads"
    artifact = uploads / f"{paper_id}.mineru"
    artifact.mkdir(parents=True)
    (uploads / f"{paper_id}.pdf").write_bytes(b"%PDF-1.4\n")
    (artifact / "source.md").write_text("# Paper\n\nbody", encoding="utf-8")
    (artifact / "content_list.json").write_text("[]", encoding="utf-8")
    (artifact / "manifest.json").write_text(
        json.dumps({"paper_id": paper_id, "status": "completed"}),
        encoding="utf-8",
    )
    return uploads
