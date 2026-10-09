"""migrate_chunks_v2 dry-run 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.has_collection.return_value = False
    client.get_collection_stats.return_value = {"row_count": 100}
    return client


def test_dry_run_no_v2_exists(mock_client):
    mock_client.has_collection.side_effect = lambda collection_name: (
        collection_name != "study_assistant_chunks_v2"
    )

    with patch("backend.scripts.migrate_chunks_v2.settings") as mock_settings:
        mock_settings.milvus_collection = "paper_chunks"
        mock_settings.milvus_collection_v2 = "study_assistant_chunks_v2"

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

        from scripts.migrate_chunks_v2 import dry_run
        report = dry_run(mock_client)

    assert report["v2_exists"] is False
    assert report["old_exists"] is True
    assert report["old_count"] == 100
    assert "create v2" in report["action"]


def test_dry_run_v2_already_exists(mock_client):
    mock_client.has_collection.return_value = True

    with patch("backend.scripts.migrate_chunks_v2.settings") as mock_settings:
        mock_settings.milvus_collection = "paper_chunks"
        mock_settings.milvus_collection_v2 = "study_assistant_chunks_v2"

        from scripts.migrate_chunks_v2 import dry_run
        report = dry_run(mock_client)

    assert report["v2_exists"] is True
    assert "skip" in report["action"]
