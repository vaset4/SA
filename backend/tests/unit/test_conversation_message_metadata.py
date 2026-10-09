"""Conversation message metadata persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from app.repositories.conversation_repository import _message_data


def test_message_data_merges_metadata_and_preserves_base_columns() -> None:
    row = {
        "id": "msg-1",
        "conversation_id": "conv-1",
        "role": "assistant",
        "content": "回答",
        "title": "内部研究回答",
        "tone": None,
        "render_mode": "markdown",
        "metadata": {
            "content": "metadata 不应覆盖正文",
            "synthesis_state": {"status": "ready"},
            "evidence_records": [{"layer": "milvus"}],
            "trace_summary": {"source_status": {"milvus": "ok"}},
        },
        "created_at": datetime(2026, 5, 26, tzinfo=timezone.utc),
    }

    message = _message_data(row)

    assert message["content"] == "回答"
    assert message["synthesis_state"]["status"] == "ready"
    assert message["evidence_records"][0]["layer"] == "milvus"
    assert message["trace_summary"]["source_status"]["milvus"] == "ok"
    assert "metadata" not in message
