from __future__ import annotations

import app.services.session_mapping_service as session_mapping_service


def test_ensure_session_id_reuses_existing_value() -> None:
    assert session_mapping_service.ensure_session_id("sess-1") == "sess-1"


def test_ensure_session_id_generates_uuid_when_missing() -> None:
    session_id = session_mapping_service.ensure_session_id(None)

    assert isinstance(session_id, str)
    assert len(session_id) >= 32


def test_build_session_identity_keeps_conversation_boundary_explicit() -> None:
    payload = session_mapping_service.build_session_identity(
        session_id="sess-1",
        conversation_id="conv-1",
    )

    assert payload == {
        "session_id": "sess-1",
        "thread_id": "conv-1",
        "conversation_id": "conv-1",
    }
