from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.services import retrieval_service


def test_run_vector_retrieval_fails_fast_when_milvus_health_probe_is_down() -> None:
    class FakeStore:
        def probe_health(self) -> bool:
            return False

        def last_health_error(self) -> str:
            return "health endpoint timeout"

    def fake_get_vector_store():  # noqa: ANN202
        return FakeStore()

    def fake_run_rewritten_search(store, queries, filter_expr):  # noqa: ANN001
        raise AssertionError("run_rewritten_search should not be called when health probe is down")

    settings = SimpleNamespace(rag_vector_query_timeout_seconds=12.0)

    try:
        asyncio.run(
            retrieval_service.run_vector_retrieval(
                "milvus",
                ["论文贡献"],
                "",
                settings=settings,
                get_vector_store=fake_get_vector_store,
                run_rewritten_search_fn=fake_run_rewritten_search,
            )
        )
    except Exception as exc:  # noqa: BLE001
        assert "Milvus health probe failed" in str(exc)
        assert "health endpoint timeout" in str(exc)
    else:
        raise AssertionError("expected run_vector_retrieval to fail fast when health probe is down")
