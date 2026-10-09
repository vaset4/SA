from __future__ import annotations

from types import SimpleNamespace

from fastapi import Request

from app.services.rate_limit_service import resolve_rate_limit_rule


def _request(method: str, path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "client": ("127.0.0.1", 8000),
            "server": ("testserver", 80),
        }
    )


def test_task_polling_routes_use_relaxed_rate_limit() -> None:
    settings = SimpleNamespace(rate_limit="60/minute")

    detail_rule = resolve_rate_limit_rule(_request("GET", "/api/v1/tasks/task-1"), settings)
    events_rule = resolve_rate_limit_rule(_request("GET", "/api/v1/tasks/task-1/events"), settings)

    assert detail_rule == (180, 60.0)
    assert events_rule == (180, 60.0)


def test_non_task_routes_keep_default_rate_limit() -> None:
    settings = SimpleNamespace(rate_limit="60/minute")

    rule = resolve_rate_limit_rule(_request("GET", "/api/v1/projects"), settings)

    assert rule == (60, 60.0)
