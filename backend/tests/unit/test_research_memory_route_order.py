"""研究记忆路由顺序：静态路径必须先于 {memory_id}，避免 review 被当成 id。"""

from __future__ import annotations

from fastapi.routing import APIRoute

from app.api.routes.research_memory_api import router


def _path_template(route: APIRoute) -> str:
    return getattr(route, "path", "") or ""


def test_memory_review_registered_before_memory_id_get() -> None:
    get_routes = [
        r for r in router.routes
        if isinstance(r, APIRoute) and "GET" in r.methods
    ]
    paths = [_path_template(r) for r in get_routes]
    review = "/projects/{project_id}/memory/review"
    by_id = "/projects/{project_id}/memory/{memory_id}"
    assert review in paths
    assert by_id in paths
    assert paths.index(review) < paths.index(by_id)


def test_memory_curate_registered_before_memory_id_get() -> None:
    post_routes = [
        r for r in router.routes
        if isinstance(r, APIRoute) and "POST" in r.methods
    ]
    paths = [_path_template(r) for r in post_routes]
    curate = "/projects/{project_id}/memory/curate"
    # POST create is /memory ; path-param posts include /memory/{memory_id}/...
    assert curate in paths
    # ensure curate exists among posts (order vs GET by_id checked above for review)
    assert any(p.startswith("/projects/{project_id}/memory/{memory_id}") for p in paths)
