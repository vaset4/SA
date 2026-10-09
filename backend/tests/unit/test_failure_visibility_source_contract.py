from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_SCOPED_FILES = (
    REPO_ROOT / "backend/app/services/paper_db.py",
    REPO_ROOT / "backend/app/services/rag_trace_service.py",
    REPO_ROOT / "backend/app/services/recommendation.py",
    REPO_ROOT / "backend/app/middleware/observability.py",
)
FRONTEND_SCOPED_FILES = (
    REPO_ROOT / "frontend/app/(workspace)/tasks/page.jsx",
    REPO_ROOT / "frontend/app/(workspace)/evolution/page.jsx",
    REPO_ROOT / "frontend/app/(workspace)/projects/[projectId]/page.jsx",
)


def _is_exception_type(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Name) and node.id == "Exception"


def _silent_exception_handlers(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and _is_exception_type(node.type)
        and any(isinstance(statement, ast.Pass) for statement in node.body)
    ]


def _bare_trace_tasks(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "create_task" or not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Call) and getattr(first_arg.func, "id", "") == "save_rag_trace":
            lines.append(node.lineno)
    return lines


def test_scoped_backend_files_do_not_hide_side_channel_failures() -> None:
    violations = {
        str(path.relative_to(REPO_ROOT)): _silent_exception_handlers(path)
        for path in BACKEND_SCOPED_FILES
    }
    violations = {path: lines for path, lines in violations.items() if lines}

    assert not violations


def test_request_path_trace_persistence_has_no_bare_create_task() -> None:
    path = REPO_ROOT / "backend/app/middleware/observability.py"

    assert _bare_trace_tasks(path) == []


def test_scoped_frontend_files_do_not_hide_fetch_failures() -> None:
    violations = {
        str(path.relative_to(REPO_ROOT)): text.count("catch " + "{}")
        for path in FRONTEND_SCOPED_FILES
        for text in [path.read_text(encoding="utf-8")]
    }
    violations = {path: count for path, count in violations.items() if count}

    assert not violations
