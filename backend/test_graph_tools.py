from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys


def test_graph_search_tool_can_be_imported_as_langchain_tool() -> None:
    os.environ.pop("SSLKEYLOGFILE", None)
    sys.modules.pop("app.nodes.graph_tools", None)

    module = importlib.import_module("app.nodes.graph_tools")

    assert module.graph_search_tool.name == "graph_search"
    assert module.graph_search_tool.description


def test_app_main_ignores_unwritable_sslkeylogfile() -> None:
    backend_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["SSLKEYLOGFILE"] = r"D:\ssl_key\ssl_key.log"

    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=backend_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
