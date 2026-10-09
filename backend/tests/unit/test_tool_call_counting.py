"""Phase 1 — Slice E: 工具调用计数契约测试。

对应任务 05-23-cross-paper-qa-retrieval-fix。

策略：每个 tool 的 Command return 是单一字典字面量。本测用 `inspect.getsource`
读源码，验证计数字段是否被正确加入 / 排除。这是 contract-level 测试，比起
行为测试更耐 refactor（只要 update 字典里的字面量正确就过）。

计数规则（design.md §2）：
- vector_call_count: execute_local_rag_search + execute_scoped_paper_search
- graph_call_count:  graph_search_tool + neo4j_graph_search_tool
- 不计数: execute_read_chunk + execute_project_paper_inventory
"""

from __future__ import annotations

import inspect

from app.nodes import graph_tools


def _src_for_tool(tool_obj):
    """LangGraph @tool 装饰器把原函数挂在 .coroutine（async）或 .func（sync）。"""
    underlying = getattr(tool_obj, "coroutine", None) or getattr(tool_obj, "func", None) or tool_obj
    return inspect.getsource(underlying)


# ----------------------------- 向量类工具：必须 +1 -----------------------------


# ----------------------------- 图类工具：必须 +1 -----------------------------


def test_graph_search_tool_increments_graph_count() -> None:
    src = _src_for_tool(graph_tools.graph_search_tool)
    assert '"graph_call_count": 1' in src, (
        "graph_search_tool 的 Command.update 必须包含 "
        '"graph_call_count": 1'
    )


def test_neo4j_graph_search_tool_increments_graph_count() -> None:
    src = _src_for_tool(graph_tools.neo4j_graph_search_tool)
    assert '"graph_call_count": 1' in src, (
        "neo4j_graph_search_tool 的 Command.update 必须包含 "
        '"graph_call_count": 1（包括 success / degraded / unknown 3 个分支）'
    )
    # 必须覆盖所有真正发起调用的分支：success + Neo4jDegradedError + unknown Exception
    # paper_ids 空那个分支（早返回）不计数
    assert src.count('"graph_call_count": 1') >= 3, (
        f"neo4j_graph_search_tool 计数字段应至少出现 3 次（success / degraded / unknown），"
        f"实际 {src.count(chr(34) + 'graph_call_count' + chr(34) + ': 1')} 次"
    )


# ----------------------------- 不计数：inventory + read_chunk -----------------------------


# ----------------------------- 反向断言：reducer 在 RAGState 必须存在 -----------------------------


def test_reducer_already_present_on_state() -> None:
    """Slice B 已经把 reducer 加上；本测试是回归保险，防止 reducer 被错误移除。"""
    import operator
    from typing import get_type_hints

    from app.models.models import RAGState

    hints = get_type_hints(RAGState, include_extras=True)
    for field in ("vector_call_count", "graph_call_count"):
        metadata = getattr(hints[field], "__metadata__", ())
        assert operator.add in metadata, (
            f"{field!r} 必须保留 operator.add reducer，否则 Slice E 计数会被覆盖而非累加"
        )
