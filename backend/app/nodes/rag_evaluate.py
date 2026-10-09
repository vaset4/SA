"""RAG 写回节点辅助。"""

from __future__ import annotations

from typing import Any

import app.services.rag_trace_service as rag_trace_service

# 这里故意保持为“薄包装节点辅助”，真正的复杂逻辑下沉到 service：
#   - 节点文件负责维持图节点接口形状
#   - service 负责实际 trace 写回
# 这样可以避免 `rag_node.py` 本身继续膨胀。


async def run_write_back(
    state: dict[str, Any],
    *,
    paper_db: Any,
    settings: Any,
    observability_module: Any,
    with_trace_fn: Any,
    update_trace_generation_fn: Any,
) -> dict[str, Any]:
    """把最终 trace / 观测信息写回存储；属于收尾节点，而不是推理节点。"""
    return await rag_trace_service.execute_write_back(
        state,
        paper_db=paper_db,
        settings=settings,
        observability_module=observability_module,
        with_trace_fn=with_trace_fn,
        update_trace_generation_fn=update_trace_generation_fn,
    )
