"""RAG 首轮结果 grading 节点辅助。"""

from __future__ import annotations

from typing import Any

import app.services.retrieval_service as retrieval_service

# 这个模块的定位和 `rag_evaluate.py` 类似：保留图节点接口，真正逻辑放到 service。
# 区别在于这里处理的是“文档命中后是否值得继续用”的 grading，而不是最终答案质量评估。


async def run_document_grading(
    state: dict[str, Any],
    *,
    grade_documents: Any,
    with_trace: Any,
    format_search_results_fn: Any,
) -> dict[str, Any]:
    """对首轮检索命中文档做 grading，供后续回答生成/补检索决策使用。"""
    return await retrieval_service.execute_document_grading(
        state,
        grade_documents=grade_documents,
        with_trace=with_trace,
        format_search_results_fn=format_search_results_fn,
    )
