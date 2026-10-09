"""
首轮检索结果的相关性评分。

这个模块位于 RAG 检索链的早期过滤阶段，负责把首轮命中的候选文档粗分成
`relevant / irrelevant` 两类，供 `rag_node.document_grading_node` 使用。

需要特别区分：
  - 它不是最终答案质量评估层
  - 它不是 reranker，目标不是精细排序分数
  - 它的职责只是尽早剔除明显无关的证据，减少后续 ReAct 决策噪音
"""

from __future__ import annotations

# Any 用于描述检索结果字典中的动态字段。
from typing import Any

# 导入配置读取函数，用于判断是否启用文档分级。
from app.config import get_settings
# 导入结构化输出模型，要求 LLM 返回 `relevant / irrelevant` 二分类结果。
from app.models.structured_outputs import DocumentGrade
# 导入统一 LLM 服务。
from app.middleware.trace_context import trace_label
from app.services.llm_service import get_llm

# 保留可 monkeypatch 的模块级入口，默认延迟初始化。
settings = None


def _settings():
    """按需读取配置，避免模块导入阶段触发全局初始化。"""
    return settings or get_settings()

# 提示词要求模型做“宽松的二分类过滤”，而不是严格排序。
_GRADING_PROMPT = """You are a relevance grader for an academic research system.

Given a user question and a retrieved document chunk, determine whether the document
is relevant to answering the question.

Rules:
- "relevant": The document contains information that could help answer the question
- "irrelevant": The document is off-topic or does not contribute to the answer
- Be generous: partial relevance counts as "relevant"
- Academic papers discussing the same topic or method are typically "relevant"

User question: {question}

Document text:
---
{document_text}
---"""


async def grade_documents(question: str, documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    对检索文档做相关性二分类。

    输入是首轮检索命中文档列表，输出是 `(relevant_docs, irrelevant_docs)`。
    当前不会直接修改外部状态，而是返回带 `grade/grade_reasoning` 的新字典。
    """
    # 关闭文档分级或没有候选文档时，直接原样放行，不做额外过滤。
    # 这意味着 grading 是“可插拔优化层”，而不是主链硬依赖。
    if not _settings().document_grading_enabled or not documents:
        return documents, []

    # 构造结构化输出 grader，要求模型返回 `DocumentGrade`。
    grader = get_llm().with_structured_output(DocumentGrade)
    # 收集相关文档。
    relevant: list[dict[str, Any]] = []
    # 收集无关文档，便于 trace 或调试时观察被过滤结果。
    irrelevant: list[dict[str, Any]] = []
    for document in documents:
        # 优先用 preview，其次用 text；并截断到 2000 字符控制提示词长度。
        prompt = _GRADING_PROMPT.format(
            question=question,
            document_text=(document.get("preview") or document.get("text") or "")[:2000],
        )
        # 调用模型完成单文档相关性判断。
        with trace_label("document_grading", "grade"):
            grade = await grader.ainvoke(prompt)
        # 把分级结果附加回原文档字典，形成新的 graded_document。
        graded_document = {**document, "grade": grade.binary_score, "grade_reasoning": grade.reasoning}
        # 相关文档进入 relevant 列表，供后续推理继续使用。
        if grade.binary_score == "relevant":
            relevant.append(graded_document)
        else:
            # 其余一律视为 irrelevant，供调试和 trace 统计。
            irrelevant.append(graded_document)
    return relevant, irrelevant
