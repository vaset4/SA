"""
RAGAS 离线评估入口。

这个模块不属于线上请求主链，而是离线评估工具链的一部分。当前职责是：
  - 加载黄金问答数据集
  - 对每个问题调用一次 RAG 子图，收集回答与检索上下文
  - 调用 RAGAS 计算核心评估指标
  - 按需生成文本版评估报告

它和线上 `/chat`、`/search/*` 之类接口的关系是“复用同一套 RAG 能力做离线测评”，
不是在线请求路径上的必经环节。
"""

from __future__ import annotations

# 导入 asyncio，用于在 CLI 入口里运行异步评估流程。
import asyncio
# 导入 json，用于读取黄金数据集和打印结果。
import json
# Path 用于定位默认数据集文件。
from pathlib import Path
# Any 用于描述评估行和结果中的动态结构。
from typing import Any
from uuid import uuid4

# 导入报告渲染函数；这里负责执行评估，报告文本拼装交给 report.py。
from app.evaluation.report import render_evaluation_report


# 默认黄金数据集文件名，和当前模块放在同一目录。
DATASET_FILE = "golden_dataset.json"


def _dataset_path(dataset_path: str | None = None) -> Path:
    """解析黄金数据集路径；未显式传入时回退到同目录默认文件。"""
    # 调用方显式传入路径时优先使用外部文件。
    if dataset_path:
        return Path(dataset_path)
    # 否则使用 `evaluation/golden_dataset.json`。
    return Path(__file__).with_name(DATASET_FILE)


def load_golden_dataset(dataset_path: str | None = None) -> list[dict[str, Any]]:
    """加载并校验黄金数据集的最小结构。"""
    # 先解析实际使用的数据集文件路径。
    path = _dataset_path(dataset_path)
    # 读取 JSON 文件。
    with path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    # 顶层必须是列表，每一项表示一条评估样本。
    if not isinstance(items, list):
        raise ValueError(f"黄金数据集格式无效: {path}")
    for item in items:
        # 每条样本都必须是对象。
        if not isinstance(item, dict):
            raise ValueError(f"黄金数据集条目必须是对象: {item!r}")
        # 当前评估最少依赖 `question` 和 `ground_truth` 两个字段。
        if not item.get("question") or not item.get("ground_truth"):
            raise ValueError(f"黄金数据集缺少 question/ground_truth: {item!r}")
    return items


async def run_rag_pipeline(question: str) -> dict[str, Any]:
    """
    参数为单个问题；运行一次独立 RAG 子图，返回问题、回答和检索上下文。

    这是离线评估对线上能力的复用入口：它会实际调用 `rag_node`，但目的不是
    服务用户请求，而是产出可供评估的数据行。
    """
    # 延迟导入 RAG 子图构造函数，避免单纯导入评估模块时拉起整条检索依赖。
    from app.nodes.rag_node import create_rag_node

    # 为当前样本创建一个 RAG 子图实例。
    rag_graph = create_rag_node()
    # 1. 【评估入口】【研究身份】每个新问题执行只生成一次身份，图内回边沿用该输入
    result = await rag_graph.ainvoke({"user_query": question, "run_id": str(uuid4())})
    # 从结果中提取搜索候选，作为评估里的 contexts。
    search_results = result.get("search_results", [])
    # 对每条检索命中优先取 preview，其次 text/full_text，统一成字符串列表。
    contexts = [
        item.get("preview") or item.get("text") or item.get("full_text") or ""
        for item in search_results
        if isinstance(item, dict)
    ]
    return {
        # 原始问题，便于后续构造评估数据行。
        "question": question,
        # RAG 子图最终答案。
        "answer": result.get("answer", ""),
        # 供 RAGAS 使用的上下文列表。
        "contexts": contexts,
        # 1. 【研究评测】【实际交付】观察随原始回答保留，不进入评分输入或真值
        **{key: result[key] for key in ("final_delivery_report", "final_citation_bindings",
                                       "final_generation_attempt", "final_delivery_applicability") if key in result},
    }


async def build_evaluation_rows(dataset_path: str | None = None) -> list[dict[str, Any]]:
    """把黄金数据集和 RAG 实跑结果拼成 RAGAS 所需的数据行列表。"""
    # 先加载人工标注的黄金样本。
    golden_dataset = load_golden_dataset(dataset_path)
    # 逐条样本收集 RAG 输出。
    rows: list[dict[str, Any]] = []
    for item in golden_dataset:
        # 对当前问题实际跑一遍 RAG。
        rag_result = await run_rag_pipeline(item["question"])
        # 拼成 RAGAS 评估常用的 question/ground_truth/answer/contexts 结构。
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"],
            }
        )
    return rows


def _require_ragas():
    """延迟导入 RAGAS 及其依赖，缺失时显式报错。"""
    try:
        # Dataset 用于把 Python 列表转成 RAGAS 可消费的数据集对象。
        from datasets import Dataset
        # evaluate 是 RAGAS 的核心评估入口。
        from ragas import evaluate
        # 当前固定使用四个核心指标。
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError("运行评估需要先安装 ragas 和 datasets 依赖") from exc
    return Dataset, evaluate, context_precision, context_recall, faithfulness, answer_relevancy


def generate_evaluation_report(result: dict[str, Any]) -> str:
    """把评估结果交给报告层渲染成文本。"""
    return render_evaluation_report(result)


async def run_evaluation(
    dataset_path: str | None = None,
    include_report: bool = False,
) -> dict[str, Any]:
    """
    运行完整离线评估流程。

    返回值包含：
      - `rows`: 每条样本的 question/ground_truth/answer/contexts
      - `scores`: RAGAS 输出的逐指标结果列表
      - `report`: 可选，文本版摘要报告
    """
    # 先确认评估依赖已安装，并拿到当前要使用的指标对象。
    Dataset, evaluate, context_precision, context_recall, faithfulness, answer_relevancy = _require_ragas()
    # 构造样本级评估行。
    rows = await build_evaluation_rows(dataset_path)
    # 转成 RAGAS 需要的数据集格式。
    dataset = Dataset.from_list(rows)
    # 对整份数据集运行四个核心指标评估。
    scores = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )
    # 把评分结果转成普通字典，方便 JSON 输出和报告渲染。
    result = {
        "rows": rows,
        "scores": scores.to_pandas().to_dict(orient="list"),
    }
    # 调用方要求时再补充文本报告，保持默认输出更偏结构化。
    if include_report:
        result["report"] = generate_evaluation_report(result)
    return result


def main() -> int:
    """CLI 入口：运行完整评估并把结果打印到标准输出。"""
    # 在同步入口中运行异步评估流程。
    result = asyncio.run(run_evaluation())
    # 直接打印完整结构化结果，便于离线保存或重定向。
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 以 0 表示执行成功。
    return 0


if __name__ == "__main__":
    # 作为脚本运行时，返回 CLI 退出码。
    raise SystemExit(main())
