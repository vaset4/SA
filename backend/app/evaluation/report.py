"""
评估报告渲染。

这个模块只负责把已经算好的评估结果整理成摘要文本，不负责：
  - 运行 RAG 子图
  - 调用 RAGAS 做评分
  - 加载黄金数据集

因此它在评估体系中属于“结果整理/展示层”，不是执行层。
"""

from __future__ import annotations

# Any 用于描述从评估流水线传来的动态结果结构。
from typing import Any


# 当前报告层强依赖的核心指标名称；缺任意一个都视为结果不完整。
REQUIRED_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def _metric_average(scores: dict[str, Any], metric_name: str) -> float:
    """提取单个指标的均值，并对缺失/非法格式做显式校验。"""
    # 从整体分数字典里取当前指标对应的列表。
    values = scores.get(metric_name)
    # 指标缺失或不是非空列表时，直接视为评估结果不完整。
    if not isinstance(values, list) or not values:
        raise ValueError(f"评估结果缺少核心指标: {metric_name}")
    try:
        # 把各条样本的分数统一转成 float。
        numeric_values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"评估指标不是数值列表: {metric_name}") from exc
    # 返回样本级指标均值。
    return sum(numeric_values) / len(numeric_values)


def extract_metric_summary(scores: dict[str, Any]) -> dict[str, float]:
    """从完整分数字典里提取核心指标摘要。"""
    # 整体 scores 结构必须是字典。
    if not isinstance(scores, dict):
        raise ValueError("评估结果 scores 必须是字典")
    # 对每个核心指标都计算一次均值摘要。
    return {
        metric_name: _metric_average(scores, metric_name)
        for metric_name in REQUIRED_METRICS
    }


def render_evaluation_report(result: dict[str, Any]) -> str:
    """把结构化评估结果渲染成简洁文本报告。"""
    # rows 用来统计样本数量。
    rows = result.get("rows")
    # 报告层要求 rows 必须是列表。
    if not isinstance(rows, list):
        raise ValueError("评估结果 rows 必须是列表")
    # 从 scores 中提取四个核心指标的均值。
    summary = extract_metric_summary(result.get("scores", {}))
    # 先写标题和样本数。
    lines = ["评估结果", f"样本数: {len(rows)}"]
    # 再逐项追加指标均值。
    for metric_name in REQUIRED_METRICS:
        lines.append(f"{metric_name}: {summary[metric_name]:.4f}")
    # 返回多行文本，供 CLI 或上层展示。
    return "\n".join(lines)
