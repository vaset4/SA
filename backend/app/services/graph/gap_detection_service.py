"""确定性 gap 检测服务。"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from app.models.controversy_models import GapCandidate
from app.services.graph.research_object_extractor import extract_research_objects

_service: "GapDetectionService | None" = None


class GapDetectionService:
    """从 limitation / task / dataset 分布中识别第一版 gap。"""

    def detect_gaps(self, papers: list[dict[str, Any]], topic: str = "") -> list[GapCandidate]:
        gaps: list[GapCandidate] = []
        limitations = self._collect_limitations(papers)
        gaps.extend(self._limitation_gaps(limitations))
        gaps.extend(self._coverage_gaps(papers, topic))
        return gaps

    def _collect_limitations(self, papers: list[dict[str, Any]]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for paper in papers:
            analysis = paper.get("analysis") or {}
            limitation = str(analysis.get("limitation", "") or "").strip()
            if not limitation:
                continue
            key = limitation.lower()
            grouped[key].append(paper.get("paper_id", ""))
        return grouped

    def _limitation_gaps(self, limitations: dict[str, list[str]]) -> list[GapCandidate]:
        gaps: list[GapCandidate] = []
        for limitation, paper_ids in limitations.items():
            deduped_ids = [paper_id for paper_id in dict.fromkeys(paper_ids) if paper_id]
            if len(deduped_ids) < 2:
                continue
            gaps.append(
                {
                    "gap_id": self._gap_id("limitation", limitation),
                    "gap_type": "limitation_unresolved",
                    "topic_key": limitation[:80],
                    "support_paper_ids": deduped_ids,
                    "evidence_items": [f"重复 limitation: {limitation[:120]}"],
                    "severity": min(1.0, 0.4 + 0.2 * len(deduped_ids)),
                    "summary": "多篇论文重复提到相似局限，但当前材料中没有明确后续解决路径。",
                    "next_research_question": "哪些后续方法能直接针对这一局限设计对策并给出验证？",
                }
            )
        return gaps

    def _coverage_gaps(self, papers: list[dict[str, Any]], topic: str) -> list[GapCandidate]:
        task_methods: dict[str, set[str]] = defaultdict(set)
        task_datasets: dict[str, set[str]] = defaultdict(set)
        task_metrics: dict[str, set[str]] = defaultdict(set)

        for paper in papers:
            extraction = extract_research_objects(paper)
            tasks = extraction.get("tasks") or [topic or "unknown"]
            methods = {item for item in extraction.get("methods") or [] if item}
            datasets = {item for item in extraction.get("datasets") or [] if item}
            metrics = {item for item in extraction.get("metrics") or [] if item}
            for task in tasks:
                if not task:
                    continue
                task_methods[task].update(methods)
                task_datasets[task].update(datasets)
                task_metrics[task].update(metrics)

        gaps: list[GapCandidate] = []
        for task, methods in task_methods.items():
            dataset_count = len(task_datasets.get(task, set()))
            metric_count = len(task_metrics.get(task, set()))
            if len(methods) >= 3 and dataset_count <= 1:
                gaps.append(
                    {
                        "gap_id": self._gap_id("dataset", task),
                        "gap_type": "dataset_gap",
                        "topic_key": task,
                        "support_paper_ids": self._paper_ids_for_task(papers, task),
                        "evidence_items": [f"方法数 {len(methods)}，数据集数 {dataset_count}"],
                        "severity": 0.7,
                        "summary": f"{task} 下方法在增加，但数据集覆盖仍然单一。",
                        "next_research_question": "是否需要引入跨域或更具挑战性的数据集来验证泛化能力？",
                    }
                )
            if len(methods) >= 3 and metric_count <= 1:
                gaps.append(
                    {
                        "gap_id": self._gap_id("evaluation", task),
                        "gap_type": "evaluation_gap",
                        "topic_key": task,
                        "support_paper_ids": self._paper_ids_for_task(papers, task),
                        "evidence_items": [f"方法数 {len(methods)}，指标数 {metric_count}"],
                        "severity": 0.65,
                        "summary": f"{task} 的评测维度偏单一，可能遮蔽真实取舍。",
                        "next_research_question": "除了当前主指标，是否需要加入效率、稳定性或鲁棒性指标？",
                    }
                )
        return gaps

    def _paper_ids_for_task(self, papers: list[dict[str, Any]], task: str) -> list[str]:
        matched: list[str] = []
        lowered = task.lower()
        for paper in papers:
            extraction = extract_research_objects(paper)
            if any(item.lower() == lowered for item in extraction.get("tasks") or []):
                matched.append(paper.get("paper_id", ""))
        return [paper_id for paper_id in dict.fromkeys(matched) if paper_id]

    def _gap_id(self, prefix: str, topic_key: str) -> str:
        digest = hashlib.md5(topic_key.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{digest}"


def get_gap_detection_service() -> GapDetectionService:
    global _service
    if _service is None:
        _service = GapDetectionService()
    return _service
