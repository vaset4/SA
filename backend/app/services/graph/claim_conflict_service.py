"""确定性争议检测服务。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from app.models.controversy_models import ClaimTriple, ConflictCluster
from app.services.graph.research_object_extractor import extract_research_objects

_service: "ClaimConflictService | None" = None

_BETTER_HINTS = ("better", "improve", "improved", "higher", "state-of-the-art", "outperform", "gain", "提升", "优于", "更好")
_WORSE_HINTS = ("worse", "lower", "drop", "degrade", "trade-off", "slower", "下降", "更差", "更慢")
_SIMILAR_HINTS = ("similar", "comparable", "on par", "接近", "相当", "类似")
_EFFICIENCY_HINTS = ("efficient", "efficiency", "latency", "memory", "cost", "faster", "轻量", "效率", "更快", "更省")


class ClaimConflictService:
    """从论文 analysis 构建 ClaimTriple 并识别冲突簇。"""

    def build_claim_triples(self, papers: list[dict[str, Any]]) -> list[ClaimTriple]:
        triples: list[ClaimTriple] = []
        for paper in papers:
            extraction = extract_research_objects(paper)
            analysis = paper.get("analysis") or {}
            task = (extraction.get("tasks") or [""])[0]
            method = (extraction.get("methods") or [analysis.get("method", "") or ""])[0]
            dataset = (extraction.get("datasets") or [""])[0]
            metric = (extraction.get("metrics") or [""])[0]
            claims = extraction.get("claims") or analysis.get("contribution") or []
            if not claims:
                claims = [analysis.get("tldr", "")] if analysis.get("tldr") else []
            for claim_text in claims:
                normalized = str(claim_text or "").strip()
                if not normalized:
                    continue
                triples.append(
                    {
                        "paper_id": paper.get("paper_id", ""),
                        "task": task,
                        "method": method,
                        "dataset": dataset,
                        "metric": metric,
                        "claim_direction": self._infer_direction(normalized),
                        "claim_text": normalized,
                        "evidence_excerpt": normalized[:220],
                    }
                )
        return triples

    def detect_conflicts(self, triples: list[ClaimTriple]) -> list[ConflictCluster]:
        conflicts = self._metric_direction_conflicts(triples)
        conflicts.extend(self._method_tradeoff_conflicts(triples))
        return conflicts

    def _metric_direction_conflicts(self, triples: list[ClaimTriple]) -> list[ConflictCluster]:
        grouped: dict[tuple[str, str, str], list[ClaimTriple]] = defaultdict(list)
        for triple in triples:
            key = (triple["task"], triple["dataset"], triple["metric"])
            if any(key):
                grouped[key].append(triple)

        conflicts: list[ConflictCluster] = []
        for key, claims in grouped.items():
            directions = {claim["claim_direction"] for claim in claims if claim["claim_direction"] in {"better", "worse"}}
            papers = sorted({claim["paper_id"] for claim in claims if claim["paper_id"]})
            if len(directions) < 2 or len(papers) < 2:
                continue
            topic_key = " / ".join(item or "unknown" for item in key)
            conflicts.append(
                {
                    "cluster_id": self._cluster_id("metric", topic_key),
                    "topic_key": topic_key,
                    "papers": papers,
                    "conflict_reason": "相同 task + dataset + metric 下出现方向相反的结论",
                    "conflict_type": "metric_direction_conflict",
                    "support_claims": claims,
                    "summary": f"{topic_key} 上存在指标方向冲突，需回到实验设定核对差异。",
                }
            )
        return conflicts

    def _method_tradeoff_conflicts(self, triples: list[ClaimTriple]) -> list[ConflictCluster]:
        grouped: dict[str, list[ClaimTriple]] = defaultdict(list)
        for triple in triples:
            if triple["task"]:
                grouped[triple["task"]].append(triple)

        conflicts: list[ConflictCluster] = []
        for task, claims in grouped.items():
            efficiency_claims = [claim for claim in claims if self._looks_efficiency_tradeoff(claim["claim_text"])]
            quality_claims = [
                claim for claim in claims
                if claim["claim_direction"] == "better" and claim not in efficiency_claims
            ]
            papers = sorted({claim["paper_id"] for claim in claims if claim["paper_id"]})
            if len(efficiency_claims) == 0 or len(quality_claims) == 0 or len(papers) < 2:
                continue
            support_claims = efficiency_claims[:2] + quality_claims[:2]
            conflicts.append(
                {
                    "cluster_id": self._cluster_id("tradeoff", task),
                    "topic_key": task,
                    "papers": sorted({claim["paper_id"] for claim in support_claims if claim["paper_id"]}),
                    "conflict_reason": "同一任务下同时出现效果优先与资源优先的“更优”主张",
                    "conflict_type": "method_tradeoff_conflict",
                    "support_claims": support_claims,
                    "summary": f"{task} 存在方法取舍冲突，需要区分效果提升与资源效率的目标。",
                }
            )
        return conflicts

    def _infer_direction(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in _BETTER_HINTS):
            return "better"
        if any(token in lowered for token in _WORSE_HINTS):
            return "worse"
        if any(token in lowered for token in _SIMILAR_HINTS):
            return "similar"
        return "uncertain"

    def _looks_efficiency_tradeoff(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in _EFFICIENCY_HINTS)

    def _cluster_id(self, prefix: str, topic_key: str) -> str:
        digest = hashlib.md5(topic_key.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{digest}"


def get_claim_conflict_service() -> ClaimConflictService:
    global _service
    if _service is None:
        _service = ClaimConflictService()
    return _service
