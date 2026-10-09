"""项目内研究检索转换层。

职责：把 Milvus / Neo4j 检索结果转成统一 EvidenceRecord，供 planner 的取证适配器消费。
规则版取证计划与三路证据协调随 ReAct 链一起摘除，转换函数是这个模块留下的现役能力。
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any

from app.models.knowledge_access_contracts import KnowledgeMaterial, READ_CONTRACT_VERSION, SourceSpan
from app.models.research_contracts import EvidenceRecord
from app.models.evidence_layers import (
    EVIDENCE_LAYER_DIRECT_TEXT,
    EVIDENCE_LAYER_STRUCTURE_HINT,
    EVIDENCE_SOURCE_MILVUS,
    EVIDENCE_SOURCE_NEO4J,
    VERIFICATION_STATUS_UNVERIFIED,
    VERIFICATION_STATUS_VERIFIED,
)
from app.services.internal_research.evidence_identity import make_read_evidence_id


def convert_knowledge_materials(
    materials: tuple[KnowledgeMaterial, ...],
    *,
    hits: list[dict[str, Any]] | None = None,
    provenance: str = "retrieved_chunk",
) -> list[EvidenceRecord]:
    """参数为核验后的材料组、可选排名命中与来源类型；返回带范围身份的真实正文证据。"""
    records: list[EvidenceRecord] = []
    scores = {str(hit.get("chunk_id") or ""): float(hit.get("score") or 0.0) for hit in (hits or [])}
    parent_scores: dict[tuple[str, str], float] = {}
    for hit in hits or []:
        if not hit.get("parent_id"):
            continue
        key = (str(hit.get("paper_id") or ""), str(hit["parent_id"]))
        parent_scores[key] = max(parent_scores.get(key, 0.0), float(hit.get("score") or 0.0))
    for material in materials:
        # 1. 【知识取证】【证据转换】正文只来自已核验发布材料，索引预览不参与正文或身份
        spans: list[SourceSpan] = [deepcopy(span) for span in material.source_spans]
        record = EvidenceRecord(
            evidence_id=make_read_evidence_id(
                paper_id=material.paper_id, material_group_id=material.material_group_id,
                source_spans=material.source_spans, text=material.text,
            ),
            source=EVIDENCE_SOURCE_MILVUS,
            layer=EVIDENCE_LAYER_DIRECT_TEXT,
            paper_id=material.paper_id,
            chunk_id=material.chunk_id,
            parent_id=material.parent_id,
            section=material.section_path or material.section,
            page_num=material.page_num,
            excerpt=material.text,
            expanded_text="",
            summary="",
            dimensions=[],
            confidence=max(parent_scores.get((material.paper_id, material.parent_id), 0.0),
                           max((scores.get(span["chunk_id"], 0.0) for span in spans), default=0.0)),
            provenance=provenance,
            verification_status=VERIFICATION_STATUS_VERIFIED,
            usable_as_citation=True,
            read_contract_version=READ_CONTRACT_VERSION,
            material_group_id=material.material_group_id,
            source_spans=spans,
            source_location_status=material.source_location_status,
        )
        records.append(record)
    return records


def convert_neo4j_evidence(
    graph_evidence: list[dict[str, Any]],
) -> list[EvidenceRecord]:
    """
    把 Neo4j 图谱结果转换成 structure_hint 证据。

    Args:
        graph_evidence: 图谱抽取结果列表

    Returns:
        list[EvidenceRecord]: 不可直接引用的 structure_hint 证据记录
    """
    records: list[EvidenceRecord] = []
    # 1. 只保留有有效 summary 的结构化图谱线索
    for item in graph_evidence or []:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue
        # 2. 把图谱关系提示统一映射为 structure_hint 层 EvidenceRecord
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{uuid.uuid4().hex[:12]}",
                source=EVIDENCE_SOURCE_NEO4J,
                layer=EVIDENCE_LAYER_STRUCTURE_HINT,
                paper_id=str(item.get("paper_id") or ""),
                chunk_id="",
                parent_id="",
                section="",
                page_num=0,
                excerpt=summary,
                expanded_text="",
                summary=summary[:200],
                dimensions=[],
                confidence=0.6,
                provenance="ai_extracted_graph",
                verification_status=VERIFICATION_STATUS_UNVERIFIED,
                usable_as_citation=False,
            )
        )
    return records
