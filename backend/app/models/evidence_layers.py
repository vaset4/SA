"""Shared constants for research evidence sources and layers."""

from __future__ import annotations

from typing import Final, Literal


EVIDENCE_SOURCE_MILVUS: Final = "milvus"
EVIDENCE_SOURCE_LIGHTRAG: Final = "lightrag"
EVIDENCE_SOURCE_NEO4J: Final = "neo4j"
EVIDENCE_SOURCE_INVENTORY: Final = "inventory"
# 外部 arXiv 检索记录的来源标识：外部产物没有本地 chunk_id，只进规划证据、不进引用池
EVIDENCE_SOURCE_ARXIV: Final = "arxiv"

EVIDENCE_LAYER_DIRECT_TEXT: Final = "direct_text"
EVIDENCE_LAYER_GRAPH_HINT: Final = "graph_hint"
EVIDENCE_LAYER_STRUCTURE_HINT: Final = "structure_hint"
EVIDENCE_LAYER_METADATA: Final = "metadata"

VERIFICATION_STATUS_VERIFIED: Final = "verified"
VERIFICATION_STATUS_UNVERIFIED: Final = "unverified"
VERIFICATION_STATUS_FAILED: Final = "verification_failed"

PLANNER_SOURCE_LOCAL_DIRECT_TEXT: Final = "local_direct_text"
PLANNER_SOURCE_LOCAL_METADATA: Final = "local_metadata"
PLANNER_SOURCE_LOCAL_STRUCTURE_HINT: Final = "local_structure_hint"
PLANNER_SOURCE_EXTERNAL_CONTEXT: Final = "external_context_needed"
PLANNER_SOURCE_INFERENCE_FROM_UNITS: Final = "inference_from_units"
PLANNER_SOURCE_LOCAL_SECTION_STRUCTURE: Final = "local_section_structure"
PLANNER_SOURCE_LOCAL_SECTION_TEXT: Final = "local_section_text"
PLANNER_SOURCE_LOCAL_CHUNK_METADATA: Final = "local_chunk_metadata"

EvidenceSource = Literal["milvus", "lightrag", "neo4j", "inventory", "arxiv"]
EvidenceLayer = Literal["direct_text", "graph_hint", "structure_hint", "metadata"]
VerificationStatus = Literal["verified", "unverified", "verification_failed"]
