"""
上传入库流水线模型。

这个模块只定义 Phase 5 需要的统一数据结构语义，避免上传链继续在多个服务里
随手拼匿名 dict。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, TypedDict

from app.models.models import Chunk, Paper
from app.models.knowledge_access_contracts import SourceItem

IngestStage = Literal["parse", "analyze", "index_write", "graph_write", "done"]
IngestStatus = Literal["start", "done", "error", "skipped"]


class IngestStepResult(TypedDict):
    """流水线单步执行结果。"""

    stage: IngestStage
    status: IngestStatus
    duration_ms: int
    detail: NotRequired[str]
    payload: NotRequired[Dict[str, Any]]


class IngestedDocument(TypedDict):
    """纯解析阶段产物。"""

    file_path: str
    chunks: List[Chunk]
    full_text: str
    file_type: str
    metadata: NotRequired["IngestedDocumentMetadata"]
    parser_mode: NotRequired[str]
    artifact_dir: NotRequired[str]
    source_markdown_path: NotRequired[str]
    sidecar_path: NotRequired[str]
    manifest_path: NotRequired[str]
    warnings: NotRequired[List[str]]
    normalized_page_count: NotRequired[int]
    content_schema_version: NotRequired[int]
    source_items: NotRequired[dict[str, SourceItem]]


class IngestedDocumentMetadata(TypedDict):
    """上传阶段抽取出的结构化元数据。"""

    title: str
    authors: List[str]
    published_date: str
    abstract: str


class IngestAnalysisResult(TypedDict):
    """结构化分析阶段产物。"""

    paper: Paper
    analysis_applied: bool


class IngestIndexResult(TypedDict):
    """索引阶段产物。"""

    saved_id: str
    vector_count: int


class IngestJob(TypedDict):
    """上传流水线的统一作业视图。"""

    job_id: str
    paper_id: str
    library_id: str
    filename: str
    status: Literal["running", "done", "error"]
    file_path: str
    file_type: str
    chunk_count: int
    parser_mode: NotRequired[str]
    warnings: NotRequired[List[Any]]
    index_status: NotRequired[str]
    graph_status: NotRequired[str]
    fallback_reason: NotRequired[str | None]
    steps: List[IngestStepResult]
    error: NotRequired[str]
