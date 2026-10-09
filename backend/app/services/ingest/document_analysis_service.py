"""
上传文档结构化分析服务。

这是上传流水线的第二阶段：基于全文调用 paper_analyzer 子图，补出 `paper.analysis`。
"""

from __future__ import annotations

from app.models.library_models import DEFAULT_LIBRARY_ID
from app.models.ingest_models import IngestAnalysisResult, IngestedDocument, IngestedDocumentMetadata
from app.models.models import Paper, PaperAnalyzerState


def _resolve_upload_metadata(ingested: IngestedDocument) -> IngestedDocumentMetadata:
    return ingested.get("metadata") or {
        "title": "",
        "authors": [],
        "published_date": "",
        "abstract": "",
    }


def build_upload_paper(filename: str, ingested: IngestedDocument, paper_id: str) -> Paper:
    """把解析结果包装成上传来源的 Paper 实体。"""
    metadata = _resolve_upload_metadata(ingested)
    return Paper(
        library_id=DEFAULT_LIBRARY_ID,
        paper_id=paper_id,
        arxiv_id=None,
        doi=None,
        title=metadata.get("title") or filename or "Untitled",
        authors=[str(author).strip() for author in metadata.get("authors", []) if str(author).strip()],
        abstract=metadata.get("abstract") or "",
        pdf_url="",
        pdf_path=ingested["file_path"],
        categories=[],
        published_date=metadata.get("published_date") or "",
        source="upload",
        analysis=None,
        tags=[],
    )


def apply_analysis(paper: Paper, analysis) -> Paper:
    """把分析结果合并回 Paper。"""
    return {**paper, "analysis": analysis}


async def analyze_paper(
    analyzer,
    paper: Paper,
    ingested: IngestedDocument,
) -> IngestAnalysisResult:
    """调用 paper_analyzer 子图补全 analysis。"""
    state: PaperAnalyzerState = {
        "paper": paper,
        "full_text": ingested["full_text"],
        "analysis": None,
    }
    result = await analyzer.ainvoke(state)
    analysis = result.get("analysis")
    return {
        "paper": apply_analysis(paper, analysis),
        "analysis_applied": analysis is not None,
    }
